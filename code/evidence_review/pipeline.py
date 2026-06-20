"""
Pipeline orchestrator: ties together data loading, prompting, the VLM
client, and the guardrails layer into a single per-claim review step and
a batch CSV-to-CSV runner.

Cost discipline (per problem_statement.md's "operational analysis" ask):
- exactly ONE VLM call per claim row, with all of that claim's images
  attached together, rather than one call per image.
- disk-backed response cache so re-running during development never
  re-spends a call for an unchanged (prompt, images) pair.
- usage (input/output tokens) is tracked per call and summed, so the
  evaluation report's cost estimate is measured, not guessed.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

from .data import Claim, EvidenceRequirement, UserHistory, requirements_for_object
from .guardrails import build_review_result, fallback_result, parse_model_json
from .prompt import build_messages
from .schema import OUTPUT_COLUMNS
from .vlm_client import ResponseCache, VLMClient


@dataclass
class RunStats:
    rows_processed: int = 0
    vlm_calls_made: int = 0
    cache_hits: int = 0
    failures: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_images: int = 0
    total_latency_s: float = 0.0
    per_row: list[dict] = field(default_factory=list)

    def record(self, *, cached: bool, failed: bool, input_tokens: int, output_tokens: int,
               n_images: int, latency_s: float, user_id: str) -> None:
        self.rows_processed += 1
        if cached:
            self.cache_hits += 1
        else:
            self.vlm_calls_made += 1
        if failed:
            self.failures += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_images += n_images
        self.total_latency_s += latency_s
        self.per_row.append({
            "user_id": user_id,
            "cached": cached,
            "failed": failed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "n_images": n_images,
            "latency_s": round(latency_s, 3),
        })


def review_claim(
    claim: Claim,
    client: VLMClient,
    user_histories: dict[str, UserHistory],
    requirements: list[EvidenceRequirement],
    cache: ResponseCache | None,
    stats: RunStats,
) -> dict:
    """Run one claim through the pipeline and return a finished output row (dict)."""
    history = user_histories.get(claim.user_id)
    reqs = requirements_for_object(requirements, claim.claim_object)
    system_prompt, user_prompt, image_paths = build_messages(claim, history, reqs)

    failed = False
    try:
        response = client.generate(system_prompt, user_prompt, image_paths, cache=cache)
        raw = parse_model_json(response.text)
        result = build_review_result(raw, claim, history)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cached = response.cached
        latency = response.latency_s
    except Exception as e:  # noqa: BLE001 -- any failure must still produce a row, not crash the batch
        import traceback
        print(f"  [WARN] review failed for {claim.user_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed = True
        cached = False
        latency = 0.0
        input_tokens = output_tokens = 0
        result = fallback_result(claim, reason=str(e)[:200])

    stats.record(
        cached=cached,
        failed=failed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        n_images=len(claim.image_abs_paths),
        latency_s=latency,
        user_id=claim.user_id,
    )
    return result.as_row(claim)


def run_batch(
    claims: list[Claim],
    client: VLMClient,
    user_histories: dict[str, UserHistory],
    requirements: list[EvidenceRequirement],
    output_csv_path: str,
    cache_dir: str | None = None,
    verbose: bool = True,
) -> RunStats:
    cache = ResponseCache(cache_dir) if cache_dir else None
    stats = RunStats()

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    # Open in write mode once, flush after every row. If the process is
    # killed/interrupted partway through (Ctrl+C, crash, quota exhaustion
    # raised as an unexpected error somewhere outside review_claim's own
    # try/except), every row completed so far is already safely on disk
    # instead of being lost because we only wrote at the very end.
    f = open(output_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    f.flush()

    rows = []
    try:
        for i, claim in enumerate(claims, 1):
            if verbose:
                print(f"[{i}/{len(claims)}] reviewing claim for {claim.user_id} "
                      f"({claim.claim_object}, {len(claim.image_abs_paths)} image(s))...")
            row = review_claim(claim, client, user_histories, requirements, cache, stats)
            rows.append(row)
            writer.writerow(row)
            f.flush()
    finally:
        f.close()

    if verbose:
        print(f"\nWrote {len(rows)} rows to {output_csv_path}")
        print(f"VLM calls: {stats.vlm_calls_made}, cache hits: {stats.cache_hits}, "
              f"failures: {stats.failures}")
        print(f"Tokens: {stats.total_input_tokens} in / {stats.total_output_tokens} out")

    return stats
