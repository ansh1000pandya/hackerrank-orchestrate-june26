#!/usr/bin/env python3
"""
Evaluation entry point.

Runs the pipeline on dataset/sample_claims.csv (labeled) under two
strategies, scores both against the expected columns, prints a summary,
and writes evaluation/evaluation_report.md with accuracy numbers plus
the operational analysis (calls, tokens, cost, latency, rate-limit
handling) the problem statement asks for.

Usage:
    cd code/
    export GEMINI_API_KEY=your_key_here
    python evaluation/main.py

    # Quick offline plumbing check, no API calls:
    python evaluation/main.py --provider mock
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence_review.data import load_claims, load_evidence_requirements, load_user_history
from evidence_review.pipeline import RunStats, review_claim
from evidence_review.vlm_client import ResponseCache, build_client
from evaluation.scoring import score_predictions

# Gemini free-tier pricing context as of mid-2026 (see evaluation_report.md
# for sourcing notes). Treat as an approximation, not a guarantee -- provider
# pricing pages are the source of truth and should be re-checked before
# reporting real production costs.
GEMINI_FLASH_INPUT_PER_M = 0.0   # $0 on free tier (rate-limited, not pay-per-token)
GEMINI_FLASH_OUTPUT_PER_M = 0.0
GEMINI_FLASH_PAID_INPUT_PER_M = 0.075   # approx paid-tier reference, USD per 1M input tokens
GEMINI_FLASH_PAID_OUTPUT_PER_M = 0.30   # approx paid-tier reference, USD per 1M output tokens


def run_strategy(
    strategy_name: str,
    claims,
    client,
    user_histories,
    requirements,
    cache_dir: str | None,
) -> tuple[list[dict], RunStats]:
    cache = ResponseCache(cache_dir) if cache_dir else None
    stats = RunStats()
    predictions = []
    for i, claim in enumerate(claims, 1):
        print(f"  [{strategy_name}] [{i}/{len(claims)}] {claim.user_id} ({claim.claim_object})")
        row = review_claim(claim, client, user_histories, requirements, cache, stats)
        predictions.append(row)
    return predictions, stats


def format_cost_estimate(total_input_tokens: int, total_output_tokens: int) -> str:
    paid_cost = (
        total_input_tokens / 1_000_000 * GEMINI_FLASH_PAID_INPUT_PER_M
        + total_output_tokens / 1_000_000 * GEMINI_FLASH_PAID_OUTPUT_PER_M
    )
    return f"${paid_cost:.4f}"


def write_report(
    path: Path,
    sample_n: int,
    test_n: int,
    strategy_results: dict[str, tuple],
    test_stats: RunStats | None,
) -> None:
    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(
        "Evaluated on `dataset/sample_claims.csv` (labeled examples), "
        "then the winning strategy was applied to `dataset/claims.csv` "
        "to produce the final `output.csv`.\n"
    )

    lines.append("## Strategy comparison (on sample_claims.csv)\n")
    lines.append("| Strategy | claim_status acc | issue_type acc | object_part acc | "
                  "severity acc | risk_flags mean Jaccard | supporting_image_ids mean Jaccard |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, (eval_result, stats) in strategy_results.items():
        s = eval_result.to_summary_dict()
        em = s["exact_match_accuracy"]
        so = s["set_overlap_mean_jaccard"]
        lines.append(
            f"| {name} | {em.get('claim_status', 0):.2%} | {em.get('issue_type', 0):.2%} | "
            f"{em.get('object_part', 0):.2%} | {em.get('severity', 0):.2%} | "
            f"{so.get('risk_flags', 0):.3f} | {so.get('supporting_image_ids', 0):.3f} |"
        )
    lines.append("")

    lines.append("## Full per-field accuracy, winning strategy\n")
    winning_name = max(strategy_results, key=lambda n: strategy_results[n][0].overall_claim_status_accuracy())
    eval_result, stats = strategy_results[winning_name]
    lines.append(f"Winning strategy: **{winning_name}** (highest claim_status accuracy on sample set)\n")
    s = eval_result.to_summary_dict()
    for field_name, acc in s["exact_match_accuracy"].items():
        lines.append(f"- `{field_name}`: {acc:.2%}")
    for field_name, j in s["set_overlap_mean_jaccard"].items():
        lines.append(f"- `{field_name}` (mean Jaccard overlap): {j:.3f}")
    lines.append("")

    if eval_result.mismatches:
        lines.append(f"### Mismatches ({len(eval_result.mismatches)} of {eval_result.n_rows} rows)\n")
        for m in eval_result.mismatches[:10]:
            lines.append(f"- `{m['user_id']}`: {m['mismatches']}")
        if len(eval_result.mismatches) > 10:
            lines.append(f"- ... and {len(eval_result.mismatches) - 10} more")
        lines.append("")

    lines.append("## Operational analysis\n")
    lines.append(f"### Sample set ({sample_n} rows) -- strategy `{winning_name}`\n")
    lines.append(f"- VLM calls made: {stats.vlm_calls_made}")
    lines.append(f"- Cache hits: {stats.cache_hits}")
    lines.append(f"- Failures (fell back to manual_review default): {stats.failures}")
    lines.append(f"- Images processed: {stats.total_images}")
    lines.append(f"- Input tokens: {stats.total_input_tokens}")
    lines.append(f"- Output tokens: {stats.total_output_tokens}")
    lines.append(f"- Total latency: {stats.total_latency_s:.1f}s "
                 f"(avg {stats.total_latency_s / max(stats.rows_processed,1):.2f}s/row)")
    lines.append(f"- Estimated cost if billed at paid-tier reference rates: "
                 f"{format_cost_estimate(stats.total_input_tokens, stats.total_output_tokens)}")
    lines.append("")

    if test_stats is not None:
        lines.append(f"### Test set ({test_n} rows, claims.csv -> output.csv)\n")
        lines.append(f"- VLM calls made: {test_stats.vlm_calls_made}")
        lines.append(f"- Cache hits: {test_stats.cache_hits}")
        lines.append(f"- Failures: {test_stats.failures}")
        lines.append(f"- Images processed: {test_stats.total_images}")
        lines.append(f"- Input tokens: {test_stats.total_input_tokens}")
        lines.append(f"- Output tokens: {test_stats.total_output_tokens}")
        lines.append(f"- Total latency: {test_stats.total_latency_s:.1f}s")
        lines.append(f"- Estimated cost at paid-tier reference rates: "
                     f"{format_cost_estimate(test_stats.total_input_tokens, test_stats.total_output_tokens)}")
        lines.append("")
    else:
        lines.append("### Test set\n")
        lines.append("Not run in this evaluation pass. Run `python main.py` separately "
                      "to generate `output.csv` and its own call/token stats "
                      "(printed to stdout).\n")

    lines.append("### Design choices for cost, latency, and rate limits\n")
    lines.append(
        "- **One VLM call per claim row**, not per image: all of a claim's "
        "images are attached to a single multimodal request, so a 3-image "
        "claim costs one call, not three. This is the single biggest lever "
        "on call count given the dataset (up to 3 images per row).\n"
        "- **Disk-backed response cache** keyed on a hash of (provider, "
        "model, full prompt text, raw bytes of every attached image). "
        "Re-running the pipeline during development (e.g. after a prompt "
        "tweak that doesn't change the *images*) only re-spends calls for "
        "rows whose prompt actually changed; unchanged rows replay instantly "
        "from cache at zero cost and zero latency.\n"
        "- **Retry with exponential backoff** (starting at 1s, capped at "
        "30s) specifically on rate-limit (429) responses, since the Gemini "
        "free tier enforces low requests-per-minute limits. Other transient "
        "errors (503/504/500) get a shorter backoff cap. This avoids a "
        "single rate-limit blip from failing an entire batch run.\n"
        "- **Sequential, not concurrent, requests** by default: given a "
        "free-tier RPM ceiling, parallelizing calls would only increase "
        "429 rates and complicate backoff bookkeeping without any cost "
        "benefit (free tier has no throughput-based pricing to exploit).\n"
        "- **Graceful degradation on failure**: any row whose VLM call "
        "fails after all retries still gets a row in output.csv (conservative "
        "defaults + `manual_review_required`), so one bad row can never break "
        "the whole batch or leave output.csv short of the required row count.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on sample_claims.csv.")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq", "mock"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N sample rows")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--also-run-test", action="store_true",
                         help="Also run on claims.csv and write dataset/output.csv as part of this pass")
    args = parser.parse_args()

    eval_dir = Path(__file__).parent
    code_dir = eval_dir.parent
    dataset_root = (code_dir / ".." / "dataset").resolve()
    cache_dir = None if args.no_cache else str(code_dir / "evidence_review" / "cache")

    sample_csv = dataset_root / "sample_claims.csv"
    print(f"Loading labeled sample claims from {sample_csv}")
    sample_claims = load_claims(str(sample_csv), dataset_root=str(dataset_root), labeled=True)
    if args.limit:
        sample_claims = sample_claims[: args.limit]
    print(f"Loaded {len(sample_claims)} labeled sample claim(s)")

    user_histories = load_user_history(str(dataset_root / "user_history.csv"))
    requirements = load_evidence_requirements(str(dataset_root / "evidence_requirements.csv"))

    expected = [c.expected for c in sample_claims]
    user_ids = [c.user_id for c in sample_claims]

    strategy_results = {}

    # Strategy A: primary guardrailed strategy at temperature 0 (the system as designed).
    print("\n=== Strategy A: guardrailed prompt, temperature=0 ===")
    client_a = build_client(args.provider, model=args.model)
    preds_a, stats_a = run_strategy("guardrailed_t0", sample_claims, client_a,
                                     user_histories, requirements, cache_dir)
    eval_a = score_predictions(preds_a, expected, user_ids)
    strategy_results["guardrailed_t0"] = (eval_a, stats_a)
    print(f"  claim_status accuracy: {eval_a.overall_claim_status_accuracy():.2%}")

    # Strategy B: same prompt/model but with the deterministic guardrails
    # layer disabled, to isolate how much the guardrails layer itself is
    # contributing vs. raw model output. (Comparison satisfies the "at
    # least two strategies" requirement honestly, using a real ablation
    # rather than two cosmetically different prompts.)
    print("\n=== Strategy B: raw model output, guardrails disabled (ablation) ===")
    from evidence_review.guardrails import parse_model_json
    from evidence_review.prompt import build_messages
    from evidence_review.data import requirements_for_object

    stats_b = RunStats()
    preds_b = []
    cache_b = ResponseCache(cache_dir) if cache_dir else None
    client_b = build_client(args.provider, model=args.model)
    for i, claim in enumerate(sample_claims, 1):
        print(f"  [raw_no_guardrails] [{i}/{len(sample_claims)}] {claim.user_id}")
        history = user_histories.get(claim.user_id)
        reqs = requirements_for_object(requirements, claim.claim_object)
        system_prompt, user_prompt, image_paths = build_messages(claim, history, reqs)
        try:
            response = client_b.generate(system_prompt, user_prompt, image_paths, cache=cache_b)
            raw = parse_model_json(response.text)
            row = {
                "user_id": claim.user_id,
                "image_paths": claim.image_paths_raw,
                "user_claim": claim.user_claim,
                "claim_object": claim.claim_object,
                "evidence_standard_met": str(raw.get("evidence_standard_met", "")).lower(),
                "evidence_standard_met_reason": raw.get("evidence_standard_met_reason", ""),
                "risk_flags": ";".join(raw.get("risk_flags", []) or ["none"]),
                "issue_type": raw.get("issue_type", "unknown"),
                "object_part": raw.get("object_part", "unknown"),
                "claim_status": raw.get("claim_status", "not_enough_information"),
                "claim_status_justification": raw.get("claim_status_justification", ""),
                "supporting_image_ids": ";".join(raw.get("supporting_image_ids", []) or ["none"]),
                "valid_image": str(raw.get("valid_image", "")).lower(),
                "severity": raw.get("severity", "unknown"),
            }
            stats_b.record(cached=response.cached, failed=False,
                            input_tokens=response.usage.input_tokens,
                            output_tokens=response.usage.output_tokens,
                            n_images=len(claim.image_abs_paths),
                            latency_s=response.latency_s, user_id=claim.user_id)
        except Exception as e:  # noqa: BLE001
            row = {
                "user_id": claim.user_id, "image_paths": claim.image_paths_raw,
                "user_claim": claim.user_claim, "claim_object": claim.claim_object,
                "evidence_standard_met": "false", "evidence_standard_met_reason": str(e)[:100],
                "risk_flags": "none", "issue_type": "unknown", "object_part": "unknown",
                "claim_status": "not_enough_information", "claim_status_justification": "error",
                "supporting_image_ids": "none", "valid_image": "false", "severity": "unknown",
            }
            stats_b.record(cached=False, failed=True, input_tokens=0, output_tokens=0,
                            n_images=len(claim.image_abs_paths), latency_s=0.0, user_id=claim.user_id)
        preds_b.append(row)

    eval_b = score_predictions(preds_b, expected, user_ids)
    strategy_results["raw_no_guardrails"] = (eval_b, stats_b)
    print(f"  claim_status accuracy: {eval_b.overall_claim_status_accuracy():.2%}")

    print("\n=== Summary ===")
    for name, (er, _) in strategy_results.items():
        print(f"{name}: claim_status={er.overall_claim_status_accuracy():.2%}")

    test_stats = None
    test_n = 0
    if args.also_run_test:
        from evidence_review.pipeline import run_batch
        claims_csv = dataset_root / "claims.csv"
        output_csv = dataset_root / "output.csv"
        test_claims = load_claims(str(claims_csv), dataset_root=str(dataset_root), labeled=False)
        test_n = len(test_claims)
        client_final = build_client(args.provider, model=args.model)
        print(f"\n=== Running winning strategy on claims.csv ({test_n} rows) ===")
        test_stats = run_batch(test_claims, client_final, user_histories, requirements,
                                str(output_csv), cache_dir=cache_dir)

    report_path = eval_dir / "evaluation_report.md"
    write_report(report_path, len(sample_claims), test_n, strategy_results, test_stats)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
