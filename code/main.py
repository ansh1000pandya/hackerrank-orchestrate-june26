#!/usr/bin/env python3
"""
Multi-Modal Evidence Review -- main entry point.

Reads dataset/claims.csv (test rows, no labels) and produces output.csv
with one prediction row per input row, per the schema in
problem_statement.md.

Usage:
    cd code/
    export GEMINI_API_KEY=your_key_here
    python main.py

    # Options:
    python main.py --provider mock              # offline, no API calls (for plumbing tests)
    python main.py --model gemini-2.5-flash      # pick a specific Gemini model
    python main.py --limit 5                     # only process first N rows (quick smoke test)
    python main.py --no-cache                    # force fresh calls, ignore cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evidence_review.data import load_claims, load_evidence_requirements, load_user_history
from evidence_review.pipeline import run_batch
from evidence_review.vlm_client import build_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-modal evidence review pipeline.")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq", "mock"],
                         help="VLM backend to use (default: gemini)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows")
    parser.add_argument("--no-cache", action="store_true", help="Disable response caching")
    parser.add_argument(
        "--claims-csv", default=None,
        help="Path to input claims CSV (default: ../dataset/claims.csv relative to this file)",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Path to write predictions (default: ../dataset/output.csv)",
    )
    args = parser.parse_args()

    code_dir = Path(__file__).parent
    dataset_root = (code_dir / ".." / "dataset").resolve()

    claims_csv = Path(args.claims_csv) if args.claims_csv else dataset_root / "claims.csv"
    output_csv = Path(args.output_csv) if args.output_csv else dataset_root / "output.csv"
    user_history_csv = dataset_root / "user_history.csv"
    evidence_req_csv = dataset_root / "evidence_requirements.csv"
    cache_dir = code_dir / "evidence_review" / "cache"

    print(f"Loading claims from {claims_csv}")
    claims = load_claims(str(claims_csv), dataset_root=str(dataset_root), labeled=False)
    if args.limit:
        claims = claims[: args.limit]
    print(f"Loaded {len(claims)} claim(s)")

    user_histories = load_user_history(str(user_history_csv))
    requirements = load_evidence_requirements(str(evidence_req_csv))

    missing_total = sum(len(c.missing_images) for c in claims)
    if missing_total:
        print(f"WARNING: {missing_total} referenced image path(s) could not be found on disk.")

    client = build_client(args.provider, model=args.model)
    print(f"Using provider={client.name} model={client.model}")

    start = time.time()
    stats = run_batch(
        claims=claims,
        client=client,
        user_histories=user_histories,
        requirements=requirements,
        output_csv_path=str(output_csv),
        cache_dir=None if args.no_cache else str(cache_dir),
    )
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
