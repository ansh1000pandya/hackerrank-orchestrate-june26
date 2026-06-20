# Evaluation Report

Evaluated on `dataset/sample_claims.csv` (labeled examples), then the winning strategy was applied to `dataset/claims.csv` to produce the final `output.csv`.

## Strategy comparison (on sample_claims.csv)

| Strategy | claim_status acc | issue_type acc | object_part acc | severity acc | risk_flags mean Jaccard | supporting_image_ids mean Jaccard |
|---|---|---|---|---|---|---|
| guardrailed_t0 | 65.00% | 55.00% | 70.00% | 40.00% | 0.643 | 0.775 |
| raw_no_guardrails | 65.00% | 55.00% | 70.00% | 40.00% | 0.582 | 0.775 |

## Full per-field accuracy, winning strategy

Winning strategy: **guardrailed_t0** (highest claim_status accuracy on sample set)

- `evidence_standard_met`: 70.00%
- `claim_status`: 65.00%
- `issue_type`: 55.00%
- `object_part`: 70.00%
- `valid_image`: 75.00%
- `severity`: 40.00%
- `risk_flags` (mean Jaccard overlap): 0.643
- `supporting_image_ids` (mean Jaccard overlap): 0.775

### Mismatches (17 of 20 rows)

- `user_002`: {'evidence_standard_met': {'predicted': 'true', 'expected': 'false'}, 'claim_status': {'predicted': 'supported', 'expected': 'not_enough_information'}, 'issue_type': {'predicted': 'scratch', 'expected': 'broken_part'}, 'severity': {'predicted': 'low', 'expected': 'unknown'}, 'risk_flags': {'predicted': [], 'expected': ['claim_mismatch', 'manual_review_required', 'wrong_object']}, 'supporting_image_ids': {'predicted': ['img_1'], 'expected': ['img_1', 'img_2']}}
- `user_004`: {'issue_type': {'predicted': 'glass_shatter', 'expected': 'crack'}, 'severity': {'predicted': 'high', 'expected': 'medium'}}
- `user_007`: {'issue_type': {'predicted': 'glass_shatter', 'expected': 'broken_part'}}
- `user_005`: {'claim_status': {'predicted': 'supported', 'expected': 'contradicted'}, 'issue_type': {'predicted': 'dent', 'expected': 'scratch'}, 'severity': {'predicted': 'medium', 'expected': 'low'}, 'risk_flags': {'predicted': ['user_history_risk'], 'expected': ['claim_mismatch', 'manual_review_required', 'user_history_risk']}}
- `user_006`: {'object_part': {'predicted': 'unknown', 'expected': 'headlight'}, 'valid_image': {'predicted': 'false', 'expected': 'true'}, 'risk_flags': {'predicted': [], 'expected': ['damage_not_visible', 'wrong_angle']}}
- `user_003`: {'evidence_standard_met': {'predicted': 'false', 'expected': 'true'}, 'claim_status': {'predicted': 'not_enough_information', 'expected': 'supported'}, 'valid_image': {'predicted': 'false', 'expected': 'true'}, 'severity': {'predicted': 'unknown', 'expected': 'medium'}, 'risk_flags': {'predicted': ['blurry_image', 'manual_review_required', 'possible_manipulation'], 'expected': ['blurry_image']}}
- `user_008`: {'evidence_standard_met': {'predicted': 'false', 'expected': 'true'}, 'claim_status': {'predicted': 'not_enough_information', 'expected': 'contradicted'}, 'issue_type': {'predicted': 'unknown', 'expected': 'broken_part'}, 'object_part': {'predicted': 'unknown', 'expected': 'front_bumper'}, 'severity': {'predicted': 'unknown', 'expected': 'high'}, 'risk_flags': {'predicted': ['blurry_image', 'manual_review_required', 'possible_manipulation', 'user_history_risk', 'wrong_angle'], 'expected': ['claim_mismatch', 'manual_review_required', 'non_original_image', 'user_history_risk']}, 'supporting_image_ids': {'predicted': [], 'expected': ['img_1']}}
- `user_009`: {'issue_type': {'predicted': 'glass_shatter', 'expected': 'crack'}, 'severity': {'predicted': 'high', 'expected': 'medium'}}
- `user_012`: {'severity': {'predicted': 'medium', 'expected': 'low'}}
- `user_018`: {'issue_type': {'predicted': 'glass_shatter', 'expected': 'crack'}, 'severity': {'predicted': 'high', 'expected': 'medium'}}
- ... and 7 more

## Operational analysis

### Sample set (20 rows) -- strategy `guardrailed_t0`

- VLM calls made: 20
- Cache hits: 0
- Failures (fell back to manual_review default): 2
- Images processed: 29
- Input tokens: 62813
- Output tokens: 2432
- Total latency: 165.0s (avg 8.25s/row)
- Estimated cost if billed at paid-tier reference rates: $0.0054

### Test set

Not run in this evaluation pass. Run `python main.py` separately to generate `output.csv` and its own call/token stats (printed to stdout).

### Design choices for cost, latency, and rate limits

- **One VLM call per claim row**, not per image: all of a claim's images are attached to a single multimodal request, so a 3-image claim costs one call, not three. This is the single biggest lever on call count given the dataset (up to 3 images per row).
- **Disk-backed response cache** keyed on a hash of (provider, model, full prompt text, raw bytes of every attached image). Re-running the pipeline during development (e.g. after a prompt tweak that doesn't change the *images*) only re-spends calls for rows whose prompt actually changed; unchanged rows replay instantly from cache at zero cost and zero latency.
- **Retry with exponential backoff** (starting at 1s, capped at 30s) specifically on rate-limit (429) responses, since the Gemini free tier enforces low requests-per-minute limits. Other transient errors (503/504/500) get a shorter backoff cap. This avoids a single rate-limit blip from failing an entire batch run.
- **Sequential, not concurrent, requests** by default: given a free-tier RPM ceiling, parallelizing calls would only increase 429 rates and complicate backoff bookkeeping without any cost benefit (free tier has no throughput-based pricing to exploit).
- **Graceful degradation on failure**: any row whose VLM call fails after all retries still gets a row in output.csv (conservative defaults + `manual_review_required`), so one bad row can never break the whole batch or leave output.csv short of the required row count.
