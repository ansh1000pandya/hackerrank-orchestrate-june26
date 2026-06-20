"""
Deterministic post-processing layer.

The VLM produces a first-pass structured judgment. This module is the
non-negotiable, code-enforced layer on top of it:

1. Validate every field against the allowed-value lists in schema.py,
   coercing near-misses and falling back to safe defaults instead of
   ever writing an invalid value to output.csv.
2. Merge in user-history risk signals as ADDITIONAL risk flags only --
   never let history flip a claim_status the model derived from images.
   This directly encodes the spec's rule: "User history can add risk
   context, but should not override clear visual evidence by itself."
3. Force `manual_review_required` onto genuinely risky combinations
   (e.g. high rejection rate + contradicted claim) so risk doesn't
   silently disappear if the model forgets to flag it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .data import Claim, UserHistory
from .schema import (
    CLAIM_STATUS_VALUES,
    ISSUE_TYPE_VALUES,
    RISK_FLAG_VALUES,
    SEVERITY_VALUES,
    closest_allowed,
    closest_object_part,
    normalize_bool,
)

HIGH_RISK_REJECTION_RATE = 0.4  # >=40% of a user's resolved claims were rejected
HIGH_RISK_RECENT_CLAIMS = 4  # claims in the last 90 days


@dataclass
class ReviewResult:
    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str
    severity: str

    def as_row(self, claim: Claim) -> dict:
        return {
            "user_id": claim.user_id,
            "image_paths": claim.image_paths_raw,
            "user_claim": claim.user_claim,
            "claim_object": claim.claim_object,
            "evidence_standard_met": self.evidence_standard_met,
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": self.risk_flags,
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": self.supporting_image_ids,
            "valid_image": self.valid_image,
            "severity": self.severity,
        }


def parse_model_json(raw_text: str) -> dict:
    """Parse the model's JSON, tolerating stray markdown fences some models add anyway."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    return json.loads(text)


def _history_risk_flags(history: UserHistory | None) -> list[str]:
    flags = []
    if history is None:
        return flags
    if history.has_risk_flag:
        flags.append("user_history_risk")
    if history.rejection_rate >= HIGH_RISK_REJECTION_RATE and history.past_claim_count > 0:
        flags.append("user_history_risk")
    if history.last_90_days_claim_count >= HIGH_RISK_RECENT_CLAIMS:
        flags.append("user_history_risk")
    return flags


def build_review_result(
    raw: dict,
    claim: Claim,
    history: UserHistory | None,
) -> ReviewResult:
    claim_object = claim.claim_object

    # --- categorical fields, coerced into allowed values ---
    issue_type = closest_allowed(str(raw.get("issue_type", "unknown")), ISSUE_TYPE_VALUES)
    object_part = closest_object_part(str(raw.get("object_part", "unknown")), claim_object)
    claim_status = closest_allowed(
        str(raw.get("claim_status", "not_enough_information")),
        CLAIM_STATUS_VALUES,
        default="not_enough_information",
    )
    severity = closest_allowed(str(raw.get("severity", "unknown")), SEVERITY_VALUES, default="unknown")

    # --- booleans ---
    evidence_standard_met = normalize_bool(raw.get("evidence_standard_met", False))
    valid_image = normalize_bool(raw.get("valid_image", bool(claim.image_abs_paths)))

    # If no images actually loaded, evidence cannot possibly be sufficient,
    # regardless of what the model claims -- a hard code-level override.
    if not claim.image_abs_paths:
        evidence_standard_met = "false"
        valid_image = "false"
        if claim_status == "supported":
            claim_status = "not_enough_information"

    # --- risk flags: model's flags + history-derived flags, deduped, validated ---
    model_flags_raw = raw.get("risk_flags", []) or []
    if isinstance(model_flags_raw, str):
        model_flags_raw = [f.strip() for f in model_flags_raw.split(";") if f.strip()]
    model_flags = [closest_allowed(f, RISK_FLAG_VALUES, default=None) for f in model_flags_raw]
    model_flags = [f for f in model_flags if f and f != "none"]

    history_flags = _history_risk_flags(history)

    all_flags = list(dict.fromkeys(model_flags + history_flags))  # dedupe, preserve order

    # Escalate to manual review if claim is contradicted/uncertain AND there's
    # any risk signal at all -- risk should not silently vanish.
    risky_status = claim_status in {"contradicted", "not_enough_information"}
    if risky_status and all_flags and "manual_review_required" not in all_flags:
        all_flags.append("manual_review_required")

    risk_flags_str = ";".join(all_flags) if all_flags else "none"

    # --- supporting image ids: must be a subset of actually-loaded image ids ---
    sup_raw = raw.get("supporting_image_ids", []) or []
    if isinstance(sup_raw, str):
        sup_raw = [s.strip() for s in sup_raw.split(";") if s.strip()]
    valid_ids = set(claim.image_ids)
    sup_ids = [s for s in sup_raw if s in valid_ids]
    supporting_image_ids = ";".join(sup_ids) if sup_ids else "none"

    evidence_reason = str(raw.get("evidence_standard_met_reason", "")).strip() or "No reason provided."
    justification = str(raw.get("claim_status_justification", "")).strip() or "No justification provided."

    return ReviewResult(
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=evidence_reason,
        risk_flags=risk_flags_str,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=supporting_image_ids,
        valid_image=valid_image,
        severity=severity,
    )


def fallback_result(claim: Claim, reason: str) -> ReviewResult:
    """Used when the VLM call fails entirely (after retries) or returns unparseable JSON."""
    return ReviewResult(
        evidence_standard_met="false",
        evidence_standard_met_reason=f"Automated review failed: {reason}",
        risk_flags="manual_review_required",
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification=f"Could not complete automated review: {reason}. Needs manual review.",
        supporting_image_ids="none",
        valid_image="false" if not claim.image_abs_paths else "true",
        severity="unknown",
    )
