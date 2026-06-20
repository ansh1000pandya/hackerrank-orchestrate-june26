"""
Schema constants for the Multi-Modal Evidence Review system.

Single source of truth for:
- required output columns (order matters for output.csv)
- allowed values per categorical field (per problem_statement.md)
- object-specific allowed `object_part` values

Keeping this separate from logic means validation and prompting both
import the *same* lists, so the model's allowed-value menu can never
drift out of sync with what we accept when writing rows.
"""

from __future__ import annotations

# Exact column order required by problem_statement.md
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_OBJECTS = {"car", "laptop", "package"}

CLAIM_STATUS_VALUES = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE_VALUES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}

OBJECT_PART_VALUES = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}

RISK_FLAG_VALUES = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}

SEVERITY_VALUES = {"none", "low", "medium", "high", "unknown"}

BOOL_STRINGS = {"true", "false"}


def closest_allowed(value: str, allowed: set[str], default: str = "unknown") -> str:
    """
    Map a free-text model output to the closest allowed value.

    The spec says "use the closest matching value from these lists" -- models
    occasionally emit a near-miss (e.g. "broken" instead of "broken_part").
    This does exact match first (case/whitespace-insensitive), then a cheap
    substring heuristic, then falls back to `default` rather than raising,
    so one bad field never crashes a whole row.
    """
    if value is None:
        return default
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    for a in allowed:
        if v == a:
            return a
    for a in allowed:
        if v in a or a in v:
            return a
    return default


def closest_object_part(value: str, claim_object: str) -> str:
    allowed = OBJECT_PART_VALUES.get(claim_object, {"unknown"})
    return closest_allowed(value, allowed, default="unknown")


def normalize_bool(value) -> str:
    """Normalize a python bool / string into the literal 'true'/'false' the CSV needs."""
    if isinstance(value, bool):
        return "true" if value else "false"
    v = str(value).strip().lower()
    if v in {"true", "yes", "1"}:
        return "true"
    if v in {"false", "no", "0"}:
        return "false"
    return "false"  # conservative default: don't claim evidence is valid/sufficient unless told so
