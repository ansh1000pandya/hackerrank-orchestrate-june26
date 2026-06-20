"""
Scoring logic for evaluating predictions against sample_claims.csv's
labeled expected outputs.

Per-field metrics:
- exact-match accuracy for single-valued categorical fields
  (claim_status, issue_type, object_part, severity, evidence_standard_met,
  valid_image)
- set-overlap (Jaccard-style) for multi-valued fields (risk_flags,
  supporting_image_ids), since "close enough" matters more than exact
  string equality for sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _as_set(value: str) -> set[str]:
    if not value:
        return set()
    parts = {p.strip() for p in value.split(";") if p.strip()}
    parts.discard("none")
    return parts


@dataclass
class FieldScore:
    field_name: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class SetFieldScore:
    field_name: str
    jaccard_sum: float = 0.0
    total: int = 0

    @property
    def mean_jaccard(self) -> float:
        return self.jaccard_sum / self.total if self.total else 0.0


EXACT_MATCH_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "object_part",
    "valid_image",
    "severity",
]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]


@dataclass
class EvalResult:
    n_rows: int = 0
    exact_scores: dict[str, FieldScore] = field(default_factory=dict)
    set_scores: dict[str, SetFieldScore] = field(default_factory=dict)
    mismatches: list[dict] = field(default_factory=list)

    def overall_claim_status_accuracy(self) -> float:
        return self.exact_scores["claim_status"].accuracy if "claim_status" in self.exact_scores else 0.0

    def to_summary_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "exact_match_accuracy": {
                k: round(v.accuracy, 4) for k, v in self.exact_scores.items()
            },
            "set_overlap_mean_jaccard": {
                k: round(v.mean_jaccard, 4) for k, v in self.set_scores.items()
            },
        }


def score_predictions(predictions: list[dict], expected: list[dict], user_ids: list[str]) -> EvalResult:
    """
    `predictions` and `expected` must be aligned lists (same order, same rows).
    `user_ids` is used only for readable mismatch reporting.
    """
    result = EvalResult(n_rows=len(predictions))
    for f in EXACT_MATCH_FIELDS:
        result.exact_scores[f] = FieldScore(field_name=f)
    for f in SET_FIELDS:
        result.set_scores[f] = SetFieldScore(field_name=f)

    for pred, exp, uid in zip(predictions, expected, user_ids):
        row_mismatches = {}
        for f in EXACT_MATCH_FIELDS:
            p_val = str(pred.get(f, "")).strip().lower()
            e_val = str(exp.get(f, "")).strip().lower()
            result.exact_scores[f].total += 1
            if p_val == e_val:
                result.exact_scores[f].correct += 1
            else:
                row_mismatches[f] = {"predicted": p_val, "expected": e_val}

        for f in SET_FIELDS:
            p_set = _as_set(str(pred.get(f, "")))
            e_set = _as_set(str(exp.get(f, "")))
            result.set_scores[f].total += 1
            if not p_set and not e_set:
                jaccard = 1.0
            elif not p_set or not e_set:
                jaccard = 0.0
            else:
                jaccard = len(p_set & e_set) / len(p_set | e_set)
            result.set_scores[f].jaccard_sum += jaccard
            if jaccard < 1.0:
                row_mismatches[f] = {"predicted": sorted(p_set), "expected": sorted(e_set)}

        if row_mismatches:
            result.mismatches.append({"user_id": uid, "mismatches": row_mismatches})

    return result
