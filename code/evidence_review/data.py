"""
Data loading for the Multi-Modal Evidence Review system.

Reads the four input CSVs and joins them into a single in-memory
`Claim` record per row, with image paths resolved to absolute paths
on disk (and filtered against junk files like .DS_Store).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvidenceRequirement:
    requirement_id: str
    claim_object: str  # 'car' | 'laptop' | 'package' | 'all'
    applies_to: str
    minimum_image_evidence: str


@dataclass
class UserHistory:
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str

    @property
    def has_risk_flag(self) -> bool:
        return self.history_flags.strip().lower() not in {"none", ""}

    @property
    def rejection_rate(self) -> float:
        total = self.accept_claim + self.manual_review_claim + self.rejected_claim
        if total == 0:
            return 0.0
        return self.rejected_claim / total


@dataclass
class Claim:
    user_id: str
    image_paths_raw: str  # original semicolon-joined string, preserved verbatim for output
    user_claim: str
    claim_object: str
    image_abs_paths: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)

    # Only present for sample_claims.csv (labeled) rows; None for claims.csv
    expected: dict | None = None


def _image_id_from_path(path: str) -> str:
    return Path(path).stem


def load_evidence_requirements(path: str) -> list[EvidenceRequirement]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            EvidenceRequirement(
                requirement_id=row["requirement_id"],
                claim_object=row["claim_object"],
                applies_to=row["applies_to"],
                minimum_image_evidence=row["minimum_image_evidence"],
            )
            for row in reader
        ]


def load_user_history(path: str) -> dict[str, UserHistory]:
    out: dict[str, UserHistory] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uh = UserHistory(
                user_id=row["user_id"],
                past_claim_count=int(row["past_claim_count"] or 0),
                accept_claim=int(row["accept_claim"] or 0),
                manual_review_claim=int(row["manual_review_claim"] or 0),
                rejected_claim=int(row["rejected_claim"] or 0),
                last_90_days_claim_count=int(row["last_90_days_claim_count"] or 0),
                history_flags=row["history_flags"],
                history_summary=row["history_summary"],
            )
            out[uh.user_id] = uh
    return out


def load_claims(path: str, dataset_root: str, labeled: bool = False) -> list[Claim]:
    """
    Load claims.csv or sample_claims.csv.

    `dataset_root` is the directory that `image_paths` entries are relative
    to (the `dataset/` folder), since CSV paths look like
    'images/test/case_001/img_1.jpg'.

    If `labeled` is True, also captures the expected-output columns from
    sample_claims.csv into `Claim.expected` for evaluation use.
    """
    claims: list[Claim] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_paths = row["image_paths"]
            parts = [p.strip() for p in raw_paths.split(";") if p.strip()]

            abs_paths, ids, missing = [], [], []
            for p in parts:
                full = os.path.join(dataset_root, p)
                ids.append(_image_id_from_path(p))
                if os.path.isfile(full):
                    abs_paths.append(full)
                else:
                    missing.append(p)

            claim = Claim(
                user_id=row["user_id"],
                image_paths_raw=raw_paths,
                user_claim=row["user_claim"],
                claim_object=row["claim_object"],
                image_abs_paths=abs_paths,
                image_ids=ids,
                missing_images=missing,
            )

            if labeled:
                claim.expected = {
                    "evidence_standard_met": row.get("evidence_standard_met", ""),
                    "evidence_standard_met_reason": row.get("evidence_standard_met_reason", ""),
                    "risk_flags": row.get("risk_flags", ""),
                    "issue_type": row.get("issue_type", ""),
                    "object_part": row.get("object_part", ""),
                    "claim_status": row.get("claim_status", ""),
                    "claim_status_justification": row.get("claim_status_justification", ""),
                    "supporting_image_ids": row.get("supporting_image_ids", ""),
                    "valid_image": row.get("valid_image", ""),
                    "severity": row.get("severity", ""),
                }

            claims.append(claim)
    return claims


def requirements_for_object(
    requirements: list[EvidenceRequirement], claim_object: str
) -> list[EvidenceRequirement]:
    """Requirements that apply to this object: object-specific ones plus 'all'."""
    return [r for r in requirements if r.claim_object in (claim_object, "all")]
