"""Routing-policy validation and persistence.

The HARD SAFETY LIMITS below live in code, not in YAML, so a tuned or hand-edited policy file can never make
the gate more permissive than they allow. Tuning may only move thresholds toward MORE caution.
These are research-demo limits, not clinical limits.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.schemas.labels import SPECIALTIES
from backend.services.risk_model import NUMERIC_FEATURES
from backend.utils.errors import PolicyError

# Trigger thresholds: the rule fires when value >= threshold, so a LOWER number is more cautious.
# The policy's number must be <= this ceiling.
MAX_LIMITS: dict[str, float] = {
    "human_review_probability": 0.50,
    "infection_probability": 0.50,   # requirement: any high infection-concern signal -> a human
    "same_day_probability": 0.50,
    "urgency_score": 1.50,           # halfway between PRIORITY (1) and URGENT (2)
    "urgent_tail_probability": 0.40,  # P(URGENT) + P(EMERGENCY)
    "risk_score": 0.50,
}
# Floors: the rule fires when value < floor, so a HIGHER number is more cautious.
# The policy's number must be >= this minimum.
MIN_LIMITS: dict[str, float] = {
    "urgency_confidence_min": 0.40,
    "specialty_confidence_min": 0.60,
    "specialty_top_probability_min": 0.55,
    "boolean_certainty_min": 0.20,
}
LEGACY_REQUIRED = [
    "human_review_probability", "urgency_score_urgent", "same_day_probability", "infection_probability_urgent",
    "specialty_confidence", "specialty_top_probability", "urgency_confidence_min",
]
ROUTES = {"escalate_to_human", "urgent_review", "automated_specialty_route", "human_review"}


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Raise PolicyError listing every problem; return the policy unchanged when it is acceptable."""
    problems: list[str] = []
    rules = policy.get("rules")
    th = policy.get("thresholds") or {}
    if rules not in ("legacy", "conservative"):
        raise PolicyError("Routing policy has no valid 'rules' type (legacy | conservative).")
    for key in ("confidence_bands", "route_labels", "specialty_labels"):
        if key not in policy:
            problems.append(f"missing section '{key}'")
    missing_routes = ROUTES - set(policy.get("route_labels", {}))
    if missing_routes:
        problems.append(f"route_labels missing {sorted(missing_routes)}")

    if rules == "legacy":
        problems += [f"missing threshold '{k}'" for k in LEGACY_REQUIRED if k not in th]
    else:
        for k in (*MAX_LIMITS, *MIN_LIMITS):
            if k not in th:
                problems.append(f"missing threshold '{k}'")
                continue
            v = th[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
                problems.append(f"threshold '{k}' is not a number")
            elif k in MAX_LIMITS and not 0 <= v <= MAX_LIMITS[k]:
                problems.append(f"threshold '{k}'={v} is more permissive than the safety ceiling {MAX_LIMITS[k]}")
            elif k in MIN_LIMITS and not MIN_LIMITS[k] <= v <= 1:
                problems.append(f"threshold '{k}'={v} is more permissive than the safety floor {MIN_LIMITS[k]}")
        bad = [s for s in policy.get("human_only_specialties", []) if s not in SPECIALTIES]
        if bad:
            problems.append(f"unknown specialties in human_only_specialties: {bad}")
        rm = policy.get("risk_model")
        if rm:
            n = len(rm.get("features", []))
            if rm.get("features") != NUMERIC_FEATURES or not (len(rm.get("coef", [])) == len(rm.get("mean", [])) == len(rm.get("scale", [])) == n):
                problems.append("risk_model features/coefficients are inconsistent")
    if problems:
        raise PolicyError("Routing policy is invalid: " + "; ".join(problems), detail=" | ".join(problems))
    return policy


def dump_policy(policy: dict[str, Any], path: Path, header: str) -> None:
    body = yaml.safe_dump(policy, sort_keys=False, default_flow_style=False, width=100)
    path.write_text(header.rstrip() + "\n\n" + body, encoding="utf-8")


def provenance_stub(**fields: Any) -> dict[str, Any]:
    return {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
