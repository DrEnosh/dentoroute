"""Deterministic rule engine for the synthetic benchmark labels.

Reads data/ground_truth_rules.yaml. Neither Claude nor Jev is involved here.
Educational synthetic benchmark rules - NOT clinical guidelines.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from backend.schemas.case import CaseState, ReferenceLabels
from backend.utils.config import DATA_DIR

RULES_PATH = DATA_DIR / "ground_truth_rules.yaml"

_ANTICOAG = ("anticoag", "warfarin", "apixaban", "rivaroxaban", "clopidogrel", "blood thinner")
_IMMUNO = ("immunocompromised", "chemotherapy", "uncontrolled diabetes", "hiv", "transplant", "immunosuppress")


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> dict:
    with open(path or RULES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _contains_any_text(items: list[str], needles: tuple[str, ...]) -> bool:
    blob = " ".join(items).lower()
    return any(n in blob for n in needles)


def build_context(state: CaseState) -> dict[str, Any]:
    ctx = state.model_dump()
    ctx["medical_history_count"] = len(state.medical_history)
    ctx["medication_count"] = len(state.medications)
    ctx["allergies_present"] = len(state.allergies) > 0
    ctx["has_anticoagulant"] = _contains_any_text(state.medications + state.medical_history, _ANTICOAG)
    ctx["immunocompromised"] = _contains_any_text(state.medical_history, _IMMUNO)
    ctx["tobacco_use"] = _contains_any_text(state.medical_history, ("tobacco", "smok"))
    return ctx


# ---- condition evaluation --------------------------------------------------
def _leaf(cond: dict, ctx: dict) -> bool:
    field, op, expected = cond["field"], cond.get("op", "eq"), cond.get("value")
    actual = ctx.get(field)
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op in ("gt", "gte", "lt", "lte"):
        if actual is None:
            return False
        return {
            "gt": actual > expected,
            "gte": actual >= expected,
            "lt": actual < expected,
            "lte": actual <= expected,
        }[op]
    if op == "in":
        return actual in expected
    if op == "contains":
        return actual is not None and expected in actual
    if op == "contains_any":
        return actual is not None and any(v in actual for v in expected)
    raise ValueError(f"Unknown operator: {op}")


def evaluate_condition(cond: dict, ctx: dict) -> bool:
    if "all" in cond:
        return all(evaluate_condition(c, ctx) for c in cond["all"])
    if "any" in cond:
        return any(evaluate_condition(c, ctx) for c in cond["any"])
    if "not" in cond:
        return not evaluate_condition(cond["not"], ctx)
    return _leaf(cond, ctx)


# ---- per-dimension evaluation ---------------------------------------------
def _max_level(section: dict, ctx: dict) -> tuple[str, list[str]]:
    levels: list[str] = section["levels"]
    best = levels.index(section["default"])
    fired: list[str] = []
    for rule in section["rules"]:
        if evaluate_condition(rule["when"], ctx):
            fired.append(rule["id"])
            best = max(best, levels.index(rule["level"]))
    return levels[best], fired


def _any_true(section: dict, ctx: dict) -> tuple[bool, list[str]]:
    fired = [r["id"] for r in section["rules"] if evaluate_condition(r["when"], ctx)]
    return bool(fired), fired


def label_case(state: CaseState, rules: dict | None = None) -> ReferenceLabels:
    rules = rules or load_rules()
    ctx = build_context(state)
    fired: dict[str, list[str]] = {}

    urgency, fired["urgency"] = _max_level(rules["urgency"], ctx)
    ctx["urgency_idx"] = rules["urgency"]["levels"].index(urgency)

    infection, fired["infection_concern"] = _any_true(rules["infection_concern"], ctx)
    ctx["infection_concern"] = infection

    same_day, fired["same_day_review"] = _any_true(rules["same_day_review"], ctx)
    ctx["same_day_review"] = same_day

    # specialty: first match wins
    spec_rules = rules["specialty"]
    specialty, fired["specialty"] = spec_rules["default"], []
    for rule in spec_rules["rules"]:
        if evaluate_condition(rule["when"], ctx):
            specialty, fired["specialty"] = rule["specialty"], [rule["id"]]
            break

    human, fired["human_review"] = _any_true(rules["human_review"], ctx)

    comp = rules["complexity"]
    points, fired["complexity"] = 0, []
    for rule in comp["rules"]:
        if evaluate_condition(rule["when"], ctx):
            points += rule["points"]
            fired["complexity"].append(rule["id"])
    th = comp["thresholds"]
    if points >= th["HIGHLY_COMPLEX"]:
        complexity = "HIGHLY_COMPLEX"
    elif points >= th["COMPLEX"]:
        complexity = "COMPLEX"
    elif points >= th["MODERATE"]:
        complexity = "MODERATE"
    else:
        complexity = "SIMPLE"
    ctx["complexity_idx"] = comp["levels"].index(complexity)

    referral, fired["referral_priority"] = _max_level(rules["referral_priority"], ctx)

    return ReferenceLabels(
        urgency=urgency,
        specialty=specialty,
        infection_concern=infection,
        same_day_review=same_day,
        human_review=human,
        complexity=complexity,
        referral_priority=referral,
        fired_rules=fired,
    )
