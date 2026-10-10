"""Benchmark metrics: pure functions over benchmark rows (no I/O, no model calls).

A benchmark row (see benchmark_service) holds `reference`, `predictions`, `probabilities`, `confidences`,
`latency_ms`, `cost_usd`, `schema_valid`, `error_kind`. Rows with `predictions is None` are failures and are
excluded from accuracy numbers but counted in the error / schema-compliance numbers.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.schemas.labels import (
    BOOLEAN_QUESTIONS, COMPLEXITY_LEVELS, REFERRAL_LEVELS, SPECIALTIES, URGENCY_LEVELS,
)

CATEGORICAL: dict[str, tuple[list[str], bool]] = {  # question -> (labels, ordinal?)
    "urgency": (URGENCY_LEVELS, True),
    "specialty": (SPECIALTIES, False),
    "complexity": (COMPLEXITY_LEVELS, True),
    "referral_priority": (REFERRAL_LEVELS, True),
}
QUESTIONS = ["urgency", "specialty", "infection_concern", "same_day_review", "human_review", "complexity", "referral_priority"]


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("predictions")]


def is_correct(row: dict[str, Any], q: str) -> bool:
    return row["predictions"][q] == row["reference"][q]


# ----------------------------------------------------------------- categorical --
def categorical_metrics(rows: list[dict[str, Any]], q: str) -> dict[str, Any]:
    labels, ordinal = CATEGORICAL[q]
    ok = valid_rows(rows)
    n = len(ok)
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)  # rows = reference, cols = predicted
    for r in ok:
        mat[idx[r["reference"][q]], idx[r["predictions"][q]]] += 1
    out: dict[str, Any] = {
        "n": n, "labels": labels, "confusion_matrix": mat.tolist(),
        "accuracy": float(np.trace(mat) / n) if n else None,
    }
    # macro-F1 over labels that occur in the reference or the predictions
    f1s = []
    for i in range(len(labels)):
        tp, fp, fn = mat[i, i], mat[:, i].sum() - mat[i, i], mat[i, :].sum() - mat[i, i]
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    out["macro_f1"] = float(np.mean(f1s)) if f1s else None
    if ordinal and n:
        diffs = np.array([abs(idx[r["predictions"][q]] - idx[r["reference"][q]]) for r in ok])
        out["within_one"] = float((diffs <= 1).mean())
        out["mae_levels"] = float(diffs.mean())
        under = [idx[r["predictions"][q]] < idx[r["reference"][q]] for r in ok]
        out["under_triage_rate"] = float(np.mean(under))  # predicted LOWER than reference
        out["over_triage_rate"] = float(np.mean([idx[r["predictions"][q]] > idx[r["reference"][q]] for r in ok]))
    if q == "specialty" and n:
        with_probs = [r for r in ok if r.get("probabilities", {}).get("specialty")]
        if with_probs:
            hits = 0
            for r in with_probs:
                top3 = sorted(r["probabilities"]["specialty"].items(), key=lambda kv: -kv[1])[:3]
                hits += r["reference"][q] in {k for k, _ in top3}
            out["top3_accuracy"] = hits / len(with_probs)
        else:
            out["top3_accuracy"] = None  # provider returns no probability distribution
    return out


# --------------------------------------------------------------------- boolean --
def boolean_metrics(rows: list[dict[str, Any]], q: str) -> dict[str, Any]:
    ok = valid_rows(rows)
    n = len(ok)
    tp = sum(1 for r in ok if r["predictions"][q] and r["reference"][q])
    fp = sum(1 for r in ok if r["predictions"][q] and not r["reference"][q])
    fn = sum(1 for r in ok if not r["predictions"][q] and r["reference"][q])
    tn = n - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    probs = [(r["probabilities"].get(q), r["reference"][q]) for r in ok if r.get("probabilities", {}).get(q) is not None]
    brier = float(np.mean([(p - float(y)) ** 2 for p, y in probs])) if probs else None
    return {
        "n": n, "accuracy": (tp + tn) / n if n else None, "precision": prec, "recall": rec, "f1": f1,
        "brier": brier, "n_positive": tp + fn, "confusion_matrix": [[tn, fp], [fn, tp]],
        "labels": ["no", "yes"],
    }


# ------------------------------------------------------------ latency / cost ---
def latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None and not r.get("cached")]
    if not vals:
        return {"n": 0, "status": "not measured"}
    a = np.array(vals, dtype=float)
    return {
        "n": len(vals), "status": "measured", "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
    }


def cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    if not vals:
        return {"status": "not measured", "total_usd": None, "per_case_usd": None}
    return {"status": "measured", "total_usd": float(sum(vals)), "per_case_usd": float(np.mean(vals)), "n": len(vals)}


def token_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    u = [r["usage"] for r in rows if r.get("usage")]
    if not u:
        return {"status": "not reported"}
    tin = sum(x.get("input_tokens", 0) for x in u)
    tout = sum(x.get("output_tokens", 0) for x in u)
    return {"status": "reported", "input_tokens": tin, "output_tokens": tout, "n": len(u)}


# ---------------------------------------------------------- gated routing ------
def routing_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """What the confidence-gated policy did, and whether automation was safe (rows must carry `route`)."""
    ok = [r for r in valid_rows(rows) if r.get("route")]
    if not ok:
        return None
    n = len(ok)
    counts: dict[str, int] = {}
    for r in ok:
        counts[r["route"]] = counts.get(r["route"], 0) + 1
    auto = [r for r in ok if r["route"] == "automated_specialty_route"]
    manual = [r for r in ok if r["route"] != "automated_specialty_route"]

    def acc(rs: list[dict[str, Any]], q: str) -> float | None:
        return float(np.mean([is_correct(r, q) for r in rs])) if rs else None

    high_ref = [r for r in ok if r["reference"]["urgency"] in ("URGENT", "EMERGENCY")]
    unsafe = [r for r in auto if r["reference"]["urgency"] in ("URGENT", "EMERGENCY") or r["reference"]["human_review"]]
    return {
        "n": n, "route_counts": counts, "automation_coverage": len(auto) / n,
        "specialty_accuracy_automated": acc(auto, "specialty"), "specialty_accuracy_escalated": acc(manual, "specialty"),
        "urgency_accuracy_automated": acc(auto, "urgency"), "urgency_accuracy_escalated": acc(manual, "urgency"),
        "high_urgency_reference_cases": len(high_ref),
        "high_urgency_reaching_human": sum(1 for r in high_ref if r["route"] != "automated_specialty_route"),
        "high_urgency_human_rate": (sum(1 for r in high_ref if r["route"] != "automated_specialty_route") / len(high_ref)) if high_ref else None,
        "unsafe_automations": len(unsafe),
        "unsafe_automation_ids": [r["case_id"] for r in unsafe][:25],
    }


# ------------------------------------------------------------------ aggregate --
def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(rows)
    ok = valid_rows(rows)
    invalid = [r for r in rows if r.get("error_kind") == "invalid_output"]
    api_err = [r for r in rows if r.get("error_kind") == "api"]
    received = len(ok) + len(invalid)
    metrics: dict[str, Any] = {
        "n_cases": n_total, "n_valid": len(ok), "n_api_errors": len(api_err), "n_invalid_outputs": len(invalid),
        "schema_compliance": (len(ok) / received) if received else None,
        "per_question": {}, "latency": latency_summary(rows), "cost": cost_summary(rows), "tokens": token_summary(rows),
    }
    for q in QUESTIONS:
        metrics["per_question"][q] = categorical_metrics(rows, q) if q in CATEGORICAL else boolean_metrics(rows, q)
    accs = [m["accuracy"] for m in metrics["per_question"].values() if m["accuracy"] is not None]
    metrics["mean_accuracy_across_questions"] = float(np.mean(accs)) if accs else None
    metrics["routing"] = routing_summary(rows)
    return metrics


# ----------------------------------------------------------- paired comparison --
def paired_comparison(
    a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], *, a_name: str = "jev", b_name: str = "claude",
    n_boot: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Per-question accuracy difference (a - b) on the cases BOTH providers answered, with a bootstrap 95% CI.

    The verdict only names a winner if the CI excludes zero; otherwise it says there is no clear difference.
    """
    b_by_id = {r["case_id"]: r for r in valid_rows(b_rows)}
    pairs = [(r, b_by_id[r["case_id"]]) for r in valid_rows(a_rows) if r["case_id"] in b_by_id]
    out: dict[str, Any] = {"n_paired": len(pairs), "a": a_name, "b": b_name, "per_question": {}}
    if not pairs:
        return out
    rng = np.random.default_rng(seed)
    for q in QUESTIONS:
        ca = np.array([is_correct(x, q) for x, _ in pairs], dtype=float)
        cb = np.array([is_correct(y, q) for _, y in pairs], dtype=float)
        diff = ca - cb
        idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boots = diff[idx].mean(axis=1)
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        if lo > 0:
            verdict = f"{a_name} higher (95% CI excludes 0)"
        elif hi < 0:
            verdict = f"{b_name} higher (95% CI excludes 0)"
        else:
            verdict = "no clear difference (95% CI includes 0)"
        out["per_question"][q] = {
            f"{a_name}_accuracy": float(ca.mean()), f"{b_name}_accuracy": float(cb.mean()),
            "difference": float(diff.mean()), "ci95": [lo, hi], "verdict": verdict,
        }
    return out


def nan_to_none(obj: Any) -> Any:
    """Make nested metrics JSON-safe (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
