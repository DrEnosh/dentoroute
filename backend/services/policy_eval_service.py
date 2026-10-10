"""Cost-sensitive evaluation, threshold selection and old-vs-new comparison for the routing safety gate.

Data discipline (enforced in code, not just by convention):
  * `tune()` refuses any row whose split is not train/calibration (HoldoutLeakError).
  * The risk model is fit on TRAIN only. Thresholds are chosen on CALIBRATION only.
  * The holdout split is evaluated once, after the policy is frozen.

Definitions (reference = the deterministic synthetic rubric, not clinical truth):
  urgent         reference urgency is URGENT or EMERGENCY
  needs_human    urgent OR reference infection concern OR same-day review OR human review
  automated      the policy chose `automated_specialty_route`
  missed urgent  automated AND urgent                      <- the headline safety metric
  false automation  automated AND needs_human (a superset of missed urgent)
  unnecessary review  routed to a human AND NOT needs_human
  coverage       automated / all cases
  automation accuracy  accuracy of the specialty / urgency answers among the automated cases
All thresholds are research-demo values, not clinical thresholds.
"""
from __future__ import annotations

import copy
import itertools
from typing import Any

import numpy as np
from scipy.stats import beta

from backend.services import risk_model
from backend.services.metrics_service import nan_to_none
from backend.services.policy_service import MAX_LIMITS, MIN_LIMITS, provenance_stub, validate_policy
from backend.services.routing_service import route_features
from backend.services.split_service import FRACTIONS, SEED, SPLIT_NAMES, TUNING_SPLITS, attach_splits, get_splits
from backend.utils.config import load_routing_policy
from backend.utils.errors import BenchmarkError, DentoRouteError

AUTOMATED = "automated_specialty_route"
DEFAULT_COSTS = {"missed_urgent": 100.0, "other_false_automation": 20.0, "unnecessary_review": 1.0}
COST_RATIOS = (2, 5, 10, 50, 100, 500)  # missed-urgent : unnecessary-review cost ratios for the sensitivity table


class HoldoutLeakError(DentoRouteError):
    code = "holdout_leak"


class PolicyTuningError(BenchmarkError):  # a user-fixable input problem: HTTP 400
    code = "policy_tuning_error"


# ------------------------------------------------------------------ labels -----
def is_urgent(ref: dict[str, Any]) -> bool:
    return ref["urgency"] in ("URGENT", "EMERGENCY")


def needs_human(ref: dict[str, Any]) -> bool:
    return is_urgent(ref) or bool(ref["infection_concern"] or ref["same_day_review"] or ref["human_review"])


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float | None:
    """One-sided (1 - alpha) upper bound on a rate seen k times in n trials. Zero misses is not zero risk."""
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


# ------------------------------------------------------------- evaluation ------
def _prepare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in rows if r.get("predictions")]
    return ok


def evaluate(rows: list[dict[str, Any]], policy: dict[str, Any], costs: dict[str, float] | None = None) -> dict[str, Any]:
    costs = costs or DEFAULT_COSTS
    rows = _prepare(rows)
    n = len(rows)
    outs = [route_features(risk_model.features_from_row(r), policy) for r in rows]
    auto = [o.route == AUTOMATED for o in outs]
    refs = [r["reference"] for r in rows]
    urgent = [is_urgent(x) for x in refs]
    need = [needs_human(x) for x in refs]

    n_auto = sum(auto)
    n_urgent, n_need = sum(urgent), sum(need)
    n_noneed = n - n_need
    missed = sum(a and u for a, u in zip(auto, urgent))
    false_auto = sum(a and nd for a, nd in zip(auto, need))
    unnecessary = sum((not a) and (not nd) for a, nd in zip(auto, need))
    other_fa = false_auto - missed
    legacy_unsafe = sum(a and (u or x["human_review"]) for a, u, x in zip(auto, urgent, refs))
    auto_rows = [r for r, a in zip(rows, auto) if a]

    def acc(q: str) -> float | None:
        return float(np.mean([r["predictions"][q] == r["reference"][q] for r in auto_rows])) if auto_rows else None

    joint = float(np.mean([r["predictions"]["specialty"] == r["reference"]["specialty"] and r["predictions"]["urgency"] == r["reference"]["urgency"]
                           for r in auto_rows])) if auto_rows else None
    cost = costs["missed_urgent"] * missed + costs["other_false_automation"] * other_fa + costs["unnecessary_review"] * unnecessary
    counts: dict[str, int] = {}
    for o in outs:
        counts[o.route] = counts.get(o.route, 0) + 1
    return {
        "n": n, "n_automated": n_auto, "n_human": n - n_auto,
        "coverage": n_auto / n if n else None, "human_review_rate": (n - n_auto) / n if n else None,
        "n_urgent_reference": n_urgent, "missed_urgent": missed,
        "missed_urgent_rate": missed / n_urgent if n_urgent else None,
        "missed_urgent_upper95": clopper_pearson_upper(missed, n_urgent),
        "n_needs_human_reference": n_need, "false_automation": false_auto,
        "false_automation_rate": false_auto / n_need if n_need else None, "false_automation_other": other_fa,
        "n_no_human_needed_reference": n_noneed, "unnecessary_human_review": unnecessary,
        "unnecessary_review_rate": unnecessary / n_noneed if n_noneed else None,
        "automation_accuracy_specialty": acc("specialty"), "automation_accuracy_urgency": acc("urgency"),
        "automation_accuracy_joint": joint, "legacy_unsafe_automations": legacy_unsafe,
        "cost": cost, "cost_per_case": cost / n if n else None, "route_counts": counts,
        "missed_urgent_ids": [r["case_id"] for r, a, u in zip(rows, auto, urgent) if a and u],
    }


# ---------------------------------------------------------------- tuning -------
def _assert_tuning_rows(rows: list[dict[str, Any]], allowed: tuple[str, ...] = TUNING_SPLITS) -> None:
    bad = sorted({r.get("split", "<none>") for r in rows if r.get("split") not in allowed})
    if bad:
        raise HoldoutLeakError(f"Threshold selection may only use {list(allowed)} rows; got {bad}. The holdout split must stay untouched.")


def _cautious_position(key: str, idx: int, n: int) -> int:
    """Higher = more cautious. Trigger thresholds are cautious when low; floors are cautious when high."""
    return (n - 1 - idx) if key in MAX_LIMITS else idx


def _fast_cost(feats: list[dict[str, Any]], urgent: list[bool], need: list[bool], policy: dict[str, Any], costs: dict[str, float]) -> tuple[int, int, int, int]:
    missed = other = unnec = auto_n = 0
    for f, u, nd in zip(feats, urgent, need):
        if route_features(f, policy).route == AUTOMATED:
            auto_n += 1
            if u:
                missed += 1
            elif nd:
                other += 1
        elif not nd:
            unnec += 1
    return missed, other, unnec, auto_n


def tune(
    rows_train: list[dict[str, Any]], rows_cal: list[dict[str, Any]], base_policy: dict[str, Any] | None = None,
    costs: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the risk model on TRAIN, pick thresholds on CALIBRATION. Returns (policy, selection_report)."""
    _assert_tuning_rows(rows_train, ("train",))
    _assert_tuning_rows(rows_cal, ("calibration",))
    base = copy.deepcopy(base_policy or load_routing_policy("conservative"))
    cfg = base["tuning"]
    costs = costs or cfg.get("costs") or DEFAULT_COSTS
    budget = int(cfg.get("budget_missed_urgent_calibration", 0))
    margin = int(cfg.get("safety_margin_steps", 1))
    grid: dict[str, list[float]] = {k: sorted(cfg["grid"][k]) for k in cfg["tunable"]}
    for k, vals in grid.items():  # the search space itself can never exceed the hard limits
        if (k in MAX_LIMITS and max(vals) > MAX_LIMITS[k]) or (k in MIN_LIMITS and min(vals) < MIN_LIMITS[k]):
            raise PolicyTuningError(f"Tuning grid for '{k}' includes values looser than the hard safety limit.")

    train = [r for r in rows_train if r.get("predictions")]
    cal = [r for r in rows_cal if r.get("predictions")]
    if not train or not cal:
        raise PolicyTuningError("Need non-empty train and calibration sets.")
    base["risk_model"] = risk_model.fit([risk_model.features_from_row(r) for r in train], [needs_human(r["reference"]) for r in train])

    feats = [risk_model.features_from_row(r) for r in cal]
    urgent = [is_urgent(r["reference"]) for r in cal]
    need = [needs_human(r["reference"]) for r in cal]
    keys = list(grid)
    results = []
    for combo in itertools.product(*[range(len(grid[k])) for k in keys]):
        cand = copy.deepcopy(base)
        for k, i in zip(keys, combo):
            cand["thresholds"][k] = grid[k][i]
        missed, other, unnec, auto_n = _fast_cost(feats, urgent, need, cand, costs)
        cost = costs["missed_urgent"] * missed + costs["other_false_automation"] * other + costs["unnecessary_review"] * unnec
        caution = sum(_cautious_position(k, i, len(grid[k])) for k, i in zip(keys, combo))
        results.append({"combo": combo, "missed": missed, "other": other, "unnec": unnec, "auto": auto_n, "cost": cost, "caution": caution})

    feasible = [r for r in results if r["missed"] <= budget]
    if feasible:
        best = min(feasible, key=lambda r: (r["cost"], -r["caution"], -r["auto"]))
    else:  # nothing meets the budget: be as safe as the grid allows and say so
        best = min(results, key=lambda r: (r["missed"], r["cost"], -r["caution"]))

    chosen = {k: grid[k][i] for k, i in zip(keys, best["combo"])}
    final = copy.deepcopy(base)
    final["thresholds"].update(chosen)
    stepped: dict[str, float] = {}
    for k, i in zip(keys, best["combo"]):  # safety margin: move each tuned threshold toward caution
        j = max(0, i - margin) if k in MAX_LIMITS else min(len(grid[k]) - 1, i + margin)
        stepped[k] = grid[k][j]
    final["thresholds"].update(stepped)
    validate_policy(final)

    # calibration-set Pareto frontier: most coverage for each number of missed urgent cases
    best_cov: dict[int, int] = {}
    for r in results:
        best_cov[r["missed"]] = max(best_cov.get(r["missed"], 0), r["auto"])
    frontier, running = [], -1
    for m in sorted(best_cov):
        if best_cov[m] > running:
            running = best_cov[m]
            frontier.append({"missed_urgent": m, "coverage": best_cov[m] / len(cal)})

    report = {
        "n_candidates": len(results), "n_feasible": len(feasible), "feasible": bool(feasible),
        "budget_missed_urgent_calibration": budget, "safety_margin_steps": margin, "costs": costs,
        "selected_before_margin": chosen, "selected_final": {k: final["thresholds"][k] for k in keys},
        "calibration_before_margin": {"missed_urgent": best["missed"], "other_false_automation": best["other"],
                                      "unnecessary_review": best["unnec"], "automated": best["auto"], "n": len(cal)},
        "calibration_frontier": frontier, "risk_model": {k: base["risk_model"][k] for k in ("n_train", "n_positive", "regularization_c")},
        "grid": grid,
        "note": None if feasible else "No threshold combination met the missed-urgent budget on the calibration set; the most cautious option was chosen.",
    }
    return final, report


# ------------------------------------------------------------ comparison -------
def _case_costs(rows: list[dict[str, Any]], policy: dict[str, Any], costs: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    c, a = [], []
    for r in rows:
        out = route_features(risk_model.features_from_row(r), policy)
        auto = out.route == AUTOMATED
        u, nd = is_urgent(r["reference"]), needs_human(r["reference"])
        c.append(costs["missed_urgent"] if auto and u else costs["other_false_automation"] if auto and nd else costs["unnecessary_review"] if (not auto and not nd) else 0.0)
        a.append(1.0 if auto else 0.0)
    return np.array(c), np.array(a)


def _paired_bootstrap(old: np.ndarray, new: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    diff = new - old
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return {"mean_difference_new_minus_old": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def run_policy_study(rows: list[dict[str, Any]], *, data_mode: str, model: str, costs: dict[str, float] | None = None) -> dict[str, Any]:
    """Tune on train+calibration, then evaluate legacy vs revised on every split, headline = holdout."""
    splits = get_splits()
    rows = attach_splits([r for r in rows if r.get("predictions")], splits)
    by = {s: [r for r in rows if r["split"] == s] for s in SPLIT_NAMES}
    # A subset would give a thin holdout (maybe no urgent cases at all), so require near-complete coverage of every split.
    sizes = splits.get("counts") or {s: sum(1 for v in splits["assignment"].values() if v == s) for s in SPLIT_NAMES}
    if any(len(by[s]) < 0.9 * sizes[s] for s in SPLIT_NAMES):
        raise PolicyTuningError("The benchmark run does not cover all three splits. Run the full benchmark (all cases) first.")
    base = load_routing_policy("conservative")
    costs = costs or base["tuning"].get("costs") or DEFAULT_COSTS
    legacy = load_routing_policy("legacy")

    tuned, report = tune(by["train"], by["calibration"], base, costs)  # holdout rows are never passed in

    def both(rs: list[dict[str, Any]]) -> dict[str, Any]:
        return {"legacy": evaluate(rs, legacy, costs), "revised": evaluate(rs, tuned, costs)}

    per_split = {s: both(by[s]) for s in SPLIT_NAMES}
    per_split["all"] = both(rows)  # context only; includes the tuning data

    hold = by["holdout"]
    sens = []
    for ratio in COST_RATIOS:
        cc = {"missed_urgent": float(ratio), "other_false_automation": ratio / 5, "unnecessary_review": 1.0}
        sens.append({"missed_to_unnecessary_cost_ratio": ratio,
                     "legacy_cost_per_case": evaluate(hold, legacy, cc)["cost_per_case"],
                     "revised_cost_per_case": evaluate(hold, tuned, cc)["cost_per_case"]})
    old_c, old_a = _case_costs(hold, legacy, costs)
    new_c, new_a = _case_costs(hold, tuned, costs)
    study = {
        "headline_split": "holdout",
        "holdout_used_for_selection": False,
        "simulated": data_mode == "demo",
        "provenance": {
            "data_mode": data_mode, "model": model, "split_seed": SEED, "split_fractions": FRACTIONS,
            "split_counts": {s: len(by[s]) for s in SPLIT_NAMES}, "tuned_on": list(TUNING_SPLITS),
            "reference": "deterministic synthetic rubric (not clinical truth)",
        },
        "definitions": {
            "urgent": "reference urgency URGENT or EMERGENCY",
            "needs_human": "urgent, or reference infection concern / same-day review / human review",
            "false_automation": "automated AND needs_human", "missed_urgent": "automated AND urgent",
            "unnecessary_review": "sent to a human AND NOT needs_human",
        },
        "costs": costs, "selection": report, "per_split": per_split, "cost_sensitivity_holdout": sens,
        "paired_holdout": {"cost_per_case": _paired_bootstrap(old_c, new_c), "coverage": _paired_bootstrap(old_a, new_a)},
        "tuned_policy": {"thresholds": tuned["thresholds"], "human_only_specialties": tuned["human_only_specialties"]},
        "limits": {"max": MAX_LIMITS, "min": MIN_LIMITS},
        "_policy": tuned,  # full policy (with risk model) for scripts; stripped before storing/returning via API
    }
    return study


def public_study(study: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in study.items() if k != "_policy"}
    return nan_to_none(out)


def policy_for_writing(study: dict[str, Any]) -> dict[str, Any]:
    pol = copy.deepcopy(study["_policy"])
    sel = study["selection"]
    pol["provenance"] = provenance_stub(
        data_mode=study["provenance"]["data_mode"], model=study["provenance"]["model"],
        simulated=study["simulated"], tuned_on=study["provenance"]["tuned_on"], holdout_used_for_selection=False,
        split_seed=SEED, split_counts=study["provenance"]["split_counts"], n_candidates=sel["n_candidates"],
        feasible_on_calibration=sel["feasible"], selection_objective="min expected cost s.t. missed urgent <= budget on calibration",
        costs=study["costs"], budget_missed_urgent_calibration=sel["budget_missed_urgent_calibration"],
        safety_margin_steps=sel["safety_margin_steps"],
        warning=("Selected on SIMULATED Jev outputs. Re-run scripts/tune_policy.py on a live benchmark before relying on these numbers."
                 if study["simulated"] else "Selected on live Jev outputs from a synthetic benchmark. Still research-demo, not clinical."),
    )
    validate_policy(pol)
    return pol


# ----------------------------------------------------------- orchestration -----
def run_and_store_study() -> dict[str, Any]:
    """Study on the latest stored Jev benchmark (live preferred). Evaluates a CANDIDATE policy; never changes the active one."""
    from backend.repositories import benchmark_repo
    from backend.utils.errors import BenchmarkError

    run = benchmark_repo.latest_run("jev", "live") or benchmark_repo.latest_run("jev")
    if run is None:
        raise BenchmarkError("No stored Jev benchmark. Run the benchmark on all cases first (Benchmark Lab or scripts/run_benchmark.py).")
    study = run_policy_study(run["rows"], data_mode=run["mode"], model=run["model"])
    run_id = benchmark_repo.create_run("policy", run["mode"], run["model"])
    benchmark_repo.finish_run(run_id, run["n_cases"], 0, public_study(study))
    return study


def latest_study() -> dict[str, Any] | None:
    from backend.repositories import benchmark_repo

    run = benchmark_repo.latest_run("policy")
    return run["summary"] if run else None
