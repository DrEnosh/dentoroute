"""Benchmark runner: Jev vs Claude on the SAME synthetic cases, scored against the deterministic rubric.

Honesty rules baked in:
 * Demo-mode runs are labelled SIMULATED (the demo simulator is not Jev) and are never compared with Claude.
 * Cost is "not measured" unless token prices are configured; latency is only recorded for real API calls.
 * The paired comparison only names a winner if the bootstrap CI excludes zero.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from backend.repositories import benchmark_repo
from backend.schemas.case import SyntheticCase
from backend.schemas.decision import ClaudeDecision, JevDecision
from backend.services import calibration_service, claude_service, jev_service, metrics_service, routing_service
from backend.services.ground_truth_service import label_case
from backend.services.synthetic_case_service import load_dataset
from backend.utils.config import get_settings
from backend.utils.errors import BenchmarkError, DentoRouteError, InvalidModelOutputError, MissingApiKeyError
from backend.utils.logging_utils import get_logger, log_event

logger = get_logger("benchmark")
ProgressFn = Callable[[int, int], None]


# ---------------------------------------------------------------- case sampling --
def select_cases(limit: int | None = None, seed: int = 0, cases: list[SyntheticCase] | None = None) -> list[SyntheticCase]:
    """All cases, or a seeded category-stratified subset (round-robin over categories)."""
    cases = cases if cases is not None else load_dataset()
    if limit is None or limit >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    by_cat: dict[str, list[SyntheticCase]] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    for v in by_cat.values():
        rng.shuffle(v)
    cats = sorted(by_cat)
    picked: list[SyntheticCase] = []
    while len(picked) < limit and any(by_cat.values()):
        for cat in cats:
            if by_cat[cat] and len(picked) < limit:
                picked.append(by_cat[cat].pop())
    return picked


# -------------------------------------------------------------------- row builders --
def _reference(case: SyntheticCase) -> dict[str, Any]:
    ref = case.reference or label_case(case.state)
    return {q: getattr(ref, q) for q in metrics_service.QUESTIONS}


def _base_row(case: SyntheticCase, provider: str, model: str, mode: str) -> dict[str, Any]:
    return {
        "case_id": case.patient_id, "category": case.category, "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider, "model": model, "mode": mode, "reference": _reference(case),
        "predictions": None, "probabilities": {}, "confidences": {}, "scores": {}, "route": None,
        "latency_ms": None, "cached": False, "usage": None, "cost_usd": None, "schema_valid": False,
        "error": None, "error_kind": None, "confidence_kind": None,
    }


def jev_row(case: SyntheticCase, decision: JevDecision, mode: str) -> dict[str, Any]:
    s = get_settings()
    row = _base_row(case, "jev", decision.model, mode)
    d = decision
    row.update(
        predictions={
            "urgency": d.urgency.label, "specialty": d.specialty.label,
            "infection_concern": d.infection_concern.prediction, "same_day_review": d.same_day_review.prediction,
            "human_review": d.human_review.prediction, "complexity": d.complexity.label,
            "referral_priority": d.referral_priority.label,
        },
        probabilities={
            "urgency": d.urgency.probabilities, "specialty": d.specialty.probabilities,
            "complexity": d.complexity.probabilities, "referral_priority": d.referral_priority.probabilities,
            "infection_concern": d.infection_concern.probability, "same_day_review": d.same_day_review.probability,
            "human_review": d.human_review.probability,
        },
        confidences={
            "urgency": d.urgency.confidence, "specialty": d.specialty.confidence,
            "complexity": d.complexity.confidence, "referral_priority": d.referral_priority.confidence,
        },
        scores={"urgency": d.urgency.score, "complexity": d.complexity.score, "referral_priority": d.referral_priority.score},
        route=routing_service.route_decision(d).route,
        latency_ms=d.latency_ms, cached=d.cached, usage=d.usage, schema_valid=True, confidence_kind="calibrated_distribution",
        cost_usd=claude_service.estimate_cost(d.usage, s.jev_price_input_per_mtok, s.jev_price_output_per_mtok),
    )
    return row


def claude_row(case: SyntheticCase, d: ClaudeDecision) -> dict[str, Any]:
    s = get_settings()
    row = _base_row(case, "claude", d.model, "live")
    row.update(
        predictions={
            "urgency": d.urgency, "specialty": d.specialty, "infection_concern": d.infection_concern,
            "same_day_review": d.same_day_review, "human_review": d.human_review,
            "complexity": d.complexity, "referral_priority": d.referral_priority,
        },
        probabilities={k: v for k, v in {
            "infection_concern": d.infection_probability, "same_day_review": d.same_day_probability,
            "human_review": d.human_review_probability,
        }.items() if v is not None},
        confidences={k: v for k, v in {"urgency": d.urgency_confidence, "specialty": d.specialty_confidence}.items() if v is not None},
        latency_ms=d.latency_ms, usage=d.usage, schema_valid=True, confidence_kind="verbalized",
        cost_usd=claude_service.estimate_cost(d.usage, s.claude_price_input_per_mtok, s.claude_price_output_per_mtok),
    )
    return row


def error_row(case: SyntheticCase, provider: str, model: str, mode: str, exc: Exception) -> dict[str, Any]:
    row = _base_row(case, provider, model, mode)
    row["error"] = exc.user_message if isinstance(exc, DentoRouteError) else "Unexpected error"
    row["error_kind"] = "invalid_output" if isinstance(exc, InvalidModelOutputError) else "api"
    return row


# ------------------------------------------------------------------------ runners --
def _run_parallel(cases: list[SyntheticCase], fn: Callable[[SyntheticCase], dict[str, Any]], workers: int,
                  progress: ProgressFn | None) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, c): c for c in cases}
        for fut in as_completed(futures):
            c = futures[fut]
            rows[c.patient_id] = fut.result()
            done += 1
            if progress:
                progress(done, len(cases))
    return [rows[c.patient_id] for c in cases]  # keep dataset order


def run_jev(cases: list[SyntheticCase], demo: bool, workers: int, progress: ProgressFn | None = None,
            service: jev_service.JevService | None = None) -> list[dict[str, Any]]:
    svc = service or jev_service.get_jev_service()
    mode = "demo" if demo else "live"
    model = "demo-simulator (NOT Jev)" if demo else get_settings().jev_model

    def one(case: SyntheticCase) -> dict[str, Any]:
        try:
            return jev_row(case, svc.evaluate(case.state, demo=demo, use_cache=True), mode)
        except DentoRouteError as exc:
            return error_row(case, "jev", model, mode, exc)
        except Exception as exc:  # noqa: BLE001 - a single bad case must not abort the run
            logger.warning("unexpected benchmark failure on %s: %s", case.patient_id, exc.__class__.__name__)
            return error_row(case, "jev", model, mode, exc)

    return _run_parallel(cases, one, workers, progress)


def run_claude(cases: list[SyntheticCase], workers: int, progress: ProgressFn | None = None,
               service: claude_service.ClaudeService | None = None) -> list[dict[str, Any]]:
    svc = service or claude_service.get_claude_service()
    if not svc.available:
        raise MissingApiKeyError("ANTHROPIC_API_KEY is not set, so the Claude benchmark cannot run.")
    model = get_settings().claude_model

    def one(case: SyntheticCase) -> dict[str, Any]:
        try:
            return claude_row(case, svc.decide(case.state))
        except DentoRouteError as exc:
            return error_row(case, "claude", model, "live", exc)
        except Exception as exc:  # noqa: BLE001
            return error_row(case, "claude", model, "live", exc)

    return _run_parallel(cases, one, workers, progress)


# ------------------------------------------------------------------- orchestration --
def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    policy_bands = __import__("backend.utils.config", fromlist=["load_routing_policy"]).load_routing_policy()["confidence_bands"]
    return metrics_service.nan_to_none({
        "simulated": mode == "demo",
        "metrics": metrics_service.compute_metrics(rows),
        "calibration": calibration_service.calibration_report(rows, high=policy_bands["high"], medium=policy_bands["medium"]),
    })


def run_benchmark(
    provider: str = "jev", limit: int | None = None, demo: bool | None = None, workers: int | None = None,
    seed: int = 0, progress: ProgressFn | None = None, cases: list[SyntheticCase] | None = None,
) -> dict[str, Any]:
    """Run, persist and summarise a benchmark. Returns {'runs': {provider: run_dict}, 'skipped': {...}, 'comparison': ...}."""
    if provider not in ("jev", "claude", "both"):
        raise BenchmarkError("provider must be 'jev', 'claude' or 'both'.")
    settings = get_settings()
    workers = workers or settings.benchmark_workers
    selected = select_cases(limit, seed, cases)
    if not selected:
        raise BenchmarkError("No cases to benchmark.")
    is_demo = settings.resolve_demo_mode(demo)
    runs: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    if provider in ("jev", "both"):
        runs["jev"] = _execute("jev", selected, is_demo, workers, progress)
    if provider in ("claude", "both"):
        if provider == "both" and is_demo:
            skipped["claude"] = ("Not run: Demo Mode simulates Jev, so a Jev-vs-Claude comparison would be meaningless. "
                                 "Set TYPESAFE_API_KEY and DENTOROUTE_DEMO_MODE=false for a real comparison.")
        elif not claude_service.get_claude_service().available:
            if provider == "claude":
                raise MissingApiKeyError("ANTHROPIC_API_KEY is not set, so the Claude benchmark cannot run.")
            skipped["claude"] = "Not run: ANTHROPIC_API_KEY is not set."
        else:
            runs["claude"] = _execute("claude", selected, False, workers, progress)

    comparison = None
    if "jev" in runs and "claude" in runs and runs["jev"]["mode"] == "live":
        comparison = build_comparison(runs["jev"], runs["claude"])
    return {"runs": runs, "skipped": skipped, "comparison": comparison}


def _execute(provider: str, cases: list[SyntheticCase], demo: bool, workers: int, progress: ProgressFn | None) -> dict[str, Any]:
    settings = get_settings()
    mode = "demo" if demo else "live"
    model = ("demo-simulator (NOT Jev)" if demo else settings.jev_model) if provider == "jev" else settings.claude_model
    run_id = benchmark_repo.create_run(provider, mode, model)
    log_event(logger, "benchmark_start", case_id="*", model=model, success=True, validation=f"{provider}:{mode}:{len(cases)}")
    rows = run_jev(cases, demo, workers, progress) if provider == "jev" else run_claude(cases, workers, progress)
    summary = summarize(rows, mode)
    n_err = sum(1 for r in rows if r["error"])
    benchmark_repo.save_results(run_id, rows)
    benchmark_repo.finish_run(run_id, len(rows), n_err, summary)
    log_event(logger, "benchmark_done", case_id="*", model=model, success=True, validation=f"errors={n_err}/{len(rows)}")
    return {"run_id": run_id, "provider": provider, "mode": mode, "model": model, "n_cases": len(rows),
            "n_errors": n_err, "summary": summary, "rows": rows}


def build_comparison(jev_run: dict[str, Any], claude_run: dict[str, Any]) -> dict[str, Any]:
    paired = metrics_service.paired_comparison(jev_run["rows"], claude_run["rows"])
    jm, cm = jev_run["summary"]["metrics"], claude_run["summary"]["metrics"]
    return metrics_service.nan_to_none({
        "paired": paired,
        "table": {
            "latency": {"jev": jm["latency"], "claude": cm["latency"]},
            "cost": {"jev": jm["cost"], "claude": cm["cost"]},
            "tokens": {"jev": jm["tokens"], "claude": cm["tokens"]},
            "schema_compliance": {"jev": jm["schema_compliance"], "claude": cm["schema_compliance"]},
            "invalid_outputs": {"jev": jm["n_invalid_outputs"], "claude": cm["n_invalid_outputs"]},
            "api_errors": {"jev": jm["n_api_errors"], "claude": cm["n_api_errors"]},
            "probability_distributions": {"jev": "yes (per-class probabilities)", "claude": "no (self-reported confidence only)"},
        },
        "caveat": ("Both providers are scored against a deterministic synthetic rubric that was written by the same "
                   "author as the prompts. Differences describe this benchmark, not clinical performance."),
    })


def get_results() -> dict[str, Any]:
    """Latest stored runs (any mode) plus a comparison when a live Jev run and a Claude run both exist."""
    jev_live = benchmark_repo.latest_run("jev", "live")
    jev_any = jev_live or benchmark_repo.latest_run("jev")
    claude = benchmark_repo.latest_run("claude")
    comparison = build_comparison(jev_live, claude) if jev_live and claude and jev_live["summary"] and claude["summary"] else None
    return {"jev": jev_any, "claude": claude, "comparison": comparison}
