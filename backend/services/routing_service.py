"""Deterministic, confidence-gated routing policy (the "Python = policy layer").

Jev PRODUCES decisions; this module decides what the SOFTWARE does with them. Jev never triggers anything
directly. All thresholds come from a policy YAML (research-demo values, NOT clinical thresholds).

Two rule sets exist:

CONSERVATIVE (default, config/routing_policy.yaml) - automation is the exception, and every gate must pass.
First match wins:
  R0  invalid / missing model output                      -> human_review         (fail-safe, see fail_safe_routing)
  R1  human-review probability high                       -> escalate_to_human
  R2  infection-concern probability high                  -> urgent_review
  R3  urgent-risk pattern: urgency score, P(URGENT or EMERGENCY), same-day probability,
      or the learned reference-risk score is high         -> urgent_review
  R4  low confidence: urgency / specialty confidence or top specialty probability below its floor,
      or an uncertain answer on infection / same-day / human review      -> human_review
  R5  predicted specialty is on the human-only list       -> urgent_review / human_review
  R6  every gate passed                                   -> automated_specialty_route

LEGACY (config/routing_policy_legacy.yaml) - the v1 baseline, kept only for comparison. It lets clean URGENT
calls through to automation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.schemas.decision import JevDecision
from backend.schemas.routing import RoutingResult
from backend.services import risk_model
from backend.services.policy_service import validate_policy
from backend.utils.config import load_routing_policy
from backend.utils.confidence import confidence_band

BOOLEAN_SAFETY_FEATURES = (("infection concern", "p_infection"), ("same-day review", "p_same_day"), ("human review", "p_human_review"))


@dataclass
class Outcome:
    route: str
    rule: str
    reasons: list[str] = field(default_factory=list)
    specialty: str | None = None


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


# ------------------------------------------------------------------ legacy v1 ----
def _legacy(f: dict[str, Any], th: dict[str, float]) -> Outcome:
    if f["p_human_review"] >= th["human_review_probability"]:
        return Outcome("escalate_to_human", "R1_human_review_probability",
                       [f"Human-review probability {_pct(f['p_human_review'])} >= threshold {_pct(th['human_review_probability'])}."])
    urgent: list[str] = []
    if f["urgency_score"] >= th["urgency_score_urgent"]:
        urgent.append(f"Urgency score {f['urgency_score']:.2f} (scale 0-3) >= threshold {th['urgency_score_urgent']:.2f}.")
    if f["p_same_day"] >= th["same_day_probability"]:
        urgent.append(f"Same-day review probability {_pct(f['p_same_day'])} >= threshold {_pct(th['same_day_probability'])}.")
    if f["p_infection"] >= th["infection_probability_urgent"]:
        urgent.append(f"Infection-concern probability {_pct(f['p_infection'])} >= threshold {_pct(th['infection_probability_urgent'])}.")
    if urgent:
        return Outcome("urgent_review", "R2_urgent_signals", urgent)
    gates = {
        "specialty confidence": (f["specialty_confidence"], th["specialty_confidence"]),
        "top specialty probability": (f["specialty_top_probability"], th["specialty_top_probability"]),
        "urgency confidence": (f["urgency_confidence"], th["urgency_confidence_min"]),
    }
    failed = [f"{k} {_pct(v)} < {_pct(t)}" for k, (v, t) in gates.items() if v < t]
    if not failed:
        return Outcome("automated_specialty_route", "R3_confident_specialty", [
            f"Specialty '{f['specialty']}' with probability {_pct(f['specialty_top_probability'])} and confidence "
            f"{_pct(f['specialty_confidence'])} passed the confidence gates.",
            f"Human-review probability {_pct(f['p_human_review'])} is below the escalation threshold."], f["specialty"])
    return Outcome("human_review", "R4_low_confidence",
                   ["Not confident enough to automate: " + "; ".join(failed) + ". The system does not force an answer."])


# ------------------------------------------------------------ conservative v2 ----
def _conservative(f: dict[str, Any], th: dict[str, float], policy: dict[str, Any]) -> Outcome:
    # R1 - the model itself says a person is needed
    if f["p_human_review"] >= th["human_review_probability"]:
        return Outcome("escalate_to_human", "R1_human_review_probability",
                       [f"Human-review probability {_pct(f['p_human_review'])} >= threshold {_pct(th['human_review_probability'])}."])
    # R2 - any meaningful infection-concern signal goes to a person
    if f["p_infection"] >= th["infection_probability"]:
        return Outcome("urgent_review", "R2_infection_concern",
                       [f"Infection-concern probability {_pct(f['p_infection'])} >= threshold {_pct(th['infection_probability'])}."])
    # R3 - urgent / emergency risk patterns default to a person
    risk = risk_model.score(policy.get("risk_model"), f)
    why: list[str] = []
    if f["urgency_score"] >= th["urgency_score"]:
        why.append(f"Urgency score {f['urgency_score']:.2f} (scale 0-3) >= threshold {th['urgency_score']:.2f}.")
    if f["urgent_tail"] >= th["urgent_tail_probability"]:
        why.append(f"Probability of URGENT or EMERGENCY {_pct(f['urgent_tail'])} >= threshold {_pct(th['urgent_tail_probability'])}.")
    if f["p_same_day"] >= th["same_day_probability"]:
        why.append(f"Same-day review probability {_pct(f['p_same_day'])} >= threshold {_pct(th['same_day_probability'])}.")
    if risk is not None and risk >= th["risk_score"]:
        why.append(f"Learned reference-risk score {risk:.2f} >= threshold {th['risk_score']:.2f} "
                   "(pattern resembles cases the rubric marks as needing a human).")
    if why:
        return Outcome("urgent_review", "R3_urgent_risk", why)
    # R4 - low confidence, or uncertain safety answers
    low: list[str] = []
    for label, key, floor_key in (
        ("urgency confidence", "urgency_confidence", "urgency_confidence_min"),
        ("specialty confidence", "specialty_confidence", "specialty_confidence_min"),
        ("top specialty probability", "specialty_top_probability", "specialty_top_probability_min"),
    ):
        if f[key] < th[floor_key]:
            low.append(f"{label} {_pct(f[key])} < {_pct(th[floor_key])}")
    for label, key in BOOLEAN_SAFETY_FEATURES:
        certainty = abs(2 * f[key] - 1)
        if certainty < th["boolean_certainty_min"]:
            low.append(f"{label} answer is uncertain (p(yes)={_pct(f[key])}, certainty {_pct(certainty)} < {_pct(th['boolean_certainty_min'])})")
    if low:
        return Outcome("human_review", "R4_low_confidence", ["Not confident enough to automate: " + "; ".join(low) + ". The system does not force an answer."])
    # R5 - some destinations are never automated
    if f["specialty"] in policy.get("human_only_specialties", []):
        route = "urgent_review" if f["specialty"] == "emergency_dental_service" else "human_review"
        return Outcome(route, "R5_specialty_requires_human",
                       [f"Specialty '{f['specialty']}' is on the human-only list: a person always reviews these."])
    # R6 - all gates passed
    return Outcome("automated_specialty_route", "R6_all_gates_passed", [
        f"Specialty '{f['specialty']}' (probability {_pct(f['specialty_top_probability'])}, confidence {_pct(f['specialty_confidence'])}) "
        "passed every confidence gate.",
        "No infection, urgency, same-day, human-review or learned-risk signal reached its threshold."], f["specialty"])


def route_features(feats: dict[str, Any], policy: dict[str, Any]) -> Outcome:
    """Pure function of (features, policy). Used by live routing AND by the offline policy study."""
    th = policy["thresholds"]
    return _legacy(feats, th) if policy["rules"] == "legacy" else _conservative(feats, th, policy)


def _checked(policy: dict[str, Any]) -> dict[str, Any]:
    """Validate on every call (a few dict lookups): a looser-than-allowed policy must never route a case."""
    return validate_policy(policy)


def _result(policy: dict[str, Any], out: Outcome, feats: dict[str, Any]) -> RoutingResult:
    labels, spec_labels = policy["route_labels"], policy["specialty_labels"]
    if out.route == "automated_specialty_route":
        label = labels[out.route].format(specialty=spec_labels.get(out.specialty or "", (out.specialty or "").upper()))
    else:
        label = labels[out.route]
    bands = policy["confidence_bands"]
    band = confidence_band(min(feats["specialty_confidence"], feats["urgency_confidence"]), bands["high"], bands["medium"])
    disclaimer = policy.get("disclaimer", "Research-demo thresholds, not clinical thresholds.")
    prov = policy.get("provenance") or {}
    if prov.get("data_mode") == "demo":
        disclaimer += " Thresholds were selected on SIMULATED model outputs; re-tune on a live benchmark before relying on them."
    return RoutingResult(
        route=out.route, workflow_label=label, destination_specialty=out.specialty, triggered_rule=out.rule,
        reasons=out.reasons, confidence_band=band, requires_human=out.route != "automated_specialty_route",
        thresholds=dict(policy["thresholds"]), policy_name=policy.get("name", "legacy"), disclaimer=disclaimer,
    )


def route_decision(decision: JevDecision, policy: dict[str, Any] | None = None) -> RoutingResult:
    policy = _checked(policy or load_routing_policy())
    feats = risk_model.features_from_decision(decision)
    return _result(policy, route_features(feats, policy), feats)


def fail_safe_routing(reason: str, policy: dict[str, Any] | None = None) -> RoutingResult:
    """R0: whenever a model output is missing or invalid, the answer is a human. Never an automated route."""
    policy = policy or load_routing_policy()
    return RoutingResult(
        route="human_review", workflow_label=policy["route_labels"]["human_review"], destination_specialty=None,
        triggered_rule="R0_fail_safe",
        reasons=[f"{reason.rstrip('.')}. No automated action was taken; a person must review this case."],
        confidence_band="low", requires_human=True, thresholds={}, policy_name=policy.get("name", "legacy"),
        disclaimer=policy.get("disclaimer", "Research-demo thresholds, not clinical thresholds."),
    )
