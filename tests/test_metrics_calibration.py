"""Metrics and calibration math, checked against small hand-computed examples."""
from __future__ import annotations

import pytest

from backend.services import calibration_service as cal
from backend.services import metrics_service as m

REF = dict(urgency="URGENT", specialty="endodontics", infection_concern=True, same_day_review=False,
           human_review=False, complexity="MODERATE", referral_priority="HIGH")


def row(case_id="C1", pred=None, probs=None, conf=None, ref=None, route=None, **extra):
    pred = {**REF, **(pred or {})}
    r = dict(case_id=case_id, category="x", provider="jev", reference={**REF, **(ref or {})}, predictions=pred,
             probabilities=probs or {}, confidences=conf or {}, route=route, latency_ms=None, cached=False,
             usage=None, cost_usd=None, schema_valid=True, error=None, error_kind=None)
    r.update(extra)
    return r


def fail(case_id, kind):
    return dict(case_id=case_id, reference=REF, predictions=None, probabilities={}, confidences={}, route=None,
                error="x", error_kind=kind, latency_ms=None, usage=None, cost_usd=None, cached=False)


# --------------------------------------------------------------- categorical ---
def test_accuracy_within_one_and_mae():
    rows = [row("a"), row("b", pred={"urgency": "PRIORITY"}), row("c", pred={"urgency": "ROUTINE"})]
    u = m.categorical_metrics(rows, "urgency")
    assert u["accuracy"] == pytest.approx(1 / 3)
    assert u["within_one"] == pytest.approx(2 / 3)  # URGENT vs PRIORITY is 1 off, vs ROUTINE is 2 off
    assert u["mae_levels"] == pytest.approx((0 + 1 + 2) / 3)
    assert u["under_triage_rate"] == pytest.approx(2 / 3) and u["over_triage_rate"] == 0


def test_confusion_matrix_orientation_rows_are_reference():
    rows = [row("a", pred={"urgency": "EMERGENCY"})]
    u = m.categorical_metrics(rows, "urgency")
    assert u["confusion_matrix"][2][3] == 1  # reference URGENT(2) predicted EMERGENCY(3)


def test_specialty_top3_uses_probabilities_and_is_none_without():
    probs = {"specialty": {"general_dentistry": 0.5, "periodontics": 0.3, "oral_surgery": 0.15, "endodontics": 0.05}}
    r = row("a", pred={"specialty": "general_dentistry"}, probs=probs)
    s = m.categorical_metrics([r], "specialty")
    assert s["accuracy"] == 0 and s["top3_accuracy"] == 0  # endodontics is 4th
    probs["specialty"]["endodontics"], probs["specialty"]["oral_surgery"] = 0.15, 0.05
    assert m.categorical_metrics([r], "specialty")["top3_accuracy"] == 1
    assert m.categorical_metrics([row("b")], "specialty")["top3_accuracy"] is None


# ------------------------------------------------------------------- boolean ---
def test_boolean_precision_recall_f1_brier():
    rows = [
        row("a", pred={"infection_concern": True}, probs={"infection_concern": 0.9}),                      # TP
        row("b", pred={"infection_concern": True}, ref={"infection_concern": False}, probs={"infection_concern": 0.7}),  # FP
        row("c", pred={"infection_concern": False}, probs={"infection_concern": 0.2}),                     # FN
        row("d", pred={"infection_concern": False}, ref={"infection_concern": False}, probs={"infection_concern": 0.1}),  # TN
    ]
    b = m.boolean_metrics(rows, "infection_concern")
    assert (b["precision"], b["recall"], b["f1"]) == (0.5, 0.5, 0.5)
    assert b["confusion_matrix"] == [[1, 1], [1, 1]]
    assert b["brier"] == pytest.approx((0.01 + 0.49 + 0.64 + 0.01) / 4)


# --------------------------------------------------------------- aggregate -----
def test_schema_compliance_and_error_accounting():
    rows = [row("a"), row("b"), fail("c", "invalid_output"), fail("d", "api")]
    res = m.compute_metrics(rows)
    assert res["n_cases"] == 4 and res["n_valid"] == 2
    assert res["n_invalid_outputs"] == 1 and res["n_api_errors"] == 1
    assert res["schema_compliance"] == pytest.approx(2 / 3)  # network errors do not count as schema failures
    assert res["per_question"]["urgency"]["n"] == 2


def test_latency_and_cost_not_measured_unless_present():
    assert m.latency_summary([row()])["status"] == "not measured"
    assert m.cost_summary([row()])["status"] == "not measured"
    rows = [row("a", latency_ms=100.0), row("b", latency_ms=300.0), row("c", latency_ms=999.0, cached=True)]
    lat = m.latency_summary(rows)
    assert lat["n"] == 2 and lat["mean_ms"] == 200.0  # cached responses do not count as latency
    assert m.cost_summary([row(cost_usd=0.01), row("b", cost_usd=0.03)])["total_usd"] == pytest.approx(0.04)


def test_routing_summary_flags_unsafe_automation():
    auto = "automated_specialty_route"
    rows = [
        row("a", route=auto),                                                      # URGENT reference automated -> unsafe
        row("b", route=auto, ref={"urgency": "ROUTINE", "human_review": False}),   # fine
        row("c", route="urgent_review"),
    ]
    s = m.routing_summary(rows)
    assert s["automation_coverage"] == pytest.approx(2 / 3)
    assert s["unsafe_automations"] == 1 and s["unsafe_automation_ids"] == ["a"]
    assert s["high_urgency_human_rate"] == pytest.approx(1 / 2)


def test_paired_comparison_does_not_declare_winner_on_ties_or_tiny_samples():
    a = [row(f"c{i}") for i in range(20)]
    b = [row(f"c{i}") for i in range(20)]
    res = m.paired_comparison(a, b)
    assert res["n_paired"] == 20
    assert all("no clear difference" in v["verdict"] for v in res["per_question"].values())


def test_paired_comparison_detects_a_large_real_gap():
    a = [row(f"c{i}") for i in range(80)]
    b = [row(f"c{i}", pred={"urgency": "ROUTINE"}) for i in range(80)]
    v = m.paired_comparison(a, b)["per_question"]["urgency"]
    assert v["difference"] == 1.0 and "jev higher" in v["verdict"]


def test_paired_comparison_only_uses_cases_both_answered():
    a = [row("c1"), row("c2")]
    b = [row("c1"), fail("c2", "api")]
    assert m.paired_comparison(a, b)["n_paired"] == 1


def test_nan_to_none():
    assert m.nan_to_none({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}


# --------------------------------------------------------------- calibration ---
def test_perfectly_calibrated_samples_have_zero_ece():
    samples = [(0.8, 1)] * 8 + [(0.8, 0)] * 2
    assert cal.expected_calibration_error(cal.reliability_bins(samples)) == pytest.approx(0.0, abs=1e-9)


def test_overconfident_samples_have_ece_equal_to_gap():
    samples = [(0.95, 1)] * 5 + [(0.95, 0)] * 5  # says 95%, is right 50%
    assert cal.expected_calibration_error(cal.reliability_bins(samples)) == pytest.approx(0.45)


def test_reliability_bins_edges_and_counts():
    bins = cal.reliability_bins([(0.0, 0), (0.55, 1), (1.0, 1)], n_bins=10)
    assert sum(b["n"] for b in bins) == 3 and bins[-1]["n"] == 1 and bins[5]["n"] == 1


def test_brier_binary_and_multiclass():
    r = row(probs={"infection_concern": 0.8, "urgency": {"ROUTINE": 0.1, "PRIORITY": 0.1, "URGENT": 0.7, "EMERGENCY": 0.1}})
    assert cal.brier_score([r], "infection_concern") == pytest.approx(0.04)
    assert cal.brier_score([r], "urgency") == pytest.approx(0.01 + 0.01 + 0.09 + 0.01)
    assert cal.brier_score([row()], "urgency") is None


def test_top_class_samples_binary_uses_max_of_p_and_complement():
    r = row(pred={"infection_concern": False}, ref={"infection_concern": False}, probs={"infection_concern": 0.2})
    assert cal.top_class_samples([r], "infection_concern") == [(pytest.approx(0.8), 1)]


def test_verbalized_confidence_used_for_claude_style_rows():
    r = row(conf={"urgency": 0.7}, confidence_kind="verbalized")
    assert cal.top_class_samples([r], "urgency") == [(0.7, 1)]
    assert cal.top_class_samples([row(conf={"urgency": 0.7})], "urgency") == []  # Jev rows need a distribution


def test_confidence_relationship_detects_monotonic_signal():
    samples = [(0.2, 0)] * 12 + [(0.2, 1)] * 3 + [(0.6, 1)] * 10 + [(0.6, 0)] * 10 + [(0.95, 1)] * 18 + [(0.95, 0)] * 2
    rel = cal.confidence_accuracy_relationship(samples)
    assert rel["monotonic"] is True and rel["spearman_rho"] > 0.3
    assert rel["answer"].startswith("Yes")


def test_confidence_relationship_refuses_to_conclude_on_small_n():
    rel = cal.confidence_accuracy_relationship([(0.9, 1), (0.1, 0), (0.5, 1)])
    assert "Not enough data" in rel["answer"]


def test_confidence_relationship_says_no_when_flat():
    samples = [(c, y) for c in (0.2, 0.6, 0.95) for y in [1] * 10 + [0] * 10]
    assert cal.confidence_accuracy_relationship(samples)["answer"].startswith("No clear")


def test_band_assignment_matches_policy_bands():
    assert [cal.band_of(x) for x in (0.1, 0.5, 0.79, 0.8, 1.0)] == ["low", "medium", "medium", "high", "high"]


def test_risk_coverage_is_monotone_in_coverage():
    rows = [row(f"c{i}", conf={"urgency": c, "specialty": c}) for i, c in enumerate([0.1, 0.3, 0.5, 0.7, 0.9] * 4)]
    rc = cal.risk_coverage(rows, thresholds=[0.0, 0.4, 0.8])
    assert [x["coverage"] for x in rc] == [1.0, 0.6, 0.2]
    assert rc[0]["escalated"] == 0 and rc[2]["n_automated"] == 4


def test_histograms_cover_all_samples():
    rows = [row(f"c{i}", conf={"urgency": 0.1 * i}) for i in range(10)]
    h = cal.confidence_histograms(rows, "urgency")
    assert sum(h["all"]["counts"]) == 10 and sum(h["correct"]["counts"]) + sum(h["incorrect"]["counts"]) == 10


def test_calibration_report_shape_on_empty_input():
    rep = cal.calibration_report([])
    assert set(rep["per_question"]) == set(m.QUESTIONS)
    assert rep["pooled"]["ece"] is None and rep["per_question"]["urgency"]["brier"] is None
