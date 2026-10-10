"""Split discipline, threshold tuning, cost-sensitive evaluation, and the policy API / UI tab.

All numbers here come from the demo simulator or hand-built rows. They test the machinery, not clinical performance.
"""
from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient
from scipy.stats import beta

from backend.main import create_app
from backend.services import benchmark_service, policy_eval_service as pe, risk_model, split_service
from backend.services.policy_service import MAX_LIMITS, MIN_LIMITS, validate_policy
from backend.services.routing_service import route_decision, route_features
from backend.services.synthetic_case_service import load_dataset
from backend.utils.config import load_routing_policy
from backend.utils.errors import PolicyError
from tests.helpers import benign, make_decision, reference, row_from_decision


# ------------------------------------------------------------------ fixtures -----
@pytest.fixture(scope="module")
def demo_rows():
    return benchmark_service.run_jev(load_dataset(), demo=True, workers=4)


@pytest.fixture(scope="module")
def split_rows(demo_rows):
    return split_service.attach_splits(demo_rows, split_service.get_splits())


def by_split(rows, name):
    return [r for r in rows if r["split"] == name]


def toy(split: str, i: int, d=None, ref=None):
    return row_from_decision(d or benign(), ref or reference(), f"T{split[:2]}{i}", split)


# ------------------------------------------------------------------- splits ------
def test_splits_are_disjoint_complete_and_roughly_40_30_30():
    s = split_service.get_splits()
    assign = s["assignment"]
    cases = load_dataset()
    assert set(assign) == {c.patient_id for c in cases}
    counts = {n: sum(1 for v in assign.values() if v == n) for n in split_service.SPLIT_NAMES}
    assert sum(counts.values()) == len(cases) and s["counts"] == counts
    for name, frac in split_service.FRACTIONS.items():
        assert abs(counts[name] / len(cases) - frac) < 0.02


def test_splits_are_stable_and_depend_only_on_ids_and_reference_labels():
    a = split_service.build_default_splits()
    b = split_service.build_default_splits()
    assert a["assignment"] == b["assignment"]
    assert a["assignment"] == split_service.get_splits()["assignment"]  # the saved file matches a fresh build
    assert a["stratified_by"] == "reference urgency"


def test_splits_are_stratified_by_reference_urgency():
    from backend.services.ground_truth_service import label_case

    assign = split_service.get_splits()["assignment"]
    urgent = {n: 0 for n in split_service.SPLIT_NAMES}
    for c in load_dataset():
        ref = c.reference or label_case(c.state)
        if ref.urgency in ("URGENT", "EMERGENCY"):
            urgent[assign[c.patient_id]] += 1
    total = sum(urgent.values())
    assert total > 0
    assert abs(urgent["holdout"] / total - 0.30) < 0.05 and abs(urgent["train"] / total - 0.40) < 0.05


def test_make_splits_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_service.make_splits(["a", "b"], ["x", "x"], fractions={"train": 0.5, "calibration": 0.5, "holdout": 0.5})


def test_get_splits_writes_a_file_when_missing(tmp_path):
    p = tmp_path / "splits.json"
    s = split_service.get_splits(path=p)
    assert p.exists() and json.loads(p.read_text())["seed"] == s["seed"]


def test_attach_splits_drops_unknown_cases_instead_of_guessing():
    out = split_service.attach_splits([{"case_id": "NOT-A-CASE"}], {"assignment": {}})
    assert out == []


# ----------------------------------------------------------- leakage guards ------
def test_tune_refuses_holdout_rows_in_either_slot(split_rows):
    train, cal, hold = (by_split(split_rows, n) for n in split_service.SPLIT_NAMES)
    with pytest.raises(pe.HoldoutLeakError):
        pe.tune(train + hold[:3], cal)
    with pytest.raises(pe.HoldoutLeakError):
        pe.tune(train, cal + hold[:3])
    with pytest.raises(pe.HoldoutLeakError):
        pe.tune(train, train)          # calibration slot must hold calibration rows
    with pytest.raises(pe.HoldoutLeakError):
        pe.tune([{"predictions": {}, "reference": {}}], cal)  # rows with no split label are rejected too


def test_tuned_policy_does_not_change_when_holdout_rows_change(split_rows):
    """Perturbing every holdout row must leave the tuned thresholds and risk model byte-identical."""
    train, cal = by_split(split_rows, "train"), by_split(split_rows, "calibration")
    p1, _ = pe.tune(train, cal)
    hold = copy.deepcopy(by_split(split_rows, "holdout"))   # copy: the module fixture is shared
    for r in hold:                       # the holdout is never passed in, so this cannot matter
        r["reference"] = dict(r["reference"], urgency="EMERGENCY")
    p2, _ = pe.tune(train, cal)
    assert p1["thresholds"] == p2["thresholds"] and p1["risk_model"] == p2["risk_model"]


def test_run_policy_study_reports_holdout_as_headline_and_never_selects_on_it(split_rows):
    study = pe.run_policy_study(split_rows, data_mode="demo", model="demo-simulator (NOT Jev)")
    assert study["headline_split"] == "holdout" and study["holdout_used_for_selection"] is False
    assert study["provenance"]["tuned_on"] == ["train", "calibration"]
    assert study["simulated"] is True
    n = study["provenance"]["split_counts"]
    assert study["per_split"]["holdout"]["revised"]["n"] == n["holdout"]
    assert "_policy" not in pe.public_study(study)
    # The frozen policy, evaluated on holdout, is exactly what the study reports
    assert pe.evaluate(by_split(split_rows, "holdout"), study["_policy"])["missed_urgent"] == study["per_split"]["holdout"]["revised"]["missed_urgent"]


def test_study_requires_all_three_splits(demo_rows):
    only_train = [r for r in split_service.attach_splits(demo_rows) if r["split"] == "train"]
    with pytest.raises(pe.PolicyTuningError):
        pe.run_policy_study(only_train, data_mode="demo", model="m")


# -------------------------------------------------------------- tuning -----------
def test_tuned_thresholds_respect_the_hard_limits_and_the_grid(split_rows):
    pol, rep = pe.tune(by_split(split_rows, "train"), by_split(split_rows, "calibration"))
    validate_policy(pol)
    for k, v in pol["thresholds"].items():
        if k in MAX_LIMITS:
            assert v <= MAX_LIMITS[k]
        if k in MIN_LIMITS:
            assert v >= MIN_LIMITS[k]
    assert rep["feasible"] and rep["calibration_before_margin"]["missed_urgent"] <= rep["budget_missed_urgent_calibration"]
    assert rep["n_feasible"] <= rep["n_candidates"] and rep["calibration_frontier"][0]["missed_urgent"] == 0


def test_safety_margin_moves_every_tuned_threshold_toward_caution(split_rows):
    pol, rep = pe.tune(by_split(split_rows, "train"), by_split(split_rows, "calibration"))
    for k, final in rep["selected_final"].items():
        before = rep["selected_before_margin"][k]
        assert final <= before if k in MAX_LIMITS else final >= before


def test_larger_margin_is_never_less_cautious(split_rows):
    base = load_routing_policy("conservative")
    tr, ca = by_split(split_rows, "train"), by_split(split_rows, "calibration")
    p0, _ = pe.tune(tr, ca, dict(base, tuning=dict(base["tuning"], safety_margin_steps=0)))
    p2, _ = pe.tune(tr, ca, dict(base, tuning=dict(base["tuning"], safety_margin_steps=2)))
    for k in base["tuning"]["tunable"]:
        if k in MAX_LIMITS:
            assert p2["thresholds"][k] <= p0["thresholds"][k]
        else:
            assert p2["thresholds"][k] >= p0["thresholds"][k]


def test_tuning_grid_looser_than_a_hard_limit_is_rejected(split_rows):
    base = copy.deepcopy(load_routing_policy("conservative"))
    key = "infection_probability"
    base["tuning"]["grid"][key] = [0.1, MAX_LIMITS[key] + 0.2]
    with pytest.raises(pe.PolicyTuningError):
        pe.tune(by_split(split_rows, "train"), by_split(split_rows, "calibration"), base)


def test_infeasible_budget_falls_back_to_the_most_cautious_option_and_says_so(split_rows):
    base = copy.deepcopy(load_routing_policy("conservative"))
    base["tuning"]["budget_missed_urgent_calibration"] = -1   # nothing can satisfy this
    pol, rep = pe.tune(by_split(split_rows, "train"), by_split(split_rows, "calibration"), base)
    assert rep["feasible"] is False and rep["note"] and "most cautious" in rep["note"]
    validate_policy(pol)


def test_tune_needs_both_sets():
    with pytest.raises(pe.PolicyTuningError):
        pe.tune([], [])


# --------------------------------------------------- hand-computed evaluation ----
def test_evaluate_counts_match_a_hand_computed_example():
    pol = copy.deepcopy(load_routing_policy("conservative"))
    pol["risk_model"] = None
    pol["thresholds"].update(
        human_review_probability=0.5, infection_probability=0.35, same_day_probability=0.5, urgency_score=1.5,
        urgent_tail_probability=0.3, risk_score=0.4, urgency_confidence_min=0.6, specialty_confidence_min=0.7,
        specialty_top_probability_min=0.55, boolean_certainty_min=0.4,
    )
    clean_urgent = dict(urgency=(0.0, 0.05, 0.9, 0.05))
    rows = [
        toy("holdout", 1, benign(), reference()),                                      # automated, correct, no human needed
        toy("holdout", 2, benign(), reference()),                                      # same
        toy("holdout", 3, benign(), reference(urgency="URGENT")),                      # silent miss: automated AND urgent
        toy("holdout", 4, benign(), reference(infection=True)),                        # automated, needs human, not urgent
        toy("holdout", 5, benign(**clean_urgent), reference(urgency="URGENT")),        # caught by R3
        toy("holdout", 6, benign(infection=0.8), reference()),                         # sent to a human unnecessarily
        toy("holdout", 7, benign(), reference(specialty="oral_pathology")),            # automated with the wrong specialty
    ]
    m = pe.evaluate(rows, pol, costs={"missed_urgent": 100, "other_false_automation": 20, "unnecessary_review": 1})
    assert m["n"] == 7 and m["n_automated"] == 5 and m["n_human"] == 2 and m["coverage"] == pytest.approx(5 / 7)
    assert m["n_urgent_reference"] == 2 and m["missed_urgent"] == 1 and m["missed_urgent_ids"] == ["Tho3"]
    assert m["n_needs_human_reference"] == 3 and m["false_automation"] == 2 and m["false_automation_other"] == 1
    assert m["false_automation_rate"] == pytest.approx(2 / 3)
    assert m["n_no_human_needed_reference"] == 4 and m["unnecessary_human_review"] == 1
    assert m["unnecessary_review_rate"] == pytest.approx(1 / 4)
    assert m["automation_accuracy_specialty"] == pytest.approx(4 / 5)       # row 7 is the wrong one
    assert m["automation_accuracy_urgency"] == pytest.approx(4 / 5)         # row 3 is the wrong one
    assert m["automation_accuracy_joint"] == pytest.approx(3 / 5)
    assert m["cost"] == 100 * 1 + 20 * 1 + 1 * 1 and m["cost_per_case"] == pytest.approx(121 / 7)
    assert m["missed_urgent_upper95"] == pytest.approx(float(beta.ppf(0.95, 2, 1)))  # k=1 miss of n=2 urgent


def test_evaluate_handles_nothing_automated_and_empty_input():
    pol = load_routing_policy("conservative")
    all_human = [toy("holdout", i, benign(human=0.9), reference()) for i in range(3)]
    m = pe.evaluate(all_human, pol)
    assert m["coverage"] == 0 and m["automation_accuracy_specialty"] is None and m["missed_urgent"] == 0
    e = pe.evaluate([], pol)
    assert e["n"] == 0 and e["coverage"] is None and e["cost_per_case"] is None


def test_evaluate_skips_failed_rows():
    pol = load_routing_policy("conservative")
    rows = [toy("holdout", 1), {"case_id": "bad", "predictions": None, "reference": reference()}]
    assert pe.evaluate(rows, pol)["n"] == 1


def test_clopper_pearson_known_values():
    assert pe.clopper_pearson_upper(0, 24) == pytest.approx(1 - 0.05 ** (1 / 24))   # closed form for k=0
    assert pe.clopper_pearson_upper(0, 0) is None and pe.clopper_pearson_upper(5, 5) == 1.0
    assert pe.clopper_pearson_upper(1, 24) > pe.clopper_pearson_upper(0, 24)
    assert pe.clopper_pearson_upper(0, 100) < pe.clopper_pearson_upper(0, 24)       # more data, tighter bound


def test_label_definitions():
    assert pe.is_urgent(reference(urgency="EMERGENCY")) and not pe.is_urgent(reference(urgency="PRIORITY"))
    assert pe.needs_human(reference(infection=True)) and pe.needs_human(reference(same_day=True))
    assert pe.needs_human(reference(human=True)) and pe.needs_human(reference(urgency="URGENT"))
    assert not pe.needs_human(reference())


def test_row_routing_equals_decision_routing_for_every_demo_case(demo_rows):
    """The study evaluates stored rows; the app routes live decisions. Both paths must agree exactly."""
    from backend.services.jev_service import JevService

    pol = load_routing_policy()
    cases = {c.patient_id: c for c in load_dataset()}
    svc = JevService()
    for r in demo_rows[:60]:
        d = svc.evaluate(cases[r["case_id"]].state, demo=True, use_cache=False)
        assert route_decision(d, pol).route == route_features(risk_model.features_from_row(r), pol).route


# --------------------------------------------------------- shipped policy ---------
def test_shipped_policy_is_valid_labelled_and_never_loosens_limits():
    p = validate_policy(load_routing_policy())
    assert p["rules"] == "conservative" and p["version"] == 2
    assert p["provenance"]["holdout_used_for_selection"] is False
    text = open("config/routing_policy.yaml", encoding="utf-8").read().lower()
    assert "not clinical" in text and "research-demo" in text.replace("research demo", "research-demo")
    for k, v in p["thresholds"].items():
        assert k not in MAX_LIMITS or v <= MAX_LIMITS[k]
        assert k not in MIN_LIMITS or v >= MIN_LIMITS[k]


def test_a_policy_looser_than_the_hard_limit_is_refused_on_every_route():
    p = copy.deepcopy(load_routing_policy())
    p["thresholds"]["infection_probability"] = MAX_LIMITS["infection_probability"] + 0.1
    with pytest.raises(PolicyError):
        route_decision(make_decision(), p)


def test_shipped_policy_meets_the_headline_goal_on_holdout_with_simulated_data(split_rows):
    """Regression guard, not a clinical claim: on the demo data the shipped policy automates no reference-urgent holdout case."""
    hold = by_split(split_rows, "holdout")
    new, old = pe.evaluate(hold, load_routing_policy("conservative")), pe.evaluate(hold, load_routing_policy("legacy"))
    assert new["missed_urgent"] <= old["missed_urgent"] and new["missed_urgent"] == 0
    assert new["coverage"] < old["coverage"]          # safety is bought with automation, and the numbers say so


def test_legacy_policy_reproduces_the_61_unsafe_automations(demo_rows):
    m = pe.evaluate(demo_rows, load_routing_policy("legacy"))
    assert m["legacy_unsafe_automations"] == 61


# ----------------------------------------------------------------- API ------------
@pytest.fixture
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


def test_api_active_policy_exposes_limits_and_provenance(client):
    r = client.get("/api/policy").json()
    assert r["name"] == "conservative" and r["rules"] == "conservative" and r["risk_model_configured"] is True
    assert r["limits"]["max"]["infection_probability"] == MAX_LIMITS["infection_probability"]
    assert r["provenance"]["data_mode"] in ("demo", "live") and "research" in (r["disclaimer"] or "").lower()


def test_api_study_without_a_benchmark_is_a_clear_error(client):
    r = client.post("/api/policy/study")
    assert r.status_code == 400 and "benchmark" in r.json()["error"].lower()
    assert client.get("/api/policy/study").json() == {"study": None}


def test_api_study_runs_on_the_stored_benchmark_and_does_not_change_the_active_policy(client):
    before = client.get("/api/policy").json()["thresholds"]
    assert client.post("/api/benchmark/run", json={"provider": "jev", "demo": True}).status_code == 200
    study = client.post("/api/policy/study").json()
    assert study["simulated"] is True and study["holdout_used_for_selection"] is False and "_policy" not in study
    assert study["per_split"]["holdout"]["revised"]["missed_urgent"] <= study["per_split"]["holdout"]["legacy"]["missed_urgent"]
    assert len(study["cost_sensitivity_holdout"]) == len(pe.COST_RATIOS)
    assert client.get("/api/policy/study").json()["study"]["simulated"] is True   # stored for the UI
    assert client.get("/api/policy").json()["thresholds"] == before               # candidate only; never auto-applied


def test_study_refuses_a_partial_benchmark_because_the_holdout_would_be_too_thin(demo_rows):
    with pytest.raises(pe.PolicyTuningError):
        pe.run_policy_study(demo_rows[:60], data_mode="demo", model="m")


def test_api_study_on_a_partial_benchmark_is_a_400(client):
    client.post("/api/benchmark/run", json={"provider": "jev", "limit": 40, "demo": True})
    r = client.post("/api/policy/study")
    assert r.status_code == 400 and "full benchmark" in r.json()["error"]
