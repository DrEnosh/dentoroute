import json
import random

import pytest

from backend.schemas.case import CaseState
from backend.schemas.labels import CATEGORIES, SPECIALTIES, URGENCY_LEVELS
from backend.services import ground_truth_service as gt
from backend.services import synthetic_case_service as svc


@pytest.fixture(scope="module")
def dataset():
    return svc.generate_dataset(per_category=25, seed=42)


def test_dataset_size_and_balance(dataset):
    assert len(dataset) >= 500
    counts = svc.category_counts(dataset)
    assert set(counts) == set(CATEGORIES)
    assert set(counts.values()) == {25}


def test_unique_synthetic_ids(dataset):
    ids = [c.patient_id for c in dataset]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("SYN-") for i in ids)


def test_generation_is_reproducible():
    a = svc.generate_dataset(per_category=3, seed=7)
    b = svc.generate_dataset(per_category=3, seed=7)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]
    c = svc.generate_dataset(per_category=3, seed=8)
    assert [x.model_dump() for x in a] != [x.model_dump() for x in c]


def test_every_label_dimension_is_covered(dataset):
    assert {c.reference.urgency for c in dataset} == set(URGENCY_LEVELS)
    assert {c.reference.specialty for c in dataset} == set(SPECIALTIES)
    assert {c.reference.human_review for c in dataset} == {True, False}
    assert {c.reference.infection_concern for c in dataset} == {True, False}


def test_every_case_has_all_required_labels(dataset):
    for c in dataset:
        r = c.reference
        assert r is not None
        assert r.urgency in URGENCY_LEVELS and r.specialty in SPECIALTIES
        assert r.complexity and r.referral_priority


def test_state_sent_to_jev_has_no_labels_or_category(dataset):
    state = dataset[0].state.to_jev_state()
    for forbidden in ("category", "reference", "urgency", "specialty", "patient_id"):
        assert forbidden not in state
    assert all(v is not None for v in state.values())


def test_saved_files_are_consistent(tmp_path):
    cases = svc.generate_dataset(per_category=2, seed=1)
    cp, tp = tmp_path / "cases.json", tmp_path / "truth.json"
    svc.save_dataset(cases, cp, tp)
    rows = json.loads(cp.read_text())
    truth = json.loads(tp.read_text())
    assert len(rows) == len(cases) == len(truth)
    for row in rows:
        assert truth[row["patient_id"]]["urgency"] == row["reference"]["urgency"]


def test_committed_dataset_matches_regeneration():
    saved = svc.load_dataset()
    fresh = svc.generate_dataset(svc.DEFAULT_PER_CATEGORY, svc.DEFAULT_SEED)
    assert [c.model_dump() for c in saved] == [c.model_dump() for c in fresh]


def test_demo_cases_include_ambiguous_human_review_case():
    demo = svc.load_demo_cases()
    assert len(demo) == 10
    amb = next(c for c in demo if c.category == "ambiguous_vague")
    assert amb.reference.human_review is True


def test_get_case_and_not_found():
    assert svc.get_case("SYN-0001").patient_id == "SYN-0001"
    assert svc.get_case("DEMO-04").state.swelling is True
    with pytest.raises(Exception) as exc:
        svc.get_case("SYN-9999")
    assert "SYN-9999" in str(exc.value)


def test_random_case_filters_category():
    c = svc.random_case(category="oral_lesion", seed=1)
    assert c.category == "oral_lesion"
    fresh = svc.random_case(seed=3, from_dataset=False)
    assert fresh.patient_id.startswith("SYN-GEN-")


# ------------------------------------------------------------ rule engine ----
def test_rule_example_from_spec(severe_case):
    labels = gt.label_case(severe_case)
    assert labels.urgency in ("URGENT", "EMERGENCY")
    assert labels.infection_concern is True
    assert labels.human_review is True
    assert labels.same_day_review is True
    assert "U_U1_swelling_fever" in labels.fired_rules["urgency"]


def test_routine_case_labels(routine_case):
    labels = gt.label_case(routine_case)
    assert (labels.urgency, labels.specialty, labels.complexity, labels.referral_priority) == (
        "ROUTINE", "general_dentistry", "SIMPLE", "LOW",
    )
    assert not (labels.infection_concern or labels.same_day_review or labels.human_review)


def test_specialty_first_match_wins():
    # swelling+fever+trismus matches emergency (S02) before oral surgery (S09)
    s = CaseState(swelling=True, fever=True, trismus=True, progressive_symptoms=True)
    assert gt.label_case(s).specialty == "emergency_dental_service"


def test_condition_operators():
    ctx = {"a": 3, "b": ["x", "y"], "c": None, "d": "k"}
    ev = gt.evaluate_condition
    assert ev({"field": "a", "op": "gte", "value": 3}, ctx)
    assert not ev({"field": "a", "op": "gt", "value": 3}, ctx)
    assert ev({"field": "b", "op": "contains", "value": "x"}, ctx)
    assert ev({"field": "b", "op": "contains_any", "value": ["q", "y"]}, ctx)
    assert ev({"field": "d", "op": "in", "value": ["k", "z"]}, ctx)
    assert not ev({"field": "c", "op": "gte", "value": 1}, ctx)  # None never satisfies ordering
    assert ev({"not": {"field": "a", "op": "eq", "value": 4}}, ctx)
    assert ev({"any": [{"field": "a", "op": "eq", "value": 0}, {"field": "d", "op": "eq", "value": "k"}]}, ctx)
    with pytest.raises(ValueError):
        ev({"field": "a", "op": "nope", "value": 1}, ctx)


def test_rules_file_is_labeled_educational():
    rules = gt.load_rules()
    assert "Educational synthetic benchmark rules" in rules["meta"]["disclaimer"]
    text = gt.RULES_PATH.read_text()
    assert "not clinical guidelines" in text.lower()


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        svc.generate_case("nope", random.Random(1))
