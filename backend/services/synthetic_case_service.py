"""Synthetic dental case generator. Entirely synthetic - no real patient data, no PHI.

Cases are built from scenario templates with seeded randomness, then labeled by the
deterministic rule engine (ground_truth_service). No LLM is involved at any point.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Callable

from backend.schemas.case import CaseState, ReferenceLabels, SyntheticCase
from backend.schemas.labels import CATEGORIES
from backend.services.ground_truth_service import label_case
from backend.utils.config import DATA_DIR
from backend.utils.errors import CaseNotFoundError

CASES_PATH = DATA_DIR / "synthetic_cases.json"
TRUTH_PATH = DATA_DIR / "ground_truth.json"
DEMO_CASES_PATH = DATA_DIR / "demo_cases.json"
DEFAULT_PER_CATEGORY = 25
DEFAULT_SEED = 42

REGIONS = [
    "maxillary molar", "mandibular molar", "maxillary premolar",
    "mandibular premolar", "maxillary incisor", "mandibular incisor",
]
SIDES = ["right", "left"]
HISTORY_POOL = [
    "hypertension", "type 2 diabetes", "asthma", "hypothyroidism", "tobacco use",
    "anticoagulant therapy", "gastro-oesophageal reflux", "osteoporosis",
]
RISK_HISTORY = ["immunocompromised", "uncontrolled diabetes"]
MED_POOL = [
    "metformin", "amlodipine", "levothyroxine", "atorvastatin", "omeprazole",
    "salbutamol inhaler", "losartan", "vitamin D",
]
ALLERGY_POOL = ["penicillin", "latex", "sulfa drugs", "ibuprofen"]

Override = dict
Generator = Callable[[random.Random], Override]


# ---------------------------------------------------------------- helpers ----
def _chance(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def _region(rng: random.Random) -> str:
    return rng.choice(REGIONS)


def _side_region_phrase(rng: random.Random, region: str) -> str:
    return f"{rng.choice(SIDES)} {region}"


def _temperature(rng: random.Random, fever: bool) -> float | None:
    if fever:
        return round(rng.uniform(38.0, 40.0), 1)
    return round(rng.uniform(36.4, 37.4), 1) if _chance(rng, 0.6) else None


def _history(rng: random.Random, age: int) -> tuple[list[str], list[str], list[str]]:
    p_hist = 0.1 if age < 30 else 0.35 if age < 50 else 0.65
    history: list[str] = []
    meds: list[str] = []
    if age >= 13 and _chance(rng, p_hist):
        history = rng.sample(HISTORY_POOL, k=rng.choice([1, 1, 2, 3]))
        meds = rng.sample(MED_POOL, k=min(len(MED_POOL), rng.choice([1, 2, 3, 4])))
        if "anticoagulant therapy" in history:
            meds.append("warfarin")
    allergies = rng.sample(ALLERGY_POOL, k=1) if _chance(rng, 0.18) else []
    return history, meds, allergies


# ------------------------------------------------------ category templates ---
def _routine(rng):
    return dict(
        age=rng.randint(18, 82),
        chief_complaint=rng.choice([
            "routine dental check-up and cleaning", "six-monthly examination, no complaints",
            "annual check-up, no pain", "scale and polish, no symptoms",
        ]),
        pain_severity=0,
    )


def _caries(rng):
    region = _region(rng)
    return dict(
        age=rng.randint(15, 70), tooth_region=region, pain_severity=rng.randint(1, 4),
        pain_character="sharp_brief", pain_trigger=rng.sample(["sweets", "cold", "hot"], k=rng.choice([1, 2])),
        duration=rng.choice(["2 weeks", "1 month", "several weeks"]),
        chief_complaint=f"sensitivity to sweet foods in a {region} tooth, visible dark spot",
    )


def _reversible_pulpitis(rng):
    region = _region(rng)
    return dict(
        age=rng.randint(15, 65), tooth_region=region, pain_severity=rng.randint(3, 5),
        pain_character="sharp_brief", pain_trigger=rng.choice([["cold"], ["cold", "hot"]]),
        duration=rng.choice(["3 days", "1 week", "5 days"]),
        chief_complaint=f"short sharp pain with cold drinks in {region} area that stops quickly",
    )


def _severe_pulpal(rng):
    region = _region(rng)
    return dict(
        age=rng.randint(18, 60), tooth_region=region, pain_severity=rng.randint(7, 10),
        pain_character=rng.choice(["throbbing", "spontaneous", "lingering"]),
        pain_trigger=rng.sample(["hot", "cold", "lying down"], k=rng.choice([1, 2])),
        duration=rng.choice(["2 days", "3 days", "4 days"]),
        chief_complaint=f"severe {rng.choice(['throbbing', 'constant', 'night-time'])} pain in a {region} tooth, woke me at night",
    )


def _periapical(rng):
    region = _region(rng)
    swelling = _chance(rng, 0.25)
    fever = _chance(rng, 0.1)
    return dict(
        age=rng.randint(20, 70), tooth_region=region, pain_severity=rng.randint(4, 8),
        pain_character=rng.choice(["dull", "throbbing", "dull"]),
        pain_trigger=rng.choice([["biting"], ["biting", "pressure"], ["biting", "hot"]]),
        previous_root_canal=_chance(rng, 0.4),
        swelling=swelling, swelling_location="buccal vestibule near the affected tooth" if swelling else None,
        pus_or_drainage=_chance(rng, 0.2), fever=fever, temperature_c=_temperature(rng, fever),
        duration=rng.choice(["5 days", "1 week", "10 days"]),
        chief_complaint=f"tooth feels high and painful to bite on, {region} region",
    )


def _post_op(rng):
    region = _region(rng)
    fever = _chance(rng, 0.1)
    return dict(
        age=rng.randint(18, 75), tooth_region=region, recent_extraction_or_procedure=True,
        pain_severity=rng.randint(2, 8), pain_character=rng.choice(["dull", "throbbing"]),
        bleeding=_chance(rng, 0.3), swelling=_chance(rng, 0.15), swelling_location="extraction site area",
        fever=fever, temperature_c=_temperature(rng, fever),
        duration=rng.choice(["2 days", "3 days", "4 days after procedure"]),
        chief_complaint=f"pain returning after an extraction in the {region} area",
    )


def _periodontal(rng):
    return dict(
        age=rng.randint(35, 82), gingival_involvement=True, bleeding=_chance(rng, 0.7),
        mobility=_chance(rng, 0.35), pain_severity=rng.randint(0, 4), pain_character=rng.choice(["dull", None]),
        duration=rng.choice(["several months", "over a year"]),
        chief_complaint=rng.choice([
            "receding gums and teeth feeling loose", "bad taste and gums that bleed when brushing",
            "gums are swollen and sore around several teeth",
        ]),
    )


def _gingival_bleeding(rng):
    return dict(
        age=rng.randint(18, 62), gingival_involvement=True, bleeding=True,
        pain_severity=rng.randint(0, 2), duration=rng.choice(["2 weeks", "1 month", "several weeks"]),
        chief_complaint="gums bleed when brushing and flossing",
    )


def _perio_abscess(rng):
    region = _region(rng)
    fever = _chance(rng, 0.3)
    return dict(
        age=rng.randint(30, 75), tooth_region=region, gingival_involvement=True,
        swelling=True, swelling_location=f"gingival swelling near {region} teeth",
        pain_severity=rng.randint(5, 8), pain_character=rng.choice(["throbbing", "dull"]),
        pus_or_drainage=_chance(rng, 0.5), mobility=_chance(rng, 0.5),
        fever=fever, temperature_c=_temperature(rng, fever),
        duration=rng.choice(["3 days", "4 days", "1 week"]),
        chief_complaint="painful swollen lump on the gum with bad taste",
    )


def _third_molar(rng):
    swelling = _chance(rng, 0.3)
    fever = _chance(rng, 0.1)
    return dict(
        age=rng.randint(17, 32), tooth_region="mandibular third molar", third_molar_symptoms=True,
        pain_severity=rng.randint(3, 8), pain_character=rng.choice(["dull", "throbbing"]),
        pain_trigger=rng.choice([["chewing"], ["biting"], []]),
        swelling=swelling, swelling_location="angle of the jaw near the wisdom tooth" if swelling else None,
        trismus=_chance(rng, 0.2), fever=fever, temperature_c=_temperature(rng, fever),
        duration=rng.choice(["3 days", "1 week", "recurring over months"]),
        chief_complaint="pain and tenderness behind the last lower tooth, partially erupted wisdom tooth",
    )


def _trauma(rng):
    kind = rng.choices(["avulsion", "fracture", "luxation"], weights=[0.2, 0.4, 0.4])[0]
    base = dict(
        age=rng.randint(6, 45), trauma=True, tooth_region=rng.choice(["maxillary incisor", "mandibular incisor"]),
        pain_severity=rng.randint(2, 9), bleeding=_chance(rng, 0.4),
        duration=rng.choice(["a few hours ago", "yesterday", "this morning"]),
    )
    if kind == "avulsion":
        base.update(tooth_avulsion=True, chief_complaint="front tooth knocked out completely in a fall")
    elif kind == "fracture":
        base.update(tooth_fracture=True, chief_complaint="front tooth chipped and broken during sports injury")
    else:
        base.update(mobility=True, chief_complaint="front tooth loose and pushed out of position after a fall")
    return base


def _fractured(rng):
    region = _region(rng)
    return dict(
        age=rng.randint(20, 75), tooth_fracture=True, tooth_region=region,
        pain_severity=rng.randint(1, 7), pain_trigger=rng.choice([["biting"], ["biting", "cold"], ["cold"]]),
        trauma=_chance(rng, 0.2), duration=rng.choice(["2 days", "1 week"]),
        chief_complaint=f"piece of a {region} tooth broke off while chewing, sharp edge",
    )


def _prosth(rng):
    return dict(
        age=rng.randint(40, 86), restoration_or_prosthesis_issue=True, pain_severity=rng.randint(0, 4),
        duration=rng.choice(["1 week", "several weeks"]),
        chief_complaint=rng.choice([
            "crown has come loose", "denture rubs and causes a sore spot", "bridge feels loose when eating",
            "filling fell out leaving a rough edge",
        ]),
    )


def _ortho(rng):
    return dict(
        age=rng.randint(13, 40), orthodontic_appliance_issue=True, pain_severity=rng.randint(0, 4),
        duration=rng.choice(["2 days", "1 week"]),
        chief_complaint=rng.choice([
            "wire poking into the cheek", "bracket came off", "braces feel tight and sore after adjustment",
            "retainer cracked",
        ]),
    )


def _pediatric(rng):
    fever = _chance(rng, 0.15)
    swelling = _chance(rng, 0.08)
    return dict(
        age=rng.randint(3, 12), sex=rng.choice(["male", "female"]), pain_severity=rng.randint(2, 7),
        pain_character=rng.choice(["dull", "sharp_brief", "throbbing"]), pain_trigger=rng.choice([["chewing"], ["cold"], ["sweets"]]),
        tooth_region=rng.choice(["maxillary molar", "mandibular molar"]),
        fever=fever, temperature_c=_temperature(rng, fever),
        swelling=swelling, swelling_location="cheek next to the sore tooth" if swelling else None,
        info_completeness=rng.choice(["complete", "complete", "partial"]),
        duration=rng.choice(["2 days", "4 days"]),
        chief_complaint="child complains of toothache and avoids chewing (parent-reported)",
    )


def _lesion(rng):
    site = rng.choice(["lateral tongue", "buccal mucosa", "floor of mouth", "palate", "gingiva"])
    hist = ["tobacco use"] if _chance(rng, 0.4) else []
    return dict(
        age=rng.randint(25, 82), lesion_present=True, lesion_duration_weeks=rng.randint(1, 16),
        pain_severity=rng.randint(0, 4), medical_history=hist,
        chief_complaint=f"non-healing ulcer on the {site}",
        duration="several weeks",
    )


def _tmj(rng):
    return dict(
        age=rng.randint(18, 60), jaw_clicking=True, jaw_locking=_chance(rng, 0.35),
        pain_severity=rng.randint(2, 7), pain_character=rng.choice(["dull", "aching"]),
        pain_trigger=rng.choice([["chewing"], ["chewing", "yawning"], ["yawning"]]),
        duration=rng.choice(["3 weeks", "2 months"]),
        chief_complaint="jaw pain and clicking when opening wide, headaches in the morning",
    )


def _facial_swelling(rng):
    fever = _chance(rng, 0.45)
    side = rng.choice(SIDES)
    return dict(
        age=rng.randint(15, 75), swelling=True,
        swelling_location=rng.choice([f"{side} cheek", f"{side} submandibular region", f"{side} mandibular buccal region"]),
        pain_severity=rng.randint(3, 8), pain_character=rng.choice(["throbbing", "dull"]),
        fever=fever, temperature_c=_temperature(rng, fever), trismus=_chance(rng, 0.3),
        progressive_symptoms=_chance(rng, 0.4), lymphadenopathy=_chance(rng, 0.4),
        pus_or_drainage=_chance(rng, 0.25), difficulty_swallowing=_chance(rng, 0.05),
        tooth_region=_region(rng), duration=rng.choice(["2 days", "3 days", "5 days"]),
        chief_complaint="facial swelling with a painful tooth",
    )


def _severe_infection(rng):
    side = rng.choice(SIDES)
    hist = [rng.choice(RISK_HISTORY)] if _chance(rng, 0.3) else []
    return dict(
        age=rng.randint(18, 75), swelling=True,
        swelling_location=rng.choice([f"{side} submandibular region", f"{side} neck and jaw", f"{side} facial space"]),
        fever=True, temperature_c=round(rng.uniform(38.4, 40.2), 1),
        trismus=_chance(rng, 0.55), difficulty_swallowing=_chance(rng, 0.4),
        difficulty_breathing=_chance(rng, 0.12), progressive_symptoms=True,
        lymphadenopathy=_chance(rng, 0.7), pus_or_drainage=_chance(rng, 0.5),
        pain_severity=rng.randint(6, 10), pain_character="throbbing", tooth_region=_region(rng),
        medical_history=hist, duration=rng.choice(["3 days", "4 days", "1 week"]),
        chief_complaint="rapidly spreading facial swelling with high fever after toothache",
    )


def _admin(rng):
    return dict(
        age=rng.randint(20, 80), administrative_request=True, pain_severity=0,
        chief_complaint=rng.choice([
            "request to reschedule a booked appointment", "asking for a copy of dental records for transfer",
            "insurance pre-authorisation question", "billing query about a previous visit",
            "request for a referral letter to be sent to another clinic",
        ]),
    )


def _ambiguous(rng):
    mode = rng.choice(["sparse", "sparse", "partial"])
    base = dict(
        age=rng.randint(18, 75),
        chief_complaint=rng.choice([
            "mouth feels off lately", "some discomfort, not sure where", "something doesn't feel right with my teeth",
        ]),
        info_completeness=mode, pain_severity=rng.randint(1, 6),
    )
    if mode == "partial":
        flag = rng.choice(["swelling", "fever", "mobility", "bleeding"])
        base[flag] = True
        if flag == "fever":
            base["temperature_c"] = round(rng.uniform(37.8, 38.6), 1)
        if flag == "swelling":
            base["swelling_location"] = "unsure, 'somewhere on one side'"
    return base


GENERATORS: dict[str, Generator] = {
    "routine_examination": _routine, "caries_related": _caries, "reversible_pulpitis_like": _reversible_pulpitis,
    "severe_pulpal_pain": _severe_pulpal, "periapical_symptoms": _periapical, "post_operative_pain": _post_op,
    "periodontal_complaint": _periodontal, "gingival_bleeding": _gingival_bleeding,
    "periodontal_abscess_like": _perio_abscess, "impacted_third_molar": _third_molar, "dental_trauma": _trauma,
    "fractured_tooth": _fractured, "prosthodontic_complaint": _prosth, "orthodontic_complaint": _ortho,
    "pediatric_complaint": _pediatric, "oral_lesion": _lesion, "tmj_complaint": _tmj,
    "facial_swelling": _facial_swelling, "severe_infection_warning": _severe_infection,
    "administrative_nonurgent": _admin, "ambiguous_vague": _ambiguous,
}
assert set(GENERATORS) == set(CATEGORIES), "every category needs a generator"


# --------------------------------------------------------------- public API --
def generate_case(category: str, rng: random.Random, patient_id: str = "SYN-0000") -> SyntheticCase:
    if category not in GENERATORS:
        raise ValueError(f"Unknown category: {category}")
    over = GENERATORS[category](rng)
    age = over.get("age", 30)
    history, meds, allergies = _history(rng, age)
    data: dict = dict(
        patient_id=patient_id, sex=rng.choice(["male", "female"]),
        medical_history=history, medications=meds, allergies=allergies,
    )
    if category in ("routine_examination", "caries_related", "administrative_nonurgent", "ambiguous_vague"):
        pass
    data.update(over)
    # temperature coherence: fever without explicit reading gets a plausible one
    if data.get("fever") and data.get("temperature_c") is None:
        data["temperature_c"] = _temperature(rng, True)
    if "temperature_c" not in data:
        data["temperature_c"] = _temperature(rng, bool(data.get("fever")))
    if category == "ambiguous_vague":
        data["temperature_c"] = data.get("temperature_c") if data.get("fever") else None
        data["medical_history"], data["medications"], data["allergies"] = [], [], []
    state = CaseState(**data)
    return SyntheticCase(patient_id=patient_id, category=category, state=state, reference=label_case(state))


def generate_dataset(per_category: int = DEFAULT_PER_CATEGORY, seed: int = DEFAULT_SEED) -> list[SyntheticCase]:
    rng = random.Random(seed)
    drafts: list[tuple[str, random.Random]] = []
    for cat in CATEGORIES:
        for _ in range(per_category):
            drafts.append((cat, random.Random(rng.randrange(2**31))))
    rng.shuffle(drafts)
    cases = []
    for i, (cat, case_rng) in enumerate(drafts, start=1):
        cases.append(generate_case(cat, case_rng, patient_id=f"SYN-{i:04d}"))
    return cases


def save_dataset(cases: list[SyntheticCase], cases_path: Path = CASES_PATH, truth_path: Path = TRUTH_PATH) -> None:
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump(mode="json") for c in cases], f, indent=1)
    truth = {c.patient_id: c.reference.model_dump(mode="json") for c in cases if c.reference}
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=1)


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[SyntheticCase, ...]:
    with open(path, encoding="utf-8") as f:
        return tuple(SyntheticCase(**row) for row in json.load(f))


def load_dataset(path: Path | None = None) -> list[SyntheticCase]:
    p = path or CASES_PATH
    if not Path(p).exists():
        cases = generate_dataset()
        save_dataset(cases)
    return list(_load(str(p)))


def load_demo_cases() -> list[SyntheticCase]:
    if not DEMO_CASES_PATH.exists():
        save_demo_cases()
    return list(_load(str(DEMO_CASES_PATH)))


def get_case(patient_id: str) -> SyntheticCase:
    for case in load_demo_cases() + load_dataset():
        if case.patient_id == patient_id:
            return case
    raise CaseNotFoundError(f"No synthetic case with ID '{patient_id}'.")


def random_case(category: str | None = None, seed: int | None = None, from_dataset: bool = True) -> SyntheticCase:
    rng = random.Random(seed)
    if from_dataset:
        pool = [c for c in load_dataset() if category in (None, c.category)]
        if not pool:
            raise ValueError(f"No cases for category '{category}'")
        return rng.choice(pool)
    cat = category or rng.choice(CATEGORIES)
    return generate_case(cat, rng, patient_id=f"SYN-GEN-{rng.randrange(10**5):05d}")


def category_counts(cases: list[SyntheticCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.category] = counts.get(c.category, 0) + 1
    return counts


# ----------------------------------------------------------- LinkedIn demo ---
def _demo(pid: str, category: str, **fields) -> SyntheticCase:
    state = CaseState(patient_id=pid, **fields)
    return SyntheticCase(patient_id=pid, category=category, state=state, reference=label_case(state))


def build_demo_cases() -> list[SyntheticCase]:
    """~10 hand-written, particularly interesting synthetic cases for demos."""
    return [
        _demo("DEMO-01", "routine_examination", age=34, sex="female",
              chief_complaint="routine dental check-up and cleaning", pain_severity=0, temperature_c=36.7),
        _demo("DEMO-02", "reversible_pulpitis_like", age=27, sex="male", tooth_region="maxillary premolar",
              chief_complaint="short sharp pain with cold drinks that stops quickly", duration="1 week",
              pain_severity=3, pain_character="sharp_brief", pain_trigger=["cold"], temperature_c=36.8),
        _demo("DEMO-03", "severe_pulpal_pain", age=41, sex="female", tooth_region="mandibular molar",
              chief_complaint="severe throbbing pain in a lower molar, woke me at night", duration="3 days",
              pain_severity=9, pain_character="throbbing", pain_trigger=["hot", "lying down"], temperature_c=36.9),
        _demo("DEMO-04", "facial_swelling", age=28, sex="male", tooth_region="mandibular molar",
              chief_complaint="severe lower right posterior tooth pain with facial swelling", duration="2 days",
              pain_severity=8, pain_character="throbbing", pain_trigger=["biting"], swelling=True,
              swelling_location="right mandibular buccal region", fever=True, temperature_c=38.1,
              previous_root_canal=True, progressive_symptoms=True),
        _demo("DEMO-05", "impacted_third_molar", age=22, sex="female", tooth_region="mandibular third molar",
              chief_complaint="pain behind the last lower tooth, partially erupted wisdom tooth", duration="1 week",
              pain_severity=5, pain_character="dull", pain_trigger=["chewing"], third_molar_symptoms=True,
              temperature_c=36.8),
        _demo("DEMO-06", "periodontal_complaint", age=58, sex="male",
              chief_complaint="receding gums and teeth feeling loose", duration="over a year",
              pain_severity=2, gingival_involvement=True, bleeding=True, mobility=True,
              medical_history=["type 2 diabetes", "tobacco use"], medications=["metformin"]),
        _demo("DEMO-07", "dental_trauma", age=16, sex="male", tooth_region="maxillary incisor",
              chief_complaint="front tooth knocked out completely in a sports injury", duration="a few hours ago",
              pain_severity=6, trauma=True, tooth_avulsion=True, bleeding=True),
        _demo("DEMO-08", "oral_lesion", age=63, sex="male",
              chief_complaint="non-healing ulcer on the lateral tongue", duration="several weeks",
              pain_severity=1, lesion_present=True, lesion_duration_weeks=8, medical_history=["tobacco use"]),
        _demo("DEMO-09", "orthodontic_complaint", age=19, sex="female",
              chief_complaint="wire poking into the cheek", duration="2 days", pain_severity=2,
              orthodontic_appliance_issue=True),
        _demo("DEMO-10", "ambiguous_vague", age=45, sex="female",
              chief_complaint="something doesn't feel right with my teeth", pain_severity=3,
              info_completeness="sparse"),
    ]


def save_demo_cases() -> None:
    cases = build_demo_cases()
    DEMO_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_CASES_PATH, "w", encoding="utf-8") as f:
        json.dump([c.model_dump(mode="json") for c in cases], f, indent=1)
    _load.cache_clear()
