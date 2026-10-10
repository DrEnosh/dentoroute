"""Train / calibration / holdout split for the safety-gate study.

  train        fit the learned risk score
  calibration  choose thresholds (the ONLY place threshold decisions are made)
  holdout      evaluated once, after everything is frozen; never used to fit, select or compare candidates

The split is stratified by reference urgency, seeded, and saved to data/splits.json so it is auditable and
stable. It is computed from case IDs and reference labels only, never from model outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from backend.services.ground_truth_service import label_case
from backend.services.synthetic_case_service import load_dataset
from backend.utils.config import DATA_DIR

SPLITS_PATH = DATA_DIR / "splits.json"
SPLIT_NAMES = ("train", "calibration", "holdout")
FRACTIONS = {"train": 0.40, "calibration": 0.30, "holdout": 0.30}
SEED = 20261006
TUNING_SPLITS = ("train", "calibration")


def make_splits(case_ids: list[str], strata: list[str], seed: int = SEED, fractions: dict[str, float] | None = None) -> dict[str, str]:
    f = fractions or FRACTIONS
    if abs(sum(f.values()) - 1) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    ids = list(case_ids)
    train_ids, rest_ids, _, rest_strata = train_test_split(ids, strata, train_size=f["train"], stratify=strata, random_state=seed)
    cal_share = f["calibration"] / (f["calibration"] + f["holdout"])
    cal_ids, hold_ids = train_test_split(rest_ids, train_size=cal_share, stratify=rest_strata, random_state=seed + 1)
    out = {i: "train" for i in train_ids}
    out.update({i: "calibration" for i in cal_ids})
    out.update({i: "holdout" for i in hold_ids})
    return out


def build_default_splits() -> dict[str, Any]:
    cases = load_dataset()
    ids = [c.patient_id for c in cases]
    strata = [(c.reference or label_case(c.state)).urgency for c in cases]
    assignment = make_splits(ids, strata)
    counts = {s: sum(1 for v in assignment.values() if v == s) for s in SPLIT_NAMES}
    return {"seed": SEED, "fractions": FRACTIONS, "stratified_by": "reference urgency", "counts": counts, "assignment": assignment}


def get_splits(path: Path | None = None, rebuild: bool = False) -> dict[str, Any]:
    p = path or SPLITS_PATH
    if p.exists() and not rebuild:
        return json.loads(p.read_text(encoding="utf-8"))
    data = build_default_splits()
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def attach_splits(rows: list[dict[str, Any]], splits: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Copy of the rows with a `split` field. Rows for cases outside the split file are dropped (never guessed)."""
    assignment = (splits or get_splits())["assignment"]
    return [{**r, "split": assignment[r["case_id"]]} for r in rows if r["case_id"] in assignment]
