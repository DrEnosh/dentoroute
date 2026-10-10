"""Shared fixtures. Tests never need live API keys and never touch the real database."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Make sure no real key leaks into tests, and force an isolated DB.
for var in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(var, None)
os.environ["DENTOROUTE_DEMO_MODE"] = "true"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    from backend.repositories import db
    from backend.utils import config

    monkeypatch.setenv("DENTOROUTE_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DENTOROUTE_DEMO_MODE", "true")
    config.reset_settings_cache()
    db.reset_engine()
    yield
    db.reset_engine()
    config.reset_settings_cache()


@pytest.fixture
def severe_case():
    from backend.schemas.case import CaseState

    return CaseState(
        patient_id="SYN-TEST-1", age=28, sex="male",
        chief_complaint="severe lower right posterior tooth pain", duration="2 days",
        pain_severity=8, pain_character="throbbing", pain_trigger=["biting"],
        swelling=True, swelling_location="right mandibular buccal region",
        fever=True, temperature_c=38.1, previous_root_canal=True,
        tooth_region="mandibular molar", progressive_symptoms=True,
    )


@pytest.fixture
def routine_case():
    from backend.schemas.case import CaseState

    return CaseState(patient_id="SYN-TEST-2", age=34, chief_complaint="routine check-up and cleaning")
