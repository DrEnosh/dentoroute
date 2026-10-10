"""Structured (synthetic) patient state. NO real patient data / PHI, ever."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CaseState(BaseModel):
    """The only thing sent to Jev. Free of reference labels and scenario categories."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(default="SYN-MANUAL", description="Synthetic ID, e.g. SYN-0001")
    age: int = Field(default=30, ge=0, le=110)
    sex: Literal["male", "female", "other"] = "female"

    chief_complaint: str | None = None
    duration: str | None = None
    pain_severity: int = Field(default=0, ge=0, le=10, description="0-10 self-rated")
    pain_character: str | None = None  # throbbing | spontaneous | lingering | sharp_brief | dull | ...
    pain_trigger: list[str] = Field(default_factory=list)

    swelling: bool = False
    swelling_location: str | None = None
    progressive_symptoms: bool = False
    pus_or_drainage: bool = False
    fever: bool = False
    temperature_c: float | None = None
    trismus: bool = False
    lymphadenopathy: bool = False
    difficulty_swallowing: bool = False
    difficulty_breathing: bool = False

    trauma: bool = False
    tooth_fracture: bool = False
    tooth_avulsion: bool = False
    previous_root_canal: bool = False
    recent_extraction_or_procedure: bool = False
    tooth_region: str | None = None
    bleeding: bool = False
    mobility: bool = False
    gingival_involvement: bool = False
    third_molar_symptoms: bool = False

    lesion_present: bool = False
    lesion_duration_weeks: int | None = Field(default=None, ge=0)
    jaw_clicking: bool = False
    jaw_locking: bool = False
    restoration_or_prosthesis_issue: bool = False
    orthodontic_appliance_issue: bool = False
    administrative_request: bool = False

    info_completeness: Literal["complete", "partial", "sparse"] = "complete"
    medical_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)

    def to_jev_state(self) -> dict:
        """JSON state for Jev: drops the synthetic ID and null values; keeps explicit negatives."""
        data = self.model_dump(exclude={"patient_id"})
        return {k: v for k, v in data.items() if v is not None}


class ReferenceLabels(BaseModel):
    """Deterministic rule-based labels (data/ground_truth_rules.yaml). Educational only."""

    urgency: str
    specialty: str
    infection_concern: bool
    same_day_review: bool
    human_review: bool
    complexity: str
    referral_priority: str
    fired_rules: dict[str, list[str]] = Field(default_factory=dict)


class SyntheticCase(BaseModel):
    patient_id: str
    category: str
    state: CaseState
    reference: ReferenceLabels | None = None
