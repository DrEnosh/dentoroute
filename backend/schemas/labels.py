"""Closed label sets shared by the whole project (the model may not invent labels)."""
from __future__ import annotations

URGENCY_LEVELS = ["ROUTINE", "PRIORITY", "URGENT", "EMERGENCY"]
COMPLEXITY_LEVELS = ["SIMPLE", "MODERATE", "COMPLEX", "HIGHLY_COMPLEX"]
REFERRAL_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
SPECIALTIES = [
    "general_dentistry",
    "endodontics",
    "periodontics",
    "oral_surgery",
    "prosthodontics",
    "orthodontics",
    "pediatric_dentistry",
    "oral_medicine",
    "oral_pathology",
    "emergency_dental_service",
]
BOOLEAN_QUESTIONS = ["infection_concern", "same_day_review", "human_review"]
ALL_QUESTIONS = [
    "urgency",
    "specialty",
    "infection_concern",
    "same_day_review",
    "human_review",
    "complexity",
    "referral_priority",
]

# Scenario categories used by the synthetic generator (coverage list from the spec).
CATEGORIES = [
    "routine_examination",
    "caries_related",
    "reversible_pulpitis_like",
    "severe_pulpal_pain",
    "periapical_symptoms",
    "post_operative_pain",
    "periodontal_complaint",
    "gingival_bleeding",
    "periodontal_abscess_like",
    "impacted_third_molar",
    "dental_trauma",
    "fractured_tooth",
    "prosthodontic_complaint",
    "orthodontic_complaint",
    "pediatric_complaint",
    "oral_lesion",
    "tmj_complaint",
    "facial_swelling",
    "severe_infection_warning",
    "administrative_nonurgent",
    "ambiguous_vague",
]
