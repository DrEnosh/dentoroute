"""Extract the *positive* facts present in a case, and detect claims that are not supported by them.

Used by the explanation layer so Claude (or the template) can only talk about facts that exist in the
input, and to reject explanations that mention findings the case does not contain.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from backend.schemas.case import CaseState

# term to look for in generated prose -> predicate over the case that must be TRUE for the term to be allowed
_CLAIM_RULES: list[tuple[str, str, Callable[[CaseState], bool]]] = [
    ("fever", r"\bfever(ish)?\b|\bpyrexia\b|\bfebrile\b", lambda s: s.fever),
    ("swelling", r"\bswell(ing|en)\b", lambda s: s.swelling or s.pus_or_drainage),
    ("trismus", r"\btrismus\b|restricted (mouth|jaw) opening|limited (mouth|jaw) opening", lambda s: s.trismus),
    ("difficulty swallowing", r"difficulty swallowing|trouble swallowing|dysphagia", lambda s: s.difficulty_swallowing),
    ("difficulty breathing", r"difficulty breathing|trouble breathing|shortness of breath|airway", lambda s: s.difficulty_breathing or s.difficulty_swallowing),
    ("lymph node enlargement", r"lymph", lambda s: s.lymphadenopathy),
    ("trauma", r"\btrauma\b|\binjur(y|ed)\b", lambda s: s.trauma),
    ("bleeding", r"\bbleed(ing|s)?\b", lambda s: s.bleeding),
    ("pus or drainage", r"\bpus\b|\bdrainage\b|\bdischarge\b", lambda s: s.pus_or_drainage),
    ("root canal", r"root canal", lambda s: s.previous_root_canal),
    ("tooth loss (avulsion)", r"knocked[- ]out|avuls", lambda s: s.tooth_avulsion),
    ("fracture", r"\bfractur(e|ed)\b|\bchipped\b|\bbroken tooth\b", lambda s: s.tooth_fracture),
    ("lesion or ulcer", r"\blesion\b|\bulcer\b", lambda s: s.lesion_present),
    ("wisdom tooth", r"wisdom|third molar", lambda s: s.third_molar_symptoms),
    ("pain on biting", r"(pain|tender\w*|sore\w*) (on|when|with) (biting|chewing)|biting pain", lambda s: "biting" in s.pain_trigger or "chewing" in s.pain_trigger),
    ("severe pain", r"\bsevere pain\b|\bsevere\b[^.]{0,20}\bpain\b", lambda s: s.pain_severity >= 7),
    ("jaw joint symptoms", r"\bTMJ\b|jaw (clicking|locking)", lambda s: s.jaw_clicking or s.jaw_locking),
    ("diabetes", r"diabet", lambda s: any("diabet" in h.lower() for h in s.medical_history)),
]


def positive_signals(s: CaseState) -> list[str]:
    """Human-readable list of findings that are actually present in the case."""
    out: list[str] = []
    if s.pain_severity >= 1:
        desc = f"pain rated {s.pain_severity}/10"
        if s.pain_character:
            desc += f" ({s.pain_character.replace('_', ' ')})"
        out.append(desc)
    if s.pain_trigger:
        out.append("pain triggered by " + ", ".join(s.pain_trigger))
    if s.swelling:
        out.append("swelling" + (f" ({s.swelling_location})" if s.swelling_location else ""))
    if s.fever:
        out.append("fever" + (f" ({s.temperature_c} °C)" if s.temperature_c else ""))
    flags = {
        "progressive_symptoms": "symptoms progressing", "pus_or_drainage": "pus or drainage",
        "trismus": "restricted jaw opening (trismus)", "lymphadenopathy": "lymph node enlargement",
        "difficulty_swallowing": "difficulty swallowing", "difficulty_breathing": "difficulty breathing",
        "trauma": "recent trauma", "tooth_fracture": "fractured tooth", "tooth_avulsion": "tooth knocked out",
        "previous_root_canal": "previous root canal treatment",
        "recent_extraction_or_procedure": "recent extraction or procedure", "bleeding": "bleeding",
        "mobility": "tooth mobility", "gingival_involvement": "gum involvement",
        "third_molar_symptoms": "wisdom-tooth symptoms", "lesion_present": "mouth lesion",
        "jaw_clicking": "jaw clicking", "jaw_locking": "jaw locking",
        "restoration_or_prosthesis_issue": "restoration/prosthesis problem",
        "orthodontic_appliance_issue": "orthodontic appliance problem", "administrative_request": "administrative request",
    }
    for field, text in flags.items():
        if getattr(s, field):
            out.append(text)
    if s.lesion_present and s.lesion_duration_weeks:
        out.append(f"lesion present for about {s.lesion_duration_weeks} weeks")
    if s.info_completeness != "complete":
        out.append(f"{s.info_completeness} information provided")
    out.extend(s.medical_history)
    return out


def find_unsupported_claims(text: str, s: CaseState) -> list[str]:
    """Return clinical terms mentioned in `text` that the case does NOT support (possible invention).

    Negated mentions ("no fever", "without swelling", "fever: false") are allowed.
    """
    issues: list[str] = []
    for name, pattern, allowed in _CLAIM_RULES:
        if allowed(s):
            continue
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            window = text[max(0, m.start() - 40) : m.start()].lower()
            if re.search(r"\b(no|not|without|absent|absence of|denies|negative for|free of|nor)\b[^.;]{0,35}$", window):
                continue
            issues.append(f"mentions '{m.group(0)}' ({name}), which is not present in the case data")
            break
    return issues
