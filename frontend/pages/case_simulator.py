from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend import api_client  # noqa: E402
from frontend.components import results, theme  # noqa: E402

theme.page_header("Case Simulator", "Pick or build a synthetic case, run it through Jev and the routing policy.")

PAIN_CHARACTERS = ["", "throbbing", "spontaneous", "lingering", "sharp_brief", "dull", "pressure"]
TRIGGERS = ["biting", "chewing", "cold", "heat", "sweet", "lying down", "spontaneous"]
FLAGS = [
    ("fever", "Fever"), ("swelling", "Swelling"), ("progressive_symptoms", "Symptoms progressing"), ("pus_or_drainage", "Pus / drainage"),
    ("trismus", "Restricted jaw opening"), ("lymphadenopathy", "Lymph node enlargement"), ("difficulty_swallowing", "Difficulty swallowing"),
    ("difficulty_breathing", "Difficulty breathing"), ("trauma", "Recent trauma"), ("tooth_fracture", "Fractured tooth"),
    ("tooth_avulsion", "Tooth knocked out"), ("previous_root_canal", "Previous root canal"), ("bleeding", "Bleeding"),
    ("mobility", "Tooth mobility"), ("third_molar_symptoms", "Wisdom-tooth symptoms"), ("lesion_present", "Mouth lesion"),
    ("jaw_clicking", "Jaw clicking"), ("jaw_locking", "Jaw locking"),
]


@st.cache_data(show_spinner=False, ttl=60)
def _demo_cases() -> list[dict]:
    return api_client.demo_cases()


@st.cache_data(show_spinner=False, ttl=60)
def _categories() -> dict[str, int]:
    return api_client.categories()


def manual_form() -> dict:
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 0, 110, 34)
    sex = c2.selectbox("Sex", ["female", "male", "other"])
    completeness = c3.selectbox("Information provided", ["complete", "partial", "sparse"])
    complaint = st.text_input("Chief complaint", "pain in a lower molar")
    c1, c2, c3 = st.columns(3)
    duration = c1.text_input("Duration", "2 days")
    pain = c2.slider("Pain (0-10)", 0, 10, 5)
    character = c3.selectbox("Pain character", PAIN_CHARACTERS)
    triggers = st.multiselect("Pain triggers", TRIGGERS)
    cols = st.columns(3)
    state: dict = {}
    for i, (key, label) in enumerate(FLAGS):
        state[key] = cols[i % 3].checkbox(label, key=f"flag_{key}")
    temp = st.number_input("Temperature (°C, optional)", 35.0, 42.0, 37.0, 0.1) if state["fever"] else None
    history = st.text_input("Medical history (comma separated)", "")
    meds = st.text_input("Medications (comma separated)", "")
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    state.update(
        age=age, sex=sex, info_completeness=completeness, chief_complaint=complaint or None, duration=duration or None,
        pain_severity=pain, pain_character=character or None, pain_trigger=triggers, temperature_c=temp,
        medical_history=split(history), medications=split(meds), patient_id="SYN-MANUAL",
    )
    return state


source = st.radio("Case source", ["Demo cases (featured)", "Random synthetic case", "Build manually"], horizontal=True)
case_id: str | None = None
case_state: dict | None = None
reference_hint = ""

try:
    if source.startswith("Demo"):
        demos = _demo_cases()
        labels = {f"{c['patient_id']} · {c['category'].replace('_', ' ')}": c for c in demos}
        pick = st.selectbox("Featured case", list(labels))
        chosen = labels[pick]
        case_id, case_state = chosen["patient_id"], chosen["state"]
    elif source.startswith("Random"):
        cats = _categories()
        c1, c2, c3 = st.columns([2, 1, 1])
        cat = c1.selectbox("Scenario category", ["any"] + list(cats))
        seed = c2.number_input("Seed", 0, 10_000, 1)
        if c3.button("Draw case", width="stretch") or "drawn" not in st.session_state:
            st.session_state["drawn"] = api_client.generate_case(None if cat == "any" else cat, int(seed))
        chosen = st.session_state["drawn"]
        case_id, case_state = chosen["patient_id"], chosen["state"]
        st.caption(f"Drawn {case_id} ({chosen['category'].replace('_', ' ')}) from the 500+ case synthetic dataset.")
    else:
        case_state = manual_form()
except api_client.ApiError as exc:
    st.error(exc.message)
    st.stop()

with st.expander("Structured case state (exactly what Jev receives)", expanded=False):
    shown = {k: v for k, v in (case_state or {}).items() if v not in (None, False, [], "")}
    st.json(shown, expanded=True)
    st.caption("Fields that are false/empty are still sent as explicit negatives. The synthetic ID and any reference labels are never sent.")

explain = st.checkbox("Generate plain-language explanation", value=True)
if st.button("Evaluate with DentoRoute", type="primary"):
    with st.spinner("Asking Jev seven questions in one request..."):
        try:
            res = api_client.evaluate(case_id=case_id, case=None if case_id else case_state, demo=theme.demo_override(), explain=explain)
            st.session_state["current"] = res
        except api_client.ApiError as exc:
            st.session_state["current"] = {"error": exc.message}

res = st.session_state.get("current")
if res:
    st.divider()
    if not res.get("error"):
        st.subheader(f"Result for {res['patient_id']}")
        if res["mode"] == "demo":
            st.caption("Simulated Jev response (Demo Mode).")
    results.render(res, key="sim")
    if not res.get("error"):
        theme.nav_link("pages/decision_dashboard.py", "Open this result in the Decision Dashboard", ":material/dashboard:")
