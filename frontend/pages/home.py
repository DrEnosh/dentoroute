from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend import api_client  # noqa: E402
from frontend.components import theme  # noqa: E402
from backend.utils.config import TAGLINE  # noqa: E402

theme.page_header("DentoRoute", f"{TAGLINE}  Powered by TypeSafe Jev.")

st.markdown(
    "DentoRoute turns a synthetic dental case into **structured, probabilistic decisions**, applies a transparent "
    "**confidence-gated routing policy**, and explains the result in plain language. It is a research prototype for "
    "studying how decision-layer confidence can decide *when software may act and when a human must look*."
)

try:
    h = api_client.health()
    cols = st.columns(4)
    cols[0].markdown(theme.card("Synthetic cases", str(h["dataset_cases"]), "21 scenario categories"), unsafe_allow_html=True)
    cols[1].markdown(theme.card("Decision questions", "7", "asked in one Jev request"), unsafe_allow_html=True)
    cols[2].markdown(theme.card("Mode", theme.current_mode().upper(), "Demo = simulated Jev"), unsafe_allow_html=True)
    cols[3].markdown(theme.card("Decision model", h["jev_model"] if theme.current_mode() == "live" else "simulator", "explanations: " + h["claude_model"]), unsafe_allow_html=True)
except api_client.ApiError as exc:
    st.error(exc.message)

st.subheader("How a case flows through the system")
steps = [
    ("1. Dental case", "A synthetic patient state: symptoms, findings, history. No real patient data, ever."),
    ("2. Jev decision layer", "Seven atomic questions in one request: urgency, specialty, infection concern, same-day review, human review, complexity, referral priority. Every answer carries probabilities."),
    ("3. Routing policy", "Plain Python and a YAML file. Thresholds on those probabilities pick one of four routes. Jev never triggers actions itself."),
    ("4. Claude explanation", "Writes a short summary from the facts and decisions. It cannot change a decision, and an invented finding is rejected."),
    ("5. Human-readable summary", "Route, reasons, uncertainty and a clear note when a human must review."),
]
for title, text in steps:
    st.markdown(f'<div class="dr-step"><b>{theme.esc(title)}</b><br>{theme.esc(text)}</div>', unsafe_allow_html=True)

st.subheader("Start here")
c1, c2, c3 = st.columns(3)
theme.nav_link("pages/case_simulator.py", "Try a demo case", ":material/stethoscope:", c1)
theme.nav_link("pages/benchmark_lab.py", "Run the benchmark", ":material/science:", c2)
theme.nav_link("pages/architecture.py", "See the architecture", ":material/account_tree:", c3)
st.caption("Tip: the ambiguous demo case (DEMO-10) shows the point of the project: when Jev is unsure, DentoRoute hands the case to a person instead of guessing.")
