"""DentoRoute - Streamlit entry point.   streamlit run frontend/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from frontend.components import theme  # noqa: E402

st.set_page_config(page_title="DentoRoute", page_icon=":material/dentistry:", layout="wide")

PAGES = Path(__file__).parent / "pages"
nav = st.navigation([
    st.Page(str(PAGES / "home.py"), title="Home", icon=":material/home:", default=True),
    st.Page(str(PAGES / "case_simulator.py"), title="Case Simulator", icon=":material/stethoscope:"),
    st.Page(str(PAGES / "decision_dashboard.py"), title="Decision Dashboard", icon=":material/dashboard:"),
    st.Page(str(PAGES / "benchmark_lab.py"), title="Benchmark Lab", icon=":material/science:"),
    st.Page(str(PAGES / "case_history.py"), title="Case History", icon=":material/history:"),
    st.Page(str(PAGES / "architecture.py"), title="Architecture", icon=":material/account_tree:"),
    st.Page(str(PAGES / "about_safety.py"), title="About / Safety", icon=":material/shield:"),
])
theme.sidebar_controls()
nav.run()
