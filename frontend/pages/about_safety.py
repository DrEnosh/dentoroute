from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from backend.utils.config import DOCS_DIR  # noqa: E402
from frontend.components import theme  # noqa: E402

theme.page_header("About / Safety", "What DentoRoute is, and what it is not.")

st.error("**Research prototype. Not for clinical diagnosis or treatment.** All patient data is synthetic. "
         "Outputs must never be used to make or delay a real care decision.")
doc = DOCS_DIR / "safety.md"
st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "docs/safety.md not found.")
