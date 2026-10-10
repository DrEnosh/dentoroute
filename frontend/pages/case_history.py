from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backend.schemas.labels import SPECIALTIES, URGENCY_LEVELS  # noqa: E402
from frontend import api_client  # noqa: E402
from frontend.components import results, theme  # noqa: E402

theme.page_header("Case History", "Every evaluation is stored locally (SQLite), including failed ones.")

c1, c2, c3, c4 = st.columns(4)
urgency = c1.selectbox("Urgency", ["any"] + URGENCY_LEVELS)
specialty = c2.selectbox("Specialty", ["any"] + SPECIALTIES)
conf_range = c3.slider("Routing confidence", 0.0, 1.0, (0.0, 1.0), 0.05)
search = c4.text_input("Case ID contains")

try:
    runs = api_client.history(
        urgency=None if urgency == "any" else urgency, specialty=None if specialty == "any" else specialty,
        min_confidence=conf_range[0] if conf_range[0] > 0 else None, max_confidence=conf_range[1] if conf_range[1] < 1 else None,
        search=search or None, limit=500,
    )
except api_client.ApiError as exc:
    st.error(exc.message)
    st.stop()

if not runs:
    st.info("Nothing here yet. Evaluate a case in the Case Simulator, or relax the filters.")
    st.stop()

df = pd.DataFrame([{
    "run": r["run_id"], "time": r["created_at"][:19].replace("T", " "), "case": r["patient_id"], "mode": r["mode"],
    "urgency": r["urgency"], "specialty": r["specialty"], "route": r["route"],
    "confidence": None if r["min_confidence"] is None else round(r["min_confidence"], 2),
    "latency_ms": None if r["latency_ms"] is None else round(r["latency_ms"]), "error": r["error"],
} for r in runs])
st.caption(f"{len(df)} runs")
event = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="hist_table")
st.download_button("Export filtered history (CSV)", df.to_csv(index=False), "dentoroute_history.csv", "text/csv")

sel = event.selection.rows if event and event.selection else []
if sel:
    run = runs[sel[0]]
    st.divider()
    st.subheader(f"Run {run['run_id']}: {run['patient_id']}")
    try:
        reference = api_client.case_detail(run["patient_id"])["reference"]
    except api_client.ApiError:
        reference = None  # manual cases have no reference labels
    results.render({
        "patient_id": run["patient_id"], "mode": run["mode"], "decision": run["decision"], "routing": run["routing"],
        "explanation": run["explanation"], "error": run["error"], "reference": reference,
    }, key=f"hist{run['run_id']}")
    with st.expander("Case state"):
        st.json(run["state"], expanded=False)
else:
    st.caption("Select a row to open the full result.")
