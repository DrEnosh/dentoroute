from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backend.services.jev_service import QUESTION_SPECS  # noqa: E402
from backend.utils.config import load_routing_policy  # noqa: E402
from frontend.components import theme  # noqa: E402

theme.page_header("Architecture", "Jev decides, plain Python applies policy, Claude explains. Each layer has one job.")

DOT = """digraph G {
 rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.45;
 node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#cbd5e1", fillcolor="#f1f5f9"];
 edge [color="#64748b", fontname="Helvetica", fontsize=10];
 case [label="Synthetic dental case\\n(structured state)"];
 jev  [label="TypeSafe Jev\\n7 atomic questions, 1 request\\nprobabilities + confidence", fillcolor="#0f766e", fontcolor="white", color="#0f766e"];
 pol  [label="Routing policy\\nPython + YAML thresholds\\n(deterministic)", fillcolor="#e0f2fe"];
 route [label="Route\\nautomated / urgent /\\nhuman review / escalate", fillcolor="#fef3c7"];
 claude [label="Claude explanation\\ngrounded in input facts\\ncannot change decisions", fillcolor="#ede9fe"];
 out [label="Human-readable summary"];
 cache [label="SQLite\\ncache + history", shape=cylinder, fillcolor="#e2e8f0"];
 case -> jev -> pol -> route -> claude -> out;
 jev -> cache [style=dashed]; route -> cache [style=dashed];
}"""
st.graphviz_chart(DOT, width="stretch")

st.subheader("Layer responsibilities")
st.dataframe(pd.DataFrame([
    {"Layer": "Jev (decision layer)", "Does": "Answers 7 questions with probabilities and confidence", "Never does": "Trigger actions, write explanations"},
    {"Layer": "Routing policy", "Does": "Maps probabilities to one of 4 routes using YAML thresholds", "Never does": "Call a model; hide why it chose a route"},
    {"Layer": "Claude (explanation)", "Does": "Summarizes facts + decisions in plain language", "Never does": "Change a decision, add findings, diagnose or recommend treatment"},
    {"Layer": "Demo simulator", "Does": "Produces Jev-shaped responses with no API key", "Never does": "Claim to be Jev; is always labelled"},
]), hide_index=True, width="stretch")

st.subheader("The seven Jev questions")
rows = []
for qid, spec in QUESTION_SPECS.items():
    crit = spec["criteria"]
    options = list(crit) if not isinstance(crit, list) else [c.split(":")[0] if len(c) < 40 else c[:38] + "..." for c in crit]
    rows.append({"id": qid, "primitive": spec["type"].upper(), "levels / labels": ", ".join(options) if spec["type"] != "noul" else "yes / no (probability)"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Atomic questions in one request, as the Jev docs recommend. Closed label sets: the model cannot invent a specialty or urgency level.")

st.subheader("Routing policy: conservative safety gate (live from config/routing_policy.yaml)")
policy = load_routing_policy()
th = policy["thresholds"]
fmt = lambda k: th[k]  # noqa: E731
st.dataframe(pd.DataFrame([
    {"Rule": "R0", "Fires when": "model output is invalid or missing", "Threshold": "-", "Route": "human review (fail-safe)"},
    {"Rule": "R1", "Fires when": "human-review probability >=", "Threshold": fmt("human_review_probability"), "Route": "escalate to human"},
    {"Rule": "R2", "Fires when": "infection-concern probability >=", "Threshold": fmt("infection_probability"), "Route": "urgent human review"},
    {"Rule": "R3", "Fires when": "urgency score >=", "Threshold": fmt("urgency_score"), "Route": "urgent human review"},
    {"Rule": "R3", "Fires when": "P(URGENT) + P(EMERGENCY) >=", "Threshold": fmt("urgent_tail_probability"), "Route": "urgent human review"},
    {"Rule": "R3", "Fires when": "same-day review probability >=", "Threshold": fmt("same_day_probability"), "Route": "urgent human review"},
    {"Rule": "R3", "Fires when": "learned reference-risk score >=", "Threshold": fmt("risk_score"), "Route": "urgent human review"},
    {"Rule": "R4", "Fires when": "urgency confidence <", "Threshold": fmt("urgency_confidence_min"), "Route": "human review"},
    {"Rule": "R4", "Fires when": "specialty confidence <", "Threshold": fmt("specialty_confidence_min"), "Route": "human review"},
    {"Rule": "R4", "Fires when": "top specialty probability <", "Threshold": fmt("specialty_top_probability_min"), "Route": "human review"},
    {"Rule": "R4", "Fires when": "infection / same-day / human-review certainty |2p-1| <", "Threshold": fmt("boolean_certainty_min"), "Route": "human review"},
    {"Rule": "R5", "Fires when": "specialty is one of " + ", ".join(policy.get("human_only_specialties", [])), "Threshold": "-", "Route": "human review"},
    {"Rule": "R6", "Fires when": "none of the above fired", "Threshold": "-", "Route": "automated specialty route"},
]).astype(str), hide_index=True, width="stretch")
st.caption(policy["disclaimer"] + " First matching rule wins. Automation is the exception: a case is automated only when every gate passes. "
           "Hard safety limits live in code (backend/services/policy_service.py); a policy file looser than them is rejected.")
prov = policy.get("provenance") or {}
if prov.get("data_mode") == "demo":
    st.warning("These thresholds were selected on SIMULATED Jev outputs (train + calibration splits only). "
               "Re-run scripts/tune_policy.py on a live benchmark before relying on them.")

st.subheader("Code layout")
st.code("""backend/
  services/   jev_service.py (only file that touches the TypeSafe SDK) · routing_service.py · claude_service.py
              benchmark_service.py · metrics_service.py · calibration_service.py · pipeline_service.py
  routers/    FastAPI endpoints          repositories/  SQLite access          schemas/  Pydantic models
config/routing_policy.yaml   data/ground_truth_rules.yaml   data/synthetic_cases.json
frontend/     Streamlit pages + Plotly charts""", language="text")
