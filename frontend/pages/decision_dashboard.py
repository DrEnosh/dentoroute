from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backend.services.calibration_service import histogram  # noqa: E402
from frontend import api_client  # noqa: E402
from frontend.components import charts, theme  # noqa: E402
from frontend.components.results import show_error, summary_cards  # noqa: E402

theme.page_header("Decision Dashboard", "Probabilities, confidence and the routing path for one case, plus a view across all evaluated cases.")

tab_case, tab_all = st.tabs(["This case", "All evaluated cases"])

with tab_case:
    res = st.session_state.get("current")
    if not res:
        st.info("No case evaluated yet. Use the Case Simulator first.")
        theme.nav_link("pages/case_simulator.py", "Go to Case Simulator", ":material/stethoscope:")
    elif res.get("error"):
        show_error(res)
    else:
        d, r = res["decision"], res["routing"]
        st.subheader(f"{res['patient_id']}")
        summary_cards(res)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            theme.plot(charts.urgency_bars(d), "dash_urg")
        with c2:
            theme.plot(charts.specialty_bars(d), "dash_spec")
        st.markdown("##### Confidence and concern gauges")
        th = r["thresholds"]
        t_spec = th.get("specialty_confidence_min", th.get("specialty_confidence"))
        t_inf = th.get("infection_probability", th.get("infection_probability_urgent"))
        g = st.columns(5)
        with g[0]:
            theme.plot(charts.gauge(d["urgency"]["confidence"], "Urgency confidence"), "g1")
        with g[1]:
            theme.plot(charts.gauge(d["specialty"]["confidence"], "Specialty confidence", threshold=t_spec), "g2")
        with g[2]:
            theme.plot(charts.gauge(d["infection_concern"]["probability"], "Infection concern", invert=True, threshold=t_inf), "g3")
        with g[3]:
            theme.plot(charts.gauge(d["same_day_review"]["probability"], "Same-day review", invert=True, threshold=th["same_day_probability"]), "g4")
        with g[4]:
            theme.plot(charts.gauge(d["human_review"]["probability"], "Human review", invert=True, threshold=th["human_review_probability"]), "g5")
        st.caption(f"Red marker = routing threshold of the active policy ({r.get('policy_name', '?')}, research-demo values, not clinical thresholds).")
        c1, c2 = st.columns(2)
        with c1:
            theme.plot(charts.probability_bars(d["complexity"]["probabilities"], "Complexity (Jev Score)"), "dash_cx")
        with c2:
            theme.plot(charts.probability_bars(d["referral_priority"]["probabilities"], "Referral priority (Jev Score)"), "dash_rp")
        st.markdown("##### Routing path")
        st.caption("The highlighted rule is the first one that fired. Rules are checked top to bottom.")
        st.graphviz_chart(charts.routing_path_dot(r), width="stretch")
        for line in r["reasons"]:
            st.markdown(f'<div class="dr-step">{theme.esc(line)}</div>', unsafe_allow_html=True)

with tab_all:
    try:
        runs = [x for x in api_client.history(limit=1000) if x.get("decision")]
    except api_client.ApiError as exc:
        st.error(exc.message)
        runs = []
    if not runs:
        st.info("No evaluations stored yet.")
        if st.button("Evaluate the 10 featured demo cases"):
            with st.spinner("Evaluating..."):
                for c in api_client.demo_cases():
                    api_client.evaluate(case_id=c["patient_id"], demo=theme.demo_override(), explain=False)
            st.rerun()
    else:
        df = pd.DataFrame([{"case": x["patient_id"], "urgency": x["urgency"], "specialty": x["specialty"], "route": x["route"],
                            "min_confidence": x["min_confidence"], "mode": x["mode"]} for x in runs])
        st.caption(f"{len(df)} stored evaluations" + (" (includes simulated Demo Mode runs)" if (df['mode'] == 'demo').any() else ""))
        c1, c2 = st.columns(2)
        with c1:
            theme.plot(charts.route_distribution(df["route"].value_counts().to_dict()), "all_route")
        with c2:
            order = ["ROUTINE", "PRIORITY", "URGENT", "EMERGENCY"]
            vc = df["urgency"].value_counts()
            theme.plot(charts.category_bars({k: int(vc.get(k, 0)) for k in order}, "Urgency distribution", theme.URGENCY_COLORS), "all_urg")
        c1, c2 = st.columns(2)
        with c1:
            theme.plot(charts.histogram_bars(histogram(df["min_confidence"].dropna().tolist()), "Routing confidence (min of urgency, specialty)", xtitle="Confidence"), "all_conf")
        with c2:
            vc = df["specialty"].value_counts().sort_values()
            theme.plot(charts.probability_bars({k.replace("_", " "): float(v) / len(df) for k, v in vc.items()}, "Predicted specialty share", horizontal=True), "all_spec")
