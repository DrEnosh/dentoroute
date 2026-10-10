"""Render one evaluation (the EvaluateCaseResponse dict) - shared by Simulator, Dashboard and History."""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

from backend.services import export_service
from frontend.components import charts
from frontend.components.theme import (
    BAND_COLORS, ROUTE_COLORS, ROUTE_TEXT, URGENCY_COLORS, card, chip, esc, research_mode, urgency_chip,
)


def show_error(res: dict) -> None:
    st.error(res.get("error") or "Evaluation failed.")
    st.caption("Nothing was routed automatically. Fix the problem above or switch on Demo Mode in the sidebar. "
               "The failed attempt is recorded in Case History.")


def summary_cards(res: dict) -> None:
    d, r = res["decision"], res["routing"]
    cols = st.columns(4)
    cols[0].markdown(card("Urgency", d["urgency"]["label"], f"score {d['urgency']['score']:.2f} / 3",
                          URGENCY_COLORS.get(d["urgency"]["label"])), unsafe_allow_html=True)
    cols[1].markdown(card("Best-matching service", d["specialty"]["label"].replace("_", " "),
                          f"top probability {d['specialty']['probabilities'][d['specialty']['label']]:.0%}"), unsafe_allow_html=True)
    cols[2].markdown(card("Workflow route", r["workflow_label"], f"rule {r['triggered_rule']}", ROUTE_COLORS.get(r["route"])), unsafe_allow_html=True)
    cols[3].markdown(card("Confidence band", r["confidence_band"].upper(),
                          "human review required" if r["requires_human"] else "automated, no human gate triggered",
                          BAND_COLORS[r["confidence_band"]]), unsafe_allow_html=True)


def decision_table(d: dict) -> list[dict[str, Any]]:
    return [
        {"Question": "Urgency", "Jev answer": d["urgency"]["label"], "Detail": f"score {d['urgency']['score']:.2f}", "Confidence": f"{d['urgency']['confidence']:.0%}"},
        {"Question": "Specialty", "Jev answer": d["specialty"]["label"], "Detail": f"p={d['specialty']['probabilities'][d['specialty']['label']]:.0%}", "Confidence": f"{d['specialty']['confidence']:.0%}"},
        {"Question": "Infection concern", "Jev answer": "yes" if d["infection_concern"]["probability"] >= .5 else "no", "Detail": f"p(yes)={d['infection_concern']['probability']:.0%}", "Confidence": f"{abs(2 * d['infection_concern']['probability'] - 1):.0%}"},
        {"Question": "Same-day review", "Jev answer": "yes" if d["same_day_review"]["probability"] >= .5 else "no", "Detail": f"p(yes)={d['same_day_review']['probability']:.0%}", "Confidence": f"{abs(2 * d['same_day_review']['probability'] - 1):.0%}"},
        {"Question": "Human review", "Jev answer": "yes" if d["human_review"]["probability"] >= .5 else "no", "Detail": f"p(yes)={d['human_review']['probability']:.0%}", "Confidence": f"{abs(2 * d['human_review']['probability'] - 1):.0%}"},
        {"Question": "Complexity", "Jev answer": d["complexity"]["label"], "Detail": f"score {d['complexity']['score']:.2f}", "Confidence": f"{d['complexity']['confidence']:.0%}"},
        {"Question": "Referral priority", "Jev answer": d["referral_priority"]["label"], "Detail": f"score {d['referral_priority']['score']:.2f}", "Confidence": f"{d['referral_priority']['confidence']:.0%}"},
    ]


def explanation_block(e: dict | None) -> None:
    if not e:
        return
    label = "Claude explanation" if e["source"] == "claude" else "Template explanation (Claude not used)"
    st.subheader("Plain-language summary")
    st.caption(label + (f" - {e['model']}" if e.get("model") else ""))
    st.write(e["summary"])
    st.write(e["engine_findings"])
    st.write(e["why_route"])
    c1, c2 = st.columns(2)
    c1.markdown(f"**Human review.** {e['human_review_note']}")
    c2.markdown(f"**Uncertainty.** {e['uncertainty']}")
    if e["urgency_signals"]:
        st.markdown("**Signals from the case:** " + " ".join(chip(s, "#475569") for s in e["urgency_signals"]), unsafe_allow_html=True)
    for w in e.get("warnings", []):
        st.info(w)
    st.caption(e["disclaimer"])


def research_panel(res: dict) -> None:
    d = res["decision"]
    with st.expander("Research Mode: provider details", expanded=True):
        c = st.columns(4)
        c[0].metric("Model", d["model"])
        c[1].metric("Latency", f"{d['latency_ms']:.0f} ms" if d.get("latency_ms") is not None else "not measured")
        c[2].metric("Cache", "hit" if d.get("cached") else "miss")
        c[3].metric("Mode", d["mode"])
        if d.get("request_id"):
            st.caption(f"request_id: {d['request_id']}")
        if d.get("usage"):
            st.caption(f"usage: {d['usage']}")
        r = res["routing"]
        st.markdown("**Policy thresholds in force** (config/routing_policy.yaml, research-demo values):")
        st.json(r["thresholds"], expanded=False)
        st.markdown("**All probabilities**")
        st.dataframe(
            [{"question": q, "label": lab, "probability": p} for q in ("urgency", "specialty", "complexity", "referral_priority")
             for lab, p in d[q]["probabilities"].items()], width="stretch", hide_index=True, height=240)
        if res.get("reference"):
            st.markdown("**Rule-based reference labels** (educational synthetic rubric, not clinical truth)")
            ref = {k: v for k, v in res["reference"].items() if k != "fired_rules"}
            st.json(ref, expanded=False)
            if res["reference"].get("fired_rules"):
                st.caption("Rules that fired: " + json.dumps(res["reference"]["fired_rules"]))
        st.markdown("**Raw model response**")
        st.json(d.get("raw_response") or {}, expanded=False)


def exports(res: dict, key: str) -> None:
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Report (Markdown)", export_service.case_report_markdown(res), f"{res['patient_id']}_report.md",
                       "text/markdown", key=f"md_{key}")
    c2.download_button("Report (JSON)", export_service.case_report_json(res, include_raw=research_mode()),
                       f"{res['patient_id']}_report.json", "application/json", key=f"js_{key}")


def render(res: dict, key: str = "r") -> None:
    """Full evaluation view."""
    if res.get("error"):
        show_error(res)
        return
    d, r = res["decision"], res["routing"]
    summary_cards(res)
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        explanation_block(res.get("explanation"))
    with right:
        st.subheader("Why this route")
        st.markdown(f"{chip(ROUTE_TEXT[r['route']], ROUTE_COLORS[r['route']])}", unsafe_allow_html=True)
        for line in r["reasons"]:
            st.markdown(f'<div class="dr-step">{esc(line)}</div>', unsafe_allow_html=True)
        st.caption(r["disclaimer"])
    st.subheader("Jev's seven answers")
    st.dataframe(decision_table(d), width="stretch", hide_index=True)
    exports(res, key)
    if research_mode():
        research_panel(res)
