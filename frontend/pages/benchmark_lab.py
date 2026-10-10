from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backend.services import export_service  # noqa: E402
from backend.utils.config import DOCS_DIR  # noqa: E402
from frontend import api_client  # noqa: E402
from frontend.components import charts, theme  # noqa: E402

theme.page_header("Benchmark Lab", "Jev vs Claude on the same synthetic cases, scored against a deterministic rule set. Calibration included.")

try:
    health = api_client.health()
except api_client.ApiError as exc:
    st.error(exc.message)
    st.stop()

QUESTION_NAMES = ["urgency", "specialty", "infection_concern", "same_day_review", "human_review", "complexity", "referral_priority"]


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


# ------------------------------------------------------------------- run controls --
with st.expander("Run a benchmark", expanded=False):
    mode = theme.current_mode()
    c1, c2, c3 = st.columns(3)
    providers = ["jev", "both", "claude"] if health["claude_key_configured"] else ["jev"]
    provider = c1.selectbox("Providers", providers, help="'both' needs a live Jev key and an Anthropic key.")
    size = c2.select_slider("Cases", options=[25, 50, 100, 250, "All"], value="All" if mode == "demo" else 50)
    c3.markdown(f"**Mode:** {mode.upper()}")
    if mode == "demo":
        st.info("Demo Mode: Jev answers come from a deterministic simulator. The run is labelled SIMULATED and Claude is not compared.")
    else:
        st.warning("Live mode sends one request per case to the Jev API (and Claude if selected). Start small.")
    if not health["claude_key_configured"]:
        st.caption("Claude benchmark disabled: ANTHROPIC_API_KEY is not set.")
    if st.button("Run benchmark", type="primary"):
        with st.spinner("Running..."):
            try:
                out = api_client.run_benchmark(provider, None if size == "All" else int(size), theme.demo_override())
                for k, why in out.get("skipped", {}).items():
                    st.info(f"{k}: {why}")
                st.success("Benchmark finished and saved.")
            except api_client.ApiError as exc:
                st.error(exc.message)

try:
    results = api_client.benchmark_results(include_rows=True)
except api_client.ApiError as exc:
    st.error(exc.message)
    st.stop()

if not results["jev"] and not results["claude"]:
    st.info("No benchmark has been run yet. Open 'Run a benchmark' above (Demo Mode finishes in about a second).")
    st.stop()

options = {k: results[k] for k in ("jev", "claude") if results[k]}
which = st.radio("Inspect", list(options), format_func=lambda k: f"{k.title()} ({options[k]['mode']}, {options[k]['n_cases']} cases)", horizontal=True)
run = options[which]
summary = run["summary"]
metrics, calib = summary["metrics"], summary["calibration"]
rows = run["rows"]

if summary["simulated"]:
    st.warning("SIMULATED RUN. These numbers describe the demo simulator, which is NOT Jev. They show how the benchmark works, not how Jev performs.")
st.caption(f"Run #{run['run_id']} · model {run['model']} · finished {run['finished_at']}")

(t_acc, t_cm, t_cal, t_gate, t_pol, t_cmp, t_exp, t_method) = st.tabs(
    ["Accuracy", "Confusion matrices", "Calibration", "Confidence-gated routing", "Safety policy", "Jev vs Claude", "Export", "Method"])

# ------------------------------------------------------------------------ accuracy --
with t_acc:
    lat, cost = metrics["latency"], metrics["cost"]
    c = st.columns(5)
    c[0].markdown(theme.card("Cases scored", f"{metrics['n_valid']}/{metrics['n_cases']}", f"{metrics['n_api_errors']} API errors"), unsafe_allow_html=True)
    c[1].markdown(theme.card("Schema compliance", pct(metrics["schema_compliance"]), f"{metrics['n_invalid_outputs']} invalid outputs"), unsafe_allow_html=True)
    c[2].markdown(theme.card("Mean accuracy", pct(metrics["mean_accuracy_across_questions"]), "across 7 questions"), unsafe_allow_html=True)
    c[3].markdown(theme.card("Median latency", f"{lat['p50_ms']:.0f} ms" if lat["status"] == "measured" else "not measured", f"p95 {lat['p95_ms']:.0f} ms" if lat["status"] == "measured" else "no real API calls"), unsafe_allow_html=True)
    c[4].markdown(theme.card("Cost", f"${cost['total_usd']:.4f}" if cost["status"] == "measured" else "not measured", "set price env vars to enable"), unsafe_allow_html=True)
    st.write("")
    theme.plot(charts.accuracy_bars(metrics["per_question"]), "acc_bars")
    table = []
    for q in QUESTION_NAMES:
        m = metrics["per_question"][q]
        table.append({
            "question": q, "n": m["n"], "accuracy": m["accuracy"], "within ±1 level": m.get("within_one"), "MAE (levels)": m.get("mae_levels"),
            "macro-F1": m.get("macro_f1") if m.get("macro_f1") is not None else m.get("f1"), "top-3": m.get("top3_accuracy"),
            "precision": m.get("precision"), "recall": m.get("recall"), "Brier": m.get("brier"),
        })
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    if lat["status"] == "measured":
        theme.plot(charts.latency_histogram([r["latency_ms"] for r in rows if r.get("latency_ms") and not r.get("cached")]), "lat_hist")
    under = metrics["per_question"]["urgency"].get("under_triage_rate")
    if under is not None:
        st.caption(f"Urgency: {pct(under)} of cases were rated LOWER than the rule-based reference (under-triage) and "
                   f"{pct(metrics['per_question']['urgency'].get('over_triage_rate'))} HIGHER (over-triage).")

# ------------------------------------------------------------------------ confusion --
with t_cm:
    q = st.selectbox("Question", QUESTION_NAMES, key="cm_q")
    norm = st.toggle("Normalise rows", key="cm_norm")
    m = metrics["per_question"][q]
    theme.plot(charts.confusion_heatmap(m["confusion_matrix"], m["labels"], f"{q.replace('_', ' ').title()}: reference vs predicted", norm), "cm_fig")

# ------------------------------------------------------------------------ calibration --
with t_cal:
    st.markdown("Does a stated 80% mean right about 80% of the time? And does higher confidence mean higher accuracy?")
    if which == "claude":
        st.info("Claude returns no probability distributions, only a self-reported confidence for urgency/specialty and probabilities for the three yes/no questions. Treat these plots as verbalized confidence, not calibrated probabilities.")
    qsel = st.selectbox("Question", ["all questions (pooled)"] + QUESTION_NAMES, key="cal_q")
    if qsel.startswith("all"):
        block = {"reliability_bins": calib["pooled"]["reliability_bins"], "ece": calib["pooled"]["ece"], "brier": None, "confidence": calib["pooled"]["confidence"], "confidence_histograms": None}
    else:
        block = calib["per_question"][qsel]
    c1, c2 = st.columns(2)
    with c1:
        theme.plot(charts.reliability_diagram(block["reliability_bins"], "Reliability diagram", block["ece"]), "rel")
    with c2:
        conf = block["confidence"]
        st.markdown("##### Does higher confidence mean higher accuracy?")
        st.markdown(f"**{conf['answer']}**")
        st.caption(f"n={conf['n']}" + (f", Spearman rho={conf['spearman_rho']:.2f}" if conf["spearman_rho"] is not None else ""))
        theme.plot(charts.band_accuracy_bars(conf["bands"], "Accuracy by confidence band"), "bands")
    m1, m2 = st.columns(2)
    m1.metric("Expected calibration error (ECE)", f"{block['ece']:.3f}" if block["ece"] is not None else "n/a", help="Lower is better. Uses the probability of the predicted answer, 10 equal-width bins.")
    m2.metric("Brier score", f"{block['brier']:.3f}" if block.get("brier") is not None else ("n/a (per question)" if qsel.startswith("all") else "n/a"), help="Lower is better.")
    c1, c2 = st.columns(2)
    with c1:
        if block.get("confidence_histograms"):
            h = block["confidence_histograms"]
            theme.plot(charts.histogram_bars(h["all"], "Confidence histogram (correct vs incorrect)", {"correct": h["correct"], "incorrect": h["incorrect"]}, "Confidence"), "chist")
        else:
            st.caption("Pick a single question to see its confidence histogram.")
    with c2:
        theme.plot(charts.histogram_bars(calib["probability_histogram"], "Every probability the model emitted", xtitle="Probability"), "phist")
    st.caption(calib["note"])

# ------------------------------------------------------------------------ gated routing --
with t_gate:
    st.markdown("Confidence-gated routing: the software acts only when the model is sure enough, otherwise a human looks.")
    rs = metrics.get("routing")
    if rs is None:
        st.info("Routing statistics exist only for Jev runs.")
    else:
        c = st.columns(4)
        c[0].markdown(theme.card("Automated", pct(rs["automation_coverage"]), "of cases"), unsafe_allow_html=True)
        c[1].markdown(theme.card("Specialty acc. (automated)", pct(rs["specialty_accuracy_automated"]), f"escalated: {pct(rs['specialty_accuracy_escalated'])}"), unsafe_allow_html=True)
        c[2].markdown(theme.card("Urgency acc. (automated)", pct(rs["urgency_accuracy_automated"]), f"escalated: {pct(rs['urgency_accuracy_escalated'])}"), unsafe_allow_html=True)
        c[3].markdown(theme.card("High-urgency reaching a human", pct(rs["high_urgency_human_rate"]), f"{rs['high_urgency_reaching_human']} of {rs['high_urgency_reference_cases']} reference URGENT/EMERGENCY"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            theme.plot(charts.route_distribution(rs["route_counts"]), "gate_routes")
        with c2:
            theme.plot(charts.risk_coverage_fig(calib["risk_coverage"]), "gate_rc")
        st.warning(f"Under the ACTIVE policy, {rs['unsafe_automations']} cases were automated although the rule-based reference says "
                   "URGENT/EMERGENCY or human review. See the 'Safety policy' tab for the legacy-vs-revised comparison on a held-out split. "
                   "Thresholds are research-demo values, not clinical thresholds.")

# ------------------------------------------------------------------------ safety policy --
with t_pol:
    st.markdown("Old policy vs the conservative safety gate. Thresholds are chosen on a **calibration** split; the risk score is fit on **train**; "
                "everything below the headline is reported on a **holdout** split that was never used for fitting or selection.")
    if which == "claude":
        st.info("The safety policy applies to Jev decisions. Select the Jev run.")
    else:
        if st.button("Run policy study on this Jev benchmark"):
            with st.spinner("Fitting on train, selecting on calibration, evaluating on holdout..."):
                try:
                    api_client.run_policy_study()
                except api_client.ApiError as exc:
                    st.error(exc.message)
        try:
            study = api_client.policy_study()
        except api_client.ApiError as exc:
            st.error(exc.message)
            study = None
        if not study:
            st.info("No policy study yet. Run a full benchmark (all cases), then click the button above.")
        else:
            prov, ps = study["provenance"], study["per_split"]
            if study["simulated"]:
                st.warning("SIMULATED: the study ran on demo-simulator outputs, not Jev. It shows the method and the trade-off, not Jev's real behaviour.")
            st.caption(f"Splits (seed {prov['split_seed']}): {prov['split_counts']} · tuned on {prov['tuned_on']} · holdout used for selection: {study['holdout_used_for_selection']}")
            h = ps["holdout"]
            L, R = h["legacy"], h["revised"]
            c = st.columns(4)
            c[0].markdown(theme.card("Missed urgent (holdout)", f"{L['missed_urgent']} → {R['missed_urgent']}",
                                     f"of {L['n_urgent_reference']} reference-urgent · revised upper 95% bound {pct(R['missed_urgent_upper95'])}"), unsafe_allow_html=True)
            c[1].markdown(theme.card("Coverage", f"{pct(L['coverage'])} → {pct(R['coverage'])}", "share automated"), unsafe_allow_html=True)
            c[2].markdown(theme.card("Unnecessary human review", f"{L['unnecessary_human_review']} → {R['unnecessary_human_review']}",
                                     f"of {L['n_no_human_needed_reference']} cases needing no human"), unsafe_allow_html=True)
            c[3].markdown(theme.card("Automation accuracy", f"{pct(L['automation_accuracy_specialty'])} → {pct(R['automation_accuracy_specialty'])}",
                                     "specialty, among automated"), unsafe_allow_html=True)
            st.write("")
            st.markdown(f"Zero missed urgent cases in {R['n_urgent_reference']} holdout cases does **not** mean zero risk: with that sample the true miss rate could still be up to about "
                        f"{pct(R['missed_urgent_upper95'])} (95% upper bound)." if R["missed_urgent"] == 0 else
                        f"The revised policy still automated {R['missed_urgent']} reference-urgent holdout case(s): {', '.join(R['missed_urgent_ids'])}.")
            g1, g2 = st.columns(2)
            with g1:
                theme.plot(charts.policy_count_bars(h), "pol_counts")
            with g2:
                theme.plot(charts.policy_rate_bars(h), "pol_rates")
            pc = study["paired_holdout"]
            st.caption(f"Paired bootstrap on the holdout (revised minus legacy): coverage {pc['coverage']['mean_difference_new_minus_old']:+.1%} "
                       f"(95% CI {pc['coverage']['ci95'][0]:+.1%} to {pc['coverage']['ci95'][1]:+.1%}); expected cost per case "
                       f"{pc['cost_per_case']['mean_difference_new_minus_old']:+.2f} (95% CI {pc['cost_per_case']['ci95'][0]:+.2f} to {pc['cost_per_case']['ci95'][1]:+.2f}).")
            st.markdown("##### Every split")
            table = []
            for sp in ("train", "calibration", "holdout", "all"):
                for pol in ("legacy", "revised"):
                    e = ps[sp][pol]
                    table.append({"split": sp + (" (includes tuning data)" if sp == "all" else ""), "policy": pol, "n": e["n"], "coverage": e["coverage"],
                                  "missed urgent": f"{e['missed_urgent']}/{e['n_urgent_reference']}", "false automation": e["false_automation"],
                                  "unnecessary review": e["unnecessary_human_review"], "specialty acc. (auto)": e["automation_accuracy_specialty"],
                                  "cost / case": e["cost_per_case"]})
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            sel = study["selection"]
            c1, c2 = st.columns(2)
            with c1:
                theme.plot(charts.cost_sensitivity_fig(study["cost_sensitivity_holdout"]), "pol_cost")
                st.caption("The revised policy is not better at every cost ratio: when a missed urgent case is only about as bad as a couple of "
                           "unnecessary reviews, the legacy policy's higher automation can win.")
            with c2:
                theme.plot(charts.calibration_frontier_fig(sel["calibration_frontier"], ps["calibration"]["legacy"], ps["calibration"]["revised"], prov["split_counts"]["calibration"]), "pol_frontier")
            st.markdown("##### Selected thresholds vs hard safety limits")
            lim = study["limits"]
            tt = study["tuned_policy"]["thresholds"]
            st.dataframe(pd.DataFrame([{
                "threshold": k, "selected": v, "tuned": k in sel["selected_final"],
                "hard limit": (f"<= {lim['max'][k]}" if k in lim["max"] else f">= {lim['min'][k]}" if k in lim["min"] else "-")} for k, v in tt.items()]),
                hide_index=True, width="stretch")
            st.caption(f"{sel['n_candidates']} candidates searched on the calibration split, {sel['n_feasible']} met the missed-urgent budget "
                       f"({sel['budget_missed_urgent_calibration']}); one-step safety margin applied. Cost weights: {study['costs']}. "
                       "Thresholds are research-demo thresholds, not clinical thresholds. The study evaluates a candidate; it does not change the active policy "
                       "(run scripts/tune_policy.py --write to apply).")
            with st.expander("Definitions"):
                st.json(study["definitions"])

# ------------------------------------------------------------------------ comparison --
with t_cmp:
    cmp_ = results.get("comparison")
    if cmp_ is None:
        st.info("No Jev vs Claude comparison yet. It needs a LIVE Jev run and a Claude run on the same cases "
                "(TYPESAFE_API_KEY + ANTHROPIC_API_KEY, Demo Mode off). Demo Mode never produces a comparison because the simulator is not Jev.")
    else:
        paired = cmp_["paired"]
        theme.plot(charts.comparison_bars(paired), "cmp_bars")
        st.dataframe(pd.DataFrame([{"question": q, "Jev": v["jev_accuracy"], "Claude": v["claude_accuracy"], "difference": v["difference"],
                                    "95% CI": f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]", "verdict": v["verdict"]} for q, v in paired["per_question"].items()]),
                     hide_index=True, width="stretch")
        t = cmp_["table"]
        st.dataframe(pd.DataFrame([
            {"metric": "Schema compliance", "Jev": pct(t["schema_compliance"]["jev"]), "Claude": pct(t["schema_compliance"]["claude"])},
            {"metric": "Invalid outputs", "Jev": t["invalid_outputs"]["jev"], "Claude": t["invalid_outputs"]["claude"]},
            {"metric": "API errors", "Jev": t["api_errors"]["jev"], "Claude": t["api_errors"]["claude"]},
            {"metric": "Median latency (ms)", "Jev": t["latency"]["jev"].get("p50_ms", "not measured"), "Claude": t["latency"]["claude"].get("p50_ms", "not measured")},
            {"metric": "Cost", "Jev": t["cost"]["jev"]["status"], "Claude": t["cost"]["claude"]["status"]},
            {"metric": "Probability distributions", "Jev": t["probability_distributions"]["jev"], "Claude": t["probability_distributions"]["claude"]},
        ]), hide_index=True, width="stretch")
        st.caption(cmp_["caveat"])

# ------------------------------------------------------------------------ export --
with t_exp:
    st.markdown("Per-case results: case ID, timestamp, model, latency, predictions, probabilities, confidences, errors.")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Download CSV", export_service.rows_to_csv(rows), f"dentoroute_benchmark_{which}_{run['mode']}.csv", "text/csv")
    c2.download_button("Download JSON", export_service.benchmark_json(run), f"dentoroute_benchmark_{which}_{run['mode']}.json", "application/json")
    st.caption("From the command line: python scripts/run_benchmark.py --demo")

# ------------------------------------------------------------------------ method --
with t_method:
    doc = DOCS_DIR / "benchmark_methodology.md"
    st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "docs/benchmark_methodology.md not found.")
