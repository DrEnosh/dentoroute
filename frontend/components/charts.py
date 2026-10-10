"""Plotly figures. All take plain dicts (the JSON the API returns) and return a go.Figure."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from frontend.components.theme import BAND_COLORS, ROUTE_COLORS, ROUTE_TEXT, TEAL, URGENCY_COLORS

_LAYOUT = dict(margin=dict(l=10, r=10, t=45, b=10), template="plotly_white", font=dict(size=13))


def _fig(title: str, height: int = 320) -> go.Figure:
    f = go.Figure()
    f.update_layout(title=dict(text=title, font=dict(size=15)), height=height, **_LAYOUT)
    return f


def probability_bars(probs: dict[str, float], title: str, colors: dict[str, str] | None = None, horizontal: bool = False,
                     top_n: int | None = None, height: int = 320) -> go.Figure:
    items = list(probs.items())
    if horizontal:
        items = sorted(items, key=lambda kv: kv[1])
        if top_n:
            items = items[-top_n:]
    keys, vals = [k for k, _ in items], [v for _, v in items]
    marker = [colors.get(k, TEAL) for k in keys] if colors else TEAL
    f = _fig(title, height)
    f.add_bar(x=vals if horizontal else keys, y=keys if horizontal else vals, orientation="h" if horizontal else "v",
              marker_color=marker, text=[f"{v:.0%}" for v in vals], textposition="outside", cliponaxis=False)
    rng = [0, 1.15]
    f.update_layout(xaxis=dict(range=rng, tickformat=".0%") if horizontal else {}, yaxis={} if horizontal else dict(range=rng, tickformat=".0%"))
    return f


def urgency_bars(decision: dict) -> go.Figure:
    return probability_bars(decision["urgency"]["probabilities"], "Urgency probability (Jev Score)", URGENCY_COLORS)


def specialty_bars(decision: dict) -> go.Figure:
    labels = {k: k.replace("_", " ") for k in decision["specialty"]["probabilities"]}
    probs = {labels[k]: v for k, v in decision["specialty"]["probabilities"].items()}
    return probability_bars(probs, "Specialty distribution (Jev Choice)", horizontal=True, top_n=10, height=380)


def gauge(value: float, title: str, steps: tuple[float, float] = (0.5, 0.8), invert: bool = False, threshold: float | None = None,
          height: int = 230) -> go.Figure:
    """0-100% gauge. `invert` colours high values as 'bad' (probabilities of concern) instead of 'good' (confidence)."""
    good, mid, bad = "#bbf7d0", "#fef08a", "#fecaca"
    cols = [bad, mid, good] if not invert else [good, mid, bad]
    f = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100, number=dict(suffix="%", valueformat=".0f"),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#0f172a", thickness=0.25),
                   steps=[dict(range=[0, steps[0] * 100], color=cols[0]), dict(range=[steps[0] * 100, steps[1] * 100], color=cols[1]),
                          dict(range=[steps[1] * 100, 100], color=cols[2])],
                   threshold=dict(line=dict(color="#dc2626", width=3), value=threshold * 100) if threshold is not None else None),
    ))
    f.update_layout(height=height, margin=dict(l=30, r=30, t=50, b=5), template="plotly_white")
    return f


def routing_path_dot(routing: dict) -> str:
    """Graphviz DOT for the policy ladder with the rule that fired highlighted."""
    fired = routing["triggered_rule"][:2]
    esc, urg, hum, auto = (ROUTE_COLORS["escalate_to_human"], ROUTE_COLORS["urgent_review"],
                           ROUTE_COLORS["human_review"], ROUTE_COLORS["automated_specialty_route"])
    if routing.get("policy_name") == "legacy":
        ladder = [("R1", "R1  human-review\\nprobability high?", "Escalate to human", esc),
                  ("R2", "R2  urgency / same-day /\\ninfection high?", "Urgent human review", urg),
                  ("R3", "R3  specialty confident\\nand urgency not split?", "Automated specialty route", auto)]
        tail = ("R4", "R4  low confidence\\n(do not force an answer)", "Human review", hum)
    else:
        ladder = [("R1", "R1  human-review\\nprobability high?", "Escalate to human", esc),
                  ("R2", "R2  infection-concern\\nprobability high?", "Urgent human review", urg),
                  ("R3", "R3  urgent-risk pattern?\\n(urgency, same-day, learned risk)", "Urgent human review", urg),
                  ("R4", "R4  any low confidence /\\nuncertain safety answer?", "Human review", hum),
                  ("R5", "R5  human-only specialty?", "Human review", hum)]
        tail = ("R6", "R6  all gates passed", "Automated specialty route", auto)
    lines = ['digraph G {', ' rankdir=LR; nodesep=0.3; ranksep=0.35; bgcolor="transparent"; fontname="Helvetica";',
             ' node [fontname="Helvetica", fontsize=11]; edge [fontname="Helvetica", fontsize=10, color="#94a3b8"];',
             ' start [label="Jev decisions\\n(R0: invalid output -> human)", shape=oval, style=filled, fillcolor="#e2e8f0", color="#cbd5e1"];']
    def box(rid: str, label: str) -> str:
        on = rid == fired
        return (f' {rid} [label="{label}", shape=box, style="rounded,filled", fillcolor="{"#0f766e" if on else "#f1f5f9"}", '
                f'fontcolor="{"white" if on else "#334155"}", color="{"#0f766e" if on else "#cbd5e1"}"];')
    def end(rid: str, text: str, col: str) -> str:
        return f' {rid}_end [label="{text}", shape=box, style="filled", fillcolor="{col}", fontcolor="white", color="{col}"];'
    nodes = ladder + [tail]
    for rid, label, etext, col in nodes:
        lines += [box(rid, label), end(rid, etext, col)]
    lines.append(" start -> %s;" % nodes[0][0])
    for i, (rid, _, _, _) in enumerate(nodes[:-1]):
        lines.append(f' {rid} -> {rid}_end [label="yes"]; {rid} -> {nodes[i + 1][0]} [label="no"];')
    lines.append(f" {nodes[-1][0]} -> {nodes[-1][0]}_end;")
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------- benchmark ------
def accuracy_bars(per_question: dict[str, dict], title: str = "Accuracy vs rule-based reference") -> go.Figure:
    qs = list(per_question)
    vals = [per_question[q]["accuracy"] or 0 for q in qs]
    f = _fig(title, 340)
    f.add_bar(x=[q.replace("_", " ") for q in qs], y=vals, marker_color=TEAL, text=[f"{v:.0%}" for v in vals], textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(range=[0, 1.1], tickformat=".0%"))
    return f


def comparison_bars(paired: dict) -> go.Figure:
    pq = paired["per_question"]
    qs = list(pq)
    a, b = paired["a"], paired["b"]
    f = _fig(f"{a.title()} vs {b.title()}: accuracy on {paired['n_paired']} paired cases", 380)
    f.add_bar(name=a.title(), x=[q.replace("_", " ") for q in qs], y=[pq[q][f"{a}_accuracy"] for q in qs], marker_color=TEAL)
    f.add_bar(name=b.title(), x=[q.replace("_", " ") for q in qs], y=[pq[q][f"{b}_accuracy"] for q in qs], marker_color="#94a3b8")
    f.update_layout(barmode="group", yaxis=dict(range=[0, 1.05], tickformat=".0%"), legend=dict(orientation="h", y=1.12))
    return f


def confusion_heatmap(cm: list[list[int]], labels: list[str], title: str, normalize: bool = False) -> go.Figure:
    z = [[float(x) for x in row] for row in cm]
    text = [[str(int(x)) for x in row] for row in cm]
    if normalize:
        z = [[(x / sum(row)) if sum(row) else 0 for x in row] for row in z]
    pretty = [l.replace("_", " ") for l in labels]
    f = _fig(title, 420 if len(labels) > 5 else 340)
    f.add_trace(go.Heatmap(z=z, x=pretty, y=pretty, text=text, texttemplate="%{text}", colorscale="Teal", showscale=False,
                           hovertemplate="reference %{y}<br>predicted %{x}<br>count %{text}<extra></extra>"))
    f.update_layout(xaxis=dict(title="Predicted", side="bottom"), yaxis=dict(title="Reference (rule-based)", autorange="reversed"))
    return f


def reliability_diagram(bins: list[dict], title: str, ece: float | None = None) -> go.Figure:
    pts = [b for b in bins if b["n"]]
    f = _fig(title + (f"  (ECE {ece:.3f})" if ece is not None else ""), 380)
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8"), name="Perfect calibration"))
    if pts:
        f.add_trace(go.Scatter(
            x=[b["mean_confidence"] for b in pts], y=[b["accuracy"] for b in pts], mode="lines+markers", name="Observed",
            line=dict(color=TEAL), marker=dict(color=TEAL, size=[max(7, min(30, 6 + b["n"] ** 0.5 * 2)) for b in pts]),
            text=[f"n={b['n']}" for b in pts], hovertemplate="confidence %{x:.2f}<br>accuracy %{y:.2f}<br>%{text}<extra></extra>"))
    f.update_layout(xaxis=dict(title="Stated probability of the predicted answer", range=[0, 1.02]),
                    yaxis=dict(title="Observed accuracy", range=[0, 1.02]), legend=dict(orientation="h", y=-0.25))
    return f


def histogram_bars(h: dict, title: str, series: dict[str, dict] | None = None, xtitle: str = "Value") -> go.Figure:
    mids = [(h["edges"][i] + h["edges"][i + 1]) / 2 for i in range(len(h["counts"]))]
    f = _fig(title, 300)
    if series:
        colors = {"correct": "#16a34a", "incorrect": "#dc2626"}
        for name, hh in series.items():
            f.add_bar(x=mids, y=hh["counts"], name=name, marker_color=colors.get(name, TEAL))
        f.update_layout(barmode="stack", legend=dict(orientation="h", y=1.15))
    else:
        f.add_bar(x=mids, y=h["counts"], marker_color=TEAL)
    f.update_layout(xaxis=dict(title=xtitle, range=[0, 1]), yaxis=dict(title="Cases"))
    return f


def band_accuracy_bars(bands: dict[str, dict], title: str) -> go.Figure:
    order = ["low", "medium", "high"]
    f = _fig(title, 300)
    f.add_bar(x=[f"{b} (n={bands[b]['n']})" for b in order], y=[bands[b]["accuracy"] or 0 for b in order],
              marker_color=[BAND_COLORS[b] for b in order],
              text=[f"{bands[b]['accuracy']:.0%}" if bands[b]["accuracy"] is not None else "no cases" for b in order],
              textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Accuracy"))
    return f


def risk_coverage_fig(rc: list[dict], title: str = "Confidence-gated routing: accuracy vs automation") -> go.Figure:
    pts = [r for r in rc if r["joint_accuracy"] is not None]
    f = _fig(title, 360)
    f.add_trace(go.Scatter(x=[r["threshold"] for r in pts], y=[r["joint_accuracy"] for r in pts], mode="lines+markers",
                           name="Accuracy of automated cases (urgency AND specialty correct)", line=dict(color=TEAL)))
    f.add_trace(go.Scatter(x=[r["threshold"] for r in rc], y=[r["coverage"] for r in rc], mode="lines", name="Share automated (coverage)",
                           line=dict(color="#6366f1", dash="dot")))
    f.update_layout(xaxis=dict(title="Minimum confidence required to automate"), yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                    legend=dict(orientation="h", y=-0.3))
    return f


def route_distribution(counts: dict[str, int], title: str = "Where cases were routed") -> go.Figure:
    keys = [k for k in ROUTE_TEXT if k in counts]
    f = _fig(title, 300)
    f.add_bar(x=[ROUTE_TEXT[k] for k in keys], y=[counts[k] for k in keys], marker_color=[ROUTE_COLORS[k] for k in keys],
              text=[counts[k] for k in keys], textposition="outside", cliponaxis=False)
    f.update_layout(yaxis=dict(title="Cases"))
    return f


def category_bars(values: dict[str, float], title: str, colors: dict[str, str] | None = None, height: int = 300) -> go.Figure:
    f = _fig(title, height)
    f.add_bar(x=list(values), y=list(values.values()), marker_color=[colors.get(k, TEAL) for k in values] if colors else TEAL)
    return f


def latency_histogram(latencies_ms: list[float], title: str = "Latency per request (ms)") -> go.Figure:
    f = _fig(title, 300)
    f.add_trace(go.Histogram(x=latencies_ms, marker_color=TEAL, nbinsx=25))
    f.update_layout(xaxis=dict(title="ms"), yaxis=dict(title="Requests"))
    return f


# ------------------------------------------------------------- safety policy ------
POLICY_COLORS = {"legacy": "#94a3b8", "revised": "#0f766e"}


def policy_count_bars(h: dict, title: str = "Errors that matter (holdout)") -> go.Figure:
    labels = [("missed_urgent", "Missed urgent\n(automated)"), ("false_automation", "False automation\n(any human-needed)"),
              ("unnecessary_human_review", "Unnecessary\nhuman review")]
    f = _fig(title, 340)
    for pol in ("legacy", "revised"):
        vals = [h[pol][k] for k, _ in labels]
        f.add_bar(name=pol.title(), x=[l.replace("\n", " ") for _, l in labels], y=vals, marker_color=POLICY_COLORS[pol],
                  text=vals, textposition="outside", cliponaxis=False)
    f.update_layout(barmode="group", yaxis=dict(title="Cases"), legend=dict(orientation="h", y=1.15))
    return f


def policy_rate_bars(h: dict, title: str = "Coverage and automation accuracy (holdout)") -> go.Figure:
    labels = [("coverage", "Coverage"), ("automation_accuracy_specialty", "Specialty accuracy\n(automated)"),
              ("automation_accuracy_urgency", "Urgency accuracy\n(automated)"), ("unnecessary_review_rate", "Unnecessary review\nrate")]
    f = _fig(title, 340)
    for pol in ("legacy", "revised"):
        vals = [h[pol][k] or 0 for k, _ in labels]
        f.add_bar(name=pol.title(), x=[l.replace("\n", " ") for _, l in labels], y=vals, marker_color=POLICY_COLORS[pol],
                  text=[f"{v:.0%}" for v in vals], textposition="outside", cliponaxis=False)
    f.update_layout(barmode="group", yaxis=dict(range=[0, 1.15], tickformat=".0%"), legend=dict(orientation="h", y=1.15))
    return f


def cost_sensitivity_fig(sens: list[dict], title: str = "Expected cost per case vs how bad a missed urgent case is (holdout)") -> go.Figure:
    f = _fig(title, 340)
    x = [s["missed_to_unnecessary_cost_ratio"] for s in sens]
    f.add_trace(go.Scatter(x=x, y=[s["legacy_cost_per_case"] for s in sens], name="Legacy", mode="lines+markers", line=dict(color=POLICY_COLORS["legacy"])))
    f.add_trace(go.Scatter(x=x, y=[s["revised_cost_per_case"] for s in sens], name="Revised", mode="lines+markers", line=dict(color=POLICY_COLORS["revised"])))
    f.update_layout(xaxis=dict(title="Cost of a missed urgent case, in units of one unnecessary review", type="log"),
                    yaxis=dict(title="Cost per case"), legend=dict(orientation="h", y=1.15))
    return f


def calibration_frontier_fig(frontier: list[dict], legacy_cal: dict, revised_cal: dict, n_cal: int,
                             title: str = "Calibration split: coverage vs missed urgent cases") -> go.Figure:
    f = _fig(title, 340)
    f.add_trace(go.Scatter(x=[p["missed_urgent"] for p in frontier], y=[p["coverage"] for p in frontier], mode="lines+markers",
                           name="Best achievable (searched grid)", line=dict(color="#6366f1", shape="hv")))
    for name, e in (("Legacy", legacy_cal), ("Revised (after safety margin)", revised_cal)):
        f.add_trace(go.Scatter(x=[e["missed_urgent"]], y=[e["coverage"]], mode="markers", name=name,
                               marker=dict(size=14, color=POLICY_COLORS["legacy" if name == "Legacy" else "revised"])))
    f.update_layout(xaxis=dict(title="Missed urgent cases (automated)", dtick=1), yaxis=dict(title="Coverage", range=[0, 1], tickformat=".0%"),
                    legend=dict(orientation="h", y=-0.3))
    return f
