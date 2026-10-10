"""Exports: JSON / Markdown case reports and CSV / JSON benchmark results."""
from __future__ import annotations

import csv
import io
import json
from typing import Any

from backend.utils.config import DEMO_BANNER, SAFETY_BANNER, TAGLINE


def case_report_json(response: dict[str, Any], include_raw: bool = False) -> str:
    data = dict(response)
    dec = data.get("decision")
    if dec and not include_raw:
        dec = {k: v for k, v in dec.items() if k != "raw_response"}
        data["decision"] = dec
    data["safety_notice"] = SAFETY_BANNER
    return json.dumps(data, indent=2, default=str)


def _pct(x: float) -> str:
    return f"{x:.0%}"


def case_report_markdown(response: dict[str, Any]) -> str:
    d, r, e = response.get("decision"), response.get("routing"), response.get("explanation")
    lines = [f"# DentoRoute case report: {response.get('patient_id')}", "", f"_{TAGLINE}_", "", f"> {SAFETY_BANNER}"]
    if response.get("mode") == "demo":
        lines += ["", f"> {DEMO_BANNER}"]
    if response.get("error"):
        lines += ["", "## Error", response["error"]]
        return "\n".join(lines) + "\n"
    if d:
        lines += ["", "## Decision-layer output", "",
                  f"- Source: {d['source']} ({d['mode']}), model `{d['model']}`",
                  f"- Urgency: **{d['urgency']['label']}** (score {d['urgency']['score']:.2f}/3, confidence {_pct(d['urgency']['confidence'])})",
                  f"- Specialty: **{d['specialty']['label']}** (confidence {_pct(d['specialty']['confidence'])})",
                  f"- Infection concern probability: {_pct(d['infection_concern']['probability'])}",
                  f"- Same-day review probability: {_pct(d['same_day_review']['probability'])}",
                  f"- Human review probability: {_pct(d['human_review']['probability'])}",
                  f"- Complexity: {d['complexity']['label']}; referral priority: {d['referral_priority']['label']}"]
    if r:
        lines += ["", "## Routing policy", "", f"- Route: **{r['workflow_label']}** (`{r['route']}`, rule `{r['triggered_rule']}`)",
                  f"- Confidence band: {r['confidence_band']}; human required: {r['requires_human']}"]
        lines += [f"- {x}" for x in r["reasons"]]
    if e:
        lines += ["", f"## Explanation ({e['source']})", "", e["summary"], "", e["engine_findings"], "", e["why_route"],
                  "", f"**Human review:** {e['human_review_note']}", "", f"**Uncertainty:** {e['uncertainty']}"]
        if e.get("warnings"):
            lines += ["", *[f"_Note: {w}_" for w in e["warnings"]]]
        lines += ["", f"_{e['disclaimer']}_"]
    ref = response.get("reference")
    if ref:
        lines += ["", "## Rule-based reference labels (educational synthetic rubric, not clinical truth)", ""]
        lines += [f"- {k}: {v}" for k, v in ref.items() if k != "fired_rules"]
    return "\n".join(lines) + "\n"


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    flat = [_flatten(r) for r in rows]
    cols: list[str] = []
    for r in flat:
        for k in r:
            if k not in cols:
                cols.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(flat)
    return buf.getvalue()


def _flatten(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def benchmark_json(run: dict[str, Any]) -> str:
    return json.dumps(run, indent=2, default=str)
