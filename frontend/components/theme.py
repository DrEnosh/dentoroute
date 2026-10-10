"""Shared look & feel, banners and small HTML helpers (all text content is escaped)."""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from backend.utils.config import DEMO_BANNER, SAFETY_BANNER, TAGLINE  # noqa: E402
from frontend import api_client  # noqa: E402

TEAL = "#0f766e"
URGENCY_COLORS = {"ROUTINE": "#16a34a", "PRIORITY": "#ca8a04", "URGENT": "#ea580c", "EMERGENCY": "#dc2626"}
ROUTE_COLORS = {
    "automated_specialty_route": "#0f766e", "urgent_review": "#ea580c",
    "human_review": "#6366f1", "escalate_to_human": "#dc2626",
}
ROUTE_TEXT = {
    "automated_specialty_route": "Automated specialty route", "urgent_review": "Urgent human review",
    "human_review": "Human review (low confidence)", "escalate_to_human": "Escalated to human",
}
BAND_COLORS = {"high": "#16a34a", "medium": "#ca8a04", "low": "#dc2626"}

_CSS = """
<style>
.block-container {padding-top: 3.6rem; max-width: 1250px;}
.dr-safety {background:#fef3c7; border:1px solid #f59e0b; color:#78350f; padding:.45rem .8rem; border-radius:8px;
            font-size:.88rem; margin-bottom:.5rem;}
.dr-demo {background:#e0f2fe; border:1px solid #38bdf8; color:#0c4a6e; padding:.45rem .8rem; border-radius:8px;
          font-size:.88rem; margin-bottom:.8rem;}
.dr-card {border:1px solid #e2e8f0; border-radius:12px; padding:1rem 1.1rem; background:#fff; min-height:150px;}
.dr-card h4 {margin:0 0 .3rem 0; font-size:.8rem; letter-spacing:.04em; text-transform:uppercase; color:#64748b;}
.dr-big {font-size:1.45rem; font-weight:700; line-height:1.15;}
.dr-sub {color:#64748b; font-size:.85rem; margin-top:.2rem;}
.dr-chip {display:inline-block; padding:.12rem .6rem; border-radius:999px; color:#fff; font-size:.78rem; font-weight:600;}
.dr-step {border-left:4px solid #0f766e; padding:.4rem .8rem; background:#f8fafc; border-radius:0 8px 8px 0; margin-bottom:.4rem;}
.dr-tag {font-size:.8rem; color:#64748b;}
</style>
"""


def esc(x: object) -> str:
    return html.escape(str(x))


def chip(text: str, color: str) -> str:
    return f'<span class="dr-chip" style="background:{color}">{esc(text)}</span>'


def card(title: str, big: str, sub: str = "", color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="dr-card"><h4>{esc(title)}</h4><div class="dr-big"{style}>{esc(big)}</div>'
            f'<div class="dr-sub">{esc(sub)}</div></div>')


def research_mode() -> bool:
    return bool(st.session_state.get("research_mode", False))


def demo_override() -> bool | None:
    """True forces Demo Mode for requests; None lets the backend decide."""
    return True if st.session_state.get("force_demo") else None


@st.cache_data(ttl=5, show_spinner=False)
def _health() -> dict:
    return api_client.health()


def current_mode() -> str:
    try:
        backend_mode = _health()["mode"]
    except Exception:  # noqa: BLE001
        backend_mode = "demo"
    return "demo" if (demo_override() or backend_mode == "demo") else "live"


def page_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="dr-safety"><b>{esc(SAFETY_BANNER)}</b> Synthetic data only.</div>', unsafe_allow_html=True)
    if current_mode() == "demo":
        st.markdown(f'<div class="dr-demo"><b>{esc(DEMO_BANNER)}</b> Results below come from a deterministic simulator, not from the real model.</div>',
                    unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def sidebar_controls() -> None:
    """Rendered once, from app.py."""
    st.sidebar.markdown(f"### DentoRoute\n_{TAGLINE}_")
    try:
        h = _health()
    except Exception:  # noqa: BLE001
        h = {"mode": "demo", "jev_key_configured": False, "claude_key_configured": False}
    backend_demo = h["mode"] == "demo"
    st.sidebar.toggle("Research Mode", key="research_mode",
                      help="Show raw Jev JSON, probabilities, thresholds, latency, cache status and rubric details.")
    st.sidebar.checkbox("Force Demo Mode", key="force_demo", value=backend_demo, disabled=backend_demo,
                        help="Use simulated Jev responses. Always on when no TYPESAFE_API_KEY is set." )
    st.sidebar.caption(f"Mode: **{current_mode().upper()}**  \nJev key: {'set' if h['jev_key_configured'] else 'not set'}  \n"
                       f"Claude key: {'set' if h['claude_key_configured'] else 'not set'}  \nBackend: {api_client.backend_kind()}")
    st.sidebar.divider()
    st.sidebar.caption("Research prototype. Not for clinical diagnosis or treatment.")


def urgency_chip(label: str) -> str:
    return chip(label, URGENCY_COLORS.get(label, "#64748b"))


def plot(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def nav_link(page: str, label: str, icon: str, container=None) -> None:
    """page_link only works inside st.navigation; skip quietly when a page is run standalone."""
    from streamlit.errors import StreamlitPageNotFoundError

    try:
        (container or st).page_link(page, label=label, icon=icon)
    except StreamlitPageNotFoundError:
        pass
