"""Streamlit smoke tests: every page renders, and the main flows work end to end in Demo Mode."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["case_simulator", "decision_dashboard", "benchmark_lab", "case_history", "architecture", "about_safety"]
APP = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
SAFETY = "Research prototype. Not for clinical diagnosis or treatment."


@pytest.fixture
def app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [e.value for e in at.error]
    parts += [e.value for e in at.info] + [w.value for w in at.warning] + [t.value for t in at.title]
    return "\n".join(str(p) for p in parts)


def click(at: AppTest, label: str) -> AppTest:
    return next(b for b in at.button if b.label == label).click().run()


def test_home_renders_with_safety_and_demo_banners(app):
    assert not app.exception
    body = texts(app)
    assert SAFETY in body
    assert "Demo Mode — using simulated Jev responses." in body
    assert "From dental cases to structured decisions." in body


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(app, page):
    app.switch_page(f"pages/{page}.py").run()
    assert not app.exception, [e.value for e in app.exception]
    assert SAFETY in texts(app)


def test_simulator_evaluates_a_demo_case_and_dashboard_shows_it(app):
    app.switch_page("pages/case_simulator.py").run()
    click(app, "Evaluate with DentoRoute")
    assert not app.exception, [e.value for e in app.exception]
    assert "Simulated Jev response" in texts(app)
    assert "Result for DEMO-01" in [s.value for s in app.subheader]
    assert "Jev's seven answers" in [s.value for s in app.subheader]
    app.switch_page("pages/decision_dashboard.py").run()
    assert not app.exception
    assert len(app.tabs) == 2


def test_ambiguous_demo_case_is_not_automated(app):
    app.switch_page("pages/case_simulator.py").run()
    sel = next(s for s in app.selectbox if s.label == "Featured case")
    sel.select(next(o for o in sel.options if o.startswith("DEMO-10"))).run()
    click(app, "Evaluate with DentoRoute")
    res = app.session_state["current"]
    assert res["routing"]["route"] in ("human_review", "escalate_to_human") and res["routing"]["requires_human"]


def test_manual_case_builder_runs(app):
    app.switch_page("pages/case_simulator.py").run()
    app.radio[0].set_value("Build manually").run()
    app.checkbox(key="flag_fever").check().run()
    app.checkbox(key="flag_swelling").check().run()
    click(app, "Evaluate with DentoRoute")
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["current"]["decision"]["urgency"]["label"]


def test_research_mode_shows_provider_details(app):
    app.switch_page("pages/case_simulator.py").run()
    app.session_state["research_mode"] = True  # AppTest drops sidebar widget keys on switch_page
    click(app, "Evaluate with DentoRoute")
    assert not app.exception
    assert any("Research Mode" in e.label for e in app.expander)


def test_benchmark_lab_runs_and_shows_simulated_warning(app):
    app.switch_page("pages/benchmark_lab.py").run()
    assert "No benchmark has been run yet" in texts(app)
    app.select_slider[0].set_value(25).run()
    click(app, "Run benchmark")
    assert not app.exception, [e.value for e in app.exception]
    body = texts(app)
    assert "SIMULATED RUN" in body and "NOT Jev" in body
    assert "No Jev vs Claude comparison yet" in body  # never compared in demo mode
    assert any("Does higher confidence mean higher accuracy?" in m.value for m in app.markdown)


def test_history_lists_evaluated_cases(app):
    app.switch_page("pages/case_simulator.py").run()
    click(app, "Evaluate with DentoRoute")
    app.switch_page("pages/case_history.py").run()
    assert not app.exception
    assert len(app.dataframe) >= 1
    assert "DEMO-01" in app.dataframe[0].value.to_string()


def test_live_mode_without_key_shows_friendly_error_not_traceback(app, monkeypatch):
    from backend.utils import config
    from frontend import api_client

    app.switch_page("pages/case_simulator.py").run()
    monkeypatch.setenv("DENTOROUTE_DEMO_MODE", "false")
    config.reset_settings_cache()
    api_client._cache.update(transport=None, kind=None, checked=0.0)
    app.session_state["force_demo"] = False
    click(app, "Evaluate with DentoRoute")
    assert not app.exception
    assert any("TYPESAFE_API_KEY" in e.value for e in app.error)


def test_safety_policy_tab_runs_the_study_and_labels_it_simulated(app):
    app.switch_page("pages/benchmark_lab.py").run()
    click(app, "Run benchmark")                       # demo mode default: all cases
    assert not app.exception, [e.value for e in app.exception]
    body = texts(app)
    assert "No policy study yet" in body
    click(app, "Run policy study on this Jev benchmark")
    assert not app.exception, [e.value for e in app.exception]
    body = texts(app)
    assert "SIMULATED: the study ran on demo-simulator outputs" in body
    assert "holdout used for selection: False" in body
    assert "does **not** mean zero risk" in body or "still automated" in body
    assert "not better at every cost ratio" in body


def test_policy_study_without_a_full_benchmark_shows_a_friendly_error(app):
    app.switch_page("pages/benchmark_lab.py").run()
    app.select_slider[0].set_value(25).run()
    click(app, "Run benchmark")
    click(app, "Run policy study on this Jev benchmark")
    assert not app.exception
    assert any("all three splits" in e.value or "benchmark" in e.value.lower() for e in app.error)
