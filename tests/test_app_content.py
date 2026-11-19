"""Inhalt der App per AppTest: die angezeigten Zahlen stammen aus der Rechnung bei den Reglern (Schlüsselwort-Falle), das Beispiel folgt dem Schritt-Regler, die Teststärke erst auf Klick."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hoh_constants as C
import hoh_evaluation as ev
import hoh_exact as E

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning] + [i.value for i in at.info] + [s.value for s in at.success])


def _metric(at, label_start):
    return next(m for m in at.metric if m.label.startswith(label_start))


def test_metrics_show_the_values_of_the_analysis_at_the_slider_settings():
    at = _app()
    at.sidebar.slider(key="n_slider").set_value(30)
    at.sidebar.slider(key="k_slider").set_value(2)
    at.sidebar.slider(key="q_slider").set_value(65)
    at.sidebar.slider(key="m_slider").set_value(40)
    at.sidebar.number_input(key="seed_input").set_value(8)
    at.run()
    assert not at.exception
    a = ev.analyse(ev.Settings(n=30, k=2, q_pct=65, m=40, seed=8))
    assert _metric(at, "Wahre Quote").value == "0,65"
    assert _metric(at, "Exakte Erwartung").value == ev.fmt(a.exact.mean_est)
    assert _metric(at, "Mittel der").value == ev.fmt(a.mean_est) and _metric(at, "Mittel der").label == f"Mittel der {a.n_defined} Folgenschätzer"
    assert _metric(at, "Gepoolter").value == ev.fmt(a.pooled)
    assert ev.verdict_main(a)[1] in _texts(at)


def test_every_preset_loads_and_shows_its_exact_expectation():
    for name, preset in C.PRESETS.items():
        at = _app()
        at.sidebar.button(key=f"preset_{name}").click().run()
        assert not at.exception, name
        a = ev.analyse(ev.Settings(**preset))
        assert _metric(at, "Exakte Erwartung").value == ev.fmt(a.exact.mean_est), name
        assert at.sidebar.slider(key="n_slider").value == preset["n"] and at.sidebar.slider(key="q_slider").value == preset["q_pct"]


def test_exact_probe_table_agrees_and_uses_at_most_sixteen_throws():
    at = _app()
    # Standard n = 20: die Aufzählung deckelt auf 16 Würfe und sagt es
    caps = [c.value for c in at.caption if c.value.startswith("Aufzählung aller")]
    assert caps and f"2^{C.ENUM_MAX} = 65.536 Folgen der Länge n = {C.ENUM_MAX}" in caps[0]
    assert any("stimmen in allen 5 Zeilen" in s.value for s in at.success)
    at.sidebar.slider(key="n_slider").set_value(10)
    at.run()
    assert any(c.value.startswith("Aufzählung aller 2^10 = 1.024 Folgen der Länge n = 10") for c in at.caption)


def test_example_step_follows_the_slider_the_button_and_the_length():
    at = _app()
    step = next(s for s in at.slider if s.key == "example_step")
    assert step.value == 12 and step.max == 20
    at.button(key="example_next").click().run()
    assert next(s for s in at.slider if s.key == "example_step").value == 13
    next(s for s in at.slider if s.key == "example_step").set_value(5).run()
    a = ev.analyse(ev.Settings())
    assert ev.verdict_example(a, 5)[1] in _texts(at)
    at.sidebar.slider(key="n_slider").set_value(8).run()                  # kürzere Länge als der gewählte Schritt: der Schritt rastet ein
    assert not at.exception
    step = next(s for s in at.slider if s.key == "example_step")
    assert step.max == 8 and step.value <= 8
    for _ in range(12):                                                    # über das Ende hinaus weiter schalten: bleibt bei n
        at.button(key="example_next").click().run()
    assert next(s for s in at.slider if s.key == "example_step").value == 8


def test_power_experiment_runs_only_on_click_and_resets_when_the_sliders_change():
    at = _app()
    assert any("Noch nicht berechnet" in i.value for i in at.info)
    assert "power_chart" not in [e.key for e in at.get("plotly_chart")]
    at.sidebar.slider(key="n_slider").set_value(30)
    at.sidebar.slider(key="m_slider").set_value(5)
    at.run()
    at.button(key="power_run").click().run()
    assert not at.exception
    rows = at.session_state["power_result"]["rows"]
    assert [r["q_pct"] for r in rows] == list(C.Q_GRID_PCT)
    assert any(t.value.startswith("n = 30, k = 3, 5 Folgen, je Wert 100 Läufe") for t in at.info)
    at.sidebar.slider(key="k_slider").set_value(2).run()
    assert any("neu berechnen" in i.value for i in at.info)


def test_the_caps_of_the_power_run_are_named():
    at = _app()
    at.sidebar.slider(key="n_slider").set_value(150)
    at.sidebar.slider(key="m_slider").set_value(60)
    at.run()
    assert any(f"Läuft für n = {C.POWER_N_CAP}, k = 3, {C.POWER_M_CAP} Folgen" in c.value for c in at.caption)


def test_page_has_the_expanders_and_sections_by_title():
    at = _app()
    labels = [e.label for e in at.expander]
    for needed in ("So funktioniert das Maß", "📋 Verfahren und Begriffe: wer macht was?", "⚠️ Annahmen, Grenzen und bekannte Probleme", "📐 Mathematische Formulierung"):
        assert needed in labels
    subs = [s.value for s in at.subheader]
    assert any(s.startswith("🚧 Wo die Annahmen enden") for s in subs) and len([s for s in subs if s.startswith("🔬")]) == 5
    table = next(m.value for m in at.markdown if "| Annahme |" in m.value)
    assert "`siegtabelle-demo`" in table and "`rangintervalle-demo`" in table and "Wer setzt an" in table


def test_every_method_of_the_table_is_named_on_the_page():
    at = _app()
    names = [r["Name"] for r in ev.method_rows(ev.analyse(ev.Settings()))]
    tables = [df.value for df in at.dataframe]
    method_table = next(t for t in tables if "Name" in t.columns)
    assert list(method_table["Name"]) == names and all(method_table["Was"]) and all(method_table["Wert"])


def test_no_literature_claim_beyond_not_read_in_full_and_no_basketball_claim():
    text = _texts(_app())
    assert "nicht im Volltext gelesen" in text
    assert "Basketball" in text and "nicht auf Basketball" in text


def test_exact_table_used_by_the_page_is_the_demo_recursion():
    a = ev.analyse(ev.Settings())
    assert a.exact.mean_est == pytest.approx(E.exact_table(20, 3, 0.5)[20].mean_est)
