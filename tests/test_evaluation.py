"""Auswertung: Messreihen (Struktur, Zweig-Tests gegen tote Varianten), Analyse bei den Reglern, Hilfsfunktionen."""

import pytest

import hoh_constants as C
import hoh_evaluation as ev
import hoh_exact as E


def test_analysis_uses_the_settings_by_keyword_and_the_exact_stats_of_the_same_settings():
    s = ev.Settings(n=18, k=2, q_pct=60, m=15, seed=9)
    a = ev.analyse(s)
    assert a.settings == s and a.x.shape == (15, 18) and a.truth == pytest.approx(0.6)
    assert a.exact.mean_est == pytest.approx(E.exact_table(18, 2, 0.6)[18].mean_est)
    assert a.n_defined == int((a.O > 0).sum()) and a.positions == tuple(t for t in range(2, 18) if a.x[0][t - 2] and a.x[0][t - 1])


def test_settings_default_equals_the_first_preset():
    assert ev.Settings(**C.PRESETS["Faire Münze, 20 Würfe"]) == ev.Settings()
    assert ev.Settings(q_pct=65).q == pytest.approx(0.65)


def test_sweep_length_rows_are_complete_and_the_band_orders_correctly():
    rows = ev.sweep_length(3, 50)
    assert [r["n"] for r in rows] == list(C.N_SIM_GRID)
    for r in rows:
        assert 0 < r["exact"] < 0.5 and r["low"] <= r["sim"] <= r["high"] and 0 < r["p_defined"] <= 1
    assert [r["p_defined"] for r in rows] == sorted(r["p_defined"] for r in rows)


def test_sweep_length_variants_differ_branch_test():
    """Zweig-Test (Nullspalte = Bug): k und q ändern die Reihe wirklich; es gibt keine Spalte mit lauter gleichen Werten."""
    grid = (10, 20, 50)
    base = ev.sweep_length(3, 50, grid=grid, per_seed=500)
    for other in (ev.sweep_length(1, 50, grid=grid, per_seed=500), ev.sweep_length(3, 65, grid=grid, per_seed=500)):
        assert all(abs(a["exact"] - b["exact"]) > 0.02 for a, b in zip(base, other))
        assert all(abs(a["sim"] - b["sim"]) > 0.01 for a, b in zip(base, other))
    assert len({round(r["sim"], 6) for r in base}) == 3


def test_exact_curve_covers_every_length_and_agrees_with_the_table():
    curve = ev.exact_curve(2, 50, n_max=60)
    assert [n for n, _ in curve] == list(range(3, 61))
    assert dict(curve)[40] == pytest.approx(E.exact_table(40, 2, 0.5)[40].mean_est)


def test_sweep_k_has_one_row_per_k_and_falls_with_k_at_the_measured_lengths():
    for n in (20, 50, 100):
        rows = ev.sweep_k(n, 50)
        assert [r["k"] for r in rows] == list(range(C.K_MIN, C.K_MAX + 1))
        vals = [r["exact"] for r in rows]
        assert vals == sorted(vals, reverse=True) and len(set(vals)) == C.K_MAX, n      # fair, bei diesen n: je größer k, desto kleiner der Erwartungswert
        assert all(r["ratio"] == pytest.approx(0.5) for r in rows)


def test_sweep_m_mean_stays_and_pooled_rises_branch_test():
    rows = ev.sweep_m(20, 3, 50, grid=(1, 5, 50), per_seed=1000)
    assert [r["m"] for r in rows] == [1, 5, 50]
    assert rows[0]["mean"] == rows[0]["pooled"]                       # eine Folge: dasselbe Maß
    assert rows[0]["pooled"] < rows[1]["pooled"] < rows[2]["pooled"]  # gepoolt steigt mit m
    assert all(abs(r["mean"] - rows[0]["exact_single"]) < 0.03 for r in rows)


def test_power_curve_rows_and_the_two_tests_differ_branch_test():
    rows = ev.power_curve(40, 2, 10, q_grid=(50, 80), seeds=C.SWEEP_SEEDS[:3], reps=6, count=39)
    assert [r["q_pct"] for r in rows] == [50, 80] and rows[0]["runs"] == 18
    for r in rows:
        for key in ("naive_hot", "perm_hot", "naive_cold", "perm_cold"):
            assert 0 <= r[key + "_low"] <= r[key] <= r[key + "_high"] <= 1
    assert rows[1]["perm_hot"] > rows[0]["perm_hot"] and rows[1]["naive_hot"] > rows[0]["naive_hot"]
    assert rows[0]["naive_cold"] > rows[0]["perm_cold"]               # die naive Prüfung meldet bei fairer Münze häufiger „kalt“


def test_first_q_reaching_uses_the_smallest_q_above_fifty_and_none_when_never():
    rows = [{"q_pct": 50, "x": 0.9}, {"q_pct": 60, "x": 0.5}, {"q_pct": 70, "x": 0.8}, {"q_pct": 80, "x": 1.0}]
    assert ev.first_q_reaching(rows, "x") == 70                       # q = 50 zählt nicht (dort ist es die Fehlalarmquote), 0,8 genau erreicht
    assert ev.first_q_reaching(rows, "x", 1.01) is None


def test_enumeration_rows_agree_across_methods_and_name_the_formula_rows():
    rows = ev.enumeration_rows(12, 2, 0.7)
    assert len(rows) == 5
    for r in rows:
        assert r["enum"] == pytest.approx(r["recursion"], abs=1e-12)
    with_formula = [r for r in rows if r["formula"] is not None]
    assert len(with_formula) == 3 and all(r["enum"] == pytest.approx(r["formula"], abs=1e-12) for r in with_formula)


def test_method_rows_cover_every_named_method_and_show_values():
    a = ev.analyse(ev.Settings())
    rows = ev.method_rows(a)
    names = [r["Name"] for r in rows]
    assert names == ["Gelegenheit", "Schätzer einer Folge", "Mittel der Folgenschätzer", "Gepoolter Schätzer", "Rekursion (exakt)", "Aufzählung (exakt)", "Simulation", "Naive Prüfung", "Permutationstest"]
    assert all(r["Was"] and r["Wert"] for r in rows)
    assert ev.fmt(a.mean_est) in [r["Wert"] for r in rows]
