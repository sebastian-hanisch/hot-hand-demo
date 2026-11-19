"""Jede Zahl der README steht hier (Zahl-zu-Test-Tabelle der README). Exakte Größen (Rekursion, Aufzählung) werden auf die gedruckte Stelle verglichen, Simulationen und Teststärken als Band
(feste Zufalls-Seeds, aber Summationsreihenfolgen und Vergleiche können auf anderen Plattformen in der letzten Binärstelle abweichen: DEMO-PLAYBOOK §5).

Aufbau: Abschnitte wie in der README, „Ergebnis in Kürze“ und „Befunde“."""

import functools
import re
import subprocess
import sys
from pathlib import Path

import pytest

import hoh_constants as C
import hoh_evaluation as ev
import hoh_exact as E

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


@functools.lru_cache(maxsize=None)
def _table(n_max, k, q_pct):
    return E.exact_table(n_max, k, q_pct / 100.0)


@functools.lru_cache(maxsize=None)
def _length(k, q_pct):
    return ev.sweep_length(k, q_pct)


@functools.lru_cache(maxsize=None)
def _m_rows():
    return ev.sweep_m(20, 3, 50)


@functools.lru_cache(maxsize=None)
def _power(n, k, m):
    return ev.power_curve(n, k, m)


def _de(x, digits=3):
    return f"{x:.{digits}f}".replace(".", ",")


# --- Exakte Erwartung: die Verzerrung der Folgenschätzer -------------------------------------------------------------------------------------

def test_exact_expectation_k3_fair_by_length():
    """Tabelle „Verzerrung über die Länge“, Spalte exakt (Rekursion, k = 3, faire Münze)."""
    t = _table(200, 3, 50)
    assert [_de(t[n].mean_est) for n in (10, 20, 50, 100, 200)] == ["0,353", "0,361", "0,418", "0,460", "0,481"]
    assert [_de(t[n].mean_est - 0.5) for n in (20, 200)] == ["-0,139", "-0,019"]


def test_the_plan_numbers_of_the_premeasurement_lie_within_the_sampling_error_of_the_exact_values():
    """Die Vormessung (20 000 Folgen je Länge, anderer Zufallsgenerator) nannte 0,359 / 0,418 / 0,459 / 0,480; exakt sind 0,361 / 0,418 / 0,460 / 0,481: größte Abweichung 0,002."""
    pre = {20: 0.359, 50: 0.418, 100: 0.459, 200: 0.480}
    t = _table(200, 3, 50)
    assert max(abs(t[n].mean_est - v) for n, v in pre.items()) == pytest.approx(0.0023, abs=0.0004)
    assert max(abs(t[n].mean_est - v) for n, v in pre.items()) < 0.003


def test_opportunities_and_defined_share_at_k3():
    t = _table(200, 3, 50)
    assert _de(t[20].p_defined, 3) == "0,768" and _de(t[20].e_o, 3) == "2,125" and _de(t[200].e_o, 1) == "24,6"
    assert t[200].p_defined > 0.999999


def test_pooled_limit_equals_q_exactly_for_every_length_and_q():
    for q_pct in (50, 65):
        assert all(s.ratio == pytest.approx(q_pct / 100, rel=1e-9) for s in _table(200, 3, q_pct).values())


def test_dependence_is_hidden_by_the_bias_at_short_length():
    """Persistenz q: bei n = 20 und k = 3 erwartet man vom Schätzer 0,411 (q = 0,55), 0,464 (q = 0,60), 0,518 (q = 0,65), 0,574 (q = 0,70); Verzerrung zwischen −0,139 und −0,126."""
    vals = {q: E.exact_table(20, 3, q / 100)[20].mean_est for q in (50, 55, 60, 65, 70)}
    assert [_de(vals[q]) for q in (55, 60, 65, 70)] == ["0,411", "0,464", "0,518", "0,574"]
    assert [round(vals[q] - q / 100, 3) for q in (50, 55, 60, 65, 70)] == [-0.139, -0.139, -0.136, -0.132, -0.126]
    assert vals[60] < 0.5 < vals[65]


def test_n50_k2_values():
    assert _de(E.exact_table(50, 2, 0.5)[50].mean_est) == "0,467" and _de(E.exact_table(50, 2, 0.6)[50].mean_est) == "0,569"


def test_k_dependence_at_n20_and_n50():
    for n, expected in ((20, ["0,474", "0,415", "0,361", "0,339", "0,331"]), (50, ["0,490", "0,467", "0,418", "0,367", "0,338"])):
        rows = ev.sweep_k(n, 50)
        assert [_de(r["exact"]) for r in rows] == expected, n
    r20 = ev.sweep_k(20, 50)
    assert [_de(r["p_defined"] * 100, 1) for r in r20] == ["100,0", "97,9", "76,8", "45,8", "23,7"]


def test_bias_is_not_monotone_in_length_largest_length_per_k():
    """Größte Verzerrung (fair): k = 1 bei n = 4 (0,405), k = 2 bei n = 8 (0,370), k = 3 bei n = 12 (0,351), k = 5 bei n = 26 (0,330); bei n = k + 1 gar keine."""
    out = {}
    for k in (1, 2, 3, 5):
        t = _table(200, k, 50)
        n_min = min(t, key=lambda n: t[n].mean_est)
        out[k] = (n_min, _de(t[n_min].mean_est))
        assert t[k + 1].mean_est == pytest.approx(0.5)
    assert out == {1: (4, "0,405"), 2: (8, "0,370"), 3: (12, "0,351"), 5: (26, "0,330")}


# --- Simulation ------------------------------------------------------------------------------------------------------------------------------

def test_simulation_bands_contain_the_exact_expectation_in_five_of_five_lengths():
    rows = _length(3, 50)
    assert sum(1 for r in rows if r["low"] <= r["exact"] <= r["high"]) == 5
    assert [round(r["sim"], 3) for r in rows] == pytest.approx([0.348, 0.364, 0.419, 0.459, 0.481], abs=0.0006)
    assert max(abs(r["sim"] - r["exact"]) for r in rows) < 0.005


def test_more_folgen_table():
    """n = 20, k = 3, faire Münze: Mittel der Folgenschätzer bleibt bei 0,36 bis 0,37, gepoolt steigt auf 0,503 (m = 100)."""
    rows = _m_rows()
    assert [r["m"] for r in rows] == list(C.M_GRID)
    assert [round(r["mean"], 3) for r in rows] == pytest.approx([0.360, 0.361, 0.359, 0.367, 0.358, 0.363, 0.365], abs=0.0006)
    assert [round(r["pooled"], 3) for r in rows] == pytest.approx([0.360, 0.401, 0.455, 0.487, 0.488, 0.494, 0.503], abs=0.0006)
    assert sum(1 for r in rows if r["mean_low"] <= r["exact_single"] <= r["mean_high"]) == 6
    assert rows[-1]["pooled_low"] < 0.5 < rows[-1]["pooled_high"]
    assert (round(rows[-1]["pooled_low"], 3), round(rows[-1]["pooled_high"], 3)) == pytest.approx((0.496, 0.507), abs=0.0006)
    outside = [r["m"] for r in rows if not r["mean_low"] <= r["exact_single"] <= r["mean_high"]]
    assert outside == [10]                                                   # README: bei m = 10 knapp außerhalb
    assert max(abs(r["mean"] - r["exact_single"]) for r in rows) < 0.01


# --- Beispiel bei den Voreinstellungen -----------------------------------------------------------------------------------------------------

def test_default_page_example():
    a = ev.analyse(ev.Settings())
    assert a.n_defined == 80 and _de(a.mean_est) == "0,339" and _de(a.pooled) == "0,495"
    p = a.perm
    assert int(a.x[0].sum()) == 12 and _de(p["t_obs"]) == "0,667" and p["null_mean"] == pytest.approx(0.436, abs=0.002) and p["p_upper"] == pytest.approx(0.294, abs=0.01)
    assert p["valid"] == pytest.approx(978, abs=5)


def test_enumeration_n16_k2_values():
    rows = ev.enumeration_rows(16, 2, 0.5)
    assert [_de(r["recursion"], 3) for r in rows] == ["0,951", "3,500", "1,750", "0,500", "0,400"]
    assert all(r["enum"] == pytest.approx(r["recursion"], abs=1e-12) for r in rows)
    assert _de(rows[-1]["recursion"], 4) == "0,3996"
    assert all(r["enum"] == pytest.approx(r["formula"], abs=1e-12) for r in rows if r["formula"] is not None) and sum(r["formula"] is not None for r in rows) == 3


# --- Teststärke -------------------------------------------------------------------------------------------------------------------------------

def test_power_n50_k2_m20():
    """Tabelle „Teststärke“ (n = 50, k = 2, 20 Folgen, 100 Läufe je q)."""
    rows = {r["q_pct"]: r for r in _power(50, 2, 20)}
    assert rows[50]["naive_hot"] == pytest.approx(0.01, abs=0.03) and rows[50]["perm_hot"] == pytest.approx(0.05, abs=0.04)
    assert rows[50]["naive_cold"] == pytest.approx(0.19, abs=0.04) and rows[50]["perm_cold"] == pytest.approx(0.04, abs=0.03)
    assert rows[55]["perm_hot"] == pytest.approx(0.47, abs=0.04) and rows[55]["naive_hot"] == pytest.approx(0.18, abs=0.04)
    assert rows[60]["perm_hot"] == pytest.approx(0.93, abs=0.04) and rows[60]["naive_hot"] == pytest.approx(0.71, abs=0.04)
    assert rows[65]["perm_hot"] >= 0.97 and rows[65]["naive_hot"] == pytest.approx(0.95, abs=0.04)
    assert all(rows[q]["perm_hot"] > rows[q]["naive_hot"] for q in (55, 60, 65))
    assert ev.first_q_reaching(list(rows.values()), "perm_hot") == 60 and ev.first_q_reaching(list(rows.values()), "naive_hot") == 65
    assert all(r["perm_hot_low"] >= 0 and r["perm_hot_high"] <= 1 for r in rows.values())
    assert (rows[50]["naive_cold_low"], rows[50]["naive_cold_high"]) == pytest.approx((0.10, 0.25), abs=1e-9)        # README: je Zufalls-Seed (20 Läufe) 10 bis 25 %


def test_power_n20_k3_m20():
    rows = {r["q_pct"]: r for r in _power(20, 3, 20)}
    assert (rows[50]["naive_cold_low"], rows[50]["naive_cold_high"]) == pytest.approx((0.20, 0.55), abs=1e-9)        # README: je Zufalls-Seed (20 Läufe) 20 bis 55 %
    assert [round(rows[q]["perm_hot"], 2) for q in C.Q_GRID_PCT] == pytest.approx([0.03, 0.12, 0.27, 0.48, 0.74, 0.94], abs=0.04)
    assert [round(rows[q]["naive_hot"], 2) for q in C.Q_GRID_PCT] == pytest.approx([0.00, 0.00, 0.03, 0.11, 0.39, 0.92], abs=0.04)
    assert rows[50]["naive_cold"] == pytest.approx(0.43, abs=0.05) and rows[55]["naive_cold"] == pytest.approx(0.34, abs=0.05) and rows[50]["perm_cold"] == pytest.approx(0.02, abs=0.03)
    assert ev.first_q_reaching(list(rows.values()), "perm_hot") == 80


def test_false_alarm_rate_of_the_permutation_test_stays_near_five_percent_everywhere():
    for key in ((50, 2, 20), (20, 3, 20)):
        r50 = next(r for r in _power(*key) if r["q_pct"] == 50)
        assert r50["perm_hot"] <= 0.07 and r50["perm_cold"] <= 0.07, key             # Mittel über die 5 Zufalls-Seeds (100 Läufe): Standardfehler bei 5 % rund 2,2 Punkte
        assert r50["perm_hot_high"] <= 0.10 and r50["perm_cold_high"] <= 0.10, key   # je Zufalls-Seed (20 Läufe) höchstens 10 %, wie die README sagt


# --- Testzahl ----------------------------------------------------------------------------------------------------------------------------------

def test_the_test_count_in_the_readme_matches_the_suite():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    m = re.search(r"(\d+) tests? collected", out) or re.search(r"^(\d+) tests", out, re.M)
    n = int(m.group(1))
    declared = re.search(r"\*\*(\d+) Tests\*\*", README)
    assert declared and int(declared.group(1)) == n, (declared.group(1) if declared else None, n)


def test_every_decimal_in_the_summary_is_german():
    summary = README.split("## Ergebnis in Kürze", 1)[1].split("\n## ", 1)[0]
    assert not re.search(r"\b\d\.\d+\b", summary), "Dezimalpunkt statt -komma"


def test_the_printed_numbers_appear_in_the_readme():
    """Die Zahlen der Tests oben stehen auch im Text (Kurzfassung der README-Tabellen): fehlt eine, ist die README oder der Test veraltet."""
    printed = ["0,353", "0,361", "0,418", "0,460", "0,481", "0,348", "0,364", "0,419", "0,459", "76,8 %", "2,125", "24,6", "0,405", "0,370", "0,351", "0,330",
               "0,474", "0,415", "0,339", "0,331", "0,490", "0,467", "0,367", "0,338", "100,0 %", "97,9 %", "45,8 %", "23,7 %",
               "0,360", "0,401", "0,455", "0,487", "0,488", "0,494", "0,503", "0,365", "0,496", "0,507",
               "0,411", "0,464", "0,518", "0,574", "−0,139", "−0,136", "−0,132", "−0,126", "0,569",
               "0,951", "3,500", "1,750", "0,3996", "0,436", "0,294", "0,667", "0,339", "0,495", "978", "47 %", "93 %", "71 %", "95 %", "19 %", "43 %", "74 %", "94 %", "39 %", "12 %", "27 %", "48 %"]
    missing = [x for x in printed if x not in README]
    assert not missing, missing
