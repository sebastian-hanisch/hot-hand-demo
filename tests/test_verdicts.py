"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier."""

import dataclasses

import hoh_evaluation as ev
import hoh_exact as E


def _analysis(mean_exact=0.361, truth_q=50, p_defined=0.77, mean_est=0.339, pooled=0.495):
    a = ev.analyse(ev.Settings(n=20, k=3, q_pct=truth_q, m=100, seed=35))
    exact = E.ExactStats(20, 3, truth_q / 100, p_defined, mean_exact, 2.0, 1.0)     # E[H] / E[O] = 0,5
    return dataclasses.replace(a, exact=exact, mean_est=mean_est, pooled=pooled)


def test_main_verdict_names_exact_value_truth_bias_and_sample():
    kind, text = ev.verdict_main(_analysis())
    assert kind == "warning"
    assert "n = 20 Würfen und k = 3" in text and "**0,361**" in text and "**0,50**" in text and "**−0,139**" in text
    assert "gegen 0,500" in text and "Mittel dieser 100 Folgen: 0,339 (gepoolt 0,495)" in text


def test_main_verdict_threshold_between_info_and_warning():
    assert ev.verdict_main(_analysis(mean_exact=0.496))[0] == "info"          # Verzerrung −0,004
    kind, text = ev.verdict_main(_analysis(mean_exact=0.494))                  # −0,006
    assert kind == "warning" and "unter 0,005" not in text
    assert "unter 0,005" in ev.verdict_main(_analysis(mean_exact=0.502))[1]    # auch nach oben zählt der Betrag


def test_main_verdict_without_possible_opportunity():
    kind, text = ev.verdict_main(_analysis(p_defined=0.0, mean_exact=float("nan")))
    assert kind == "warning" and "nicht definiert" in text


def _length_rows():
    return [{"n": 10, "exact": 0.353, "low": 0.34, "high": 0.36, "sim": 0.35}, {"n": 20, "exact": 0.361, "low": 0.37, "high": 0.38, "sim": 0.375},
            {"n": 200, "exact": 0.481, "low": 0.478, "high": 0.482, "sim": 0.48}]


def test_length_verdict_counts_bands_and_names_the_worst_length():
    curve = [(4, 0.40), (12, 0.351), (30, 0.43), (200, 0.481)]
    kind, text = ev.verdict_length(_length_rows(), 0.5, 3, curve)
    assert kind == "warning"
    assert "bei n = 10 bei 0,353, bei n = 200 bei 0,481 (wahr 0,50)" in text and "von −0,147 auf −0,019" in text
    assert "Am größten ist sie bei n = 12 (−0,149)" in text and "k + 1 = 4 ist der Schätzer unverzerrt" in text
    assert "In 2 von 3 simulierten Längen" in text        # n = 20: 0,361 liegt nicht in [0,37, 0,38]; die beiden anderen Längen liegen im Band


def test_length_verdict_info_when_the_bias_at_the_longest_length_is_below_the_threshold():
    rows = [{"n": 10, "exact": 0.4, "low": 0.3, "high": 0.5, "sim": 0.4}, {"n": 200, "exact": 0.499, "low": 0.4, "high": 0.6, "sim": 0.5}]
    assert ev.verdict_length(rows, 0.5, 2, [(3, 0.5), (200, 0.499)])[0] == "info"
    rows[1]["exact"] = 0.49
    assert ev.verdict_length(rows, 0.5, 2, [(3, 0.5), (200, 0.49)])[0] == "warning"


def test_k_verdict():
    rows = [{"k": 1, "exact": 0.474, "p_defined": 1.0}, {"k": 2, "exact": 0.415, "p_defined": 0.979}, {"k": 5, "exact": 0.331, "p_defined": 0.237}]
    kind, text = ev.verdict_k(rows, 20, 0.5)
    assert kind == "info"
    assert "Bei n = 20: Die Verzerrung ist −0,026 bei k = 1 und −0,169 bei k = 5; am größten ist sie bei k = 5 (−0,169)" in text
    assert "von 100,0 % auf 23,7 %" in text


def test_m_verdict_counts_the_bands_that_contain_the_exact_value():
    rows = [{"m": 1, "mean": 0.360, "mean_low": 0.355, "mean_high": 0.364, "pooled": 0.360, "exact_single": 0.3613, "ratio": 0.5},
            {"m": 10, "mean": 0.37, "mean_low": 0.3641, "mean_high": 0.3708, "pooled": 0.487, "exact_single": 0.3613, "ratio": 0.5},
            {"m": 100, "mean": 0.365, "mean_low": 0.357, "mean_high": 0.374, "pooled": 0.503, "exact_single": 0.3613, "ratio": 0.5}]
    kind, text = ev.verdict_m(rows)
    assert kind == "warning"
    assert "bei m = 1 0,360, bei m = 100 0,365 (exakte Erwartung 0,361; sie liegt in 2 von 3 Bändern" in text      # m = 10: 0,3613 liegt unter 0,3641
    assert "von 0,360 bei m = 1 auf 0,503 bei m = 100" in text and "ist 0,500" in text


def _power_rows():
    base = {"runs": 100}
    return [dict(base, q_pct=50, perm_hot=0.05, naive_hot=0.01, perm_cold=0.04, naive_cold=0.19),
            dict(base, q_pct=55, perm_hot=0.47, naive_hot=0.18, perm_cold=0.0, naive_cold=0.01),
            dict(base, q_pct=60, perm_hot=0.93, naive_hot=0.71, perm_cold=0.0, naive_cold=0.0),
            dict(base, q_pct=65, perm_hot=1.0, naive_hot=0.95, perm_cold=0.0, naive_cold=0.0)]


def test_power_verdict_names_first_q_reaching_eighty_percent_and_the_false_alarm_rates():
    kind, text = ev.verdict_power(_power_rows(), 50, 2, 20)
    assert kind == "info"
    assert "Permutationstest** erreicht 80 % Teststärke ab q = 0,60, die **naive Prüfung** ab q = 0,65" in text
    assert "„heiß“ in 5,0 %, die naive Prüfung in 1,0 %" in text
    assert "„kalt“ meldet bei fairer Münze der Permutationstest in 4,0 %, die naive Prüfung in 19,0 %" in text
    assert "n = 50, k = 2, 20 Folgen, je Wert 100 Läufe" in text


def test_power_verdict_when_a_test_never_reaches_the_target():
    rows = [dict(r, naive_hot=min(r["naive_hot"], 0.5)) for r in _power_rows()]
    text = ev.verdict_power(rows, 50, 2, 20)[1]
    assert "naive Prüfung** im Raster bis q = 0,65 nicht" in text


def test_example_verdict_counts_opportunities_up_to_the_step():
    a = ev.analyse(ev.Settings(n=20, k=1, q_pct=50, m=1, seed=3))
    x = a.x[0]
    seen_some = False
    for step in (1, 2, 3, 7, 20):
        o = sum(1 for t in range(1, step) if x[t - 1])
        h = sum(int(x[t]) for t in range(1, step) if x[t - 1])
        text = ev.verdict_example(a, step)[1]
        if o == 0:
            assert "noch keine Gelegenheit" in text
        else:
            seen_some = True
            assert text == f"Nach {step} Würfen: {o} Gelegenheiten, davon {h} Treffer, Schätzer {h}/{o} = {ev.fmt(h / o)}."
    assert seen_some and ev.verdict_example(a, 1)[1].startswith("Nach 1 Wurf gibt es noch keine Gelegenheit")


def test_reference_verdict_names_hits_reference_and_p_values():
    a = ev.analyse(ev.Settings())
    text = ev.verdict_reference(a)[1]
    p = a.perm
    assert f"{int(a.x[0].sum())} Treffer in 20 Würfen" in text and f"**{ev.fmt(p['null_mean'])}**" in text
    assert f"nach oben {ev.fmt(p['p_upper'], 3)}, nach unten {ev.fmt(p['p_lower'], 3)}" in text and "999 Umordnungen" in text
    none = dataclasses.replace(a, perm=dict(p, t_obs=float("nan")))
    assert "nicht definiert" in ev.verdict_reference(none)[1]
