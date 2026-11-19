"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Längen, k an der Grenze, eine einzige Folge, lauter gleiche Würfe, Extremwerte der Regler."""

import itertools

import numpy as np
import pytest

import hoh_algorithm as A
import hoh_constants as C
import hoh_evaluation as ev
import hoh_exact as E
from hoh_scenario import make_sequences


def test_exactly_two_throws_and_k_one():
    """n = 2, k = 1: eine einzige mögliche Gelegenheit (der zweite Wurf nach einem Treffer): Zweige `n < 2` dürfen nicht greifen."""
    O, H = A.opportunity_counts(np.array([[1, 1], [1, 0], [0, 1], [0, 0]]), 1)
    assert O.tolist() == [1, 1, 0, 0] and H.tolist() == [1, 0, 0, 0]
    for q in (0.3, 0.5, 0.9):
        assert E.exact_table(2, 1, q)[2].mean_est == pytest.approx(q)
        assert E.enumerate_stats(2, 1, q).mean_est == pytest.approx(q)


def test_k_is_inclusive_at_both_ends():
    """k = 1 und k = n - 1 sind erlaubt, k = 0 und k = n nicht."""
    x = np.ones((1, 6), dtype=int)
    assert int(A.opportunity_counts(x, 1)[0][0]) == 5 and int(A.opportunity_counts(x, 5)[0][0]) == 1
    for k in (0, 6):
        with pytest.raises(ValueError):
            A.opportunity_counts(x, k)


def test_all_hits_and_all_misses():
    O, H = A.opportunity_counts(np.array([[1] * 9, [0] * 9]), 3)
    assert O.tolist() == [6, 0] and H.tolist() == [6, 0]
    assert A.mean_of_estimators(O, H) == 1.0                        # die Folge ohne Gelegenheit fällt weg, die andere hat Schätzer 1


def test_a_single_folge_makes_mean_and_pooled_the_same_measure():
    for seed in range(20):
        x = make_sequences(1, 30, 0.5, seed)
        O, H = A.opportunity_counts(x, 2)
        a, b = A.mean_of_estimators(O, H), A.pooled_estimate(O, H)
        assert (a == b) or (np.isnan(a) and np.isnan(b))


def test_every_slider_corner_gives_a_complete_analysis():
    corners = itertools.product((C.N_MIN, C.N_MAX), (C.K_MIN, C.K_MAX), (C.Q_PCT_MIN, C.Q_PCT_MAX), (C.M_MIN, C.M_MAX))
    for n, k, q, m in corners:
        a = ev.analyse(ev.Settings(n=n, k=k, q_pct=q, m=m, seed=C.SEED_MAX))
        assert a.x.shape == (m, n) and 0 <= a.n_defined <= m
        assert a.exact.p_defined > 0
        ev.verdict_main(a)
        ev.verdict_reference(a)
        ev.verdict_example(a, n)
        ev.method_rows(a)


def test_smallest_instance_k_max_has_few_opportunities_but_a_defined_exact_value():
    """n = 8 und k = 5: nur 3 Positionen können Gelegenheit sein; die Aufzählung gibt es und stimmt mit der Rekursion."""
    en, rc = E.enumerate_stats(C.N_MIN, C.K_MAX, 0.5), E.exact_table(C.N_MIN, C.K_MAX, 0.5)[C.N_MIN]
    assert 0 < rc.p_defined < 0.2 and rc.mean_est == pytest.approx(en.mean_est, abs=1e-13)


def test_k_max_fits_into_the_shortest_sequence():
    assert C.K_MAX <= C.N_MIN - 3 and C.ENUM_MAX >= C.N_MIN


def test_fmt_handles_nan_and_signs():
    assert ev.fmt(float("nan")) == "–" and ev.pct(float("nan")) == "–"
    assert ev.fmt(-0.1394, 3) == "−0,139" and ev.fmt(0.5, 2) == "0,50"
    assert ev.pct(0.05) == "5,0 %" and ev.pct(0.123) == "12,3 %"
    assert ev.fmt(-0.0001) == "0,000" and ev.fmt(-0.0006) == "−0,001"


def test_example_without_any_opportunity_says_so():
    a = ev.analyse(ev.Settings(n=8, k=5, q_pct=30, m=1, seed=1))
    kind, text = ev.verdict_example(a, 3)
    assert kind == "info" and "noch keine Gelegenheit" in text
