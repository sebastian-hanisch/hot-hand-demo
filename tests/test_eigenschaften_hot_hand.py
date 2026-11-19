"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Beziehungen, die gelten müssen, ohne dass man das richtige Ergebnis kennt. Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed."""

import numpy as np
import pytest

import hoh_algorithm as A
import hoh_evaluation as ev
import hoh_exact as E
from hoh_scenario import derive_seed, make_sequences


def _batch(i, m=12, n=25, q=0.5):
    return make_sequences(m, n, q, derive_seed(77, i))


def test_the_order_of_the_folgen_does_not_matter():
    rng = np.random.default_rng(0)  # nur zum Mischen der Reihenfolge im Test, kein Ergebnis hängt an den Zahlen
    for i in range(30):
        x = _batch(i)
        O, H = A.opportunity_counts(x, 2)
        perm = rng.permutation(len(x))
        O2, H2 = A.opportunity_counts(x[perm], 2)
        assert A.mean_of_estimators(O, H) == pytest.approx(A.mean_of_estimators(O2, H2), abs=1e-12)
        assert A.pooled_estimate(O, H) == pytest.approx(A.pooled_estimate(O2, H2), abs=1e-12)


def test_duplicating_all_folgen_changes_neither_measure():
    for i in range(30):
        x = _batch(i)
        xx = np.concatenate([x, x])
        O, H = A.opportunity_counts(x, 3)
        O2, H2 = A.opportunity_counts(xx, 3)
        assert A.mean_of_estimators(O, H) == pytest.approx(A.mean_of_estimators(O2, H2), abs=1e-12)
        assert A.pooled_estimate(O, H) == pytest.approx(A.pooled_estimate(O2, H2), abs=1e-12)


def test_pooled_estimate_of_two_groups_is_the_weighted_combination():
    for i in range(30):
        a, b = _batch(i, m=6), _batch(i + 100, m=9)
        (Oa, Ha), (Ob, Hb) = A.opportunity_counts(a, 2), A.opportunity_counts(b, 2)
        Oc, Hc = A.opportunity_counts(np.concatenate([a, b]), 2)
        assert A.pooled_estimate(Oc, Hc) == pytest.approx((Ha.sum() + Hb.sum()) / (Oa.sum() + Ob.sum()), abs=1e-12)


def test_more_required_hits_never_create_more_opportunities():
    for i in range(30):
        x = _batch(i, m=10, n=30)
        counts = [A.opportunity_counts(x, k)[0] for k in (1, 2, 3, 4, 5)]
        assert all((a >= b).all() for a, b in zip(counts, counts[1:]))
        assert counts[0].sum() > counts[-1].sum()


def test_estimates_lie_between_zero_and_one_and_hits_never_exceed_opportunities():
    for i in range(30):
        x = _batch(i, q=[0.3, 0.5, 0.8][i % 3])
        O, H = A.opportunity_counts(x, 2)
        e = A.estimates(O, H)
        assert (H <= O).all() and (e[~np.isnan(e)] >= 0).all() and (e[~np.isnan(e)] <= 1).all()


def test_complementing_all_throws_leaves_the_sequence_probabilities_unchanged():
    x = E.all_sequences(9)
    for q in (0.2, 0.5, 0.8):
        assert np.allclose(E.sequence_probabilities(x, q), E.sequence_probabilities(1 - x, q), rtol=1e-12)


def test_complete_persistence_makes_every_sequence_all_hits_or_all_misses():
    x = make_sequences(30, 15, 1.0, 3)
    O, H = A.opportunity_counts(x, 2)
    for row, o, h in zip(x, O, H):
        assert (row.all() and (o, h) == (13, 13)) or (not row.any() and (o, h) == (0, 0))
    assert A.mean_of_estimators(O, H) == 1.0 and A.pooled_estimate(O, H) == 1.0


def test_exact_estimator_expectation_rises_with_q_on_the_grid():
    for n, k in [(20, 3), (8, 5), (50, 2), (30, 1)]:
        vals = [E.exact_table(n, k, q / 100)[n].mean_est for q in range(30, 95, 5)]
        assert all(b > a for a, b in zip(vals, vals[1:])), (n, k)


def test_the_estimator_is_unbiased_only_where_one_opportunity_is_possible_and_biased_for_the_fair_coin_otherwise():
    for k in (1, 2, 3, 4):
        table = E.exact_table(60, k, 0.5)
        assert table[k + 1].mean_est == pytest.approx(0.5, abs=1e-15)
        assert all(table[n].mean_est < 0.5 - 1e-6 for n in range(k + 2, 61)), k    # fair: ab n = k + 2 über die ganze Länge bis 60 unter 1/2


def test_the_pooled_limit_does_not_depend_on_n_or_k():
    for k in (1, 2, 3):
        table = E.exact_table(40, k, 0.6)
        assert all(s.ratio == pytest.approx(0.6, rel=1e-9) for s in table.values())


def test_same_settings_same_analysis_and_other_seed_other_analysis():
    a = ev.analyse(ev.Settings(n=15, k=2, q_pct=60, m=20, seed=4))
    b = ev.analyse(ev.Settings(n=15, k=2, q_pct=60, m=20, seed=4))
    c = ev.analyse(ev.Settings(n=15, k=2, q_pct=60, m=20, seed=5))
    assert (a.x == b.x).all() and a.mean_est == b.mean_est and a.perm == b.perm
    assert not (a.x == c.x).all()


def test_permutation_does_not_change_the_hit_count_and_the_test_depends_only_on_the_hit_counts_and_order():
    for i in range(20):
        x = _batch(i, m=3, n=18)
        res1 = A.permutation_test(x, 2, 49, 5)
        res2 = A.permutation_test(x[::-1], 2, 49, 5)      # andere Reihenfolge der Folgen: derselbe Prüfwert
        assert res1["t_obs"] == pytest.approx(res2["t_obs"], abs=1e-12)
