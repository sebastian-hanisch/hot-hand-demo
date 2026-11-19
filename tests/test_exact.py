"""Exakte Erwartungen gegen Handrechnung (kleinste Fälle) und gegeneinander: Rekursion, Aufzählung, Formeln."""

import numpy as np
import pytest

import hoh_exact as E


def test_n3_k1_fair_coin_by_hand():
    """Von Hand (8 Folgen): Schätzer 0, 1, 0, 0, 1/2, 1 in den sechs Folgen 010, 011, 100, 101, 110, 111; 000 und 001 haben keine Gelegenheit. Mittel 5/12; E[O] = 8/8, E[H] = 4/8."""
    for s in (E.enumerate_stats(3, 1, 0.5), E.exact_table(3, 1, 0.5)[3]):
        assert s.mean_est == pytest.approx(5 / 12, abs=1e-15)
        assert s.p_defined == pytest.approx(6 / 8)
        assert s.e_o == pytest.approx(1.0) and s.e_h == pytest.approx(0.5)


def test_n4_k2_fair_coin_by_hand():
    """Von Hand (16 Folgen): Schätzer 0, 0, 1/2, 1 für 1100, 1101, 1110, 1111 und 0, 1 für 0110, 0111; alle übrigen haben keine Gelegenheit. Mittel 5/12, E[O] = 8/16, E[H] = 4/16."""
    for s in (E.enumerate_stats(4, 2, 0.5), E.exact_table(4, 2, 0.5)[4]):
        assert s.mean_est == pytest.approx(5 / 12, abs=1e-15)
        assert s.p_defined == pytest.approx(6 / 16)
        assert s.e_o == pytest.approx(0.5) and s.e_h == pytest.approx(0.25)


def test_n3_k1_with_persistence_three_fifths_by_hand():
    """q = 3/5: Gewichte der sechs Folgen 1/2·(2/5)², 1/2·(2/5)(3/5), 1/2·(2/5)(3/5), 1/2·(2/5)², 1/2·(3/5)(2/5), 1/2·(3/5)²; definierte Masse 0,7, Zähler 0,36, also 18/35."""
    for s in (E.enumerate_stats(3, 1, 0.6), E.exact_table(3, 1, 0.6)[3]):
        assert s.p_defined == pytest.approx(0.7) and s.mean_est == pytest.approx(18 / 35)


def test_one_possible_opportunity_is_unbiased_for_every_q():
    """n = k + 1: höchstens der letzte Wurf kann Gelegenheit sein; dann ist der Schätzer dieser Wurf selbst, und nach k Treffern ist er mit Wahrscheinlichkeit q ein Treffer."""
    for k in (1, 2, 3):
        for q in (0.3, 0.5, 0.8):
            assert E.exact_table(k + 1, k, q)[k + 1].mean_est == pytest.approx(q)
            assert E.enumerate_stats(k + 1, k, q).mean_est == pytest.approx(q)


@pytest.mark.parametrize("q", [0.3, 0.5, 0.65, 0.8])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_recursion_equals_enumeration_on_a_grid(k, q):
    table = E.exact_table(12, k, q)
    for n in range(k + 1, 13):
        en = E.enumerate_stats(n, k, q)
        rc = table[n]
        assert rc.p_defined == pytest.approx(en.p_defined, abs=1e-12)
        assert rc.mean_est == pytest.approx(en.mean_est, abs=1e-12)
        assert rc.e_o == pytest.approx(en.e_o, abs=1e-12) and rc.e_h == pytest.approx(en.e_h, abs=1e-12)


def test_closed_form_expectations_equal_enumeration():
    for n, k, q in [(10, 1, 0.5), (10, 3, 0.5), (12, 2, 0.7), (9, 4, 0.35), (16, 3, 0.5)]:
        en = E.enumerate_stats(n, k, q)
        assert E.expected_opportunities(n, k, q) == pytest.approx(en.e_o, rel=1e-12)
        assert E.expected_hits(n, k, q) == pytest.approx(en.e_h, rel=1e-12)
        assert en.ratio == pytest.approx(q, rel=1e-12)           # E[H] / E[O] = q


def test_pooled_limit_is_q_for_every_length_in_the_recursion():
    table = E.exact_table(60, 3, 0.7)
    assert all(s.ratio == pytest.approx(0.7, rel=1e-9) for s in table.values())


def test_recursion_for_one_length_does_not_depend_on_n_max():
    short, long = E.exact_table(20, 3, 0.5)[20], E.exact_table(80, 3, 0.5)[20]
    assert short.mean_est == pytest.approx(long.mean_est, abs=1e-14) and short.p_defined == pytest.approx(long.p_defined, abs=1e-14)


def test_pmf_sums_to_one_and_has_h_at_most_o():
    pmf = E.exact_pmf(12, 2, 0.6)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    assert not np.triu(pmf, 1).any()                      # pmf[o, h] = 0 für h > o
    assert pmf[0, 1:].sum() == 0 and pmf[1:, :].sum() > 0.5


def test_sequence_probabilities_sum_to_one_and_all_sequences_are_distinct():
    x = E.all_sequences(8)
    assert len({tuple(r) for r in x.tolist()}) == 256 and x.shape == (256, 8)
    for q in (0.2, 0.5, 0.9):
        assert E.sequence_probabilities(x, q).sum() == pytest.approx(1.0)
    assert (E.sequence_probabilities(x, 0.5) == 2.0 ** -8).all()


def test_all_sequences_bit_order_is_most_significant_first():
    assert E.all_sequences(3).tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]


def test_invalid_arguments_raise():
    for fn, args in ((E.exact_table, (5, 5, 0.5)), (E.exact_table, (5, 0, 0.5)), (E.exact_pmf, (4, 4, 0.5)), (E.all_sequences, (0,)), (E.all_sequences, (21,))):
        with pytest.raises(ValueError):
            fn(*args)
