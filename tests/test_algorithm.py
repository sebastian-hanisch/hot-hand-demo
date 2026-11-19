"""Kern gegen Handrechnung: Gelegenheiten, Schätzer, Permutationsumordnung, naive Prüfung (DEMO-PLAYBOOK §1b)."""

import itertools
import random

import numpy as np
import pytest

import hoh_algorithm as A


def test_opportunities_on_the_mini_fold(mini):
    assert A.opportunity_positions(mini, 2) == [2, 3, 6]
    O, H = A.opportunity_counts(np.array(mini), 2)
    assert (int(O), int(H)) == (3, 2)


def test_k_one_by_hand():
    # 0 1 1 0 1: Gelegenheiten bei t = 2 (davor 1, Treffer) und t = 3 (davor 1, Fehlwurf); t = 1 und t = 4 haben davor eine 0
    x = np.array([0, 1, 1, 0, 1])
    assert A.opportunity_positions(x, 1) == [2, 3]
    O, H = A.opportunity_counts(x, 1)
    assert (int(O), int(H)) == (2, 1)


def test_no_opportunity_when_no_run_of_k_exists():
    O, H = A.opportunity_counts(np.array([1, 0, 1, 0, 1, 0]), 2)
    assert (int(O), int(H)) == (0, 0)
    assert np.isnan(A.estimates(O, H)).all()


def test_all_hits_every_position_after_k_is_a_hit_opportunity():
    O, H = A.opportunity_counts(np.ones((1, 10), dtype=int), 3)
    assert (int(O[0]), int(H[0])) == (7, 7)


def test_vectorised_counts_equal_the_loop_on_random_sequences():
    rng = random.Random(5)
    for _ in range(300):
        n = rng.randint(2, 30)
        k = rng.randint(1, n - 1)
        seq = [rng.randint(0, 1) for _ in range(n)]
        pos = A.opportunity_positions(seq, k)
        O, H = A.opportunity_counts(np.array(seq), k)
        assert int(O) == len(pos) and int(H) == sum(seq[t] for t in pos), (seq, k)


def test_counts_work_on_a_batch_axis():
    x = np.array([[1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 0]])
    O, H = A.opportunity_counts(x, 2)
    assert O.tolist() == [3, 1] and H.tolist() == [2, 0]   # zweite Folge 0 1 1 0 1 0 0: nur t = 3 (davor 1 1) ist Gelegenheit, dort ein Fehlwurf


@pytest.mark.parametrize("bad_k", [0, -1, 7, 8])
def test_invalid_k_raises(bad_k):
    with pytest.raises(ValueError):
        A.opportunity_counts(np.zeros((1, 7), dtype=int), bad_k)
    with pytest.raises(ValueError):
        A.opportunity_positions([0] * 7, bad_k)


def test_estimates_mean_and_pooled_by_hand():
    O, H = np.array([2, 0, 1]), np.array([1, 0, 0])
    e = A.estimates(O, H)
    assert e[0] == 0.5 and np.isnan(e[1]) and e[2] == 0.0
    assert A.mean_of_estimators(O, H) == pytest.approx(0.25)          # (1/2 + 0) / 2: die Folge ohne Gelegenheit fällt weg
    assert A.pooled_estimate(O, H) == pytest.approx(1 / 3)            # 1 Treffer an 3 Gelegenheiten


def test_undefined_everywhere_gives_nan():
    O, H = np.zeros(4, dtype=int), np.zeros(4, dtype=int)
    assert np.isnan(A.mean_of_estimators(O, H)) and np.isnan(A.pooled_estimate(O, H))


def test_mean_and_pooled_differ_when_folgen_have_different_numbers_of_opportunities():
    O, H = np.array([1, 4]), np.array([1, 0])
    assert A.mean_of_estimators(O, H) == pytest.approx(0.5) and A.pooled_estimate(O, H) == pytest.approx(0.2)


def test_statistic_works_per_repetition():
    O = np.array([[2, 0, 1], [0, 0, 0]])
    H = np.array([[1, 0, 0], [0, 0, 0]])
    t = A.statistic(O, H)
    assert t[0] == pytest.approx(0.25) and np.isnan(t[1])


def test_permute_sequences_keeps_every_row_sum_and_shape():
    x = np.array([[1, 1, 0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    p = A.permute_sequences(x, 50, 3)
    assert p.shape == (50, 3, 8)
    assert (p.sum(axis=-1) == x.sum(axis=-1)).all()
    assert (p[:, 2, :] == 1).all()                                            # eine Folge aus lauter Treffern bleibt es
    assert len({tuple(row) for row in p[:, 1, :].tolist()}) > 3               # die Einzelfolge wird wirklich umgeordnet (8 mögliche Lagen des einen Treffers)


def test_permutations_are_reproducible_and_depend_on_the_seed():
    x = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 1]], dtype=np.uint8)
    assert (A.permute_sequences(x, 20, 1) == A.permute_sequences(x, 20, 1)).all()
    assert not (A.permute_sequences(x, 20, 1) == A.permute_sequences(x, 20, 2)).all()


def test_every_arrangement_appears_for_a_tiny_sequence():
    """Vier Würfe mit zwei Treffern: alle C(4,2) = 6 Anordnungen kommen in 600 Umordnungen vor."""
    x = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    seen = {tuple(r) for r in A.permute_sequences(x, 600, 9)[:, 0, :].tolist()}
    assert seen == {tuple(s) for s in itertools.product((0, 1), repeat=4) if sum(s) == 2}


def test_permutation_test_on_a_sequence_with_no_opportunity_is_undefined():
    res = A.permutation_test(np.array([[1, 0, 1, 0, 1, 0]]), 2, 99, 1)
    assert res["p_upper"] == 1.0 and res["p_lower"] == 1.0 and np.isnan(res["t_obs"])


def test_permutation_test_all_hits_is_not_significant():
    """Nur Treffer: jede Umordnung ist dieselbe Folge, der echte Wert ist nicht extremer als jeder andere: p = 1 in beiden Richtungen."""
    res = A.permutation_test(np.ones((2, 12), dtype=np.uint8), 2, 99, 1)
    assert res["p_upper"] == 1.0 and res["p_lower"] == 1.0 and res["t_obs"] == 1.0


def test_permutation_test_flags_one_long_run_as_hot():
    """Zwei Treffer-Läufe der Länge 12 in 24 Würfen sind unter Umordnung selten (der echte Schätzer liegt weit oben)."""
    x = np.array([[1] * 12 + [0] * 12], dtype=np.uint8)
    res = A.permutation_test(x, 2, 999, 4)
    assert res["t_obs"] == pytest.approx(10 / 11) and res["p_upper"] <= 0.01 and res["p_lower"] > 0.9


def test_permutation_p_values_stay_in_range_and_have_the_smallest_possible_value():
    x = np.array([[1] * 12 + [0] * 12], dtype=np.uint8)
    res = A.permutation_test(x, 2, 99, 4)
    assert 1 / 100 <= res["p_upper"] <= 1.0 and 1 / 100 <= res["p_lower"] <= 1.0


def test_permutation_test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        A.permutation_test(np.array([1, 0, 1, 1]), 1, 9, 1)


def test_naive_test_by_hand():
    # zwei Folgen mit je 4 Gelegenheiten, alle Treffer: T = 1, SE = sqrt(2 · 0,25 / 4) / 2, z = 0,5 / SE
    res = A.naive_test(np.array([4, 4]), np.array([4, 4]))
    se = (2 * 0.25 / 4) ** 0.5 / 2
    assert res["t"] == 1.0 and res["z"] == pytest.approx(0.5 / se) and res["hot"] and not res["cold"]
    cold = A.naive_test(np.array([4, 4]), np.array([0, 0]))
    assert cold["z"] == pytest.approx(-0.5 / se) and cold["cold"] and not cold["hot"]


def test_naive_test_threshold_and_empty_case():
    mid = A.naive_test(np.array([100]), np.array([50]))
    assert mid["z"] == 0.0 and not mid["hot"] and not mid["cold"]
    none = A.naive_test(np.array([0, 0]), np.array([0, 0]))
    assert np.isnan(none["z"]) and not none["hot"] and not none["cold"]
