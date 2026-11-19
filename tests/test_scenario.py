"""Szenario: der vektorisierte Zufall gegen den Generator mit Zustand, die Folgen gegen eine Schleifenfassung, Eigenschaften der Münze."""

import pytest

from hoh_rng import SplitMix64
from hoh_scenario import derive_seed, make_sequences, uniform_stream


@pytest.mark.parametrize("seed", [0, 1, 35, 123456789, 2**63 + 5, 2**64 - 1])
def test_uniform_stream_equals_the_sequential_generator_bit_for_bit(seed):
    rng = SplitMix64(seed)
    assert [rng.random() for _ in range(200)] == list(uniform_stream(seed, 200))


def test_uniform_stream_is_in_the_unit_interval_and_roughly_uniform():
    u = uniform_stream(7, 100000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.005  # Standardfehler 0,0009


def test_derive_seed_is_deterministic_and_separates_parts():
    assert derive_seed(5, 1, 2) == derive_seed(5, 1, 2)
    assert len({derive_seed(5, 1, 2), derive_seed(5, 2, 1), derive_seed(5, 1), derive_seed(6, 1, 2), derive_seed(5, 1, 3)}) == 5


def _loop_sequences(m, n, q, seed):
    """Schleifenfassung ohne numpy: erster Wurf aus der ersten Zufallszahl, jeder weitere wechselt, wenn die Zufallszahl mindestens q ist."""
    rng = SplitMix64(seed)
    out = []
    for _ in range(m):
        row = []
        for t in range(n):
            u = rng.random()
            row.append(1 if u < 0.5 else 0) if t == 0 else row.append(row[-1] if u < q else 1 - row[-1])
        out.append(row)
    return out


@pytest.mark.parametrize("m,n,q,seed", [(1, 5, 0.5, 1), (4, 9, 0.7, 2), (6, 20, 0.3, 3), (3, 1, 0.9, 4), (7, 12, 0.5, 99)])
def test_make_sequences_matches_the_loop_version(m, n, q, seed):
    assert make_sequences(m, n, q, seed).tolist() == _loop_sequences(m, n, q, seed)


def test_extreme_persistence_gives_constant_and_alternating_rows():
    const = make_sequences(50, 12, 1.0, 5)
    assert all(len(set(row)) == 1 for row in const.tolist())
    alt = make_sequences(50, 12, 0.0, 5)
    assert all(row[t] != row[t + 1] for row in alt.tolist() for t in range(11))


def test_fair_coin_is_independent_and_persistent_coin_repeats_with_probability_q():
    x = make_sequences(20000, 20, 0.5, 7).astype(float)
    assert abs(x.mean() - 0.5) < 0.004                                   # 400 000 Würfe, Standardfehler 0,0008 (mit Abhängigkeit nur q = 0,5: unabhängig)
    assert abs((x[:, 1:] * x[:, :-1]).mean() - 0.25) < 0.004             # P(zwei Treffer in Folge) = 1/4
    y = make_sequences(20000, 20, 0.7, 7)
    assert abs((y[:, 1:] == y[:, :-1]).mean() - 0.7) < 0.004
    assert abs(y.mean() - 0.5) < 0.006                                   # die Einzelquote bleibt 1/2, auch bei q = 0,7


def test_same_arguments_same_sequences_other_seed_other_sequences():
    a = make_sequences(10, 30, 0.6, 11)
    assert (a == make_sequences(10, 30, 0.6, 11)).all()
    assert not (a == make_sequences(10, 30, 0.6, 12)).all()


def test_invalid_arguments_raise():
    for args in ((0, 5, 0.5, 1), (3, 0, 0.5, 1), (3, 5, -0.1, 1), (3, 5, 1.1, 1)):
        with pytest.raises(ValueError):
            make_sequences(*args)
