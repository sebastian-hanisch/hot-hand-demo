"""Orakel (anderer Rechenweg, ORACLE-PLAYBOOK §2 und §5): Der Kern der Demo rechnet die Erwartung des Schätzers über Fenstersummen (Aufzählung) und über eine Rekursion
über den Zustand (hoh_exact). Hier steht ein dritter Weg ohne numpy und ohne Module der Demo: reine Python-Schleifen mit exakten Brüchen (`fractions`), Zählen über Teilfolgen.

Geprüft werden (1) die exakte Erwartung des Schätzers, E[O], E[H] und die geschlossenen Formeln gegen die Bruchrechnung, (2) die Simulation gegen die exakte Erwartung mit
vier Standardfehlern Abstand (feste Zufalls-Seeds), (3) die Aussagen zur Zahl der Folgen (Mittel der Folgenschätzer unabhängig von m, gepoolt nähert sich q) durch Aufzählung aller
Folgenpaare und -tripel, (4) der Permutationstest: der p-Wert der Demo gegen den exakten p-Wert aus allen Anordnungen, und seine Gültigkeit (Irrtumswahrscheinlichkeit höchstens 5 %)
durch Aufzählung aller Folgen."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

import hoh_algorithm as A
import hoh_exact as E
import hoh_scenario as S


# --- der Orakel-Weg: reine Schleifen, exakte Brüche -------------------------------------------------------------------------------------------

def opp_by_slices(seq, k):
    """(O, H) durch Vergleich der k Würfe davor mit k Einsen (Teilfolgen statt Fenstersummen oder Zustand)."""
    O = H = 0
    for t in range(k, len(seq)):
        if tuple(seq[t - k:t]) == (1,) * k:
            O += 1
            H += seq[t]
    return O, H


def weight(seq, q):
    """Wahrscheinlichkeit der Folge: 1/2 für den ersten Wurf, q je gleichem und 1 - q je wechselndem Übergang."""
    same = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    return F(1, 2) * q ** same * (1 - q) ** (len(seq) - 1 - same)


def exact_fractions(n, k, q):
    """(P(O>=1), E[H/O | O>=1], E[O], E[H]) als Brüche durch Aufzählung aller 2^n Folgen."""
    pdef = num = e_o = e_h = F(0)
    for seq in itertools.product((0, 1), repeat=n):
        w = weight(seq, q)
        O, H = opp_by_slices(seq, k)
        e_o += w * O
        e_h += w * H
        if O > 0:
            pdef += w
            num += w * F(H, O)
    return pdef, num / pdef, e_o, e_h


Q_GRID = (F(1, 2), F(3, 5), F(1, 3), F(4, 5))


def test_the_oracle_itself_on_the_hand_examples():
    """Das Orakel wird zuerst an den Handwerten geprüft: n = 3, k = 1 (5/12) und q = 3/5 (18/35), n = 4, k = 2 (5/12)."""
    assert exact_fractions(3, 1, F(1, 2))[1] == F(5, 12)
    assert exact_fractions(3, 1, F(3, 5))[1] == F(18, 35)
    assert exact_fractions(4, 2, F(1, 2))[1] == F(5, 12)
    assert exact_fractions(3, 1, F(1, 2))[0] == F(3, 4)


@pytest.mark.parametrize("q", Q_GRID)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_recursion_and_enumeration_equal_the_fraction_enumeration(k, q):
    table = E.exact_table(11, k, float(q))
    for n in range(k + 1, 12):
        pdef, mean_est, e_o, e_h = exact_fractions(n, k, q)
        assert float(pdef) == pytest.approx(table[n].p_defined, abs=1e-12), (n, k, q)
        assert float(mean_est) == pytest.approx(table[n].mean_est, abs=1e-12), (n, k, q)
        assert float(e_o) == pytest.approx(table[n].e_o, abs=1e-12) and float(e_h) == pytest.approx(table[n].e_h, abs=1e-12)
        if n <= 10:
            en = E.enumerate_stats(n, k, float(q))
            assert float(mean_est) == pytest.approx(en.mean_est, abs=1e-12)


@pytest.mark.parametrize("q", Q_GRID)
def test_closed_forms_hold_exactly_in_fractions(q):
    """E[O] = (n-k)/2 · q^(k-1) und E[H] = q · E[O] gelten als Bruch ohne Rest (nicht nur auf Rundung)."""
    for n, k in [(6, 1), (7, 2), (8, 3), (9, 2), (5, 4)]:
        _, _, e_o, e_h = exact_fractions(n, k, q)
        assert e_o == F(n - k, 2) * q ** (k - 1)
        assert e_h == q * e_o


def test_the_fair_coin_bias_is_real_exactly_in_fractions():
    """Der Kern der Seite als Bruch: Bei fairer Münze liegt E[H/O | O>=1] für n = 8 und k = 1, 2, 3 echt unter 1/2, aber E[H]/E[O] ist genau 1/2."""
    for k in (1, 2, 3):
        _, mean_est, e_o, e_h = exact_fractions(8, k, F(1, 2))
        assert mean_est < F(1, 2) and e_h / e_o == F(1, 2)


def test_simulation_agrees_with_the_exact_expectation_within_four_standard_errors():
    for n, k, q in [(20, 3, 0.5), (12, 2, 0.7), (30, 1, 0.4)]:
        exact = E.exact_table(n, k, q)[n].mean_est
        vals = []
        for sd in (11, 23, 37, 41, 59):
            x = S.make_sequences(4000, n, q, S.derive_seed(sd, 99, n))
            O, H = A.opportunity_counts(x, k)
            e = A.estimates(O, H)
            vals.extend(e[~np.isnan(e)].tolist())
        vals = np.array(vals)
        se = vals.std(ddof=1) / math.sqrt(len(vals))
        assert abs(vals.mean() - exact) < 4 * se, (n, k, q, vals.mean(), exact, se)


def _pair_stats(n, k, q, m):
    """E[Mittel der Folgenschätzer] und E[gepoolt], jeweils bedingt darauf, dass es überhaupt eine Gelegenheit gibt, durch Aufzählung aller m-Tupel von Folgen."""
    seqs = list(itertools.product((0, 1), repeat=n))
    w = [weight(s, q) for s in seqs]
    oh = [opp_by_slices(s, k) for s in seqs]
    den = num_mean = num_pool = F(0)
    for idx in itertools.product(range(len(seqs)), repeat=m):
        p = F(1)
        for i in idx:
            p *= w[i]
        d = [oh[i] for i in idx if oh[i][0] > 0]
        if not d:
            continue
        den += p
        num_mean += p * sum(F(h, o) for o, h in d) / len(d)
        num_pool += p * F(sum(h for _, h in d), sum(o for o, _ in d))
    return num_mean / den, num_pool / den


def test_more_folgen_do_not_change_the_mean_of_estimators_but_pooling_approaches_q():
    """n = 5, k = 1, faire Münze, m = 1, 2, 3 Folgen (alle 32^m Tupel): Das Mittel der Folgenschätzer hat für jedes m genau dieselbe Erwartung (Bruch gleich),
    der gepoolte Schätzer liegt bei m = 1 gleich, dann näher an 1/2 (Beträge der Abweichung fallen)."""
    stats = [_pair_stats(5, 1, F(1, 2), m) for m in (1, 2, 3)]
    assert stats[0][0] == stats[1][0] == stats[2][0] == stats[0][1]
    dev = [abs(p - F(1, 2)) for _, p in stats]
    assert dev[0] > dev[1] > dev[2] > 0


# --- Permutationstest: exakter p-Wert und Gültigkeit ----------------------------------------------------------------------------------------

def stat_single(seq, k):
    O, H = opp_by_slices(seq, k)
    return F(H, O) if O else None


def perm_distribution(n, h, k):
    """Verteilung des Schätzers über alle C(n, h) Anordnungen von h Treffern in n Würfen (Zähler je Wert; None = keine Gelegenheit)."""
    dist = {}
    for pos in itertools.combinations(range(n), h):
        arr = [0] * n
        for i in pos:
            arr[i] = 1
        t = stat_single(arr, k)
        dist[t] = dist.get(t, 0) + 1
    return dist


def p_values_from(dist, t_obs):
    """Exakter einseitiger p-Wert nach oben und unten aus der Verteilung: Anteil der Anordnungen mit mindestens bzw. höchstens so extremem Wert."""
    if t_obs is None:
        return F(1), F(1)
    total = sum(dist.values())
    hi = sum(c for t, c in dist.items() if t is not None and t >= t_obs)
    lo = sum(c for t, c in dist.items() if t is not None and t <= t_obs)
    return F(hi, total), F(lo, total)


def exact_perm_p(seq, k):
    return p_values_from(perm_distribution(len(seq), sum(seq), k), stat_single(seq, k))


def test_the_permutation_p_value_matches_the_exact_p_value_within_sampling_error():
    """Je Folge 4 Standardfehler (Binomial) mit B = 999 Umordnungen; die Demo rechnet (1 + Zahl) / (B + 1), also bis zu 1/(B+1) höher."""
    B = 999
    for seq, k in [((1, 1, 1, 0, 0, 1, 0, 1, 1, 0), 2), ((0, 1, 1, 1, 1, 0, 0, 0, 1, 0), 1), ((1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1), 2)]:
        p_hi, p_lo = (float(v) for v in exact_perm_p(seq, k))
        res = A.permutation_test(np.array([seq], dtype=np.uint8), k, B, 17)
        for exact, got in ((p_hi, res["p_upper"]), (p_lo, res["p_lower"])):
            tol = 4 * math.sqrt(exact * (1 - exact) / B) + 1 / (B + 1)
            assert abs(got - exact) <= tol, (seq, k, exact, got)


def test_the_exact_permutation_test_has_size_at_most_five_percent_for_the_fair_coin():
    """Alle 2^12 Folgen einer fairen Münze (je 2^-12), gruppiert nach Trefferzahl h und Wert des Schätzers: Die Summe der Wahrscheinlichkeiten der Folgen mit exaktem p ≤ 0,05 ist
    höchstens 0,05, in beiden Richtungen (die Gültigkeit des Verfahrens, an das die Demo mit Zufallsumordnungen herankommt)."""
    n, k = 12, 2
    size_hi = size_lo = F(0)
    for h in range(n + 1):
        dist = perm_distribution(n, h, k)
        for t, count in dist.items():                           # genau count Folgen mit h Treffern haben diesen Wert
            p_hi, p_lo = p_values_from(dist, t)
            if p_hi <= F(1, 20):
                size_hi += F(count, 2 ** n)
            if p_lo <= F(1, 20):
                size_lo += F(count, 2 ** n)
    assert 0 < size_hi <= F(1, 20) and 0 < size_lo <= F(1, 20)
