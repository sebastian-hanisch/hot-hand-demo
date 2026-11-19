"""Exakte Erwartungen ohne Simulation: Rekursion über den Zustand (für alle Längen bis n_max in einem Durchgang) und Aufzählung aller 2^n Folgen (n <= 16).

Beide rechnen die Verteilung der Paare (O, H) einer Folge: O = Zahl der Gelegenheiten (vor dem Wurf standen k Treffer in Folge), H = Treffer an Gelegenheiten.
Daraus folgen, jeweils bedingt auf O >= 1 (nur dort ist der Schätzer H / O definiert):
  mean_est = E[H / O | O >= 1]  (die Erwartung des Folgenschätzers; das Mittel der Folgenschätzer über viele Folgen nähert sich ihr an),
  und unbedingt E[O], E[H] mit dem Verhältnis E[H] / E[O] (das, wogegen der gepoolte Schätzer bei vielen Folgen strebt).
Modell: symmetrische Markow-Kette mit Persistenz q (hoh_scenario)."""

from dataclasses import dataclass

import numpy as np

from hoh_algorithm import opportunity_counts


@dataclass(frozen=True)
class ExactStats:
    n: int
    k: int
    q: float
    p_defined: float    # P(O >= 1)
    mean_est: float     # E[H / O | O >= 1]; nan, wenn O >= 1 unmöglich ist
    e_o: float          # E[O]
    e_h: float          # E[H]

    @property
    def ratio(self) -> float:
        """E[H] / E[O]: der Grenzwert des gepoolten Schätzers bei vielen Folgen (nan ohne mögliche Gelegenheit)."""
        return self.e_h / self.e_o if self.e_o > 0 else float("nan")


def expected_opportunities(n: int, k: int, q: float) -> float:
    """Geschlossene Formel: E[O] = (n - k) · q^(k-1) / 2 (jede der n - k Positionen ist mit Wahrscheinlichkeit q^(k-1)/2 eine Gelegenheit)."""
    return (n - k) * q ** (k - 1) / 2.0


def expected_hits(n: int, k: int, q: float) -> float:
    """Geschlossene Formel: E[H] = q · E[O] (an jeder Gelegenheit folgt ein Treffer mit Wahrscheinlichkeit q, gleich welche Vorgeschichte)."""
    return q * expected_opportunities(n, k, q)


def _transition_step(S: np.ndarray, k: int, q: float, olim: int) -> np.ndarray:
    """Ein Wurf der Rekursion. S[r, o, h]: Wahrscheinlichkeit, dass die Folge bisher o Gelegenheiten und h Treffer an Gelegenheiten hat und mit r = min(Länge des
    Treffer-Laufs am Ende, k) endet. Eine Gelegenheit entsteht beim nächsten Wurf genau dann, wenn r = k."""
    new = np.zeros_like(S)
    for r in range(k + 1):
        last_hit = r > 0
        p1 = q if last_hit else 1.0 - q          # P(nächster Wurf Treffer | letzter Wurf)
        p0 = 1.0 - p1
        src = S[r, :olim, :olim]
        if r == k:                               # Gelegenheit: o + 1, bei Treffer auch h + 1
            new[0, 1:olim + 1, 0:olim] += p0 * src
            new[k, 1:olim + 1, 1:olim + 1] += p1 * src
        else:
            new[0, :olim, :olim] += p0 * src
            new[r + 1, :olim, :olim] += p1 * src
    return new


def exact_table(n_max: int, k: int, q: float) -> dict:
    """Für jede Länge n = k + 1 .. n_max: ExactStats (aus einer einzigen Rekursion). Rückgabe {n: ExactStats}."""
    if not 1 <= k < n_max:
        raise ValueError("1 <= k < n_max verlangt")
    L = n_max - k + 2
    S = np.zeros((k + 1, L, L))
    S[0, 0, 0] = 0.5                              # erster Wurf: Fehlwurf (r = 0) ...
    S[min(1, k), 0, 0] += 0.5                     # ... oder Treffer (r = 1, bei k = 1 gleich k)
    oo, hh = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    ratio = np.where(oo > 0, hh / np.maximum(oo, 1), 0.0)
    out = {}
    for n in range(2, n_max + 1):
        olim = min(n - 1 - k + 1, L - 1) if n - 1 >= k else 1   # nach n - 1 Würfen höchstens n - 1 - k Gelegenheiten
        S = _transition_step(S, k, q, max(olim, 1))
        if n > k:
            tot = S.sum(axis=0)
            p_def = float(tot[1:, :].sum())
            e_o = float((tot * oo).sum())
            e_h = float((tot * hh).sum())
            mean_est = float((tot * ratio).sum() / p_def) if p_def > 0 else float("nan")
            out[n] = ExactStats(n, k, q, p_def, mean_est, e_o, e_h)
    return out


def exact_pmf(n: int, k: int, q: float) -> np.ndarray:
    """Gemeinsame Verteilung von (O, H) einer Folge der Länge n als Matrix pmf[o, h] aus der Rekursion (Summe 1)."""
    if not 1 <= k < n:
        raise ValueError("1 <= k < n verlangt")
    L = n - k + 2
    S = np.zeros((k + 1, L, L))
    S[0, 0, 0] = 0.5
    S[min(1, k), 0, 0] += 0.5
    for t in range(2, n + 1):
        S = _transition_step(S, k, q, max(min(t - k, L - 1), 1))
    return S.sum(axis=0)


def all_sequences(n: int) -> np.ndarray:
    """Alle 2^n Folgen als 0/1-Array (2^n, n); Zeile i ist die Binärdarstellung von i (höchstes Bit zuerst)."""
    if not 1 <= n <= 20:
        raise ValueError("n zwischen 1 und 20 verlangt")
    idx = np.arange(1 << n, dtype=np.int64)[:, None]
    return ((idx >> np.arange(n - 1, -1, -1, dtype=np.int64)[None, :]) & 1).astype(np.uint8)


def sequence_probabilities(x: np.ndarray, q: float) -> np.ndarray:
    """Wahrscheinlichkeit jeder Folge in x (Zeilen): 1/2 für den ersten Wurf, dann q je gleichem und 1 - q je wechselndem Wurf."""
    same = x[:, 1:] == x[:, :-1]
    return 0.5 * np.prod(np.where(same, q, 1.0 - q), axis=1)


def enumerate_stats(n: int, k: int, q: float) -> ExactStats:
    """Die Erwartungen durch Aufzählung aller 2^n Folgen (nur für kleine n) mit den Gelegenheiten aus opportunity_counts (Fenstersummen, nicht die Rekursion)."""
    x = all_sequences(n)
    w = sequence_probabilities(x, q)
    O, H = opportunity_counts(x, k)
    defined = O > 0
    p_def = float(w[defined].sum())
    mean_est = float((w[defined] * (H[defined] / O[defined])).sum() / p_def) if p_def > 0 else float("nan")
    return ExactStats(n, k, q, p_def, mean_est, float((w * O).sum()), float((w * H).sum()))
