"""Szenario: Folgen von Würfen (Treffer 1, Fehlwurf 0) aus einer Münze mit Persistenz q und der dafür nötige, plattformfeste Zufall.

Modell: Der erste Wurf ist ein Treffer mit Wahrscheinlichkeit 1/2. Jeder weitere Wurf geht mit Wahrscheinlichkeit q wie sein Vorgänger aus, sonst
umgekehrt (symmetrische Markow-Kette). Dann ist jeder Wurf einzeln ein Treffer mit Wahrscheinlichkeit 1/2, und die wahre Trefferquote nach einem
Treffer ist q, gleich nach wie vielen Treffern davor. q = 1/2 ist die faire Münze ohne Gedächtnis (unabhängige Würfe).

Zufall: der Strom von SplitMix64 (hoh_rng.SplitMix64), hier als Zähler-Variante in numpy: Die i-te Ausgabe ist mix(Zufalls-Seed + i · γ), also genau die i-te Ausgabe
des Generators mit Zustand; das erlaubt, ganze Blöcke ohne Python-Schleife zu ziehen und bleibt auf jeder Plattform bitgleich (reine Ganzzahlarithmetik)."""

import numpy as np

from hoh_rng import MASK, SplitMix64

GAMMA = 0x9E3779B97F4A7C15
_U = np.uint64


def derive_seed(base: int, *parts: int) -> int:
    """Teil-Zufalls-Seed aus einem Zufalls-Seed und ganzen Zahlen (kein hash(): auf jeder Plattform und in jedem Prozess gleich)."""
    s = int(base) & MASK
    for p in parts:
        s = SplitMix64((s + (int(p) + 1) * GAMMA) & MASK).next_u64()
    return s


def uniform_stream(seed: int, count: int) -> np.ndarray:
    """Die ersten `count` Ausgaben von SplitMix64(seed).random(), vektorisiert (float64 in [0, 1) mit 53 Bit)."""
    i = np.arange(1, count + 1, dtype=np.uint64)
    z = _U(int(seed) & MASK) + i * _U(GAMMA)               # Überlauf modulo 2^64 gewollt (Arrays warnen nicht)
    z = (z ^ (z >> _U(30))) * _U(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U(27))) * _U(0x94D049BB133111EB)
    z = z ^ (z >> _U(31))
    return (z >> _U(11)).astype(np.float64) / float(1 << 53)


def make_sequences(m: int, n: int, q: float, seed: int) -> np.ndarray:
    """m Folgen der Länge n als 0/1-Array (m, n). Zufallszahl [i, 0] bestimmt den ersten Wurf (< 1/2: Treffer), [i, t] den Wechsel (>= q: anderes Ergebnis wie davor)."""
    if m < 1 or n < 1 or not 0.0 <= q <= 1.0:
        raise ValueError("m, n >= 1 und 0 <= q <= 1 verlangt")
    u = uniform_stream(seed, m * n).reshape(m, n)
    first = (u[:, 0] < 0.5).astype(np.uint8)
    flips = (u[:, 1:] >= q).astype(np.uint8)
    parity = np.concatenate([np.zeros((m, 1), dtype=np.uint8), np.cumsum(flips, axis=1, dtype=np.int64).astype(np.uint8) & 1], axis=1)
    return first[:, None] ^ parity
