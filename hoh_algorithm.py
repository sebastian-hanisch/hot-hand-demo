"""Kern: Gelegenheiten, Schätzer und Permutationstest (numpy). Jede Regel ist eine eigene Funktion (DEMO-PLAYBOOK §1b).

Begriffe (0-basierte Positionen t = 0 .. n-1 in einer Folge x):
* Gelegenheit zu k: Position t >= k, vor der die k Würfe x[t-k .. t-1] alle Treffer waren.
* O = Zahl der Gelegenheiten einer Folge, H = Zahl der Treffer an Gelegenheiten.
* Schätzer je Folge: H / O (nur definiert, wenn O >= 1).
* Mittel der Folgenschätzer: Mittel von H / O über die Folgen, in denen er definiert ist (das Maß dieser Seite).
* gepoolter Schätzer: Summe H geteilt durch Summe O über alle Folgen zusammen.
Die exakte Erwartung steht in hoh_exact.py (Rekursion und Aufzählung)."""

import numpy as np

from hoh_scenario import uniform_stream

TOL = 1e-12  # Vergleiche von Mittelwerten gleicher Brüche in verschiedener Summationsreihenfolge


def _check_k(n: int, k: int) -> None:
    if not 1 <= k < n:
        raise ValueError(f"k = {k} muss zwischen 1 und n - 1 = {n - 1} liegen")


def opportunity_counts(x, k: int):
    """(O, H) für jede Folge der letzten Achse von x (0/1, Form (..., n)): Gelegenheiten und Treffer an Gelegenheiten."""
    x = np.asarray(x)
    n = x.shape[-1]
    _check_k(n, k)
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,), dtype=np.int32), np.cumsum(x, axis=-1, dtype=np.int32)], axis=-1)   # c[j] = Treffer unter den ersten j Würfen
    opp = (c[..., k:n] - c[..., 0:n - k]) == k                                                             # Position t: die k Würfe davor sind Treffer
    hit = opp & (x[..., k:] == 1)
    return opp.sum(axis=-1), hit.sum(axis=-1)


def opportunity_positions(seq, k: int) -> list[int]:
    """Positionen der Gelegenheiten einer einzelnen Folge (einfache Schleife; die Anzeige im Beispiel und das Gegenstück zu opportunity_counts)."""
    seq = [int(v) for v in seq]
    _check_k(len(seq), k)
    return [t for t in range(k, len(seq)) if all(seq[t - k:t])]


def estimates(O, H) -> np.ndarray:
    """Schätzer H / O je Folge; nan, wo es keine Gelegenheit gab."""
    O = np.asarray(O, dtype=float)
    H = np.asarray(H, dtype=float)
    return np.where(O > 0, H / np.maximum(O, 1.0), np.nan)


def mean_of_estimators(O, H) -> float:
    """Mittel der Folgenschätzer über die Folgen mit mindestens einer Gelegenheit; nan, wenn keine Folge eine hat."""
    e = estimates(O, H)
    ok = ~np.isnan(e)
    return float(e[ok].mean()) if ok.any() else float("nan")


def pooled_estimate(O, H) -> float:
    """Gepoolter Schätzer: alle Gelegenheiten aller Folgen zusammen; nan ohne Gelegenheit."""
    o = float(np.sum(O))
    return float(np.sum(H)) / o if o > 0 else float("nan")


def statistic(O, H):
    """Prüfgröße des Permutationstests für Daten mit letzter Achse = Folgen: Mittel der Folgenschätzer, nan ohne definierte Folge."""
    e = estimates(O, H)
    ok = ~np.isnan(e)
    cnt = ok.sum(axis=-1)
    tot = np.where(ok, e, 0.0).sum(axis=-1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def permute_sequences(x, count: int, seed: int) -> np.ndarray:
    """`count` zufällige Umordnungen jeder Folge einzeln (gleiche Zahl der Treffer je Folge), Form (count, m, n); Zufallsschlüssel aus uniform_stream, sortiert."""
    x = np.asarray(x)
    m, n = x.shape
    keys = uniform_stream(seed, count * m * n).reshape(count, m, n)
    order = np.argsort(keys, axis=-1) + (np.arange(m) * n)[None, :, None]   # Spaltenindex der Umordnung, verschoben auf die Zeile der Folge in x.ravel()
    return np.ascontiguousarray(x).reshape(-1)[order]


def permutation_test(x, k: int, count: int, seed: int) -> dict:
    """Permutationstest der Prüfgröße T = Mittel der Folgenschätzer gegen „Reihenfolge innerhalb jeder Folge ist zufällig“.

    Einseitig nach oben (p_upper: Treffer folgt häufiger auf Treffer) und nach unten (p_lower). p = (1 + Zahl der Umordnungen, die mindestens so extrem sind) / (count + 1).
    Ist T bei den echten Daten nicht definiert, ist p = 1; eine Umordnung ohne definierte Folge zählt in keinem Ast als extrem. Unter „unabhängige Würfe mit je Folge
    eigener Trefferquote“ sind alle Umordnungen gleich wahrscheinlich, der Test hat dann höchstens die Irrtumswahrscheinlichkeit alpha."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x hat die Form (Folgen, Würfe)")
    O, H = opportunity_counts(x, k)
    t_obs = float(statistic(O, H))
    perms = permute_sequences(x, count, seed)
    Op, Hp = opportunity_counts(perms, k)
    t_perm = statistic(Op, Hp)
    valid = ~np.isnan(t_perm)
    if np.isnan(t_obs):
        p_up = p_lo = 1.0
    else:
        p_up = (1 + int(np.sum(valid & (t_perm >= t_obs - TOL)))) / (count + 1)
        p_lo = (1 + int(np.sum(valid & (t_perm <= t_obs + TOL)))) / (count + 1)
    null_mean = float(t_perm[valid].mean()) if valid.any() else float("nan")
    return {"t_obs": t_obs, "p_upper": p_up, "p_lower": p_lo, "null_mean": null_mean, "valid": int(valid.sum()), "count": count}


def naive_test(O, H, z_crit: float = 1.6448536269514722) -> dict:
    """Die naive Prüfung: Mittel der Folgenschätzer gegen 0,5 mit der Binomial-Näherung für den Standardfehler (jede Folge ein Anteil aus O Gelegenheiten, Varianz 1/(4·O)).

    z = (T - 0,5) / SE mit SE = sqrt(Summe 1/(4·O_i)) / Zahl der definierten Folgen; „heiß“ bei z > z_crit, „kalt“ bei z < -z_crit. Ohne definierte Folge: weder noch."""
    O = np.asarray(O, dtype=float)
    H = np.asarray(H, dtype=float)
    ok = O > 0
    d = int(ok.sum())
    if d == 0:
        return {"t": float("nan"), "z": float("nan"), "hot": False, "cold": False}
    t = float(np.mean(H[ok] / O[ok]))
    se = float(np.sqrt(np.sum(0.25 / O[ok]))) / d
    z = (t - 0.5) / se
    return {"t": t, "z": z, "hot": z > z_crit, "cold": z < -z_crit}
