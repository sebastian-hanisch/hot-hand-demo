"""Kennzahlen, Messreihen (Sweeps) und Verdikte als reine Funktionen: `verdict_*(...) -> (Art, Text)`, die App ruft `getattr(st, Art)(Text)`."""

from dataclasses import dataclass

import numpy as np

import hoh_algorithm as A
import hoh_constants as C
import hoh_exact as E
from hoh_scenario import derive_seed, make_sequences


def fmt(x: float, digits: int = 3) -> str:
    """Zahl mit deutschem Dezimalkomma; nan als „–“."""
    if x != x:
        return "–"
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")                      # „−0,000“ gibt es nicht
    return text.replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 1) -> str:
    return "–" if x != x else f"{100 * x:.{digits}f} %".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    k: int = C.DEFAULT_K
    q_pct: int = C.DEFAULT_Q_PCT
    m: int = C.DEFAULT_M
    seed: int = C.DEFAULT_SEED

    @property
    def q(self) -> float:
        return self.q_pct / 100.0


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: np.ndarray            # (m, n) 0/1
    O: np.ndarray            # Gelegenheiten je Folge
    H: np.ndarray            # Treffer an Gelegenheiten je Folge
    est: np.ndarray          # Schätzer je Folge (nan: keine Gelegenheit)
    mean_est: float          # Mittel der Folgenschätzer
    pooled: float            # gepoolter Schätzer
    n_defined: int           # Folgen mit mindestens einer Gelegenheit
    truth: float             # wahre Trefferquote nach einem Treffer: q
    exact: E.ExactStats      # exakte Erwartung bei diesen Reglern
    positions: tuple         # Gelegenheiten der ersten Folge (Beispiel)
    perm: dict               # Permutationstest der ersten Folge (Beispiel)


def analyse(settings: Settings, table=None) -> Analysis:
    s = settings
    x = make_sequences(s.m, s.n, s.q, derive_seed(s.seed, 1))
    O, H = A.opportunity_counts(x, s.k)
    est = A.estimates(O, H)
    exact = (E.exact_table(s.n, s.k, s.q) if table is None else table)[s.n]
    perm = A.permutation_test(x[:1], s.k, C.EXAMPLE_PERM_B, derive_seed(s.seed, 2))
    return Analysis(s, x, O, H, est, A.mean_of_estimators(O, H), A.pooled_estimate(O, H), int((O > 0).sum()), s.q, exact,
                    tuple(A.opportunity_positions(x[0], s.k)), perm)


# --- Messreihen ---------------------------------------------------------------------------------------------------------------------------

def sweep_length(k: int, q_pct: int, grid=C.N_SIM_GRID, seeds=C.SWEEP_SEEDS, per_seed=C.SIM_SEQUENCES_PER_SEED) -> list[dict]:
    """Je Länge: die exakte Erwartung des Folgenschätzers (Rekursion) und die Simulation (je Zufalls-Seed das Mittel der Folgenschätzer über `per_seed` Folgen)."""
    q = q_pct / 100.0
    table = E.exact_table(max(grid), k, q)
    rows = []
    for n in grid:
        vals = []
        for sd in seeds:
            x = make_sequences(per_seed, n, q, derive_seed(sd, 10, n))
            O, H = A.opportunity_counts(x, k)
            vals.append(A.mean_of_estimators(O, H))
        rows.append({"n": n, "exact": table[n].mean_est, "sim": float(np.mean(vals)), "low": float(min(vals)), "high": float(max(vals)),
                     "p_defined": table[n].p_defined, "ratio": table[n].ratio})
    return rows


def exact_curve(k: int, q_pct: int, n_max: int = C.N_MAX) -> list[tuple[int, float]]:
    """Exakte Erwartung des Folgenschätzers für jede Länge von k + 1 bis n_max (für die glatte Kurve)."""
    return [(n, s.mean_est) for n, s in E.exact_table(n_max, k, q_pct / 100.0).items()]


def sweep_k(n: int, q_pct: int) -> list[dict]:
    """Je k = K_MIN .. K_MAX bei Länge n: exakte Erwartung des Folgenschätzers und Anteil der Folgen mit mindestens einer Gelegenheit."""
    q = q_pct / 100.0
    rows = []
    for k in range(C.K_MIN, C.K_MAX + 1):
        s = E.exact_table(n, k, q)[n]
        rows.append({"k": k, "exact": s.mean_est, "p_defined": s.p_defined, "e_o": s.e_o, "ratio": s.ratio})
    return rows


def sweep_m(n: int, k: int, q_pct: int, grid=C.M_GRID, seeds=C.SWEEP_SEEDS, per_seed=C.SIM_SEQUENCES_PER_SEED) -> list[dict]:
    """Je Zahl der Folgen m: Mittel der Folgenschätzer und gepoolter Schätzer, gemittelt über Wiederholungen (je Zufalls-Seed per_seed // m Wiederholungen
    zu je m Folgen, also gleich viele Folgen für jedes m); Band = kleinster und größter Wert der Zufalls-Seeds. Wiederholungen ohne jede Gelegenheit zählen nicht."""
    q = q_pct / 100.0
    single = E.exact_table(n, k, q)[n]
    rows = []
    for m in grid:
        reps = max(per_seed // m, 1)
        mean_vals, pool_vals = [], []
        for sd in seeds:
            x = make_sequences(reps * m, n, q, derive_seed(sd, 20, m)).reshape(reps, m, n)
            O, H = A.opportunity_counts(x, k)
            t_mean = A.statistic(O, H)
            o_tot = O.sum(axis=1)
            t_pool = np.where(o_tot > 0, H.sum(axis=1) / np.maximum(o_tot, 1), np.nan)
            mean_vals.append(float(np.nanmean(t_mean)))
            pool_vals.append(float(np.nanmean(t_pool)))
        rows.append({"m": m, "mean": float(np.mean(mean_vals)), "mean_low": min(mean_vals), "mean_high": max(mean_vals),
                     "pooled": float(np.mean(pool_vals)), "pooled_low": min(pool_vals), "pooled_high": max(pool_vals),
                     "exact_single": single.mean_est, "ratio": single.ratio})
    return rows


def power_curve(n: int, k: int, m: int, q_grid=C.Q_GRID_PCT, seeds=C.SWEEP_SEEDS, reps=C.POWER_REPS_PER_SEED, count=C.PERM_B, alpha=C.ALPHA) -> list[dict]:
    """Teststärke der naiven Prüfung und des Permutationstests: je Persistenz q und Zufalls-Seed `reps` Läufe mit m Folgen der Länge n.

    „heiß“ = einseitige Entscheidung nach oben (Treffer folgt häufiger auf Treffer), „kalt“ = nach unten. Bei q = 50 ist „heiß“ die Fehlalarmquote."""
    rows = []
    for q_pct in q_grid:
        per = {"naive_hot": [], "perm_hot": [], "naive_cold": [], "perm_cold": []}
        for sd in seeds:
            c = dict.fromkeys(per, 0)
            for r in range(reps):
                s = derive_seed(sd, 30, q_pct, r)
                x = make_sequences(m, n, q_pct / 100.0, derive_seed(s, 1))
                O, H = A.opportunity_counts(x, k)
                nv = A.naive_test(O, H)
                pt = A.permutation_test(x, k, count, derive_seed(s, 2))
                c["naive_hot"] += bool(nv["hot"])
                c["naive_cold"] += bool(nv["cold"])
                c["perm_hot"] += pt["p_upper"] <= alpha
                c["perm_cold"] += pt["p_lower"] <= alpha
            for key in per:
                per[key].append(c[key] / reps)
        row = {"q_pct": q_pct, "runs": reps * len(seeds)}
        for key, vals in per.items():
            row[key] = float(np.mean(vals))
            row[key + "_low"] = float(min(vals))
            row[key + "_high"] = float(max(vals))
        rows.append(row)
    return rows


def first_q_reaching(rows: list[dict], key: str, target: float = C.POWER_TARGET):
    """Kleinster Persistenz-Wert (in Prozent) der Reihe, bei dem `key` mindestens `target` erreicht; None, wenn keiner im Raster."""
    for r in sorted(rows, key=lambda r: r["q_pct"]):
        if r["q_pct"] > 50 and r[key] >= target:
            return r["q_pct"]
    return None


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage: Liegt das Mittel der Folgenschätzer bei diesen Reglern merklich neben der wahren Quote?"""
    s = a.settings
    bias = a.exact.mean_est - a.truth
    if a.exact.p_defined == 0 or bias != bias:
        return "warning", "Bei dieser Länge und diesem k kann es keine Gelegenheit geben: Der Schätzer ist nicht definiert."
    head = (f"Bei n = {s.n} Würfen und k = {s.k} ist die **exakte Erwartung** des Schätzers einer Folge **{fmt(a.exact.mean_est)}**, die wahre Trefferquote nach {s.k} "
            f"Treffern ist **{fmt(a.truth, 2)}**: Verzerrung **{fmt(bias, 3)}**. Der gepoolte Schätzer strebt bei vielen Folgen gegen {fmt(a.exact.ratio)}.")
    if abs(bias) < 0.005:
        return "info", head + " Die Verzerrung liegt hier unter 0,005."
    return "warning", head + f" Das Mittel dieser {s.m} Folgen: {fmt(a.mean_est)} (gepoolt {fmt(a.pooled)}); eine Stichprobe streut um die Erwartung."


def verdict_length(rows: list[dict], truth: float, k: int, curve: list[tuple[int, float]]) -> tuple[str, str]:
    """Verdikt zur Länge: Endpunkte der Reihe, die Länge mit der größten Verzerrung (die Kurve ist nicht monoton) und die Zählung der Simulation gegen die exakte Erwartung."""
    first, last = rows[0], rows[-1]
    inside = sum(1 for r in rows if r["low"] <= r["exact"] <= r["high"])
    bias_first, bias_last = first["exact"] - truth, last["exact"] - truth
    n_worst, v_worst = max(curve, key=lambda nv: abs(nv[1] - truth))
    text = (f"k = {k}: Die exakte Erwartung liegt bei n = {first['n']} bei {fmt(first['exact'])}, bei n = {last['n']} bei {fmt(last['exact'])} (wahr {fmt(truth, 2)}); "
            f"die Verzerrung geht von {fmt(bias_first)} auf {fmt(bias_last)} zurück und ist bei n = {last['n']} nicht null. Am größten ist sie bei n = {n_worst} ({fmt(v_worst - truth)}); "
            f"bei n = k + 1 = {k + 1} ist der Schätzer unverzerrt. In {inside} von {len(rows)} simulierten Längen liegt die exakte Erwartung im Band der {len(C.SWEEP_SEEDS)} Zufalls-Seeds.")
    return ("warning" if abs(bias_last) >= 0.005 else "info"), text


def verdict_k(rows: list[dict], n: int, truth: float) -> tuple[str, str]:
    biases = [r["exact"] - truth for r in rows]
    k_worst = rows[int(np.argmin(biases))]["k"]
    return "info", (f"Bei n = {n}: Die Verzerrung ist {fmt(biases[0])} bei k = {rows[0]['k']} und {fmt(biases[-1])} bei k = {rows[-1]['k']}; am größten ist sie bei k = {k_worst} "
                    f"({fmt(min(biases))}). Mit k sinkt der Anteil der Folgen mit mindestens einer Gelegenheit von {pct(rows[0]['p_defined'])} auf {pct(rows[-1]['p_defined'])}.")


def verdict_m(rows: list[dict]) -> tuple[str, str]:
    first, last = rows[0], rows[-1]
    exact = first["exact_single"]
    flat = sum(1 for r in rows if r["mean_low"] <= exact <= r["mean_high"])
    return "warning", (f"Mehr Folgen verkleinern die Verzerrung des **Mittels der Folgenschätzer** nicht: bei m = {first['m']} {fmt(first['mean'])}, bei m = {last['m']} {fmt(last['mean'])} "
                       f"(exakte Erwartung {fmt(exact)}; sie liegt in {flat} von {len(rows)} Bändern der Zufalls-Seeds). Der **gepoolte Schätzer** geht von {fmt(first['pooled'])} bei m = {first['m']} "
                       f"auf {fmt(last['pooled'])} bei m = {last['m']}; der Grenzwert für sehr viele Folgen ist {fmt(first['ratio'])}.")


def verdict_power(rows: list[dict], n: int, k: int, m: int) -> tuple[str, str]:
    base = next(r for r in rows if r["q_pct"] == 50)
    naive_q, perm_q = first_q_reaching(rows, "naive_hot"), first_q_reaching(rows, "perm_hot")

    def where(q):
        return f"ab q = {fmt(q / 100, 2)}" if q is not None else f"im Raster bis q = {fmt(max(r['q_pct'] for r in rows) / 100, 2)} nicht"

    return "info", (f"n = {n}, k = {k}, {m} Folgen, je Wert {base['runs']} Läufe: Der **Permutationstest** erreicht {pct(C.POWER_TARGET, 0)} Teststärke {where(perm_q)}, die **naive Prüfung** {where(naive_q)}. "
                    f"Bei fairer Münze (q = 0,50) meldet der Permutationstest „heiß“ in {pct(base['perm_hot'])}, die naive Prüfung in {pct(base['naive_hot'])} der Läufe; "
                    f"„kalt“ meldet bei fairer Münze der Permutationstest in {pct(base['perm_cold'])}, die naive Prüfung in {pct(base['naive_cold'])}.")


def verdict_example(a: Analysis, step: int) -> tuple[str, str]:
    """Verdikt unter dem wachsenden Beispiel: Stand der ersten Folge nach `step` Würfen (ab step <= k noch keine Gelegenheit möglich)."""
    seq = a.x[0][:step]
    k = a.settings.k
    pos = [t for t in A.opportunity_positions(a.x[0], k) if t < step] if step > k else []
    o = len(pos)
    h = int(sum(seq[t] for t in pos))
    if o == 0:
        return "info", f"Nach {step} {'Wurf' if step == 1 else 'Würfen'} gibt es noch keine Gelegenheit (vor ihr müssen {k} Treffer in Folge stehen): Der Schätzer ist noch nicht definiert."
    return "info", f"Nach {step} Würfen: {o} Gelegenheiten, davon {h} Treffer, Schätzer {h}/{o} = {fmt(h / o)}."


def verdict_reference(a: Analysis) -> tuple[str, str]:
    """Zum wachsenden Beispiel: Was ist der Referenzwert dieser einen Folge unter zufälliger Umordnung (Permutationstest)? Die Zahlen kommen aus a.perm."""
    p = a.perm
    seq = a.x[0]
    h, n = int(seq.sum()), len(seq)
    if p["t_obs"] != p["t_obs"]:
        return "info", f"Diese Folge hat bei k = {a.settings.k} keine Gelegenheit ({h} Treffer in {n} Würfen): Der Schätzer ist nicht definiert, ein Test ist nicht möglich."
    return "info", (f"Diese Folge hat {h} Treffer in {n} Würfen (eigene Trefferquote {fmt(h / n, 2)}) und den Schätzer {fmt(p['t_obs'])}. Ordnet man genau diese Würfe zufällig um "
                    f"({p['count']} Umordnungen, in {p['valid']} davon gibt es mindestens eine Gelegenheit), liegt der Schätzer im Mittel bei **{fmt(p['null_mean'])}** – das ist der Referenzwert "
                    f"des Permutationstests, nicht 0,5. Einseitige p-Werte: nach oben {fmt(p['p_upper'], 3)}, nach unten {fmt(p['p_lower'], 3)} (eine einzelne Folge hat wenig Aussagekraft).")


def enumeration_rows(n: int, k: int, q: float) -> list[dict]:
    """Aufzählung aller 2^n Folgen, Rekursion und (wo es sie gibt) geschlossene Formel nebeneinander: Zeilen {Größe, enum, recursion, formula}."""
    en = E.enumerate_stats(n, k, q)
    rc = E.exact_table(max(n, k + 1), k, q)[n]
    return [
        {"name": "P(mindestens eine Gelegenheit)", "enum": en.p_defined, "recursion": rc.p_defined, "formula": None},
        {"name": "E[Gelegenheiten O]", "enum": en.e_o, "recursion": rc.e_o, "formula": E.expected_opportunities(n, k, q)},
        {"name": "E[Treffer an Gelegenheiten H]", "enum": en.e_h, "recursion": rc.e_h, "formula": E.expected_hits(n, k, q)},
        {"name": "E[H] / E[O]", "enum": en.ratio, "recursion": rc.ratio, "formula": q},
        {"name": "E[H / O | O ≥ 1] (Erwartung des Schätzers)", "enum": en.mean_est, "recursion": rc.mean_est, "formula": None},
    ]


def method_rows(a: Analysis) -> list[dict]:
    """Verfahren und Begriffe der App mit ihrem Wert bei den aktuellen Reglern (Spur aus der Rechnung dieser Seite)."""
    s = a.settings
    naive = A.naive_test(a.O, a.H)
    return [
        {"Name": "Gelegenheit", "Was": f"Ein Wurf, vor dem die {s.k} Würfe davor alle Treffer waren.", "Wert": f"{int(a.O.sum())} Gelegenheiten in {s.m} Folgen, {int(a.H.sum())} davon Treffer"},
        {"Name": "Schätzer einer Folge", "Was": "Treffer an Gelegenheiten geteilt durch Gelegenheiten; ohne Gelegenheit nicht definiert.",
         "Wert": f"definiert in {a.n_defined} von {s.m} Folgen"},
        {"Name": "Mittel der Folgenschätzer", "Was": "Das Maß dieser Seite: Schätzer je Folge, dann Mittel über die Folgen mit Gelegenheit.", "Wert": fmt(a.mean_est)},
        {"Name": "Gepoolter Schätzer", "Was": "Alle Gelegenheiten und Treffer aller Folgen zusammenzählen, dann einmal teilen.", "Wert": fmt(a.pooled)},
        {"Name": "Rekursion (exakt)", "Was": "Verteilung von (Gelegenheiten, Treffer) über den Zustand „Länge des Treffer-Laufs am Ende“, Wurf für Wurf; liefert die Erwartung ohne Simulation.",
         "Wert": f"Erwartung des Schätzers {fmt(a.exact.mean_est)}"},
        {"Name": "Aufzählung (exakt)", "Was": f"Alle 2^n Folgen einzeln auswerten und nach ihrer Wahrscheinlichkeit gewichten; nur bis n = {C.ENUM_MAX}.", "Wert": "siehe Tabelle der exakten Probe"},
        {"Name": "Simulation", "Was": "Folgen mit festen Zufalls-Seeds ziehen (SplitMix64) und auswerten; Band über 5 Zufalls-Seeds.", "Wert": f"{s.m} Folgen, Zufalls-Seed {s.seed}"},
        {"Name": "Naive Prüfung", "Was": "Mittel der Folgenschätzer gegen 0,5 halten, Standardfehler aus der Binomial-Näherung; „heiß“ bei z > 1,645.",
         "Wert": f"z = {fmt(naive['z'], 2)}" if naive["z"] == naive["z"] else "nicht definiert"},
        {"Name": "Permutationstest", "Was": "Würfe jeder Folge zufällig umordnen (gleiche Trefferzahl), den Schätzer neu rechnen und den echten Wert in dieser Verteilung einordnen.",
         "Wert": f"erste Folge: p nach oben {fmt(a.perm['p_upper'], 3)}"},
    ]
