"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import hoh_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_example(seq, positions, step, k, row=C.EXAMPLE_ROW):
    """Eine Folge als Reihen von Kästchen: grün = Treffer, grau = Fehlwurf, Rahmen in Orange = Gelegenheit (davor standen k Treffer in Folge).
    Gezeigt werden die ersten `step` Würfe; die übrigen stehen als leere Umrisse da, damit die Folge beim Weiterschalten nicht springt."""
    n = len(seq)
    pos = set(positions)
    xs = [t % row for t in range(n)]
    ys = [-(t // row) for t in range(n)]
    shown = [t for t in range(n) if t < step]
    fig = go.Figure()
    rest = [t for t in range(n) if t >= step]
    if rest:
        fig.add_trace(go.Scatter(x=[xs[t] for t in rest], y=[ys[t] for t in rest], mode="markers", name="noch nicht geworfen",
                                 marker=dict(symbol="square-open", size=16, color="rgba(150,150,150,0.5)"), hoverinfo="skip"))
    kinds = [
        ("Treffer", lambda t: seq[t] == 1 and t not in pos, C.COLOR_HIT, "rgba(0,0,0,0)", 0),
        ("Fehlwurf", lambda t: seq[t] == 0 and t not in pos, C.COLOR_MISS, "rgba(0,0,0,0)", 0),
        (f"Gelegenheit, Treffer (davor {k} Treffer)", lambda t: seq[t] == 1 and t in pos, C.COLOR_HIT, C.COLOR_OPP, 3),
        (f"Gelegenheit, Fehlwurf (davor {k} Treffer)", lambda t: seq[t] == 0 and t in pos, C.COLOR_MISS, C.COLOR_OPP, 3),
    ]
    for name, pred, color, line, width in kinds:
        idx = [t for t in shown if pred(t)]
        if not idx:
            continue
        fig.add_trace(go.Scatter(x=[xs[t] for t in idx], y=[ys[t] for t in idx], mode="markers", name=name,
                                 marker=dict(symbol="square", size=16, color=color, line=dict(color=line, width=width)),
                                 customdata=[t + 1 for t in idx], hovertemplate="Wurf %{customdata}<extra>" + name + "</extra>"))
    rows = (n - 1) // row + 1
    fig.update_xaxes(visible=False, range=[-0.7, row - 0.3])
    fig.update_yaxes(visible=False, range=[-(rows - 1) - 0.7, 0.7])
    return _base(fig, height=max(150, 60 + 38 * rows))


def build_estimates(est, truth, mean_est, pooled, exact):
    """Verteilung der Folgenschätzer (nur Folgen mit Gelegenheit) in 10 Klassen; senkrechte Linien: wahre Quote, exakte Erwartung, Mittel der Folgenschätzer, gepoolt."""
    e = np.asarray(est, dtype=float)
    e = e[~np.isnan(e)]
    counts, edges = np.histogram(e, bins=10, range=(0.0, 1.0)) if len(e) else (np.zeros(10), np.linspace(0, 1, 11))
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, width=0.085, marker_color="rgba(150,150,150,0.6)", name="Folgenschätzer",
                           hovertemplate="%{x:.2f}: %{y} Folgen<extra></extra>"))
    ymax = max(float(counts.max()), 1.0) * 1.18
    for value, name, color, dash in ((truth, "wahre Quote", C.COLOR_TRUE, "solid"), (exact, "exakte Erwartung des Schätzers", C.COLOR_MEAN, "dot"),
                                     (mean_est, "Mittel der Folgenschätzer", C.COLOR_MEAN, "solid"), (pooled, "gepoolt", C.COLOR_POOLED, "solid")):
        if value == value:
            fig.add_trace(go.Scatter(x=[value, value], y=[0, ymax], mode="lines", line=dict(color=color, width=3, dash=dash), name=f"{name}: {value:.3f}".replace(".", ","),
                                     hoverinfo="skip"))
    fig.update_xaxes(title_text="Schätzer einer Folge (Treffer an Gelegenheiten / Gelegenheiten)", range=[-0.06, 1.06])
    fig.update_yaxes(title_text="Zahl der Folgen", range=[0, ymax])
    return _base(fig, height=380)


def build_length(rows, curve, truth, k):
    """Verzerrung über die Länge: exakte Erwartung für jede Länge (Linie), Simulation an einzelnen Längen (Punkte mit Band über die Zufalls-Seeds), wahre Quote."""
    fig = go.Figure()
    xs = [r["n"] for r in rows]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r["high"] for r in rows] + [r["low"] for r in rows][::-1], fill="toself", fillcolor="rgba(192,57,43,0.15)",
                             line=dict(width=0), hoverinfo="skip", name="Simulation: Band der Zufalls-Seeds"))
    fig.add_trace(go.Scatter(x=[n for n, _ in curve], y=[v for _, v in curve], mode="lines", line=dict(color=C.COLOR_MEAN, width=3), name="exakte Erwartung des Schätzers",
                             hovertemplate="n = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["sim"] for r in rows], mode="markers", marker=dict(color=C.COLOR_MEAN, size=9, symbol="diamond"), name="Simulation (Mittel der Seeds)",
                             hovertemplate="n = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[min(n for n, _ in curve), max(n for n, _ in curve)], y=[truth, truth], mode="lines", line=dict(color=C.COLOR_TRUE, dash="dash", width=2),
                             name="wahre Quote", hoverinfo="skip"))
    fig.update_xaxes(title_text="Länge der Folge n (Würfe)", type="log", tickvals=[8, 10, 20, 50, 100, 200], ticktext=["8", "10", "20", "50", "100", "200"])
    fig.update_yaxes(title_text=f"Erwartung des Schätzers (k = {k})", range=[min(0.25, min(v for _, v in curve) - 0.03), max(0.75, truth + 0.1)])
    return _base(fig)


def build_k(rows, truth):
    fig = go.Figure(go.Bar(x=[f"k = {r['k']}" for r in rows], y=[r["exact"] for r in rows], marker_color=C.COLOR_MEAN, name="exakte Erwartung des Schätzers",
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=truth, line=dict(color=C.COLOR_TRUE, dash="dash", width=2), annotation_text="wahre Quote", annotation_position="top left")
    fig.update_yaxes(title_text="Erwartung des Schätzers", range=[0, max(0.75, truth + 0.1)])
    return _base(fig, height=320)


def build_m(rows):
    """Mittel der Folgenschätzer und gepoolter Schätzer über die Zahl der Folgen (Simulation, Band über die Zufalls-Seeds), dazu die zwei exakten Grenzwerte."""
    fig = go.Figure()
    xs = [r["m"] for r in rows]
    for key, name, color in (("mean", "Mittel der Folgenschätzer", C.COLOR_MEAN), ("pooled", "gepoolter Schätzer", C.COLOR_POOLED)):
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r[key + "_high"] for r in rows] + [r[key + "_low"] for r in rows][::-1], fill="toself", line=dict(width=0),
                                 fillcolor="rgba(192,57,43,0.15)" if key == "mean" else "rgba(76,120,168,0.18)", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=3), name=name, hovertemplate="m = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=rows[0]["exact_single"], line=dict(color=C.COLOR_MEAN, dash="dot", width=2), annotation_text="Erwartung bei einer Folge", annotation_position="bottom right")
    fig.add_hline(y=rows[0]["ratio"], line=dict(color=C.COLOR_POOLED, dash="dot", width=2), annotation_text="E[Treffer] / E[Gelegenheiten]", annotation_position="top right")
    fig.update_xaxes(title_text="Zahl der Folgen m", type="log", tickvals=list(xs), ticktext=[str(v) for v in xs])
    fig.update_yaxes(title_text="Schätzwert", range=[0.25, 0.75])
    return _base(fig)


def build_power(rows, q_now_pct=None):
    """Teststärke über die Persistenz q: Permutationstest und naive Prüfung (Band: kleinster bis größter Wert der Zufalls-Seeds)."""
    fig = go.Figure()
    xs = [r["q_pct"] / 100 for r in rows]
    for key, name, color in (("perm_hot", "Permutationstest", C.COLOR_PERM), ("naive_hot", "naive Prüfung gegen 0,5", C.COLOR_MEAN)):
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r[key + "_high"] for r in rows] + [r[key + "_low"] for r in rows][::-1], fill="toself", line=dict(width=0),
                                 fillcolor="rgba(47,107,101,0.15)" if key == "perm_hot" else "rgba(192,57,43,0.15)", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=3), name=name, hovertemplate="q = %{x:.2f}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=C.POWER_TARGET, line=dict(color="rgba(120,120,120,0.8)", dash="dot", width=1), annotation_text="80 %", annotation_position="bottom right")
    fig.add_hline(y=C.ALPHA, line=dict(color="rgba(120,120,120,0.8)", dash="dot", width=1), annotation_text="5 %", annotation_position="top right")
    if q_now_pct is not None and min(xs) <= q_now_pct / 100 <= max(xs):
        fig.add_vline(x=q_now_pct / 100, line=dict(color=C.COLOR_TRUE, dash="dash", width=1), annotation_text="Regler q", annotation_position="top")
    fig.update_xaxes(title_text="Persistenz q (wahre Trefferquote nach einem Treffer)", tickvals=xs)
    fig.update_yaxes(title_text="Anteil der Läufe mit Entscheidung „heiß“", range=[-0.02, 1.05], tickformat=".0%")
    return _base(fig)
