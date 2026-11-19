"""Der Hot-Hand-Messfehler: warum eine faire Münze nach Treffern schwächer aussieht - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Kernfrage: Wie misst man, ob einem Treffer ein Treffer folgt, und warum ist dieses Maß verzerrt? Aufbau nach DEMO-PLAYBOOK §2 (Konzepte-Demo).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import hoh_constants as C
from hoh_evaluation import (
    Settings,
    analyse,
    enumeration_rows,
    exact_curve,
    fmt,
    method_rows,
    power_curve,
    sweep_k,
    sweep_length,
    sweep_m,
    verdict_example,
    verdict_k,
    verdict_length,
    verdict_m,
    verdict_main,
    verdict_power,
    verdict_reference,
)
from hoh_layout import dunkles_design, kopf_css
from hoh_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from hoh_visualization import build_estimates, build_example, build_k, build_length, build_m, build_power

st.set_page_config(page_title="Der Hot-Hand-Messfehler – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _length(k, q_pct):
    return sweep_length(k, q_pct), exact_curve(k, q_pct)


@st.cache_data(show_spinner=False)
def _by_k(n, q_pct):
    return sweep_k(n, q_pct)


@st.cache_data(show_spinner=False)
def _by_m(n, k, q_pct):
    return sweep_m(n, k, q_pct)


@st.cache_data(show_spinner=False)
def _enumeration(n, k, q_pct):
    return enumeration_rows(n, k, q_pct / 100.0)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in hoh_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔥 Der Hot-Hand-Messfehler: warum eine faire Münze nach Treffern schwächer aussieht")
st.markdown(
    """
**Wie misst man, ob einem Treffer ein Treffer folgt?** Das in der Hot-Hand-Literatur verbreitete Maß (ob genau diese Variante in einer bestimmten Studie benutzt wurde, prüft die Seite nicht): In jeder Folge von Würfen alle Stellen suchen, vor denen **k Treffer in Folge** standen, den Anteil der Treffer an
diesen **Gelegenheiten** bestimmen und über die Folgen mitteln. Bei einer **fairen Münze** müsste dabei im Erwartungswert 0,5 herauskommen. Es kommt weniger heraus: **Das Maß ist in endlichen Folgen verzerrt**, und die
Verzerrung schrumpft nur langsam mit der Länge. Die Seite rechnet das **exakt** (Rekursion und Aufzählung aller Folgen), vergleicht den gepoolten Schätzer, lässt eine echte **Abhängigkeit** (Persistenz q) hinzu und
fragt, ab wann ein **Permutationstest** sie findet.

Wer den Messfehler zuerst ohne Fachbegriffe verstehen will, öffnet „💬 In einfachen Worten“. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Alles bezieht sich auf das Zufallsmodell, nicht auf Basketball oder andere Daten.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/hot-hand-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Man wirft eine faire Münze immer wieder und schaut, wie oft nach mehreren Treffern in Folge wieder ein Treffer kommt. Bei einer fairen Münze erwartet man die Hälfte. Wertet man aber jede kurze Wurffolge einzeln aus und bildet dann den Durchschnitt über die Folgen, kommt weniger heraus: Das Maß hat einen eingebauten Messfehler.

Die Demo rechnet diesen Fehler exakt aus, zeigt, dass mehr Folgen ihn beim Durchschnitt über die Folgen nicht beseitigen, beim Zusammenwerfen aller Gelegenheiten aber bei vielen Folgen fast ganz, und misst, ab wann ein Permutationstest eine echte Neigung der Münze findet. Alles gilt für das Zufallsmodell der Demo, nicht für Basketball oder andere echte Daten.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/hot-hand-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert das Maß", expanded=True):
    st.markdown(
        """
1. **Eine Folge** ist eine Reihe von Würfen: Treffer (1) oder Fehlwurf (0). Hier wirft eine Münze mit **Persistenz q**: Jeder Wurf geht mit Wahrscheinlichkeit q wie der davor aus, sonst anders.
   Bei q = 0,5 sind die Würfe unabhängig (faire Münze); jeder einzelne Wurf ist immer ein Treffer mit Wahrscheinlichkeit 0,5, und nach einem Treffer folgt mit Wahrscheinlichkeit **q** wieder einer.
2. **Gelegenheiten:** Ein Wurf ist eine Gelegenheit, wenn die **k Würfe davor alle Treffer** waren. Der **Schätzer einer Folge** ist der Anteil der Treffer an ihren Gelegenheiten. Hat eine Folge keine Gelegenheit, ist er nicht definiert.
3. **Mittel der Folgenschätzer:** Der Schätzer jeder Folge zählt einmal, die Folgen ohne Gelegenheit fallen weg. Alternativ zählt der **gepoolte Schätzer** alle Gelegenheiten aller Folgen zusammen.
4. **Warum das Mittel zu niedrig liegt** (Deutung, nicht einzeln gemessen; gemessen ist die Größe der Verzerrung in den Experimenten): Die Treffer einer Folge sind ein endlicher Vorrat. Wer gerade k Treffer
   „verbraucht“ hat, findet in den übrigen Würfen der Folge einen kleineren Anteil Treffer (wie Ziehen ohne Zurücklegen). Der Referenzwert des Permutationstests unten zeigt an der ersten Folge, wo der Schätzer
   bei zufällig umgeordneten Würfen liegt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Folgen**")
    n = st.slider("Länge n (Würfe je Folge)", *bounds("n_slider"), key="n_slider", help="Wie viele Würfe eine Folge hat.")
    m = st.slider("Zahl der Folgen m", *bounds("m_slider"), key="m_slider", help="Wie viele Folgen (je eine „Schützin“) gezogen werden.")
    st.markdown("**Maß**")
    k = st.slider("k Treffer in Folge davor", *bounds("k_slider"), key="k_slider", help="Eine Gelegenheit ist ein Wurf, vor dem k Treffer in Folge standen.")
    st.markdown("**Münze**")
    q_pct = st.slider("Persistenz q in % (50 = faire Münze)", *bounds("q_slider"), step=C.Q_PCT_STEP, key="q_slider",
                      help="Wahrscheinlichkeit, dass ein Wurf wie der davor ausgeht. 50: unabhängige Würfe; über 50: Treffer folgt häufiger auf Treffer; unter 50: seltener.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Folgen ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "k_slider": int(k), "q_slider": int(q_pct), "m_slider": int(m), "seed_input": int(seed)})

settings = Settings(n=int(n), k=int(k), q_pct=int(q_pct), m=int(m), seed=int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; es gibt keine gute oder schlechte Richtung, darum delta_color="off".
m1.metric("Wahre Quote nach einem Treffer", fmt(a.truth, 2), delta="Referenz", delta_color="off")
m2.metric("Exakte Erwartung des Schätzers", fmt(a.exact.mean_est), delta=f"{fmt(a.exact.mean_est - a.truth)} gegen wahr" if a.exact.mean_est == a.exact.mean_est else "nicht definiert", delta_color="off")
m3.metric(f"Mittel der {a.n_defined} Folgenschätzer", fmt(a.mean_est), delta=f"{fmt(a.mean_est - a.truth)} gegen wahr" if a.mean_est == a.mean_est else "nicht definiert", delta_color="off")
m4.metric("Gepoolter Schätzer", fmt(a.pooled), delta=f"{fmt(a.pooled - a.truth)} gegen wahr" if a.pooled == a.pooled else "nicht definiert", delta_color="off")
st.plotly_chart(build_estimates(a.est, a.truth, a.mean_est, a.pooled, a.exact.mean_est), width="stretch", key="est_chart")
st.caption(f"Schätzer je Folge bei {settings.m} Folgen der Länge {settings.n} (Zufalls-Seed {settings.seed}); Folgen ohne Gelegenheit ({settings.m - a.n_defined} von {settings.m}) fehlen. "
           "Die Balken sind eine Stichprobe: Mittel und gepoolter Wert streuen von Zufalls-Seed zu Zufalls-Seed um die exakte Erwartung bzw. ihren Grenzwert.")

st.markdown("## ❓ Kernfrage: Wie weit liegt das Maß neben der Wahrheit?")
kind, text = verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Wachsendes Beispiel -------------------------------------------------------------------------------------------------------------------

st.subheader("🧩 Eine Folge, Wurf für Wurf")
st.markdown("Die erste der gezogenen Folgen. Orange gerahmt: **Gelegenheiten**, also Würfe, vor denen k Treffer in Folge standen; gefüllt grün ist ein Treffer, grau ein Fehlwurf.")
if "example_step" not in st.session_state:
    st.session_state["example_step"] = min(settings.n, 12)
st.session_state["example_step"] = max(1, min(int(st.session_state["example_step"]), settings.n))


def _advance():
    st.session_state["example_step"] = min(int(st.session_state["example_step"]) + 1, int(st.session_state["n_slider"]))


c1, c2 = st.columns([4, 1])
step = c1.slider("Gezeigt: die ersten … Würfe", 1, settings.n, key="example_step")
c2.button("Einen Wurf weiter", on_click=_advance, key="example_next")
st.plotly_chart(build_example(a.x[0], a.positions, int(step), settings.k), width="stretch", key="example_chart")
kind, text = verdict_example(a, int(step))
getattr(st, kind)(text)
kind, text = verdict_reference(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Stimmen Aufzählung, Rekursion und Formel überein?")
n_enum = min(settings.n, C.ENUM_MAX)
enum_count = f"{2 ** n_enum:,}".replace(",", ".")  # deutsche Tausenderpunkte
st.caption(f"Aufzählung aller 2^{n_enum} = {enum_count} Folgen der Länge n = {n_enum} (die Aufzählung geht höchstens bis n = {C.ENUM_MAX}; ist der Regler n größer, gilt hier {C.ENUM_MAX}), k = {settings.k}, q = {fmt(settings.q, 2)}. "
           "Drei Rechenwege: Aufzählung (jede Folge einzeln, mit ihrer Wahrscheinlichkeit gewichtet), Rekursion (Zustand: Länge des Treffer-Laufs am Ende) und, wo es sie gibt, eine geschlossene Formel.")
rows = _enumeration(n_enum, settings.k, settings.q_pct)
st.dataframe(
    [{"Größe": r["name"], "Aufzählung": fmt(r["enum"], 6), "Rekursion": fmt(r["recursion"], 6), "Formel": fmt(r["formula"], 6) if r["formula"] is not None else "–"} for r in rows],
    hide_index=True, width="stretch",
)
agree = all(abs(r["enum"] - r["recursion"]) < 1e-9 for r in rows if r["enum"] == r["enum"])
(st.success if agree else st.error)(
    f"Aufzählung und Rekursion stimmen in allen {len(rows)} Zeilen auf 9 Stellen überein." if agree else "Aufzählung und Rekursion weichen ab: Die Seite zeigt dann keine Messung, sondern einen Fehler."
)

st.markdown("---")

st.subheader("🔬 Experiment: Wie hängt die Verzerrung von der Länge ab?")
st.caption(f"k = {settings.k}, q = {fmt(settings.q, 2)}. Linie: exakte Erwartung des Schätzers einer Folge (Rekursion) für jede Länge. Punkte: Simulation, je {len(C.SWEEP_SEEDS)} feste Zufalls-Seeds mit "
           f"{C.SIM_SEQUENCES_PER_SEED} Folgen; das Band reicht vom kleinsten bis zum größten Wert der Zufalls-Seeds.")
with st.spinner("Rechnet Rekursion und Simulation (einige Sekunden beim ersten Aufruf)…"):
    length_rows, curve = _length(settings.k, settings.q_pct)
st.plotly_chart(build_length(length_rows, curve, settings.q, settings.k), width="stretch", key="length_chart")
kind, text = verdict_length(length_rows, settings.q, settings.k, curve)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment: Wie hängt die Verzerrung von k ab?")
st.caption(f"Exakte Erwartung des Schätzers bei n = {settings.n} und q = {fmt(settings.q, 2)} für k = {C.K_MIN} bis {C.K_MAX} (Rekursion).")
k_rows = _by_k(settings.n, settings.q_pct)
st.plotly_chart(build_k(k_rows, settings.q), width="stretch", key="k_chart")
kind, text = verdict_k(k_rows, settings.n, settings.q)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment: Hilft es, mehr Folgen zusammenzulegen?")
st.caption(f"n = {settings.n}, k = {settings.k}, q = {fmt(settings.q, 2)}. Je Zahl der Folgen m wiederholt die Simulation ‚m Folgen ziehen, beide Schätzer bilden‘ so oft, dass überall gleich viele Folgen "
           f"eingehen ({C.SIM_SEQUENCES_PER_SEED} je Zufalls-Seed, {len(C.SWEEP_SEEDS)} Zufalls-Seeds); die Bänder zeigen den kleinsten und größten Wert der Zufalls-Seeds.")
with st.spinner("Rechnet die Simulation (einige Sekunden beim ersten Aufruf)…"):
    m_rows = _by_m(settings.n, settings.k, settings.q_pct)
st.plotly_chart(build_m(m_rows), width="stretch", key="m_chart")
kind, text = verdict_m(m_rows)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Experiment: Ab wann findet ein Test eine echte Abhängigkeit?")
n_p, m_p = min(settings.n, C.POWER_N_CAP), min(settings.m, C.POWER_M_CAP)
runs = C.POWER_REPS_PER_SEED * len(C.SWEEP_SEEDS)
st.markdown(
    f"Gleiche Folgen, zwei Entscheidungsregeln, jeweils einseitig bei 5 %: die **naive Prüfung** (Mittel der Folgenschätzer gegen 0,5) und der **Permutationstest** (Würfe jeder Folge {C.PERM_B}-mal zufällig umordnen, "
    "den echten Wert in dieser Verteilung einordnen). Gezählt wird, in wie viel Prozent der Läufe die Regel „heiß“ meldet, wenn die Münze Persistenz q hat; bei q = 0,5 ist das die Fehlalarmquote."
)
st.caption(f"Läuft für n = {n_p}, k = {settings.k}, {m_p} Folgen (n und m sind für die Rechenzeit auf {C.POWER_N_CAP} und {C.POWER_M_CAP} gedeckelt), q = " + ", ".join(fmt(v / 100, 2) for v in C.Q_GRID_PCT) +
           f"; {runs} Läufe je q ({len(C.SWEEP_SEEDS)} feste Zufalls-Seeds zu je {C.POWER_REPS_PER_SEED}). Der Regler q gilt hier nicht, er steht nur als Linie im Bild.")
power_key = (n_p, settings.k, m_p)
if st.button("🧪 Teststärke berechnen", key="power_run", help=f"{runs * len(C.Q_GRID_PCT)} Läufe mit je {C.PERM_B} Umordnungen: je nach n und m einige Sekunden bis eine halbe Minute."):
    with st.spinner(f"Rechnet {runs * len(C.Q_GRID_PCT)} Läufe mit je {C.PERM_B} Umordnungen (bei n = {n_p} und {m_p} Folgen einige Sekunden bis etwa eine halbe Minute)…"):
        st.session_state["power_result"] = {"key": power_key, "rows": power_curve(n_p, settings.k, m_p)}
result = st.session_state.get("power_result")
if result and result["key"] == power_key:
    st.plotly_chart(build_power(result["rows"], settings.q_pct), width="stretch", key="power_chart")
    kind, text = verdict_power(result["rows"], n_p, settings.k, m_p)
    getattr(st, kind)(text)
elif result:
    st.info("Die Regler n, m oder k haben sich seit der Berechnung geändert: Bitte die Teststärke neu berechnen.")
else:
    st.info("Noch nicht berechnet: Der Knopf startet den Lauf.")

st.markdown("---")

with st.expander("📋 Verfahren und Begriffe: wer macht was?"):
    st.markdown("Jede Zeile nennt ein Verfahren oder einen Begriff der Seite und seinen Wert bei den aktuellen Reglern.")
    st.dataframe(method_rows(a), hide_index=True, width="stretch")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Münze mit Persistenz q, gleiche Quote 0,5 für alle Folgen** | Echte Schützen haben unterschiedliche Quoten und unterschiedlich schwere Würfe; ob sich das mit einer Abhängigkeit vermischt, sagt diese Seite nicht. | – |
| **Gedächtnis nur über den Wurf davor (Markow-Kette erster Ordnung)** | Andere Formen von Abhängigkeit (längeres Gedächtnis, wechselnde Form) sind nicht gerechnet; ob der Permutationstest sie findet, ist nicht gemessen. | Wurzel der Linie „Wann zählt die Reihenfolge?“ (dort steht, wann Abhängigkeit überhaupt einen Reihenfolgeeffekt bewirkt) |
| **Der Folgenschätzer ist das Maß** | Eine Rangliste aus Treffern kommt mit Rauschen: Auch ohne Abhängigkeit schwanken Ränge. | `siegtabelle-demo` (Rauschen in Tabellen), `rangintervalle-demo` (wie sicher ein Rang ist) |
| **Ein Permutationstest je Folgenmenge, 5 % einseitig** | Wer mehrere Folgen einzeln testet und die auffälligste herausgreift, hat die 5 % verlassen (Mehrfachtests); das ist hier nicht gerechnet. | – |
| **Teststärke aus 100 Läufen je q** | Die Teststärke ist auf einige Prozentpunkte genau (das Band zeigt die Streuung der Zufalls-Seeds); ein Wert um 80 % kann bei anderen Zufalls-Seeds auf der anderen Seite der Schwelle liegen. | – |
    """
)
with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Würfe sind Treffer oder Fehlwurf; der erste Wurf ist mit Wahrscheinlichkeit 1/2 ein Treffer; jeder weitere geht mit Wahrscheinlichkeit q wie sein Vorgänger aus (symmetrische Markow-Kette,
Zufall aus SplitMix64 mit festen Zufalls-Seeds). Folgen sind unabhängig voneinander und gleich lang. Der Folgenschätzer ist nur in Folgen mit mindestens einer Gelegenheit definiert; alle Erwartungen
sind darauf bedingt.

**Grenzen.** Die Seite sagt etwas über dieses Zufallsmodell, nichts über Basketball, Sportdaten oder Menschen. Die Literatur (Gilovich, Vallone und Tversky 1985; Miller und Sanjurjo 2018) ist **nicht im Volltext gelesen**;
die Seite beruft sich nicht auf deren Ergebnisse. Der Permutationstest prüft nur Austauschbarkeit der Würfe innerhalb einer Folge: er verwirft sie, soweit eine Abhängigkeit die Prüfgröße (das Mittel der Folgenschätzer) verschiebt; gemessen ist nur die Markow-Persistenz q.

**Bekannte Probleme und Unsicherheiten.** Die Simulationen und die Teststärke sind Stichproben mit festen Zufalls-Seeds (Band angegeben). Der Teststärke-Lauf ist für die Rechenzeit gedeckelt (n ≤ 100, m ≤ 20).
Die Darstellung im Browser (Farben, Rahmen der Kästchen, Gedrängtheit bei n = 200) ist mit AppTest nicht sichtbar und nicht in einem echten Browser geprüft.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/hot-hand-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $m$ Folgen von je $n$ Würfen (Treffer 1, Fehlwurf 0) und eine Zahl $k$. Gesucht ist ein Maß dafür, wie oft auf $k$ Treffer in Folge wieder ein Treffer folgt, und wie stark dieses Maß von der wahren Quote $q$ abweicht.

| Symbol | Bedeutung |
|---|---|
| $x_t$ | Wurf $t$ einer Folge (1 Treffer, 0 Fehlwurf), $t = 1, \dots, n$ |
| $n$, $m$ | Länge einer Folge bzw. Zahl der Folgen |
| $k$ | Länge des Treffer-Laufs vor einer Gelegenheit |
| $q$ | Wahrscheinlichkeit, dass ein Wurf wie der davor ausgeht ($q = 1/2$: unabhängige Würfe) |
| $I$, $O$, $H$ | Menge der Gelegenheiten einer Folge, ihre Zahl, Zahl der Treffer an ihnen |
| $\hat p$ | Schätzer einer Folge: Anteil der Treffer an ihren Gelegenheiten, $H/O$ (nur für $O \ge 1$) |
| $D$ | Menge der Folgen mit mindestens einer Gelegenheit |
| $\bar p$, $\tilde p$ | Mittel der Folgenschätzer bzw. gepoolter Schätzer über alle Folgen |
| $T$, $B$ | Prüfgröße des Permutationstests (das Mittel $\bar p$) und Zahl der Umordnungen |
| $\mathrm{SE}$, $z$ | Standardfehler und Prüfgröße der naiven Prüfung gegen 0,5 |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/hot-hand-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Gegeben** sind $m$ Folgen $x^{(j)} \in \{0,1\}^n$ und eine Zahl $k \ge 1$; **gesucht** ist ein Maß dafür, wie oft ein Treffer auf $k$ Treffer in Folge folgt.

| Symbol | Bedeutung |
|---|---|
| $x_t$ | Wurf $t$ einer Folge (1 Treffer, 0 Fehlwurf), $t = 1, \dots, n$ |
| $k$ | Länge des Treffer-Laufs vor einer Gelegenheit |
| $q$ | Wahrscheinlichkeit, dass ein Wurf wie der davor ausgeht ($q = 1/2$: unabhängige Würfe) |
| $I$, $O$, $H$ | Menge der Gelegenheiten einer Folge, ihre Zahl, Zahl der Treffer an ihnen |

**Gelegenheiten, Treffer, Schätzer einer Folge.**

$$I = \{\, t > k : x_{t-k} = \dots = x_{t-1} = 1 \,\}, \qquad O = |I|, \qquad H = \sum_{t \in I} x_t$$

$$\hat p = \frac{H}{O} \quad (O \ge 1)$$

**Zwei Maße über $m$ Folgen** (mit $D$ = Menge der Folgen mit $O^{(j)} \ge 1$):

$$\bar p = \frac{1}{|D|} \sum_{j \in D} \hat p^{(j)} \qquad \text{(Mittel der Folgenschätzer)}$$

$$\tilde p = \frac{\sum_j H^{(j)}}{\sum_j O^{(j)}} \qquad \text{(gepoolt)}$$

**Modell.** Erster Wurf: Treffer mit Wahrscheinlichkeit $1/2$. Danach $P(x_t = x_{t-1}) = q$. Jeder Wurf ist einzeln ein Treffer mit Wahrscheinlichkeit $1/2$, und $P(x_t = 1 \mid x_{t-1} = 1, \dots) = q$ für jede Vorgeschichte.

**Erwartungen von $O$ und $H$** (geschlossene Form; jede der $n-k$ Positionen $t > k$ ist mit Wahrscheinlichkeit $\tfrac12 q^{k-1}$ eine Gelegenheit, und an jeder Gelegenheit folgt mit Wahrscheinlichkeit $q$ ein Treffer):

$$\mathbb{E}[O] = \frac{n-k}{2}\, q^{k-1}$$

$$\mathbb{E}[H] = q\, \mathbb{E}[O]$$

Das Verhältnis $\mathbb{E}[H]/\mathbb{E}[O] = q$ ist der Grenzwert des gepoolten Schätzers bei vielen Folgen. Der Folgenschätzer dagegen hat die Erwartung

$$\mathbb{E}\!\left[\frac{H}{O} \,\middle|\, O \ge 1\right] \neq q \quad \text{(Verzerrung, hier ausgerechnet statt abgeleitet)}$$

Kleinster Fall, von Hand: $n = 3$, $k = 1$, $q = 1/2$. Die acht Folgen haben die Schätzer $0, 1, 0, 0, \tfrac12, 1$ in den sechs Folgen mit $O \ge 1$ (010, 011, 100, 101, 110, 111) und keine Gelegenheit in 000 und 001; der Mittelwert ist $\tfrac{5}{12} < \tfrac12$.

**Rekursion.** Zustand nach $t$ Würfen: $(r, o, h)$ mit $r = \min(\text{Länge des Treffer-Laufs am Ende}, k)$. Der nächste Wurf ist eine Gelegenheit genau dann, wenn $r = k$. Die Übergangswahrscheinlichkeit des nächsten Wurfs ist $q$ (gleiches Ergebnis wie der letzte) bzw. $1-q$. Daraus ergibt sich die gemeinsame Verteilung von $(O, H)$ für jede Länge, daraus die Erwartung des Schätzers.

**Permutationstest.** Prüfgröße $T$ = Mittel der Folgenschätzer. Jede Folge wird einzeln zufällig umgeordnet (gleiche Trefferzahl), $B$-mal; $T^{(b)}$ sei der Wert der $b$-ten Umordnung. Einseitiger p-Wert nach oben:

$$p = \frac{1 + \#\{\, b : T^{(b)} \ge T_{\text{beob}} \,\}}{B + 1}$$

**Naive Prüfung.** Mit Standardfehler $\mathrm{SE} = \sqrt{\sum_{j \in D} \tfrac{1}{4 O^{(j)}}} \,/\, |D|$ aus der Binomial-Näherung:

$$z = \frac{\bar p - 0{,}5}{\mathrm{SE}}, \qquad \text{„heiß“ bei } z > 1{,}645$$

**Literatur (nicht im Volltext gelesen).** Gilovich, Vallone, Tversky (1985); Miller, Sanjurjo (2018).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Wettbewerbsverzerrung: Reihenfolge und Anreize](https://sebastianhanisch.net/konzepte-wettbewerbsverzerrung.html)."
)
