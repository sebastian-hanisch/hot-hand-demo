"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 8, 200, 20          # Länge einer Folge (Zahl der Würfe). Rechenaufwand am Maximum: O(m · n) Zufallszahlen
K_MIN, K_MAX, DEFAULT_K = 1, 5, 3             # k Treffer in Folge vor dem betrachteten Wurf; K_MAX <= N_MIN - 3, damit jede Folge Gelegenheiten haben kann
Q_PCT_MIN, Q_PCT_MAX, DEFAULT_Q_PCT = 30, 90, 50   # Persistenz q in Prozent: Wahrscheinlichkeit, dass ein Wurf wie sein Vorgänger ausgeht; 50 = faire Münze
Q_PCT_STEP = 5
M_MIN, M_MAX, DEFAULT_M = 1, 100, 100         # Zahl der Folgen (je eine „Schützin“)
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- feste Messreihen -------------------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = (11, 23, 37, 41, 59)            # feste Zufalls-Seeds aller Simulationen: nie ein einzelner Seed als Befund
SIM_SEQUENCES_PER_SEED = 4000                 # Folgen je Zufalls-Seed und Länge (5 Seeds: 20 000 Folgen je Zelle)
N_SIM_GRID = (10, 20, 50, 100, 200)           # Längen, an denen die Simulation neben der exakten Kurve steht
M_GRID = (1, 2, 5, 10, 20, 50, 100)           # Zahl der Folgen im Experiment „Mehr Folgen“
Q_GRID_PCT = (50, 55, 60, 65, 70, 80)         # Persistenz-Werte der Teststärke-Kurve
ENUM_MAX = 16                                 # so lange Folgen darf die Aufzählung aller 2^n Folgen höchstens haben (65 536 Folgen)

# --- Permutationstest und Teststärke ----------------------------------------------------------------------------------------------------------
ALPHA = 0.05                                  # Irrtumswahrscheinlichkeit beider Tests (einseitig)
PERM_B = 99                                   # Umordnungen je Test (kleinster möglicher p-Wert 1/100)
EXAMPLE_PERM_B = 999                          # Umordnungen für die Referenz im wachsenden Beispiel
POWER_REPS_PER_SEED = 20                      # Wiederholungen je Zufalls-Seed und Persistenz-Wert (5 Seeds: 100 Läufe je Wert)
POWER_N_CAP, POWER_M_CAP = 100, 20            # Deckel für Länge und Zahl der Folgen im Teststärke-Lauf (Rechenzeit)
POWER_TARGET = 0.8                            # „findet die Abhängigkeit“: Teststärke mindestens 80 %

# --- Darstellung ------------------------------------------------------------------------------------------------------------------------------
EXAMPLE_ROW = 25                              # Würfe je Zeile im wachsenden Beispiel
COLOR_HIT = "#2F6B65"
COLOR_MISS = "rgba(150,150,150,0.45)"
COLOR_OPP = "#d98e04"
COLOR_TRUE = "#222222"
COLOR_MEAN = "#c0392b"
COLOR_POOLED = "#4c78a8"
COLOR_PERM = "#2F6B65"

PRESETS = {
    "Faire Münze, 20 Würfe": {"n": DEFAULT_N, "k": DEFAULT_K, "q_pct": DEFAULT_Q_PCT, "m": DEFAULT_M, "seed": DEFAULT_SEED},
    "Lange Folgen (200 Würfe)": {"n": 200, "k": 3, "q_pct": 50, "m": 100, "seed": DEFAULT_SEED},
    "Eine einzige Folge": {"n": 20, "k": 3, "q_pct": 50, "m": 1, "seed": DEFAULT_SEED},
    "Aufzählbar (16 Würfe, k = 2)": {"n": 16, "k": 2, "q_pct": 50, "m": 100, "seed": DEFAULT_SEED},
    "Echte Persistenz (q = 0,65)": {"n": 50, "k": 2, "q_pct": 65, "m": 20, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Faire Münze, 20 Würfe": "100 Folgen zu je 20 Würfen einer fairen Münze, k = 3: Wie weit liegt das Maß unter der wahren Trefferquote 0,5?",
    "Lange Folgen (200 Würfe)": "Dieselbe faire Münze mit 200 Würfen je Folge: Die Verzerrung wird kleiner, verschwindet aber nicht.",
    "Eine einzige Folge": "Nur eine Folge: Mittel der Folgenschätzer und gepoolter Schätzer sind dann dasselbe Maß.",
    "Aufzählbar (16 Würfe, k = 2)": "16 Würfe: Die Erwartung lässt sich über alle 65 536 möglichen Folgen exakt ausrechnen und mit der Rekursion vergleichen.",
    "Echte Persistenz (q = 0,65)": "Eine Münze, die zu 65 % wie beim Wurf davor ausgeht (50 Würfe, 20 Folgen, k = 2): Findet der Permutationstest das?",
}
