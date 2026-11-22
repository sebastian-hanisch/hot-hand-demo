# Der Hot-Hand-Messfehler: warum eine faire Münze nach Treffern schwächer aussieht – Streamlit-Demo

<!-- Live-Link: direkt hierher setzen, sobald die App veröffentlicht ist (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-hot-hand-demo.streamlit.app/)** -->

Konzepte-Demo der Linie **„Wettbewerbsverzerrung erklärt“** (Bereich Wertung und Wahl), Stück **„Wie misst man Abhängigkeit?“**. Es setzt an der Wurzel der Linie an (**„Wann zählt die Reihenfolge?“**, nur beim Namen genannt): Dort ist Abhängigkeit
die Voraussetzung dafür, dass eine Reihenfolge überhaupt etwas ändert; hier geht es um die Frage, **wie man sie misst**. Das in der Hot-Hand-Literatur verbreitete Maß (ob genau diese Variante in einer bestimmten Studie benutzt wurde, prüft die Demo nicht), der Anteil der Treffer an den Stellen, vor denen **k Treffer in Folge** standen, je Folge gerechnet und über die
Folgen gemittelt, ist in endlichen Folgen verzerrt: Bei einer **fairen Münze** liegt es unter 0,5. Die Demo rechnet die Verzerrung **exakt** (Rekursion und Aufzählung aller Folgen), vergleicht den **gepoolten** Schätzer, lässt eine echte
**Abhängigkeit** (Persistenz q) zu und misst, ab wann ein **Permutationstest** sie findet. **Alles gilt für das Zufallsmodell der Seite; sie sagt nichts über Basketball, Sportdaten oder Menschen.**

## Ergebnis in Kürze

(Alle Zahlen: faire Münze, k = 3, wenn nichts anderes steht. Jede Zahl steht mit ihrem Test in der Tabelle „Befunde“.)

- **Das Maß liegt bei n = 20 Würfen im Erwartungswert bei 0,361 statt 0,5** (Verzerrung −0,139); bei n = 50 bei 0,418, bei n = 100 bei 0,460, bei n = 200 bei 0,481 (Verzerrung −0,019). Die Verzerrung schrumpft nur langsam und ist bei n = 200 nicht null.
- **Mehr Folgen helfen dem Mittel der Folgenschätzer nicht:** bei n = 20 liegt es bei m = 1 Folge bei 0,360, bei m = 100 Folgen bei 0,365 (Simulation, 5 feste Zufalls-Seeds); der **gepoolte** Schätzer steigt von 0,360 (m = 1) über 0,401 (m = 2) und 0,487 (m = 10) auf 0,503 (m = 100).
  Sein Grenzwert für sehr viele Folgen ist exakt q, bei fairer Münze 0,5.
- **Eine echte Abhängigkeit sieht bei kurzen Folgen nach wenig aus:** Hat die Münze Persistenz q = 0,65 (nach einem Treffer folgt mit 65 % wieder einer), liegt das Maß bei n = 20 und k = 3 im Erwartungswert bei 0,518; bei q = 0,60 bei 0,464, also **unter** 0,5.
- **Gegen die in der Demo definierte naive Prüfung (gegen 0,5, ohne Korrektur der Verzerrung) findet der Permutationstest die Abhängigkeit früher** (n = 50, k = 2, 20 Folgen, 100 Läufe je q, Raster 0,05): 80 % Teststärke ab q = 0,60 gegen ab q = 0,65; bei q = 0,55 meldet er in 47 % der Läufe „heiß“, die naive Prüfung in 18 %.
- **Die naive Prüfung meldet bei fairer Münze häufig „kalt“:** in etwa 19 % der Läufe (n = 50, k = 2, 20 Folgen), bei n = 20 und k = 3 in etwa 43 %, mit einer Streuung zwischen den 5 Zufalls-Seeds (je 20 Läufe) von 10 bis 25 % bzw. 20 bis 55 %; der Permutationstest in 4 % bzw. 2 % (Fehlerrate 1. Art von 5 % ist sein Ziel).

## Kernfrage

Wie misst man, ob einem Treffer ein Treffer folgt, und warum ist dieses Maß verzerrt? Gemessen wird das Maß selbst, an einem Modell, in dem die richtige Antwort bekannt ist: Jeder Wurf ist einzeln mit Wahrscheinlichkeit 1/2 ein Treffer, und nach einem Treffer folgt mit
Wahrscheinlichkeit **q** wieder einer (q = 0,5: unabhängige Würfe, die faire Münze).

## Modell

- **Münze mit Persistenz q** (`hoh_scenario.py`): erster Wurf Treffer mit Wahrscheinlichkeit 1/2, jeder weitere geht mit Wahrscheinlichkeit q wie sein Vorgänger aus, sonst umgekehrt (symmetrische Markow-Kette erster Ordnung). Dann ist jeder Wurf einzeln ein Treffer mit
  Wahrscheinlichkeit 1/2 und die wahre Trefferquote nach k Treffern ist für jedes k gleich q.
- **Gelegenheit zu k:** ein Wurf, vor dem die k Würfe davor alle Treffer waren. O = Zahl der Gelegenheiten einer Folge, H = Zahl der Treffer an Gelegenheiten. **Schätzer einer Folge** H / O, nur definiert, wenn O ≥ 1.
- **Mittel der Folgenschätzer** (das Maß dieser Seite): Mittel von H / O über die Folgen mit O ≥ 1. **Gepoolter Schätzer:** Summe aller H geteilt durch Summe aller O.
- **Geschlossene Form** (`hoh_exact.py`): E[O] = (n − k) · q^(k−1) / 2 und E[H] = q · E[O]; daher strebt der gepoolte Schätzer gegen q. Die Erwartung des Folgenschätzers E[H / O | O ≥ 1] hat für allgemeines k keine einfache Form, die diese Seite verwendet; sie wird gerechnet (siehe Methodik).
- Formeln, Symbole und der von Hand gerechnete Kleinstfall (n = 3, k = 1: Mittel 5/12) stehen im 📐-Expander der App.

## Methodik

- **Rekursion** (`hoh_exact.exact_table`): die gemeinsame Verteilung von (O, H) über den Zustand „Länge des Treffer-Laufs am Ende, höchstens k“, Wurf für Wurf; ein Durchgang liefert alle Längen bis n_max. Daraus E[H / O | O ≥ 1], P(O ≥ 1), E[O], E[H].
- **Aufzählung** (`hoh_exact.enumerate_stats`): alle 2^n Folgen (n ≤ 16, 65 536 Folgen), jede mit ihrer Wahrscheinlichkeit gewichtet, Gelegenheiten über Fenstersummen.
- **Simulation:** Folgen aus SplitMix64 (wörtliche Kopie `hoh_rng.py` der Vorlage `tools/templates/rng_splitmix64_exact.py`; `hoh_scenario.uniform_stream` zieht denselben Strom vektorisiert in numpy, Test: bitgleich). **5 feste Zufalls-Seeds** (11, 23, 37, 41, 59), das Band
  ist der kleinste bis größte Wert der Zufalls-Seeds.
- **Permutationstest** (`hoh_algorithm.permutation_test`): Prüfgröße = Mittel der Folgenschätzer; jede Folge wird einzeln zufällig umgeordnet (gleiche Trefferzahl), 99 Umordnungen (999 im wachsenden Beispiel), p = (1 + Zahl der mindestens so extremen Umordnungen) / (Umordnungen + 1),
  einseitig. Gültig unter „die Würfe jeder Folge sind austauschbar“ (Orakel-Test: exakter p-Wert aus allen Anordnungen).
- **Naive Prüfung** (`hoh_algorithm.naive_test`): dieselbe Prüfgröße gegen 0,5 gehalten, Standardfehler aus der Binomial-Näherung; „heiß“ bei z > 1,645, „kalt“ bei z < −1,645.
- **Teststärke** (`hoh_evaluation.power_curve`): je q in {0,50; 0,55; 0,60; 0,65; 0,70; 0,80} 5 Zufalls-Seeds zu je 20 Läufen (100 Läufe), je Lauf m Folgen der Länge n; gezählt wird, wie oft eine Regel „heiß“ (einseitig nach oben bei 5 %) meldet. Bei q = 0,50 ist das die Fehlerrate 1. Art (Type-I-Fehlerrate).
  Der Lauf ist auf n ≤ 100 und m ≤ 20 gedeckelt (Rechenzeit) und startet erst auf Klick.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; Simulation und Teststärke als Band (Summationsreihenfolge und Vergleiche können auf anderen Plattformen in der letzten Binärstelle abweichen), exakte Größen auf die gedruckte Stelle.

### Verzerrung über die Länge (faire Münze, k = 3; exakt = Rekursion, Simulation = 5 Zufalls-Seeds × 4 000 Folgen je Länge)

| Länge n | exakte Erwartung des Folgenschätzers | Simulation (Mittel der Zufalls-Seeds) | Anteil der Folgen mit Gelegenheit | Test |
|---|---|---|---|---|
| 10 | 0,353 | 0,348 | – | `test_exact_expectation_k3_fair_by_length`, `test_simulation_bands_contain_the_exact_expectation_in_five_of_five_lengths` |
| 20 | 0,361 | 0,364 | 76,8 % (E[O] = 2,125) | `test_opportunities_and_defined_share_at_k3` |
| 50 | 0,418 | 0,419 | – | dieselben |
| 100 | 0,460 | 0,459 | – | dieselben |
| 200 | 0,481 | 0,481 | E[O] = 24,6 | dieselben |

Die exakte Erwartung liegt in **5 von 5** simulierten Längen im Band der 5 Zufalls-Seeds; größte Abweichung Simulation gegen exakt unter 0,005. Die wahre Quote 0,5 wird bei n = 200 nicht erreicht (Verzerrung −0,019).
**Die Kurve ist nicht monoton in n:** Bei n = k + 1 gibt es nur eine mögliche Gelegenheit, dort ist der Schätzer unverzerrt (0,5); am niedrigsten liegt er bei k = 1 bei n = 4 (0,405), k = 2 bei n = 8 (0,370), k = 3 bei n = 12 (0,351), k = 5 bei n = 26 (0,330)
(`test_bias_is_not_monotone_in_length_largest_length_per_k`).

### Abhängigkeit von k (faire Münze, exakt)

| | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|
| n = 20: Erwartung des Folgenschätzers | 0,474 | 0,415 | 0,361 | 0,339 | 0,331 |
| n = 20: Anteil der Folgen mit Gelegenheit | 100,0 % | 97,9 % | 76,8 % | 45,8 % | 23,7 % |
| n = 50: Erwartung des Folgenschätzers | 0,490 | 0,467 | 0,418 | 0,367 | 0,338 |

Test: `test_k_dependence_at_n20_and_n50`. Bei diesen beiden Längen sinkt die Erwartung mit k; je größer k, desto weniger Folgen haben überhaupt eine Gelegenheit.

### Mehr Folgen: Mittel der Folgenschätzer gegen gepoolt (n = 20, k = 3, faire Münze, Simulation)

| m | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| Mittel der Folgenschätzer | 0,360 | 0,361 | 0,359 | 0,367 | 0,358 | 0,363 | 0,365 |
| gepoolt | 0,360 | 0,401 | 0,455 | 0,487 | 0,488 | 0,494 | 0,503 |

Je Zahl der Folgen gleich viele Folgen insgesamt (5 Zufalls-Seeds × 4 000). Das Mittel der Folgenschätzer bleibt bei der exakten Erwartung 0,361 (diese liegt in **6 von 7** Bändern der Zufalls-Seeds; bei m = 10 knapp außerhalb), der gepoolte Schätzer steigt; sein Band bei m = 100
(0,496 bis 0,507) enthält 0,5. Exakt gleich für m = 1, 2, 3 ist das Mittel der Folgenschätzer im Orakel-Test (n = 5, k = 1, alle 32^m Tupel von Folgen in Brüchen), der gepoolte Schätzer liegt dort für m = 1 gleich und für m = 2 und 3 näher an 1/2 (`test_more_folgen_do_not_change_the_mean_of_estimators_but_pooling_approaches_q`).
Test der Tabelle: `test_more_folgen_table`. **Der gepoolte Schätzer ist bei wenigen Folgen nicht besser:** bei m = 2 liegt er bei 0,401.

### Echte Abhängigkeit hinter der Verzerrung (exakt, n = 20, k = 3)

| Persistenz q | 0,50 | 0,55 | 0,60 | 0,65 | 0,70 |
|---|---|---|---|---|---|
| Erwartung des Folgenschätzers | 0,361 | 0,411 | 0,464 | 0,518 | 0,574 |
| Verzerrung gegen q | −0,139 | −0,139 | −0,136 | −0,132 | −0,126 |

Test: `test_dependence_is_hidden_by_the_bias_at_short_length`. Bei n = 50 und k = 2 sind es 0,467 (q = 0,50) und 0,569 (q = 0,60); `test_n50_k2_values`.
Ein Maß, das die Abhängigkeit zeigen soll, liegt hier bei einer wirklich persistenten Münze (q = 0,60) unter 0,5: Wer es mit 0,5 vergleicht, liest Gleichgültigkeit oder sogar „kalte Hand“.

### Aufzählung gegen Rekursion gegen Formel (n = 16, k = 2, faire Münze)

P(mindestens eine Gelegenheit) 0,951; E[O] = 3,500; E[H] = 1,750; E[H] / E[O] = 0,500; Erwartung des Folgenschätzers 0,400 (genau 0,3996). Alle Zeilen stimmen in Aufzählung und Rekursion auf 12 Stellen überein, die drei Formelzeilen ebenfalls
(`test_enumeration_n16_k2_values`; unabhängig in Brüchen: `tests/test_oracle_hot_hand.py`).

### Wachsendes Beispiel bei den Voreinstellungen (n = 20, k = 3, q = 0,5, m = 100, Zufalls-Seed 35)

80 der 100 Folgen haben mindestens eine Gelegenheit; Mittel der Folgenschätzer 0,339, gepoolt 0,495 (eine Stichprobe: die exakte Erwartung ist 0,361). Die erste Folge hat 12 Treffer in 20 Würfen und den Schätzer 0,667; ordnet man ihre Würfe 999-mal zufällig um, liegt der Schätzer im Mittel bei 0,436
(in 978 Umordnungen gibt es eine Gelegenheit), der einseitige p-Wert nach oben ist 0,294 (`test_default_page_example`).

### Teststärke: naive Prüfung gegen Permutationstest (n = 50, k = 2, 20 Folgen, 100 Läufe je q, 5 Zufalls-Seeds zu je 20 Läufen, 99 Umordnungen)

| Persistenz q | 0,50 | 0,55 | 0,60 | 0,65 | 0,70 | 0,80 |
|---|---|---|---|---|---|---|
| „heiß“: Permutationstest | 5 % | 47 % | 93 % | 100 % | 100 % | 100 % |
| „heiß“: naive Prüfung | 1 % | 18 % | 71 % | 95 % | 100 % | 100 % |
| „kalt“: Permutationstest | 4 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| „kalt“: naive Prüfung | 19 % | 1 % | 0 % | 0 % | 0 % | 0 % |

80 % Teststärke erreicht der Permutationstest ab q = 0,60, die naive Prüfung ab q = 0,65. Standardfehler einer Zahl bei 100 Läufen bis 5 Prozentpunkte (bei 5 % rund 2 Punkte); Test mit Band: `test_power_n50_k2_m20`.

Dieselbe Rechnung mit n = 20, k = 3 (kürzere Folgen, größeres k; `test_power_n20_k3_m20`):

| Persistenz q | 0,50 | 0,55 | 0,60 | 0,65 | 0,70 | 0,80 |
|---|---|---|---|---|---|---|
| „heiß“: Permutationstest | 3 % | 12 % | 27 % | 48 % | 74 % | 94 % |
| „heiß“: naive Prüfung | 0 % | 0 % | 3 % | 11 % | 39 % | 92 % |
| „kalt“: naive Prüfung bei q = 0,50 | 43 % | | | | | |

Hier erreicht der Permutationstest 80 % Teststärke erst bei q = 0,80 im Raster (bei q = 0,70 sind es 74 %); in beiden Tabellen bleibt die Fehlerrate 1. Art des Permutationstests bei q = 0,5 im Mittel über die 5 Zufalls-Seeds unter 7 % (je Zufalls-Seed bis 10 %; `test_false_alarm_rate_of_the_permutation_test_stays_near_five_percent_everywhere`).

## Befunde und Korrekturen gegenüber dem Plan

Der Plan (Vormessung M2, 2026-11-18) kannte nur die Simulation der Verzerrung. Alle Zahlen dieser README sind **neu gemessen** (SplitMix64 statt NumPy-Generator, exakte Rekursion statt 20 000 Simulationen je Länge).

- **Die Vormessung stimmt innerhalb ihrer Stichprobenstreuung:** sie nannte 0,359 / 0,418 / 0,459 / 0,480 für n = 20 / 50 / 100 / 200; exakt sind 0,361 / 0,418 / 0,460 / 0,481, größte Abweichung 0,002 (`test_the_plan_numbers_of_the_premeasurement_lie_within_the_sampling_error_of_the_exact_values`).
- **Der gepoolte Schätzer ist nicht „anders verzerrt“, sondern bei einer Folge identisch und bei vielen Folgen nahezu unverzerrt:** bei m = 1 dasselbe Maß, bei m = 2 noch bei 0,401, bei m = 100 bei 0,503 (Band 0,496 bis 0,507). Die Verzerrung des Mittels der Folgenschätzer verschwindet dagegen mit m nie.
- **Neu gegenüber dem Plan:** die Kurve ist **nicht monoton in n** (größte Verzerrung bei mittlerer Länge, nicht bei der kürzesten); die naive Prüfung gegen 0,5 hat bei fairer Münze eine **hohe „kalt“-Fehlalarmquote** (etwa 19 % und etwa 43 %); bei q = 0,65 und n = 20 liegt das Maß im Erwartungswert nur bei 0,518.
- **Deutung des Mechanismus:** Die Seite erklärt die Verzerrung als Ziehen ohne Zurücklegen aus dem Vorrat an Treffern einer Folge (Deutung, nicht einzeln gemessen). Gemessen ist die Größe, der Referenzwert des Permutationstests an einer Folge (Beispiel oben) und der Fehlalarm der naiven Prüfung.
- **Teststärke-Lauf gedeckelt:** n ≤ 100 und m ≤ 20; die Regler gehen für die übrigen Experimente bis n = 200 und m = 100. Über n = 100 und m = 20 hinaus wurde die Teststärke nicht gemessen.

## Abgrenzung zu Nachbar-Demos

| Demo | Was sie misst | Verhältnis zu dieser |
|---|---|---|
| [`siegtabelle-demo`](https://sebastianhanisch-siegtabelle-demo.streamlit.app/) | Wie oft der Stärkste in einer Tabelle vorn steht (Rauschen in Rundenturnieren) | Rauschen in einer Rangliste; hier die Verzerrung eines Maßes für Abhängigkeit in Folgen von Würfen |
| [`rangintervalle-demo`](https://sebastianhanisch-rangintervalle-demo.streamlit.app/) | Wie sicher ein Rang ist (Intervalle, Shrinkage) | Unsicherheit eines Rangs; hier Unsicherheit und Verzerrung eines Schätzers einer Quote |

Beide Demos behandeln Rauschen in Ranglisten; diese Demo behandelt, wie ein Maß für Abhängigkeit ohne Rauschen systematisch daneben liegt (Erwartungswert, nicht Streuung) und wie ein Test mit seiner eigenen Referenzverteilung damit umgeht. Sie kennt keinen Rang.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Münze mit Persistenz q, gleiche Quote 0,5 für alle Folgen** | Echte Schützen haben unterschiedliche Quoten und unterschiedlich schwere Würfe; ob sich das mit einer Abhängigkeit vermischt, sagt die Seite nicht. | – |
| **Gedächtnis nur über den Wurf davor (Markow-Kette erster Ordnung)** | Andere Formen von Abhängigkeit (längeres Gedächtnis, wechselnde Form) sind nicht gerechnet; ob der Permutationstest sie findet, ist nicht gemessen. | Wurzel der Linie „Wann zählt die Reihenfolge?“ (dort steht, wann Abhängigkeit überhaupt einen Reihenfolgeeffekt bewirkt) |
| **Der Folgenschätzer ist das Maß** | Eine Rangliste aus Treffern kommt mit Rauschen: Auch ohne Abhängigkeit schwanken Ränge. | `siegtabelle-demo`, `rangintervalle-demo` |
| **Ein Permutationstest je Folgenmenge, 5 % einseitig** | Wer mehrere Folgenmengen einzeln testet und die auffälligste herausgreift, hat die 5 % verlassen (Mehrfachtests); das ist nicht gerechnet. | – |
| **Teststärke aus 100 Läufen je q** | Die Teststärke ist auf einige Prozentpunkte genau (das Band zeigt die Streuung der Zufalls-Seeds); ein Wert um 80 % kann bei anderen Zufalls-Seeds auf der anderen Seite der Schwelle liegen. | – |

- **Nur das Zufallsmodell.** Keine Aussage über Basketball, über echte Würfe, Schützen oder Daten; die Seite arbeitet mit selbst erzeugten Folgen.
- **Mechanismus nur gedeutet.** Warum das Mittel zu niedrig liegt, steht als Deutung (Ziehen ohne Zurücklegen); die Seite misst die Größe und den Referenzwert, nicht die Ursache einzeln.
- **Der Permutationstest prüft nur Austauschbarkeit der Würfe innerhalb einer Folge.** Er verwirft sie, soweit eine Abhängigkeit die Prüfgröße (das Mittel der Folgenschätzer) verschiebt; gemessen ist nur die Markow-Persistenz q, andere Formen der Abhängigkeit findet er nicht unbedingt.
- **Teststärke gilt für die gemessenen Einstellungen** (n = 50, k = 2, 20 Folgen und n = 20, k = 3, 20 Folgen, 5 % einseitig, 99 Umordnungen); andere Längen und Folgenzahlen sind über die Regler messbar, aber nicht in dieser README.
- **Darstellung im Browser nicht geprüft:** Farben, Rahmen der Kästchen und Gedrängtheit bei n = 200 sind mit AppTest nicht sichtbar und in keinem echten Browser angesehen; auch KaTeX-Fehler im 📐-Expander sieht nur der Browser.
- **Linux-/Docker-Prüfung und Streamlit-Deploy** sind nicht gelaufen.

## Bewusst nicht umgesetzt

- **Reale Daten und Literaturvergleich:** keine Felddaten, keine Aussage über Studien; die Literatur ist nicht im Volltext gelesen.
- **Andere Maße:** Differenz der Trefferquote nach k Treffern und nach k Fehlwürfen, Bayes-Schätzer, bedingte Maße mit festem Trefferanteil: nicht umgesetzt.
- **Heterogene Schützen** (verschiedene Quoten je Folge) und längeres Gedächtnis: nicht modelliert.
- **PDF-Export:** keiner (kein Ergebnisbericht nötig; die Seite ist die Messung).

## Literatur (nicht im Volltext gelesen)

Die Seite stützt keine ihrer Zahlen auf diese Arbeiten; sie benennt nur den Kontext: Gilovich, Vallone, Tversky (1985) und Miller, Sanjurjo (2018) zur „Hot Hand“. Beide Arbeiten sind **nicht im Volltext gelesen**, Titel und Fundstellen sind hier nicht aufgeführt, weil sie nicht gegen die Quelle geprüft sind.

## Verifikation

- **Kern:** Gelegenheiten und Schätzer gegen Handrechnung (`tests/conftest.py`, `tests/test_algorithm.py`: Mini-Folge 1 1 1 0 1 1 1), vektorisierte gegen Schleifenfassung auf 300 Zufallsfolgen, naive Prüfung und Permutationsumordnung von Hand.
- **Exakte Werte von Hand:** n = 3, k = 1 (5/12), n = 4, k = 2 (5/12), q = 3/5 (18/35) in `tests/test_exact.py`; Rekursion gegen Aufzählung auf einem Raster (k = 1 bis 4, vier Werte q, n bis 12).
- **Orakel** (`tests/test_oracle_hot_hand.py`, anderer Rechenweg ohne numpy und ohne Demo-Module: Python-Schleifen mit exakten Brüchen): Erwartung des Schätzers, E[O], E[H] und die geschlossenen Formeln gegen die Bruch-Aufzählung (n bis 11, k bis 3, vier Werte q); Simulation gegen exakt mit vier Standardfehlern;
  Zahl der Folgen durch Aufzählung aller Folgenpaare und -tripel in Brüchen; Permutationstest gegen den exakten p-Wert aus allen Anordnungen und seine Gültigkeit (Irrtumswahrscheinlichkeit höchstens 5 %, n = 12, k = 2, alle 4 096 Folgen).
- **Eigenschaftstests** (`tests/test_eigenschaften_hot_hand.py`): Reihenfolge der Folgen, Verdoppeln, Zusammenlegen, k nie mehr Gelegenheiten, Spiegelung der Wahrscheinlichkeiten, vollständige Persistenz, Monotonie in q auf dem Raster, Unverzerrtheit bei n = k + 1.
- **Zweig-Tests** (Nullspalte = Bug): k und q ändern die Messreihen wirklich, gepoolt steigt mit m, die naive Prüfung meldet „kalt“ häufiger als der Permutationstest (`tests/test_evaluation.py`).
- **Zufall:** Kopie der SplitMix64-Vorlage mit Kopie-Test; der vektorisierte Strom bitgleich zum Generator mit Zustand (`tests/test_scenario.py`).
- **Presets, Layout, Oberfläche:** `tests/test_presets.py`, `tests/test_layout.py`, `tests/test_app.py` (jeder Knopf, jeder Regler an beiden Grenzen), `tests/test_app_content.py` (angezeigte Zahlen aus der Rechnung bei den Reglern, Beispiel-Schritt, Teststärke erst auf Klick), Footer-Test.
- **Verdikte** als reine Funktionen (`tests/test_verdicts.py`).

**196 Tests**, grün (Stand 2026-11-18, Python 3.14): 39 s seriell (ein Lauf, die Zeit hängt stark von der Auslastung des Rechners ab); den größten Teil der Zeit brauchen die AppTest-Läufe (jeder Knopf, jeder Regler an beiden Grenzen, Layout). Die Laufzeit ist nicht als Test festgeschrieben.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `hoh_algorithm.py` | Kern: Gelegenheiten, Schätzer, Permutationstest, naive Prüfung |
| `hoh_exact.py` | Rekursion, Aufzählung, geschlossene Formeln |
| `hoh_constants.py` | Regler-Grenzen, feste Annahmen (Zufalls-Seeds, Raster), Farben, Presets |
| `hoh_evaluation.py` | Analyse, Messreihen, Verdikte, Tabellen der App |
| `hoh_layout.py` | Fester Kopf (Stylesheet) |
| `hoh_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `hoh_rng.py` | SplitMix64 (wörtliche Kopie der Vorlage) |
| `hoh_scenario.py` | Folgen aus der Münze mit Persistenz, vektorisierter Zufallsstrom |
| `hoh_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `ruff.toml` | Lint-Einstellung (O und H als Namen der Größen erlaubt) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Exakt, Orakel, Eigenschaften, Auswertung, Claims, Verdikte, Presets, Layout, App, Footer, Laufzeit-Importe, Kopie |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Wettbewerbsverzerrung: Reihenfolge und Anreize](https://sebastianhanisch.net/konzepte-wettbewerbsverzerrung.html).
