* Uebersetzung sonderfall
* Schwieriger
* Nutzen von einem beliebigen grossen dataset (~20k)
  * Erweitern mit Variante 1 -> 40k Data
  * Gucken wie model mit nur orig data, dann mit allen data, dann nur mit augmented
* ML: Huggingface top
  * BERT zb guter anfangspunkt
  * Einfach eingeben, zig tutorials
  * Unterteilen in Training/Testset
  * Training und Testset immer gleich lassen 

### 07/07/2022
* Datasplits -> Eigentlich 3 Splits (Train, Test, Validation)
  * Validation bei Training mitverwenden
  * Test Set bei Training gar nicht verwenden -> Eval, nie veraendern
  * Am ende predict methode nutzen -> Darauf dann metriken evaluieren
  * Random seed als const fuer den Split um Methoden vergleichbar zu halten
  * Seed festsetze: Python random, numpy, pytorch(deterministic results)
  * Forschungsprojekt Praesi: 13.09, 20-25 min, Folien auf Englisch, Praesi auf Deutsch 

### 04/08/2022
* Wenn Graphen dann y-achse immer gleich
* matplotlib interval tick integers googlen
* Analyseansatz: Gucken, welche saetze werden falsch vorrausgesagt um schwaechen von baseline und augmentierung zu filtern -> gezielte augmentierung
* Haeufigste falsch vorrausgesagte Woerter finden
* Evaluierungsdaten haendisch falsch tippen

### Herausforderungen
* Tokenization -> Word Embeddings
* Ordentliches Einsetzen der neuen Woerter
* Anpassungen an real life data

### 25/11/2022
* Visualisierung ueber Balkendiagramme
  * Standarddatensatz als Referenzwert
  * Visualisieren welche Augmentierung wieviel +-
  * Aufsummieren ueber Datensaetze
* Den Datensatz, wo es anscheinend am besten funktioniert noch genauer augmentieren
  * 10, 20, 30... % und auch die Konkatenierung anpassen