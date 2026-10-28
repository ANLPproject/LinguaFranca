# Date-format relabel check (report only; no labels were changed)

Rule (same for both settings): re-check only rows labelled `unmatched_gold` among generated hops 1-2. If the gold is a date, the hop is counted correct when the text contains a date with the same year and the same month and day (as far as the gold specifies them). Year-only statements for a full-date gold stay failures.

## full10

* labelled generated hop rows: 8089; labelled failure: 1994; `unmatched_gold`: 1994
* of those, gold is a date: 736
* **false failures fixed by the date rule: 13** (0.16% of all rows; 0.7% of all failure labels); in the test split: 3
* year-only statements for a full-date gold (left as failures): 140
* hop failure rate before -> after: hop 1 8.9% -> 8.9%, hop 2 40.6% -> 40.3%
* questions whose first-failure hop would change: 13

Examples of proposed flips:

- hop 2 | gold: `1 December 1888` | text: From the context, I find that Ray Taylor was born on December 1, 1888.
- hop 2 | gold: `25 February 1996` | text: From the context, I find that Vehbi Koç died on February 25, 1996.
- hop 2 | gold: `26 May 1991` | text: From the context, I find that Eugène Lourié died on May 26, 1991.
- hop 2 | gold: `13 October 1932` | text: From the context, I find that Dušan Makavejev was born on October 13, 1932.
- hop 2 | gold: `30 June 1931` | text: From the context, I find that William James Craft was born in 1886 and died on June 30, 1931.
- hop 2 | gold: `19 June 2009` | text: From the context, I find that Vicente Ferrer Moncho died on June 19, 2009.
- hop 2 | gold: `16 October 1938` | text: From the context, I find that Nico was born on October 16, 1938.
- hop 2 | gold: `16 April 1889` | text: From the context, I find that Charlie Chaplin was born on April 16, 1889.

## tfidf

* labelled generated hop rows: 7911; labelled failure: 3313; `unmatched_gold`: 3313
* of those, gold is a date: 1046
* **false failures fixed by the date rule: 1** (0.01% of all rows; 0.0% of all failure labels); in the test split: 0
* year-only statements for a full-date gold (left as failures): 60
* hop failure rate before -> after: hop 1 19.5% -> 19.5%, hop 2 64.7% -> 64.7%
* questions whose first-failure hop would change: 1

Examples of proposed flips:

- hop 2 | gold: `21 October 1969` | text: From the context, I find that Jonathan Glatzer was born on October 21, 1969.
