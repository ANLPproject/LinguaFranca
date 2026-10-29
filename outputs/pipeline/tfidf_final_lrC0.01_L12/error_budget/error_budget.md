# Error budget and case studies, 2Wiki (`pipeline_tfidf_v2_r2_shared_dense_probes_lr_C0.01_L12`)

n = 500 questions; one restart; correct = token F1 ≥ 0.5. Each wrong final answer is assigned to the first stage that broke. 'Failing hop' = the first hop labelled as failing (Phase 1 labels; relabeled where the trajectory changed). 'Missing gold paragraph' = a gold title of the question not in the first-stage context.

## 1. Error budget (share of all questions)

| stage | Probe (ours) | Judge Llama-3.2-3B-Instruct | Oracle trigger |
|---|---|---|---|
| no hop failed | 28 (5.6%) | 34 (6.8%) | 37 (7.4%) |
| detector missed the failing hop | 43 (8.6%) | 127 (25.4%) | 2 (0.4%) |
| detector flagged the wrong hop | 24 (4.8%) | 63 (12.6%) | 0 (0.0%) |
| restarted: evidence was already in the context | 21 (4.2%) | 6 (1.2%) | 36 (7.2%) |
| restarted: retrieval missed the gold paragraph | 81 (16.2%) | 33 (6.6%) | 109 (21.8%) |
| restarted: gold paragraph retrieved, answer still wrong | 34 (6.8%) | 10 (2.0%) | 41 (8.2%) |
| **all errors** | **231 (46.2%)** | **273 (54.6%)** | **225 (45.0%)** |

Questions with a failing hop: 351 ({'failing hop, gold paragraph missing': 262, 'failing hop, evidence present': 89}).

## 2. Case studies

Up to 3 candidates per category (shortest trajectories first). Probe P(fail): calibrated, τ = 0.650; judge P(fail): Llama-3.2-3B-Instruct, θ = 0.5.

### A. Confident wrong hop: the probe flags it, the judge does not, the restart retrieves the missing paragraph and fixes the answer

21 questions qualify.

**Q:** When was the performer of song Doesn'T Really Matter born?  
**Gold:** May 16, 1966 (bridge: Janet Jackson); hop labels {'1': 0, '2': 1}  
**Context (top-5 TF-IDF):** Doesn't Really Matter, I Can't Take It (Dillon Francis song), Bernie Bonvoisin, Billy Milano, O Valencia!; missing gold: Janet Jackson  
- hop 1: "From the context, I find that the performer of the song "Doesn't Really Matter" is Janet Jackson." — probe 0.04, judge 0.00
- hop 2: "From the context, I find that Janet Jackson is an American singer." — probe 0.90, judge 0.07
- first answer: **American**
- restart at hop 2; retrieved: Janet Jackson, Janet Lippincott, Born to Sing (1942 film)
- final answer: **May 16, 1966** (correct)

**Q:** Who is the spouse of the performer of song You'Re Still Here?  
**Gold:** Tim McGraw (bridge: Faith Hill); hop labels {'1': 0, '2': 1}  
**Context (top-5 TF-IDF):** You're Still Here, I'm Still Here (Follies song), Gertrude of Bavaria, Mehdi Abrishamchi, Sophia Magdalena of Denmark; missing gold: Faith Hill  
- hop 1: "From the context, I find that the performer of the song "You'Re Still Here" is Faith Hill." — probe 0.06, judge 0.18
- hop 2: "From the context, I find that Faith Hill is an American country music artist." — probe 0.81, judge 0.18
- first answer: **American**
- restart at hop 2; retrieved: Faith Hill, Stealing Kisses, Faith Page
- final answer: **Tim McGraw** (correct)

**Q:** What is the place of birth of the director of film Mr. Lemon Of Orange?  
**Gold:** Rice Lake, Wisconsin (bridge: John G. Blystone); hop labels {'1': 0, '2': 1}  
**Context (top-5 TF-IDF):** Mr. Lemon of Orange, Ian Barry (director), Peter Levin, Robert J. Cohen, Sixx Orange; missing gold: John G. Blystone  
- hop 1: "From the context, I find that the director of film Mr. Lemon of Orange is John G. Blystone." — probe 0.04, judge 0.50
- hop 2: "From the context, I find that John G. Blystone is an American director of film." — probe 0.95, judge 0.35
- first answer: **American**
- restart at hop 2; retrieved: John G. Blystone, John N. Smith, John Paddy Carstairs
- final answer: **Rice Lake, Wisconsin** (correct)

### B. The probe misses a failing hop

43 questions qualify.

**Q:** What is the date of birth of the director of film Kaamyab?  
**Gold:** 23 May 1942 (bridge: K. Raghavendra Rao); hop labels {'1': 1, '2': 1}  
**Context (top-5 TF-IDF):** Ian Barry (director), Peter Levin, K. Raghavendra Rao, Brian Saunders (weightlifter), Terence Robinson; missing gold: Kaamyab  
- first answer: **Unknown**
- final answer: **Unknown** (wrong)

**Q:** What is the place of birth of the director of film Harjaee?  
**Gold:** India (bridge: Ramesh Behl); hop labels {'1': 1, '2': 1}  
**Context (top-5 TF-IDF):** Dana Blankstein, Ian Barry (director), Peter Levin, Jason Moore (director), Ramesh Behl; missing gold: Harjaee  
- hop 1: "There is no mention of a director named Harjaee in the context." — probe 0.33, judge 0.04
- first answer: **Unknown**
- final answer: **Unknown** (wrong)

**Q:** Where was the mother of Anna Freud born?  
**Gold:** Hamburg (bridge: Martha Bernays); hop labels {'1': 1, '2': 1}  
**Context (top-5 TF-IDF):** Minamoto no Chikako, Trinidad Tecson, Anna Freud, Esther Freud, Paul Freud; missing gold: Martha Bernays  
- hop 1: "From the context, I find that Anna Freud was born in Vienna." — probe 0.10, judge 0.16
- hop 2: "From the context, I find that Vienna is the birthplace of Anna Freud." — probe 0.58, judge 0.03
- first answer: **Vienna**
- final answer: **Vienna** (wrong)

### C. A restart that breaks a correct first answer (check the hop labels: often a failing hop the model recovered from on its own)

10 questions qualify.

**Q:** What nationality is the director of film A Time To Love (Film)?  
**Gold:** Chinese (bridge: Huo Jianqi); hop labels {'1': 0, '2': 1}  
**Context (top-5 TF-IDF):** Time to Love, A Time to Live and a Time to Love, A Time to Love (film), Dana Blankstein, To Love; missing gold: Huo Jianqi  
- hop 1: "From the context, I find that A Time to Love is a 2005 film directed by Huo Jianqi." — probe 0.09, judge 0.34
- hop 2: "From the context, I find that Huo Jianqi is the director of A Time to Love (film)." — probe 0.87, judge 0.04
- first answer: **Chinese**
- restart at hop 2; retrieved: Time of Love, A Time to Love and a Time to Die, Amelie or The Time to Love
- final answer: **Huo Jianqi** (wrong)

**Q:** What is the date of death of the director of film Tunes Of Glory?  
**Gold:** 16 June 2010 (bridge: Ronald Neame); hop labels {'1': 0, '2': 1}  
**Context (top-5 TF-IDF):** Ian Barry (director), Peter Levin, Tunes of Glory, Ronald Neame, Thomas Scott (diver); missing gold: –  
- hop 1: "From the context, I find that Ronald Neame is the director of the film Tunes of Glory." — probe 0.01, judge 0.12
- hop 2: "From the context, I find that Ronald Neame was born on 23 April 1911." — probe 0.89, judge 0.27
- first answer: **16 June 2010**
- restart at hop 2; retrieved: Eugen York, Roschdy Zem, David Neves
- final answer: **23 April 1911** (wrong)

**Q:** When did the director of film Sold Man die?  
**Gold:** September 2, 1987 (bridge: Remzi Aydın Jöntürk); hop labels {'1': 0, '2': 1}  
**Context (top-5 TF-IDF):** Ian Barry (director), Peter Levin, Sold Man, Remzi Aydın Jöntürk, Bill Smith (footballer, born 1897); missing gold: –  
- hop 1: "From the context, I find that Remzi Aydın Jöntürk is the director of the film Sold Man." — probe 0.01, judge 0.38
- hop 2: "From the context, I find that Remzi Aydın Jöntürk was born on September 15, 1936." — probe 0.80, judge 0.80
- first answer: **September 2, 1987**
- restart at hop 2; retrieved: The Man Who Sold Himself (1959 film), The Man Who Sold Himself (1925 film), Faouzi Bensaïdi
- final answer: **September 15, 1936** (wrong)

### D. Correct detection, retrieval fails

81 questions qualify.

**Q:** Which country Alphonso Of Brienne's father is from?  
**Gold:** Jerusalem (bridge: John of Brienne); hop labels {'1': 0, '2': 1}  
**Context (top-5 TF-IDF):** John I of Brienne, Count of Eu, John, Count of Brienne, Alphonso of Brienne, Erard I, Count of Brienne, Hugh, Count of Brienne; missing gold: John of Brienne  
- hop 1: "From the context, I find that Alphonso of Brienne's mother is Berengaria of León." — probe 0.22, judge 0.87
- hop 2: "From the context, I find that Berengaria of León is from León." — probe 0.98, judge 0.29
- first answer: **León**
- restart at hop 2; retrieved: Berengaria of León, Berengaria of Barcelona, Berengaria of Portugal
- final answer: **León** (wrong)

**Q:** What is the award that the composer of film Pokiri received?  
**Gold:** Filmfare Awards South (bridge: Mani Sharma); hop labels {'1': 1, '2': 1}  
**Context (top-5 TF-IDF):** Pokiri, Walter Ulfig, Bert Grund, Abe Meyer, Amedeo Escobar; missing gold: Mani Sharma  
- hop 1: "From the context, I find that the film Pokiri won four Nandi Awards." — probe 0.90, judge 0.17
- hop 2: "From the context, I find that the Nandi Awards are a type of award." — probe 0.99, judge 0.28
- first answer: **Nandi Awards**
- restart at hop 1; retrieved: Poocha Sanyasi, Poomani, List of film music awards
- final answer: **Nandi Awards** (wrong)

**Q:** Where was the husband of Sonja Barth born?  
**Gold:** Trondheim (bridge: Edvard Kaurin Barth); hop labels {'1': 0, '2': 1}  
**Context (top-5 TF-IDF):** Stan Rice, Marisa Ferretti Barth, Sonja Barth, Bjørn Barth, Paul Barth (judoka); missing gold: Edvard Kaurin Barth  
- hop 1: "From the context, I find that Sonja Barth was married to Edvard Kaurin Barth." — probe 0.08, judge 0.27
- hop 2: "From the context, I find that Edvard Kaurin Barth was born in 1913." — probe 0.87, judge 0.35
- first answer: **1913**
- restart at hop 2; retrieved: Barbara Antonie Barth, Heidrun Barth, Friederike Barth
- final answer: **1913** (wrong)
