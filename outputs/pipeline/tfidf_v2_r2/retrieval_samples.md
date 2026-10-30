# What comes out after retrieval — real examples (`pipeline_tfidf_v2_r2`)

Main system: one shared probe, repair v2 (entity-led hop-2 query + cross-encoder re-ranking), one restart. Each example shows the model's first answer, where the probe stopped it, what the retriever returned (★ = a gold paragraph), and the restarted answer.

## How often does the repair retrieval find the missing gold paragraph? (2Wiki, probe-triggered restarts)

- Restarts triggered: **300** of 500 questions.
- Restarts where a gold paragraph was missing from the prompt: **228**; the top-3 retrieved paragraphs contain a missing gold paragraph in **113 (50%)**.

| gold paragraph retrieved? | n | answer fixed (wrong → right) | broken (right → wrong) | still wrong | still right |
|---|---|---|---|---|---|
| yes | 113 | 76 | 1 | 30 | 6 |
| no | 115 | 4 | 2 | 105 | 4 |

Restarts where both gold paragraphs were already in the prompt: 72 (fixed 12, broken 8).

Note: 296 of 2977 restarts (10%) — and a similar share of first answers — never write `<answer>`: the model keeps adding `<hopN>` steps until the 400-token cap. They count as wrong.

## Fixed — gold paragraph retrieved, hop-1 failure

**Q:** Where did Desiderata Of The Lombards's mother die?  
**Gold answer:** Liège  · gold paragraphs: Ansa, Queen of the Lombards, Desiderata of the Lombards

*First-stage context (TF-IDF top-5):* ★ Desiderata of the Lombards, Gundeberga, Gundeberga, Tato, Alboin

*First answer (no retrieval):*
```
<hop1>From the context, I find that Desiderius King of the Lombards's wife, Ansa, Queen of the Lombards, died no information is given about her death.</hop1>
<hop2>From the context, I find that Ansa, Queen of the Lombards, was the daughter of duke Garibald I of Bavaria.</hop2>
<answer>(none — ran to the token cap)</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.91, hop 2 = 0.86 (τ = 0.64) → **stopped at `</hop1>`**

*Retrieval query:* `Where did Desiderata Of The Lombards's mother die? From the context, I find that Desiderius King of the Lombards's wife, Ansa, Queen of the Lombards, died no information is given about her death.`

*Retrieved (top 3, prepended to the context):*
- **★ Ansa, Queen of the Lombards** — Ansa or Ansia (died after 774) belonged to an aristocratic family of Brescia and was the wife of Desiderius (756-774), King of the Lombards. The Latin name does …
- **Despoina** — In Greek mythology, Despoina, Despoena or Despoine, was the daughter of Demeter and Poseidon and sister of Arion. She was the goddess of mysteries of Arcadian c …
- **Basilina** — Basilina( died 332) was the wife of Julius Constantius and the mother of Roman Emperor Julian, who in her honour gave the name" Basilinopolis" to a city in Bith …

*Restarted answer:*
```
<hop1>From the context, I find that Ansa, the mother of Desiderata of the Lombards, died after 774.</hop1>
<hop2>From the context, I find that Ansa died in a location where she was locked up with her husband and daughter without a name, maybe Desiderata, at a monastery of the Carolingian Empire, in Liège or Corbie.</hop2>
<answer>Liège or Corbie</answer>   → ✅
```

---

**Q:** Where did the director of film The Great Mike die?  
**Gold answer:** Hollywood  · gold paragraphs: The Great Mike, Wallace Fox

*First-stage context (TF-IDF top-5):* ★ The Great Mike, Dana Blankstein, Ian Barry (director), Peter Levin, The Time, the Place and the Girl (1946 film)

*First answer (no retrieval):*
```
<hop1>From the context, I find that the director of the 1944 film The Great Mike is Wallace Fox.</hop1>
<hop2>From the context, I find that Wallace Fox is American.</hop2>
<answer>Unknown</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.82, hop 2 = 0.86 (τ = 0.64) → **stopped at `</hop1>`**

*Retrieval query:* `Where did the director of film The Great Mike die? From the context, I find that the director of the 1944 film The Great Mike is Wallace Fox.`

*Retrieved (top 3, prepended to the context):*
- **Mike (film)** — Mike is a 1926 American drama film directed by Marshall Neilan. The film is a modest production, featuring Sally O'Neil and William Haines.
- **★ Wallace Fox** — Wallace Fox (March 9, 1895 – June 30, 1958) was an American film director. He directed 84 films between 1927 and 1953. He was born in Purcell, Oklahoma, and die …
- **Behind the Mike (film)** — Behind the Mike is a 1937 American comedy film directed by Sidney Salkow, which stars William Gargan, Judith Barrett, Don Wilson, and Sterling Holloway. The scr …

*Restarted answer:*
```
<hop1>From the context, I find that Wallace Fox is an American film director.</hop1>
<hop2>From the context, I find that Wallace Fox died in Hollywood, California.</hop2>
<answer>Hollywood, California</answer>   → ✅
```

---

## Fixed — gold paragraph retrieved, hop-2 failure

**Q:** Where did the director of film Love Begins At 20 die?  
**Gold answer:** Oxnard  · gold paragraphs: Frank McDonald (director), Love Begins at 20

*First-stage context (TF-IDF top-5):* ★ Love Begins at 20, When Love Begins, Ian Barry (director), Peter Levin, Simon Van Booy

*First answer (no retrieval):*
```
<hop1>From the context, I find that the director of film Love Begins At 20 is Frank McDonald.</hop1>
<hop2>From the context, I find that the director of film Love Begins At 20 is an American.</hop2>
<answer>There is no information in the context about the death of the director of film Love Begins At 20.</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.04, hop 2 = 0.98 (τ = 0.64) → **stopped at `</hop2>`**

*Retrieval query:* `Frank McDonald. Where did the director of film Love Begins At 20 die? From the context, I find that the director of film Love Begins At 20 is an American.`

*Retrieved (top 3, prepended to the context):*
- **★ Frank McDonald (director)** — Frank McDonald (November 9, 1899 Baltimore, Maryland – March 8, 1980 Oxnard, California) was an American film and television director, active from 1935 to 1966. …
- **Love (1920 film)** — Love is an extant American 1920 silent era romance drama film starring Louise Glaum, James Kirkwood, and Joseph Kilgour. Directed by Wesley Ruggles and produced …
- **Love Never Dies (1921 film)** — Love Never Dies is a 1921 American silent drama film directed by King Vidor. Prints of the film survive in several film archives.

*Restarted answer:*
```
<hop1>From the context, I find that the director of film Love Begins At 20 is Frank McDonald.</hop1>
<hop2>From the context, I find that Frank McDonald was an American film and television director.</hop2>
<answer>Oxnard, California</answer>   → ✅
```

---

**Q:** What is the date of death of Olimpia Giustiniani's husband?  
**Gold answer:** 28 November 1685  · gold paragraphs: Maffeo Barberini (1631–1685), Olimpia Giustiniani

*First-stage context (TF-IDF top-5):* ★ Olimpia Giustiniani, Thomas Scott (diver), Theodred II (Bishop of Elmham), Harry Wainwright (footballer), Bill Smith (footballer, born 1897)

*First answer (no retrieval):*
```
<hop1>From the context, I find that Olimpia Giustiniani's husband is Maffeo Barberini, Prince of Palestrina.</hop1>
<hop2>From the context, I find that Maffeo Barberini is the husband of Olimpia Giustiniani.</hop2>
<answer>Unknown</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.03, hop 2 = 0.99 (τ = 0.64) → **stopped at `</hop2>`**

*Retrieval query:* `Maffeo Barberini. What is the date of death of Olimpia Giustiniani's husband? From the context, I find that Maffeo Barberini is the husband of Olimpia Giustiniani.`

*Retrieved (top 3, prepended to the context):*
- **★ Maffeo Barberini (1631–1685)** — Maffeo Barberini (19 August 1631 – 28 November 1685) was an Italian nobleman of the Barberini and "Prince of Palestrina". He was appointed Gonfalonier of the Ch …
- **Olimpia Ludovisi** — Olimpia Ludovisi (1656 – 27 November 1700) was the eldest daughter of Niccolò Ludovisi and his third wife Costanza Pamphili, sister of Vatican cardinal Camillo  …
- **Clara Maffei** — Elena Clara Antonia Carrara Spinelli( 13 March 1814, in Bergamo – 13 July 1886, in Milan) was an Italian woman of letters and backer of the Risorgimento, usuall …

*Restarted answer:*
```
<hop1>From the context, I find that Olimpia Giustiniani was the wife of Maffeo Barberini, Prince of Palestrina.</hop1>
<hop2>From the context, I find that Maffeo Barberini died on 28 November 1685.</hop2>
<answer>28 November 1685</answer>   → ✅
```

---

## Gold paragraph retrieved, answer still wrong

**Q:** What is the date of death of the director of film The French Lieutenant'S Woman (Film)?  
**Gold answer:** 25 November 2002  · gold paragraphs: Karel Reisz, The French Lieutenant's Woman (film)

*First-stage context (TF-IDF top-5):* Theodred II (Bishop of Elmham), John Bloom (film editor), Don Sharpe, Assheton Gorton, Ann Mollo

*First answer (no retrieval):*
```
<hop1>From the context, I find that the film "The French Lieutenant's Woman" has a director, but the director's name is not specified.</hop1>
<hop2>From the context, I find that the film "The French Lieutenant's Woman" has a nomination for an Academy Award in the category Best Art Direction, and the nominees for this category include Ann Mollo and Assheton Gorton.</hop2>
<answer>(none — ran to the token cap)</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.78, hop 2 = 0.89 (τ = 0.64) → **stopped at `</hop1>`**

*Retrieval query:* `What is the date of death of the director of film The French Lieutenant'S Woman (Film)? From the context, I find that the film "The French Lieutenant's Woman" has a director, but the director's name i …`

*Retrieved (top 3, prepended to the context):*
- **★ The French Lieutenant's Woman (film)** — The French Lieutenant's Woman is a 1981 British romantic drama film directed by Karel Reisz, produced by Leon Clore, and adapted by playwright Harold Pinter. It …
- **Jacques Poitrenaud** — Jacques Poitrenaud( born 22 May 1922, Lille – died 5 April 2005, Paris) was a French film director and actor.
- **Gérard Patris** — Gérard Patris( 1931 − 1990) was a French film director and television director who died in a car accident in 1990 in Chailles. His works include the documentary …

*Restarted answer:*
```
<hop1>From the context, I find that the director of the film "The French Lieutenant's Woman" is Karel Reisz.</hop1>
<hop2>From the context, I find that Karel Reisz is a French film director.</hop2>
<answer>5 April 2005</answer>   → ❌
```

---

**Q:** Which country the composer of song I'Ll Make A Man Of You is from?  
**Gold answer:** British  · gold paragraphs: Arthur Wimperis, I'll Make a Man of You

*First-stage context (TF-IDF top-5):* To Make a Man (Feel Like a Man), I'll Make a Man Out of You, I'll Make You Music, Alonso Mudarra, Walter Robinson (composer)

*First answer (no retrieval):*
```
<hop1>From the context, I find that the song "I'll Make a Man Out of You" is performed by Donny Osmond as the singing voice of Captain Li Shang in the film "Mulan".</hop1>
<hop2>From the context, I find that Donny Osmond is American.</hop2>
<answer>American</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.92, hop 2 = 0.43 (τ = 0.64) → **stopped at `</hop1>`**

*Retrieval query:* `Which country the composer of song I'Ll Make A Man Of You is from? From the context, I find that the song "I'll Make a Man Out of You" is performed by Donny Osmond as the singing voice of Captain Li S …`

*Retrieved (top 3, prepended to the context):*
- **★ I'll Make a Man of You** — "I'll Make a Man of You" is a World War I recruiting song that was sung across Britain in hopes of rallying young men to enlist in the military. It is sung from …
- **Working for the Man (Roy Orbison song)** — "Working for the Man" is a song composed and sung by rock and roll performer Roy Orbison. Released in 1962 as a double A-side with "Leah", it reached number 1 i …
- **Making a Man** — Making a Man is a 1922 American silent drama film produced by Adolph Zukor and Jesse L. Lasky, distributed by Paramount Pictures, and directed by Joseph Henaber …

*Restarted answer:*
```
<hop1>From the context, I find that the song "I'll Make a Man Out of You" is from Walt Disney Pictures' 36th animated feature film "Mulan" (1998).</hop1>
<hop2>From the context, I find that the film "Mulan" is an American film.</hop2>
<answer>United States</answer>   → ❌
```

---

## Gold paragraph NOT retrieved

**Q:** Who is the paternal grandfather of Helvis Of Cyprus?  
**Gold answer:** Hugh VIII of Lusignan  · gold paragraphs: Aimery of Cyprus, Helvis of Cyprus

*First-stage context (TF-IDF top-5):* Helvis of Brunswick-Grubenhagen, ★ Helvis of Cyprus, Kaya Alp, John of Lusignan, Henry Krause

*First answer (no retrieval):*
```
<hop1>From the context, I find that Helvis of Cyprus was the daughter of Amalric II of Jerusalem, King of Cyprus.</hop1>
<hop2>From the context, I find that Amalric II of Jerusalem, King of Cyprus, was the father of Helvis of Cyprus.</hop2>
<answer>King Hugh</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.18, hop 2 = 0.89 (τ = 0.64) → **stopped at `</hop2>`**

*Retrieval query:* `Helvis of Cyprus. Who is the paternal grandfather of Helvis Of Cyprus? From the context, I find that Amalric II of Jerusalem, King of Cyprus, was the father of Helvis of Cyprus.`

*Retrieved (top 3, prepended to the context):*
- **John II of Jerusalem** — John II of Jerusalem (1259 or ca. 1267 – 20 May 1285 in Nicosia) was the eldest son of Hugh III de Poitiers, king of Cyprus and Isabella of Ibelin. He succeeded …
- **Hugh IV of Cyprus** — Hugh IV( 1293×96 – 10 October 1359) was King of Cyprus from 31 March 1324 to his abdication, on 24 November 1358 and, nominally, King of Jerusalem, as Hugh II,  …
- **Helvis of Ibelin** — Helvis of Ibelin( after 1178 – before 1 June 1216) was a daughter of Balian of Ibelin and his wife, Maria Komnene, who was the dowager Queen of Jerusalem. Helvi …

*Restarted answer:*
```
<hop1>From the context, I find that Helvis of Cyprus was the daughter of Amalric II of Jerusalem, King of Cyprus.</hop1>
<hop2>From the context, I find that Amalric II of Jerusalem was married to Eschive d'Ibelin.</hop2>
<answer>(none — ran to the token cap)</answer>   → ❌
```

---

**Q:** What is the date of death of Sabina Franklyn's father?  
**Gold answer:** 31 October 2006  · gold paragraphs: Sabina Franklyn, William Franklyn

*First-stage context (TF-IDF top-5):* ★ Sabina Franklyn, Harry Wainwright (footballer), Thomas Scott (diver), Harry Johnson (wrestler), Fred Bradley (rower)

*First answer (no retrieval):*
```
<hop1>From the context, I find that Sabina Franklyn's father's name is William Franklyn.</hop1>
<hop2>From the context, I find that William Franklyn is the father of Sabina Franklyn.</hop2>
<answer>Not specified</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.01, hop 2 = 0.90 (τ = 0.64) → **stopped at `</hop2>`**

*Retrieval query:* `William Franklyn. What is the date of death of Sabina Franklyn's father? From the context, I find that William Franklyn is the father of Sabina Franklyn.`

*Retrieved (top 3, prepended to the context):*
- **William I, Marquis of Namur** — William I, Marquis of Namur," the Rich",( 1324 – 1 October 1391) was Count of Namur from 1337 until his death.
- **William of Bavaria-Munich** — William of Bavaria- Munich( 1435 – 1435) was a German nobleman. He was the son of Margaret of Cleves and her first husband William III, Duke of Bavaria. He was  …
- **William Fitzalan, 9th Earl of Arundel** — William Fitzalan, 9th Earl of Arundel, 6th Baron Maltravers( 23 November 1417 – 1487) was an English nobleman. Born on 23 November 1417, William was the second  …

*Restarted answer:*
```
<hop1>From the context, I find that William Franklyn is the father of Sabina Franklyn.</hop1>
<hop2>From the context, I find that William Franklyn's birth year is 1954.</hop2>
<answer>Not mentioned</answer>   → ❌
```

---

## Broken — right before the restart, wrong after

**Q:** When did the director of film Sold Man die?  
**Gold answer:** September 2, 1987  · gold paragraphs: Remzi Aydın Jöntürk, Sold Man

*First-stage context (TF-IDF top-5):* Ian Barry (director), Peter Levin, ★ Sold Man, ★ Remzi Aydın Jöntürk, Bill Smith (footballer, born 1897)

*First answer (no retrieval):*
```
<hop1>From the context, I find that Remzi Aydın Jöntürk is the director of the film Sold Man.</hop1>
<hop2>From the context, I find that Remzi Aydın Jöntürk was born on September 15, 1936.</hop2>
<answer>September 2, 1987</answer>   → ✅
```
*Probe:* P(fail) hop 1 = 0.01, hop 2 = 0.73 (τ = 0.64) → **stopped at `</hop2>`**

*Retrieval query:* `Remzi Aydın Jöntürk. When did the director of film Sold Man die? From the context, I find that Remzi Aydın Jöntürk was born on September 15, 1936.`

*Retrieved (top 3, prepended to the context):*
- **The Man Who Sold Himself (1959 film)** — The Man Who Sold Himself is a 1959 West German crime film directed by Josef von Báky and starring Hildegard Knef, Hansjörg Felmy and Antje Weisgerber. It was sh …
- **The Man Who Sold Himself (1925 film)** — The Man Who Sold Himself is a 1925 German silent film directed by Hans Steinhoff and starring Hans Mierendorff, Vivian Gibson and Olaf Fjord. The film's art dir …
- **Faouzi Bensaïdi** — Faouzi Bensaïdi( born 14 March 1967) is a Moroccan film director, actor, screenwriter and artist. His film" A Thousand Months" was screened in the Un Certain Re …

*Restarted answer:*
```
<hop1>From the context, I find that Remzi Aydın Jöntürk is the director of the 1976 Turkish film "Satılmış Adam (Sold Man)".</hop1>
<hop2>From the context, I find that Remzi Aydın Jöntürk was born on September 15, 1936.</hop2>
<answer>September 15, 1936</answer>   → ❌
```

---

**Q:** Who is George Carew, 4Th Baron Carew's paternal grandmother?  
**Gold answer:** Jane Catherine Cliffe  · gold paragraphs: George Carew, 4th Baron Carew, Robert Carew, 2nd Baron Carew

*First-stage context (TF-IDF top-5):* ★ George Carew, 4th Baron Carew, Gerald Carew, 5th Baron Carew, Nicholas Carew (died 1311), ★ Robert Carew, 2nd Baron Carew, George Carew (priest)

*First answer (no retrieval):*
```
<hop1>From the context, I find that Sir George Richard Philips, 2nd Baronet, is the father of Emily Anne Philips, who is the wife of Robert Carew, 2nd Baron Carew.</hop1>
<hop2>From the context, I find that Sir George Richard Philips, 2nd Baronet, is the father of Emily Anne Philips.</hop2>
<answer>Jane Catherine Cliffe</answer>   → ✅
```
*Probe:* P(fail) hop 1 = 0.01, hop 2 = 0.98 (τ = 0.64) → **stopped at `</hop2>`**

*Retrieval query:* `Sir George Richard Philips, 2nd Baronet. Who is George Carew, 4Th Baron Carew's paternal grandmother? From the context, I find that Sir George Richard Philips, 2nd Baronet, is the father of Emily Anne …`

*Retrieved (top 3, prepended to the context):*
- **George Carew (diplomat)** — Sir George Carew( died 13 November 1612) was an English diplomat, historian and Member of Parliament.
- **Sir George Agnew, 7th Baronet** — Sir George Anthony Agnew, 7th Baronet( born 18 August 1953) is a British Baronet and landowner. The son of Sir George Keith Agnew, 5th Baronet, he succeeded to  …
- **Sir Richard Grosvenor, 2nd Baronet** — Sir Richard Grosvenor, 2nd Baronet( c. 1604 – 31 January 1665) was an ancestor of the modern day Dukes of Westminster. He was the son of Sir Richard Grosvenor,  …

*Restarted answer:*
```
<hop1>From the context, I find that Sir George Carew, 4th Baron Carew's mother is Emily Anne Philips, daughter of Sir George Richard Philips, 2nd Baronet.</hop1>
<hop2>From the context, I find that Sir George Richard Philips, 2nd Baronet is the father of Emily Anne Philips.</hop2>
<answer>(none — ran to the token cap)</answer>   → ❌
```

---

## HotpotQA (out of domain) — fixed

**Q:** Who directed the second film in a British series of action comedy film parodying the James Bond secret agent genre with comedy similar to Rowan Atkinson's Mr. Bean character?  
**Gold answer:** Oliver Parker

*First-stage context (TF-IDF top-5):* Johnny English Reborn, Johnny English, Johnny English (film series), Mr. Bean's Holiday, Mr. Bean (character)

*First answer (no retrieval):*
```
<hop1>From the context, I find that the film series in question is the "Johnny English film series".</hop1>
<hop2>From the context, I find that the "Johnny English film series" is a series of action comedy film parodying the James Bond secret agent genre.</hop2>
<answer>British</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.96, hop 2 = 0.99 (τ = 0.64) → **stopped at `</hop1>`**

*Retrieval query:* `Who directed the second film in a British series of action comedy film parodying the James Bond secret agent genre with comedy similar to Rowan Atkinson's Mr. Bean character? From the context, I find  …`

*Retrieved (top 3, prepended to the context):*
- **No. 1 of the Secret Service** — No. 1 of the Secret Service is a 1977 imitation James Bond film starring Nicky Henson as British secret agent Charles Bind. It was directed and written by Linds …
- **James Tont operazione D.U.E.** — James Tont operazione D.U.E. or The Wacky World of James Tont is a 1966 French/Italian international co-production spy film spoof based on James Bond's "Thunder …
- **The Last of the Secret Agents?** — The Last of the Secret Agents? is a 1966 American film that spoofs the spy film genre, starring the then- popular comedy team of Allen& Rossi.

*Restarted answer:*
```
<hop1>From the context, I find that the series is Johnny English.</hop1>
<hop2>From the context, I find that the series Johnny English is a British series of action comedy film parodying the James Bond secret agent genre.</hop2>
<answer>Oliver Parker</answer>   → ✅
```

---

**Q:** Sporobolus and Zea are in the same what?  
**Gold answer:** family

*First-stage context (TF-IDF top-5):* Sporobolus, Sporobolus compositus, Leopoldo Zea Aguilar, Zea perennis, Microplitis croceipes

*First answer (no retrieval):*
```
<hop1>From the context, I find that Sporobolus is a genus of plants in the grass family.</hop1>
<hop2>From the context, I find that Zea is a genus of grasses.</hop2>
<answer>same genus</answer>   → ❌
```
*Probe:* P(fail) hop 1 = 0.90, hop 2 = 0.94 (τ = 0.64) → **stopped at `</hop1>`**

*Retrieval query:* `Sporobolus and Zea are in the same what? From the context, I find that Sporobolus is a genus of plants in the grass family.`

*Retrieved (top 3, prepended to the context):*
- **Zea (plant)** — Zea is a genus of plants in the grass family. The best-known species is "Z. mays", variously called maize, corn, or Indian corn, one of the most important crops …
- **Zea nicaraguensis** — Zea nicaraguensis is a true grass species in the genus "Zea". It is considered to be phenotypically the most distinctive, as well as the most threatened teosint …
- **Axinandra zeylanica** — Axinandra zeylanica is a species of plant in the Crypteroniaceae family. It is endemic to Sri Lanka, where locally known as" pol hunna- පොල් හුන්න".

*Restarted answer:*
```
<hop1>From the context, I find that Sporobolus is a genus of plants in the grass family.</hop1>
<hop2>From the context, I find that Zea is a genus of plants in the grass family.</hop2>
<answer>The grass family</answer>   → ✅
```

---
