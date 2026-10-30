# Dense vs hybrid repair retrieval (same probe, same first pass)

Both runs: tuned shared probe (C = 0.01, layer 12, τ = 0.650), repair v2, up to 2 restarts, 500 2Wiki + 500 HotpotQA.
Only the **restart retrieval** differs:

- dense: all-MiniLM-L6-v2 top 20 → cross-encoder re-rank
- hybrid: dense + BM25 (RRF) top 30 → cross-encoder re-rank

The first pass (no intervention), the probe's detection numbers and the judges' detection numbers are identical, because they don't depend on the repair retriever.

Dense run: [`../tfidf_final_lrC0.01_L12/`](../tfidf_final_lrC0.01_L12/). Hybrid run: this folder.

## Repair retrieval (2Wiki, restarts with ≥ 1 gold paragraph missing)

| restart at | n | dense: found a missing gold paragraph | hybrid: found |
|---|---|---|---|
| hop 1 | 296 | 0.247 | 0.264 |
| hop 2 | 293 | 0.532 | **0.659** |

## End-to-end accuracy (token F1 ≥ 0.5)

| system | 2Wiki dense | 2Wiki hybrid | HotpotQA dense | HotpotQA hybrid |
|---|---|---|---|---|
| No intervention | 0.342 | 0.342 | 0.504 | 0.504 |
| Probe trigger (ours) | 0.538 | **0.558** | **0.570** | 0.550 |
| Probe trigger, up to 2 restarts | 0.594 | **0.604** | **0.564** | 0.552 |
| Probe trigger, accuracy-optimal τ | 0.554 | **0.582** | **0.580** | 0.566 |
| Judge trigger (Llama-3.2-3B) | 0.454 | 0.452 | 0.526 | 0.536 |
| Judge trigger (Qwen2.5-7B) | 0.508 | 0.510 | 0.546 | 0.546 |
| Always restart after hop 2 | 0.554 | **0.582** | 0.586 | 0.586 |
| Oracle trigger | 0.550 | 0.566 | – | – |

## Reading

- **2Wiki, in-domain: hybrid helps.** It finds the missing hop-2 paragraph 66% of the time vs 53%. With one restart, the probe goes from 53.8% to 55.8%, and with 2 restarts from 59.4% to 60.4%. Restarts that break a correct answer drop from 10 to 5.
- **HotpotQA, out-of-domain: hybrid is no better.** The probe falls 2 points (57.0 → 55.0) and "always restart after hop 2" is unchanged (58.6). The differences are within the paired-bootstrap noise (CIs of ±3–4 points).
- **The ranking doesn't change.** "Always restart after hop 2" still matches or beats the probe on accuracy (2Wiki 58.2 vs 55.8), and with fair thresholds every judge again reaches 56–59% (see `robustness/robustness.md` §1). The probe still has the best accuracy at matched restart budgets of 40–80% on 2Wiki (§4).
- **Latency:** hybrid retrieval costs 154 ms per call vs 105 ms.

**Recommendation:** report dense as the main system (it's the pre-registered configuration and better on HotpotQA). Report hybrid as a retrieval ablation: better repair retrieval improves in-domain accuracy, and the gain comes from the repair, not the trigger.
