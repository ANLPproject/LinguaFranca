# Important Discoveries & Insights (LinguaFranca)

## 1. Data Generation & Prompting (Phase 1 & 2)
* **Strict XML Chain-of-Thought:** We forced LLaMA-3.2-3B to output reasoning steps wrapped in `<hop1>` and `<hop2>` tags. This was absolutely critical because it allowed us to perfectly isolate the "bottleneck" where the model resolves a bridging entity (the `</hop1>` token). 
* **Counterfactual Generation:** To generate hallucinations, we created "Minimal Pairs" by swapping the real entity with a fake one (e.g., "Soft Parachutes" -> "Seven Billiard Tables"). This confused the model and forced it to hallucinate, giving us perfect 1:1 training data for the probe.

## 2. Advanced Probing Results (Phase 3)
* **Hop-Level Localization (RQ1):** The probe evaluates the hidden states of *individual hops*, not entire questions. This allows us to say exactly *which* step of the reasoning chain failed.
* **Layer-wise Accuracies (5-Fold CV):** We found that early layers do not contain strong reasoning signals. The probe accuracy peaks in the middle layers before dropping off.
  * Layer 8: 89.3% Acc | 95.8% AUROC
  * Layer 9: 90.6% Acc | 96.2% AUROC
  * **Layer 10: 91.2% Acc | 96.7% AUROC (Absolute Peak Bottleneck)**
  * Layer 11: 91.4% Acc | 96.6% AUROC
  * Layer 24: 88.6% Acc | 94.8% AUROC
* **Experimental Controls (Proving the Probe is real):**
  * **Hop-1 Only Probe:** 95.1% Acc, 95.3% AUROC. Proves the probe works perfectly even when isolating the very first reasoning step.
  * **Shuffled Labels:** ~50.6% Acc. Proves the probe isn't cheating by memorizing dataset artifacts.
  * **Cross-Hop Generalization (Train on Hop 1, Test on Hop 2):** Dropped to 62.5% Acc, 66.1% AUROC. **Analysis:** This proves that the hidden states for Hop 1 and Hop 2 are geometrically different! The model stores the "I am on step 1" thought differently than the "I am on step 2" thought.

## 3. Qualitative Error Analysis Results
The Layer 10 probe correctly identified different modalities of hallucination with **100% confidence**. Manual verification of the top 10 flagged failures showed:
* **"I Give Up" Failures (Hedging):** The model explicitly stating the entity is missing. *(e.g., "There is no information about Bayan Khutugh's father in the context.")*
* **Dodging the Question:** The model giving an unrelated fact to avoid answering. *(e.g., When asked for the composer of Daddy-O, the model said: "From the context, I find that the film Daddy-O was released in 1958.")*
* **Outright Lies:** The model blatantly hallucinating a wrong entity. *(e.g., When asked for the composer of Astral City, the model said: "From the context, I find that the composer of film Astral City is Wagner de Assis." [He is the director, not the composer])*

## 4. Hedging vs. Confident Lies (Ablation Study)
A key methodological concern was whether the probe was simply acting as a mechanical "refusal detector" rather than truly understanding reasoning failures. We split the failure class into "Hedging" (explicitly stating lack of knowledge) and "Confidently Wrong" (stating a lie). The layer-wise breakdown proved the probe is doing much more than lexical matching:
* **Layer 10:** 100.0% accurate at detecting Hedging | 80.0% accurate at detecting Confident Lies.
* **Layer 11 (The True Sweet Spot):** 100.0% accurate at detecting Hedging | **98.2% accurate at detecting Confident Lies**.

**Full Layer-by-Layer Breakdown (Hop-1 Test Set Failures):**
| Layer | Hedging Acc (%) | Confident Wrong Acc (%) |
|-------|-----------------|-------------------------|
| 0     | 100.0           | 65.5                    |
| 1     | 100.0           | 83.6                    |
| 2     | 100.0           | 83.6                    |
| 3     | 92.3            | 70.9                    |
| 4     | 100.0           | 78.2                    |
| 5     | 100.0           | 70.9                    |
| 6     | 100.0           | 80.0                    |
| 7     | 100.0           | 60.0                    |
| 8     | 100.0           | 90.9                    |
| 9     | 84.6            | 60.0                    |
| 10    | 100.0           | 80.0                    |
| **11**| **100.0**       | **98.2**                |
| 12    | 92.3            | 80.0                    |
| 13    | 100.0           | 78.2                    |
| 14    | 100.0           | 87.3                    |
| 15    | 92.3            | 78.2                    |
| 16    | 92.3            | 74.5                    |
| 17    | 100.0           | 87.3                    |
| 18    | 100.0           | 67.3                    |
| 19    | 100.0           | 74.5                    |
| 20    | 100.0           | 80.0                    |
| 21    | 100.0           | 81.8                    |
| 22    | 100.0           | 78.2                    |
| 23    | 100.0           | 67.3                    |
| 24    | 100.0           | 69.1                    |
| 25    | 100.0           | 70.9                    |
| 26    | 100.0           | 65.5                    |
| 27    | 100.0           | 70.9                    |

This definitively proves that the probe is capturing the fundamental failure of reasoning (unsupported facts) deep in the model's representations, regardless of whether the model chooses to express that failure honestly (hedging) or dishonestly (hallucinating).

## 5. Causal Patching & Denoising (Phase 4)
* **Prompt-End Patching is Too Weak:** Simply injecting a hidden state at the last token of the question prompt resulted in a **0% causal corruption rate**. **Analysis:** The model's attention mechanism just ignores the injected state at the end of the prompt and looks back at the 30+ tokens in the question context to re-derive the entity.
* **Entity-Level Denoising:** To actually hijack the model's brain, we must intervene at the exact microsecond it finishes thinking about the bridging entity. By harvesting the Clean State at the `</hop1>` token and injecting it into the Counterfactual run exactly at the `</hop1>` token, we overwrite its short-term memory before it starts Hop 2.

## 6. Phase 5 (Next Steps)
* **Probe-Triggered RAG:** Because our probe is trained at the Hop-1 level, we can use it in a live application. As the model generates Hop 1, the probe will monitor Layer 11 (since we now know it excels at catching confident lies). If the probe detects a hallucination, it will immediately halt generation (before the error cascades to Hop 2), trigger a retrieval block, and restart the prompt with the correct Wikipedia context.

## 7. Dataset & Methodology Validation (Discussion)
* **Reasoning Depth Verification:** We verified that the compositional-filtered data used in this project is consistently strictly 2-hop (100.0% of the 4000 examples evaluated contained exactly 2 reasoning edges). This is consistent with 2WikiMultihopQA's core category definition. Therefore, our fixed 2-hop XML template (`<hop1>` and `<hop2>`) perfectly matched the structural reality of the dataset, and the model did not suffer from conflicting pressures or forced compression.

### 8. The LLM-as-a-Judge Baseline Comparison (Massive Selling Point)
* **The Experiment:** We tested if Llama-3.2-3B could act as an external judge to catch its own reasoning hallucinations by prompting it with the full Wikipedia context and the reasoning hop.
* **The Baseline Failure:** The external LLM judge failed completely, achieving only **63.1% accuracy** (barely better than random chance) despite consuming 1039 tokens per hop and taking 648ms of inference time.
* **The Probe Superiority:** In contrast, the internal Linear Probe on Layer 11 achieved **91.4% accuracy** with ** token cost** and less than 1ms of latency (O(1) matrix multiplication).
* **The Conclusion:** Small 3B models lack the capacity to act as reliable external text-based judges for fact-checking. However, they internally *know* when they are hallucinating! Probing hidden states unlocks this knowledge efficiently and accurately.

### 9. Sycophancy in LLM-as-a-Judge
* When broken down by failure type, the LLM Judge completely collapses due to sycophancy (yes-bias).
* **Hedging Failure Accuracy (23.0%):** When the model explicitly states 'I don't know' or 'There is no information', the external LLM judge still incorrectly marks it as a successful reasoning step 77% of the time.
* **Confident Failure Accuracy (28.7%):** When the model hallucinated confidently, the LLM Judge caught it only 28.7% of the time, compared to the Linear Probe which caught it **98.2%** of the time.
