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
* **"I Give Up" Failures:** The model explicitly stating the entity is missing. *(e.g., "There is no information about Bayan Khutugh's father in the context.")*
* **Dodging the Question:** The model giving an unrelated fact to avoid answering. *(e.g., When asked for the composer of Daddy-O, the model said: "From the context, I find that the film Daddy-O was released in 1958.")*
* **Outright Lies:** The model blatantly hallucinating a wrong entity. *(e.g., When asked for the composer of Astral City, the model said: "From the context, I find that the composer of film Astral City is Wagner de Assis." [He is the director, not the composer])*

## 4. Causal Patching & Denoising (Phase 4)
* **Prompt-End Patching is Too Weak:** Simply injecting a hidden state at the last token of the question prompt resulted in a **0% causal corruption rate**. **Analysis:** The model's attention mechanism just ignores the injected state at the end of the prompt and looks back at the 30+ tokens in the question context to re-derive the entity.
* **Entity-Level Denoising:** To actually hijack the model's brain, we must intervene at the exact microsecond it finishes thinking about the bridging entity. By harvesting the Clean State at the `</hop1>` token and injecting it into the Counterfactual run exactly at the `</hop1>` token, we overwrite its short-term memory before it starts Hop 2.

## 5. Phase 5 (Next Steps)
* **Probe-Triggered RAG:** Because our probe is trained at the Hop-1 level, we can use it in a live application. As the model generates Hop 1, the probe will monitor Layer 10. If the probe detects a hallucination, it will immediately halt generation (before the error cascades to Hop 2), trigger a retrieval block, and restart the prompt with the correct Wikipedia context.
