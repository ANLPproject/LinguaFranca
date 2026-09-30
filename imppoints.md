# Important Discoveries & Insights (LinguaFranca)

## 1. Data Generation & Prompting (Phase 1 & 2)
* **Strict XML Chain-of-Thought:** We forced LLaMA-3.2-3B to output reasoning steps wrapped in `<hop1>` and `<hop2>` tags. This was absolutely critical because it allowed us to perfectly isolate the "bottleneck" where the model resolves a bridging entity (the `</hop1>` token). 
* **Counterfactual Generation:** To generate hallucinations, we created "Minimal Pairs" by swapping the real entity with a fake one (e.g., "Soft Parachutes" -> "Seven Billiard Tables"). This confused the model and forced it to hallucinate, giving us perfect 1:1 training data for the probe.

## 2. Advanced Probing (Phase 3)
* **Hop-Level Localization (RQ1):** The probe evaluates the hidden states of *individual hops*, not entire questions. This allows us to say exactly *which* step of the reasoning chain failed.
* **Layer 10 is the Bottleneck:** Across all layers (0-28), Layer 10 and 11 represent the absolute peak of the model's reasoning process. The Logistic Regression probe achieved **91% Accuracy / 96.7% AUROC** at Layer 10.
* **Strict Data Hygiene:** We grouped the 5-fold cross-validation by `clean_pair_id` (the base question ID) to ensure that Hop 1 and Hop 2 of the same question never bleed across train/test splits. 

## 3. Qualitative Error Analysis
The Layer 10 probe doesn't just detect one specific failure mode; it detects the fundamental neurological signature of a hallucination. When reviewing the top 10 most confident failure predictions (all 100% confidence), the probe successfully caught:
* **"I Give Up" Failures:** The model explicitly stating the entity is missing from the context.
* **Dodging the Question:** The model giving a film's release date when asked for its composer (e.g., *Daddy-O*).
* **Outright Lies:** The model stating the director of a film is the composer (e.g., *Astral City*).

## 4. Causal Patching & Denoising (Phase 4)
* **Prompt-End Patching is Too Weak:** Simply injecting a hidden state at the last token of the question prompt results in a 0% causal effect. The model's attention mechanism just ignores the injected state and looks back at the rest of the prompt.
* **Entity-Level Denoising:** To actually hijack the model's brain, we must intervene at the exact microsecond it finishes thinking about the bridging entity. By harvesting the Clean State at the `</hop1>` token and injecting it into the Counterfactual run exactly at the `</hop1>` token, we can overwrite its short-term memory before it starts Hop 2.

## 5. Phase 5 (Next Steps)
* **Probe-Triggered RAG:** Because our probe is trained at the Hop-1 level, we can use it in a live application. As the model generates Hop 1, the probe will monitor Layer 10. If the probe detects a hallucination, it will immediately halt generation (before the error cascades to Hop 2), trigger a retrieval block, and restart the prompt with the correct Wikipedia context.
