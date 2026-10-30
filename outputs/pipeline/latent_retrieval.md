# Latent retrieval: search with the model's hidden state (layer 12)

Ridge map (alpha = 100, chosen by grouped 5-fold CV on the train split) from the mean-pooled hop state to the all-MiniLM-L6-v2 embedding of the paragraph that hop needs. Trained on all labelled hops 1–2 of the probe's train split (6315 rows). Tested on the probe's **test** split: hops that failed and whose needed paragraph was missing from the prompt (397 rows; hop 1: 20, hop 2: 377). A hit = the needed paragraph is among the top results (paragraphs already in the prompt excluded). The text query is the pipeline's current repair query (repair v2). Corpus: the pipeline's 94,031 paragraphs.

| query | hit@3 | hit@3 hop 1 | hit@3 hop 2 | hit@10 | hit@3 after cross-encoder re-rank of top 20 |
|---|---|---|---|---|---|
| text query (current) | 0.438 | 0.750 | 0.422 | 0.547 | 0.531 |
| hidden-state query | 0.156 | 0.000 | 0.164 | 0.212 | 0.224 |
| blend (w = 0.25 on hidden state, chosen on train) | 0.463 | 0.700 | 0.451 | 0.569 | 0.544 |
| blend (w = 0.5) | 0.448 | 0.650 | 0.438 | 0.547 | 0.534 |
