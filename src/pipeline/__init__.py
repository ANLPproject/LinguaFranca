"""
src/pipeline
────────────
Phases 3–5: probe-triggered Naive Restart RAG, the Doctor-RAG-style text
baseline, and the end-to-end evaluation.

Modules
───────
common.py     settings, data loading, canonical split, prompts, answer metrics
probes.py     train hop-1 / hop-2 probes, CV threshold calibration (F0.5)
retriever.py  passage corpus + dense retriever (all-MiniLM-L6-v2, FAISS)
generation.py live token-by-token generation with hop-boundary probing
judge.py      post-hoc LLM judge (Doctor-RAG-style trigger)
systems.py    the five systems (no-intervention, probe, judge, always-retrieve, oracle)
run.py        GPU stages with checkpoint/resume  (python -m src.pipeline.run --help)
evaluate.py   metrics, tables, figures         (python -m src.pipeline.evaluate --help)

Design note
───────────
Decoding is greedy, so a trajectory that the probe halts at </hopK> is a
token-for-token prefix of the uninterrupted trajectory.  The runner therefore
generates every trajectory ONCE with the probe in monitor mode (scores
recorded, never halting) and pre-computes the restart outputs; every system
and every threshold is then evaluated from these cached outputs.  A subset is
also run truly live (halt → retrieve → restart) to verify this equivalence and
to measure wall-clock latency.
"""
