"""
phase1_dataset/hop_rules.py
───────────────────────────
Dependency-free labeling rules shared by label_hops.py and the
canonical-dataset script (scripts/make_canonical_dataset.py).

The "extra hop" rule
────────────────────
2WikiMultihopQA compositional questions have exactly two gold edges, but the
model frequently emits more <hopN> blocks than that (1,329 / 4,000 examples in
the Phase 1 run produced 3–18 hops).  A hop whose index is larger than the gold
reasoning graph has NO gold entity, so we cannot say whether it succeeded or
failed.  The original labeler marked every such hop as a failure
("unmatched_gold"), which made 64 % of all failure labels artefacts of hop
position.  These hops are now labeled -1 (unlabeled) and excluded everywhere,
exactly like hops whose gold entity is blank.

Because the bipartite assignment processes hops in generation order, extra hops
can only claim gold entities that hops 1..N left unclaimed, so excluding them
never changes the labels of hops 1..N.  Applying this rule post-hoc to an
already-labeled file is therefore identical to re-running the labeler.
"""

from __future__ import annotations

EXTRA_HOP_METHOD = "extra_hop"


def apply_extra_hop_rule(example: dict) -> dict:
    """
    In-place: label hops beyond the gold reasoning graph as -1, back-fill blank
    `bridging_entity_gold` fields from the graph, and recompute first_fail_hop.
    Returns the same dict for convenience.
    """
    graph = example.get("reasoning_graph", []) or []
    gold_by_hop = {node.get("hop"): (node.get("gold_entity") or "") for node in graph}
    n_gold = len(graph)

    for hop in example.get("hops", []):
        idx = hop.get("hop_idx")
        if idx is None:
            continue
        if idx > n_gold:
            if hop.get("label") in (0, 1) and "label_v1" not in hop:
                hop["label_v1"] = hop["label"]          # keep the old label for auditing
            hop["label"] = -1
            hop["match_method"] = EXTRA_HOP_METHOD
            hop["bridging_entity_gold"] = ""
        elif not (hop.get("bridging_entity_gold") or "").strip():
            hop["bridging_entity_gold"] = gold_by_hop.get(idx, "")

    example["first_fail_hop"] = first_fail_hop(example.get("hops", []))
    return example


def first_fail_hop(hops: list[dict]):
    """Smallest hop_idx with label 1 (labeled hops only), else None."""
    fails = [h["hop_idx"] for h in hops if h.get("label") == 1]
    return min(fails) if fails else None
