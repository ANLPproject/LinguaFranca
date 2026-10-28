"""
tests/test_probe_suite.py
Synthetic end-to-end test of lf_probe_suite.py (CPU, ~1 min):
  1. the whole suite runs (all four models, tuning, cells, report, resume),
  2. `lr_default` reproduces the OLD protocol (lf_common rows + load_hidden + make_probe + canonical_split) exactly,
  3. rows whose hop was never generated are dropped, not scored on a zero vector,
  4. train and test never share a question group.
    python tests/test_probe_suite.py
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lf_probe_suite as S                                                    # noqa: E402
from lf_common import binary_metrics, canonical_split, hop_rows, load_hidden, make_probe, rows_arrays  # noqa: E402

N_EX, D, N_LAYERS = 120, 16, 4


def make_data(tmp: Path):
    rng = np.random.RandomState(0)
    examples = []
    for i in range(N_EX):
        cf = i % 15 == 0
        ex = dict(id=f"ex{i}", is_counterfactual=cf, clean_pair_id=f"q{i // 2 if not cf else i}", final_answer_correct=bool(i % 2),
                  hop_spans=[dict(hop_idx=1), dict(hop_idx=2), dict(hop_idx=3)], hops=[])
        n_hops = 3
        pooled = rng.randn(N_LAYERS, n_hops, D).astype(np.float16)
        for k in (1, 2, 3):
            y = int(rng.rand() < (0.3 if k == 1 else 0.6))
            if k <= 2:
                pooled[1:, k - 1, :3] += (2.0 * y - 1.0)                     # signal in layers >= 1
            txt = "there is no information" if (y and rng.rand() < 0.2) else f"hop {k} text {rng.randint(100)}"
            ex["hops"].append(dict(hop_idx=k, text=txt, label=(y if k <= 2 else -1)))
        if i == 7:                                                          # hop 2 labelled but never generated
            ex["hop_spans"] = [dict(hop_idx=1)]
            pooled = pooled[:, :1]
        torch.save(dict(pooled=torch.from_numpy(pooled)), tmp / f"ex{i}.pt")
        examples.append(ex)
    return examples


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        hs = tmp / "hs"
        hs.mkdir()
        examples = make_data(hs)
        args = argparse.Namespace(models=list(S.ALL_MODELS), layers="all", tune_layers="1,2", jobs=2, n_boot=50,
                                  mlp_epochs=300, fast=True)
        R = S.run_setting("synthetic", examples, hs, args, tmp / "out")
        final = S.report(R, args, None)
        out = tmp / "out" / "synthetic"
        for f in ("summary.md", "cells.csv", "selected.json", "cells.jsonl", "tuning.json", "info.json"):
            assert (out / f).exists(), f
        # saved probes: one per (model, task) at the selected layer; reloading reproduces the stored test scores
        by_cell = {(c["model"], c["task"], c["layer"]): c for c in R["cells"]}
        tr0, te0 = canonical_split(examples)
        for r in final:
            base = out / "probes" / f"{r['model']}_{r['task']}_L{r['layer']}"
            assert base.with_suffix(".joblib").exists() and base.with_suffix(".json").exists(), base
            score, meta = S.load_probe(base.with_suffix(".joblib"))
            assert meta["layer"] == r["layer"] and meta["task"] == r["task"] and meta["feature_dim"] == D
            tem = R["masks"][r["task"]][1]
            Xte = R["X"][np.flatnonzero(tem), r["layer"]].astype(np.float32)
            ref = np.array(by_cell[(r["model"], r["task"], r["layer"])]["test_scores"])
            assert np.allclose(score(Xte), ref, atol=1e-4), (r["model"], r["task"])
            if r["model"] in ("lr_default", "lr", "svm"):
                with np.load(base.with_suffix(".npz")) as z:        # plain-array export gives the same scores
                    raw = ((Xte - z["mean"]) / z["scale"]) @ z["coef"] + z["intercept"]
                port = raw if r["model"] == "svm" else 1 / (1 + np.exp(-raw))
                assert np.allclose(port, ref, atol=1e-3), ("npz", r["model"], r["task"])
        n_cells = len(R["cells"])
        assert n_cells == len(S.ALL_MODELS) * 3 * N_LAYERS, n_cells
        # 3. never-generated hop dropped
        assert R["info"]["dropped"]["no_span"] == 1, R["info"]["dropped"]
        # 4. grouped split
        tr, te = canonical_split(examples)
        g = lambda idx: {examples[i]["clean_pair_id"] for i in idx}
        assert not (g(tr) & g(te))
        # resume: a second run must not recompute anything
        R2 = S.run_setting("synthetic", examples, hs, args, tmp / "out")
        assert len(R2["cells"]) == n_cells
        assert len((out / "cells.jsonl").read_text().splitlines()) == n_cells
        # 2. lr_default == old protocol, on the rows both can see (drop the never-generated hop 2 of ex7 from the old rows)
        old_rows = [r for r in hop_rows(examples) if not (r["id"] == "ex7" and r["hop"] == 2)]
        A = rows_arrays(old_rows)
        tr_ex = np.zeros(len(examples), bool)
        tr_ex[tr] = True
        trr = tr_ex[A["ei"]]
        XL = load_hidden(examples, old_rows, hs, [2])[2]
        by = {(c["model"], c["task"], c["layer"]): c for c in R["cells"]}
        for task, hops in S.TASKS.items():
            tm = np.isin(A["hop"], hops)
            trm, tem = trr & tm, ~trr & tm
            clf = make_probe().fit(XL[trm], A["y"][trm])
            ref = binary_metrics(A["y"][tem], clf.predict_proba(XL[tem])[:, 1])
            new = by[("lr_default", task, 2)]["test"]["all"]
            assert new["n"] == ref["n"], (task, new["n"], ref["n"])
            assert abs(new["auroc"] - ref["auroc"]) < 1e-9 and abs(new["acc"] - ref["acc"]) < 1e-12, (task, new, ref)
        # every model learned something on the planted signal (layer 2 test AUROC well above chance)
        for m in S.ALL_MODELS:
            a = by[(m, "hop12", 2)]["test"]["all"]["auroc"]
            assert a > 0.75, (m, a)
        assert len(final) == len(S.ALL_MODELS) * 3
        # legacy rows: identical to the old scripts INCLUDING the never-generated hop (zero vector)
        argsL = argparse.Namespace(**{**vars(args), "legacy_rows": True, "models": ["lr_default"]})
        RL = S.run_setting("synthetic_legacy", examples, hs, argsL, tmp / "out")
        all_rows = hop_rows(examples)
        AL = rows_arrays(all_rows)
        trl = tr_ex[AL["ei"]]
        XLo = load_hidden(examples, all_rows, hs, [2])[2]
        byL = {(c["model"], c["task"], c["layer"]): c for c in RL["cells"]}
        for task, hops in S.TASKS.items():
            tm = np.isin(AL["hop"], hops)
            trm, tem = trl & tm, ~trl & tm
            clf = make_probe().fit(XLo[trm], AL["y"][trm])
            ref = binary_metrics(AL["y"][tem], clf.predict_proba(XLo[tem])[:, 1])
            new = byL[("lr_default", task, 2)]["test"]["all"]
            assert new["n"] == ref["n"] and abs(new["auroc"] - ref["auroc"]) < 1e-9 and abs(new["acc"] - ref["acc"]) < 1e-12, (task, new, ref)
        assert RL["info"]["dropped"] == {"no_span": 0, "no_column": 0}
        print("[test_probe_suite] OK  cells:", n_cells, "| lr_default == old protocol (clean and legacy rows) | resume ok | dropped rows ok")


if __name__ == "__main__":
    main()
