"""
src/pipeline/evaluate.py
────────────────────────
Phase 5 evaluation from the cached pipeline outputs (CPU only).

  RQ1  hop-level detection (probe vs judge on identical rows), trajectory-level
       detection (P / R / F0.5), first-failure localization, detection earliness
  RQ3  end-to-end accuracy (EM / F1 / acc) for every system, fixed vs broken
       answers, paired-bootstrap CIs, token cost, wall-clock latency (live subset),
       accuracy-vs-cost threshold sweeps
  RQ4  per-layer CV AUROC (from probe training), earliest decodable layer
  sanity  live-vs-stored feature parity, simulated-vs-live equivalence

Usage:
    python -m src.pipeline.evaluate --out outputs/pipeline/tfidf
Writes results.md, results.json, tables.tex, sweep.csv, sweep.png, per_item.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.pipeline.common import answer_correct, detection_metrics, read_jsonl
from src.pipeline.systems import judge_trigger, outcome, probe_trigger

SYSTEMS = ["no_intervention", "probe_restart", "judge_restart", "always_retrieve", "oracle_restart"]
LABEL = {"no_intervention": "No intervention", "probe_restart": "Probe trigger (ours)",
         "judge_restart": "Judge trigger (Doctor-RAG-style)", "always_retrieve": "Always retrieve",
         "oracle_restart": "Oracle trigger",
         "probe_restart_acc": "Probe trigger, accuracy-optimal τ (2-fold CV)",
         "probe_restart_2r": "Probe trigger, up to 2 restarts", "judge_restart_2r": "Judge trigger, up to 2 restarts"}
MODE_TEXT = {"shared": "one shared probe for both hops", "dedicated": "dedicated hop-1 / hop-2 probes"}
TAU_GRID = np.round(np.linspace(0, 1, 51), 2)


def _fold(uid: str) -> int:
    import hashlib
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2


def _acc_optimal_tau(arrs: dict) -> tuple[float, float]:
    """(τ1, τ2) maximising one-round probe-restart accuracy on the given items; ties -> fewer restarts."""
    c0, c1, c2, p1, p2 = (arrs[k] for k in ("c0", "c1", "c2", "p1", "p2"))
    best_key, best = None, (1.0, 1.0)
    for t1 in TAU_GRID:
        f1 = p1 >= t1
        for t2 in TAU_GRID:
            f2 = ~f1 & (p2 >= t2)
            key = (float(np.where(f1, c1, np.where(f2, c2, c0)).mean()), -float((f1 | f2).mean()))
            if best_key is None or key > best_key:
                best_key, best = key, (float(t1), float(t2))
    return best


def _merged(out: Path, stem: str) -> list[dict]:
    """All records of one stage: <stem>.jsonl plus <stem>.shardN.jsonl (not other stems sharing a prefix)."""
    recs = []
    for p in sorted([*out.glob(f"{stem}.jsonl"), *out.glob(f"{stem}.shard*.jsonl")]):
        recs += read_jsonl(p)
    return recs


def _safe(uid: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)


def _paired_ci(a, b, n_boot: int = 2000, seed: int = 0):
    """95 % CI of mean(a) - mean(b) for paired 0/1 arrays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def _relabel(items: list[dict], s0: dict) -> None:
    """Hop labels for S0 trajectories that differ from the stored Phase 1 text (presence check, no LLM tier)."""
    todo = [it for it in items if it["dataset"] == "2wiki" and s0[it["uid"]].get("reproduced") is False]
    for it in items:
        it["label_source"] = "phase1" if it["dataset"] == "2wiki" else None
    if not todo:
        return
    try:
        from utils.matching import EntityMatcher
        matcher = EntityMatcher(aliases={}, sbert_threshold=0.70)
    except Exception as e:                                         # noqa: BLE001
        print(f"[evaluate] cannot relabel {len(todo)} non-reproduced trajectories ({e}); excluding their labels")
        for it in todo:
            it["labels"], it["first_fail"], it["label_source"] = {}, None, "unavailable"
        return
    for it in todo:
        labels = {}
        for k in ("1", "2"):
            g, t = it["gold_entities"].get(k, ""), s0[it["uid"]]["hop_text"].get(k, "")
            if g.strip():
                labels[k] = 0 if (t and matcher.match(g, t).matched) else 1
        it["labels"] = labels
        fails = [int(k) for k, v in labels.items() if v == 1]
        it["first_fail"] = min(fails) if fails else None
        it["label_source"] = "relabeled"


def evaluate(out, probe_dir=None, probe_mode: str = "shared") -> dict:
    out = Path(out)
    probe_dir = Path(probe_dir) if probe_dir else out / "probes"
    meta = json.loads((probe_dir / "probes_meta.json").read_text())
    items_all = read_jsonl(out / "eval_set.jsonl")
    s0 = {r["uid"]: r for r in _merged(out, "s0")}
    judge = {r["uid"]: r for r in _merged(out, "judge")}
    rs = defaultdict(dict)
    for r in _merged(out, "restarts") + _merged(out, "restarts2"):
        rs[r["uid"]][r["key"]] = r
    for sc in _merged(out, "rescore"):
        if sc["key"] in rs[sc["uid"]]:
            rs[sc["uid"]][sc["key"]]["rescore"] = sc
    has_round2 = any(">" in k for v in rs.values() for k in v)
    live = {r["uid"]: r for r in _merged(out, "live")}

    items = [it for it in items_all if it["uid"] in s0 and "q" in rs[it["uid"]]
             and all(f"h{k}" in rs[it["uid"]] for k in ("1", "2")
                     if k in s0[it["uid"]]["boundaries"] and s0[it["uid"]]["hop_text"][k])]
    print(f"[evaluate] {len(items)}/{len(items_all)} items have all outputs")
    _relabel(items, s0)

    # Probe scores are recomputed from the saved live features with THIS probe bundle, so a run made with
    # an older (e.g. uncalibrated) bundle or the other probe mode is evaluated on the same scale as `tau`.
    from src.pipeline.probes import PROBE_MODES, ProbeBundle
    alt_mode = next(m for m in PROBE_MODES if m != probe_mode)
    pb, pb_alt = ProbeBundle(probe_dir, mode=probe_mode), ProbeBundle(probe_dir, mode=alt_mode)
    tau, layers, tau_alt = pb.tau, pb.layer, pb_alt.tau
    LABEL["probe_restart_alt"] = f"Probe trigger, {MODE_TEXT[alt_mode]} (ablation)"
    LABEL["probe_restart_2r_alt"] = f"Probe trigger, {MODE_TEXT[alt_mode]}, up to 2 restarts (ablation)"

    def scores_of(uid, bundle):
        f = out / "s0_feats" / f"{_safe(uid)}.npz"
        if f.exists():
            z = np.load(f)
            return {k: bundle.p_fail(k, z[f"h{k}"]) for k in (1, 2) if f"h{k}" in z.files}
        return None

    probe_scores = {it["uid"]: scores_of(it["uid"], pb) for it in items}
    if any(v is None for v in probe_scores.values()):
        raise SystemExit("s0_feats/ is missing: probe scores cannot be recomputed for this probe bundle")
    alt_scores = {it["uid"]: scores_of(it["uid"], pb_alt) for it in items}
    # second-round triggers were scored during the run with the run's probe mode
    run_args = json.loads((out / "run_args.json").read_text()) if (out / "run_args.json").exists() else {}
    run_mode = run_args.get("probe_mode", "dedicated")       # runs before the switch used dedicated probes

    R: dict = dict(setting=meta["setting"], probe_mode=probe_mode, n_items=len(items), probe_layers=layers, tau=tau)

    # ── RQ1: detection on 2Wiki ──────────────────────────────────────────────
    wiki = [it for it in items if it["dataset"] == "2wiki"]
    det = {}
    for k in (1, 2):
        rows = [(it, probe_scores[it["uid"]][k], judge.get(it["uid"], {}).get(str(k), {}).get("p_fail"))
                for it in wiki if it["labels"].get(str(k)) in (0, 1) and k in probe_scores[it["uid"]]
                and judge.get(it["uid"], {}).get(str(k)) is not None]
        if not rows:
            continue
        y = [it["labels"][str(k)] for it, _, _ in rows]
        pp = np.array([p for _, p, _ in rows]); pj = np.array([j for _, _, j in rows])
        pa = np.array([alt_scores[it["uid"]][k] for it, _, _ in rows])
        det[f"hop{k}"] = {"probe": detection_metrics(y, pp >= tau[k], pp), "judge": detection_metrics(y, pj >= 0.5, pj),
                          f"probe_{alt_mode}": detection_metrics(y, pa >= tau_alt[k], pa)}
    traj_rows = [it for it in wiki if it["label_source"] != "unavailable"]
    if traj_rows:
        y = [int(it["first_fail"] in (1, 2)) for it in traj_rows]
        kp = [probe_trigger(probe_scores[it["uid"]], tau) for it in traj_rows]
        kj = [judge_trigger(judge.get(it["uid"])) for it in traj_rows]
        ka = [probe_trigger(alt_scores[it["uid"]], tau_alt) for it in traj_rows]
        traj = {"probe": detection_metrics(y, [k is not None for k in kp]),
                "judge": detection_metrics(y, [k is not None for k in kj]),
                f"probe_{alt_mode}": detection_metrics(y, [k is not None for k in ka])}
        fail = [i for i, v in enumerate(y) if v]
        for name, ks in (("probe", kp), ("judge", kj), (f"probe_{alt_mode}", ka)):
            traj[name]["localization_acc_on_failures"] = (
                float(np.mean([ks[i] == traj_rows[i]["first_fail"] for i in fail])) if fail else float("nan"))
            traj[name]["exact_first_fail_acc_all"] = float(np.mean(
                [ks[i] == (traj_rows[i]["first_fail"] if traj_rows[i]["first_fail"] in (1, 2) else None)
                 for i in range(len(traj_rows))]))
        det["trajectory"] = traj
    R["detection_2wiki"] = det
    R["label_sources"] = {s: sum(it["label_source"] == s for it in wiki) for s in ("phase1", "relabeled", "unavailable")}

    # ── accuracy-optimal τ: 2-fold CV over the 2Wiki eval questions; HotpotQA uses τ fit on all of 2Wiki ──
    def acc_arrays(its):
        def corr(it, key):
            r = rs[it["uid"]].get(key)
            return answer_correct(r["answer"], it["gold_answer"]) if r else None
        c0 = np.array([answer_correct(s0[it["uid"]]["answer"], it["gold_answer"]) for it in its], float)
        a = dict(c0=c0)
        for k in (1, 2):
            ck = [corr(it, f"h{k}") for it in its]
            a[f"c{k}"] = np.array([c0[i] if v is None else v for i, v in enumerate(ck)], float)
            a[f"p{k}"] = np.array([probe_scores[it["uid"]].get(k, -1.0) if ck[i] is not None else -1.0
                                   for i, it in enumerate(its)], float)
        return a

    acc_tau = {}
    if wiki:
        folds = {f: [it for it in wiki if _fold(it["uid"]) == f] for f in (0, 1)}
        fit = {f: _acc_optimal_tau(acc_arrays(folds[1 - f])) for f in (0, 1) if folds[1 - f]}
        fit_all = _acc_optimal_tau(acc_arrays(wiki))
        for it in items:
            t = fit.get(_fold(it["uid"]), fit_all) if it["dataset"] == "2wiki" else fit_all
            acc_tau[it["uid"]] = {1: t[0], 2: t[1]}
        R["acc_optimal_tau"] = {"fold0_fit_on_fold1": fit.get(0), "fold1_fit_on_fold0": fit.get(1),
                                "all_2wiki_used_for_hotpotqa": fit_all}

    # ── RQ3: end-to-end ──────────────────────────────────────────────────────
    p2_key = "probe_restart_2r" if run_mode == probe_mode else "probe_restart_2r_alt"
    per_item = []
    for it in items:
        row = dict(uid=it["uid"], dataset=it["dataset"], condition=it.get("retrieval_condition"),
                   gold=it["gold_answer"], first_fail=it.get("first_fail"))
        kw = dict(probe_scores=probe_scores[it["uid"]], tau=tau, judge_rec=judge.get(it["uid"]))
        for sysname in SYSTEMS:
            if sysname == "oracle_restart" and it["dataset"] != "2wiki":
                continue
            row[sysname] = outcome(sysname, it, s0[it["uid"]], rs[it["uid"]], **kw)
        if acc_tau:
            row["probe_restart_acc"] = outcome("probe_restart", it, s0[it["uid"]], rs[it["uid"]],
                                               **dict(kw, tau=acc_tau[it["uid"]]))
        alt_kw = dict(kw, probe_scores=alt_scores[it["uid"]], tau=tau_alt)
        row["probe_restart_alt"] = outcome("probe_restart", it, s0[it["uid"]], rs[it["uid"]], **alt_kw)
        if has_round2:
            # second-round probe scores were computed during the run, so the 2-round probe system
            # exists for the run's probe mode only (main row if it matches, ablation row otherwise)
            row[p2_key] = outcome("probe_restart", it, s0[it["uid"]], rs[it["uid"]], rounds=2,
                                  **(kw if run_mode == probe_mode else alt_kw))
            row["judge_restart_2r"] = outcome("judge_restart", it, s0[it["uid"]], rs[it["uid"]], rounds=2, **kw)
        per_item.append(row)

    systems_present = SYSTEMS + (["probe_restart_acc"] if acc_tau else []) + \
        ([p2_key, "judge_restart_2r"] if has_round2 else []) + ["probe_restart_alt"]
    e2e = {}
    for ds in ("2wiki", "hotpotqa"):
        rows = [r for r in per_item if r["dataset"] == ds]
        if not rows:
            continue
        base = np.array([r["no_intervention"]["correct"] for r in rows], float)
        e2e[ds] = {}
        for sysname in systems_present:
            if sysname not in rows[0]:
                continue
            o = [r[sysname] for r in rows]
            c = np.array([x["correct"] for x in o], float)
            m = dict(n=len(rows), acc=float(c.mean()), em=float(np.mean([x["em"] for x in o])),
                     f1=float(np.mean([x["f1"] for x in o])), fire_rate=float(np.mean([x["fired"] for x in o])),
                     gen_tokens=float(np.mean([x["gen_tokens"] for x in o])),
                     processed_tokens=float(np.mean([x["processed_tokens"] for x in o])),
                     fixed=int(((c == 1) & (base == 0)).sum()), broken=int(((c == 0) & (base == 1)).sum()))
            dt = [x["detect_tokens"] for x in o if x["fired"] and x["detect_tokens"] is not None]
            m["mean_detect_tokens"] = float(np.mean(dt)) if dt else None
            if sysname != "no_intervention":
                m["acc_delta_vs_none"] = m["acc"] - float(base.mean())
                m["acc_delta_ci95"] = _paired_ci(c, base)
            e2e[ds][sysname] = m
        pr = np.array([r["probe_restart"]["correct"] for r in rows], float)
        jr = np.array([r["judge_restart"]["correct"] for r in rows], float)
        e2e[ds]["probe_minus_judge_acc_ci95"] = _paired_ci(pr, jr)
        if has_round2:
            if p2_key == "probe_restart_2r":
                pr2 = np.array([r["probe_restart_2r"]["correct"] for r in rows], float)
                jr2 = np.array([r["judge_restart_2r"]["correct"] for r in rows], float)
                e2e[ds]["probe2r_minus_judge2r_acc_ci95"] = _paired_ci(pr2, jr2)
            e2e[ds]["second_restart_rate"] = {s: float(np.mean([r[s]["n_restarts"] == 2 for r in rows]))
                                              for s in (p2_key, "judge_restart_2r")}
        # breakdown by first-stage retrieval condition
        by = defaultdict(list)
        for r in rows:
            by[r["condition"]].append(r)
        e2e[ds]["by_condition"] = {
            cond: {s: float(np.mean([r[s]["correct"] for r in rr])) for s in systems_present if s in rr[0]} | {"n": len(rr)}
            for cond, rr in sorted(by.items())}
    R["end_to_end"] = e2e

    # HotpotQA (and 2Wiki) final-answer failure prediction from the internal signal
    item_rows = {r["uid"]: r for r in per_item}
    fap = {}
    for ds in ("2wiki", "hotpotqa"):
        rows = [it for it in items if it["dataset"] == ds]
        y = [int(not item_rows[it["uid"]]["no_intervention"]["correct"]) for it in rows]
        sp = [max(probe_scores[it["uid"]].values()) if probe_scores[it["uid"]] else np.nan for it in rows]
        sj = [max((v["p_fail"] for v in judge.get(it["uid"], {}).values() if isinstance(v, dict)), default=np.nan)
              for it in rows]
        if rows and len(set(y)) > 1:
            fap[ds] = dict(probe_max_pfail=detection_metrics(y, np.nan_to_num(sp) >= 0.5, sp).get("auroc"),
                           judge_max_pfail=detection_metrics(y, np.nan_to_num(sj) >= 0.5, sj).get("auroc"),
                           n=len(rows), base_failure_rate=float(np.mean(y)))
    R["final_answer_failure_auroc"] = fap

    # ── sweeps ───────────────────────────────────────────────────────────────
    grid = np.round(np.linspace(0, 1, 41), 3)
    sweep = []
    for ds in ("2wiki", "hotpotqa"):
        rows_it = [it for it in items if it["dataset"] == ds]
        for t in grid:
            for trig in ("probe", "judge"):
                o = [outcome("probe_restart" if trig == "probe" else "judge_restart", it, s0[it["uid"]], rs[it["uid"]],
                             probe_scores=probe_scores[it["uid"]], tau={1: t, 2: t}, judge_rec=judge.get(it["uid"]),
                             theta=t) for it in rows_it]
                if o:
                    sweep.append(dict(dataset=ds, trigger=trig, threshold=float(t),
                                      fire_rate=float(np.mean([x["fired"] for x in o])),
                                      acc=float(np.mean([x["correct"] for x in o])),
                                      gen_tokens=float(np.mean([x["gen_tokens"] for x in o]))))
    with open(out / "sweep.csv", "w", encoding="utf-8") as f:
        f.write("dataset,trigger,threshold,fire_rate,acc,gen_tokens\n")
        for s in sweep:
            f.write(f"{s['dataset']},{s['trigger']},{s['threshold']},{s['fire_rate']:.4f},{s['acc']:.4f},{s['gen_tokens']:.2f}\n")
    _plot_sweep(out, sweep, e2e)

    # ── latency & equivalence (live subset) ──────────────────────────────────
    lat = {}
    it_by = {it["uid"]: it for it in items}
    lv = [u for u in live if u in it_by]
    if lv:
        sims = {u: item_rows[u] for u in lv}

        def judge_ms(u):
            return sum(v["ms"] for v in judge.get(u, {}).values() if isinstance(v, dict))

        wall = dict(
            no_intervention=[s0[u]["t_total_ms"] for u in lv],
            probe_restart=[live[u]["probe_restart"]["wall_ms"] for u in lv],
            judge_restart=[s0[u]["t_total_ms"] + judge_ms(u) + (live[u]["judge_restart"]["wall_ms"]
                                                                if "judge_restart" in live[u] else 0) for u in lv],
            always_retrieve=[live[u]["always_retrieve"]["wall_ms"] for u in lv])
        lat["wall_ms"] = {k: dict(mean=float(np.mean(v)), median=float(np.median(v))) for k, v in wall.items()}
        lat["n"] = len(lv)
        pms = [b["probe_ms"] for u in lv for b in s0[u]["boundaries"].values() if b.get("p_fail") is not None]
        lat["probe_ms_per_boundary"] = float(np.mean(pms)) if pms else None
        lat["judge_ms_per_trajectory"] = float(np.mean([judge_ms(u) for u in lv]))
        rms = [r["retrieval_ms"] for u in lv for r in rs[u].values()]
        lat["retrieval_ms"] = float(np.mean(rms)) if rms else None
        # the live subset runs the probe system with the run's --rounds; compare with the same simulated system
        live_rounds = max(live[u]["probe_restart"].get("rounds", 1) for u in lv)
        # (the live probe ran in the run's probe mode)
        if live_rounds == 2 and has_round2:
            sim = p2_key
        else:
            sim = "probe_restart" if run_mode == probe_mode else "probe_restart_alt"
        lat["live_probe_system"] = sim
        lat["equivalence"] = dict(
            first_pass_is_prefix=float(np.mean([s0[u]["text"].startswith(live[u]["probe_restart"]["first_pass_text"])
                                                for u in lv])),
            same_trigger=float(np.mean([live[u]["probe_restart"]["stopped_at"] == sims[u][sim]["hop"] for u in lv])),
            # every non-empty stop triggered one restart (the last generation runs with the probe off)
            same_n_restarts=float(np.mean([
                sum(s is not None for s in live[u]["probe_restart"].get("stops", [live[u]["probe_restart"]["stopped_at"]]))
                == sims[u][sim]["n_restarts"] for u in lv])),
            same_answer=float(np.mean([live[u]["probe_restart"]["answer"].strip() == sims[u][sim]["answer"].strip()
                                       for u in lv])),
            same_correctness=float(np.mean([
                answer_correct(live[u]["probe_restart"]["answer"], it_by[u]["gold_answer"]) == sims[u][sim]["correct"]
                for u in lv])))
    R["latency_live_subset"] = lat

    # ── RQ4 + parity ─────────────────────────────────────────────────────────
    rq4 = {}
    for t in ("hop1", "hop2", "hop12"):
        cv = {int(L): a for L, a in meta.get("cv_auroc", {}).get(t, {}).items()}
        if len(cv) > 2:
            best = max(cv.values())
            rq4[t] = dict(cv_auroc=cv, best_layer=max(cv, key=cv.get), best=best,
                          earliest_within_0_01=min(L for L, a in cv.items() if a >= best - 0.01))
    R["rq4_layers"] = rq4
    R["probe_training"] = {t: {k: v for k, v in p.items() if k != "test_strata"} | {"test_strata": p.get("test_strata")}
                           for t, p in meta["probes"].items()}
    pj = out / "parity.json"
    if pj.exists():
        par = json.loads(pj.read_text())
        rows = par.get("rows", [])
        R["parity"] = dict(n_reproduced=par.get("n_reproduced"), n_2wiki_s0=par.get("n_2wiki_s0"), n_hops=len(rows),
                           mean_cos_probe_layer=float(np.mean([r["cos_probe_layer"] for r in rows])) if rows else None,
                           mean_abs_dp=float(np.mean([abs(r["p_live"] - r["p_stored"]) for r in rows])) if rows else None,
                           max_abs_dp=float(np.max([abs(r["p_live"] - r["p_stored"]) for r in rows])) if rows else None)

    (out / "results.json").write_text(json.dumps(R, indent=1, default=float))
    with open(out / "per_item.jsonl", "w", encoding="utf-8") as f:
        for r in per_item:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "results.md").write_text(_markdown(R), encoding="utf-8")
    (out / "tables.tex").write_text(_latex(R), encoding="utf-8")
    print((out / "results.md").read_text(encoding="utf-8"))
    return R


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _markdown(R: dict) -> str:
    L = [f"# Phase 3–5 results — setting `{R['setting']}`", "",
         f"Items evaluated: {R['n_items']}.  Main probe: {MODE_TEXT[R['probe_mode']]}; layers {R['probe_layers']}, thresholds "
         f"{ {k: round(v, 3) for k, v in R['tau'].items()} } (F0.5-optimal on train-split CV).", ""]
    det = R.get("detection_2wiki", {})
    if det:
        L += ["## RQ1 — failure detection (2Wiki held-out test, live trajectories)", "",
              f"Label sources: {R.get('label_sources')}", "",
              "| rows | trigger | n | fail | AUROC | precision | recall | F0.5 | FPR | flag rate |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for rowname in ("hop1", "hop2", "trajectory"):
            for trig in sorted(det.get(rowname, {}), key=lambda t: (t != "probe", t == "judge")):
                m = det.get(rowname, {}).get(trig)
                if m:
                    L.append(f"| {rowname} | {trig} | {m['n']} | {m['n_fail']} | {_f(m.get('auroc'))} | {_f(m['precision'])} | "
                             f"{_f(m['recall'])} | {_f(m['f05'])} | {_f(m['fpr'])} | {_f(m['flag_rate'])} |")
        t = det.get("trajectory")
        if t:
            L += ["", f"First-failure localization accuracy (on failing trajectories): probe "
                      f"{_f(t['probe']['localization_acc_on_failures'])}, judge {_f(t['judge']['localization_acc_on_failures'])}. "
                      f"Exact first-fail hop incl. 'no failure' (all trajectories): probe "
                      f"{_f(t['probe']['exact_first_fail_acc_all'])}, judge {_f(t['judge']['exact_first_fail_acc_all'])}.", ""]
    for ds, e in R.get("end_to_end", {}).items():
        L += [f"## RQ3 — end-to-end accuracy, {ds}", "",
              "| system | acc | Δ vs none [95% CI] | EM | F1 | restart rate | fixed | broken | gen tok | processed tok | detect tok |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, m in e.items():
            if not isinstance(m, dict) or "acc" not in m:
                continue
            ci = m.get("acc_delta_ci95")
            delta = f"{m['acc_delta_vs_none']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "–"
            L.append(f"| {LABEL.get(s, s)} | {_f(m['acc'])} | {delta} | {_f(m['em'])} | {_f(m['f1'])} | {_f(m['fire_rate'])} | "
                     f"{m['fixed']} | {m['broken']} | {_f(m['gen_tokens'], 1)} | {_f(m['processed_tokens'], 0)} | "
                     f"{_f(m['mean_detect_tokens'], 1)} |")
        ci = e.get("probe_minus_judge_acc_ci95")
        if ci:
            L += ["", f"Probe − judge accuracy, paired bootstrap 95% CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]"]
        ci2 = e.get("probe2r_minus_judge2r_acc_ci95")
        if ci2:
            L += [f"Probe − judge accuracy, both with up to 2 restarts: [{ci2[0]:+.3f}, {ci2[1]:+.3f}]; "
                  f"share of questions restarted twice: {e['second_restart_rate']}"]
        if ds == "2wiki" and R.get("acc_optimal_tau"):
            L += [f"Accuracy-optimal τ (τ1, τ2), chosen on the other half of the 2Wiki questions: {R['acc_optimal_tau']}"]
        bc = e.get("by_condition")
        if bc:
            names = [s for s in LABEL if any(s in v for v in bc.values())]
            L += ["", "Accuracy by first-stage retrieval condition:", "",
                  "| condition | n | " + " | ".join(LABEL[s] for s in names) + " |",
                  "|---|---|" + "---|" * len(names)]
            for cond, v in bc.items():
                L.append(f"| {cond} | {v['n']} | " + " | ".join(_f(v.get(s)) for s in names) + " |")
        L.append("")
    fap = R.get("final_answer_failure_auroc")
    if fap:
        L += ["## Final-answer failure prediction (AUROC of max hop P(fail) vs. wrong final answer)", "",
              "| dataset | n | failure rate | probe | judge |", "|---|---|---|---|---|"]
        for ds, m in fap.items():
            L.append(f"| {ds} | {m['n']} | {_f(m['base_failure_rate'])} | {_f(m['probe_max_pfail'])} | {_f(m['judge_max_pfail'])} |")
        L.append("")
    lat = R.get("latency_live_subset")
    if lat:
        L += [f"## Latency (sequential, live subset n={lat['n']})", "", "| system | mean ms | median ms |", "|---|---|---|"]
        for s, m in lat["wall_ms"].items():
            name = LABEL[lat.get("live_probe_system", s)] if s == "probe_restart" else LABEL[s]
            L.append(f"| {name} | {m['mean']:.0f} | {m['median']:.0f} |")
        L += ["", f"Probe cost per hop boundary: {_f(lat['probe_ms_per_boundary'], 2)} ms; judge cost per trajectory: "
                  f"{_f(lat['judge_ms_per_trajectory'], 1)} ms; retrieval: {_f(lat['retrieval_ms'], 1)} ms.",
              f"Simulated-vs-live equivalence: {lat['equivalence']}", ""]
    rq4 = R.get("rq4_layers")
    if rq4:
        L += ["## RQ4 — where failure becomes linearly decodable (train-split CV AUROC)", ""]
        for t, v in rq4.items():
            L.append(f"* {t}: best layer {v['best_layer']} (AUROC {v['best']:.3f}); earliest layer within 0.01 of best: "
                     f"{v['earliest_within_0_01']}")
        L.append("")
    par = R.get("parity")
    if par:
        L += ["## Sanity: live features vs stored Phase 1 features", "",
              f"Reproduced the Phase 1 trajectory exactly: {par['n_reproduced']}/{par['n_2wiki_s0']}. On {par['n_hops']} hops: "
              f"mean cosine at probe layer {_f(par['mean_cos_probe_layer'], 5)}, |ΔP(fail)| mean {_f(par['mean_abs_dp'], 4)}, "
              f"max {_f(par['max_abs_dp'], 4)}.", ""]
    return "\n".join(L)


def _latex(R: dict) -> str:
    L = []
    for ds, e in R.get("end_to_end", {}).items():
        L += [f"% end-to-end, {ds} ({R['setting']})", r"\begin{tabular}{lrrrrr}", r"\toprule",
              r"System & Acc & EM & F1 & Restart rate & Gen.\ tokens \\", r"\midrule"]
        for s, m in e.items():
            if isinstance(m, dict) and "acc" in m:
                L.append(f"{LABEL.get(s, s)} & {m['acc']:.3f} & {m['em']:.3f} & {m['f1']:.3f} & {m['fire_rate']:.2f} & "
                         f"{m['gen_tokens']:.1f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
    det = R.get("detection_2wiki", {})
    if det:
        L += [f"% detection, 2Wiki ({R['setting']})", r"\begin{tabular}{llrrrrr}", r"\toprule",
              r"Rows & Trigger & AUROC & P & R & F$_{0.5}$ & FPR \\", r"\midrule"]
        for rowname in ("hop1", "hop2", "trajectory"):
            for trig in sorted(det.get(rowname, {}), key=lambda t: (t != "probe", t == "judge")):
                m = det.get(rowname, {}).get(trig)
                if m:
                    L.append(f"{rowname} & {trig.replace('_', ' ')} & {_f(m.get('auroc'))} & {m['precision']:.3f} & {m['recall']:.3f} & "
                             f"{m['f05']:.3f} & {_f(m['fpr'])} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def _plot_sweep(out: Path, sweep: list[dict], e2e: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    dss = sorted({s["dataset"] for s in sweep})
    if not dss:
        return
    fig, axes = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        for trig, color, name in (("probe", "#2563eb", "Probe trigger (ours)"), ("judge", "#ea580c", "Judge trigger")):
            pts = sorted((s["gen_tokens"], s["acc"]) for s in sweep if s["dataset"] == ds and s["trigger"] == trig)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=2.5, lw=1.5, color=color, label=name)
        for s, mk, name in (("no_intervention", "s", "No intervention"), ("always_retrieve", "^", "Always retrieve"),
                            ("oracle_restart", "*", "Oracle trigger"), ("probe_restart", "D", "Probe @ calibrated τ")):
            m = e2e.get(ds, {}).get(s)
            if m:
                ax.plot(m["gen_tokens"], m["acc"], mk, color="#111827", ms=7, label=name)
        ax.set_title(ds)
        ax.set_xlabel("mean generated tokens per question")
        ax.set_ylabel("answer accuracy (F1 ≥ 0.5)")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-dir", default=None)
    ap.add_argument("--probe-mode", default="shared", choices=["shared", "dedicated"])
    args = ap.parse_args()
    evaluate(args.out, args.probe_dir, args.probe_mode)


if __name__ == "__main__":
    main()
