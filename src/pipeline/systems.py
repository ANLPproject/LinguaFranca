"""
src/pipeline/systems.py
───────────────────────
The compared systems, computed from cached generations (see __init__.py):

  no_intervention   S0: one uninterrupted greedy trajectory.
  probe_restart     ours: halt at the first hop K whose probe P(fail) >= τ_K,
                    retrieve with (question + hop-K text), flush, restart.
  judge_restart     Doctor-RAG-style: generate the full trajectory, judge hop 1
                    and hop 2 post hoc, restart from the earliest flagged hop.
  always_retrieve   retrieve with the question before generating (no trigger).
  oracle_restart    restart at the gold first-failure hop (2Wiki only):
                    upper bound for any trigger with this repair.

Token accounting
────────────────
gen_tokens       decode steps (the dominant cost), incl. restarts.
processed_tokens prefill + decode tokens, incl. judge prompts and restart prompts.
detect_tokens    tokens generated before the failure was flagged
                 (probe: up to </hopK>; judge: the whole trajectory).
"""

from __future__ import annotations

from src.pipeline.common import answer_correct, answer_em, answer_f1

RESTART_KEY = {0: "q", 1: "h1", 2: "h2"}


def probe_trigger(scores: dict, tau: dict):
    """scores/tau: {1: p, 2: p} (p may be None if the hop never closed)."""
    for k in (1, 2):
        p = scores.get(k)
        if p is not None and p >= tau[k]:
            return k
    return None


def judge_trigger(judge_rec: dict | None, theta: float = 0.5):
    if not judge_rec:
        return None
    for k in (1, 2):
        r = judge_rec.get(str(k))
        if r is not None and r["p_fail"] >= theta:
            return k
    return None


def _restart(rs: dict, k: int):
    return rs.get(RESTART_KEY[k])


def _second_round_probe(rs: dict, k: int, tau: dict):
    """(restart-2 record, tokens of restart 1 before its halt) if the probe also flags restart 1, else (None, None)."""
    r = _restart(rs, k)
    sc = (r or {}).get("rescore", {}).get("probe", {})
    j = probe_trigger({int(h): v["p_fail"] for h, v in sc.items()}, tau)
    r2 = rs.get(f"{RESTART_KEY[k]}>{j}") if j is not None else None
    return (r2, sc[str(j)]["n_tokens"]) if r2 is not None else (None, None)


def _second_round_judge(rs: dict, k: int, theta: float):
    """(restart-2 record or None, judge tokens spent re-checking restart 1)."""
    jr = (_restart(rs, k) or {}).get("rescore", {}).get("judge")
    if jr is None:
        return None, 0
    tok = sum(v["ntok"] for v in jr.values())
    j = judge_trigger(jr, theta)
    return (rs.get(f"{RESTART_KEY[k]}>{j}") if j is not None else None), tok


def outcome(system: str, item: dict, s0: dict, rs: dict, *, probe_scores: dict | None = None,
            tau: dict | None = None, judge_rec: dict | None = None, theta: float = 0.5, rounds: int = 1) -> dict:
    """
    One system on one item.  rs: restart key -> restart record (round-1 records may carry "rescore";
    round-2 records are keyed "h1>2" etc.).  rounds=2 lets probe/judge trigger one more restart.
    """
    n_prompt, n_gen = s0["n_prompt"], s0["n_gen"]
    base = dict(answer=s0["answer"], fired=False, hop=None, gen_tokens=n_gen,
                processed_tokens=n_prompt + n_gen, detect_tokens=None, n_restarts=0)

    if system == "no_intervention":
        res = base
    elif system == "always_retrieve":
        r = rs["q"]
        res = dict(answer=r["answer"], fired=True, hop=0, gen_tokens=r["n_gen"],
                   processed_tokens=r["n_prompt"] + r["n_gen"], detect_tokens=0, n_restarts=1)
    elif system in ("probe_restart", "oracle_restart"):
        if system == "probe_restart":
            k = probe_trigger(probe_scores or {}, tau)
        else:
            k = item.get("first_fail")
            if k is not None and str(k) not in s0["boundaries"]:
                k = None                       # the hop never closed: no boundary to act on
        r = _restart(rs, k) if k is not None else None
        if r is None:
            res = base
        else:
            t_k = s0["boundaries"][str(k)]["n_tokens"]
            res = dict(answer=r["answer"], fired=True, hop=k, gen_tokens=t_k + r["n_gen"],
                       processed_tokens=n_prompt + t_k + r["n_prompt"] + r["n_gen"], detect_tokens=t_k, n_restarts=1)
            if rounds == 2 and system == "probe_restart":
                r2, t2 = _second_round_probe(rs, k, tau)
                if r2 is not None:            # restart 1 is halted at its own </hopJ> and restarted once more
                    res.update(answer=r2["answer"], gen_tokens=t_k + t2 + r2["n_gen"], n_restarts=2,
                               processed_tokens=n_prompt + t_k + r["n_prompt"] + t2 + r2["n_prompt"] + r2["n_gen"])
    elif system == "judge_restart":
        judge_tok = sum((judge_rec or {})[k]["ntok"] for k in ("1", "2") if k in (judge_rec or {}))
        k = judge_trigger(judge_rec, theta)
        r = _restart(rs, k) if k is not None else None
        res = dict(base, processed_tokens=n_prompt + n_gen + judge_tok)
        if r is not None:
            res.update(answer=r["answer"], fired=True, hop=k, gen_tokens=n_gen + r["n_gen"],
                       processed_tokens=n_prompt + n_gen + judge_tok + r["n_prompt"] + r["n_gen"],
                       detect_tokens=n_gen, n_restarts=1)
            if rounds == 2:                   # the judge re-checks the complete restart, then restarts once more
                r2, tok2 = _second_round_judge(rs, k, theta)
                res["processed_tokens"] += tok2
                if r2 is not None:
                    res.update(answer=r2["answer"], gen_tokens=res["gen_tokens"] + r2["n_gen"], n_restarts=2,
                               processed_tokens=res["processed_tokens"] + r2["n_prompt"] + r2["n_gen"])
    else:
        raise ValueError(system)

    gold = item["gold_answer"]
    res.update(correct=answer_correct(res["answer"], gold), em=answer_em(res["answer"], gold),
               f1=answer_f1(res["answer"], gold))
    return res
