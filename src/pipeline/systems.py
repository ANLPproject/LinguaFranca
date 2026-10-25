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


def outcome(system: str, item: dict, s0: dict, rs: dict, *, probe_scores: dict | None = None,
            tau: dict | None = None, judge_rec: dict | None = None, theta: float = 0.5) -> dict:
    """One system on one item.  rs: restart key -> restart record."""
    n_prompt, n_gen = s0["n_prompt"], s0["n_gen"]
    base = dict(answer=s0["answer"], fired=False, hop=None, gen_tokens=n_gen,
                processed_tokens=n_prompt + n_gen, detect_tokens=None)

    if system == "no_intervention":
        res = base
    elif system == "always_retrieve":
        r = rs["q"]
        res = dict(answer=r["answer"], fired=True, hop=0, gen_tokens=r["n_gen"],
                   processed_tokens=r["n_prompt"] + r["n_gen"], detect_tokens=0)
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
                       processed_tokens=n_prompt + t_k + r["n_prompt"] + r["n_gen"], detect_tokens=t_k)
    elif system == "judge_restart":
        judge_tok = sum((judge_rec or {})[k]["ntok"] for k in ("1", "2") if k in (judge_rec or {}))
        k = judge_trigger(judge_rec, theta)
        r = _restart(rs, k) if k is not None else None
        res = dict(base, processed_tokens=n_prompt + n_gen + judge_tok)
        if r is not None:
            res.update(answer=r["answer"], fired=True, hop=k, gen_tokens=n_gen + r["n_gen"],
                       processed_tokens=n_prompt + n_gen + judge_tok + r["n_prompt"] + r["n_gen"],
                       detect_tokens=n_gen)
    else:
        raise ValueError(system)

    gold = item["gold_answer"]
    res.update(correct=answer_correct(res["answer"], gold), em=answer_em(res["answer"], gold),
               f1=answer_f1(res["answer"], gold))
    return res
