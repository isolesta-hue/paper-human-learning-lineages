"""Exam v1 per-part probe (2026-10-20; the user's decision 10-20 "응, 두가지 다 진행해도 좋을거 같아." -- after the sixteenth
sibling reading: Sia's NH 144 measured, criterion by criterion, what its body learned and what its added parts answer, and
the first paper (ledger P6) should state Dami's design reliance per criterion, not in one line). D67 diagnosis -- Dami is
judged as a whole; the parts are diagnosis only. Read-only:
  * a few of Dami's v1 exam lives run again with the exam's own code (fingerprint 0551dd3a140d checked first), learning
    exactly as in the exam: every answer row of the probe life must equal the recorded one (checked, exact), and the body
    must not change while answering (run_exam's own check, plus one around the extra pass below);
  * at each item's JUDGED session (the grader: "every" / "end" -> after session 15, "after_session_k" -> k) the answer is
    also split into its parts (finding 112's exact split, run_long.answer_parts: habit similarity "sim", habit connection
    table "assoc", templates "pattern", whole-sentence memory "chunk", self-made side part "side", world model "wm" when
    there is one) and recomputed (a) with each part removed, (b) with each part alone, (c) habit = sim + assoc removed /
    alone, (d) without the kind gate (finding 90) -- ANSWER-TIME removal only (the life learned with every part; Sia's
    NH 144 removed parts for whole lives, which this does not do);
  * one extra pass: every item answered by the newborn body before session 0 ("experience 0": what the structure answers
    with nothing heard).
No key is read (the edition's keys stay sealed until grading); analyse_exam_parts.py judges the variants afterwards with
the grader's own rules. Output per life: OUT/LIFE.parts.jsonl (one row per judged item + one per item before session 0:
{id, after_session, format, variants: {name: scores}}), OUT/LIFE.answers.jsonl (the probe life's own answer file),
OUT/LIFE.json (checks, cost). ANSWERS_DIR "-" = no recorded answers to compare (a development life, mechanics only).
    python3 diag_exam_parts.py EDITION_DIR ANSWERS_DIR OUT_DIR LIFE[,LIFE...] [WORKERS] [--sessions N]"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

import conv_v1 as CV
import run_exam as RE
import run_long as RL
from lab2 import pad

FP = "0551dd3a140d"
PARTS = ("sim", "assoc", "pattern", "chunk", "side", "wm")
LAST = 15
REC, STATE = [], {}
_answer = CV.answer
_convert = CV.convert_life


def judged(when):
    return int(when.rsplit("_", 1)[1]) if when.startswith("after_session_") else LAST


def variant_logits(model, ctx):
    """{variant: logits [1, V]} from the exact split; "real" = the model's own complete (what the exam answered with)."""
    parts, blocked, err = RL.answer_parts(model, ctx)
    have = [p for p in PARTS if p in parts]         # every part the body has (the same variants on every row) ...
    nonzero = [p for p in have if float(parts[p].abs().max()) > 0]    # ... and the ones that spoke on this row

    def finish(v):                                  # the kind gate (when the body has it on) and the known-word mask
        return model.only_known(model.kind_first(ctx, v))
    full = sum(parts[p] for p in parts)
    with torch.no_grad():
        real = model.complete(ctx)
    out = {"real": real, "full": finish(full), "nogate": model.only_known(full)}
    for p in have:
        out["-" + p] = finish(full - parts[p])
        out[p] = finish(parts[p])
    if "sim" in have and "assoc" in have:
        out["-habit"] = finish(full - parts["sim"] - parts["assoc"])
        out["habit"] = finish(parts["sim"] + parts["assoc"])
    dev = float((out["full"] - real).masked_fill(real < -1e8, 0).abs().max())
    return out, err, dev, nonzero


def probe_row(body, q, after_session):
    """the item's scores under every variant, computed exactly as conv_v1.answer computes the real ones."""
    model, seq = body.me.model, q.sequence(body.w.boundary)
    if q.format in ("choice", "pair"):
        per_t, errs, devs, spoke = {}, [], [], set()
        for lab, states in q.options:
            for t, _ in states:
                if t not in per_t:
                    lg, err, dev, nz = variant_logits(model, pad([seq + [t]]))
                    errs.append(err)
                    devs.append(dev)
                    spoke.update(nz)
                    per_t[t] = {}
                    for name, v in lg.items():
                        p = torch.log_softmax(v, dim=1)[0].exp()     # as DamiBody.state_probs (log-softmax, then exp)
                        d = {c: float(p[c]) for c in body.cells}
                        d[CV.ABSENT] = max(0.0, 1.0 - sum(d.values()))
                        per_t[t][name] = d
        names = list(next(iter(per_t.values())))
        var = {}
        for name in names:
            ll = {lab: sum(float(np.log(max(float(per_t[t][name].get(a, 0.0)), 1e-12))) for t, a in states)
                  for lab, states in q.options}
            if q.format == "pair":
                m = max(ll.values())
                z = sum(np.exp(v - m) for v in ll.values())
                ll = {lab: float(np.exp(v - m) / z) for lab, v in ll.items()}
            var[name] = ll
        return {"id": q.id, "after_session": after_session, "format": q.format, "variants": var,
                "err": max(errs), "dev": max(devs), "spoke": sorted(spoke)}
    lg, err, dev, nz = variant_logits(model, pad([seq]))
    var = {}
    for name, v in lg.items():
        lp = torch.log_softmax(v, dim=1)[0]
        var[name] = {k: (0.0 if isinstance(tok, tuple) else float(lp[tok])) for k, tok in q.cands}
    return {"id": q.id, "after_session": after_session, "format": q.format, "variants": var, "err": err, "dev": dev,
            "spoke": nz}


def answer_and_probe(body, q, after_session):
    row = _answer(body, q, after_session)
    if after_session == judged(q.when):
        pr = probe_row(body, q, after_session)
        pr["real_equal"] = pr["variants"]["real"] == row["scores"]     # the probe's "real" = the exam's own answer
        REC.append(pr)
    return row


def convert_and_keep(life_dir, tag=None):
    w, qs = _convert(life_dir, tag) if tag is not None else _convert(life_dir)
    STATE["w"], STATE["qs"] = w, qs
    return w, qs


class ProbeLearner(RE.Learner3):
    """the exam's learner, unchanged, plus one answering pass before its first session (experience 0)."""
    def session(self, *a, **k):
        if not STATE.get("pre_done"):
            STATE["pre_done"] = True
            body, before = RE.DamiBody(self, STATE["w"]), RE.learner_fp(self)
            t0 = time.process_time()
            for q in STATE["qs"]:
                REC.append(probe_row(body, q, -1))
            STATE["pre_unchanged"] = RE.learner_fp(self) == before
            STATE["pre_cpu"] = time.process_time() - t0
        return super().session(*a, **k)


def one_life(task):
    life_dir, answers_dir, out_dir, n = task
    life = os.path.basename(os.path.normpath(life_dir))
    REC.clear()
    STATE.clear()
    CV.answer, CV.convert_life, RE.Learner3 = answer_and_probe, convert_and_keep, ProbeLearner
    t_wall = time.time()
    ans = os.path.join(out_dir, f"{life}.answers.jsonl")
    res = RE.exam_life(life_dir, ans, seed=RE.life_seed(life_dir), n_sessions=n)
    with open(os.path.join(out_dir, f"{life}.parts.jsonl"), "w") as f:
        for r in REC:
            f.write(json.dumps(r) + "\n")
    got = {(r["id"], r["after_session"]): r["scores"] for r in map(json.loads, open(ans, encoding="utf-8"))}
    ref = None
    if answers_dir != "-":                          # "-": a development life with no recorded answers (mechanics test)
        ref = {}
        for line in open(os.path.join(answers_dir, f"{life}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if n is None or r["after_session"] < n:
                ref[(r["id"], r["after_session"])] = r["scores"]
    summary = dict(life=life, sessions=n or "all", rows=len(got), compared=ref is not None,
                   same_rows=None if ref is None else set(got) == set(ref),
                   exact=None if ref is None else sum(got[k] == ref.get(k) for k in got),
                   recorded=None if ref is None else len(ref),
                   unchanged_while_answering=res["unchanged_while_answering"],
                   pre_unchanged=STATE.get("pre_unchanged"), pre_cpu_s=round(STATE.get("pre_cpu", 0.0), 1),
                   probe_rows=len(REC), judged_rows=sum(r["after_session"] >= 0 for r in REC),
                   real_equal=sum(bool(r.get("real_equal")) for r in REC if r["after_session"] >= 0),
                   largest_split_err=max((r["err"] for r in REC), default=0.0),
                   largest_full_dev=max((r["dev"] for r in REC), default=0.0),
                   variants=sorted(REC[0]["variants"]) if REC else [], cpu_s=res["cpu_s"],
                   wall_s=round(time.time() - t_wall, 1))
    with open(os.path.join(out_dir, f"{life}.json"), "w") as f:
        json.dump(summary, f)
    return summary


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[args.index("--sessions") + 1]) if "--sessions" in args else None
    if "--sessions" in args:
        i = args.index("--sessions")
        args = args[:i] + args[i + 2:]
    edition, answers, out = args[0], (args[1] if args[1] == "-" else os.path.abspath(args[1])), os.path.abspath(args[2])
    lives = args[3].split(",")
    workers = int(args[4]) if len(args) > 4 else len(lives)
    fp = subprocess.run([sys.executable, "exam_fingerprint.py"], capture_output=True, text=True).stdout.split()[0]
    if fp != FP:
        sys.exit(f"fingerprint {fp} != the exam's {FP}: not the exam's code")
    os.makedirs(out, exist_ok=True)
    tasks = [(os.path.join(edition, "lives", lf), answers, out, n) for lf in lives]
    print(f"probe: lives {lives}, sessions {n or 'all'}, workers {workers}, fingerprint {fp}", flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        for s in pool.imap_unordered(one_life, tasks):
            print(json.dumps(s), flush=True)
    print("exit 0")
