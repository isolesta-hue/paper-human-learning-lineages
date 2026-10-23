"""Diagnosis lives with conv-v1j (finding 166 next (a); the user's decision 10-21 "응, 진단시험을 하자."; pre-registration
DIAG_JOINED.md, fixed before any life runs). Read-only towards the exam: v1's keys are public, the v1 answers stay as graded.
A few of Dami's v1 lives are run again with the exam's own code (fingerprint 0551dd3a140d checked first) and the same body
seed, changing ONE thing: the converter's grouping of the exposure (conv_v1j: one exposure line = one heard unit).
A conv-v1j life is about twice as long as a conv-v1 life (the 2-session smoke: 315 s vs about 150 s), longer than the
2-hour work limit, so the loop below is run_exam.exam_life's loop line for line plus a checkpoint after every session and a
deadline: before a session the life pauses if the last session's time x 1.25 would pass the deadline, and the next piece
goes on from the checkpoint. Checked before the lives (CHECK in DIAG_JOINED.md): cut after session 0 + resume == the
uncut run_exam.exam_life of the smoke, every answer to the last digit.
Each life writes OUT/LIFE.jsonl (its answer file, the shared format) and OUT/LIFE.json (checks, cost):
  - rebuild: every exposure line comes back exactly from the joined units (sentence, line ids, cells), in order;
  - rows: the answer rows are exactly the (id, session) rows the edition asks (the same rows as the recorded answers);
  - keys: every row's score keys are the item's candidates; unchanged: the learner is the same before and after answering.
analyse_joined.py then judges these answers and the recorded conv-v1 answers of the same lives with the grader's rules.
    python3 diag_joined.py EDITION_DIR RECORDED_ANSWERS_DIR OUT_DIR LIFE[,LIFE...] [WORKERS] [--sessions N]
                           [--deadline UNIX_TIME] [--stop-after S]"""
import json
import os
import pickle
import random
import subprocess
import sys
import time

import numpy as np
import torch

import conv_v1 as CV
import conv_v1j as CJ
import run_exam as RE
from integrated import Learner3, START

FP = "78809e282067"   # 0551dd3a140d (the exam) + the memory-only conf_matrix change of 10-21 (run_schema3.iter_template_matrices):
                      # the exam's conv-v1 answers and the conv-v1j smoke are byte-identical before and after (DIAG_JOINED.md)


def rebuild_ok(life_dir, w):
    expo = CV.load_jsonl(os.path.join(life_dir, "exposure.jsonl"))
    want = [{"session": r["session"], "sentence": r["sentence"],
             "scene": {"lines": [{"id": ln["id"], "cells": [list(c) for c in ln["cells"]]} for ln in r["scene"]["lines"]]}}
            for r in expo]
    got = CJ.rebuild_records(w)
    return got == want, len(got), len(want)


def exam_life_resumable(life_dir, out_path, seed, ckpt, n_sessions=None, deadline=None, stop_after=None):
    """run_exam.exam_life's loop, line for line, + a checkpoint after each session (the learner, its random stream, the rows,
    the global random states) and a deadline. Returns exam_life's result dict, or {"paused_after": s} when it stops early."""
    t0 = time.process_time()
    w, qs = CV.convert_life(life_dir)
    st = None
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            st = pickle.load(f)
    if st is not None:
        me, rng, rows, seen, unchanged, s0, cpu0 = (st["me"], st["rng"], st["rows"], st["seen"], st["unchanged"],
                                                    st["next"], st["cpu"])
        w = me.w                                    # the learner's own world (equal to the one just converted)
        torch.set_rng_state(st["torch"])
        np.random.set_state(st["np"])
        random.setstate(st["py"])
        print(f"resumed at session {s0} (cpu so far {cpu0:.0f}s)", flush=True)
    else:
        me = Learner3(w, seed, content=True, where="linear", rehearse=True)
        rng = np.random.default_rng(70_000 + seed)
        rows, seen, unchanged, s0, cpu0 = [], [0], True, 0, 0.0
    settings = dict(START, kind_first=True, cue_templates=True)
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    n = len(w.sessions) if n_sessions is None else n_sessions
    last = None
    for s in range(s0, n):
        if deadline is not None and last is not None and time.time() + 1.25 * last > deadline:
            return {"paused_after": s - 1, "cpu_s": round(cpu0 + time.process_time() - t0, 1)}
        t_s = time.time()
        c = w.sessions[s]
        me.session(sents[c], {o: sents[o] for o in seen if sents[o]}, settings, rng)
        seen.append(c)
        body, before = RE.DamiBody(me, w), RE.learner_fp(me)
        for q in qs:
            if s in CV.sessions_to_answer(q.when):
                rows.append(CV.answer(body, q, s))
        unchanged = unchanged and RE.learner_fp(me) == before
        cpu = cpu0 + time.process_time() - t0
        print(f"session {s}: rows {len(rows)}, cpu {cpu:.0f}s, unchanged {unchanged}", flush=True)
        if s + 1 < n:
            state = {"me": me, "rng": rng, "rows": rows, "seen": seen, "unchanged": unchanged, "next": s + 1, "cpu": cpu,
                     "torch": torch.get_rng_state(), "np": np.random.get_state(), "py": random.getstate()}
            with open(ckpt + ".tmp", "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(ckpt + ".tmp", ckpt)
            if stop_after is not None and s == stop_after:
                return {"paused_after": s, "cpu_s": round(cpu, 1)}
        last = time.time() - t_s
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    due = sum(len([x for x in CV.sessions_to_answer(q.when) if x < n]) for q in qs)
    keys_ok = all(set(r["scores"]) == {k for k, _ in q.cands} for q in qs for r in rows if r["id"] == q.id)
    return {"rows": len(rows), "due": due, "keys_match": keys_ok, "unchanged_while_answering": unchanged,
            "cpu_s": round(cpu0 + time.process_time() - t0, 1), "V": w.V, "items": len(qs)}


def one_life(task):
    life_dir, recorded, out_dir, n, deadline, stop_after = task
    life = os.path.basename(os.path.normpath(life_dir))
    CV.convert_life = CJ.convert_life                  # the one change (the loop calls CV.convert_life, as run_exam does)
    t0 = time.time()
    w, _ = CJ.convert_life(life_dir)
    ok, n_units, n_lines = rebuild_ok(life_dir, w)
    ans = os.path.join(out_dir, f"{life}.jsonl")
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    ckpt = os.path.join(out_dir, "ckpt", f"{life}.pkl")
    res = exam_life_resumable(life_dir, ans, RE.life_seed(life_dir), ckpt, n, deadline, stop_after)
    if "paused_after" in res:
        summary = dict(life=life, converter=CJ.VERSION, paused_after=res["paused_after"], cpu_s=res["cpu_s"],
                       wall_s_this_piece=round(time.time() - t0, 1))
        print(json.dumps(summary), flush=True)
        return summary
    got = [(r["id"], r["after_session"]) for r in map(json.loads, open(ans, encoding="utf-8"))]
    ref = {(r["id"], r["after_session"]) for r in map(json.loads, open(os.path.join(recorded, f"{life}.jsonl"), encoding="utf-8"))
           if n is None or r["after_session"] < n}
    summary = dict(life=life, converter=CJ.VERSION, sessions=n or "all", rebuild_ok=ok, units=n_units, exposure_lines=n_lines,
                   rows=len(got), duplicated=len(got) - len(set(got)), same_rows_as_recorded=set(got) == ref,
                   due=res["due"], keys_match=res["keys_match"],
                   unchanged_while_answering=res["unchanged_while_answering"], cpu_s=res["cpu_s"],
                   wall_s_last_piece=round(time.time() - t0, 1), V=res["V"], items=res["items"])
    with open(os.path.join(out_dir, f"{life}.json"), "w") as f:
        json.dump(summary, f)
    return summary


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = {}
    for flag in ("--sessions", "--deadline", "--stop-after"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            args = args[:i] + args[i + 2:]
    n = int(opts["--sessions"]) if "--sessions" in opts else None
    deadline = float(opts["--deadline"]) if "--deadline" in opts else None
    stop_after = int(opts["--stop-after"]) if "--stop-after" in opts else None
    edition, recorded, out = args[0], os.path.abspath(args[1]), os.path.abspath(args[2])
    lives = args[3].split(",")
    workers = int(args[4]) if len(args) > 4 else len(lives)
    fp = subprocess.run([sys.executable, "exam_fingerprint.py"], capture_output=True, text=True).stdout.split()[0]
    if fp != FP:
        sys.exit(f"fingerprint {fp} != the exam's {FP}: not the exam's code")
    os.makedirs(out, exist_ok=True)
    todo = [lf for lf in lives if not os.path.exists(os.path.join(out, f"{lf}.json"))]
    tasks = [(os.path.join(edition, "lives", lf), recorded, out, n, deadline, stop_after) for lf in todo]
    print(f"diagnosis (conv-v1j): lives {todo} (done before: {sorted(set(lives) - set(todo))}), sessions {n or 'all'}, "
          f"workers {workers}, deadline {time.strftime('%H:%M:%SZ', time.gmtime(deadline)) if deadline else '-'}, "
          f"fingerprint {fp}", flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
        for s in pool.imap_unordered(one_life, tasks):
            print(json.dumps(s), flush=True)
    print("exit 0")
