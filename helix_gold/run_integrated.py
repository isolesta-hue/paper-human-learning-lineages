"""Integration step B: the method layer on the unified learner, and whether the two layers help or hinder each other
(roadmap step 6, 2026-09-25; design agreed with the user).

Learner3 (integrated.py: the content layer of findings 57-66 as one learner) lives the schema world's life; the method
layer (finding 67's successive halving + finding 50's cross-validated judge) fixes its knobs every session:
  review share {0.1, 0.25, 0.5, 0.75} (start 0.5), learning rate {0.003, 0.01, 0.03} (0.01), order {mixed, new-first},
  review picking {uniform, curious (finding 41)}.
Every session (from the second on): the new categories' sentences minus a quarter held aside, the older categories minus
a fresh quarter; every branch (the current settings + every one-knob move) prepares the session on a copy and practises
50 epochs, the better half on to 100, the better half of those to 200, the best by the judge (score on the held-out
quarters, template and memory routes included) is chosen; then the real session runs with all sentences.
2 x 2: content layer off / on x method layer off / on:
  none     -- the standardised habit alone, start settings
  content  -- all content parts, start settings (= integrated.life_fixed; checked equal to run_schema3)
  method   -- the habit alone + the method layer
  both     -- all content parts + the method layer
Worlds: conj (schema + XOR block) and schema (the user: test both).
Correction found in the first check (one seed, before any run): the judge's held-out OLD sentences had been heard in
earlier sessions and whole-sentence memory recalled them verbatim, so the judge graded recall (all parts + method layer
ended 0.47 against 0.60 with start settings). The judge now leaves each sentence's own memory out (Learner3.judge_score). Second check: the judge still preferred a slow
learning rate that the outside test did not -- on OLD categories finding 50's held-out sentences had been practised in
earlier sessions, so the judge graded familiarity (finding 48's bias; in this world remembering sentences and generalising
to new combinations part ways: judge old 0.458 vs 0.377 for learning rate 0.003 vs 0.03, outside old 0.500 vs 0.561).
Old categories are now judged on new combinations the learner makes itself (Learner3.recombined_score: a word with a
strongly tied partner, the partner, and other known words of the category -- the outside test's shape, never heard).
Completion test, fixed before looking (README roadmap step 6): gap = both - content, per session, outside test (never-heard
items): (1) the gap at the end > 0, and (2) the mean gap of sessions 6-8 > that of sessions 2-4, each by the sign test
(p < 0.01), 100 seeds. Seeds: 30 first, cached; 100 if a key comparison is undecided.
"""
import math
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

import data_v3 as W
from run_unified_continual import sign_p
from run_schema import D_positions, PER_SESSION, EPOCHS, acc_items
from integrated import Learner3, START, build_world, measure

KNOBS = {"review": [0.1, 0.25, 0.5, 0.75], "lr": [0.003, 0.01, 0.03], "order": ["mixed", "new-first"],
         "pick": ["uniform", "curious"]}
MOVES = ["review+", "review-", "lr+", "lr-", "order", "pick"]
RUNGS = [50, 100, 200]
HOLD, EXPECT_RATE = 0.25, 0.5
WORLDS = ["conj", "schema"]
CONDITIONS = ["none", "content", "method", "both"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_int")


def apply(s, move):
    k, sign = (move[:-1], move[-1]) if move[-1] in "+-" else (move, None)
    vals = KNOBS[k]
    i = vals.index(s[k])
    j = (i + 1) % len(vals) if sign is None else i + (1 if sign == "+" else -1)
    if not 0 <= j < len(vals):
        return None
    t = dict(s)
    t[k] = vals[j]
    return t


def score_items(model, items):
    return acc_items(model, items) if items else float("nan")


def life(world_name, cond, seed, chunk=True, knobs=None, make=None):
    global KNOBS
    knobs_saved = KNOBS
    if knobs is not None:
        KNOBS = knobs
    try:
        return _life(world_name, cond, seed, chunk, make)
    finally:
        KNOBS = knobs_saved


def _life(world_name, cond, seed, chunk, make=None):
    w = build_world(world_name, seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    mrng = np.random.default_rng(60_000 + seed)       # the method layer's own draws (so "content" = life_fixed)
    me = (make or Learner3)(w, seed, content=cond in ("content", "both"), chunk=chunk)
    method = cond in ("method", "both")
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    aside = {}
    for c in range(w.n_cat):
        idx = mrng.permutation(len(sents[c]))
        aside[c] = set(idx[:int(round(len(sents[c]) * HOLD))].tolist())
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    settings = dict(START)
    seen, steps = [], []
    acc_now, expect, rpe = {}, {}, {}
    for i, new_cats in enumerate(sessions):
        old = list(seen)
        seen += new_cats
        cost = np.nan
        if method and i > 0:
            cost = 0
            trial_new = [s for c in new_cats for j, s in enumerate(sents[c]) if j not in aside[c]]
            judged = {c: [s for j, s in enumerate(sents[c]) if j in aside[c]] for c in new_cats}
            trial_old = {}
            for c in old:
                idx = mrng.permutation(len(sents[c]))
                k = int(round(len(sents[c]) * HOLD))
                judged[c] = [sents[c][j] for j in idx[:k]]
                trial_old[c] = [sents[c][j] for j in idx[k:]]
            available = [m for m in MOVES if apply(settings, m) is not None]
            branches = []
            for m in [None] + available:
                s = settings if m is None else apply(settings, m)
                b = me.clone()
                b.prepare(trial_new, trial_old, s, np.random.default_rng(mrng.integers(1 << 31)), rpe)
                branches.append((m, s, b))

            jseed = int(mrng.integers(1 << 31))

            def judge(b):
                # new categories: their held-out sentences (never practised); old categories: new combinations the learner
                # makes from their held-out sentences (they were practised before -- finding 48's familiarity bias)
                return float(np.mean([b.judge_score(ss) if c in new_cats else
                                      b.recombined_score(ss, np.random.default_rng(jseed + c)) for c, ss in judged.items()]))
            alive = branches
            for rung in RUNGS:
                for _, _, b in alive:
                    before = b.st["done"]
                    b.train_to(rung)
                    cost += b.st["done"] - before
                if rung == RUNGS[-1]:
                    break
                alive = sorted(alive, key=lambda x: -judge(x[2]))[:math.ceil(len(alive) / 2)]
            settings = dict(max(alive, key=lambda x: judge(x[2]))[1])
        checks = {"once": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "once"],
                  "conj": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind in ("conj", "long")]}
        curves = me.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, settings, rng, rpe,
                            checks=checks)
        if method:                                   # surprise per category (finding 41), for "curious" review picking
            for c in seen:
                a = score_items(me.model, D_positions(sents[c]))
                if c in acc_now:
                    lp = a - acc_now[c]
                    rpe[c] = lp - expect.get(c, 0.0)
                    expect[c] = expect.get(c, 0.0) + EXPECT_RATE * (lp - expect.get(c, 0.0))
                acc_now[c] = a
        st = measure(me, tests, seen, new_cats, curves, seed)
        st["older"] = score_items(me.model, [(cx, t) for c in old for kind, cx, t in tests[c]]) if old else np.nan
        st.update({"review": settings["review"], "lr": settings["lr"], "new-first": float(settings["order"] == "new-first"),
                   "curious": float(settings["pick"] == "curious"), "trial epochs": cost})
        steps.append(st)
    return steps


def one_seed(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    out = {(wn, c): life(wn, c, seed) for wn in WORLDS for c in CONDITIONS}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_sess = len(results[0][(WORLDS[0], CONDITIONS[0])])

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    print(f"=== {n_seeds} seeds; integration step B; knobs {KNOBS}; start {START} ===")
    for key in ["never-heard", "newest", "older", "conj all", "once all", "fmt options", "fmt candidate",
                "review", "lr", "new-first", "curious", "trial epochs"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS:
            for c in CONDITIONS:
                print(f"{wn:6} {c:8} " + " ".join(f"{m([r[(wn, c)][i][key] for r in results]):>6}" for i in range(n_sess)))

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print("\n--- paired (never-heard) ---")
    for wn in WORLDS:
        for g, b in [("content", "none"), ("method", "none"), ("both", "content"), ("both", "method"), ("both", "none")]:
            life_d = np.array([np.mean([st["never-heard"] for st in r[(wn, g)]]) - np.mean([st["never-heard"] for st in r[(wn, b)]]) for r in results])
            end_d = np.array([r[(wn, g)][-1]["never-heard"] - r[(wn, b)][-1]["never-heard"] for r in results])
            new_d = np.array([np.mean([st["newest"] for st in r[(wn, g)][1:]]) - np.mean([st["newest"] for st in r[(wn, b)][1:]]) for r in results])
            print(f"{wn}: {g:8} vs {b:8} life {sgn(life_d)}  end {sgn(end_d)}  newest s2-8 {sgn(new_d)}")
        inter = np.array([(np.mean([st["never-heard"] for st in r[(wn, "both")]]) - np.mean([st["never-heard"] for st in r[(wn, "content")]]))
                          - (np.mean([st["never-heard"] for st in r[(wn, "method")]]) - np.mean([st["never-heard"] for st in r[(wn, "none")]]))
                          for r in results])
        print(f"{wn}: interaction (method gain with content - method gain without) life {sgn(inter)}")

    print("\n--- completion test: gap = both - content, per session ---")
    for wn in WORLDS:
        gaps = np.array([[r[(wn, "both")][i]["never-heard"] - r[(wn, "content")][i]["never-heard"] for i in range(n_sess)] for r in results])
        print(f"{wn}: gap by session " + " ".join(f"{g:+.3f}" for g in gaps.mean(0)))
        early, late = gaps[:, 1:4].mean(1), gaps[:, 5:8].mean(1)
        print(f"{wn}: (1) end gap {sgn(gaps[:, -1])}   (2) late (s6-8) - early (s2-4) {sgn(late - early)}")


if __name__ == "__main__":
    main()
