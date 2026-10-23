"""Integration step C: stable/plastic split, a budget as input, the random-world control, and the outer layer's integrity
check (roadmap step 6, 2026-09-26; the user's order after finding 69).

On top of finding 69's integrated learner (all content parts + the method layer):
  split     -- finding 68: a plastic copy is forked from the stable learner at the start of every session and lives the
               session with the start settings; per bucket (categories new this session / older ones) the plastic copy
               answers while, in the session's trial, the plastic settings have beaten the chosen branch on the judge at
               least as often as they lost (finding 61's stakes rule). Finding 68 on the easy world: the newest categories
               came back to the fixed-settings level, the whole life did not change.
  budget    -- the method layer takes its budget as input (the user's principle: logic that fits whatever compute it gets):
               low = rungs 25/50/100/200 (keep the better half each time), mid = 50/100/200 (finding 69), high = every branch
               to 200 (finding 51's "full"). Graceful means: more budget -> a little better, never a cliff.
  random    -- finding 69's design named the random world as the control (does the self-correcting learner collapse where
               there are no regularities?) -- it was left out; here it is.
  integrity -- the outer layer's checksum: the original memory (Module A's vectors) is fingerprinted at birth and checked
               after every session (never changed = "intact").
Conditions: content / both (finding 69, reused from its cache where present) / both+split / both-low / both-high.
Worlds: schema (all), conj (content, both, both+split), random (content, both). Seeds: 30 first (cached).
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
import run_integrated as RI

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_int3")
PLAN = {"schema": ["content", "both", "both+split", "both-low", "both-high"],
        "conj": ["content", "both", "both+split"],
        "random": ["content", "both"]}
RUNGS = {"low": [25, 50, 100, 200], "mid": [50, 100, 200], "high": [200]}


def life(world_name, cond, seed):
    if cond in ("content", "both") and world_name in ("schema", "conj"):
        path = os.path.join(RI.CACHE, f"seed{seed}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)[(world_name, cond)]
    budget = cond.split("-")[1] if cond.startswith("both-") else "mid"
    return run(world_name, "content" if cond == "content" else "both", seed, split="split" in cond, budget=budget)


def run(world_name, cond, seed, split=False, budget="mid"):
    w = build_world(world_name, seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    mrng = np.random.default_rng(60_000 + seed)
    me = Learner3(w, seed, content=True)
    method = cond == "both"
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    aside = {}
    for c in range(w.n_cat):
        idx = mrng.permutation(len(sents[c]))
        aside[c] = set(idx[:int(round(len(sents[c]) * RI.HOLD))].tolist())
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    settings = dict(START)
    seen, steps = [], []
    acc_now, expect, rpe = {}, {}, {}
    wins = {"new": [0, 0], "old": [0, 0]}
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
                k = int(round(len(sents[c]) * RI.HOLD))
                judged[c] = [sents[c][j] for j in idx[:k]]
                trial_old[c] = [sents[c][j] for j in idx[k:]]
            available = [m for m in RI.MOVES if RI.apply(settings, m) is not None]
            branches = []
            for m in [None] + available:
                s = settings if m is None else RI.apply(settings, m)
                b = me.clone()
                b.prepare(trial_new, trial_old, s, np.random.default_rng(mrng.integers(1 << 31)), rpe)
                branches.append((m, s, b))
            jseed = int(mrng.integers(1 << 31))

            def part(b, cs):
                return float(np.mean([b.judge_score(judged[c]) if c in new_cats else
                                      b.recombined_score(judged[c], np.random.default_rng(jseed + c)) for c in cs]))

            def judge(b):
                return part(b, list(judged))
            alive = branches
            for rung in RUNGS[budget]:
                for _, _, b in alive:
                    before = b.st["done"]
                    b.train_to(rung)
                    cost += b.st["done"] - before
                if rung == RUNGS[budget][-1]:
                    break
                alive = sorted(alive, key=lambda x: -judge(x[2]))[:math.ceil(len(alive) / 2)]
            best = max(alive, key=lambda x: judge(x[2]))
            if split:
                pt = me.clone()
                pt.prepare(trial_new, trial_old, dict(START), np.random.default_rng(mrng.integers(1 << 31)), rpe)
                pt.train_to(EPOCHS)
                cost += EPOCHS
                for bucket, cs in (("new", [c for c in judged if c in new_cats]), ("old", [c for c in judged if c not in new_cats])):
                    if cs:
                        a, b_ = part(pt, cs), part(best[2], cs)
                        wins[bucket][0] += a > b_
                        wins[bucket][1] += a < b_
            settings = dict(best[1])
        plastic = None
        if split and i > 0:
            plastic = me.clone()
            plastic.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, dict(START),
                            np.random.default_rng(mrng.integers(1 << 31)), rpe)
        checks = {"once": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "once"],
                  "conj": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind in ("conj", "long")]}
        curves = me.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, settings, rng, rpe,
                            checks=checks)
        if method:
            for c in seen:
                a = acc_items(me.model, D_positions(sents[c]))
                if c in acc_now:
                    lp = a - acc_now[c]
                    rpe[c] = lp - expect.get(c, 0.0)
                    expect[c] = expect.get(c, 0.0) + RI.EXPECT_RATE * (lp - expect.get(c, 0.0))
                acc_now[c] = a
        st = measure(me, tests, seen, new_cats, curves, seed)

        def use(bucket):
            return plastic is not None and wins[bucket][0] >= wins[bucket][1] and sum(wins[bucket]) > 0
        if plastic is not None and (use("new") or use("old")):
            items = {c: [(cx, t) for kind, cx, t in tests[c]] for c in seen}
            per = {c: acc_items((plastic if use("new" if c in new_cats else "old") else me).model, items[c]) for c in seen}
            n_it = {c: len(items[c]) for c in seen}
            st["never-heard"] = sum(per[c] * n_it[c] for c in seen) / sum(n_it.values())
            st["newest"] = sum(per[c] * n_it[c] for c in new_cats) / sum(n_it[c] for c in new_cats)
        st["older"] = acc_items(me.model, [(cx, t) for c in old for kind, cx, t in tests[c]]) if old else np.nan
        st.update({"review": settings["review"], "lr": settings["lr"], "trial epochs": cost, "intact": float(me.intact),
                   "plastic new": float(use("new")) if split else np.nan, "plastic old": float(use("old")) if split else np.nan})
        steps.append(st)
    return steps


def one_seed(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    out = {(wn, c): life(wn, c, seed) for wn, conds in PLAN.items() for c in conds}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n = len(results[0][("schema", "content")])

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; integration step C ===")
    for key in ["never-heard", "newest", "older", "lr", "review", "trial epochs", "plastic new", "plastic old", "intact"]:
        print(f"\n--- {key} ---")
        for wn, conds in PLAN.items():
            for c in conds:
                print(f"{wn:6} {c:11} " + " ".join(f"{m([r[(wn, c)][i].get(key, np.nan) for r in results]):>6}" for i in range(n)))
    print("\n--- paired ---")
    for wn, conds in PLAN.items():
        for g, b in [("both", "content"), ("both+split", "both"), ("both-low", "both"), ("both-high", "both")]:
            if g not in conds or b not in conds:
                continue
            dl = np.array([np.mean([s["never-heard"] for s in r[(wn, g)]]) - np.mean([s["never-heard"] for s in r[(wn, b)]]) for r in results])
            de = np.array([r[(wn, g)][-1]["never-heard"] - r[(wn, b)][-1]["never-heard"] for r in results])
            dn = np.array([np.nanmean([s["newest"] for s in r[(wn, g)][1:]]) - np.nanmean([s["newest"] for s in r[(wn, b)][1:]]) for r in results])
            print(f"{wn}: {g:11} vs {b:8} life {sgn(dl)}  end {sgn(de)}  newest s2-8 {sgn(dn)}")
        gaps = np.array([[r[(wn, "both")][i]["never-heard"] - r[(wn, "content")][i]["never-heard"] for i in range(n)] for r in results])
        print(f"{wn}: completion gap by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0))
              + f" | end {sgn(gaps[:, -1])} | late - early {sgn(gaps[:, 5:8].mean(1) - gaps[:, 1:4].mean(1))}")
    print("\n--- budget (schema): trial epochs per life vs outside end ---")
    for c in ["both-low", "both", "both-high"]:
        print(f"{c:10} epochs {np.mean([np.nansum([s['trial epochs'] for s in r[('schema', c)]]) for r in results]):.0f}  "
              f"end {np.mean([r[('schema', c)][-1]['never-heard'] for r in results]):.3f}")


if __name__ == "__main__":
    main()
