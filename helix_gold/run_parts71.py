"""Content layer, roadmap step 5's remaining parts: templates chained, and a setter and a solver inside one learner
(2026-09-25; the user chose to test them as parts before the long test, then offer them there as options).

  compose  -- templates chained, a simulation inside the templates (roadmap 5 (2)): for every two templates in use with
              at least 4 confirmations, t1 +/- t2 is a composed template of weight w1 x w2 (integrated.composed_matrix).
              World: data_v3 compose=True -- every category also has a pair c = a + DELTA_k1 + DELTA_k2, heard only apart.
  setter   -- setter and solver (roadmap 5 (3), the user's idea after game-playing AIs): the setter makes new-combination
              questions from what was just heard (a word with a strongly tied partner, the partner, other known words of
              its category) and keeps those the solver gets wrong before practice; they are practised with the rest. The
              answers come from real heard sentences, so the setter cannot exploit a wrong model (finding 36).
  setter-random -- control: as many such questions, not chosen by failure.
All on the unified learner (integrated.Learner3: all content parts of finding 69), start settings (no method layer).
Worlds: compose (schema + composed pairs) / random (no regularities: does either part do harm there?).
Seeds: two stages (CLAUDE.md). Key comparisons fixed before looking: compose vs base, setter vs base, setter vs
setter-random (per world, life).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

import data_v3 as W
from run_unified_continual import sign_p
from run_schema import PER_SESSION
from integrated import Learner3, START, build_world, measure

WORLDS = ["compose", "random"]
CONDITIONS = ["base", "compose", "setter", "setter-random", "compose+setter"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_p71")
KEY = [("compose", "base"), ("setter", "base"), ("setter", "setter-random")]


def life(world_name, cond, seed):
    w = build_world(world_name, seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    me = Learner3(w, seed, compose="compose" in cond,
                  setter=("random" if cond == "setter-random" else "hard") if "setter" in cond else None)
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    seen, steps = [], []
    for new_cats in sessions:
        old = list(seen)
        seen += new_cats
        checks = {"once": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "once"],
                  "conj": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "compose"]}
        curves = me.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, START, rng, checks=checks)
        steps.append(measure(me, tests, seen, new_cats, curves, seed))
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


def life_mean(r, g):
    return np.mean([st["never-heard"] for st in r[g]])


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n = len(results[0][(WORLDS[0], CONDITIONS[0])])

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; templates chained, setter and solver ===")
    for key in ["never-heard", "newest", "compose all", "unseen all", "once all", "pairs all"]:
        print(f"\n--- {key} ---")
        for wn in WORLDS:
            for c in CONDITIONS:
                print(f"{wn:7} {c:15} " + " ".join(f"{m([r[(wn, c)][i][key] for r in results]):>6}" for i in range(n)))
    print("\n--- composed pairs, first sight (epoch 0) ---")
    for c in CONDITIONS:
        print(f"compose {c:15} " + " ".join(f"{m([r[('compose', c)][i]['curve conj'].get(0, np.nan) for r in results]):>6}" for i in range(n)))
    print("\n--- paired ---")
    undecided = []
    for wn in WORLDS:
        for g, b in [("compose", "base"), ("setter", "base"), ("setter", "setter-random"), ("setter-random", "base"),
                     ("compose+setter", "base")]:
            dl = np.array([life_mean(r, (wn, g)) - life_mean(r, (wn, b)) for r in results])
            de = np.array([r[(wn, g)][-1]["never-heard"] - r[(wn, b)][-1]["never-heard"] for r in results])
            dc = np.array([np.nan_to_num(r[(wn, g)][-1]["compose all"]) - np.nan_to_num(r[(wn, b)][-1]["compose all"]) for r in results])
            do = np.array([r[(wn, g)][-1]["once all"] - r[(wn, b)][-1]["once all"] for r in results])
            print(f"{wn}: {g:15} vs {b:13} life {sgn(dl)}  end {sgn(de)}  compose end {sgn(dc)}  once end {sgn(do)}")
            if (g, b) in KEY and n_seeds < 100:
                n_pos, n_neg = int((dl > 1e-9).sum()), int((dl < -1e-9).sum())
                half = 2.756 * dl.std(ddof=1) / np.sqrt(len(dl))
                if not (sign_p(n_pos, n_neg) < 0.005 or (dl.mean() - half > -0.002 and dl.mean() + half < 0.002)):
                    undecided.append(f"{wn}: {g} vs {b}")
    if n_seeds < 100:
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not undecided else "undecided -> extend to 100: " + "; ".join(undecided)))


if __name__ == "__main__":
    main()
