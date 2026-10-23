"""Screening the dormant parts before the long test (the user's suggestion, 2026-09-26: filter out the clearly useless parts with
a small test first, so the long test is cheaper).

Body: the integrated content layer (Learner3) with the start settings (no method layer: cheap), one dormant part switched on
from birth (Learner3.wake), against the same learner without it. Worlds: schema and conj (all parts), compose (the parts made
for it). 10 seeds, the normal 8-session life.
Rule, fixed before running (lenient on purpose -- the method layer's judge will decide again in the long test; this only drops
parts that never help anything):
  a part is WOKEN-ABLE in the long test if, in some world, the life mean on never-heard combinations OR its own target measure
  (once-heard pairs / XOR / heard pairs / composed pairs) is higher on average with more seeds up than down;
  otherwise it stays asleep (switched off, not removed -- the dormant-parts rule). A part that changes nothing is also dropped.
Also reported: seconds per life (a part the method layer would try costs trial time in every session)."""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

import data_v3 as W
from run_schema import PER_SESSION
from run_unified_continual import sign_p
from integrated import Learner3, START, build_world, measure

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_screen")
PARTS = {"base": {}, "arbiter": {"chunk_mode": "arbiter"}, "ladder": {"chunk_mode": "ladder"},
         "consensus": {"chunk_mode": "consensus"}, "compose": {"compose": True}, "setter-hard": {"setter": "hard"},
         "versions": {"versions": True}, "fmt-practice": {"fmt_practice": True}}
PLAN = {"schema": list(PARTS), "conj": list(PARTS), "compose": ["base", "compose", "setter-hard"]}
TARGET = {"arbiter": "once all", "ladder": "conj all", "consensus": "pairs all", "compose": "compose all",
          "setter-hard": "never-heard", "versions": "once all", "fmt-practice": "fmt options"}


def life(world_name, part, seed):
    w = build_world(world_name, seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    me = Learner3(w, seed)
    s = dict(START, **PARTS[part])
    sents = {c: [x for _, x, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    seen, steps = [], []
    t0 = time.time()
    for new_cats in sessions:
        old = list(seen)
        seen += new_cats
        curves = me.session([x for c in new_cats for x in sents[c]], {c: sents[c] for c in old}, s, rng)
        steps.append(measure(me, tests, seen, new_cats, curves, seed))
    steps[-1]["seconds"] = time.time() - t0
    return steps


def one_seed(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    out = {(wn, p): life(wn, p, seed) for wn, parts in PLAN.items() for p in parts}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))

    def sgn(dd):
        dd = dd[~np.isnan(dd)]
        if not len(dd):
            return "   -   ", False
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.4f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})", dd.mean() > 0 and n_pos > n_neg

    def lm(r, key):
        return np.nanmean([s_[key] for s_ in r])

    print(f"=== {n_seeds} seeds; dormant-part screening (content layer, start settings) ===")
    verdict = {}
    for p in PARTS:
        if p == "base":
            continue
        for wn, parts in PLAN.items():
            if p not in parts:
                continue
            dl = np.array([lm(r[(wn, p)], "never-heard") - lm(r[(wn, "base")], "never-heard") for r in results])
            dt = np.array([lm(r[(wn, p)], TARGET[p]) - lm(r[(wn, "base")], TARGET[p]) for r in results])
            changed = any(not np.allclose(np.nan_to_num([s_["never-heard"] for s_ in r[(wn, p)]]),
                                          np.nan_to_num([s_["never-heard"] for s_ in r[(wn, "base")]])) for r in results)
            (tl, gl), (tt, gt) = sgn(dl), sgn(dt)
            secs = np.mean([r[(wn, p)][-1]["seconds"] / r[(wn, "base")][-1]["seconds"] for r in results])
            print(f"{p:12} {wn:7} life {tl:18} {TARGET[p]:12} {tt:18} time x{secs:.1f}{'' if changed else '  (no change)'}")
            verdict[p] = verdict.get(p, False) or (changed and (gl or gt))
    print("\nverdict: " + "  ".join(f"{p} {'woken-able' if v else 'stays asleep'}" for p, v in verdict.items()))


if __name__ == "__main__":
    main()
