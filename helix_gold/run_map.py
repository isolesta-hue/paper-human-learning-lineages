"""The reaction-norm map (2026-09-29; the user's reading of finding 74: a dormant part is not useless, it is a trait that did not
fit THIS environment -- so measure each part's response across environments, like a reaction norm in biology, and against its
background: the parts were built on older bodies, epistasis).

Body: Voyager's content layer with the start settings (no method layer: cheap), one dormant part switched on from birth
(Learner3.wake), against the same learner without it. Optimised code (cache_map is post-optimisation only).
Parts: the chunk-memory modes arbiter / ladder / consensus / idf-arbiter (new: the arbiter's per-shape judgement on idf matching,
finding 74's note), compose, setter-hard, recomb (setter-random: self-made recombination practice), versions, fmt-practice.
Worlds (data_v3): schema, conj (XOR), parity (3-word pattern), compose (two relations chained), random (no regularities),
shift (rules change in the second half), local (each place has its own rules), back (rules change and come back).
Measured per (part, world): life mean on never-heard combinations and the kinds (once-heard pairs, heard pairs, XOR, 3-word,
composed, unseen), vs base. 30 seeds first (two-stage rule; the map is descriptive -- no single key comparison, so it stays at
30 unless one cell decides a question later).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

import data_v3 as W
from run_schema import PER_SESSION
from run_unified_continual import sign_p
from integrated import Learner3, START, build_world, measure

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_map")
PARTS = {"base": {}, "arbiter": {"chunk_mode": "arbiter"}, "ladder": {"chunk_mode": "ladder"},
         "consensus": {"chunk_mode": "consensus"}, "idf-arbiter": {"chunk_mode": "idf-arbiter"},
         "idf-route": {"chunk_mode": "idf-route"}, "idf-trust": {"chunk_mode": "idf-trust"},
         "world-only": {"hebb_source": "world"}, "thompson": {"thompson": True}, "fmt-hard": {"fmt_hard": True},
         "agree2": {"agree2": True}, "disagree": {"disagree": True}, "verify-cont": {"verify_cont": True},
         "recomb-side": {"setter": "random", "self_side": True},
         "recomb-cap": {"setter": "random", "self_side": "capacity"}, "compose": {"compose": True},
         "setter-hard": {"setter": "hard"}, "recomb": {"setter": "random"}, "versions": {"versions": True},
         "fmt-practice": {"fmt_practice": True}}
WORLDS = ["schema", "conj", "parity", "compose", "random", "shift", "local", "back"]
KINDS = {"fmt options": "fmt options", "fmt candidate": "fmt candidate", "once": "once all", "pairs": "pairs all", "xor": "conj all", "3-word": "long all", "compose": "compose all",
         "unseen": "unseen all"}


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
    for new_cats in sessions:
        old = list(seen)
        seen += new_cats
        curves = me.session([x for c in new_cats for x in sents[c]], {c: sents[c] for c in old}, s, rng)
        steps.append(measure(me, tests, seen, new_cats, curves, seed))
    return steps


def one(task):
    seed, wn, part = task
    path = os.path.join(CACHE, f"seed{seed}_{wn}_{part}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    out = life(wn, part, seed)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    tasks = [(sd, wn, p) for sd in range(n_seeds) for wn in WORLDS for p in PARTS]
    with Pool(workers) as pool:
        res = dict(zip(tasks, pool.map(one, tasks, chunksize=1)))

    def lm(sd, wn, p, key):
        v = [s_.get(key, np.nan) for s_ in res[(sd, wn, p)]]
        return np.nanmean(v) if not np.all(np.isnan(v)) else np.nan

    def cell(wn, p, key):
        d = np.array([lm(sd, wn, p, key) - lm(sd, wn, "base", key) for sd in range(n_seeds)])
        d = d[~np.isnan(d)]
        if not len(d):
            return "     .     "
        n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        mark = "*" if sign_p(n_pos, n_neg) < 0.01 else " "
        return f"{d.mean():+.3f}{mark}{n_pos:>2}/{n_neg:<2}"

    print(f"=== {n_seeds} seeds; reaction-norm map (content layer, start settings; life mean vs base; * p<0.01) ===")
    print(f"\n--- never-heard combinations ---\n{'part':13}" + "".join(f"{wn:>12}" for wn in WORLDS))
    for p in list(PARTS)[1:]:
        print(f"{p:13}" + "".join(f"{cell(wn, p, 'never-heard'):>12}" for wn in WORLDS))
    for kind, key in KINDS.items():
        print(f"\n--- {kind} ---\n{'part':13}" + "".join(f"{wn:>12}" for wn in WORLDS))
        for p in list(PARTS)[1:]:
            print(f"{p:13}" + "".join(f"{cell(wn, p, key):>12}" for wn in WORLDS))


if __name__ == "__main__":
    main()
