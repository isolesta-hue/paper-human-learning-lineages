"""The transformer control (2026-09-26; the user: build from an existing structure and fit it to the goal -- how would a
transformer with attention do, keeping the fixed centre and the fixed original memory?). integrated_tf.TLearner in place
of the integrated learner's content layer; the method layer unchanged on top.
Conditions: tf (start settings) / tf+method (the method layer on the transformer); references from finding 69's cache:
content (all our content parts, start settings) / both (+ method layer). Worlds: conj, schema.
Check before running (one seed, conj): tf end 0.15, tf+method 0.21 (ours 0.59-0.63); with the transformers' usual
learning rate 0.001 instead of our start 0.01: 0.13 -- so the start settings are kept (the method layer may move them).
Measured: the outside test, the kinds (XOR, once-heard, pairs), and whether the method layer's gap (tf+method - tf)
widens like ours -- is self-correction a property of our logic or of the method layer whatever the structure?
Seeds: 30 (cached).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_integrated as RI
from integrated_tf import TLearner

WORLDS = ["conj", "schema"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_tf")


def one_seed(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    out = {}
    ref = os.path.join(RI.CACHE, f"seed{seed}.pkl")
    refd = pickle.load(open(ref, "rb")) if os.path.exists(ref) else {}
    for wn in WORLDS:
        out[(wn, "tf")] = RI.life(wn, "none", seed, make=TLearner)
        out[(wn, "tf+method")] = RI.life(wn, "method", seed, make=TLearner)
        for c in ("content", "both"):
            out[(wn, c)] = refd[(wn, c)] if (wn, c) in refd else RI.life(wn, c, seed)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n = len(results[0][("conj", "tf")])
    conds = ["tf", "tf+method", "content", "both"]

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; transformer control ===")
    for key in ["never-heard", "newest", "older", "conj all", "once all", "pairs all", "unseen all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for wn in WORLDS:
            for c in conds:
                print(f"{wn:6} {c:10} " + " ".join(f"{m([r[(wn, c)][i].get(key, np.nan) for r in results]):>6}" for i in range(n)))
    print("\n--- paired ---")
    for wn in WORLDS:
        for g, b in [("tf+method", "tf"), ("content", "tf"), ("both", "tf+method")]:
            dl = np.array([np.mean([s["never-heard"] for s in r[(wn, g)]]) - np.mean([s["never-heard"] for s in r[(wn, b)]]) for r in results])
            de = np.array([r[(wn, g)][-1]["never-heard"] - r[(wn, b)][-1]["never-heard"] for r in results])
            print(f"{wn}: {g:10} vs {b:10} life {sgn(dl)}  end {sgn(de)}")
        gaps = np.array([[r[(wn, "tf+method")][i]["never-heard"] - r[(wn, "tf")][i]["never-heard"] for i in range(n)] for r in results])
        print(f"{wn}: method gap on the transformer by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0))
              + f" | end {sgn(gaps[:, -1])} | late - early {sgn(gaps[:, 5:8].mean(1) - gaps[:, 1:4].mean(1))}")


if __name__ == "__main__":
    main()
