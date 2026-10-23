"""Integration, next step 1: why does the gap stop widening in the XOR world? (finding 69 follow-up, 2026-09-25; the
user's order: 1 find the cause, 2 split + budget, 3 a long life with the deferred options and the integration variants).

Finding 69: in the schema world + XOR block the gap between the self-correcting and the fixed-settings learner (all parts
in both) widened to session 5 (+0.020 -> +0.039) and then flattened. Two candidate causes:
  (a) whole-sentence memory -- it harms the once-heard pairs (0.398 with both layers vs 0.461 with the method layer alone)
      and may cancel the method layer's gains: take it out of BOTH arms (content without memory, fixed vs self-correcting)
  (b) the knobs' ceiling -- the method layer had raised the learning rate to 0.025 of a maximum of 0.03: widen the
      learning-rate knob to {0.003, 0.01, 0.03, 0.1}
Conditions (world conj; gap = self-correcting - fixed, same parts):
  content / both                 -- finding 69 (reference; from its cache)
  content-nochunk / both-nochunk -- (a)
  both-widelr                    -- (b), gap against content
The cause is the one whose change makes the gap keep widening (late s6-8 minus early s2-4). Seeds: 30 first (cached),
100 if undecided.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_integrated as RI

CONDITIONS = ["content", "both", "content-nochunk", "both-nochunk", "both-widelr"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_int2")
WIDE = dict(RI.KNOBS, lr=[0.003, 0.01, 0.03, 0.1])


def life(cond, seed):
    if cond in ("content", "both"):                  # identical to finding 69's runs: reuse its cache when present
        path = os.path.join(RI.CACHE, f"seed{seed}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)[("conj", cond)]
        return RI.life("conj", cond, seed)
    if cond == "both-widelr":
        return RI.life("conj", "both", seed, knobs=WIDE)
    return RI.life("conj", cond.replace("-nochunk", ""), seed, chunk=False)


def one_seed(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    out = {c: life(c, seed) for c in CONDITIONS}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n = len(results[0]["content"])

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; conj world; why the gap flattens ===")
    for key in ["never-heard", "newest", "older", "conj all", "once all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            print(f"{c:16} " + " ".join(f"{m([r[c][i][key] for r in results]):>6}" for i in range(n)))
    print("\n--- gaps (self-correcting - fixed, same parts) ---")
    for g, b in [("both", "content"), ("both-nochunk", "content-nochunk"), ("both-widelr", "content")]:
        gaps = np.array([[r[g][i]["never-heard"] - r[b][i]["never-heard"] for i in range(n)] for r in results])
        print(f"{g:14} - {b:16} by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
        print(f"{'':33} end {sgn(gaps[:, -1])}   late - early {sgn(gaps[:, 5:8].mean(1) - gaps[:, 1:4].mean(1))}")
    for g, b in [("content-nochunk", "content"), ("both-nochunk", "both"), ("both-widelr", "both")]:
        d_life = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b]]) for r in results])
        d_end = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in results])
        print(f"{g:16} vs {b:8} life {sgn(d_life)}  end {sgn(d_end)}")


if __name__ == "__main__":
    main()
