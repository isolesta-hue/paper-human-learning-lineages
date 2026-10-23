"""Content layer, fourteenth experiment: the ladder and piece agreement on a TRUE 3-word pattern (finding 64 follow-up,
2026-09-25; the user: run it in the background, evaluate, and keep it for later while the research moves on).

Finding 64: finding 64's 3-word world was solvable by adding pairwise ties (t tied to three words, s1 to two), so it could
not show what a 3-word match is for; and piece agreement added votes, so the many chance pieces from filler words
drowned the few real ones (the user had asked for an intersection).
  parity world  -- data_v3 parity=True: one word from each of three slots, answer b for even parity, c for odd: every word
                   and every PAIR is tied to b and c equally (a 3-word XOR) -- only all three tell.
  consensus2    -- piece agreement where a piece counts by how decided it is: 1 - entropy / max entropy of the words it points
                   at (a real phrase points at one answer, a chance overlap scatters) -- an information-weighted vote, no
                   free constant.
Worlds: conj (2-word XOR) / parity (3-word XOR). Learners: masked / chunk-idf / chunk-ladder / chunk-consensus2.
Seeds: two stages (CLAUDE.md). Key comparisons fixed before looking: ladder vs idf, consensus2 vs idf (per world, life).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_schema3 as S3

WORLDS8 = ["conj", "parity"]
LEARNERS = ["masked", "chunk-idf", "chunk-ladder", "chunk-consensus2"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache14")
KEY = [("chunk-ladder", "chunk-idf"), ("chunk-consensus2", "chunk-idf")]


def life(world_name, learner, seed):
    chunk = None
    if learner == "masked+chunk":
        chunk = "always"
    elif learner.startswith("chunk-"):
        chunk = learner[len("chunk-"):]
    return S3.life(world_name, "norm+explain", seed, evidence=True, independent=True, mask_context=True, chunk=chunk)


def one_seed(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    out = {(wn, ln): life(wn, ln, seed) for wn in WORLDS8 for ln in LEARNERS}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def life_mean(r, g):
    return np.mean([st["never-heard"] for st in r[g]])


def decided(results, ga, gb):
    dd = np.array([life_mean(r, ga) - life_mean(r, gb) for r in results])
    n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
    half = 2.756 * dd.std(ddof=1) / np.sqrt(len(dd))          # 99% t interval, 29 degrees of freedom
    return sign_p(n_pos, n_neg) < 0.005 or (dd.mean() - half > -0.002 and dd.mean() + half < 0.002)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_sess = len(results[0][(WORLDS8[0], LEARNERS[0])])
    print(f"=== {n_seeds} seeds; data_v3 schema / random / local ===")
    for key, label in [("curve once", "newest ONE-SHOT pairs, epoch 0"), ("curve conj", "newest XOR / 3-word phrase items, epoch 0")]:
        print(f"\n--- {label} (sessions 1..{n_sess}) ---")
        for wn in WORLDS8:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:19} " + " ".join(f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f}" for i in range(n_sess)))
    for key in ["never-heard", "conj all", "long all", "pairs all", "once all", "unseen all"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS8:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                if not all(np.isnan(row)):
                    print(f"{wn:6} {ln:19} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", life_mean),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("pairs end", lambda r, g: r[g][-1]["pairs all"]),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("conj end", lambda r, g: r[g][-1]["conj all"]),
                         ("long end", lambda r, g: r[g][-1]["long all"]),
                         ("conj@0 s1-8", lambda r, g: np.mean([st["curve conj"][0] for st in r[g]])),
                         ("once@0 s2-8", lambda r, g: np.mean([st["curve once"][0] for st in r[g][1:]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:36} " + "  ".join(cells))
    for wn in WORLDS8:
        cmp((wn, "chunk-idf"), (wn, "masked"), f"{wn}: chunk-idf vs masked")
        for o in ("chunk-ladder", "chunk-consensus2"):
            cmp((wn, o), (wn, "chunk-idf"), f"{wn}: {o} vs chunk-idf")
            cmp((wn, o), (wn, "masked"), f"{wn}: {o} vs masked")
    if n_seeds < 100:
        und = [f"{wn}: {a} vs {b}" for wn in WORLDS8 for a, b in KEY if not decided(results, (wn, a), (wn, b))]
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not und else "undecided -> extend to 100: " + "; ".join(und)))


if __name__ == "__main__":
    main()
