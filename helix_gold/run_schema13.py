"""Content layer, thirteenth experiment: longer phrases and overlapping pieces (finding 63 follow-up, 2026-09-25; both
options proposed by the user).

Finding 63: weighting shared words by rarity (idf) kept whole-sentence memory's gain and removed most of the cost of
chance overlaps -- in this world what matters is WHICH words are shared, not how many. The user's two proposals:
  ladder     -- more conditions filter better: if 2-word matches are not accurate enough, require 3, then 4. Per rung
                (words shared 2 / 3 / 4+), the memory's help and harm on the new sentences just heard (leave-one-out) are
                counted, and a rung that harmed more than it helped is switched off (n-gram back-off, Katz 1987, the other
                way round: prefer the longest match). On idf matching.
  consensus  -- use several 2-word pieces that share a word at once ("the apple from the tree" + "from the tree fell") and
                look for their intersection: every two words of the question are a piece, each piece that some remembered
                sentence contains votes for its candidates, weighted by the rarity of its words; pieces that agree make an
                answer strong, chance overlaps scatter (a product of experts, Hinton 2002; DNA reads assembled by their
                overlaps; people link overlapping memories A-B, B-C to infer A-C, Zeithamova & Preston 2010).
Worlds: conj (finding 62: the real phrase has 2 words) / long (data_v3 long=True: a 3-word phrase (u, v, w) -> t whose
every 2-word part also belongs to another phrase -- (u, v) -> s1, (v, w) -> s2, (u, w) -> s3 -- so any 2-word match is
ambiguous and only 3 words, or the agreement of all three 2-word pieces, tell t).
Learners: masked / chunk-idf (finding 63) / chunk-ladder / chunk-consensus. Expected before running: similar in conj;
ladder and consensus ahead in long.
Seeds: two stages (CLAUDE.md). Key comparisons fixed before looking: ladder vs idf, consensus vs idf (per world, life).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_schema3 as S3

WORLDS8 = ["conj", "long"]
LEARNERS = ["masked", "chunk-idf", "chunk-ladder", "chunk-consensus"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache13")
KEY = [("chunk-ladder", "chunk-idf"), ("chunk-consensus", "chunk-idf")]


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
        for o in ("chunk-ladder", "chunk-consensus"):
            cmp((wn, o), (wn, "chunk-idf"), f"{wn}: {o} vs chunk-idf")
            cmp((wn, o), (wn, "masked"), f"{wn}: {o} vs masked")
    if n_seeds < 100:
        und = [f"{wn}: {a} vs {b}" for wn in WORLDS8 for a, b in KEY if not decided(results, (wn, a), (wn, b))]
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not und else "undecided -> extend to 100: " + "; ".join(und)))


if __name__ == "__main__":
    main()
