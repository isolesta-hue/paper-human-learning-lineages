"""Content layer, eleventh experiment: two routes to an answer -- templates and whole-sentence memory (2026-09-25; the user's
proposal: apply "recognise the pattern of the question itself" one level down -- when "the apple from the tree" comes in,
try to solve it with templates AND at the same time look up whether a remembered sentence contains that phrase whole,
and compare).
People have both routes: reading aloud uses a whole-word lookup and a letter-to-sound rule route in parallel (dual-route
model, Coltheart et al. 2001); skills become automatic when retrieving a remembered instance outraces computing the answer
(instance theory, Logan 1988); idioms and set phrases are stored whole (formulaic sequences, Wray 2002). The hippocampus
keeps episodes whole and conjunctive, and animals need it for configural problems (XOR: negative patterning, Rudy &
Sutherland 1995; configural learning, Pearce 1994). AI: a language model interpolated with a nearest-neighbour lookup of
remembered contexts (kNN-LM, Khandelwal et al. 2020).
  chunk        -- ChunkMemory (run_schema3): every heard sentence kept whole; an answer gets evidence from the remembered
                  sentences sharing the most words with the question, if at least two.
  chunk+trust  -- the same, used only while it has helped at least as often as it harmed on the new sentences just heard
                  (leave-one-out: never the sentence the question was made from; finding 61's stakes rule for a route).
World: conj = the schema world + an XOR block per category (data_v3 conj=True: (x, a) -> b, (y, a) -> c, (x, d) -> c,
(y, d) -> b -- each word alone is tied to both answers; only two together tell). rconj = the random world + the same block.
Learners: D-norm / D-norm+chunk (whole-sentence memory without templates) / masked / masked+chunk / masked+chunk+trust.
Seeds: two stages (CLAUDE.md). Key comparisons fixed before looking: masked+chunk vs masked, masked+chunk+trust vs masked
(per world, life).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_schema3 as S3

WORLDS8 = ["conj", "rconj"]
LEARNERS = ["D-norm", "D-norm+chunk", "masked", "masked+chunk", "masked+chunk+trust"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache11")
KEY = [("masked+chunk", "masked"), ("masked+chunk+trust", "masked")]


def life(world_name, learner, seed):
    chunk = ("trust" if learner.endswith("trust") else "always") if "chunk" in learner else None
    if learner.startswith("D-norm"):
        return S3.life(world_name, "D-norm", seed, chunk=chunk)
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
    for key, label in [("curve once", "newest ONE-SHOT pairs, epoch 0"), ("curve conj", "newest XOR items, epoch 0")]:
        print(f"\n--- {label} (sessions 1..{n_sess}) ---")
        for wn in WORLDS8:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:19} " + " ".join(f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f}" for i in range(n_sess)))
    for key in ["never-heard", "conj all", "pairs all", "once all", "unseen all", "chunk on"]:
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
                         ("conj@0 s1-8", lambda r, g: np.mean([st["curve conj"][0] for st in r[g]])),
                         ("once@0 s2-8", lambda r, g: np.mean([st["curve once"][0] for st in r[g][1:]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:28} " + "  ".join(cells))
    for wn in WORLDS8:
        cmp((wn, "D-norm+chunk"), (wn, "D-norm"), f"{wn}: D-norm+chunk vs D-norm")
        cmp((wn, "masked"), (wn, "D-norm"), f"{wn}: masked vs D-norm")
        cmp((wn, "masked+chunk"), (wn, "masked"), f"{wn}: masked+chunk vs masked")
        cmp((wn, "masked+chunk+trust"), (wn, "masked"), f"{wn}: +chunk+trust vs masked")
        cmp((wn, "masked+chunk+trust"), (wn, "masked+chunk"), f"{wn}: trust vs always")
    if n_seeds < 100:
        und = [f"{wn}: {a} vs {b}" for wn in WORLDS8 for a, b in KEY if not decided(results, (wn, a), (wn, b))]
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not und else "undecided -> extend to 100: " + "; ".join(und)))


if __name__ == "__main__":
    main()
