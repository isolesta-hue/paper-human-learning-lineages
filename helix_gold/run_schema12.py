"""Content layer, twelfth experiment: when is a remembered match a real phrase? Three options (finding 62 follow-up,
2026-09-25; agreed with the user).

Finding 62: whole-sentence memory solved the XOR block the pairwise habit cannot (0.45 -> 0.66), but took chance overlaps
of filler words for remembered phrases and cost the once-heard pairs (-0.099); a route-level trust rule could not
separate real phrases from coincidences -- that is a question for each match, not for the route.
  idf      -- a shared rare word is strong evidence, a shared common word weak (inverse document frequency, Sparck Jones
              1972): the best-matching sentences are those whose shared words are rarest (still at least two shared).
  chance   -- is the match more than chance? For the shared words, observed O = remembered sentences containing all of
              them, expected E = N x product (df / N) if words met independently; a match counts by 1 - E / O
              (a set phrase O >> E; fillers that happen to meet O ~ E). People check where a feeling of familiarity comes
              from (source monitoring, Johnson et al. 1993) -- common words make false recognitions (Roediger & McDermott
              1995).
  arbiter  -- the user's "compare the two routes": the memory's help and harm on the new sentences just heard (leave-one-
              out) are counted per SHAPE of the match (words shared 2 / 3 / 4+ x equally good sentences 1 / 2-3 / 4+), and
              the memory is used only for the shapes where it helped at least as often as it harmed (finding 61's stakes
              rule, per shape instead of per route -- the lowest layer of recognising the pattern of a question).
No free constant in any (the arbiter's shape bins are representative values).
Worlds: conj / rconj (finding 62). Learners: masked / masked+chunk (finding 62) / chunk-idf / chunk-chance / chunk-arbiter.
Seeds: two stages (CLAUDE.md). Key comparisons fixed before looking: each option vs masked+chunk and vs masked (life).
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
LEARNERS = ["masked", "masked+chunk", "chunk-idf", "chunk-chance", "chunk-arbiter"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache12")
KEY = [(o, b) for o in ("chunk-idf", "chunk-chance", "chunk-arbiter") for b in ("masked", "masked+chunk")]


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
        print(f"{label:36} " + "  ".join(cells))
    for wn in WORLDS8:
        cmp((wn, "masked+chunk"), (wn, "masked"), f"{wn}: masked+chunk vs masked")
        for o in ("chunk-idf", "chunk-chance", "chunk-arbiter"):
            cmp((wn, o), (wn, "masked+chunk"), f"{wn}: {o} vs masked+chunk")
            cmp((wn, o), (wn, "masked"), f"{wn}: {o} vs masked")
    if n_seeds < 100:
        und = [f"{wn}: {a} vs {b}" for wn in WORLDS8 for a, b in KEY if not decided(results, (wn, a), (wn, b))]
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not und else "undecided -> extend to 100: " + "; ".join(und)))


if __name__ == "__main__":
    main()
