"""Content layer, ninth experiment: organizing templates -- options for the pieces MDL threw away (finding 59 follow-up,
2026-09-24; agreed with the user, who asked that merging be tried as ONE of several options: since the templates got a
coordinate system, the ways to organize and choose them have multiplied, many borrowed from AI methods).

Finding 59: MDL bit accounts removed every spurious template, but also nearly every real one -- a real relation is split
into many small templates (memory space bends it; the assimilation rule then makes new ones), and no piece alone pays
for its own description. Spurious templates and pieces look the same one by one (few confirmations); they differ in
their relation to OTHER templates (a piece has close relatives, a spurious template has none) -- the next layer up.
Options (all on the standardised habit, independent evidence, context mask):
  merge       -- consolidation: after each session, two templates become one if that is shorter to describe (one
                 description instead of two, against the extra bits of coding both groups of pairs with the shared
                 direction). The closest pairs are tried first, until nothing merges. All templates with >= 2
                 confirmations are used (no bit filter) -- the effect of merging alone.
  merge+mdl   -- merging, then finding 59's bit filter.
  family+mdl  -- pooling without merging: a template is used if its family (the templates the system's own
                 assimilation rule would call the same -- |cos| >= VIGILANCE, length ratio in [0.5, 2]) has paid back
                 its bits together, paying one description. The pieces stay separate (keeping their local directions --
                 finding 59: pieces may be local straight-line pieces of a bent relation).
Correction found in the check run (before any 30-seed run): finding 59's code named a template's partner word
uniformly among the known words when no template is used (log2 #known bits), but the partner is near a in memory anyway,
so ANY direction ranks it high and every template looked cheap (the check seed merged 7 of 9 templates in the first
session). The no-template code is now the partner's rank by nearness to a alone: a template earns bits only for what its
direction adds to nearness. All bit accounts here use it; the mdl control re-tests finding 59 with it.
Controls: D-norm / masked (finding 57-58) / mdl (finding 59, with the corrected null). Other options noted for later (README): a Chinese
restaurant process for making templates (instead of the fixed vigilance), parent-child hierarchies, a template as a
learned map rather than one direction.
Worlds: schema / random / local. Seeds: two stages (30, then 100 only if a key comparison is undecided; CLAUDE.md).
Key comparisons fixed before looking: merge vs masked, merge+mdl vs masked, family+mdl vs masked (per world, life).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_schema3 as S3

WORLDS8 = ["schema", "random", "local"]
LEARNERS = ["D-norm", "masked", "mdl", "merge", "merge+mdl", "family+mdl"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache9")
KEY = [("merge", "masked"), ("merge+mdl", "masked"), ("family+mdl", "masked")]


def life(world_name, learner, seed):
    if learner == "D-norm":
        return S3.life(world_name, "D-norm", seed)
    kw = dict(evidence=True, independent=True, mask_context=True)
    if "mdl" in learner:
        kw["mdl"] = True
        kw["null"] = "near"
    if learner.startswith("merge"):
        kw["organize"] = "merge"
    elif learner.startswith("family"):
        kw["organize"] = "family"
    return S3.life(world_name, "norm+explain", seed, **kw)


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
    for key, label in [("curve once", "newest ONE-SHOT pairs, epoch 0"), ("curve unseen", "newest UNSEEN pairs, epoch 0")]:
        print(f"\n--- {label} (sessions 1..{n_sess}) ---")
        for wn in WORLDS8:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:10} " + " ".join(f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f}" for i in range(n_sess)))
    for key in ["never-heard", "pairs all", "once all", "unseen all", "used", "junk used"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS8:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                if not all(np.isnan(row)):
                    print(f"{wn:6} {ln:10} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", life_mean),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("pairs end", lambda r, g: r[g][-1]["pairs all"]),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("unseen end", lambda r, g: r[g][-1]["unseen all"]),
                         ("once@0 s2-8", lambda r, g: np.mean([st["curve once"][0] for st in r[g][1:]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:28} " + "  ".join(cells))
    for wn in WORLDS8:
        cmp((wn, "masked"), (wn, "D-norm"), f"{wn}: masked vs D-norm")
        cmp((wn, "mdl"), (wn, "masked"), f"{wn}: mdl vs masked")
        cmp((wn, "merge"), (wn, "masked"), f"{wn}: merge vs masked")
        cmp((wn, "merge+mdl"), (wn, "masked"), f"{wn}: merge+mdl vs masked")
        cmp((wn, "merge+mdl"), (wn, "merge"), f"{wn}: merge+mdl vs merge")
        cmp((wn, "family+mdl"), (wn, "masked"), f"{wn}: family+mdl vs masked")
        cmp((wn, "family+mdl"), (wn, "merge+mdl"), f"{wn}: family+mdl vs merge+mdl")
    if n_seeds < 100:
        und = [f"{wn}: {a} vs {b}" for wn in WORLDS8 for a, b in KEY if not decided(results, (wn, a), (wn, b))]
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not und else "undecided -> extend to 100: " + "; ".join(und)))


if __name__ == "__main__":
    main()
