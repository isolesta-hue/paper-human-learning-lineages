"""Content layer, eighth experiment: keep a template only if it pays for itself -- minimum description length
(finding 58 follow-up, 2026-09-24; agreed with the user).

Finding 58: choosing templates by context changed nothing, because a template only fires where a pair fits its
direction; the remaining cost is spurious templates -- about 70% of the templates in use match no real relation (and
the random world's small loss is theirs). Spurious templates are not a selection problem but a "should this exist"
problem: a regularity is real if describing the data with it is shorter than without it (MDL, Rissanen 1978; the
two-part code: the model's own length + the data's length given the model; an overfitted "law" costs more than it saves).
  mdl     -- every template keeps a bit account. Making it costs its description: the founding pair, 2 log2(#known) bits.
             Every later pair it takes in is first coded with the template as it was BEFORE that pair (prequential, so a
             template never gets credit for fitting what it was fitted to): naming the partner b of a costs log2(#known)
             bits without a template; with it: which template (adaptive frequency code, log2((N + 1) / n_k)) + sign (1 bit)
             + b's rank among the known words by distance to m_a +/- t (Elias gamma code, 2 floor(log2 rank) + 1). The
             template is used only while its account is positive. No free constant.
  count4  -- control: a template is used from 4 confirmations instead of 2 (about what a good template needs to pay off),
             to tell "the coding criterion" from "just asking for more confirmations" (a representative value, not tuned).
Worlds: schema / random / local (finding 58: 13 directions, fewer confirmations each -- the risk that MDL also drops
real templates). Learners (all on the standardised habit, independent evidence, context mask): D-norm / masked / mdl /
count4.
Seeds (the user's question, 2026-09-24 -- are 100 seeds needed?): results are cached per seed; stage 1 runs 30 seeds.
Decision rule fixed before looking: for the key comparisons (mdl vs masked, mdl vs count4, per world, life), a comparison
is decided at 30 seeds if the sign test gives p < 0.005 (stricter than 0.01, because looking twice raises the chance of a
false alarm) or the 99% interval of the mean difference lies within +/-0.002 (no meaningful effect); otherwise all
three worlds are extended to 100 seeds (the 30 are reused).
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
LEARNERS = ["D-norm", "masked", "mdl", "count4"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache8")
KEY = [("mdl", "masked"), ("mdl", "count4")]


def life(world_name, learner, seed):
    if learner == "D-norm":
        return S3.life(world_name, "D-norm", seed)
    kw = dict(evidence=True, independent=True, mask_context=True)
    if learner == "mdl":
        kw["mdl"] = True
    elif learner == "count4":
        kw["min_count"] = 4
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
                print(f"{wn:6} {ln:8} " + " ".join(f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f}" for i in range(n_sess)))
    for key in ["never-heard", "pairs all", "once all", "unseen all", "used", "junk used"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS8:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                if not all(np.isnan(row)):
                    print(f"{wn:6} {ln:8} " + " ".join(f"{v:6.3f}" for v in row))

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
        cmp((wn, "count4"), (wn, "masked"), f"{wn}: count4 vs masked")
        cmp((wn, "mdl"), (wn, "count4"), f"{wn}: mdl vs count4")
        cmp((wn, "mdl"), (wn, "D-norm"), f"{wn}: mdl vs D-norm")
    if n_seeds < 100:
        und = [f"{wn}: {a} vs {b}" for wn in WORLDS8 for a, b in KEY if not decided(results, (wn, a), (wn, b))]
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not und else "undecided -> extend to 100: " + "; ".join(und)))


if __name__ == "__main__":
    main()
