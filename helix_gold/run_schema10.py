"""Content layer, tenth experiment: two options after finding 60 (agreed with the user, 2026-09-24).

Finding 60: every way of proving templates (bit accounts, merging, family pooling) lost in the worlds with regularities,
because proof is the wrong question: whether to USE a template is a matter of expected gain -- P(real) x gain when right
- P(spurious) x harm when wrong -- and in the schema world acting on unproven templates costs almost nothing (spurious
ones hardly fire, finding 58), while in the random world it costs a little.
  stakes    -- the decision criterion depends on the stakes (signal detection theory, Green & Swets 1966; people act on
               hunches when a mistake is cheap and ask for proof when it is costly). Every session, with ALL templates in
               use, on the new sentences just heard (before practice): answers turned wrong -> right (helped) and right ->
               wrong (harmed), summed over the life. While helped >= harmed: act on hunches (all templates, as masked);
               otherwise: demand proof (only templates whose bit account is positive, finding 59-60). No free constant.
  versions  -- the user's proposal: conclude fast, but keep the conclusion open. Every merge finding 60's consolidation
               would make is not made: the merged template is added as a second version next to the two separate ones,
               both keep learning from new pairs, and each new pair is first predicted by both (prequential: bits to name
               its partner with the better separate template + 1 bit, vs with the merged one). The version that has
               predicted better so far is the one used (both start equal; ties: the older, separate versions); a version
               behind by more than a description is dropped. (Keeping several hypotheses at once -- which people, holding
               one at a time, rarely do: win-stay lose-sample, Bonawitz et al. 2014.)
Controls: D-norm / masked / mdl (all accounts with finding 60's corrected no-template code).
Worlds: schema / random / local. Seeds: two stages (CLAUDE.md). Key comparisons fixed before looking: stakes vs masked,
stakes vs mdl, versions vs masked (per world, life).
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
LEARNERS = ["D-norm", "masked", "mdl", "stakes", "versions"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache10")
KEY = [("stakes", "masked"), ("stakes", "mdl"), ("versions", "masked")]


def life(world_name, learner, seed):
    if learner == "D-norm":
        return S3.life(world_name, "D-norm", seed)
    kw = dict(evidence=True, independent=True, mask_context=True)
    kw["null"] = "near"
    if learner == "mdl":
        kw["mdl"] = True
    elif learner == "stakes":
        kw["stakes"] = True
    elif learner == "versions":
        kw["organize"] = "versions"
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
    for key in ["never-heard", "pairs all", "once all", "unseen all", "used", "junk used", "proof", "groups", "merged lead"]:
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
        cmp((wn, "stakes"), (wn, "masked"), f"{wn}: stakes vs masked")
        cmp((wn, "stakes"), (wn, "mdl"), f"{wn}: stakes vs mdl")
        cmp((wn, "versions"), (wn, "masked"), f"{wn}: versions vs masked")
    if n_seeds < 100:
        und = [f"{wn}: {a} vs {b}" for wn in WORLDS8 for a, b in KEY if not decided(results, (wn, a), (wn, b))]
        print("\nstage rule: " + ("all key comparisons decided -- stop" if not und else "undecided -> extend to 100: " + "; ".join(und)))


if __name__ == "__main__":
    main()
