"""Content layer, sixth experiment: a reality filter for the whole template channel, and how fast its trust moves
(finding 56 follow-up, 2026-09-24; design agreed with the user).

Finding 56: independent evidence brought first-sight understanding back, but in the random world (no regularity in
reality) the templates still cost a little, and per-template prediction trust stopped only part of it -- the user's
point: the logic lacks filtering by reality (a theorist who asserts laws without checking them against the world).
  context mask     -- template evidence is never given to words already in the question's sentence (the hidden word
                      is not one of them); first check: without this, the templates pushed the sentence's own partner
                      word on questions whose answer was a filler, and the filter counted that as harm
  reality filter   -- at the start of every session, before practice, on the new sentences just heard: how many
                      answers did the template evidence turn from wrong to right (helped) and from right to wrong
                      (harmed)? channel reliability = (helped + 1) / (helped + harmed + 2), multiplying every template's
                      weight. Judging and re-judging the judge happen together (the user: in a new situation the filter
                      is used and its own trust is updated at once; the brain's precision weighting).
  update speed     -- fixed: all past sessions count equally / adaptive: the record is discounted by this session's
                      surprise |observed - expected reliability| (Pearce & Hall 1980: learn faster when surprised;
                      people adapt their learning rate to how changeable the world is, Behrens et al. 2007)
Worlds: schema / random / shift (the schema world whose relation directions change in the second half -- the
filter's speed only matters when the world changes, so both questions are answered by one run).
Learners (all on the standardised habit with independent evidence, finding 56): D-norm / independent / masked /
masked+filter / masked+filter+adaptive.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
import run_schema3 as S3

WORLDS6 = ["schema", "random", "shift"]
LEARNERS = ["D-norm", "independent", "masked", "masked+filter", "masked+filter+adaptive"]


def life(world_name, learner, seed):
    if learner == "D-norm":
        return S3.life(world_name, "D-norm", seed)
    kw = dict(evidence=True, independent=True)
    if learner != "independent":
        kw["mask_context"] = True
    if "filter" in learner:
        kw["reality"] = "adaptive" if learner.endswith("adaptive") else "fixed"
    return S3.life(world_name, "norm+explain", seed, **kw)


def one_seed(seed):
    torch.set_num_threads(1)
    return {(wn, ln): life(wn, ln, seed) for wn in WORLDS6 for ln in LEARNERS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_sess = len(results[0][(WORLDS6[0], LEARNERS[0])])
    print(f"=== {n_seeds} seeds; data_v3 schema / random / shift (second half: new directions) ===")
    for key, label in [("curve once", "newest ONE-SHOT pairs, epoch 0"), ("curve unseen", "newest UNSEEN pairs, epoch 0")]:
        print(f"\n--- {label} (sessions 1..{n_sess}) ---")
        for wn in WORLDS6:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:24} " + " ".join(f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f}" for i in range(n_sess)))
    for key in ["never-heard", "pairs all", "once all", "unseen all", "rel"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS6:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                print(f"{wn:6} {ln:24} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", lambda r, g: np.mean([st["never-heard"] for st in r[g]])),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("2nd half", lambda r, g: np.mean([st["never-heard"] for st in r[g][4:]])),
                         ("pairs end", lambda r, g: r[g][-1]["pairs all"]),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("once@0 s2-8", lambda r, g: np.mean([st["curve once"][0] for st in r[g][1:]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:48} " + "  ".join(cells))
    for wn in WORLDS6:
        cmp((wn, "independent"), (wn, "D-norm"), f"{wn}: independent vs D-norm")
        cmp((wn, "masked"), (wn, "independent"), f"{wn}: masked vs independent")
        cmp((wn, "masked+filter"), (wn, "masked"), f"{wn}: masked+filter vs masked")
        cmp((wn, "masked+filter+adaptive"), (wn, "masked+filter"), f"{wn}: adaptive vs fixed speed")
        cmp((wn, "masked+filter+adaptive"), (wn, "D-norm"), f"{wn}: masked+filter+adaptive vs D-norm")


if __name__ == "__main__":
    main()
