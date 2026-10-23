"""Content layer, fourth and fifth experiments: templates on top of the standardised habit, in evidence units, trusted by
predictions that are refuted only by contradiction (finding 54 follow-up, 2026-09-24).

Finding 54: standardising the habit's scores per question (D-norm) lifted learning across the board, but on top of it
the template channel lost its first-sight effect -- its size (log(#known) x weight x match x (1 - e)) no longer matched
the habit's units -- and prediction-based trust counted "never heard together" as a refutation.
  evidence units   -- a template of weight r that points at b (match m) is combined with the habit as independent
                      evidence (naive Bayes): b's log-probability rises by log(1 + r m (V - 1)), V = number of known
                      words (the candidates) -- no free constant. Discounted by (1 - e_a) for the context word a it
                      comes from (e_a = n_a / (n_a + 1)): where the habit already has experience with a, it already
                      holds that evidence, and counting it again would count it twice. Best over context words.
  contradiction    -- a prediction a -> b is right if the two were heard together (P(b|a) >= STRONG), wrong only if a
                      is strongly tied to some OTHER word (P(x|a) >= STRONG, x != b), otherwise no verdict ("not seen"
                      is not "seen to be otherwise").
Fifth experiment (finding 56), `python run_schema4.py <seeds> <workers> 5`:
  calibrated       -- evidence + temperature scaling of the habit (Guo et al. 2017), fitted per familiarity bin on the
                      learner's record of how its habit did on just-heard, not yet practised sentences (the hypothesis:
                      the habit is overconfident on new contexts)
  independent      -- evidence without the experience discount; instead a template gives no evidence on the pairs
                      it was built from (naive Bayes needs independent evidence -- a template learned from other
                      categories is independent of this sentence; one built from this very pair is not)
  independent+trust -- the same + prediction trust with contradiction-only refutation (to silence templates whose
                      predictions keep failing, e.g. in the random world)
Learners (all on the standardised habit, same seeds, both worlds):
  D-norm           -- the habit alone (finding 54)
  old-units        -- finding 54's norm+explain (template size in the old units) -- to see the effect of the units
  evidence         -- templates weighted by confirmation count n/(n+1), evidence units, explaining away
  evidence+trust   -- templates weighted by prediction trust with contradiction-only refutation, resting of templates
                      that predict badly or never predict (finding 54), evidence units, explaining away
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch

from run_unified_continual import sign_p
from run_schema import WORLDS
import run_schema3 as S3

LEARNERS = ["D-norm", "old-units", "evidence", "evidence+trust"]
LEARNERS5 = ["D-norm", "evidence", "calibrated", "independent", "independent+trust"]


def life(world_name, learner, seed):
    if learner == "D-norm":
        return S3.life(world_name, "D-norm", seed)
    if learner == "old-units":
        return S3.life(world_name, "norm+explain", seed)
    if learner == "evidence":
        return S3.life(world_name, "norm+explain", seed, evidence=True)
    if learner == "evidence+trust":
        return S3.life(world_name, "trust+norm+explain", seed, evidence=True, contradiction=True)
    if learner == "calibrated":
        return S3.life(world_name, "norm+explain", seed, evidence=True, calibrate="bins")
    if learner == "independent":
        return S3.life(world_name, "norm+explain", seed, evidence=True, independent=True)
    return S3.life(world_name, "trust+norm+explain", seed, evidence=True, contradiction=True, independent=True)


def one_seed(args):
    seed, learners = args
    torch.set_num_threads(1)
    return {(wn, ln): life(wn, ln, seed) for wn in WORLDS for ln in learners}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    global LEARNERS
    if len(sys.argv) > 3 and sys.argv[3] == "5":
        LEARNERS = LEARNERS5
    with Pool(workers) as pool:
        results = pool.map(one_seed, [(s, LEARNERS) for s in range(n_seeds)])
    n_sess = len(results[0][(WORLDS[0], LEARNERS[0])])
    print(f"=== {n_seeds} seeds; data_v3; all learners on the standardised habit ===")
    for key, label in [("curve once", "newest ONE-SHOT pairs, epoch 0 / 200"), ("curve unseen", "newest UNSEEN pairs, epoch 0 / 200")]:
        print(f"\n--- {label} (sessions 1..{n_sess}) ---")
        for wn in WORLDS:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:17} " + " | ".join(
                    f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f} {np.nanmean([r[(wn, ln)][i][key][200] for r in results]):.3f}"
                    for i in range(n_sess)))
    for key in ["never-heard", "pairs all", "once all", "unseen all", "used", "junk used"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                print(f"{wn:6} {ln:15} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", lambda r, g: np.mean([st["never-heard"] for st in r[g]])),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("pairs end", lambda r, g: r[g][-1]["pairs all"]),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("unseen end", lambda r, g: r[g][-1]["unseen all"]),
                         ("once@0 s2-8", lambda r, g: np.mean([st["curve once"][0] for st in r[g][1:]])),
                         ("unseen@0 s2-8", lambda r, g: np.mean([st["curve unseen"][0] for st in r[g][1:]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:36} " + "  ".join(cells))
    if LEARNERS is LEARNERS5:
        for wn in WORLDS:
            cmp((wn, "evidence"), (wn, "D-norm"), f"{wn}: evidence vs D-norm")
            cmp((wn, "calibrated"), (wn, "evidence"), f"{wn}: calibrated vs evidence")
            cmp((wn, "independent"), (wn, "evidence"), f"{wn}: independent vs evidence")
            cmp((wn, "independent"), (wn, "D-norm"), f"{wn}: independent vs D-norm")
            cmp((wn, "independent+trust"), (wn, "independent"), f"{wn}: independent+trust vs independent")
            cmp((wn, "independent+trust"), (wn, "D-norm"), f"{wn}: independent+trust vs D-norm")
        print("\n--- fitted habit temperatures (calibrated, schema world; bins by the least familiar context word: <=0.6, <=0.8, <=1.0) ---")
        for i in range(n_sess):
            T = np.array([r[("schema", "calibrated")][i]["temps"] for r in results if r[("schema", "calibrated")][i]["temps"]])
            if len(T):
                print(f"s{i+1} " + " ".join(f"{v:.2f}" for v in T.mean(0)))
        return
    for wn in WORLDS:
        cmp((wn, "old-units"), (wn, "D-norm"), f"{wn}: old-units vs D-norm")
        cmp((wn, "evidence"), (wn, "D-norm"), f"{wn}: evidence vs D-norm")
        cmp((wn, "evidence+trust"), (wn, "D-norm"), f"{wn}: evidence+trust vs D-norm")
        cmp((wn, "evidence+trust"), (wn, "evidence"), f"{wn}: evidence+trust vs evidence")


if __name__ == "__main__":
    main()
