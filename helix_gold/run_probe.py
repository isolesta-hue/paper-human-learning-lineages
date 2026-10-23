"""Finding 99 (a, first check): which eye points where the outside test points? Record-only probe ("+probe"): after the
first rung every branch is scored on the old categories by each eye -- imagination (recombined_score), holding on (old heard,
judge_score), the world-shaped imagination (world_score: the core of the moment kept, the background swapped for words of
the same category and kind) -- and by the outside test on the same categories; per session the rank correlation over the
branches. Behaviour unchanged: the outside test must equal the -routeknobs lives exactly. 10 seeds per world.
Questions, fixed before running:
  P1 glyph world: is the world eye's correlation higher than the imagination eye's (sign over seeds, life mean)?
  P2 text world: is the world eye's correlation at least as high as the imagination eye's (it must not lose the generalisation
     view that finding 50 needed)?
  also: the holding-on eye's correlation in both worlds (finding 97: right in the glyph world, slow-learning bias in text).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

WORLDS = {"glyph": "both@ascii+fix-routeknobs", "text": "both+world+recomb-side-routeknobs"}
EYES = ["recomb", "heard", "world"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c + "+probe") for c in WORLDS.values() for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    for name, c in WORLDS.items():
        res = [out[(sd, c + "+probe")] for sd in range(n_seeds)]
        refs = [RL.one((sd, c)) for sd in range(n_seeds)]
        same = all(abs(a["never-heard"] - b["never-heard"]) < 1e-9 for r, q in zip(res, refs) for a, b in zip(r, q))
        print(f"\n=== {name} world, {n_seeds} seeds; record-only check: {same} ===")
        n = len(res[0])
        for e in EYES + ["spread"]:
            k = f"probe {e}"
            print(f"{k:14} " + " ".join(f"{np.nanmean([r[i].get(k, np.nan) for r in res]):6.2f}" for i in range(n)))
        life = {e: np.array([np.nanmean([r[i].get(f"probe {e}", np.nan) for i in range(n)]) for r in res]) for e in EYES}
        for e in EYES:
            print(f"  {e:7} life mean corr {np.nanmean(life[e]):+.2f}")
        d = life["world"] - life["recomb"]
        n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        print(f"  {'P1' if name == 'glyph' else 'P2'} world - recomb {d.mean():+.2f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.3f})")


if __name__ == "__main__":
    main()
