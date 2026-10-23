"""Finding 77's diagnostic: would a timely sleep help at all? Recombination practice awake from birth and put to sleep from
outside at a fixed session (8 = the end of the first half, before the new kind arrives), the method layer as in both+recomb.
If the end loss of both+recomb (-0.016 vs both at 100 seeds) disappears while the early gain stays, the problem is only WHEN
to sleep (a signal the learner lacks); if not, the cost is already paid in the first half. 30 seeds, life32, cached per life.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

COND = "both+recomb-until8"
REFS = ["content", "both", "both+recomb"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, COND) for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    refs = [RL.load_seed(sd, {}) for sd in range(n_seeds)]
    res = [{**{c: refs[sd][c] for c in REFS}, COND: out[(sd, COND)]} for sd in range(n_seeds)]

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; recombination practice asleep from session 9 (told from outside) ===")
    for key in ["never-heard", "once all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in REFS + [COND]:
            print(f"{c:20} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(16)))
    for g, b in [(COND, "both"), (COND, "both+recomb")]:
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in res])
        print(f"{g} vs {b:12} life {sgn(dl)}  end {sgn(de)}")


if __name__ == "__main__":
    main()
