"""Finding 86 (a): with self-made practice through a side part the fixed-settings learner reached the same end as the
self-correcting one -- the method layer had nothing left to correct. Give the method layer the side part's own knobs (its weight
when answering: 0 / 0.5 / 1 / 2, and how fast it forgets: 1 / 0.5 per session; representative values) and see whether the
self-correcting learner pulls ahead again. life32, 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running: K1 both+recomb-sideknobs vs both+recomb-side, end and life; K2 the completion criterion:
both+recomb-sideknobs - content+recomb-side, end and widening (s13-16 - s2-4).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content+recomb-side", "both+recomb-side", "both+recomb-sideknobs"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS[::-1] for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0][CONDS[0]])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; the method layer with the side part's knobs (life32) ===")
    for key in ["never-heard", "once all", "lr", "review", "side w", "side decay"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:24} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    dl = np.array([np.mean([s["never-heard"] for s in r["both+recomb-sideknobs"]]) - np.mean([s["never-heard"] for s in r["both+recomb-side"]]) for r in res])
    de = np.array([r["both+recomb-sideknobs"][-1]["never-heard"] - r["both+recomb-side"][-1]["never-heard"] for r in res])
    print(f"\nK1 sideknobs vs side: life {sgn(dl)}  end {sgn(de)}")
    gaps = np.array([[r["both+recomb-sideknobs"][i]["never-heard"] - r["content+recomb-side"][i]["never-heard"] for i in range(n)] for r in res])
    print("K2 completion (sideknobs - content+side) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")


if __name__ == "__main__":
    main()
