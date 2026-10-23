"""Finding 88: the two parts that passed on their own -- first-hand-only records (finding 84, 100 seeds) and self-made practice
through the side part (finding 86) -- have never been tested together. Before either becomes the default, check the combination.
life32, 30 seeds, cache_long_v2 (both+recomb-side reused).
Key comparisons, fixed before running: K1 both+world+recomb-side vs both+recomb-side, end and life; K2 the completion criterion:
both+world+recomb-side - content+world+recomb-side, end and widening (s13-16 - s2-4).

"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content+world+recomb-side", "both+recomb-side", "both+world+recomb-side"]


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

    print(f"=== {n_seeds} seeds; first-hand-only records + side-part practice together (life32) ===")
    for key in ["never-heard", "once all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:24} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    dl = np.array([np.mean([s["never-heard"] for s in r["both+world+recomb-side"]]) - np.mean([s["never-heard"] for s in r["both+recomb-side"]]) for r in res])
    de = np.array([r["both+world+recomb-side"][-1]["never-heard"] - r["both+recomb-side"][-1]["never-heard"] for r in res])
    print(f"\nK1 world+side vs side: life {sgn(dl)}  end {sgn(de)}")
    gaps = np.array([[r["both+world+recomb-side"][i]["never-heard"] - r["content+world+recomb-side"][i]["never-heard"] for i in range(n)] for r in res])
    print("K2 completion (both+world+side - content+world+side) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")


if __name__ == "__main__":
    main()
