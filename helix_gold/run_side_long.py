"""Finding 85 in the long life (life32, 30 seeds, optimised code, cache_long_v2): does self-made practice through its own side
part keep its early gain AND lose the late dependence (findings 75, 77) with the method layer over two changes?
Conditions: content, content+recomb-side, both (from run_tier2's runs), both+recomb, both+recomb-side, both+recomb-until8.
Key comparisons, fixed before running: K1 both+recomb-side vs both: end and life; K2 both+recomb-side vs both+recomb: end;
K3 completion with the side part: both+recomb-side - content+recomb-side, end and widening (s13-16 - s2-4).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content", "content+recomb-side", "both", "both+recomb", "both+recomb-side", "both+recomb-until8"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS[::-1] for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0]["both"])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    def lm(r, c):
        return np.mean([s["never-heard"] for s in r[c]])

    print(f"=== {n_seeds} seeds; self-made practice through a side part, long life (life32) ===")
    for key in ["never-heard", "once all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:22} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    print("\n--- paired (never-heard) ---")
    for g, b in [("both+recomb-side", "both"), ("both+recomb-side", "both+recomb"), ("both+recomb-side", "both+recomb-until8"),
                 ("content+recomb-side", "content")]:
        dl = np.array([lm(r, g) - lm(r, b) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in res])
        print(f"{g:20} vs {b:20} life {sgn(dl)}  end {sgn(de)}")
    gaps = np.array([[r["both+recomb-side"][i]["never-heard"] - r["content+recomb-side"][i]["never-heard"] for i in range(n)]
                     for r in res])
    print("completion (both+recomb-side - content+recomb-side) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")


if __name__ == "__main__":
    main()
