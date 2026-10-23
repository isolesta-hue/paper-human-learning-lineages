"""Finding 89: two kinds of trust in the method layer's judge (the user's proposal, 2026-09-30, after finding 87). The judge's
first-hand part (new categories' held-out sentences) is the top judge; its imagination part (old categories' self-made
recombinations) counts only as much as it has earned by agreeing with first-hand feedback (run_long "+trust2").
life32, 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running: K1 both+recomb-sideknobs+trust2 vs both+recomb-sideknobs (the fix), end and life;
K2 both+recomb-sideknobs+trust2 vs both+recomb-side (does the method layer's knob now help), end and life; K3 the completion
criterion both+recomb-sideknobs+trust2 - content+recomb-side, end and widening; K4 both+world+recomb-side+trust2 vs
both+world+recomb-side (the default candidate with the new judge), end and life.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content+recomb-side", "both+recomb-side", "both+recomb-sideknobs", "both+recomb-sideknobs+trust2",
         "both+world+recomb-side", "both+world+recomb-side+trust2"]


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

    print(f"=== {n_seeds} seeds; two kinds of trust in the judge (life32) ===")
    for key in ["never-heard", "once all", "lr", "review", "side w", "side decay", "trust imagination"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:24} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    def pair(g, b_, name):
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        print(f"{name} {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}")
    print()
    pair("both+recomb-sideknobs+trust2", "both+recomb-sideknobs", "K1")
    pair("both+recomb-sideknobs+trust2", "both+recomb-side", "K2")
    gaps = np.array([[r["both+recomb-sideknobs+trust2"][i]["never-heard"] - r["content+recomb-side"][i]["never-heard"]
                      for i in range(n)] for r in res])
    print("K3 completion (sideknobs+trust2 - content+side) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    pair("both+world+recomb-side+trust2", "both+world+recomb-side", "K4")


if __name__ == "__main__":
    main()
