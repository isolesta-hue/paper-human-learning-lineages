"""Finding 103's 100-seed extension (two-stage seed rule: the completion criterion in the text world was undecided at 30 seeds,
end +0.116 (29:1) but widening +0.019 (22:8, p=0.016)). Fixed version = the old body with the start settings (content); the
self-improving version = the same body + the method layer with the hand-found improvements as knobs (both+rediscover).
Key comparison, fixed before running: the completion criterion both+rediscover - content, end and widening (s13-16 - s2-4),
100 seeds; also R1 (both+rediscover vs both) and R2 (vs content+world+recomb-side) at 100 seeds.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content", "both", "content+world+recomb-side", "both+rediscover"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for sd in range(n_seeds) for c in CONDS]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0][CONDS[0]])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.5f})"

    print(f"=== {n_seeds} seeds; the rediscovery test, text world ===")
    for c in CONDS:
        print(f"{c:28} " + " ".join(f"{np.mean([r[c][i]['never-heard'] for r in res]):6.3f}" for i in range(n)))
    gaps = np.array([[r["both+rediscover"][i]["never-heard"] - r["content"][i]["never-heard"] for i in range(n)] for r in res])
    print("completion (both+rediscover - content) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"   end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    for b_, name in [("both", "R1"), ("content+world+recomb-side", "R2")]:
        de = np.array([r["both+rediscover"][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        print(f"{name} both+rediscover vs {b_}: end {sgn(de)}")


if __name__ == "__main__":
    main()
