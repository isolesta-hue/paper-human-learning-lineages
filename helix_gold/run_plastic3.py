"""Finding 107: per-connection plasticity with only the NEW material's share slowed ("+plastic3"; the user's choice (a),
2026-10-03). Finding 106: slowing a link also slowed its consolidation by review -- once-heard items got worse in every
variant. Here each link-table step is split by where its gradient came from: the share from this session's new sentences is
slowed (1 / (1 + the pair's age), full for a link that kept its sign three sessions in a row), the share from review (and
self-made practice) moves at full step -- new material may not shake old links, practising the same thing again may.
About twice the backward compute. As part of the body first (the diagnostic). 30 seeds, cache_long_v2.
Key comparisons, fixed before running:
  S1 (glyph) content@ascii+fix+plastic3 vs content@ascii+fix: end, life, once-heard life and s13-16;
  S2 (text) content+world+recomb-side+plastic3 vs content+world+recomb-side: end and life (must not lose).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix", "content@ascii+fix+plastic3", "content+world+recomb-side", "content+world+recomb-side+plastic3"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0][CONDS[0]])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.4f})"

    print(f"=== {n_seeds} seeds; per-connection plasticity, new material only ===")
    for key in ["never-heard", "once all", "unseen all", "pairs all"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:36} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))

    def pair(g, b_, name, key="never-heard"):
        dl = np.array([np.nanmean([s[key] for s in r[g]]) - np.nanmean([s[key] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1][key] - r[b_][-1][key] for r in res])
        late = np.array([np.mean([r[g][i][key] - r[b_][i][key] for i in range(12, 16)]) for r in res])
        print(f"{name} [{key}] {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}  s13-16 {sgn(late)}")
    print()
    for k in ["never-heard", "once all", "unseen all"]:
        pair("content@ascii+fix+plastic3", "content@ascii+fix", "S1", k)
    for k in ["never-heard", "once all", "unseen all"]:
        pair("content+world+recomb-side+plastic3", "content+world+recomb-side", "S2", k)


if __name__ == "__main__":
    main()
