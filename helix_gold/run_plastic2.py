"""Finding 106: Pioneer's idea per CONNECTION ("+plastic2"). Finding 105: per-word age slowed the positions and verbs (heard in
every session) and with them the NEW links they had to make (glyph once-heard life -0.008, 0:30; text life -0.003, 5:25). Here
only the habit's link table moves at a pair's own pace: step = 1 / (1 + sessions the pair was heard together first-hand) x the
optimizer's step, full step for a link whose change kept its sign three sessions in a row; word vectors move freely. First as
part of the body (the diagnostic: does it help the body at all?). 30 seeds, cache_long_v2.
Key comparisons, fixed before running:
  Q1 (glyph) content@ascii+fix+plastic2 vs content@ascii+fix: end, life, once-heard life and s13-16;
  Q2 (text) content+world+recomb-side+plastic2 vs content+world+recomb-side: end and life (must not lose).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix", "content@ascii+fix+plastic2", "content+world+recomb-side", "content+world+recomb-side+plastic2"]


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

    print(f"=== {n_seeds} seeds; per-connection plasticity ===")
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
        pair("content@ascii+fix+plastic2", "content@ascii+fix", "Q1", k)
    for k in ["never-heard", "once all", "unseen all"]:
        pair("content+world+recomb-side+plastic2", "content+world+recomb-side", "Q2", k)


if __name__ == "__main__":
    main()
