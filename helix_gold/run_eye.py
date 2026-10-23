"""Finding 95's follow-up: does the dashboard see the glyph world's late collapse? "dash old heard" (the stay branch on the
old categories' held-out heard sentences) is recorded next to the first indicators; behaviour unchanged, so
both@ascii+fix+eye must reproduce both@ascii+fix exactly (a record-only check). 10 seeds, cache_long_v2.
Questions, fixed before running:
  E1 does "dash old heard" fall over the life as the outside test does (s13-16 - s2-4, sign over seeds)?
  E2 within a seed, does it follow the outside test (correlation over sessions of the indicator with never-heard, and with
     once all), compared with the old indicator "dash old"?
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cond, ref = "both@ascii+fix+eye", "both@ascii+fix"
    with Pool(workers) as pool:
        res = pool.map(RL.one, [(sd, cond) for sd in range(n_seeds)], chunksize=1)
    refs = [RL.one((sd, ref)) for sd in range(n_seeds)]
    same = all(abs(a["never-heard"] - b["never-heard"]) < 1e-9 for r, q in zip(res, refs) for a, b in zip(r, q))
    print(f"=== {n_seeds} seeds; record-only check (same outside test as {ref}): {same} ===")
    n = len(res[0])
    keys = ["never-heard", "once all", "unseen all", "dash old heard", "dash old", "dash fresh before", "dash fresh after",
            "dash options"]
    for k in keys:
        print(f"{k:18} " + " ".join(f"{np.nanmean([r[i].get(k, np.nan) for r in res]):6.3f}" for i in range(n)))
    for k in ["dash old heard", "dash old", "dash fresh after"]:
        d = np.array([np.nanmean([r[i][k] for i in range(12, 16)]) - np.nanmean([r[i][k] for i in range(1, 4)]) for r in res])
        n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        cn = [np.corrcoef([r[i][k] for i in range(2, n)], [r[i]["never-heard"] for i in range(2, n)])[0, 1] for r in res]
        co = [np.corrcoef([r[i][k] for i in range(2, n)], [r[i]["once all"] for i in range(2, n)])[0, 1] for r in res]
        print(f"E1 {k}: s13-16 - s2-4 {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.3f})   "
              f"E2 corr with never-heard {np.nanmean(cn):+.2f}, with once all {np.nanmean(co):+.2f}")


if __name__ == "__main__":
    main()
