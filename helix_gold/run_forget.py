"""Finding 108: a late-life knob for the method layer (the user's choice (b), 2026-10-03; goal: keep the completion gap
widening through life -- finding 103 met the criterion at 100 seeds but the gap stopped growing after session 5). The knob:
review the categories being FORGOTTEN (the review pick weighted by how much each old category's own sentences dropped since
the last session; run_long "-forgetknob", off at birth, by the principle "learning-method improvements go to the method layer
as knobs"). Put on the rediscovery setting (the old body + the method layer with the hand-found improvements as knobs), whose
fixed version is the old body with the start settings. 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running:
  F1 (text) both+rediscover-forgetknob vs both+rediscover: end, life, s13-16;
  F2 (text) the completion criterion both+rediscover-forgetknob - content: end, widening (s13-16 - s2-4) and the late widening
     (s13-16 - s6-8) -- compared with the same numbers of both+rediscover;
  F3 (glyph) both@ascii+rediscover-forgetknob vs both@ascii+rediscover; F4 (glyph) its completion vs content@ascii.
  Also: the share of lives with the knob on, per session.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content", "both+rediscover", "both+rediscover-forgetknob",
         "content@ascii", "both@ascii+rediscover", "both@ascii+rediscover-forgetknob"]


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

    print(f"=== {n_seeds} seeds; review what is being forgotten (a late-life knob) ===")
    for key in ["never-heard", "once all", "unseen all", "forget pick", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:36} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))

    def pair(g, b_, name):
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        late = np.array([np.mean([r[g][i]["never-heard"] - r[b_][i]["never-heard"] for i in range(12, 16)]) for r in res])
        print(f"{name} {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}  s13-16 {sgn(late)}")

    def completion(g, b_, name):
        gaps = np.array([[r[g][i]["never-heard"] - r[b_][i]["never-heard"] for i in range(n)] for r in res])
        print(f"{name} completion ({g} - {b_}) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
        print(f"   end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}"
              f" | late widening s13-16 - s6-8 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 5:8].mean(1))}")
    print()
    pair("both+rediscover-forgetknob", "both+rediscover", "F1")
    completion("both+rediscover-forgetknob", "content", "F2")
    completion("both+rediscover", "content", "F2 ref")
    pair("both@ascii+rediscover-forgetknob", "both@ascii+rediscover", "F3")
    completion("both@ascii+rediscover-forgetknob", "content@ascii", "F4")
    completion("both@ascii+rediscover", "content@ascii", "F4 ref")


if __name__ == "__main__":
    main()
