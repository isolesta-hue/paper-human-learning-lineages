"""Finding 102: the attention body (finding 101) gives the method layer a lever, but its judge looked at old categories with
the recombined eye, blind there (+0.06; the world-shaped eye +0.33, attend_probe_10.out) -- the method layer again raised
the whole-sentence weight (1.53) and lowered the templates' (0.80), finding 95's wrong direction. Here the judge's old-category
eye is world_score ("+worldeye"), the same rule in every world. 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running:
  W1 (glyph) both@ascii+fix-routeknobs+attend+worldeye vs both@ascii+fix-routeknobs+attend, end and life;
  W2 (glyph, completion) both@ascii+fix-routeknobs+attend+worldeye - content@ascii+fix+attend, end and widening;
  W3 (text) both+world+recomb-side-routeknobs+attend+worldeye vs both+world+recomb-side-routeknobs+attend (must not lose);
  W4 (text, completion) both+world+recomb-side-routeknobs+attend+worldeye - content+world+recomb-side+attend.
Glyph lives first (the text lives with attention take ~40 minutes each).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix+attend", "both@ascii+fix-routeknobs+attend", "both@ascii+fix-routeknobs+attend+worldeye",
         "content+world+recomb-side+attend", "both+world+recomb-side-routeknobs+attend",
         "both+world+recomb-side-routeknobs+attend+worldeye"]


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

    print(f"=== {n_seeds} seeds; the world-shaped eye in the judge, attention body ===")
    for key in ["never-heard", "once all", "unseen all", "pattern w", "chunk w", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:50} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))

    def pair(g, b_, name):
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        print(f"{name} {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}")

    def completion(g, b_, name):
        gaps = np.array([[r[g][i]["never-heard"] - r[b_][i]["never-heard"] for i in range(n)] for r in res])
        print(f"{name} completion ({g} - {b_}) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
        print(f"   end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    print()
    pair("both@ascii+fix-routeknobs+attend+worldeye", "both@ascii+fix-routeknobs+attend", "W1")
    completion("both@ascii+fix-routeknobs+attend+worldeye", "content@ascii+fix+attend", "W2")
    pair("both+world+recomb-side-routeknobs+attend+worldeye", "both+world+recomb-side-routeknobs+attend", "W3")
    completion("both+world+recomb-side-routeknobs+attend+worldeye", "content+world+recomb-side+attend", "W4")


if __name__ == "__main__":
    main()
