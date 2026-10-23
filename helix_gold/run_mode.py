"""Finding 98: the mode layer, first step -- a "protect" mode chosen from the dashboard (the old heard indicator fell three
sessions in a row -> enter; rose three in a row -> leave; otherwise stay), in which the judge sees old categories with two eyes
(imagination + holding on); explore (default) = imagination only (run_long "+mode"). Finding 97: two eyes ALL the time
made the text world choose slow learning. With the route-weight knobs (finding 95). 30 seeds, cache_long_v2.
Key comparisons, fixed before running:
  K1 (glyph world) both@ascii+fix-routeknobs+mode vs both@ascii+fix-routeknobs, end and life;
  K2 (glyph world) the completion criterion both@ascii+fix-routeknobs+mode - content@ascii+fix, end and widening;
  K3 (text world) both+world+recomb-side-routeknobs+mode vs both+world+recomb-side-routeknobs, end and life (must not lose);
  K4 (text world) the completion criterion both+world+recomb-side-routeknobs+mode - content+world+recomb-side.
Also reported: the share of sessions in protect, per world.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix", "both@ascii+fix-routeknobs", "both@ascii+fix-routeknobs+mode", "content+world+recomb-side",
         "both+world+recomb-side-routeknobs", "both+world+recomb-side-routeknobs+mode"]


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
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.4f})"

    print(f"=== {n_seeds} seeds; the mode layer (protect) ===")
    for key in ["never-heard", "once all", "unseen all", "pattern w", "chunk w", "lr", "review", "dash old heard", "protect"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:42} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))

    def pair(g, b_, name):
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        print(f"{name} {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}")

    def completion(g, b_, name):
        gaps = np.array([[r[g][i]["never-heard"] - r[b_][i]["never-heard"] for i in range(n)] for r in res])
        print(f"{name} completion ({g} - {b_}) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
        print(f"   end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    print()
    pair("both@ascii+fix-routeknobs+mode", "both@ascii+fix-routeknobs", "K1")
    completion("both@ascii+fix-routeknobs+mode", "content@ascii+fix", "K2")
    pair("both+world+recomb-side-routeknobs+mode", "both+world+recomb-side-routeknobs", "K3")
    completion("both+world+recomb-side-routeknobs+mode", "content+world+recomb-side", "K4")


if __name__ == "__main__":
    main()
