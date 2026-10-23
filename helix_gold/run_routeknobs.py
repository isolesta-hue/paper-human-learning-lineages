"""Finding 95: the method layer gets knobs that reach the body's bottleneck -- the routes' weights when answering (templates'
term, whole-sentence memory's evidence; run_long "-routeknobs", values 0.5 / 1 / 2 / 4). Findings 86-94: the method layer's
knobs (review, learning rate, order, pick) do not touch what limits the body (finding 92: templates alone 0.96, the whole
learner 0.87). 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running:
  K1 (glyph world) both@ascii+fix-routeknobs vs both@ascii+fix, end and life;
  K2 (glyph world) the completion criterion both@ascii+fix-routeknobs - content@ascii+fix, end and widening (s13-16 - s2-4);
  K3 (text world) both+world+recomb-side-routeknobs vs both+world+recomb-side, end and life;
  K4 (text world) the completion criterion both+world+recomb-side-routeknobs - content+world+recomb-side, end and widening.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix", "both@ascii+fix", "both@ascii+fix-routeknobs", "content+world+recomb-side",
         "both+world+recomb-side", "both+world+recomb-side-routeknobs"]


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

    print(f"=== {n_seeds} seeds; route weights as method-layer knobs ===")
    for key in ["never-heard", "once all", "unseen all", "pattern w", "chunk w", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:36} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))

    def pair(g, b_, name):
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        print(f"{name} {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}")

    def completion(g, b_, name):
        gaps = np.array([[r[g][i]["never-heard"] - r[b_][i]["never-heard"] for i in range(n)] for r in res])
        print(f"{name} completion ({g} - {b_}) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
        print(f"   end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    print()
    pair("both@ascii+fix-routeknobs", "both@ascii+fix", "K1")
    completion("both@ascii+fix-routeknobs", "content@ascii+fix", "K2")
    pair("both+world+recomb-side-routeknobs", "both+world+recomb-side", "K3")
    completion("both+world+recomb-side-routeknobs", "content+world+recomb-side", "K4")


if __name__ == "__main__":
    main()
