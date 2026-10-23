"""Finding 94: the long eye -- a shadow learner keeps the state before each setting change and meets the same experience; it
re-enters every later branch race and the learner goes back to it if it wins (run_long "+longeye"). Findings 87, 89, 91, 93:
the method layer judges a change by the one session after it, while the costs (interference, over-trusting the side part)
come later. 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running:
  K1 (glyph world) the completion criterion both@ascii+fix+longeye - content@ascii+fix, end and widening (s13-16 - s2-4);
  K2 (glyph world) both@ascii+fix+longeye vs both@ascii+fix, end and life;
  K3 (text world, finding 87's short eye) both+recomb-sideknobs+longeye vs both+recomb-sideknobs, end and life.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix", "both@ascii+fix", "both@ascii+fix+longeye", "both+recomb-sideknobs",
         "both+recomb-sideknobs+longeye"]


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

    print(f"=== {n_seeds} seeds; the long eye (shadow lineage) ===")
    for key in ["never-heard", "once all", "unseen all", "lr", "review", "side w", "swaps", "changed"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:32} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))

    def pair(g, b_, name):
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        print(f"{name} {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}")
    gaps = np.array([[r["both@ascii+fix+longeye"][i]["never-heard"] - r["content@ascii+fix"][i]["never-heard"] for i in range(n)]
                     for r in res])
    print("\nK1 completion (glyph, long eye) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    pair("both@ascii+fix+longeye", "both@ascii+fix", "K2")
    pair("both+recomb-sideknobs+longeye", "both+recomb-sideknobs", "K3")


if __name__ == "__main__":
    main()
