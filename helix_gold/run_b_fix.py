"""Finding 93: the template repairs of finding 92 (kind-consistent templates, specific before general, cued templates vote
on their own pairs, the purest cue first) in the long life of the glyph world, 30 seeds (both@ascii / content@ascii reused).
Key comparisons, fixed before running: K1 both@ascii+fix vs both@ascii, end and life; K2 the completion criterion
both@ascii+fix - content@ascii+fix, end and widening (s13-16 - s2-4).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix", "both@ascii", "both@ascii+fix"]


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
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; B stage: template repairs, long life ===")
    for key in ["never-heard", "once all", "pairs all", "unseen all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:30} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    g, b_ = "both@ascii+fix", "both@ascii"
    dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
    de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
    print(f"\nK1 repairs vs none: life {sgn(dl)}  end {sgn(de)}")
    for key in ("once all", "unseen all"):
        dk = np.array([r[g][-1][key] - r[b_][-1][key] for r in res])
        print(f"   {key} end {sgn(dk)}")
    gaps = np.array([[r[g][i]["never-heard"] - r["content@ascii+fix"][i]["never-heard"] for i in range(n)] for r in res])
    print("K2 completion (both+fix - content+fix) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")


if __name__ == "__main__":
    main()
