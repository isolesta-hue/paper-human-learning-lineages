"""B stage, first formal test (finding 91): the glyph-picture world (data_ascii: "na"-centred 5 x 5 positions with role markers,
motion words as templates, drawn / converted pictures, placement rooms, the new verb, upside-down rooms) with its parts on
(kind-first answers, templates with a cue, the where channel's linear memory). Does the self-correcting learner (method
layer) pull ahead of the fixed one again in this wider world (finding 86 (b): the body had outgrown the old world)?
30 seeds, cache_long_v2.
Key comparisons, fixed before running: K1 the completion criterion both@ascii - content@ascii, end and widening (s13-16 - s2-4);
K2 the provisional default (finding 88) in this world: both+world+recomb-side@ascii vs both@ascii, end and life.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii", "both@ascii", "both+world+recomb-side@ascii"]


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

    print(f"=== {n_seeds} seeds; B stage: the glyph-picture world, long life ===")
    for key in ["never-heard", "once all", "pairs all", "unseen all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:30} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    gaps = np.array([[r["both@ascii"][i]["never-heard"] - r["content@ascii"][i]["never-heard"] for i in range(n)] for r in res])
    print("\nK1 completion (both - content) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    g, b = "both+world+recomb-side@ascii", "both@ascii"
    dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b]]) for r in res])
    de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in res])
    print(f"K2 default parts vs both: life {sgn(dl)}  end {sgn(de)}")


if __name__ == "__main__":
    main()
