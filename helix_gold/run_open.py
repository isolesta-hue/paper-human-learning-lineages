"""Finding 109: the method layer makes new options itself (the user's choice (A), 2026-10-03) -- in the TEXT world, where the
completion criterion holds (finding 103) but the gap stops growing after session 5 (finding 108's hypothesis: the menu of
options is finite and used up). run_long "+open": range extension (a knob pushed to the end of its ladder by a winning move
gets a new rung beyond it, spaced by the ladder's own ratio) and composite moves (after a session where no single move won,
the two best-ranked single moves together). On the rediscovery setting (old body + method layer + the hand-found improvements
as knobs); the fixed version is the old body with start settings. Per the user (same day): methods are found per world -- the
glyph world gets its own track. 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running:
  O1 both+rediscover+open vs both+rediscover: end, life, s13-16;
  O2 the completion criterion both+rediscover+open - content: end, widening (s13-16 - s2-4) and LATE widening (s13-16 - s6-8),
     against the same numbers for both+rediscover;
  also: rungs added and composite moves that won, per life.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content", "both+rediscover", "both+rediscover+open"]


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

    print(f"=== {n_seeds} seeds; the method layer makes new options itself (text world) ===")
    for key in ["never-heard", "once all", "unseen all", "review", "lr", "rungs added", "pairs won", "changed"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:24} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    g, b_ = "both+rediscover+open", "both+rediscover"
    dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
    de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
    late = np.array([np.mean([r[g][i]["never-heard"] - r[b_][i]["never-heard"] for i in range(12, 16)]) for r in res])
    print(f"\nO1 {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}  s13-16 {sgn(late)}")
    for x, name in [(g, "O2"), (b_, "O2 ref")]:
        gaps = np.array([[r[x][i]["never-heard"] - r["content"][i]["never-heard"] for i in range(n)] for r in res])
        print(f"{name} completion ({x} - content) by session " + " ".join(f"{v:+.3f}" for v in gaps.mean(0)))
        print(f"   end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}"
              f" | late widening s13-16 - s6-8 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 5:8].mean(1))}")


if __name__ == "__main__":
    main()
