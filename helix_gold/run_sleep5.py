"""Finding 79 diagnostic: recombination practice every other session (told from outside) -- is the trend signal's partial
gain its flicker (spaced practice)?

Earlier notes -- Finding 79 follow-up: the trend signal with a least-squares line through the whole ledger (awake while the line's value now
is positive) instead of per-session signs, which flickered. References cached.

Original notes -- Putting recombination practice to sleep by its PROGRESS, not its level (the user's proposal, 2026-09-28: humans notice
"this is working, it is getting solved" from how fast the progress goes and whether the bottleneck gives way -- judge by the
slope / acceleration of accuracy during learning, not by the final result). Finding 77: judged by the level at the end of a
session (even on clean questions) the practice always looked useful, while its cost built up over sessions.
Precedents: learning progress as the learner's own signal (Oudeyer, Kaplan & Hafner 2007; automated curricula, Graves et al.
2017); people stop studying when their rate of learning stalls (region of proximal learning, Metcalfe & Kornell 2005);
learning-curve extrapolation to stop runs early (Domhan et al. 2015). Voyager used learning progress already for review
picking (finding 41).
Two signals, each against the same clean check (new categories' held-out sentences), no new constant:
  both+recomb-slope -- within the session: the branch whose clean score still climbs more over the second half of practice
                       (epochs 100 -> 200) wins -- "fast but capped" vs "slower but still climbing"
  both+recomb-trend -- across sessions: the part's advantage (on - off) per session goes into a ledger; the part stays awake
                       while its advantage is positive and has not shrunk more often than it grew ("giving less and less")
References (cached): content, both, both+recomb, both+recomb-sleep (level, finding 77), both+recomb-until8 (told from outside).
Key comparisons, fixed before running: each vs both+recomb and vs both, end and life. 30 seeds, life32.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["both+recomb-alt"]
REFS = ["content", "both", "both+recomb", "both+recomb-trend", "both+recomb-until8", "both+recomb-fit"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    tasks += [(sd, c) for c in REFS[3:] for sd in range(n_seeds)]     # cached if run_sleep / run_sleep2 ran
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    refs = [RL.load_seed(sd, {}) for sd in range(n_seeds)]
    res = [{**{c: refs[sd][c] for c in REFS[:3]}, **{c: out[(sd, c)] for c in CONDS + REFS[3:]}} for sd in range(n_seeds)]

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; sleeping by progress (life32) ===")
    for key in ["never-heard", "recomb awake", "once all", "lr"]:
        print(f"\n--- {key} ---")
        for c in REFS + CONDS:
            print(f"{c:20} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(16)))
    print("\n--- paired (never-heard) ---")
    for g in CONDS + ["both+recomb-trend", "both+recomb-fit", "both+recomb-until8"]:
        for b in ["both+recomb", "both"]:
            dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b]]) for r in res])
            de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in res])
            print(f"{g:20} vs {b:12} life {sgn(dl)}  end {sgn(de)}")


if __name__ == "__main__":
    main()
