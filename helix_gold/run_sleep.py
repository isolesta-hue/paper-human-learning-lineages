"""Putting woken parts back to sleep with a clean judgement (2026-09-28; finding 75's follow-up, agreed with the user).

Finding 75: self-made recombination practice (and the setter, woken in 90% of sessions) gives a large early lead that shrinks
to a loss at the end (-0.016 vs the method layer alone, 34:65) -- a trait for the young learner that should fall asleep in
time, but the method layer never put it back to sleep. Suspected: its judge's old-category questions have the practice's own
shape, and a one-session trial sees the practice's gain inside the session but not its later cost.
Change (run_long.life, "clean"): parts are no longer branches of the knob search; each part's switch (wake / sleep) is tried
as its own branch practised the full session, and it changes only if it beats the stay branch on a CLEAN check -- the new
categories' held-out sentences (never practised, never recombined). No new constant.
Conditions (life32, 30 seeds, cached per life; references from run_long.py's cache):
  both+recomb-sleep  -- recombination practice awake from birth (as both+recomb), may be put to sleep (and woken again)
  both+wake-clean    -- the woken-able parts of finding 74 (as both+wake), switched by the clean check
Key comparisons, fixed before running: K1 both+recomb-sleep vs both, end (does the late loss go away?) and life (is the early
gain kept?); K2 both+wake-clean vs both+wake, end.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["both+recomb-sleep", "both+wake-clean"]
REFS = ["content", "both", "both+wake", "both+recomb"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS[::-1] for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    refs = [RL.load_seed(sd, {}) for sd in range(n_seeds)]
    res = [{**{c: refs[sd][c] for c in REFS}, **{c: out[(sd, c)] for c in CONDS}} for sd in range(n_seeds)]
    n = len(res[0]["both"])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    def lm(r, c):
        return np.mean([s["never-heard"] for s in r[c]])

    print(f"=== {n_seeds} seeds; clean part judgement (life32) ===")
    for key in ["never-heard", "once all", "recomb awake", "awake chunk_mode", "awake compose", "awake setter",
                "awake fmt_practice", "lr", "review", "trial epochs"]:
        print(f"\n--- {key} ---")
        for c in REFS + CONDS:
            print(f"{c:18} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    print("\n--- paired (never-heard) ---")
    for g, b in [("both+recomb-sleep", "both"), ("both+recomb-sleep", "both+recomb"), ("both+wake-clean", "both+wake"),
                 ("both+wake-clean", "both")]:
        dl = np.array([lm(r, g) - lm(r, b) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in res])
        print(f"{g:18} vs {b:12} life {sgn(dl)}  end {sgn(de)}")
    for g in CONDS:
        gaps = np.array([[r[g][i]["never-heard"] - r["content"][i]["never-heard"] for i in range(n)] for r in res])
        print(f"{g}: gap to content by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))


if __name__ == "__main__":
    main()
