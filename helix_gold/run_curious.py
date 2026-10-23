"""D35 (d), the first curiosity experiment (2026-10-12; user agreed "그렇게 진행해줘"). Written and fixed BEFORE the seed-0
smoke results.

Curiosity as a value signal: which old category to review. Voyager's method layer can already choose the "curious" review
pick (finding 41: weight = |surprise| + floor, surprise = the category's accuracy change last session minus its running
expectation -- a reward-prediction error). The learning-progress account (Oudeyer & Kaplan 2007; the infants' Goldilocks
effect, Kidd et al. 2012) says the drive should be the progress itself, not the surprise in it. "+lpcur": the same pick,
the same floor, the same review amount, the same knob the method layer may or may not choose -- only the signal is
|accuracy change over the last session|. No new constant.

Conditions (30 seeds): B = both@ascii+rediscover+chain+wm+wmlast (run_known's B, shared cache), C = B + "+lpcur".
Pre-registered (end = session 16, life = mean over sessions):
  C1 never-heard, C - B: end, life.
  C2 older (old categories' tests), once all (heard once), C - B: end, life (the signal includes forgetting = negative
     progress, so the old and the once-heard are where a difference is expected first).
  C3 heard-only stories / chain: max |difference| and end sign (expected ~0).
  Recorded only: "pick curious" in C (how often the method layer used the curious pick) -- if it is rarely chosen, a null
  result says little about the signal.
Two-stage seed rule: decided if p < 0.005 at 30, otherwise 100 seeds.
What this assumes: the latest session's change is the progress (no longer window); progress is per category; the method
layer still decides whether to use the curious pick at all (the signal is not forced on); the glyph-picture world of B.
"""
import sys

import numpy as np

import run_long as RL
from run_count import sgn

B = "both@ascii+rediscover+chain+wm+wmlast"
C = B + "+lpcur"
KEYS = ("never-heard", "older", "once all", "chain", "story")

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in (C, B) for sd in range(n_seeds)]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))      # D57: a dead worker fails loudly; chunksize 1
    print(f"=== {n_seeds} seeds; D35 (d) curiosity = learning progress vs surprise ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    for k in KEYS:
        for tag, c in (("C", C), ("B", B)):
            print(f"  {k:12} {tag} " + " ".join(f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    print("  pick curious C " + " ".join(f"{v:.2f}" for v in np.nanmean(g(C, "pick curious"), 0)))
    for tag, k in (("C1", "never-heard"), ("C2", "older"), ("C2", "once all")):
        d = g(C, k) - g(B, k)
        print(f"  {tag} {k}, C - B: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("chain", "story"):
        d = g(C, k) - g(B, k)
        print(f"  C3 {k}: max |d| {np.nanmax(np.abs(d)):.3f} | end {sgn(d[:, -1])}")
