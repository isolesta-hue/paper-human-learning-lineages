"""Finding 126, T4 second stage (2026-10-10; the user: postpone a little, check at 50 seeds first, go to 100 only if still
undecided -- a three-stage 30 -> 50 -> 100 for this one comparison). At 30 seeds (multi_30.out) T4 never-heard, track - bind
only: max |difference| 7.5e-02, end +0.0009 (6:6, p = 1.0). Pre-registered here, before any run beyond 30 seeds: the same two
conditions, 50 seeds (seeds 0-29 from the cache, after the cache-reuse audit check_multi50_reg.out).
  T4 never-heard, track - bind only: end, life (sign tests). Decided if p < 0.005 (kept strict: the data are looked at a
     third time) or the 99% interval of the end difference lies within +-0.002; otherwise 100 seeds.
  50-seed result (multi_50.out, compute helper, 2026-10-11): end +0.0022 (12:11, p = 1.0), life +0.0019 (18:7, p = 0.043),
     99% interval [-0.0035, +0.0084] -> undecided -> 100 seeds with this same script (`run_multi50.py 100 4 > multi_100.out`;
     seeds 0-49 from the cache). Rule for the 100-seed stage, fixed now before running: decided if p < 0.005 or the 99% interval
     of the end difference lies within +-0.002; if neither, recorded as "no direction found at 100 seeds" and closed (no 4th stage).
"""
import sys

import numpy as np

import run_long as RL
from run_multi import B, sgn

TRACK = B + "+track"

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in (TRACK, B) for sd in range(n_seeds)]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))      # D57: a dead worker fails loudly; chunksize 1
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    d = g(TRACK, "never-heard") - g(B, "never-heard")
    end = d[:, -1][~np.isnan(d[:, -1])]
    lo, hi = np.percentile(np.random.default_rng(0).choice(end, (10000, len(end))).mean(1), [0.5, 99.5])
    print(f"=== {n_seeds} seeds; finding 126 T4 (never-heard, track - bind only) ===")
    print(f"  end {sgn(end)} | life {sgn(np.nanmean(d, 1))} | max |d| {np.nanmax(np.abs(d)):.2e} | end 99% interval [{lo:+.4f}, {hi:+.4f}]")
