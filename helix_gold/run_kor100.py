"""Finding 125, second stage (2026-10-09; two-stage seed rule). At 30 seeds (kor_30.out) K1-K5 were decided (30:0) and K6
was not: never-heard, wmlast - bind, max |difference| 8.2e-02, end -0.0010 (5:2, p=0.45). Pre-registered here, before
any run beyond 30 seeds: the same two conditions, 100 seeds (seeds 0-29 from the cache).
  K6 never-heard, wmlast - bind: end, life (sign tests, p < 0.005), and max |difference|.
  Also recorded: K4 story and K5 chain, wmlast - bind, end and life.
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_kor import B, sgn

BIND, LAST = B + "+bind", B + "+bind+wmlast"

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in (BIND, LAST) for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; finding 125 second stage (K6) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    for k in ("never-heard", "story", "chain"):
        for c in (BIND, LAST):
            print(f"  {k:11} {c[len(B):]:13} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k in (("K6 never-heard", "never-heard"), ("K4 story", "story"), ("K5 chain", "chain")):
        d = g(LAST, k) - g(BIND, k)
        print(f"  {tag} wmlast - bind: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))} | max |d| {np.nanmax(np.abs(d)):.2e}")
