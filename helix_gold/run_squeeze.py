"""Finding 110 (2026-10-03, the user's question): can the memory be squeezed so it stays small while the vocabulary grows?

Voyager's connection table (habit's assoc, V x V) grows with the square of the vocabulary: at 10k words one table is ~400 MB
(~1.2 GB while learning), at 30k+ it does not fit this machine -- and the method layer clones the learner per branch. Before
designing a squeezed memory, a read-only check (the diagnosis-first habit): at the end of every session the trained table
is replaced by a squeezed copy, the outside test is taken, and the table is put back exactly.

Squeezes (each at 1/8, 1/32, 1/128 of the V x V numbers; representative values, declared here):
  rank -- low-rank product (what is shared across many words: schema-like, compressible),
  top  -- per row the few entries farthest from the row mean, the rest the row mean (specific exceptions: episode-like),
  both -- half low-rank, half top-k on the rest (complementary learning systems: shared structure + sparse exceptions).
Also recorded: the energy share of the top singular values, and the density of the Hebbian record (fraction non-zero).

Condition both+rediscover+squeeze (finding 103's setting, record only). 10 seeds (a probe, as findings 99-100).
Pre-registered:
  Q0 regression -- the scores equal the cached both+rediscover lives exactly (the probe changes nothing).
  Q1 per squeeze and budget, at the last session: kept share = squeezed / full for never-heard and once all, and the
     sign test of squeezed - full (losses).
  Q2 the same over the life (mean over sessions 2-16): does squeezing hurt more as the life fills the table?
  Q3 the smallest budget at which "both" keeps >= 0.99 of never-heard (the candidate for a trained squeezed memory).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

COND, REF = "both+rediscover+squeeze", "both+rediscover"

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in (COND, REF) for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    same = all(np.allclose([s["never-heard"] for s in out[(sd, COND)]], [s["never-heard"] for s in out[(sd, REF)]],
                           atol=1e-12) for sd in range(n_seeds))
    print(f"=== {n_seeds} seeds; squeezing the connection table (record only) ===")
    print(f"Q0 regression identical: {same}")
    life = [out[(sd, COND)] for sd in range(n_seeds)]
    print("record density (Hebbian, last session): " + f"{np.mean([l[-1]['sq record density'] for l in life]):.3f}")
    for d in RL.SQUEEZE_BUDGETS:
        print(f"energy in top r (budget 1/{d}) by session: "
              + " ".join(f"{np.mean([l[i][f'sq energy r{d}'] for l in life]):.2f}" for i in range(len(life[0]))))
    for k in ("never-heard", "once all"):
        print(f"\n--- {k} (full at the end {np.mean([l[-1][k] for l in life]):.3f}) ---")
        for how in ("rank", "top", "both"):
            for d in RL.SQUEEZE_BUDGETS:
                key = f"sq {how}{d} {k}"
                end = np.array([l[-1][key] - l[-1][k] for l in life])
                kept = np.mean([l[-1][key] for l in life]) / max(1e-9, np.mean([l[-1][k] for l in life]))
                lif = np.array([np.mean([l[i][key] - l[i][k] for i in range(1, len(l))]) for l in life])
                w, lo = int((end > 1e-12).sum()), int((end < -1e-12).sum())
                print(f"{how:4} 1/{d:<4} kept {kept:.3f}  end {end.mean():+.3f} ({w}:{lo}, p={sign_p(w, lo):.4f})"
                      f"  life {lif.mean():+.3f}")
