"""Finding 117 (2026-10-04, from finding 115): does the self-experiment's weight choice hurt late in life because it also
changes the method layer's own eye?

Finding 115: choosing the answering weights by the world-shaped self-experiment (+tune) wins in the harder world (end +0.050,
25:5) but in the plain world only speeds learning (life +0.018, 66:34) and ends level (-0.004), with the late widening lost
(-0.030, 33:67). A look at the lives: with +tune the method layer chose a lower learning rate late (0.020 vs 0.024) and the
once-heard pairs plateau (0.37-0.39). Hypothesis (logic step): the tuned weights are part of the settings every branch
inherits, so the judge that picks review / lr / parts looks THROUGH the lowered whole-sentence weight and values learning
differently. Test: +tunex -- the chosen weights only for answering (the learner's own answers and the outside test); the
method layer's branches keep weight 1.

Conditions (30 seeds): both+rediscover+tunex, both+rediscover@share+tunex; cached: +tune, no tune, content.
Pre-registered:
  X1 tunex vs tune: end, life, late widening of the completion gap (s13-16 - s6-8) -- per world.
  X2 tunex vs no tune: end, life -- per world.
  X3 the learning rate and review chosen by the method layer, sessions 9-16, tunex vs tune vs no tune.
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

WORLDS = {"plain": ("both+rediscover+tunex", "both+rediscover+tune", "both+rediscover", "content"),
          "share": ("both+rediscover@share+tunex", "both+rediscover@share+tune", "both+rediscover@share", "content@share")}


def sgn(x):
    x = np.asarray(x)
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for quad in WORLDS.values() for c in quad for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; tuned weights for answering only (the method layer's eye at weight 1) ===")
    for world, (x, t, b, f) in WORLDS.items():
        nh = {c: np.array([[st["never-heard"] for st in out[(sd, c)]] for sd in range(n_seeds)]) for c in (x, t, b, f)}
        print(f"\n##### {world}")
        for c in (x, t, b, f):
            print(f"  {c:30} " + " ".join(f"{v:.3f}" for v in nh[c].mean(0)))
        late = {c: (nh[c] - nh[f])[:, 12:16].mean(1) - (nh[c] - nh[f])[:, 5:8].mean(1) for c in (x, t, b)}
        d = nh[x] - nh[t]
        print(f"  X1 tunex - tune: end {sgn(d[:, -1])} | life {sgn(d.mean(1))} | late widening {sgn(late[x] - late[t])}")
        d = nh[x] - nh[b]
        print(f"  X2 tunex - no tune: end {sgn(d[:, -1])} | life {sgn(d.mean(1))} | late widening {sgn(late[x] - late[b])}")
        for k in ("lr", "review", "tune chunk", "assoc w"):
            print(f"  X3 {k:10} s9-16: " + " | ".join(
                f"{c.split('+')[-1] if '+tune' in c else 'no tune'} {np.nanmean([out[(sd, c)][i].get(k, np.nan) for sd in range(n_seeds) for i in range(8, 16)]):.3f}"
                for c in (x, t, b)))
