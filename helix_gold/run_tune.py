"""Finding 115 (2026-10-04, step 2 of the method-making learner, from finding 114): the method layer sets part weights by
the world-shaped self-experiment.

Finding 114: on the learner's own world-shaped questions (world_items -- the core partners stay, the background is swapped
for other known words of the same category and kind), "what if the whole-sentence memory / the connection table weighed
w times as much" points the same way as the outside test in the text world (chosen weight's outside gain: whole-sentence
memory sessions 9-16 +0.0087, 10:0; connection table sessions 2-16 +0.0089, 9:1). Here the choice is USED: at every
method session the learner computes the what-if on its own world-shaped questions (stay branch, first rung) and answers
with the best weights (CF_W = 0, 0.5, 1, 1.5; ties keep the current weight, else 1) -- `+tune`. The connection table's
answering weight is a new setting (model.assoc_w, 1 = as before); nothing is relearned, the outside test is never seen.
Text world only (finding 114: the glyph world has no room in these two weights -- per-world methods). Two worlds of the
text ladder: plain (life32) and rung 1 (common words, @share).

Conditions (30 seeds, cache_long_v2): both+rediscover+tune, both+rediscover@share+tune; references cached: both+rediscover,
both+rediscover@share, content, content@share.
Pre-registered:
  T1 tune vs no tune (same seed, same world): end, life, sessions 9-16 -- per world, sign tests.
  T2 the completion criterion with tune: end, widening (s13-16 - s2-4), late widening (s13-16 - s6-8), against no tune's.
  T3 the chosen weights over the life (mean chunk w, assoc w by session).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

WORLDS = {"plain": ("both+rediscover+tune", "both+rediscover", "content"),
          "share": ("both+rediscover@share+tune", "both+rediscover@share", "content@share")}


def sgn(x):
    x = np.asarray(x)
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if len(sys.argv) > 3:                           # the second stage only for the worlds still undecided
        WORLDS = {k: v for k, v in WORLDS.items() if k in sys.argv[3:]}
    tasks = [(sd, c) for trio in WORLDS.values() for c in trio for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the method layer sets part weights by the world-shaped self-experiment (text world) ===")
    for world, (tu, base, fixed) in WORLDS.items():
        L_ = {c: [out[(sd, c)] for sd in range(n_seeds)] for c in (tu, base, fixed)}
        n = len(L_[tu][0])
        nh = {c: np.array([[st["never-heard"] for st in l] for l in L_[c]]) for c in L_}
        print(f"\n##### {world}")
        for c in (tu, base, fixed):
            print(f"  {c:28} " + " ".join(f"{v:.3f}" for v in nh[c].mean(0)))
        d = nh[tu] - nh[base]
        print(f"  T1 tune - no tune: end {sgn(d[:, -1])} | life {sgn(d.mean(1))} | s9-16 {sgn(d[:, 8:].mean(1))}")
        for c in (tu, base):
            g = nh[c] - nh[fixed]
            print(f"  T2 {c:28} completion end {sgn(g[:, -1])} | widening {sgn(g[:, 12:16].mean(1) - g[:, 1:4].mean(1))}"
                  f" | late widening {sgn(g[:, 12:16].mean(1) - g[:, 5:8].mean(1))}")
        for k in ("tune chunk", "assoc w"):
            print(f"  T3 {k} by session: " + " ".join(f"{np.nanmean([l[i].get(k, np.nan) for l in L_[tu]]):.2f}"
                                                    for i in range(n)))
