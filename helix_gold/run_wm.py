"""World-model layer, stage A (2026-10-06, the user's design "a world model built from mathematics as raw material -- a
formula / procedure library", README "④ 월드모델 층 설계 v1"; worldmodel.py).

Question: can the learner answer questions that chain two one-step rules it only ever met apart -- "the object starts at
from(p), is pulled and rises: where does it end?" -- by keeping an explicit state and applying its books one by one? The
bag-of-words body (habit, cued templates, whole-sentence memory) only handles one moment. Chain questions: 3 per motion
room (data_ascii "chain", a separate random stream; no heard or tested item changes), two motion words never heard together,
moves that do not cancel, none crossing the grid edge.

Books from the learner's own first-hand moments (moves in its original memory, per motion word, with the rooms they were
heard in); rollout from the start position; the end cell gets one template unit of evidence per book applied
(log n_known each -- smoke: one unit total was outvoted by the cued templates' one-step ends). Smoke, seed 0: chain at the
end 0.855 with the world model vs 0.058 without; never-heard identical (the world model only speaks on chain questions).

Conditions (30 seeds, glyph world): both@ascii+rediscover+chain (record only) and both@ascii+rediscover+chain+wm.
Pre-registered:
  M1 chain accuracy, world model - none: end, life (sessions with chain questions), sign test.
  M2 the same on the upside-down rooms' chain questions (books confirmed in the room: "chain flipped").
  M3 sanity: every other measure identical (max |difference| of never-heard over all sessions and seeds = 0).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

CONDS = ("both@ascii+rediscover+chain+wm", "both@ascii+rediscover+chain")


def sgn(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; world-model stage A (rollout over the learner's books) ===")
    m, b = CONDS
    for key in ("chain", "chain flipped"):
        g = {c: np.array([[st.get(key, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)]) for c in CONDS}
        for c in CONDS:
            print(f"  {key:13} {c:34} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g[c], 0)))
        d = g[m] - g[b]
        tag = "M1" if key == "chain" else "M2"
        print(f"  {tag} {key} wm - none: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    nh = {c: np.array([[st["never-heard"] for st in out[(sd, c)]] for sd in range(n_seeds)]) for c in CONDS}
    print(f"  M3 never-heard max |wm - none| = {np.abs(nh[m] - nh[b]).max():.2e}")
