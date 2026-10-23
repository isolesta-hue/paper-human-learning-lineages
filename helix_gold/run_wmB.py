"""Finding 122 (2026-10-07): world-model stage B -- a context book from the room's surprises (README "④ 월드모델 층 설계 v1",
the user's formula-library design; NH 8-9: a changed rule needs its own context, opened by surprise).

World "@hold" (data_ascii hold=True): as the glyph world, but in every upside-down room one motion word is never heard
(separate random stream); that room's chain questions all use it. Stage A (finding 120) rolls a word unheard in the room
with its usual book -- the wrong direction there. Stage B ("+wmctx"): when the room's own books for the words heard there
run against the usual ones, the room is taken to reverse every move, and the unheard word's usual move is reversed too
(keep / reverse = the smallest library of rules about rules, by majority over the room's words; no constant).
Offline check (seed 0, untrained learner, all heard moments): held-room chain questions -- stage A 0 / 21, stage B 21 / 21.
Default worlds unchanged (data identical without hold, seeds 0 and 3; the context step runs only with +wmctx).

Conditions (30 seeds): both@ascii@hold+rediscover+chain+wm (stage A), both@ascii@hold+rediscover+chain+wmctx (stage B).
Pre-registered:
  B1 chain accuracy in the upside-down rooms ("chain flipped" = the held rooms), stage B - stage A: end, sessions 13-16.
  B2 chain accuracy over all rooms, B - A: end (only the held rooms should differ).
  B3 other measures (never-heard, once all): max |B - A| and end sign tests (expected ~0; self-made practice may leak).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

CONDS = ("both@ascii@hold+rediscover+chain+wmctx", "both@ascii@hold+rediscover+chain+wm")


def sgn(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; world-model stage B (context book from the room's surprises) ===")
    b, a = CONDS
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    for key, tag in (("chain flipped", "B1"), ("chain", "B2")):
        for c in CONDS:
            m = np.nanmean(g(c, key), 0)
            print(f"  {key:13} {c:40} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in m))
        d = g(b, key) - g(a, key)
        print(f"  {tag} {key} B - A: end {sgn(d[:, -1])}" + (f" | s13-16 {sgn(np.nanmean(d[:, 12:16], 1))}" if tag == "B1" else ""))
    for key in ("never-heard", "once all"):
        d = g(b, key) - g(a, key)
        print(f"  B3 {key}: max |B - A| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
