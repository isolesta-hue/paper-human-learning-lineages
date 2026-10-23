"""Stage-1 path step 4 (2026-10-08; the user's order "objects with properties kept apart from books"; the principle
borrowed from New Horizons NH 56 under the sibling rule in CLAUDE.md -- principle only, built for Voyager's body).

World "@objs" (data_ascii props=True = the "@multi" world plus): every object of a room is HEAVY or LIGHT (a separate
random stream; at least one of each). Stories "[B ga] at(q) GO [A ga] from(p) VERB GO [X ga] EODI": when A's move lands on
B's cell a light B is pushed one cell the same way (A takes its cell), a heavy B stops A (both stay). Heard: 2 per object as
the obstacle, with the asked object's end scene; test "objs hit" (every object as the obstacle, both objects asked) and
"objs free" (the same move missing B). Default, @kor, @hold and @multi worlds unchanged (checked, seeds 0, 3, 5).

Body part "+props" (worldmodel.py): the motion books stay shared; each object gets an entry -- light or heavy -- learned
one-shot from heard collision stories (each read twice, the collided object taken as light, then heavy; the reading that
lands on the scene names the entry); while tracking, a mover landing on another object meets its entry (light: it moves the
same way; heavy: the mover goes back; no entry: no answer). The rule's shape is given (as NH 56; NH 70 learns it from a
library -- later). Control "+propsrand": each object reads another object's entry. Without +props the tracking passes
through (the mover moves, the obstacle stays).

Conditions (30 seeds; B = both@ascii@objs+rediscover+chain+wm+bind+wmlast+track): B+"+props", B+"+propsrand", B.
Pre-registered:
  O1 collision accuracy ("objs hit"), props - B: end, life.
  O2 props - propsrand: end, life.
  O3 "objs free": props vs B (expected equal: no collision, the table is not read).
  O4 recorded: the table's size and the share of entries that are right (the world's heavy / light).
  O5 other measures (multi, story, never-heard): props vs B, max |difference|, end sign test (expected ~0; the part can
     fire while learning heard collision stories whose scene is the hidden element).
Added before the first run (2026-10-08, step 4+, the user's decision: core knowledge and infant-style tests; record only):
  O6 violation of expectation -- pairs of a possible and an impossible outcome of one story; the learner "looks longer"
     when its own surprise (-log p of the outcome shown, the whole learner's answer) is larger for the impossible one.
     Kinds: solid (a heavy B: A ends inside B's cell), push (a light B hit but stays), causal (B moves untouched), teleport
     (two steps for one). Share "looks longer" per kind and condition (end, life); props - B on solid and push (expected:
     only the property table knows solidity and pushing; causal and teleport similar in every condition).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

B = "both@ascii@objs+rediscover+chain+wm+bind+wmlast+track"
CONDS = (B + "+props", B + "+propsrand", B)


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
    print(f"=== {n_seeds} seeds; objects with properties kept apart from the books (stage-1 step 4) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    props, rand, none = CONDS
    for k in ("objs hit", "objs free", "objs table right", "objs table size"):
        for c in CONDS:
            print(f"  {k:16} {c[len(B):] or '(no table)':12} " +
                  " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, other in (("O1 props - no table", none), ("O2 props - propsrand", rand)):
        d = g(props, "objs hit") - g(other, "objs hit")
        print(f"  {tag}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    d = g(props, "objs free") - g(none, "objs free")
    print(f"  O3 free: max |props - no table| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
    for kv in ("solid", "push", "causal", "teleport"):
        print(f"  O6 voe {kv:8} looks longer, end: " + " | ".join(
            f"{c[len(B):] or 'no table'} {np.nanmean(g(c, 'voe ' + kv)[:, -1]):.3f}" for c in CONDS))
    for kv in ("solid", "push"):
        d = g(props, "voe " + kv) - g(none, "voe " + kv)
        print(f"  O6 voe {kv} props - no table: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("multi", "story", "never-heard"):
        d = g(props, k) - g(none, k)
        print(f"  O5 {k}: max |props - no table| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
