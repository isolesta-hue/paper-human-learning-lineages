"""Stage-1 path step 4+ (2026-10-09; ledger D2, second part -- goal-directed motion; Gergely & Csibra 1995 / 2003: infants
expect agents to move efficiently toward a goal). Finding 139. Written and fixed BEFORE the seed-0 smoke results.

World "@goal" (data_ascii goal=True = the "@kor" world plus): in the placement rooms the agent (named first), its goal and
another thing are named with their cells, then the agent's one-cell steps. Heard: whole shortest paths ending at the goal.
Test "goal which": the first steps, then EODIRO -- which thing is it going to? "goal next": the first steps, the next motion
word asked (unique). The given steps approach the goal and never the other thing.
Body part "+goal" (worldmodel): the agent's place rolled with the learner's own books; the goal = the thing whose distance
shrank most; the next step = the word that brings the agent closest to it. This principle is a HUMAN-GIVEN part (the
teleological stance is core knowledge); what is learned is the moves. Control "+goalrand": the other thing as the goal.
Offline logic check (seeds 0-2, true books): which / next 1.00, control 0.00.

Conditions (30 seeds; W = +rediscover+chain+wm+bind+wmlast):
  G = both@ascii@goal W +goal    Gr = ... +goalrand    Gn = ... W    body = both@ascii@goal+rediscover+chain
Pre-registered:
  R1 "goal which", G - Gn and G - Gr: end, life.
  R2 "goal next", G - Gn and G - Gr: end, life.
  R3 "never-heard", G - Gn: end, life (the leaning cost seen in findings 137 / 138's smokes).
  R4 other measures (story, chain), G vs Gn: max |difference|.
What this assumes (CLAUDE.md 2026-10-09): one agent, named first; one-cell steps; city-block distance on the 5 x 5 grid; no
obstacles (the detour / efficiency-violation test -- an obstacle removed, the old detour kept -- is v2); the goal is one of
the named things; time is story order only.
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn

W = "+rediscover+chain+wm+bind+wmlast"
G0 = "both@ascii@goal" + W
CONDS = {"G": G0 + "+goal", "Gr": G0 + "+goalrand", "Gn": G0, "body": "both@ascii@goal+rediscover+chain"}

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS.values() for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; goal-directed motion (D2, finding 139) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in range(n_seeds)])
    for k in ("goal which", "goal next"):
        for c in CONDS:
            print(f"  {k:10} {c:4} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k, a, b in (("R1", "goal which", "G", "Gn"), ("R1", "goal which", "G", "Gr"), ("R2", "goal next", "G", "Gn"),
                         ("R2", "goal next", "G", "Gr"), ("R3", "never-heard", "G", "Gn")):
        d = g(a, k) - g(b, k)
        print(f"  {tag} {k}, {a} - {b}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("story", "chain"):
        d = g("G", k) - g("Gn", k)
        print(f"  R4 {k:6} max |G - Gn| {np.nanmax(np.abs(d)):.2e}")
