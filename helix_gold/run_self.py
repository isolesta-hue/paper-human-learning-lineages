"""Stage-1 path step 4+ (2026-10-10; ledger D2, third part -- the self by contingency; Watson 1972, Bahrick & Watson 1985:
infants find what they move by the timing match between their action and a seen motion; mirror self-recognition as the
harder case). Finding 140. Written and fixed BEFORE the seed-0 smoke results.

World "@self" (data_ascii self_=True = the "@kor" world plus): Voyager's own ACTION words (ACT0-3) for the first time. In the
placement rooms one object is "me" (another one in each room -- a body seen anew). Heard: an action and the move it made
(only "me" moves). Test "self which": two actions, three things move each time; NUGU (who is me?); "self mirror": the same
with "me" seen mirrored (left / right swapped; every item has a left / right action). The others never follow the actions
under any square symmetry.
Body part "+self": the action words' moves from the learner's own books; "me" = the thing whose move follows every action
through one and the same symmetry of the square (the 8-symmetry library of finding 131 -- HUMAN-GIVEN). Control "+selfid":
the identity only. Offline logic check (seeds 0-2, true books): which 1.00 / 1.00; mirror 1.00 / identity-only 0.00.

Conditions (30 seeds; W = +rediscover+chain+wm+bind+wmlast):
  S = both@ascii@self W +self    Si = ... +selfid    Sn = ... W    body = both@ascii@self+rediscover+chain
Pre-registered:
  U1 "self which", S - Sn: end, life.
  U2 "self mirror", S - Si: end, life (the mirror needs the symmetry library).
  U3 "self which", S vs Si: end sign test (expected ~0).
  U4 "never-heard", S - Sn: end, life (the leaning question of findings 137-139).
What this assumes (CLAUDE.md 2026-10-09): one "me" per room; one-cell moves; the action words' meanings learned from moves
where only "me" moved; the 8 square symmetries as the library; two actions per question; no timing beyond the moment
(the action and the moves are in the same moment -- a delayed self, as in video-delay tests, is not covered).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn

W = "+rediscover+chain+wm+bind+wmlast"
S0 = "both@ascii@self" + W
CONDS = {"S": S0 + "+self", "Si": S0 + "+selfid", "Sn": S0, "body": "both@ascii@self+rediscover+chain"}

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS.values() for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the self by contingency (D2, finding 140) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in range(n_seeds)])
    for k in ("self which", "self mirror"):
        for c in CONDS:
            print(f"  {k:11} {c:4} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k, a, b in (("U1", "self which", "S", "Sn"), ("U2", "self mirror", "S", "Si"), ("U3", "self which", "S", "Si"),
                         ("U4", "never-heard", "S", "Sn")):
        d = g(a, k) - g(b, k)
        print(f"  {tag} {k}, {a} - {b}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
