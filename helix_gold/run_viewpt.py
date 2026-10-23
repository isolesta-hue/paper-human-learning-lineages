"""Stage-1 path step 4+ (2026-10-09; ledger D2, its viewpoint tests carried over from D17 -- the adjusted order D17 v1 ->
D2). Finding 138. Written and fixed BEFORE the seed-0 smoke results.

World "@viewpt" (data_ascii views=2 = the "@views" world, unchanged, plus): the BACK view (front seen from behind, (-x, y))
heard in some sightings only; TURNS -- a quarter turn about the vertical axis through the room's centre, (x, y, z) ->
(z, y, -x) -- seen in three views before and after. Test "viewpt back": front and top of a point never seen, the back asked
(Piaget's three mountains); "viewpt turn": three views before a turn, one after asked (mental rotation, Shepard & Metzler).
Body parts: "+views" (finding 137's axis matching -- with signs, so the back view needs nothing new) and "+vturn": each motion
word's effect per view axis as "after = sign x (an axis of a view before) + offset" (a shift and a turn in one form).
Offline logic check (seeds 0-2, true books, all heard): back 1.00 with +views alone; turn 0.13 -> 1.00 with +vturn.

Conditions (30 seeds; W = +rediscover+chain+wm+bind+wmlast):
  P = both@ascii@viewpt W +views+vturn    Pv = ... W +views    Pn = ... W    body = both@ascii@viewpt+rediscover+chain
Pre-registered:
  Q1 "viewpt back", Pv - Pn: end, life (the other side's view from matched axes and signs).
  Q2 "viewpt turn", P - Pv: end, life (a turn needs axis-trading verb maps, not shifts).
  Q3 "views fill" and "views depth", P vs Pv: end sign tests (expected ~0: shifts are a special case of the verb maps).
  Q4 "never-heard", Pv - Pn and P - Pv: end, life (the leaning cost seen in finding 137's smoke).
What this assumes (CLAUDE.md 2026-10-09; places to change when extending): view labels arrive as tags; the back view is the
front mirrored left-right; turns are quarter turns about one axis through the room's centre; one verb map per word for the
whole life (no context); image grid 5 x 5; no time axis.

AMENDMENT (2026-10-12, user decision after the review REVIEW_2026-10-12.md; written before any seed beyond the seed-0 smoke was
seen): first stage = 10 seeds (`10 4 > <name>_10.out`). A primary key is decided at 10 seeds if p < 0.005 (10:0 gives 0.002);
otherwise it goes to 30 (the 10 seeds are reused). Keys that are already 1.000 WITHOUT the part in the seed-0 smoke cannot
separate the conditions -- they are reported only, not judged. Side-effect keys (never-heard and other small effects) are
reported only at this stage; their summed cost is measured once in the integrated life (all adopted parts on vs off), not
part by part. Every key is also compared with the trivial baselines of the review (majority answer; co-occurrence in the
heard sentences) before any claim that the part is needed.
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn

W = "+rediscover+chain+wm+bind+wmlast"
P0 = "both@ascii@viewpt" + W
CONDS = {"P": P0 + "+views+vturn", "Pv": P0 + "+views", "Pn": P0, "body": "both@ascii@viewpt+rediscover+chain"}

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS.values() for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; viewpoint tests (D2, finding 138) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in range(n_seeds)])
    for k in ("viewpt back", "viewpt turn", "views fill", "views depth"):
        for c in CONDS:
            print(f"  {k:12} {c:4} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k, a, b in (("Q1", "viewpt back", "Pv", "Pn"), ("Q2", "viewpt turn", "P", "Pv"), ("Q3", "views fill", "P", "Pv"),
                         ("Q3", "views depth", "P", "Pv"), ("Q4", "never-heard", "Pv", "Pn"), ("Q4", "never-heard", "P", "Pv")):
        d = g(a, k) - g(b, k)
        print(f"  {tag} {k}, {a} - {b}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
