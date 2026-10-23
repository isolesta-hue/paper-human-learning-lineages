"""Stage-1 path step 4+ (2026-10-09; ledger D17 v1 -- the three-view world; the user's idea of 2026-09-23, revived 10-09 with
the adjusted order D13 -> D4 design -> D17 v1 -> D2). Finding 137. Written and fixed BEFORE the seed-0 smoke results.

World "@views" (data_ascii views=True = the "@kor" world plus): in the placement rooms objects stand in a 3-D room (x, y, z in
-2..2) and are seen in three pictures at once -- front (x, y), top (x, z), side (z, y) -- each an image on the same 5 x 5 image
cells as every room, with a view label after each cell. Heard: each object's three views; depth moves GOES / COMES (z +- 1)
seen in all three views before and after. Test "views fill": two views of a point never seen, the third asked; "views
depth": three views before a depth move, one view after asked (front unchanged, top / side moved).
Body part "+views" (worldmodel): image coordinates from the learner's own two most orthogonal usual moves; for each axis of
each view, the axis of another view that always shows the same number (axis matching, learned from the heard sightings);
per view, each depth word's displacement. Control "+viewsrand": each axis matched to the other axis of the same source view.
Offline logic check (seeds 0-2, true books, all heard): fill 1.00 / control 0.01-0.07; depth 1.00 both.

Conditions (30 seeds; W = +rediscover+chain+wm+bind+wmlast):
  V = both@ascii@views W +views    Vr = ... +viewsrand    Vn = both@ascii@views W (no view part)
  body = both@ascii@views+rediscover+chain
Pre-registered:
  T1 "views fill", V - Vn and V - Vr: end, life (axis matching fills a view never seen).
  T2 "views depth", V - Vn: end, life (a move seen in three views is followed in each).
  T3 recorded: "views depth front" vs "moved" per condition; "views fill" / "views depth" for body.
  T4 other measures (story, chain, never-heard), V vs Vn: max |difference| (expected exactly 0 -- the part only answers
     questions with view labels).
  T5 (added after the seed-0 smoke, before any 30-seed run): "never-heard", V - Vn: end, life. The smoke showed the
     expectation in T4 wrong for never-heard (end 0.736 vs 0.774; Vr = V exactly): the world-model part also answers the
     heard three-view sentences during practice (a hidden cell is "asked"), which changes what the body learns -- the
     leaning cost of findings 75 / 77 suspected; this measures it.
What this assumes (CLAUDE.md 2026-10-09; places to change when extending): the view labels arrive as tags (which tokens are
labels is given); three orthogonal views of one 3-D state; image grid = the rooms' 5 x 5; coordinates need two orthogonal
usual moves (a world without them gives no answer); depth moves are one word = one step; no time axis.

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
V0 = "both@ascii@views" + W
CONDS = {"V": V0 + "+views", "Vr": V0 + "+viewsrand", "Vn": V0, "body": "both@ascii@views+rediscover+chain"}

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS.values() for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the three-view world (D17 v1, finding 137) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in range(n_seeds)])
    for k in ("views fill", "views depth", "views depth front", "views depth moved"):
        for c in CONDS:
            print(f"  {k:18} {c:4} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k, a, b in (("T1", "views fill", "V", "Vn"), ("T1", "views fill", "V", "Vr"), ("T2", "views depth", "V", "Vn")):
        d = g(a, k) - g(b, k)
        print(f"  {tag} {k}, {a} - {b}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("story", "chain", "never-heard"):
        d = g("V", k) - g("Vn", k)
        print(f"  T4 {k:11} max |V - Vn| {np.nanmax(np.abs(d)):.2e}")
    d = g("V", "never-heard") - g("Vn", "never-heard")
    print(f"  T5 never-heard, V - Vn: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
