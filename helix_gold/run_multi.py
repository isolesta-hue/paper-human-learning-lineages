"""Stage-1 path step 2 (2026-10-08, the user's order: Korean first; after finding 123's sentence -> scene): multi-moment
stories -- STATE OVER TIME.

World "@multi" (data_ascii multi=True; the "@kor" world plus): Korean stories of several moments joined by the connective
GO ("-go", and then), ending with the question "[OBJ ga] EODI" (where is OBJ now?). Kinds (TEST each per room, heard 2 each
with the asked object's end scene): "same" -- one object moves twice (the second moment has no landmark: it goes on from
where it is); "two" -- the asked object moves, then ANOTHER object moves from its own landmark (a distractor motion);
"three" -- A, then B, then A again. The phrase order inside a moment is free. Separate random streams, GO / EODI made last:
the default, "@kor" and "@hold" worlds are unchanged (checked, seeds 0, 3, 5).

Body part "+track" (worldmodel.py): cut the story into moments at GO; in each moment the noun bound to the start particle
(learned, finding 123) is the landmark, the noun bound to another particle the mover; one state per object (a landmark
places the mover there; no landmark: from where it is now); the moment's motion books roll the mover only. The recipe is
given, its trust is earned: it speaks only when it has predicted the heard stories' scenes right more often than not.
Control "+trackone": one state for the whole story (a landmark resets it; every motion word moves it).
Without tracking, the finding-123 body (bind: the first landmark, all motion words) answers the stories.

Conditions (30 seeds; the finding-123/125 body B = both@ascii@multi+rediscover+chain+wm+bind+wmlast): B+"+track",
B+"+trackone", B.
Pre-registered:
  T1 multi-moment accuracy (all kinds), track - B: end, life (sessions with stories).
  T2 track - trackone: end, life.
  T3 per kind (same / two / three): end means of the three conditions; expected: B and trackone right on "same" only.
  T4 other measures (Korean one-moment stories, chain, never-heard): track vs B, max |difference| and end sign test
     (expected ~0; tracking can fire while learning heard stories whose scene is the hidden element, so not exactly 0).
Added 2026-10-08 before this runner first ran (stage-1 step 3c, record only -- run_long.tell_probe runs in every "@multi"
life and changes nothing the learner does): a sequence of pictures told as a multi-moment story.
  T5 telling accuracy per condition (book = the books read backwards; body = the body's motion word; first = control: the
     mover taken to be the first object listed), by kind, and the read-back rate (the told story read with step 2's
     tracking lands every object where the last picture shows it). Expected: book ~ high in every condition (telling uses
     the books, not the answering body); first right on "same" only.
Added 2026-10-08 before this runner first ran (step 4+, core knowledge: object permanence; record only -- the
violation-of-expectation probe in run_long reads the "voeperm" pairs of every "@multi" room):
  T6 "looks longer" share for "voe permanence" (A, unmentioned while B moves, is shown back at its landmark's place + its
     last move -- as if it had vanished and restarted) and "voe identity" (A shown where B is), per condition, end and life;
     expected: track high, bind-only and trackone lower.
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

B = "both@ascii@multi+rediscover+chain+wm+bind+wmlast"
CONDS = (B + "+track", B + "+trackone", B)


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
    print(f"=== {n_seeds} seeds; multi-moment Korean stories -- state over time (stage-1 step 2) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    track, one, none = CONDS
    for k in ("multi", "multi same", "multi two", "multi three", "multi wm fired", "multi trust"):
        for c in CONDS:
            print(f"  {k:15} {c[len(B):] or '(bind only)':12} " +
                  " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, other in (("T1 track - bind only", none), ("T2 track - trackone", one)):
        d = g(track, "multi") - g(other, "multi")
        print(f"  {tag}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for km in ("same", "two", "three"):
        print(f"  T3 {km:5}: " + " | ".join(f"{c[len(B):] or 'bind only'} {np.nanmean(g(c, 'multi ' + km)[:, -1]):.3f}"
                                         for c in CONDS))
    for k in ("tell book", "tell body", "tell first", "tell readback", "tell same book", "tell two book", "tell three book"):
        print(f"  T5 {k:16} " + " | ".join(f"{c[len(B):] or 'bind only'} {np.nanmean(g(c, k)[:, -1]):.3f}" for c in CONDS))
    for kv in ("permanence", "identity"):
        print(f"  T6 voe {kv:10} looks longer, end: " + " | ".join(
            f"{c[len(B):] or 'bind only'} {np.nanmean(g(c, 'voe ' + kv)[:, -1]):.3f}" for c in CONDS) +
              f" | track - bind only, life {sgn(np.nanmean(g(track, 'voe ' + kv) - g(none, 'voe ' + kv), 1))}")
    for k in ("story", "chain", "never-heard"):
        d = g(track, k) - g(none, k)
        print(f"  T4 {k}: max |track - bind only| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
