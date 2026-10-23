"""The leaning test (2026-10-10; the user chose it first among the waiting-time tasks). Finding 141. Written and fixed BEFORE
any run of the new conditions.

Seen in the seed-0 smokes of findings 137-140: the never-heard score of a body with a world-model part differs from the same
body without it -- lower in the three-view and viewpoint worlds (0.736 vs 0.774; 0.697 vs 0.721), higher in the goal world
(0.726 vs 0.699), exactly equal in the self world (0.742 x 3). The suspected route: where the heard sentences have the shape
of the part's questions, the part answers them DURING PRACTICE (it is added to the answer the loss is taken on), so the body
learns those sentences differently; in the self world no heard sentence has the question word, the part never answers in
practice, and the body is unchanged. "+wmtest" (run_schema3): the world model answers only when no gradient is built (tests
and the learner's own no-gradient decisions), never in practice.

Conditions (30 seeds; the cached conditions of findings 137-139 are reused):
  views:  V = both@ascii@views W +views,  Vn = ... W,  Vt = V +wmtest
  viewpt: P = both@ascii@viewpt W +views+vturn,  Pn = ... W,  Pt = P +wmtest
  goal:   G = both@ascii@goal W +goal,  Gn = ... W,  Gt = G +wmtest     (W = +rediscover+chain+wm+bind+wmlast)
SEED-0 SMOKE OF "+wmtest" (2026-10-10): never-heard identical to the part body (views 0.736 = 0.736, viewpt 0.697 = 0.697)
-- the practice passes take SPARSE bags, where the world model is silent anyway (run_schema3: "calibration acts when answering,
not on the habit's training"): the suspected route is wrong and "+wmtest" is a no-op by construction; its 30-seed run is
dropped (it would be identical). NEW route (fixed here before any run of it): the learner's own no-gradient choices inside a
session -- above all the setter, which picks the heard sentences the learner gets WRONG for extra practice: where the part
answers a heard view sentence, the body's mistake is hidden and the sentence is not practised again. "+wmmeasure": the part
is silent through each whole session (settings, options, setter, practice) and answers only when measured. The conditions
below use "+wmmeasure" in place of "+wmtest" (X_t = X +wmmeasure).
SEED-0 SMOKE OF THE FIRST "+wmmeasure" (2026-10-10, views 0.735 vs part 0.736 / none 0.774; viewpt 0.697 = part): the
silence ended right after the session, BEFORE the method layer's forgetting check (me.drop: heard sentences with a position
hidden, per room -- the part fills hidden view cells, so a view room looks "not forgotten" and the next session's review
shares change). Fixed (same route, its last leak closed): the silence now ends after that check, just before measuring.
Pre-registered:
  X1 per world, "never-heard", X_t - X_n: end, life (expected ~0 if the route is right: the leaning disappears).
  X2 per world, "never-heard", X_t - X: end, life (the size of the leaning effect, either sign).
  X3 per world, the part's own measure (views fill / viewpt turn / goal next), X_t - X: end (expected ~0 -- tests are
     answered the same; any loss would mean the part also helped the body learn its own items).
SECOND "+wmmeasure" SMOKE AND ITS TRACE (E14, 2026-10-10): views 0.739 (none 0.774, part 0.736), viewpt 0.729, goal 0.693. The
hidden-state audit (statediff_voy.py) found every body and loop value equal to the no-part life up to session 3's practice;
the world-model call probe (probe_wmcalls*.py) found 0 calls inside sessions of the "+wmmeasure" life and 38 in the no-part
life: the BASE world model (+wm, present in every condition here) answers the setter's right/wrong check (integrated.py
setter_items), so "+wmmeasure" silences more than the part, and X1 as written compares a silenced life with a speaking one.
REVISED (fixed here before any multi-seed run; the seed-0 smoke check_e14.py runs Vnt): each world gets the matched control
X_nt = X_n +wmmeasure (the no-part body, its world model silent through each session too).
  X1' per world, "never-heard", X_t - X_nt: end, life -- the part's own leaning once both bodies decide alone (expected ~0;
      the part still answers at measuring time, so items of its own kind may differ).
  X4  per world, "never-heard", X_nt - X_n: end, life -- the base world model's help in the learner's own decisions (D21's
      question: is self-assessment through the body alone better or worse for the body? either sign; the seed-0 views smoke
      suggests worse: fewer sentences picked again).
  X1 and X2, X3 as before (X1 kept as recorded, now known to be unmatched).
What this assumes: the route is the practice pass only (no-gradient decisions in a session -- review choice, rediscovery --
still see the part); three worlds; the comparison is within each world.
"""
import sys

import numpy as np

import run_long as RL
from run_count import sgn

W = "+rediscover+chain+wm+bind+wmlast"
WORLDS = {
    "views": ("both@ascii@views" + W + "+views", "both@ascii@views" + W, "views fill"),
    "viewpt": ("both@ascii@viewpt" + W + "+views+vturn", "both@ascii@viewpt" + W, "viewpt turn"),
    "goal": ("both@ascii@goal" + W + "+goal", "both@ascii@goal" + W, "goal next"),
}

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    conds = [c for x, xn, _ in WORLDS.values() for c in (x, xn, x + "+wmmeasure", xn + "+wmmeasure")]
    tasks = [(sd, c) for c in conds for sd in range(n_seeds)]
    out = dict(zip(tasks, RL.map_lives(tasks, workers)))      # D57: a dead worker fails loudly; chunksize 1
    print(f"=== {n_seeds} seeds; the leaning test (finding 141) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    for name, (x, xn, own) in WORLDS.items():
        xt, xnt = x + "+wmmeasure", xn + "+wmmeasure"
        for tag, c in (("part", x), ("none", xn), ("test-only", xt), ("none-t", xnt)):
            print(f"  {name:6} {tag:9} never-heard " + " ".join(f"{v:.3f}" for v in np.nanmean(g(c, "never-heard"), 0)))
        for lab, a, b in (("X1 t - n", xt, xn), ("X1' t - nt", xt, xnt), ("X2 t - part", xt, x), ("X4 nt - n", xnt, xn)):
            d = g(a, "never-heard") - g(b, "never-heard")
            print(f"  {name:6} {lab:11} never-heard: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
        d = g(xt, own) - g(x, own)
        print(f"  {name:6} X3 {own}, t - part: end {sgn(d[:, -1])}")
