"""Stage-1 path step 4+ (2026-10-08; core knowledge and infant-style tests, the user's decision; first item: gravity /
support).

World "@grav" (data_ascii grav=True = the "@multi" world plus): "falls" moves an object until it rests -- on the edge of the
world below it or on top of another object in its way; the other motion words move one cell. Stories "[B ga] at(q) GO
[A ga] from(p) VERB GO [A ga] EODI"; heard 2 per kind with the end scene; test "grav floor" (nothing in the way), "grav
stack" (B in the way: A rests on B), "grav step" (another word: one cell); violation-of-expectation pairs "voe float"
(rests / stops one cell down in mid-air) and "voe through" (rests on B / ends beyond B). Upside-down rooms fall upwards
(the room's book). Separate random streams; the other worlds unchanged (checked, seeds 0, 3, 5).

Body part "+grav" (worldmodel.py): a word may repeat its book move until the next move would leave the world or enter
another object's cell; WHICH words do so is learned per word from heard stories (each read once-only and until-blocked;
the reading that lands on the scene votes). Control "+gravall": every word repeats (over-generalisation). Without either:
one move (as before).

Conditions (30 seeds; B = both@ascii@grav+rediscover+chain+wm+bind+wmlast+track): B+"+grav", B+"+gravall", B.
Pre-registered:
  G1 falling accuracy (floor and stack), grav - B: end, life.
  G2 "grav step", grav - gravall: end, life (over-generalising hurts the one-cell words).
  G3 violation of expectation (float, through; share "looks longer"), grav - B: end, life.
  G4 recorded: the learned table -- "falls" repeats, the other words' share that repeat (expected 1 and 0).
  G5 other measures (multi, story, never-heard): grav vs B, max |difference|, end sign test (expected ~0).

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
from run_unified_continual import sign_p

B = "both@ascii@grav+rediscover+chain+wm+bind+wmlast+track"
CONDS = (B + "+grav", B + "+gravall", B)


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
    print(f"=== {n_seeds} seeds; gravity / support (stage-1 step 4+) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    grav, gall, none = CONDS
    for k in ("grav floor", "grav stack", "grav step", "voe float", "voe through", "grav falls repeats", "grav others repeat"):
        for c in CONDS:
            print(f"  {k:18} {c[len(B):] or '(once)':10} " +
                  " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    fall = lambda c: (g(c, "grav floor") + g(c, "grav stack")) / 2
    d = fall(grav) - fall(none)
    print(f"  G1 falling, grav - once: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    d = g(grav, "grav step") - g(gall, "grav step")
    print(f"  G2 step, grav - gravall: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for kv in ("float", "through"):
        d = g(grav, "voe " + kv) - g(none, "voe " + kv)
        print(f"  G3 voe {kv}, grav - once: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    print(f"  G4 falls repeats (end) {np.nanmean(g(grav, 'grav falls repeats')[:, -1]):.3f} | others repeat (end) "
          f"{np.nanmean(g(grav, 'grav others repeat')[:, -1]):.3f}")
    for k in ("multi", "story", "never-heard"):
        d = g(grav, k) - g(none, k)
        print(f"  G5 {k}: max |grav - once| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
