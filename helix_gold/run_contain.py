"""Stage-1 path step 4+ (2026-10-08; core knowledge: containment -- what is inside a container moves with it).

World "@contain" (data_ascii contain=True = the "@multi" world plus): two relation particles, ANE (inside) and YEOPE
(beside). Stories "[C ga] at(q) GO [A ga] at(r) [C REL] GO [C ga] VERB GO [A ga] EODI": inside (r = q) -- A moves with C;
beside (r a neighbouring cell) -- A stays. Heard 2 per relation with A's end scene; test "contain in" / "contain by";
violation-of-expectation pairs "voe inside" (carried / left behind) and "voe beside" (stays / dragged along). Separate
random streams; the other worlds unchanged (checked, seeds 0, 3, 5, seven worlds).

Body part "+contain" (worldmodel.py): particles other than the start particle and the mover particle are relation
particles; whether a relation CARRIES is learned per particle from heard stories (each read twice -- carried / not -- the
reading that lands on the scene votes); when a carrier moves, what it carries moves the same way. Without it nothing is
carried; control "+containall": every relation carries (over-generalisation).

Conditions (30 seeds; B = both@ascii@contain+rediscover+chain+wm+bind+wmlast+track): B+"+contain", B+"+containall", B.
Pre-registered:
  C1 "contain in", contain - B: end, life.
  C2 "contain by", contain - containall: end, life.
  C3 violation of expectation (inside, beside; share "looks longer"): contain vs B and vs containall, end.
  C4 recorded: the learned table right (ANE carries, YEOPE does not).
  C5 other measures (multi, story, never-heard): contain vs B, max |difference|, end sign test (expected ~0).

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

B = "both@ascii@contain+rediscover+chain+wm+bind+wmlast+track"
CONDS = (B + "+contain", B + "+containall", B)


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
    print(f"=== {n_seeds} seeds; containment (stage-1 step 4+) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    cont, call, none = CONDS
    for k in ("contain in", "contain by", "voe inside", "voe beside", "contain table right"):
        for c in CONDS:
            print(f"  {k:19} {c[len(B):] or '(none)':12} " +
                  " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k, other in (("C1 in, contain - none", "contain in", none), ("C2 by, contain - containall", "contain by", call)):
        d = g(cont, k) - g(other, k)
        print(f"  {tag}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for kv in ("inside", "beside"):
        print(f"  C3 voe {kv} looks longer, end: " + " | ".join(
            f"{c[len(B):] or 'none'} {np.nanmean(g(c, 'voe ' + kv)[:, -1]):.3f}" for c in CONDS))
    print(f"  C4 table right (end): {np.nanmean(g(cont, 'contain table right')[:, -1]):.3f}")
    for k in ("multi", "story", "never-heard"):
        d = g(cont, k) - g(none, k)
        print(f"  C5 {k}: max |contain - none| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
