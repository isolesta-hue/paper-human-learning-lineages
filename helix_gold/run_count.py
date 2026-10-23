"""Stage-1 path step 4+ (2026-10-09; core knowledge: small numbers, 1-3 -- as Wynn 1992's 1 + 1 behind a screen; ledger D1).

World "@count" (data_ascii count=True = the "@contain" world plus): things are put into a container one by one ("[X ga]
at(q) [C ane]") and sometimes one is taken out ("[X ga] at(r)"); "[C ane] MYEOT" (how many in C?) is answered by HANA /
DUL / SET. Heard 2 per final count (1, 2, 3; no taking out) with the number word; test "count put" and "count take" (the
taking-out is never heard -- the count must follow the tracked objects); violation-of-expectation pairs "voe number"
(right count / one more or one less). Separate random streams; MYEOT and the number words are made last; the other worlds
unchanged (checked, seeds 0, 3, 5, eight worlds).

Body part "+count" (worldmodel.py): the tracking keeps each object's latest relation (set when put in relation to C,
cleared when placed elsewhere); the count = the objects in that relation to C at C's cell; which number word names each
count is learned from heard stories (majority). Control "+countrand": each count read with another count's word.

Conditions (30 seeds; B = both@ascii@count+rediscover+chain+wm+bind+wmlast+track): B+"+count", B+"+countrand", B.
Pre-registered:
  N1 "count put", count - B: end, life.
  N2 "count take" (never heard), count - B: end, life.
  N3 count - countrand on put and take: end (the learned words carry the counts).
  N4 violation of expectation "voe number" ("looks longer"): per condition, end; count - B, life.
  N5 recorded: the learned table right (1 -> HANA, 2 -> DUL, 3 -> SET); other measures (contain in, multi, story,
     never-heard) count vs B, max |difference| (expected ~0).

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

B = "both@ascii@count+rediscover+chain+wm+bind+wmlast+track"
CONDS = (B + "+count", B + "+countrand", B)


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
    print(f"=== {n_seeds} seeds; small numbers (stage-1 step 4+) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    cnt, rnd, none = CONDS
    for k in ("count put", "count take", "voe number", "count table right"):
        for c in CONDS:
            print(f"  {k:17} {c[len(B):] or '(none)':11} " +
                  " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k in (("N1 put", "count put"), ("N2 take", "count take")):
        d = g(cnt, k) - g(none, k)
        print(f"  {tag}, count - none: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("count put", "count take"):
        d = g(cnt, k) - g(rnd, k)
        print(f"  N3 {k}, count - countrand: end {sgn(d[:, -1])}")
    print("  N4 voe number looks longer, end: " + " | ".join(
        f"{c[len(B):] or 'none'} {np.nanmean(g(c, 'voe number')[:, -1]):.3f}" for c in CONDS) +
          f" | count - none, life {sgn(np.nanmean(g(cnt, 'voe number') - g(none, 'voe number'), 1))}")
    print(f"  N5 table right (end): {np.nanmean(g(cnt, 'count table right')[:, -1]):.3f}")
    for k in ("contain in", "multi", "story", "never-heard"):
        d = g(cnt, k) - g(none, k)
        print(f"  N5 {k}: max |count - none| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
