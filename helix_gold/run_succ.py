"""Stage-1 path step 4+ (2026-10-09; the successor rule -- children who know "one, two, three" learn that the next word in
the counting list names one more; Gelman & Gallistel 1978, Sarnecka & Carey 2008; ledger D1a). Finding 134.

World "@succ" (data_ascii succ=True = the "@count" world plus): the counting list is heard in every room as a chant, "HANA
DUL SET NET" (number words only, in order); counts are heard with their words only up to 3, as in "@count". Test "count
four": four things put into a container (the fourth borrowed from the previous room), "[C ane] MYEOT" -- NET was never
heard with a count. Violation-of-expectation pairs "voe four" (NET / SET). Separate random stream; the other worlds
unchanged (checked, seeds 0, 3, 5, nine worlds).

Body part "+succ" (worldmodel.py): the chant gives each number word its next word (majority); a count with no word heard for
it is named by the next word after the word for one fewer. Control "+succrev": the chant read backwards (the next word of
SET is DUL) -- the order of the chant, not its mere presence, should carry the answer.

Conditions (30 seeds; B = both@ascii@succ+rediscover+chain+wm+bind+wmlast+track+count): B+"+succ", B+"+succrev", B.
Pre-registered:
  S1 "count four", succ - B: end, life.
  S2 "count four", succ - succrev: end, life.
  S3 violation of expectation "voe four" ("looks longer"): per condition, end; succ - B: end, life.
  S4 other measures (count put, count take, voe number, contain in, multi, story, never-heard), succ vs B: max |difference|
     (expected exactly 0 -- the rule only names counts that have no heard word).

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

B = "both@ascii@succ+rediscover+chain+wm+bind+wmlast+track+count"
CONDS = (B + "+succ", B + "+succrev", B)

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the successor rule (stage-1 step 4+, finding 134) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    succ, rev, none = CONDS
    for k in ("count four", "voe four", "count put"):
        for c in CONDS:
            print(f"  {k:11} {c[len(B):] or '(none)':9} " +
                  " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, other in (("S1 succ - none", none), ("S2 succ - succrev", rev)):
        d = g(succ, "count four") - g(other, "count four")
        print(f"  {tag}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for c in CONDS:
        print(f"  S3 voe four {c[len(B):] or '(none)':9} end {np.nanmean(g(c, 'voe four')[:, -1]):.3f}")
    d = g(succ, "voe four") - g(none, "voe four")
    print(f"  S3 voe four, succ - none: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("count put", "count take", "voe number", "contain in", "multi", "story", "never-heard"):
        d = g(succ, k) - g(none, k)
        print(f"  S4 {k:12} max |succ - none| {np.nanmax(np.abs(d)):.2e}")
