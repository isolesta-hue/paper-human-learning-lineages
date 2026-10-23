"""Stage-1 path step 4+ (2026-10-09; ledger D13 the number-line room + D15 scale alignment; the user's design C). Finding 136.
Written and fixed BEFORE the seed-0 smoke results were read.

Worlds (data_ascii numline=..., both = the "@kor" world plus a number line in the placement rooms -- rooms with no motion
moments): the number words (hana .. daseot) are named at the line's five cells; heard: Korean one-step stories on the line,
"[OBJ ga] [N eseo] pushed" + scene, n = 1, 2 (results up to 3); test "numline add": pushed 1-3 times to results 4, 5 (never
heard as results); test "numline sub": "pulled" 1-3 times -- a word NEVER heard on the line, only in the motion rooms.
  "@numrow" (A): the line IS the rooms' middle row (same cells, same spacing as every room).
  "@numown" (B): the line's own cells, half the rooms' spacing, in a plane of its own (a frame of a different scale).
Body part "+scale" (D15): a room whose own books are a scaled copy of the usual books gets the ratio from the learner's own
record (median over the words heard in the room of (local . usual) / (usual . usual)); a word never heard there is scaled by
it. Control "+scalerev": the inverse ratio. Body-only control: no world model (rote memory of heard stories).

Conditions (30 seeds; W = +rediscover+chain+wm+bind+wmlast):
  A  = both@ascii@numrow W     B  = both@ascii@numown W     Bs = B +scale     Br = B +scalerev
  body = both@ascii@numrow+rediscover+chain
Pre-registered:
  L1 "numline add", A - body: end, life (moves compose to results never heard).
  L2 "numline sub", A - B: end, life (shared coordinates carry the rooms' moves; a different scale does not).
  L3 "numline sub", Bs - B and Bs - Br: end, life (learned scale alignment, not merely a changed move).
  L4 "numline add", A vs B: end sign test (expected ~0 -- both learn the line's own "pushed").
  L5 recorded: "numline scale" per condition (expected ~1 in A, ~0.5 in B); "numline sub wm fired / right"; other measures
     (story, chain, never-heard) A vs B, Bs vs B: max |difference| (Bs vs B expected exactly 0 outside the line rooms).
What this assumes (CLAUDE.md, 2026-10-09; places to change when extending): numbers 1-5 fit the 5 x 5 rooms (6-8 waits for
the 9 x 9 rung); the scale part handles frames that differ by SIZE only (not by axis direction -- that is D17's axis
matching), one ratio per room; addition = repeating one motion word (no word for "plus k"); time is story order only.

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
A, B = "both@ascii@numrow" + W, "both@ascii@numown" + W
CONDS = {"A": A, "B": B, "Bs": B + "+scale", "Br": B + "+scalerev", "body": "both@ascii@numrow+rediscover+chain"}

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS.values() for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the number-line room + scale alignment (D13 / D15, finding 136) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, CONDS[c])]] for sd in range(n_seeds)])
    for k in ("numline add", "numline sub", "numline sub wm right", "numline scale"):
        for c in CONDS:
            print(f"  {k:21} {c:4} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k, a, b in (("L1", "numline add", "A", "body"), ("L2", "numline sub", "A", "B"),
                         ("L3", "numline sub", "Bs", "B"), ("L3", "numline sub", "Bs", "Br"), ("L4", "numline add", "A", "B")):
        d = g(a, k) - g(b, k)
        print(f"  {tag} {k}, {a} - {b}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("story", "chain", "never-heard"):
        for a, b in (("A", "B"), ("Bs", "B")):
            d = g(a, k) - g(b, k)
            print(f"  L5 {k:11} max |{a} - {b}| {np.nanmax(np.abs(d)):.2e}")
