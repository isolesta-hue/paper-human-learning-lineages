"""D22 (2026-10-10; frames vs reality -- the user: people tell tales, films and games from reality and still understand them;
"a world model closest to reality" as the anchor, with several tags at once (its tag, the 'me' vector, ...) so that frames
do not mix into hallucination; decision "응, D22로 적어두고 그 순서대로 진행해줘"). Written and fixed BEFORE any multi-seed run.

World "@tale" (data_ascii tale=True = "@grav" plus told tales; old worlds identical at seeds 0, 3, 5, 48/48): in every room
not upside down, tales "YENNAL [T ga] at(q) GO [A ga] from(p) falls GO [A ga] EODI to(end)" -- inside a tale only "falls" goes
UP (one rule broken, the room not turned); T = the room's tale-only thing; 3 heard per room, in half of the rooms one without
the marker. v1b "rooms known only from tales": 2 tales of each phase-1 room c are heard in room c - 1's session (always
earlier), in half of these rooms without the marker; before c is seen its real gravity questions are asked.
Pilots (seed 0, no part): the combined answer is not contaminated where reality was met (the world model's books take the
most counted move -- a reality anchor by majority), the tale frame is ignored (0.000), the body alone learns it a little
(0.27) and knows little gravity (0.2-0.3).
Parts (worldmodel.py, tale_mode): "+tk1" one tag -- marked moments go to a tale book, marked questions are answered in the
tale frame (ANCHOR + EXCEPTION: reality's one-step move times the tales' keep / reverse sign); "+tkA" anchor + several tags --
as +tk1, and an unmarked moment is also taken as a tale when its move runs against reality's usual move of its word while
its room is not turned (the room's other words keep their usual direction); "+tkig" -- marked moments dropped, marked
questions answered as real. Body-logic probe (probe_tale_logic.py, the world model alone, seeds 0-2): previews marked
Tn 0.00 / ig 1.00 / tk1 1.00 / tkA 1.00; previews unmarked 0.00 / 0.00 / 0.00 / 1.00; tale questions 0 / 0 / 1.00 / 1.00;
real rooms (clean, drop, turned) 1.00 everywhere; the moments tkA takes as tales = exactly the unmarked tales (12/15, 13/17,
11/15).

Conditions (30 seeds; B = both@ascii@tale+rediscover+chain+wm+bind+wmlast+track+grav): B (Tn), B+"+tkig", B+"+tk1", B+"+tkA".
Pre-registered (life = mean over sessions; "preview" values exist only in the sessions before the room is seen -> life only):
  T1 preview UNMARKED, combined answer, tkA - tk1: life (the user's hypothesis: several anchors stop the hallucination).
  T2 preview MARKED, combined, tk1 - Tn: life (one tag is enough when the tag is there).
  T3 tale questions ("tale in"), combined, tk1 - Tn and tkA - Tn: end, life (understanding inside the frame).
  T4 real gravity in the turned rooms ("tale real turned"), tkA - tk1: end, max |diff| (expected ~0: no false tale).
  T5 real gravity, clean / drop rooms, combined: each part - Tn, max |diff| (expected ~0: reality already held by majority).
  T6 other measures (never-heard, story, transfer): each part - Tn, max |diff| and end sign test (the setter sees the world
     model's answers to the heard marked tales -- D21 -- so never-heard may move; either sign).
  T7 recorded: the body alone (wm silent while measuring) on tale questions and previews; "tale flagged" for tkA.
What this assumes: 2D grid, one broken rule ("falls" reversed) and one tale frame for all tales (keep / reverse of the anchor,
no per-tale frames), the marker a single token at the start, tales share rooms and things with reality, the "turned" test
needs the room's other words (a real room met only through one word cannot be told from a tale), no 'me' vector yet (D22 v2:
contingency -- does it respond to my action -- as a third anchor), the anchor = reality's most counted move (one-step moments
are the most heard).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_count import sgn

B = "both@ascii@tale+rediscover+chain+wm+bind+wmlast+track+grav"
CONDS = {"Tn": B, "Tig": B + "+tkig", "T1": B + "+tk1", "TA": B + "+tkA"}
KEYS = ("tale preview unmarked", "tale preview marked", "tale in", "tale real turned", "tale real clean", "tale real drop",
        "tale transfer", "never-heard", "story", "tale in body", "tale preview unmarked body", "tale preview marked body",
        "tale flagged")

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS.values() for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; D22 frames vs reality ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    for k in KEYS:
        for tag, c in CONDS.items():
            v = np.nanmean(g(c, k), 0)
            print(f"  {k:28} {tag:4} " + " ".join("  -  " if np.isnan(x) else f"{x:.3f}" for x in v))
    life = lambda c, k: np.nanmean(g(c, k), 1)
    end = lambda c, k: g(c, k)[:, -1]
    C = CONDS
    print("  T1 preview unmarked, tkA - tk1: life", sgn(life(C["TA"], "tale preview unmarked") - life(C["T1"], "tale preview unmarked")))
    print("  T2 preview marked, tk1 - Tn: life", sgn(life(C["T1"], "tale preview marked") - life(C["Tn"], "tale preview marked")))
    for a in ("T1", "TA"):
        d = (end(C[a], "tale in") - end(C["Tn"], "tale in"), life(C[a], "tale in") - life(C["Tn"], "tale in"))
        print(f"  T3 tale in, {a} - Tn: end {sgn(d[0])} | life {sgn(d[1])}")
    d = g(C["TA"], "tale real turned") - g(C["T1"], "tale real turned")
    print(f"  T4 real turned, tkA - tk1: end {sgn(d[:, -1])} | max |diff| {np.nanmax(np.abs(d)):.3f}")
    for a in ("Tig", "T1", "TA"):
        for k in ("tale real clean", "tale real drop", "tale transfer", "never-heard", "story"):
            d = g(C[a], k) - g(C["Tn"], k)
            tagk = "T5" if k.startswith("tale real") else "T6"
            print(f"  {tagk} {k:16} {a} - Tn: max |diff| {np.nanmax(np.abs(d)):.3f} | end {sgn(d[:, -1])}")
