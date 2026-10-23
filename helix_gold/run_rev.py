"""Stage-1 path step 3a (2026-10-08, the user: can the logic that understands one side from the other run BACKWARDS --
shown how a thing moves, say it in words? Decision: "그렇게 해줘", right after step 2).

Items "describe" (data_ascii, every glyph world; a separate random stream, kept apart from the tests -- every existing item
unchanged, checked seeds 0, 3, 5 in all four worlds): an object seen at a start cell and an end cell (from(a), to(b)) plus
a background picture; the motion word is asked (in a room no two motion words make the same move). The body already fills
in any hidden word, so it may describe without help; the world model so far only answered "where does it end".
Body part "+wmrev" (worldmodel.py): the motion books read backwards -- the seen move (to - from - role shift) is looked up
among the words' books (the room's own entry, else the most counted); the one word that lands within half a cell gap gets
one decided book's evidence. No trust check (the books were built from the moments a check would use).

Conditions (30 seeds; B = both@ascii+rediscover+chain+wm+wmlast): B+"+wmrev", B.
Pre-registered:
  R1 describe accuracy, wmrev - B: end, life.
  R2 describe accuracy in the upside-down rooms (phase 2), wmrev - B: end, life (the room's own books must be used).
  R3 the body alone (B): describe accuracy by session, recorded -- does the bag-of-words body already run backwards?
  R4 other measures (chain, never-heard): wmrev vs B, max |difference| and end sign test (expected ~0; the part can fire
     while learning heard moments whose motion word is the hidden element, so not exactly 0).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

B = "both@ascii+rediscover+chain+wm+wmlast"
CONDS = (B + "+wmrev", B)


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
    print(f"=== {n_seeds} seeds; scene -> motion word, the books read backwards (stage-1 step 3a) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    rev, base = CONDS
    for k in ("describe", "describe flipped", "describe wm fired", "describe wm right"):
        for c in CONDS:
            print(f"  {k:18} {c[len(B):] or '(body only)':12} " +
                  " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k in (("R1 describe", "describe"), ("R2 describe flipped", "describe flipped")):
        d = g(rev, k) - g(base, k)
        print(f"  {tag}, wmrev - body: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    print("  R3 body only, describe by session: " + " ".join(f"{v:.3f}" for v in np.nanmean(g(base, 'describe'), 0)))
    for k in ("chain", "never-heard"):
        d = g(rev, k) - g(base, k)
        print(f"  R4 {k}: max |wmrev - body| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
