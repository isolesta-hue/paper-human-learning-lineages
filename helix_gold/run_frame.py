"""Stage-1 path step 4+ (2026-10-08; reference frames -- the user: "the centre of a baby's world rules is the ground; with
the apple as the centre, falling and flying up cannot be told apart"; decision "그렇게 적어주고 시험을 진행해줘").

World "@frame" (data_ascii frame=True = "@hold" with turned rooms): the late rooms are turned a quarter (the ground on a side
wall), a half (the ceiling -- the old upside-down rooms) or three quarters, a separate random stream; every motion word turns
with the room; one motion word is never heard in each turned room (as @hold) and its chain questions use it. The default
and the other worlds are unchanged (checked, seeds 0, 3, 5, six worlds).

Body part "+wmframe" (worldmodel.py): stage B's keep / reverse (finding 122, "+wmctx") becomes the room's FRAME -- the 8
symmetries of the square, built on the learner's two most nearly orthogonal usual books as axes; the room's frame is the one
that best maps the usual books of the words heard there onto the room's own books; an unheard word's usual move goes through
it. The library is given; which frame a room has is learned from its surprises.

Conditions (30 seeds; B = both@ascii@frame+rediscover+chain+wm): B+"+wmframe", B+"+wmctx" (stage B), B (stage A).
Pre-registered:
  F1 chain accuracy in the quarter-turned rooms, frame - ctx: end, life (stage B's keep / reverse cannot express a turn).
  F2 chain accuracy in the half-turned rooms, frame - ctx (expected ~0: both can express it).
  F3 chain accuracy, all turned rooms, frame - none: end, life.
  F4 never-heard, frame vs none, max |difference| and end sign test (expected ~0: the frame acts on chain questions).

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

B = "both@ascii@frame+rediscover+chain+wm"
CONDS = (B + "+wmframe", B + "+wmctx", B)


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
    print(f"=== {n_seeds} seeds; the room's frame (stage-1 step 4+, reference frames) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    frame, ctx, none = CONDS
    for k in ("chain", "chain quarter", "chain half", "chain flipped"):
        for c in CONDS:
            print(f"  {k:14} {c[len(B):] or '(stage A)':10} " +
                  " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, k), 0)))
    for tag, k, other in (("F1 quarter, frame - ctx", "chain quarter", ctx), ("F2 half, frame - ctx", "chain half", ctx),
                          ("F3 turned rooms, frame - none", "chain flipped", none)):
        d = g(frame, k) - g(other, k)
        print(f"  {tag}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    d = g(frame, "never-heard") - g(none, "never-heard")
    print(f"  F4 never-heard: max |frame - none| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
