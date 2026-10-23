"""Finding 119 (2026-10-06, the user's order of 2026-10-05, step 1: the learner scores its own self-experiment's predictions
and sets from them how far to trust its eye -- the first material for the surprise / confusion modes).

Findings 115-118: the world-shaped self-experiment (+tune) wins in the harder world (end +0.050, 25:5) but in the plain world
drives the whole-sentence memory weight to ~0.3 late in life and loses its lead; fixing the eye's questions did not help
(118: a partnerless background ranks better but chooses even lower). So instead of fixing the questions, grade the eye's
choices against the world: +graded -- at each method session, before choosing anew, the weights the eye chose LAST
session are graded on THIS session's held-out sentences of the new categories (fresh material the eye never saw when it
chose; the judge's leave-one-out). Trust per part = the share of graded choices not worse than weight 1, with the uniform
prior ((hits + 1) / (n + 2), 0.5 before any evidence -- declared representative value). A part whose trust is below one half
answers with weight 1; the eye keeps choosing and being graded, so trust can come back. The outside test is never seen by
the learner; the eye's latest choice is also scored on it, for checking only ("out gain").

Conditions (30 seeds): both+rediscover+tune+graded, both+rediscover@share+tune+graded; cached: +tune, knob-only, content.
Pre-registered:
  T1 graded vs tune: end, life, late widening of the completion gap (s13-16 - s6-8) -- per world.
  T2 graded vs knob-only (no tune): end, life, late widening -- per world.
  T3 the grade as an eye: per part, the agreement of the grade's sign (>= 0) with the sign of the outside gain of the same
     choice (out gain, made the session before), sessions 3-16; the trust and the share of sessions answered at weight 1
     (backed off), sessions 9-16.
Expected if the idea is right: plain world -- the chunk trust falls late, the lead of +tune is kept to the end (T1 end > 0);
harder world -- trust stays high and the +tune gain is kept (T1 ~ 0, T2 > 0).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

WORLDS = {"plain": ("both+rediscover+tune+graded", "both+rediscover+tune", "both+rediscover", "content"),
          "share": ("both+rediscover@share+tune+graded", "both+rediscover@share+tune", "both+rediscover@share",
                    "content@share")}


def sgn(x):
    x = np.asarray(x)
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for quad in WORLDS.values() for c in quad for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the eye's choices graded on the next session's fresh sentences, trust sets the weights ===")
    for world, (g, t, b, f) in WORLDS.items():
        nh = {c: np.array([[st["never-heard"] for st in out[(sd, c)]] for sd in range(n_seeds)]) for c in (g, t, b, f)}
        print(f"\n##### {world}")
        for c in (g, t, b, f):
            print(f"  {c:36} " + " ".join(f"{v:.3f}" for v in nh[c].mean(0)))
        late = {c: (nh[c] - nh[f])[:, 12:16].mean(1) - (nh[c] - nh[f])[:, 5:8].mean(1) for c in (g, t, b)}
        d = nh[g] - nh[t]
        print(f"  T1 graded - tune: end {sgn(d[:, -1])} | life {sgn(d.mean(1))} | late widening {sgn(late[g] - late[t])}")
        d = nh[g] - nh[b]
        print(f"  T2 graded - no tune: end {sgn(d[:, -1])} | life {sgn(d.mean(1))} | late widening {sgn(late[g] - late[b])}")
        for part in ("chunk", "assoc"):
            agree, n = 0, 0
            for sd in range(n_seeds):
                L_ = out[(sd, g)]
                for i in range(2, len(L_)):
                    gr, og = L_[i].get(f"grade {part}", np.nan), L_[i - 1].get(f"out gain {part}", np.nan)
                    if not (np.isnan(gr) or np.isnan(og)):
                        agree += int((gr >= 0) == (og >= 0))
                        n += 1
            tr = np.nanmean([out[(sd, g)][i].get(f"trust {part}", np.nan) for sd in range(n_seeds) for i in range(8, 16)])
            eye = [out[(sd, g)][i].get(f"eye {part}", np.nan) for sd in range(n_seeds) for i in range(8, 16)]
            used = [out[(sd, g)][i].get("tune chunk" if part == "chunk" else "assoc w", np.nan)
                    for sd in range(n_seeds) for i in range(8, 16)]
            back = np.nanmean([float(e != 1.0 and abs(u - 1.0) < 1e-9) for e, u in zip(eye, used)
                               if not (np.isnan(e) or np.isnan(u))]) if part == "assoc" else np.nan
            print(f"  T3 {part}: grade agrees with outside {agree}/{n} ({agree / max(n, 1):.2f}) | trust s9-16 {tr:.2f} | "
                  f"eye's weight s9-16 {np.nanmean(eye):.2f} | used s9-16 {np.nanmean(used):.2f}"
                  + (f" | backed off {back:.2f}" if part == "assoc" else ""))
