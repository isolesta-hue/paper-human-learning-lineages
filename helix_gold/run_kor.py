"""Finding 123 (2026-10-07, the user's order: Korean first; stage-1 path step 1 -- sentence -> scene).

World "@kor" (data_ascii korean=True): the glyph world plus Korean particle stories in every motion room. Heard (2 per
motion word): ordered phrases "[OBJ ga] [LM eseo]" in a scrambled phrase order, the motion word last (SOV), then the scene
the learner sees (from(start), to(end)) -- the object moves from where the landmark was named. Test (3 per motion word):
"[OBJ ga] [LM eseo] [LM2 reul]" scrambled, motion word last, no scene; the end cell is asked. Separate random streams; the
particles are made after every other element -- the default worlds are unchanged (checked, seeds 0 and 5, also @hold).

Body part "+bind" (worldmodel.py): each particle is bound to the element right before it (the smallest piece of word
order); which particle marks the start is learned from the learner's own ordered record of heard stories; the start is the
bound noun's naming place; the motion books then roll (stage A, finding 120). Control "+bindrand": each particle bound to
a random noun of the sentence. Offline check (seed 0, untrained, all heard): bind 318 / 318, bindrand 134 / 318, the start
particle found = "eseo".

Conditions (30 seeds): both@ascii@kor+rediscover+chain+wm+bind, ...+wm+bindrand, ...+wm (no binding).
Pre-registered:
  K1 story accuracy, bind - no binding: end, life (sessions with stories).
  K2 story accuracy, bind - bindrand: end, life.
  K3 other measures (never-heard, chain): bind vs no binding, max |difference| and end sign tests (expected ~0).
Finding 125 (added 2026-10-07 after finding 124's voting probe, before any 30-seed run): the 4th condition
...+wm+bind+wmlast -- the decided world model speaks last (run_schema3.complete). Pre-registered:
  K4 story accuracy, wmlast - bind: end, life.
  K5 chain accuracy (all rooms), wmlast - bind: end, life.
  K6 never-heard, wmlast - bind: max |difference| and end sign test (expected ~0: the world model does not speak there).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

B = "both@ascii@kor+rediscover+chain+wm"
CONDS = (B + "+bind", B + "+bindrand", B, B + "+bind+wmlast")


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
    print(f"=== {n_seeds} seeds; Korean particle stories -> scenes (finding 123) ===")
    g = lambda c, k: np.array([[st.get(k, np.nan) for st in out[(sd, c)]] for sd in range(n_seeds)])
    bind, rand, none, last = CONDS
    for c in CONDS:
        print(f"  story {c:46} " + " ".join("  -  " if np.isnan(v) else f"{v:.3f}" for v in np.nanmean(g(c, 'story'), 0)))
    for tag, other in (("K1 bind - no binding", none), ("K2 bind - bindrand", rand)):
        d = g(bind, "story") - g(other, "story")
        print(f"  {tag}: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    for k in ("never-heard", "chain"):
        d = g(bind, k) - g(none, k)
        print(f"  K3 {k}: max |bind - none| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
    for tag, k in (("K4 story", "story"), ("K5 chain", "chain")):
        d = g(last, k) - g(bind, k)
        print(f"  {tag} wmlast - bind: end {sgn(d[:, -1])} | life {sgn(np.nanmean(d, 1))}")
    d = g(last, "never-heard") - g(bind, "never-heard")
    print(f"  K6 never-heard: max |wmlast - bind| {np.nanmax(np.abs(d)):.2e} | end {sgn(d[:, -1])}")
