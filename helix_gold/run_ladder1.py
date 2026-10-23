"""Finding 111 (2026-10-03, the user's choice: (B) the world gets harder, the ladder ending at very easy real sentences;
rung order 1 -> 2 -> 3 -> 4, text world first): world ladder rung 1 -- words shared across topics.

The spiral principle: the completion criterion asks the self-modifying learner to pull AWAY from the fixed one, and since
finding 103 the gap stops growing after session 5 (findings 105-109: more knobs, self-made options -- a little). Hypothesis:
the world stopped giving the method layer something to fix. Rung 1 makes the world harder the way real text is: from the
second half on (sessions 9-16) half of the fillers are common words shared by all topics, with Zipf frequencies
(data_v3 share=True: 48 common words, P_SHARE 0.5 -- declared). New sentences now move connections that old sentences
use too. The schedule is fixed and the same for both learners (the completion criterion: the same data).

Conditions (30 seeds, cache_long_v2): content@share, both+rediscover@share; the plain-world lives (content,
both+rediscover) come from finding 103's cache.
Pre-registered:
  L1 completion in the harder world: gap = both+rediscover@share - content@share; end, widening (s13-16 - s2-4) and late
     widening (s13-16 - s6-8), sign tests.
  L2 the spiral test: late widening in the harder world minus late widening in the plain world, paired by seed (> 0 =
     a world that keeps getting harder keeps the gap growing). Also with the headroom-normalised gap.
  L3 how often the method layer still changes a setting in sessions 9-16, harder vs plain world (does it find more to fix?).
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

CONDS = ("content@share", "both+rediscover@share", "content", "both+rediscover")


def sgn(x):
    x = np.asarray(x)
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.3f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0][CONDS[0]])
    print(f"=== {n_seeds} seeds; world ladder rung 1: common words from session 9 on (text world) ===")
    for key in ("never-heard", "once all", "older", "changed"):
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:24} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    gaps = {}
    for tag, (a, b) in {"share": ("both+rediscover@share", "content@share"), "plain": ("both+rediscover", "content")}.items():
        g = np.array([[r[a][i]["never-heard"] - r[b][i]["never-heard"] for i in range(n)] for r in res])
        head = np.array([[1.0 - r[b][i]["never-heard"] for i in range(n)] for r in res])
        gaps[tag] = (g, g / np.maximum(head, 1e-9))
        print(f"\n{tag} completion by session " + " ".join(f"{v:+.3f}" for v in g.mean(0)))
        print(f"   end {sgn(g[:, -1])} | widening s13-16 - s2-4 {sgn(g[:, 12:16].mean(1) - g[:, 1:4].mean(1))}"
              f" | late widening s13-16 - s6-8 {sgn(g[:, 12:16].mean(1) - g[:, 5:8].mean(1))}")
    print()
    for j, name in ((0, "gap"), (1, "headroom-normalised gap")):
        late = {t: gaps[t][j][:, 12:16].mean(1) - gaps[t][j][:, 5:8].mean(1) for t in gaps}
        print(f"L2 {name}: late widening share - plain {sgn(late['share'] - late['plain'])}")
    ch = {c: np.array([np.nanmean([r[c][i]["changed"] for i in range(8, n)]) for r in res])
          for c in ("both+rediscover@share", "both+rediscover")}
    print(f"L3 changed in sessions 9-16: share {ch['both+rediscover@share'].mean():.3f}, plain {ch['both+rediscover'].mean():.3f}"
          f", share - plain {sgn(ch['both+rediscover@share'] - ch['both+rediscover'])}")
