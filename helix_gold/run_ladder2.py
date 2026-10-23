"""Finding 116 (2026-10-04, the user's order: world ladder rung 1 -> 2 -> 3 -> 4): world ladder rung 2 -- words without
senses, on top of rung 1 (the ladder is cumulative).

Finding 111 (100 seeds): in the rung-1 world (common words with Zipf frequencies from session 9 on) the completion gap keeps
widening late in life (+0.033, 78:22) -- the method layer's share came back once the world gave it something to fix, mostly
as "falling less" (the paired comparison with the plain world p = 0.012, gone after headroom normalisation). Rung 2: from
the second half on, half the words of every new category lose their senses (data_v3 blind=True, P_BLIND 0.5 -- declared):
their features are the global centre + noise, telling nothing of category or relation, as abstract and function words in
real language are learned from context only. Pair members are blinded too, so the templates (relations carried by the
features) cannot help there; the connection table and the whole-sentence memory must.

Conditions (30 seeds, cache_long_v2): content@blind, both+rediscover@blind; references cached: content@share,
both+rediscover@share (rung 1), content, both+rediscover (plain).
Pre-registered:
  B1 completion in the rung-2 world: end, widening (s13-16 - s2-4), late widening (s13-16 - s6-8).
  B2 the spiral: late widening rung 2 - rung 1, paired by seed (the same seed = the same sentences and common words; only
     the blinded features differ -- but the element memory is memorised from all features at birth, so the learners
     differ slightly from session 1 on: smoke seed 0), raw and headroom-normalised.
  B3 the price of senseless words: the fixed learner's and the self-modifying learner's end score, rung 2 - rung 1.
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

WORLDS = {"rung2": ("both+rediscover@blind", "content@blind"), "rung1": ("both+rediscover@share", "content@share"),
          "plain": ("both+rediscover", "content")}


def sgn(x):
    x = np.asarray(x)
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for pr in WORLDS.values() for c in pr for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; world ladder rung 2: words without senses (on top of rung 1) ===")
    nh = {c: np.array([[st["never-heard"] for st in out[(sd, c)]] for sd in range(n_seeds)]) for pr in WORLDS.values() for c in pr}
    once = {c: np.array([[st["once all"] for st in out[(sd, c)]] for sd in range(n_seeds)]) for pr in WORLDS.values() for c in pr}
    late, laten = {}, {}
    for world, (m, f) in WORLDS.items():
        print(f"\n##### {world}")
        for c in (m, f):
            print(f"  {c:24} never-heard " + " ".join(f"{v:.3f}" for v in nh[c].mean(0)))
            print(f"  {c:24} once all    " + " ".join(f"{v:.3f}" for v in once[c].mean(0)))
        g = nh[m] - nh[f]
        gn = g / np.maximum(1.0 - nh[f], 1e-9)
        late[world] = g[:, 12:16].mean(1) - g[:, 5:8].mean(1)
        laten[world] = gn[:, 12:16].mean(1) - gn[:, 5:8].mean(1)
        print("  completion by session " + " ".join(f"{v:+.3f}" for v in g.mean(0)))
        print(f"  B1 end {sgn(g[:, -1])} | widening {sgn(g[:, 12:16].mean(1) - g[:, 1:4].mean(1))} | late widening {sgn(late[world])}")
    print(f"\nB2 late widening rung2 - rung1: {sgn(late['rung2'] - late['rung1'])} | headroom-normalised {sgn(laten['rung2'] - laten['rung1'])}")
    print(f"   (rung1 - plain: {sgn(late['rung1'] - late['plain'])})")
    for (m, f) in (WORLDS["rung2"],):
        print(f"B3 end score rung2 - rung1: fixed {sgn(nh[f][:, -1] - nh['content@share'][:, -1])} | "
              f"self-modifying {sgn(nh[m][:, -1] - nh['both+rediscover@share'][:, -1])}")
