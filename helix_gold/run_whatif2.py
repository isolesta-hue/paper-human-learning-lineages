"""Finding 114 (2026-10-03, from finding 113): the self-experiment on a WORLD-SHAPED self-test -- record only.

Finding 113: the self-experiment's arithmetic is exact, but the judge's held-out own sentences are shaped unlike the world
(heard-shape sentences: the connection table always wins there, the whole-sentence memory never can), so the self-chosen
weights missed (connection table: -0.026, 0:10). Here the same what-if is computed on the judge's WORLD-SHAPED questions
(integrated.world_items, world_score's generator: the core partners stay, the background is swapped for other known words of
the same category and kind -- "the same event in company never seen", the outside test's shape made by the learner itself,
from its own record; own generator seeded per session, the method layer's draws untouched). Sources compared with the
outside test: "self" (finding 113's held-out sentences) and "world" (this one). Conditions +whatif2 (includes +whatif).
Pre-registered: W0 regression; W1 rank correlation with the outside by session, per source and part; W2 the outside gain of
the weight each source would choose, sessions 2-16 and 9-16, sign test. World beats self on W2 for the connection table
(finding 113's miss) = the self-test's shape was the problem.

--- finding 113's docstring follows ---
Finding 113 (2026-10-03, step 2 of the method-making learner, from finding 112): the self-experiment -- record only.

Finding 112 showed the answer's score splits EXACTLY into its parts, and that in the glyph world the late-life errors are
almost all made by the whole-sentence memory. So the learner can ask "what if" of itself without relearning: how many of my
own held-out questions would I get right if the whole-sentence memory (or the connection table) weighed w times as much?
(Humans cannot run their own brain at another setting; Voyager can -- the user's point, 2026-10-03.) If the answer on the
learner's OWN held-out sentences (the judge's, leave-one-out) points the same way as on the outside test, the method layer
can choose part weights by self-experiment instead of trial branches.

Recorded (+whatif): at the first rung of every method session, on the stay branch and the judge's held-out sentences,
accuracy at chunk weight x {0, 0.5, 1, 1.5} and connection-table weight x {0, 0.5, 1, 1.5} (CF_W, declared); at the end of
the session the same on the outside test (checking only -- the learner never sees it).
Conditions (10 seeds, a probe): both+rediscover+whatif (text), both@ascii+rediscover+whatif (glyph); references cached.
Pre-registered:
  W0 regression identical; sanity: self accuracy at w = 1 equals the judge's score on the same sentences (pooled vs mean of
     per-category scores may differ -- printed side by side).
  W1 per part and session: rank correlation over the four weights between self and outside accuracy, mean over seeds by
     session.
  W2 the self-chosen weight: the w with the best self accuracy; its outside gain over w = 1 (what the self-experiment would
     have bought), sessions 2-16 and 9-16, sign test over seeds. > 0 = the self-experiment is a usable eye for part weights.
"""
import sys
from multiprocessing import Pool

import numpy as np

import run_long as RL
from run_unified_continual import sign_p

PAIRS = {"text": ("both+rediscover+whatif2", "both+rediscover"), "glyph": ("both@ascii+rediscover+whatif2", "both@ascii+rediscover")}
SOURCES = ("self", "world")


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def rcorr(a, b):
    a, b = rank(np.asarray(a)), rank(np.asarray(b))
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sgn(x):
    x = np.asarray(x)
    w, lo = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    return f"{x.mean():+.4f} ({w}:{lo}, p={sign_p(w, lo):.4f})"


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for pr in PAIRS.values() for c in pr for sd in range(n_seeds)]
    with Pool(workers) as p:
        out = dict(zip(tasks, p.map(RL.one, tasks)))
    print(f"=== {n_seeds} seeds; the self-experiment: what if a part weighed w times as much (record only) ===")
    for world, (c, ref) in PAIRS.items():
        same = all(np.allclose([s["never-heard"] for s in out[(sd, c)]], [s["never-heard"] for s in out[(sd, ref)]],
                               atol=1e-12) for sd in range(n_seeds))
        lives = [out[(sd, c)] for sd in range(n_seeds)]
        n = len(lives[0])
        print(f"\n##### {world}: W0 regression identical: {same}")
        print("   sanity, self w=1 pooled vs judge (mean over sessions): "
              f"{np.nanmean([l[i].get('cf self pooled', np.nan) for l in lives for i in range(1, n)]):.3f} vs "
              f"{np.nanmean([l[i].get('cf self judge', np.nan) for l in lives for i in range(1, n)]):.3f}")
        for part in ("chunk", "assoc"):
            print(f"  --- {part} ---")
            for where in SOURCES + ("out",):
                print(f"   {where:5} acc by w (sessions 2-16 mean): " + "  ".join(
                    f"{wt}: {np.nanmean([l[i].get(f'cf {where} {part} {wt}', np.nan) for l in lives for i in range(1, n)]):.3f}"
                    for wt in RL.CF_W))
            for src in SOURCES:
                rc = [[rcorr([l[i][f"cf {src} {part} {wt}"] for wt in RL.CF_W], [l[i][f"cf out {part} {wt}"] for wt in RL.CF_W])
                       for i in range(1, n)] for l in lives]
                print(f"   W1 {src:5} rank corr with out by session: " + " ".join(f"{v:+.2f}" for v in np.nanmean(rc, axis=0)))
                gains = []
                for l in lives:
                    g = []
                    for i in range(1, n):
                        best = max(RL.CF_W, key=lambda wt: l[i][f"cf {src} {part} {wt}"])
                        g.append(l[i][f"cf out {part} {best}"] - l[i][f"cf out {part} 1.0"])
                    gains.append(g)
                gains = np.array(gains)
                print(f"   W2 {src:5} outside gain of its chosen weight: 2-16 {sgn(gains.mean(1))} | 9-16 {sgn(gains[:, 7:].mean(1))}")
                print("      by session: " + " ".join(f"{v:+.3f}" for v in gains.mean(0)))
