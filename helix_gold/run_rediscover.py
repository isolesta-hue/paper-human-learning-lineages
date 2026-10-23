"""Finding 103: the rediscovery test (the user's question after finding 100: is the misfit deeper than where it shows -- did
WE take the method layer's part by writing every improvement we found into the fixed body?). The method layer starts from the
OLD body (start settings only) and gets the hand-found improvements as two-state knobs (run_long "+rediscover"): self-made
recombination practice, its own side part, connections from first-hand evidence only; in the glyph world also the four
template repairs. The normal judge, the normal budget. 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running:
  R1 (text) both+rediscover vs both, end and life -- does it gain at all;
  R2 (text) both+rediscover vs content+world+recomb-side (the hand-made body) -- does it get there; and the share of the gap
     content -> content+world+recomb-side it closes at the end;
  R3 (glyph) both@ascii+rediscover vs both@ascii; R4 (glyph) vs content@ascii+fix, and the share of the gap closed.
  Also: per session, the share of lives with each improvement switched on.
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content", "both", "content+world+recomb-side", "both+rediscover",
         "content@ascii", "both@ascii", "content@ascii+fix", "both@ascii+rediscover"]
TOGGLES = ["setter", "self_side", "hebb_source", "cue_kinds", "cue_specific", "cue_members_vote", "cue_pure"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0][CONDS[0]])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.4f})"

    print(f"=== {n_seeds} seeds; the rediscovery test ===")
    for key in ["never-heard", "once all", "unseen all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:28} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    print("\n--- share of lives with the improvement switched on ---")
    for c in ["both+rediscover", "both@ascii+rediscover"]:
        for t in TOGGLES:
            v = [np.nanmean([r[c][i].get(f"on {t}", np.nan) for r in res]) for i in range(n)]
            if not np.all(np.isnan(v)):
                print(f"{c:24} {t:18} " + " ".join(f"{x:5.2f}" for x in v))

    def pair(g, b_, name):
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        print(f"{name} {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}")

    def share(g, old, new, name):
        e = lambda c: np.mean([r[c][-1]["never-heard"] for r in res])
        gap = e(new) - e(old)
        print(f"{name} gap closed at the end: ({g} - {old}) / ({new} - {old}) = {(e(g) - e(old)) / gap:+.2f} "
              f"(gap {gap:+.3f})")
    print()
    pair("both+rediscover", "both", "R1")
    pair("both+rediscover", "content+world+recomb-side", "R2")
    share("both+rediscover", "content", "content+world+recomb-side", "R2")
    pair("both@ascii+rediscover", "both@ascii", "R3")
    pair("both@ascii+rediscover", "content@ascii+fix", "R4")
    share("both@ascii+rediscover", "content@ascii", "content@ascii+fix", "R4")


if __name__ == "__main__":
    main()
