"""Finding 101: attention over the context words in the habit (run_long "+attend"; lab2.attention_bag -- the context mean
with weights bag_c x exp(m^T A z_c), A starting at zero, so at birth the answers are exactly the mean's; checked: max
difference 0.0 in both worlds). Finding 100: in the glyph world, early in life, the knobs barely change the outcome (the
outside test's own split-half reliability over the branches 0.10-0.22) -- the body gives the method layer no lever; the mean
lets background pictures dilute the verb and the start position. 30 seeds, cache_long_v2 (references reused).
Key comparisons, fixed before running:
  A1 (glyph, the body) content@ascii+fix+attend vs content@ascii+fix, end and life;
  A2 (glyph, completion) both@ascii+fix-routeknobs+attend - content@ascii+fix+attend, end and widening (s13-16 - s2-4);
  A3 (text, the body) content+world+recomb-side+attend vs content+world+recomb-side, end and life (must not lose);
  A4 (text, completion) both+world+recomb-side-routeknobs+attend - content+world+recomb-side+attend, end and widening.
The lever itself (finding 100's C3 with attention) is read from run_probe2.py 10 4 +attend glyph (attend_probe_10.out).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content@ascii+fix", "content@ascii+fix+attend", "both@ascii+fix-routeknobs+attend",
         "content+world+recomb-side", "content+world+recomb-side+attend", "both+world+recomb-side-routeknobs+attend"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    # the glyph world first (finding 100's lever question lives there); the text world after
    tasks = [(sd, c) for c in CONDS for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0][CONDS[0]])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.4f})"

    print(f"=== {n_seeds} seeds; attention in the habit ===")
    for key in ["never-heard", "once all", "unseen all", "pairs all", "pattern w", "chunk w", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:42} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))

    def pair(g, b_, name):
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b_]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b_][-1]["never-heard"] for r in res])
        print(f"{name} {g} vs {b_}: life {sgn(dl)}  end {sgn(de)}")

    def completion(g, b_, name):
        gaps = np.array([[r[g][i]["never-heard"] - r[b_][i]["never-heard"] for i in range(n)] for r in res])
        print(f"{name} completion ({g} - {b_}) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
        print(f"   end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    print()
    pair("content@ascii+fix+attend", "content@ascii+fix", "A1")
    completion("both@ascii+fix-routeknobs+attend", "content@ascii+fix+attend", "A2")
    pair("content+world+recomb-side+attend", "content+world+recomb-side", "A3")
    completion("both+world+recomb-side-routeknobs+attend", "content+world+recomb-side+attend", "A4")


if __name__ == "__main__":
    main()
