"""Finding 83's follow-up in the long life (life32, 30 seeds, optimised code, cache_long_v2 -- all references re-run here).
(1) adopting finding 82 (connections and templates from first-hand evidence only) as Voyager's default: does it hold with the
method layer over a long life with two changes? (2) the one tier-2 part that passed the screening -- practice where the routes
disagree -- on top of it.
Conditions: content+world, both (old default: all sources), both+world, both+world+disagree.
Key comparisons, fixed before running: K1 both+world vs both (end, life); K2 both+world+disagree vs both+world (end, life);
K3 the completion criterion under the new default: both+world - content+world, end and widening (s13-16 - s2-4).
"""
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["content+world", "both", "both+world", "both+world+disagree"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS[::-1] for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    res = [{c: out[(sd, c)] for c in CONDS} for sd in range(n_seeds)]
    n = len(res[0]["both"])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; tier 2 in the long life (life32, optimised code) ===")
    for key in ["never-heard", "once all", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in CONDS:
            print(f"{c:22} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    print("\n--- paired (never-heard) ---")
    for g, b in [("both+world", "both"), ("both+world+disagree", "both+world")]:
        dl = np.array([np.mean([s["never-heard"] for s in r[g]]) - np.mean([s["never-heard"] for s in r[b]]) for r in res])
        de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in res])
        print(f"{g:22} vs {b:12} life {sgn(dl)}  end {sgn(de)}")
    gaps = np.array([[r["both+world"][i]["never-heard"] - r["content+world"][i]["never-heard"] for i in range(n)] for r in res])
    print("completion (both+world - content+world) by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")


if __name__ == "__main__":
    main()
