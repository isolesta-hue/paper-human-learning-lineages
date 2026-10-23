"""The method ledger's first test: self-budget (2026-09-28; the user's proposal -- the learner should evaluate its own methods by
cost as well as accuracy, as we do in these experiments: accuracy at the same compute, compute at the same accuracy).

Voyager's method layer chooses how hard to think each session (run_long.life, "self"): one budget level up after a search that
found a clear win over the stay branch, one down after one that did not (levels low / mid / high; budget-aware caution for its
decisions). Findings 75-76: the method layer's gain comes at the world's changes -- so a learner that knows what its thinking
was worth should think hard around changes and little in quiet stretches.
Conditions (life32, 30 seeds, cached per life): both-self, both-high (every branch to 200 epochs: the frontier's top);
references from run_long.py's cache: content, both (mid), both-low, both-low+aware.
Key comparisons, fixed before running: K1 both-self vs both (mid): end accuracy AND trial compute (same accuracy for less
compute, or more accuracy for the same); K2 both-self vs both-low+aware: end. Also the cost-accuracy frontier of all budgets.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["both-self", "both-high"]
REFS = ["content", "both", "both-low", "both-low+aware"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS[::-1] for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    refs = [RL.load_seed(sd, {}) for sd in range(n_seeds)]
    res = [{**{c: refs[sd][c] for c in REFS}, **{c: out[(sd, c)] for c in CONDS}} for sd in range(n_seeds)]
    n = len(res[0]["both"])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    def lm(r, c):
        return np.mean([s["never-heard"] for s in r[c]])

    def ep(r, c):
        return np.nansum([s["trial epochs"] for s in r[c]])

    print(f"=== {n_seeds} seeds; self-budget (the method ledger's first test, life32) ===")
    for key in ["never-heard", "trial epochs", "budget level", "changed", "lr", "review"]:
        print(f"\n--- {key} ---")
        for c in REFS + CONDS:
            print(f"{c:15} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):7.3f}" for i in range(n)))
    print("\n--- cost-accuracy frontier (trial epochs per life / life mean / end) ---")
    for c in REFS[1:] + CONDS:
        print(f"{c:15} {np.mean([ep(r, c) for r in res]):8.0f}  {np.mean([lm(r, c) for r in res]):.3f}  "
              f"{np.mean([r[c][-1]['never-heard'] for r in res]):.3f}")
    print("\n--- paired ---")
    for g, b in [("both-self", "both"), ("both-self", "both-low+aware"), ("both-high", "both"), ("both-self", "both-high")]:
        de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in res])
        dl = np.array([lm(r, g) - lm(r, b) for r in res])
        dc = np.array([ep(r, g) - ep(r, b) for r in res])
        print(f"{g:10} vs {b:15} life {sgn(dl)}  end {sgn(de)}  compute {dc.mean():+.0f} ({(dc < 0).sum()} of {len(dc)} cheaper)")
    lv = np.array([[r["both-self"][i]["budget level"] for i in range(1, n)] for r in res])
    print("\nboth-self: mean budget level s2-8 (rule A) %.2f | s9-12 (new kind) %.2f | s13-16 (rule change) %.2f"
          % (np.nanmean(lv[:, 0:7]), np.nanmean(lv[:, 7:11]), np.nanmean(lv[:, 11:15])))


if __name__ == "__main__":
    main()
