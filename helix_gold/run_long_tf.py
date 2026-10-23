"""The transformer control in the long test (the user, 2026-09-26: the transformer version should have been a control in the
long test too). Finding 73's TLearner (the same fixed original memory, attention instead of our content layer) lives the
life32 world with fixed start settings (tf) and with the method layer on top (tf+method); references: content / both from
run_long.py's cache. Questions: (1) does the transformer keep forgetting over a life twice as long, (2) does the method layer's
gap on the transformer widen at the two changes, as ours did (finding 75)? Seeds: 30 (cached per life).
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

from run_unified_continual import sign_p
import run_long as RL

CONDS = ["tf", "tf+method"]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tasks = [(sd, c) for c in CONDS[::-1] for sd in range(n_seeds)]
    with Pool(workers) as pool:
        out = dict(zip(tasks, pool.map(RL.one, tasks, chunksize=1)))
    refs = [RL.load_seed(sd, {}) for sd in range(n_seeds)]
    res = [{**refs[sd], **{c: out[(sd, c)] for c in CONDS}} for sd in range(n_seeds)]
    n = len(res[0]["tf"])

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds; transformer control in the long test (life32) ===")
    for key in ["never-heard", "newest", "older", "once all", "review", "lr"]:
        print(f"\n--- {key} ---")
        for c in ["tf", "tf+method", "content", "both"]:
            print(f"{c:10} " + " ".join(f"{np.nanmean([r[c][i].get(key, np.nan) for r in res]):6.3f}" for i in range(n)))
    gaps = np.array([[r["tf+method"][i]["never-heard"] - r["tf"][i]["never-heard"] for i in range(n)] for r in res])
    print("\nmethod gap on the transformer by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
    print(f"end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}")
    for g, b in [("content", "tf"), ("both", "tf+method")]:
        de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in res])
        print(f"{g} vs {b}: end {sgn(de)}")
    peak = np.array([max(s["older"] for s in r["tf"][1:]) - r["tf"][-1]["older"] for r in res])
    print(f"tf forgetting (best older score in life - end older score): {sgn(peak)}")


if __name__ == "__main__":
    main()
