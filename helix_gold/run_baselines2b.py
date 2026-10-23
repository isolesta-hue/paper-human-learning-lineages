"""Q5 part 2b (2026-10-12, user decision after finding 144: "the recommended way" -- no 100-seed extension now). Written and
fixed BEFORE running; the B1 means of finding 144 are already known (that is why the intervals below are descriptive only).

What it does: the replay MLP of part 2 (MLP-re, the strongest learned baseline in the glyph world) again on the SAME 30 seeds
with the SAME chosen setting (lr 0.003, 10 epochs a session; chosen on tuning seeds 1000-1004 in part 2), by importing the very
functions of run_baselines2.py (stream, run_net -- not copied), and this time each seed's per-session scores are SAVED
(baseline2b_cache/), so the paired differences with Voyager can be given per seed.
  R  reproduction check: the mean curves must equal finding 144's MLP-re lines (baselines2_30_partB1.out) to the printed 3
     decimals; any difference is reported and the intervals are not used.
  D  descriptive 95% intervals (paired t, 29 df) of Voyager B - MLP-re, end of life and life mean, for never-heard / once all /
     unseen all (Voyager B = both@ascii+rediscover+chain+wm+wmlast, the cached 30 seeds). No non-inferiority claim: a margin
     chosen after seeing the means would not be a test.
The 100-seed extension of the undecided comparisons (two-stage rule) is deferred until after the first stage-1 score table
(ledger), because the late-life loss it measures is track B's target anyway.
What this assumes: torch on CPU with one thread is deterministic for the same seed (the reproduction check tests it); the
Voyager caches are the ones finding 143 / 144 used.
"""
import os
import sys
import pickle
from multiprocessing import Pool

import numpy as np

import run_long as RL
import run_baselines2 as B2

KEYS = ("never-heard", "once all", "unseen all")
REF = "both@ascii+rediscover+chain+wm+wmlast"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline2b_cache")
T29 = 2.045                                         # two-sided 95%, 29 degrees of freedom
B1_MLPRE = {                                        # finding 144, baselines2_30_partB1.out (MLP-re lines, 3 decimals)
    "never-heard": "0.503 0.603 0.633 0.648 0.652 0.655 0.653 0.650 0.648 0.647 0.646 0.649 0.644 0.637 0.632 0.629",
    "once all": "0.004 0.082 0.149 0.235 0.276 0.313 0.333 0.346 0.379 0.403 0.438 0.462 0.451 0.436 0.424 0.418",
    "unseen all": "0.014 0.020 0.076 0.108 0.131 0.185 0.219 0.219 0.222 0.257 0.284 0.314 0.315 0.295 0.295 0.289",
}


def job(seed):
    path = os.path.join(CACHE, f"ascii_MLP-re_seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    V, blocks, items = B2.stream("ascii", seed)
    out = B2.run_net("MLP-re", V, blocks, items, seed, 3e-3, 10)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def ci(d):
    d = np.asarray(d, dtype=float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return f"{m:+.4f} [{m - T29 * se:+.4f}, {m + T29 * se:+.4f}]"


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    seeds = list(range(30))
    with Pool(workers) as p:
        res = p.map(job, seeds)
    vy = [pickle.load(open(os.path.join(RL.CACHE, f"seed{sd}_{REF}.pkl"), "rb")) for sd in seeds]
    print("=== Q5 part 2b: replay MLP again (lr 0.003, 10 epochs), 30 seeds, glyph world ===")
    same = True
    for k in KEYS:
        m = np.array([[st[k] for st in lf] for lf in res])
        line = " ".join(f"{v:.3f}" for v in np.nanmean(m, 0))
        ok = line == B1_MLPRE[k]
        same &= ok
        print(f"  R {k:12} {'same as finding 144' if ok else 'DIFFERENT from finding 144'}: {line}")
    print(f"  R reproduction: {'exact (3 decimals)' if same else 'NOT exact -- intervals below are not used'}")
    for k in KEYS:
        b = np.array([[st[k] for st in lf] for lf in res])
        v = np.array([[st.get(k, np.nan) for st in lf] for lf in vy])[:, :b.shape[1]]
        print(f"  D {k:12} Voyager - MLP-re, end {ci(v[:, -1] - b[:, -1])} | life {ci(np.nanmean(v - b, 1))}")
