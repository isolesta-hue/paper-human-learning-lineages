"""Stage-1 step 2 checks: (a) regression -- the Korean-story life (+wmlast) is unchanged by the new code; (b) smoke, seed 0:
multi-moment stories with +track, +trackone and without tracking."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
K = "both@ascii@kor+rediscover+chain+wm+bind+wmlast"
M = "both@ascii@multi+rediscover+chain+wm+bind+wmlast"
if which == "reg":
    new = RL.life(K, 0)
    old = pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, new) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d)
else:
    c = {"track": M + "+track", "one": M + "+trackone", "none": M}[which]
    out = RL.one((0, c))
    for k in ("multi", "multi same", "multi two", "multi three", "multi wm fired", "multi wm right", "multi trust",
              "story", "chain", "never-heard"):
        print(f"{which:5} {k:15}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
