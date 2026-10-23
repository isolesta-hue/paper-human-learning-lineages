"""Step 4+ small numbers checks. reg-multi / reg-contain: the multi-moment life with tracking and the containment life
(+contain), seed 0, under the new code must equal their pickles/caches. count / none / rand: smoke on the "@count" world."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
M = "both@ascii@multi+rediscover+chain+wm+bind+wmlast+track"
C = "both@ascii@contain+rediscover+chain+wm+bind+wmlast+track+contain"
N = "both@ascii@count+rediscover+chain+wm+bind+wmlast+track"
if which.startswith("reg"):
    cond, ref = (M, "check_props_reg-new.pkl") if which == "reg-multi" else (C, f"{RL.CACHE}/seed0_{C}.pkl")
    out = RL.life(cond, 0)
    old = pickle.load(open(ref, "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d)
else:
    c = {"count": N + "+count", "none": N, "rand": N + "+countrand"}[which]
    out = RL.one((0, c))
    for k in ("count put", "count take", "count table right", "voe number", "contain in", "multi", "story", "never-heard"):
        print(f"{which:5} {k:17}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
