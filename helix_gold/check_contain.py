"""Step 4+ containment checks. reg: the multi-moment life with tracking (seed 0) under the new code must equal
check_props_reg-new.pkl. contain / none / all: smoke on the "@contain" world."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
M = "both@ascii@multi+rediscover+chain+wm+bind+wmlast+track"
C = "both@ascii@contain+rediscover+chain+wm+bind+wmlast+track"
if which == "reg":
    out = RL.life(M, 0)
    old = pickle.load(open("check_props_reg-new.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d)
else:
    c = {"contain": C + "+contain", "none": C, "all": C + "+containall"}[which]
    out = RL.one((0, c))
    for k in ("contain in", "contain by", "contain table right", "voe inside", "voe beside", "multi", "story", "never-heard"):
        print(f"{which:7} {k:19}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
