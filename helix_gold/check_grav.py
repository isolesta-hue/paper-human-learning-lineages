"""Stage-1 step 4+ (gravity) checks. reg: the multi-moment life with tracking (seed 0) under the new code must equal the
pickle saved by check_props.py reg-new (code after step 4). grav / none / all: smoke on the "@grav" world."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
M = "both@ascii@multi+rediscover+chain+wm+bind+wmlast+track"
G = "both@ascii@grav+rediscover+chain+wm+bind+wmlast+track"
if which == "reg":
    out = RL.life(M, 0)
    old = pickle.load(open("check_props_reg-new.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d)
else:
    c = {"grav": G + "+grav", "none": G, "all": G + "+gravall"}[which]
    out = RL.one((0, c))
    for k in ("grav floor", "grav stack", "grav step", "grav falls repeats", "grav others repeat", "voe float", "voe through",
              "multi", "story", "never-heard"):
        print(f"{which:5} {k:18}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
