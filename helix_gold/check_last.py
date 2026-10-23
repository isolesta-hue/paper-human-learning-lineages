"""Finding 125 checks: (a) regression -- the wm+bind life without +wmlast is unchanged by the restructured complete();
(b) smoke, seed 0: +wmlast (the decided world model speaks last)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
B = "both@ascii@kor+rediscover+chain+wm+bind"
if which == "reg":
    new = RL.life(B, 0)
    old = pickle.load(open(f"{RL.CACHE}/seed0_{B}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, new) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d)
else:
    out = RL.one((0, B + "+wmlast"))
    old = pickle.load(open(f"{RL.CACHE}/seed0_{B}.pkl", "rb"))
    for k in ("story", "chain", "chain flipped", "never-heard"):
        print(f"{k:14} last ", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
        print(f"{k:14} bind ", " ".join(f"{st.get(k, np.nan):.3f}" for st in old))
print("time", round(time.time() - t0))
