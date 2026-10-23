"""Step 4+ object permanence (record only) smoke, seed 0, on the @multi world: track (regression against
check_props_reg-new.pkl, the same learner before the probe) and bind-only."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
M = "both@ascii@multi+rediscover+chain+wm+bind+wmlast"
out = RL.life(M + ("+track" if which == "track" else ""), 0)
if which == "track":
    old = pickle.load(open("check_props_reg-new.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d)
for k in ("voe permanence", "voe identity", "multi three", "multi"):
    print(f"{which:5} {k:15}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
