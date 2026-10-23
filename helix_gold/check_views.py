"""Step 4+ three-view checks (ledger D17 v1). reg: the Korean-story life (+bind+wmlast, seed 0) under the new code must equal
its cache. Smoke (seed 0) on "@views": V, Vr, Vn, body (run_views.CONDS)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
K = "both@ascii@kor+rediscover+chain+wm+bind+wmlast"
if which == "reg":
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d)
else:
    import run_views
    out = RL.one((0, run_views.CONDS[which]))
    for k in ("views fill", "views depth", "views depth front", "views depth moved", "story", "chain", "never-heard"):
        print(f"{which:4} {k:18}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
