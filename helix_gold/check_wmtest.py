"""Leaning-test checks. reg: the self life (finding 140's S, seed 0) under the new code must equal its cache. Smoke (seed 0):
the three "+wmtest" conditions."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "reg":
    import run_self
    K = run_self.CONDS["S"]
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d)
else:
    import run_wmtest
    x, xn, own = run_wmtest.WORLDS[which]
    out = RL.one((0, x + "+wmmeasure"))
    for k in (own, "never-heard", "story", "chain"):
        print(f"{which:6} {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
