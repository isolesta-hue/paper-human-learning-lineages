"""D2 viewpoint checks. reg: the three-view life (finding 137's V, seed 0) under the new code must equal its cache (made by
check_views before the viewpoint code). Smoke (seed 0) on "@viewpt": run_viewpt.CONDS."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "reg":
    import run_views
    K = run_views.CONDS["V"]
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d)
else:
    import run_viewpt
    out = RL.one((0, run_viewpt.CONDS[which]))
    for k in ("viewpt back", "viewpt turn", "views fill", "views depth", "story", "chain", "never-heard"):
        print(f"{which:4} {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
