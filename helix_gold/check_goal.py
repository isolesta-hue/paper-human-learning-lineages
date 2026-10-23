"""D2 goal checks. reg: the viewpoint life (finding 138's P, seed 0) under the new code must equal its cache. Smoke (seed 0)
on "@goal": run_goal.CONDS."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "reg":
    import run_viewpt
    K = run_viewpt.CONDS["P"]
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d)
else:
    import run_goal
    out = RL.one((0, run_goal.CONDS[which]))
    for k in ("goal which", "goal next", "story", "chain", "never-heard"):
        print(f"{which:4} {k:11}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
