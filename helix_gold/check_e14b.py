"""E14 matched-control smokes for the other two worlds (seed 0): X_nt = X_n +wmmeasure. argv: viewpt | goal."""
import sys, time
import numpy as np, torch
import run_long as RL, run_wmtest
torch.set_num_threads(1)
t0, which = time.time(), sys.argv[1]
x, xn, own = run_wmtest.WORLDS[which]
out = RL.one((0, xn + "+wmmeasure"))
for k in (own, "never-heard", "story", "chain"):
    print(f"{which:6} nt {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
