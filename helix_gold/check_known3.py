"""D26 (1) S1 checks 3 (seed 0): K = knownonly (new records), Br = B +knownrec (records only; must equal B's cache on B's
keys), and the fair comparison on answerable items."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL, run_known as RK
torch.set_num_threads(1)
t0 = time.time()
br, b = RL.one((0, RK.BR)), pickle.load(open(f"{RL.CACHE}/seed0_{RK.B}.pkl", "rb"))
d = max(abs(x[k] - y[k]) for x, y in zip(b, br) for k in x if k in y and isinstance(x[k], float) and not np.isnan(x[k]))
print("Br vs B cache on B's keys: max diff", d)
k = RL.one((0, RK.K))
for key in ("never-heard", "never-heard answerable", "unheard answer never-heard", "chain", "chain answerable", "unheard answer chain"):
    print(f"K  {key:28}", " ".join(f"{st.get(key, np.nan):.3f}" for st in k))
    print(f"Br {key:28}", " ".join(f"{st.get(key, np.nan):.3f}" for st in br))
print("time", round(time.time() - t0))
