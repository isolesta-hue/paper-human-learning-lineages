"""D35 (d) checks. reg: run_known.B seed 0 under the new run_long (the "+lpcur" switch) must equal its cache.
smoke: C = B + "+lpcur", seed 0 (cached by RL.one)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL, run_curious as RC
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "reg":
    out, old = RL.life(RC.B, 0), pickle.load(open(f"{RL.CACHE}/seed0_{RC.B}.pkl", "rb"))
    same = len(out) == len(old) and all(set(a) == set(b) for a, b in zip(old, out))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d, "| same records/keys", same)
else:
    out, b = RL.one((0, RC.C)), pickle.load(open(f"{RL.CACHE}/seed0_{RC.B}.pkl", "rb"))
    for k in RC.KEYS:
        print(f"C {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
        print(f"B {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in b))
    print("C pick curious", " ".join(f"{st.get('pick curious', np.nan):.0f}" for st in out))
print("time", round(time.time() - t0))
