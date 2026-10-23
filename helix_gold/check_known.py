"""D26 (1) S1 checks. reg: finding 127's body B (seed 0) under the new code (only_known, classify through super) must equal
its cache. smoke: B +knownonly, seed 0 (cached by RL.one)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL, run_known
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "reg":
    out, old = RL.life(run_known.B, 0), pickle.load(open(f"{RL.CACHE}/seed0_{run_known.B}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d)
else:
    out, b = RL.one((0, run_known.K)), pickle.load(open(f"{RL.CACHE}/seed0_{run_known.B}.pkl", "rb"))
    for k in run_known.KEYS:
        print(f"K {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
        print(f"B {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in b))
print("time", round(time.time() - t0))
