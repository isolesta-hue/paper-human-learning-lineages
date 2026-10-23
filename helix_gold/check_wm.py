"""World-model stage A checks: (a) regression -- an @ascii life without +wm/+chain is unchanged; (b) smoke -- seed 0,
+chain (record only) vs +chain+wm: chain accuracy by session, everything else equal."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
base = "both@ascii+rediscover"
if which == "reg":
    new = RL.life(base, 0)
    old = pickle.load(open(f"{RL.CACHE}/seed0_{base}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, new) for k in a if isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d, "keys same", all(set(a) == set(b) for a, b in zip(old, new)))
else:
    c = base + ("+chain+wm" if which == "wm" else "+chain")
    out = RL.one((0, c))
    print(c, "chain", " ".join(f"{st.get('chain', np.nan):.3f}" for st in out))
    print(c, "never-heard", " ".join(f"{st['never-heard']:.3f}" for st in out))
print("time", round(time.time() - t0))
