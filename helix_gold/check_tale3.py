"""D22 v1b checks. reg: the gravity life (finding 130's +grav, seed 0) under the new world model (refresh split into
_book_add, the tale frame switch in term) must equal its cache. smoke: run_tale's four conditions, seed 0 (cached by RL.one)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL, run_tale
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "reg":
    G = "both@ascii@grav+rediscover+chain+wm+bind+wmlast+track+grav"
    out, old = RL.life(G, 0), pickle.load(open(f"{RL.CACHE}/seed0_{G}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d)
else:
    c = run_tale.CONDS[which]
    out = RL.one((0, c))
    for k in run_tale.KEYS:
        print(f"{which:4} {k:28}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
