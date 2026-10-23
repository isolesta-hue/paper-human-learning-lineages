"""Step 4+ number-line room checks (ledger D13 / D15). reg: the Korean-story life (+bind+wmlast, seed 0) under the new code
must equal its cache (made before the number line). Smoke (seed 0) on the "@numrow" / "@numown" worlds:
  A (row, shared spacing), B (own spacing), Bs (own + "+scale"), Br (own + "+scalerev"), body (row, no world model)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
K = "both@ascii@kor+rediscover+chain+wm+bind+wmlast"
A = "both@ascii@numrow+rediscover+chain+wm+bind+wmlast"
B = "both@ascii@numown+rediscover+chain+wm+bind+wmlast"
if which == "reg":
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d, "| keys only new:", sorted(set(out[-1]) - set(old[-1])))
else:
    c = {"A": A, "B": B, "Bs": B + "+scale", "Br": B + "+scalerev", "body": "both@ascii@numrow+rediscover+chain"}[which]
    out = RL.one((0, c))
    for k in ("numline add", "numline sub", "numline sub wm fired", "numline sub wm right", "numline scale", "story", "chain",
              "never-heard"):
        print(f"{which:4} {k:21}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
