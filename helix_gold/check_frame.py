"""Stage-1 step 4+ (reference frames) checks. reg: the "@hold" stage-B life (finding 122's condition, seed 0) under the new
code must equal its cached life. frame / ctx / none: smoke on the "@frame" world."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
H = "both@ascii@hold+rediscover+chain+wmctx"
F = "both@ascii@frame+rediscover+chain+wm"
if which == "reg":
    out = RL.life(H, 0)
    old = pickle.load(open(f"{RL.CACHE}/seed0_{H}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d)
else:
    c = {"frame": F + "+wmframe", "ctx": F + "+wmctx", "none": F}[which]
    out = RL.one((0, c))
    for k in ("chain", "chain quarter", "chain half", "never-heard"):
        print(f"{which:5} {k:14}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
