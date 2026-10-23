"""Audit (2026-10-08): the Korean-story life (+bind, seed 0) under today's code must equal its cache from before today's
changes (memory pass, naming moments without particles, tracking/props/gravity/frame/containment parts) -- these were
argued to leave the @kor world unchanged; this checks it."""
import pickle, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
t0 = time.time()
K = "both@ascii@kor+rediscover+chain+wm+bind"
out = RL.life(K, 0)
old = pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
print("audit regression max diff", d, "| keys compared", len(old[0]))
print("time", round(time.time() - t0))
