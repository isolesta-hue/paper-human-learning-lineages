"""E14 decisive smoke (seed 0): the no-part body with the world model silent through each session ("+wmmeasure" on the
control too). If the views anomaly is only the setter seeing the BASE world model's answers, this life's never-heard should
follow the silenced part life (0.739), not the speaking no-part life (0.774)."""
import sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
t0 = time.time()
c = "both@ascii@views+rediscover+chain+wm+bind+wmlast+wmmeasure"
out = RL.one((0, c))
for k in ("views fill", "never-heard", "story", "chain"):
    print(f"Vnt {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
