"""Cache-reuse audit for run_multi50: the multi-moment life (+track, seed 0) under today's code must equal its cache."""
import pickle, time
import numpy as np, torch
import run_long as RL
from run_multi import B
torch.set_num_threads(1)
t0 = time.time()
K = B + "+track"
out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
print("multi life (+track, seed 0) regression max diff", d)
print("time", round(time.time() - t0))
