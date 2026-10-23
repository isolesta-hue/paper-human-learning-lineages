"""Stage-1 step 3c smoke, seed 0: the telling probe (record only) on the multi-moment life with tracking; the learner must
be unchanged (regression against the cached life, which was run before the probe existed)."""
import pickle, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
t0 = time.time()
M = "both@ascii@multi+rediscover+chain+wm+bind+wmlast+track"
out = RL.life(M, 0)
old = pickle.load(open(f"{RL.CACHE}/seed0_{M}.pkl", "rb"))
d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
print("regression max diff (learner unchanged by the probe)", d)
for k in ("tell book", "tell body", "tell first", "tell readback", "tell segmented", "tell same book", "tell two book",
          "tell three book", "multi"):
    print(f"{k:16}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
