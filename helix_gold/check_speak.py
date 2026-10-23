"""Stage-1 step 3b smoke, seed 0: the speaking probe (record only) on the Korean-story life; the learner itself must be
unchanged (regression against the cached life without +speak)."""
import pickle, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
t0 = time.time()
K = "both@ascii@kor+rediscover+chain+wm+bind+wmlast"
out = RL.life(K + "+speak", 0)
old = pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
print("regression max diff (learner unchanged by the probe)", d)
for k in ("speak body", "speak body mon", "speak book", "speak book mon", "speak swap", "speak readback", "speak lm found",
          "speak particles ok", "story"):
    print(f"{k:20}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
