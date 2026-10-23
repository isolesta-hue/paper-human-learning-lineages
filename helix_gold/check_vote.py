"""Finding 124 smoke: the voting probe (record only) on the Korean-story life, seed 0."""
import sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
t0 = time.time()
c = "both@ascii@kor+rediscover+chain+wm+bind+vote"
out = RL.one((0, c))
for k in ("sum story", "vote story", "vec story", "agree story", "sum never-heard", "vote never-heard", "vec never-heard", "agree never-heard", "sum chain", "vote chain", "vec chain", "agree chain"):
    print(f"{k:18}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("T", {k: round(v, 2) for k, v in out[-1].items() if k.startswith("vote T")})
print("time", round(time.time() - t0))
