"""W7b seed-0 smoke of the criterion-8 world (a check, never a result): one life of "@c8" with the parts that fit a multi
world (the @multi body's condition with @c8), run through life() directly -- nothing is cached. Reports that the c8 keys
appear in every session from the first story on, their item counts, the body-alone variants, and the life's time.
    python3 smoke_c8.py"""
import time
import numpy as np
import torch
import run_long as RL

torch.set_num_threads(1)
cond = "both@ascii@c8+rediscover+chain+wm+bind+wmlast+track"
t0 = time.time()
stats = RL.life(cond, 0)
keys = sorted({k for st in stats for k in st if isinstance(k, str) and k.startswith("c8")})
print(f"smoke {cond} seed 0: {len(stats)} sessions, {time.time() - t0:.0f} s")
print(f"c8 keys ({len(keys)}): {keys}")
for k in keys:
    v = [st.get(k, np.nan) for st in stats]
    n_ok = sum(1 for x in v if isinstance(x, float) and not np.isnan(x))
    print(f"  {k:28s} present in {n_ok}/{len(stats)} sessions, values in [0, 1]: "
          f"{all(0 <= x <= 1 for x in v if isinstance(x, float) and not np.isnan(x))}")
base = set(RL.life.__code__.co_consts) and None
print("exit 0")
