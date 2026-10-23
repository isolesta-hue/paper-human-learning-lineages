"""Finding 119 regression + smoke: (a) +tune without +graded reproduces the cached life exactly; (b) +graded seed 0 runs."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which = sys.argv[1]
t0 = time.time()
if which == "reg":
    c = "both+rediscover+tune"
    new = RL.life(c, 0)
    old = pickle.load(open(f"{RL.CACHE}/seed0_{c}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, new) for k in a if isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d, "keys same", all(set(a) == set(b) for a, b in zip(old, new)))
else:
    c = "both+rediscover+tune+graded"
    out = RL.one((0, c))
    for i, st in enumerate(out):
        print(i + 1, {k: round(v, 3) for k, v in st.items() if k.split()[0] in ("grade", "eye", "trust", "out") or k in ("assoc w", "tune chunk", "never-heard")})
print("time", round(time.time() - t0))
