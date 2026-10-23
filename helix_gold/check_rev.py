"""Stage-1 step 3a checks: (a) regression -- the glyph-world life with the world model (finding 120) is unchanged by the new
code; (b) smoke, seed 0: scene -> motion word, with and without the books read backwards (+wmrev)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
R = "both@ascii+rediscover+chain+wm"
B = "both@ascii+rediscover+chain+wm+wmlast"
if which == "reg":
    new = RL.life(R, 0)
    old = pickle.load(open(f"{RL.CACHE}/seed0_{R}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, new) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print("regression max diff", d)
    print("describe (record only)", " ".join(f"{st.get('describe', np.nan):.3f}" for st in new))
else:
    c = {"rev": B + "+wmrev", "base": B}[which]
    out = RL.one((0, c))
    for k in ("describe", "describe flipped", "describe wm fired", "describe wm right", "chain", "never-heard"):
        print(f"{which:5} {k:18}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
