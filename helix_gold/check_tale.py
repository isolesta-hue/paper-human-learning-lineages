"""D22 checks. reg: the gravity life (finding 130's +grav, seed 0) under the new code (world generator with tale=, run_long's
tale records) must equal its cache. pilot: the tale world with no frame part (Tn, seed 0) -- does hearing tales in the same
rooms move the real gravity answers at all (vs the gravity world), and what does it answer to marked tale questions?"""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
G = "both@ascii@grav+rediscover+chain+wm+bind+wmlast+track+grav"
if which == "reg":
    out, old = RL.life(G, 0), pickle.load(open(f"{RL.CACHE}/seed0_{G}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d)
else:
    T = G.replace("@grav", "@tale")
    out, g = RL.one((0, T)), pickle.load(open(f"{RL.CACHE}/seed0_{G}.pkl", "rb"))
    for k in ("grav floor", "grav stack", "grav step", "tale real clean", "tale real drop", "tale real turned", "tale in",
              "tale transfer", "never-heard", "story"):
        print(f"Tn   {k:16}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
    for k in ("grav floor", "grav stack", "grav step", "never-heard", "story"):
        print(f"grav {k:16}", " ".join(f"{st.get(k, np.nan):.3f}" for st in g))
print("time", round(time.time() - t0))
