"""D22 pilot 2 (seed 0, no cache read or written): the body alone (world model silent only while measuring, D21) on the real
gravity questions, in the gravity world and the tale world without a frame part. The gravity life also serves as the
regression check of this record change (its old keys must equal the cache). argv: grav | tale."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
G = "both@ascii@grav+rediscover+chain+wm+bind+wmlast+track+grav"
c = G if which == "grav" else G.replace("@grav", "@tale")
out, old = RL.life(c, 0), pickle.load(open(f"{RL.CACHE}/seed0_{c}.pkl", "rb"))
d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
print(which, "old keys vs cache max diff", d)
for k in ("grav body unturned", "grav body turned", "tale in body", "tale in", "tale real clean", "tale real drop", "never-heard"):
    print(f"{which:5} {k:18}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
