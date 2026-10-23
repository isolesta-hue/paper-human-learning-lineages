"""Stage-1 step 4 checks. reg-old / reg-new: the multi-moment life with tracking (seed 0) under the worldmodel.py before
step 4 and under the new one -- every record must be equal (compare the two pickles). props / none / rand: smoke on the
"@objs" world."""
import importlib.util, pickle, sys, time
import numpy as np, torch
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "reg-old":
    spec = importlib.util.spec_from_file_location("worldmodel", "../wm_before_props.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["worldmodel"] = mod
    spec.loader.exec_module(mod)
import run_long as RL
M = "both@ascii@multi+rediscover+chain+wm+bind+wmlast+track"
O = "both@ascii@objs+rediscover+chain+wm+bind+wmlast+track"
if which.startswith("reg"):
    out = RL.life(M, 0)
    pickle.dump(out, open(f"check_props_{which}.pkl", "wb"))
else:
    c = {"props": O + "+props", "none": O, "rand": O + "+propsrand"}[which]
    out = RL.one((0, c))
    for k in ("objs hit", "objs free", "objs table right", "objs table size", "voe solid", "voe push", "voe causal",
              "voe teleport", "multi", "story", "never-heard"):
        print(f"{which:5} {k:16}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
