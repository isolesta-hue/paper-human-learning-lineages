"""Step 4+ successor rule checks (ledger D1a). reg-old / reg-new: the small-numbers life ("@count" world, +count), seed 0,
under the old code (run from a copy of the pushed files) and the new code must be equal. succ / count: smoke on the
"@succ" world, with and without the successor rule."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
N = "both@ascii@count+rediscover+chain+wm+bind+wmlast+track+count"
S = "both@ascii@succ+rediscover+chain+wm+bind+wmlast+track+count"
REF = "/tmp/claude-0/-home-user/23cc98b2-a804-5343-bef4-6479ad5f8346/scratchpad/check_succ_reg-old.pkl"
if which == "reg-old":
    pickle.dump(RL.life(N, 0), open(REF, "wb"))
elif which == "reg-new":
    out, old = RL.life(N, 0), pickle.load(open(REF, "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    print(which, "regression max diff", d, "| keys only new:", sorted(set(out[-1]) - set(old[-1])))
else:
    out = RL.one((0, {"succ": S + "+succ", "count": S, "rev": S + "+succrev"}[which]))
    for k in ("count four", "voe four", "count put", "count take", "voe number", "count table right", "contain in", "multi",
              "story", "never-heard"):
        print(f"{which:5} {k:17}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
