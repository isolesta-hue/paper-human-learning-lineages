"""Speed pass check (2026-10-12): CuedPatterns.__deepcopy__ (one-level copies of the immutable count tables) must leave
every life exactly as before. Old = the cached lives (written by the old code), new = RL.life under the new code, seed 0:
run_known.B (no tale, ~20 min) then run_tale.B (the D22 body). Also prints the wall time of each.
What this assumes: seed 0 of two bodies exercises the copy (both use the cued templates and the method layer's branches)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL, run_known, run_tale
torch.set_num_threads(1)
for name, cond in (("known-B", run_known.B), ("tale-B", run_tale.B)):
    t0 = time.time()
    out, old = RL.life(cond, 0), pickle.load(open(f"{RL.CACHE}/seed0_{cond}.pkl", "rb"))
    same_keys = len(out) == len(old) and all(set(a) == set(b) for a, b in zip(old, out))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    nan_same = all((isinstance(a[k], float) and np.isnan(a[k])) == (isinstance(b[k], float) and np.isnan(b[k]))
                   for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float))
    print(name, "regression max diff", d, "| same records/keys", same_keys, "| nan pattern same", nan_same,
          "| time", round(time.time() - t0), flush=True)
