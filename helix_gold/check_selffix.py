"""Review fix check (2026-10-12): `self_budget = "-self" in cond ...` (was the bare "self"). Every condition without "self" in its
name runs the same code path; checked by experiment on run_known.B seed 0 (cached by the old code): must be exactly equal."""
import pickle, time
import numpy as np, torch
import run_long as RL, run_known
torch.set_num_threads(1)
t0 = time.time()
out, old = RL.life(run_known.B, 0), pickle.load(open(f"{RL.CACHE}/seed0_{run_known.B}.pkl", "rb"))
same = len(out) == len(old) and all(set(a) == set(b) for a, b in zip(old, out))
d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
print("selffix regression max diff", d, "| same records/keys", same, "| time", round(time.time() - t0))
