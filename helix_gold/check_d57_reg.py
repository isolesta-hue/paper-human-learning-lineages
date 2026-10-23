"""D57 regression, second run: save the fresh life and list every key that differs from the cached life."""
import pickle
import sys

import numpy as np

import run_long as RL

c, sd = sys.argv[1], int(sys.argv[2])
new = RL.life(c, sd)
pickle.dump(new, open(f"reg_new_seed{sd}.pkl", "wb"))
old = pickle.load(open(f"{RL.CACHE}/seed{sd}_{c}.pkl", "rb"))
bad = {}
for i, (a, b) in enumerate(zip(new, old)):
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, "MISSING"), b.get(k, "MISSING")
        try:
            eq = (np.isnan(x) and np.isnan(y)) or x == y
        except TypeError:
            eq = repr(x) == repr(y)
        if not eq:
            bad.setdefault(k, []).append((i + 1, x, y))
print(f"life regression {c} seed {sd}: {len(bad)} keys differ")
for k, v in bad.items():
    print(f"  {k}: sessions {[s for s, _, _ in v][:8]} first new {v[0][1]!r} old {v[0][2]!r}")
