"""W6 seed-0 smoke that keeps the life: RL.one caches it (into rie's cache through the symlink), so the 10-seed run reuses
it. Valid only if this code is the final W6 code (the regression must pass before the queue step is added)."""
import sys
import time

import numpy as np

import run_bi
import run_long as RL

t0 = time.time()
which = sys.argv[1]
steps = RL.one((0, run_bi.CONDS[which]))
for k in sorted(x for x in steps[-1] if x.startswith("bi ") and " all" not in x):
    print(f"  {k:24} " + " ".join("  -  " if np.isnan(st.get(k, np.nan)) else f"{st[k]:.3f}" for st in steps), flush=True)
for k in ("story", "multi", "never-heard", "chain"):
    print(f"  {k:24} end {steps[-1].get(k, np.nan):.3f}", flush=True)
print(f"{which} done in {time.time() - t0:.0f} s", flush=True)
