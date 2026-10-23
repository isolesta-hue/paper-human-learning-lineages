"""D80 v2 seed-0 smoke (2026-10-22, before any life of run_d80c.py's seeds 20-29): the condition runs one life (seed 0) beside
the main cache and reads NO outcome key -- only that the life runs to its end, the item counts per kind, that the EQ2 keys
and the told-aware store's learned meanings are written, and the scene part's counted events.
    python3 smoke_d80c.py [OUT=smoke_d80c]"""
import os
import sys

os.environ["RIE_NO_RESUME"] = "1"

import run_long as RL
from run_d80c import COND

if __name__ == "__main__":
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "smoke_d80c")
    os.makedirs(out_dir, exist_ok=True)
    RL.CACHE, RL.PROGRESS, RL.RESUME = (os.path.join(out_dir, x) for x in ("cache", "progress", "resume"))
    res = RL.life(COND, 0)
    print(f"=== {COND} seed 0: {len(res)} sessions ===")
    for i, st in enumerate(res):
        n = {k: int(st.get(f"{k} n new", 0)) for k in ("seq now told", "seq before told", "seq now", "seq before", "seq still")}
        has = sorted({k.split()[-2] for k in st if k.startswith("seq now") and k.endswith(" new") and " n " not in k})
        print(f"session {i + 1}: items {n} | rules written {has} | told-aware meanings "
              f"{sum(1 for k in st if k.startswith('seq2 rel '))} | scene events {st.get('scene events', 'none')}")
    print("exit 0")
