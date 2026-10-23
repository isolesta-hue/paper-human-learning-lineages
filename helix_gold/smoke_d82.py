"""D82 first step seed-0 smoke (2026-10-22, before any life of run_d82.py): the NEG condition with "+d82" runs one life
(seed 0) beside the main cache and reads NO outcome key -- only R0 on that life (every "d72 ..." key equal to the stored
seed-0 NEG life) and that the "d82 ..." keys are written.
    python3 smoke_d82.py [OUT=smoke_d82]"""
import os
import pickle
import sys

os.environ["RIE_NO_RESUME"] = "1"

import numpy as np

import run_long as RL
from run_d82 import CONDS, PLAIN, same

if __name__ == "__main__":
    stored = pickle.load(open(os.path.join(RL.CACHE, f"seed0_{PLAIN['neg']}.pkl"), "rb"))
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "smoke_d82")
    os.makedirs(out_dir, exist_ok=True)
    RL.CACHE, RL.PROGRESS, RL.RESUME = (os.path.join(out_dir, x) for x in ("cache", "progress", "resume"))
    res = RL.life(CONDS["neg"], 0)
    diff = [(i, k) for i, (a, b) in enumerate(zip(res, stored)) for k in set(a) | set(b)
            if k.startswith("d72 ") and not same(a.get(k, "missing"), b.get(k, "missing"))]
    print(f"=== {CONDS['neg']} seed 0: {len(res)} sessions (stored {len(stored)}) ===")
    print(f"R0 on seed 0: d72 keys differing {len(diff)} {diff[:5]}")
    print("d82 keys written at the last session:", sorted(k for k in res[-1] if k.startswith("d82 "))[:6])
    print("exit 0")
