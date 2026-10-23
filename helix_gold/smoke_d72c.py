"""D72 subject 3 seed-0 smoke (2026-10-18, before any life of run_d72c.py): the ICE condition runs one life (seed 0) beside
the main cache (no resume state, nothing written to the main cache) and reads NO outcome key -- only that the life runs to
its end, the floors' times heard and the number of test items per session.
    python3 smoke_d72c.py [OUT=smoke_d72c]   (run in rie/, next to the cache)"""
import os
import sys

os.environ["RIE_NO_RESUME"] = "1"

import run_long as RL
from run_d72c import ICE

REC = []
_orig_keys = RL.d72_keys


def probe_keys(model, w, seen, tag):
    out = _orig_keys(model, w, seen, tag)
    if not tag:
        k = [int(sum(t_ in s_ for c in seen for k_, s_, _ in w.categories[c]["heard"] if k_ == "d72-heard"))
             for t_ in w.d72_words]
        REC.append((k, sum(1 for c in seen for x in w.categories[c].get("d72", []) if x[0].startswith("d72 ice "))))
    return out


RL.d72_keys = probe_keys

if __name__ == "__main__":
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "smoke_d72c")
    os.makedirs(out_dir, exist_ok=True)
    RL.CACHE, RL.PROGRESS, RL.RESUME = (os.path.join(out_dir, x) for x in ("cache", "progress", "resume"))
    res = RL.life(ICE, 0)
    print(f"=== {ICE} seed 0: {len(res)} sessions ===")
    for i, (k, n) in enumerate(REC):
        print(f"session {i + 1}: k {k} | ice items {n}")
    print("exit 0")
