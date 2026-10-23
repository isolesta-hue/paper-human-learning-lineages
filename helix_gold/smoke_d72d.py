"""D72 subject 4 seed-0 smoke (2026-10-22, before any life of run_d72d.py): the NEXT condition runs one life (seed 0) beside
the main cache (no resume state, nothing written to the main cache) and reads NO outcome key -- only that the life runs to
its end, the bells' times heard, the number of test items per session and that the device's " eq1" keys are written.
    python3 smoke_d72d.py [OUT=smoke_d72d]"""
import os
import sys

os.environ["RIE_NO_RESUME"] = "1"

import run_long as RL
from run_d72d import NEXT

REC = []
_orig_keys = RL.d72_keys


def probe_keys(model, w, seen, tag, scorer=None):
    out = _orig_keys(model, w, seen, tag, scorer=scorer)
    if tag in ("", " eq1"):
        k = [int(sum(t_ in s_ for c in seen for k_, s_, _ in w.categories[c]["heard"] if k_ == "d72-heard"))
             for t_ in w.d72_words]
        REC.append((tag or "usual", k, sum(1 for c in seen for x in w.categories[c].get("d72", []) if x[0].startswith("d72 next ")),
                    sorted(key for key in out if key.startswith("d72 next "))[:2]))
    return out


RL.d72_keys = probe_keys

if __name__ == "__main__":
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "smoke_d72d")
    os.makedirs(out_dir, exist_ok=True)
    RL.CACHE, RL.PROGRESS, RL.RESUME = (os.path.join(out_dir, x) for x in ("cache", "progress", "resume"))
    res = RL.life(NEXT, 0)
    print(f"=== {NEXT} seed 0: {len(res)} sessions ===")
    for i, (tag, k, n, names) in enumerate(REC):
        print(f"call {i + 1} ({tag}): k {k} | next items {n} | keys e.g. {names}")
    print("exit 0")
