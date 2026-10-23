"""D72 subject 2 seed-0 smoke (2026-10-18, before any life of run_d72b.py): the two new conditions run one life each
(seed 0) beside the main cache (no resume state, nothing written to the main cache) and the smoke reads NO outcome key:
  (1) the PREMISE of the new control (subject 1's lesson, finding 162; Pioneer's P52 check): on the chain questions, how
      often the world model speaks at all (its part is non-zero on some candidate) -- per session, with the times the two
      words were heard so far; if it never speaks once both words are heard, the control is wrong again and nothing is queued;
  (2) same learning: the CHAIN life's every key that the cached VERB life has (seed 0) is equal -- the chain questions are
      records only;
  (3) NEG: the life runs to its end, the negators' times heard and the number of test items per session.
    python3 smoke_d72b.py [OUT=smoke_d72b]   (run in rie/, next to the cache)"""
import json
import os
import pickle
import sys

os.environ["RIE_NO_RESUME"] = "1"

import numpy as np
import torch

import lab2 as L
import run_long as RL
from run_d72b import CHAIN, NEG, VERB, deep_eq

MAIN_CACHE = RL.CACHE
REC = []
_orig_keys = RL.d72_keys


def probe_keys(model, w, seen, tag):
    out = _orig_keys(model, w, seen, tag)
    if tag:
        return out
    k = [float(sum(t_ in s_ for c in seen for k_, s_, _ in w.categories[c]["heard"] if k_ == "d72-heard"))
         for t_ in w.d72_words]
    items = [x for c in seen for x in w.categories[c].get("d72", [])]
    row = {"k": k, "items": {}}
    for kind in ("neg", "chain", "verb"):
        it = [x for x in items if x[0].startswith(f"d72 {kind} ")]
        row["items"][kind] = len(it)
        if kind == "chain" and it:
            cx, _ = L.items_tensor([(s_, a_[0]) for _, s_, a_ in it])
            wm = getattr(model, "wm", None)
            with torch.no_grad():
                t = wm.term(cx) if wm is not None else torch.zeros(len(it), 1)
            spoke = (t > 0).any(1)
            both = [min(k[int(key.split()[2][0])], k[int(key.split()[2][1])]) >= 1 for key, _, _ in it]
            row["chain_spoke"] = int(spoke.sum())
            row["chain_both_heard"] = int(sum(both))
            row["chain_spoke_both_heard"] = int(sum(1 for s, b in zip(spoke.tolist(), both) if s and b))
    REC.append(row)
    return out


RL.d72_keys = probe_keys


def one(args):
    cond, out_dir = args
    torch.set_num_threads(1)
    REC.clear()
    res = RL.life(cond, 0)
    eq = None
    if cond == CHAIN:
        with open(os.path.join(MAIN_CACHE, f"seed0_{VERB}.pkl"), "rb") as f:
            v = pickle.load(f)
        eq = len(res) == len(v) and all(deep_eq({k: x.get(k) for k in y}, y) for x, y in zip(res, v))
    with open(os.path.join(out_dir, f"{'chain' if cond == CHAIN else 'neg'}_seed0.json"), "w") as f:
        json.dump({"cond": cond, "sessions": REC, "same_as_verb": eq}, f)
    return cond, eq, list(REC)


if __name__ == "__main__":
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "smoke_d72b")
    os.makedirs(out_dir, exist_ok=True)
    RL.CACHE, RL.PROGRESS, RL.RESUME = (os.path.join(out_dir, x) for x in ("cache", "progress", "resume"))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(2) as pool:
        for cond, eq, rec in pool.imap_unordered(one, [(NEG, out_dir), (CHAIN, out_dir)]):
            print(f"=== {cond} seed 0: {len(rec)} sessions ===")
            for i, r in enumerate(rec):
                extra = (f" | chain items {r['items']['chain']}, both words heard {r.get('chain_both_heard', 0)}, the world "
                         f"model spoke on {r.get('chain_spoke', 0)} (on {r.get('chain_spoke_both_heard', 0)} of those with "
                         f"both heard)" if cond == CHAIN else f" | neg items {r['items']['neg']}")
                print(f"session {i + 1}: k {[int(x) for x in r['k']]}{extra}")
            if cond == CHAIN:
                print(f"CHAIN life = cached VERB life on every VERB key: {eq}")
    print("exit 0")
