"""D72 diagnosis (2026-10-18; the user's decision Q14 "세가지 다 너의 추천대로 해보자." -- diagnose first why the category
rule goes BELOW chance with exposure, finding 162). Read-only: the same CAT lives run again with a probe beside the keys;
nothing the learner does changes (checked: every key of the probed life must equal the cached life's exactly), nothing is
written to the main cache, no resume state (RIE_NO_RESUME), and no code of the brain is changed (D72 principles 3 and 10 --
the five subjects are measured with the frozen brain; a diagnosis that led to a fix would make the subjects "used").

At every session, for every CAT test item (novel picture -> word, word -> novel picture; four candidates, the first right),
the answer's score split into its parts (run_long.answer_parts, finding 112's exact split): habit similarity "sim", habit
connection table "assoc", templates "pattern", whole-sentence memory "chunk", self-made side part "side", world model "wm"
-- and the full answer. Per candidate also the times its family word was heard so far (k). The readout asks, per part:
(1) which part puts the right candidate below the others, and from which k on; (2) whether a part scores a candidate LOWER
the more its family word was heard (the slope of the part's score on the candidate's k, within an item).
    python3 diag_d72cat.py [SEEDS=0,1,2] [OUT=diag_d72cat]   (run in rie/, next to the cache)"""
import json
import os
import pickle
import sys

os.environ["RIE_NO_RESUME"] = "1"                   # no resume state for a probe life

import numpy as np
import torch
import torch.nn.functional as F

import lab2 as L
import run_long as RL

COND = "both@ascii@d72cat+rediscover+chain+wm+wmctx+wmlast"
MAIN_CACHE = RL.CACHE
PARTS = ("sim", "assoc", "pattern", "chunk", "side", "wm")
REC = []
_orig_keys = RL.d72_keys


def probe(model, w, seen):
    words = list(w.d72_words)
    k = [float(sum(t_ in s_ for c in seen for k_, s_, _ in w.categories[c]["heard"] if k_ == "d72-heard")) for t_ in words]
    items = list(w.d72_tests)
    cx, _ = L.items_tensor([(s_, a_[0]) for _, s_, a_ in items])
    parts, blocked, err = RL.answer_parts(model, cx)
    with torch.no_grad():
        full = F.log_softmax(model.complete(cx), dim=1)
    rows = []
    for i, (key, _, a_) in enumerate(items):
        kind, f = key.split()[1], int(key.split()[2])
        fams = [f] + [g for g in range(len(words)) if g != f]          # the candidates' families, in the item's order
        a = torch.tensor(a_)
        row = {"kind": kind, "f": f, "cand_k": [k[g] for g in fams], "full": full[i, a].tolist(),
               "blocked": blocked[i, a].tolist()}
        for p in PARTS:
            if p in parts:
                row[p] = parts[p][i, a].tolist()
        rows.append(row)
    return {"k": k, "split_err": float(err), "rows": rows}


def probe_keys(model, w, seen, tag):
    out = _orig_keys(model, w, seen, tag)
    if tag == "" and getattr(w, "d72", None) == "cat":
        REC.append(probe(model, w, seen))
    return out


RL.d72_keys = probe_keys


def same(a, b):
    # (fixed after the first run, 2026-10-18: NaN inside the life's dict-valued keys made plain == call every life
    # different -- the probe's own answers reproduced every cached "d72" key exactly, 384 of 384; diag_d72cat.md)
    from run_d72b import deep_eq
    return deep_eq(a, b)


def one_seed(args):
    seed, out_dir = args
    torch.set_num_threads(1)
    REC.clear()
    res = RL.life(COND, seed)
    with open(os.path.join(MAIN_CACHE, f"seed{seed}_{COND}.pkl"), "rb") as f:
        cached = pickle.load(f)
    eq = same(res, cached)
    with open(os.path.join(out_dir, f"seed{seed}.json"), "w") as f:
        json.dump({"seed": seed, "same_as_cached": eq, "sessions": REC}, f)
    return seed, eq, len(REC)


if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
    out_dir = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "diag_d72cat")
    os.makedirs(out_dir, exist_ok=True)
    RL.CACHE = os.path.join(out_dir, "cache")         # the probe's own side paths (progress lines, cost files)
    RL.PROGRESS = os.path.join(out_dir, "progress")
    RL.RESUME = os.path.join(out_dir, "resume")
    import multiprocessing as mp
    with mp.get_context("fork").Pool(len(seeds)) as pool:
        for seed, eq, n in pool.imap_unordered(one_seed, [(s, out_dir) for s in seeds]):
            print(f"seed {seed}: sessions probed {n}, keys same as the cached life: {eq}", flush=True)
    print("exit 0")
