"""Track A W3 checks (2026-10-13).
  world: every world without the new option is the same as before W3 (data_ascii before the change, kept at
         ../data_ascii_before_w3.py): 18 worlds x seeds 0, 3, 5 -- every attribute, every list of every category.
  reg:   the D25 / finding 143-144 Voyager B life (both@ascii+rediscover+chain+wm+wmlast, seed 0) under the new code must
         equal its cache exactly (run_long / integrated changed only for "@sp").
  P / S: seed-0 smoke of run_spatial.CONDS (the "@sp" world / the "@spshuf" shuffle control): sane keys, no crash."""
import importlib.util
import os
import pickle
import sys
import time

import numpy as np
import torch

import run_long as RL

torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "world":
    import data_ascii as new
    spec = importlib.util.spec_from_file_location("old_da", os.path.join("..", "data_ascii_before_w3.py"))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    worlds = {"ascii": {}, "kor": dict(korean=True), "multi": dict(multi=True), "objs": dict(props=True),
              "grav": dict(grav=True), "tale": dict(tale=True), "frame": dict(frame=True), "contain": dict(contain=True),
              "count": dict(count=True), "succ": dict(succ=True), "numrow": dict(numline="row"),
              "numown": dict(numline="own"), "views": dict(views=True), "viewpt": dict(views=2), "goal": dict(goal=True),
              "self": dict(self_=True), "nw": dict(nw=True), "hold": dict(hold=True)}

    def same(a, b):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    bad = []
    for name, kw in worlds.items():
        for seed in (0, 3, 5):
            wo, wn = old.build(seed, **kw), new.build(seed, **kw)
            ka, kb = set(vars(wo)), set(vars(wn))
            diff = sorted(k for k in ka | kb if k not in ka or k not in kb or not same(vars(wo)[k], vars(wn)[k]))
            if diff:
                bad.append((name, seed, diff))
    print(f"world regression: {len(worlds)} worlds x 3 seeds, differing: {bad if bad else 'none'}")
elif which == "reg":
    K = "both@ascii+rediscover+chain+wm+wmlast"
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    same_keys = all(set(a) == set(b) for a, b in zip(old, out)) and len(old) == len(out)
    nan_same = all((isinstance(a[k], float) and np.isnan(a[k])) == (isinstance(b[k], float) and np.isnan(b[k]))
                   for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float))
    print(which, "regression max diff", d, "| same records/keys", same_keys, "| nan pattern same", nan_same)
else:
    import run_spatial
    out = RL.one((0, run_spatial.CONDS[which]))
    keys = ["sp word4", "sp scene4", "sp word", "sp scene", "sp word4 body", "sp scene4 body"] + \
        [f"sp {k}4 {r}" for k in ("word", "scene") for r in ("WI", "ARAE", "ANE", "YEOPE")] + \
        ["contain in", "contain by", "multi", "never-heard", "chain"]
    for k in keys:
        print(f"{which} {k:20}", " ".join("  -  " if np.isnan(st.get(k, np.nan)) else f"{st[k]:.3f}" for st in out))
print("time", round(time.time() - t0))
