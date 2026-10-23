"""Track A W4 checks (2026-10-13).
  world: every world without the new option is the same as before W4 (data_ascii before the change, kept at
         ../data_ascii_before_w4.py): 20 worlds x seeds 0, 3, 5 -- every attribute, every list of every category.
  items: every "when" item of the "@when" / "@whenshuf" worlds, seeds 0-9, read back by an independent simulation of its
         story (landmark places from the rooms' naming moments, VERBS, upside-down rooms): the asked object moves exactly
         twice, its start and two arrivals differ, the asked cell is said nowhere in the story, exactly one clause brings
         it there, the answer word is that clause's moment word; echo questions repeat one clause that is unique in the
         story; candidates = the story's n moment words, the right one first; heard answers right in "@when" and drawn
         among the story's moment words in "@whenshuf"; the answer moment's spread per n; and the order-aware word-overlap
         baseline L (each clause scored by the tokens it shares with the question, ties broken at random) on both forms.
  reg:   the D25 / finding 143-144 Voyager B life (both@ascii+rediscover+chain+wm+wmlast, seed 0) under the new code must
         equal its cache exactly (run_long / integrated changed only for "@when").
  P / S: seed-0 smoke of run_when.CONDS (the "@when" world / the "@whenshuf" shuffle control): sane keys, no crash."""
import importlib.util
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import torch

import run_long as RL

torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "world":
    import data_ascii as new
    spec = importlib.util.spec_from_file_location("old_da", os.path.join("..", os.environ.get("OLD_DA", "data_ascii_before_w4.py")))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    worlds = {"ascii": {}, "kor": dict(korean=True), "multi": dict(multi=True), "objs": dict(props=True),
              "grav": dict(grav=True), "tale": dict(tale=True), "frame": dict(frame=True), "contain": dict(contain=True),
              "count": dict(count=True), "succ": dict(succ=True), "numrow": dict(numline="row"),
              "numown": dict(numline="own"), "views": dict(views=True), "viewpt": dict(views=2), "goal": dict(goal=True),
              "self": dict(self_=True), "nw": dict(nw=True), "hold": dict(hold=True), "sp": dict(sp=True),
              "spshuf": dict(sp_shuffle=True), "when": dict(when=True), "whenshuf": dict(when_shuffle=True)}

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
elif which == "items":
    import data_ascii as D
    probs, spread, L_hits = Counter(), Counter(), Counter()
    n_items = Counter()
    for shuffle in (False, True):
        for seed in range(10):
            w = D.build(seed, when=not shuffle, when_shuffle=shuffle)
            nm = {t: w.names[t] for t in range(w.V)}
            ga, eseo, go, eonje = (w.particle_of[x] for x in ("GA", "ESEO", "GO", "EONJE"))
            mw = w.moment_words
            pos_tok = set(w.pos_of)
            lrng = np.random.default_rng(seed)
            for cat in w.categories[1:]:
                obj_at = {}
                for kind_, s_, _ in cat["heard"]:
                    if kind_.startswith("name"):
                        ps = [x for x in s_ if x in pos_tok]
                        for x in s_:
                            if w.kind_of[x] == w.kinds.index("name"):
                                obj_at[x] = w.pos_of[ps[0]]
                flipped = cat["flipped"]
                items = [(k_, s_, a_, "test") for k_, s_, a_ in cat.get("when", [])] + \
                    [(k_, s_[:-1], [s_[-1]], "heard") for k_, s_, _ in cat["heard"] if k_ == "when-heard"]
                for kind_, s_, ans, bucket in items:
                    segs, cur = [], []
                    for x in s_:
                        if x == go:
                            segs.append(cur)
                            cur = []
                        else:
                            cur.append(x)
                    q, clauses = cur, segs
                    n = len(clauses)
                    tag = f"{'shuf' if shuffle else 'when'} {bucket}"
                    n_items[tag] += 1
                    if any(x in pos_tok for cl in clauses for x in cl):
                        probs[f"{tag}: a position said in the story"] += 1
                    state, start, arrive, first = {}, {}, {}, set()
                    ok = True
                    for j, cl in enumerate(clauses):
                        mover = cl[cl.index(ga) - 1]
                        lm = cl[cl.index(eseo) - 1] if eseo in cl else None
                        verb = [x for x in cl if x in w.verb_of]
                        if len(verb) != 1 or (mover in first) == (lm is not None):
                            probs[f"{tag}: clause shape"] += 1
                            ok = False
                            break
                        if lm is not None:
                            state[mover] = obj_at[lm]
                            start[mover] = obj_at[lm]
                        first.add(mover)
                        d = D.VERBS[w.verb_of[verb[0]]][1]
                        if flipped is True:
                            d = (-d[0], -d[1])
                        state[mover] = (state[mover][0] + d[0], state[mover][1] + d[1])
                        if max(abs(state[mover][0]), abs(state[mover][1])) > D.HALF:
                            probs[f"{tag}: off the grid"] += 1
                        arrive.setdefault(mover, []).append((state[mover], j))
                    if not ok:
                        continue
                    if q[0] == eonje:                  # echo
                        form = "echo"
                        body = q[1:]
                        hits = [j for j, cl in enumerate(clauses) if sorted(cl) == sorted(body)]
                        if len(hits) != 1 or clauses[hits[0]] != body:
                            probs[f"{tag}: echo clause not unique / not repeated"] += 1
                            continue
                        j = hits[0]
                    else:
                        form = "to"
                        if len(q) != 4 or q[1] != ga or q[3] != eonje or w.role_of.get(q[2]) != "to":
                            probs[f"{tag}: question shape"] += 1
                            continue
                        x, p = q[0], w.pos_of[q[2]]
                        arr = arrive.get(x, [])
                        if len(arr) != 2:
                            probs[f"{tag}: asked object does not move twice"] += 1
                            continue
                        if len({start[x]} | {a for a, _ in arr}) != 3:
                            probs[f"{tag}: start / arrivals not distinct"] += 1
                        hits = [j for a, j in arr if a == p]
                        if len(hits) != 1:
                            probs[f"{tag}: asked cell not reached exactly once"] += 1
                            continue
                        j = hits[0]
                    right = mw[j]
                    if bucket == "test":
                        if ans[0] != right or sorted(ans) != sorted(mw[:n]):
                            probs[f"{tag}: answer / candidates wrong"] += 1
                        spread[(form, n, j)] += 1
                        # L: the clause sharing the most tokens with the question (ties at random)
                        sc = [len(set(cl) & set(q)) for cl in clauses]
                        best = [i for i, v in enumerate(sc) if v == max(sc)]
                        L_hits[(form, n)] += int(best[int(lrng.integers(len(best)))] == j)
                        L_hits[(form, n, "n")] += 1
                    else:
                        if not shuffle and ans[0] != right:
                            probs[f"{tag}: heard answer wrong"] += 1
                        if shuffle and ans[0] not in mw[:n]:
                            probs[f"{tag}: shuffled heard answer not a moment of the story"] += 1
                        if shuffle:
                            spread[("shuf heard right", n)] += int(ans[0] == right)
                            spread[("shuf heard n", n)] += 1
    print("items checked:", dict(n_items))
    print("problems:", dict(probs) if probs else "none")
    for form in ("to", "echo"):
        for n in (2, 3, 4):
            cnt = [spread[(form, n, j)] for j in range(n)]
            print(f"  {form:4} n {n}: answer moment spread {cnt} | L (word overlap) "
                  f"{L_hits[(form, n)] / max(L_hits[(form, n, 'n')], 1):.3f} (chance {1 / n:.3f})")
    for n in (2, 3, 4):
        print(f"  @whenshuf heard answers right: n {n} {spread[('shuf heard right', n)] / max(spread[('shuf heard n', n)], 1):.3f}"
              f" (expected {1 / n:.3f})")
elif which == "reg":
    K = "both@ascii+rediscover+chain+wm+wmlast"
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    same_keys = all(set(a) == set(b) for a, b in zip(old, out)) and len(old) == len(out)
    nan_same = all((isinstance(a[k], float) and np.isnan(a[k])) == (isinstance(b[k], float) and np.isnan(b[k]))
                   for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float))
    print(which, "regression max diff", d, "| same records/keys", same_keys, "| nan pattern same", nan_same)
else:
    import run_when
    out = RL.one((0, run_when.CONDS[which]))
    keys = [f"when {f}{x}" for f in ("to", "echo") for x in ("", " all", " two", " three", " four", " body")] + \
        ["multi", "multi same", "multi two", "multi three", "multi wm fired", "chain", "never-heard"]
    for k in keys:
        print(f"{which} {k:20}", " ".join("  -  " if np.isnan(st.get(k, np.nan)) else f"{st[k]:.3f}" for st in out))
print("time", round(time.time() - t0))
