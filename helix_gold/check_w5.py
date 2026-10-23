"""Track A W5 checks (2026-10-14).
  world: every world without the new option is the same as before W5 (data_ascii before the change, kept at
         ../data_ascii_before_w5.py): 22 worlds x seeds 0, 3, 5 -- every attribute, every list of every category.
  items: every "@ten" / "@tenshuf" item, seeds 0-9, read back by an independent simulation of its story (the container's
         cell, each thing put in at that cell with ANE, each take-out to another cell, the told count of "ten next"): no
         thing twice, the container never a thing, the answer word = the count's word, the candidates = the ten words with
         the right one first; "ten pick": the right story first with n, three other distinct counts, the same container and
         cell; "voe ten": the wrong word one more or one fewer; box groups (held pairs / a 6-10 / heard pairs); heard
         answers right in "@ten" and drawn at random in "@tenshuf"; heard box stories never a held pair; spreads per n.
  reg:   the D25 / finding 143-144 Voyager B life (both@ascii+rediscover+chain+wm+wmlast, seed 0) under the new code must
         equal its cache exactly (run_long / integrated changed only for "@ten").
  P / S: seed-0 smoke of run_ten.CONDS (the "@ten" world / the "@tenshuf" shuffle control): sane keys, no crash."""
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
NUMS = ("HANA", "DUL", "SET", "NET", "DASEOT", "YEOSEOT", "ILGOP", "YEODEOL", "AHOP", "YEOL")
HELD = {(3, 1), (4, 2), (5, 3)}
if which == "world":
    import data_ascii as new
    spec = importlib.util.spec_from_file_location("old_da", os.path.join("..", os.environ.get("OLD_DA", "data_ascii_before_w5.py")))
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
    probs, spread, n_items = Counter(), Counter(), Counter()
    for shuffle in (False, True):
        for seed in range(10):
            w = D.build(seed, ten=not shuffle, ten_shuffle=shuffle)
            nm = {t: w.names[t] for t in range(w.V)}
            ga, ane, go, myeot = (w.particle_of[x] for x in ("GA", "ANE", "GO", "MYEOT"))
            word = {w.numbers[x]: i + 1 for i, x in enumerate(NUMS)}
            assert [w.numbers[x] for x in NUMS] == list(w.ten_words)
            name_k = w.kinds.index("name")
            pos_at = {t: p for p, t in w.pos_at.items()}
            tag = "shuf" if shuffle else "ten"

            def simulate(s_):
                """-> (container, cell, count, told, put, take) or a problem string."""
                segs, cur = [], []
                for x in s_:
                    if x == go:
                        segs.append(cur)
                        cur = []
                    else:
                        cur.append(x)
                q_ = cur
                if len(segs) < 1 or len(segs[0]) != 3 or segs[0][1] != ga or segs[0][2] not in pos_at:
                    return "first moment"
                C, cell = segs[0][0], pos_at[segs[0][2]]
                if w.kind_of[C] != name_k:
                    return "container not a name"
                if q_ != [C, ane, myeot]:
                    return "question"
                inside, used, told, put, take = set(), set(), 0, 0, 0
                for m in segs[1:]:
                    if len(m) == 3 and m[0] == C and m[1] == ane and m[2] in word:
                        if put or take or told:
                            return "told count not first"
                        told = word[m[2]]
                    elif len(m) == 5 and m[1] == ga and m[3] == C and m[4] == ane:
                        if m[2] not in pos_at or pos_at[m[2]] != cell:
                            return "put at another cell"
                        if m[0] == C or m[0] in used or w.kind_of[m[0]] != name_k:
                            return "thing repeated / the container / not a name"
                        used.add(m[0])
                        inside.add(m[0])
                        put += 1
                    elif len(m) == 3 and m[1] == ga and m[2] in pos_at:
                        if m[0] not in inside or pos_at[m[2]] == cell:
                            return "take of a thing not inside / to the same cell"
                        inside.discard(m[0])
                        take += 1
                    else:
                        return "moment shape"
                return C, cell, len(inside) + told, told, put, take
            for cat in w.categories[1:]:
                if cat["placement"]:
                    if cat.get("ten"):
                        probs[f"{tag}: items in a placement room"] += 1
                    continue
                for k_, s_, _ in cat["heard"]:
                    if k_ != "ten-heard":
                        continue
                    n_items[f"{tag} heard"] += 1
                    r = simulate(s_[:-1])
                    if isinstance(r, str):
                        probs[f"{tag} heard: {r}"] += 1
                        continue
                    _, _, cnt, told, put, take = r
                    if take and (put, take) in HELD:
                        probs[f"{tag} heard: a held pair"] += 1
                    if take and put > 5:
                        probs[f"{tag} heard: box with more than 5 put"] += 1
                    spread[(tag, "heard right")] += int(word.get(s_[-1]) == cnt)
                    spread[(tag, "heard n")] += 1
                    spread[(tag, "heard count", cnt, "box" if take else "put")] += 1
                for k_, s_, a_ in cat.get("ten", []):
                    kp = k_.split()
                    n_items[f"{tag} {' '.join(kp[:2]) if kp[0] == 'voe' else kp[0] + ' ' + kp[1]}"] += 1
                    if kp[1] == "pick":
                        rs = [simulate(x) for x in s_]
                        if any(isinstance(r, str) for r in rs):
                            probs[f"{tag} pick: " + [r for r in rs if isinstance(r, str)][0]] += 1
                            continue
                        n = int(kp[2])
                        cnts = [r[2] for r in rs]
                        if len(rs) != 4 or cnts[0] != n or len(set(cnts)) != 4 or any(r[4] != r[2] for r in rs):
                            probs[f"{tag} pick: counts"] += 1
                        if len({(r[0], r[1]) for r in rs}) != 1:
                            probs[f"{tag} pick: container / cell differ"] += 1
                        if len(a_) != 1 or word.get(a_[0]) != n:
                            probs[f"{tag} pick: told word"] += 1
                        spread[(tag, "pick", n)] += 1
                        continue
                    r = simulate(s_)
                    if isinstance(r, str):
                        probs[f"{tag} {kp[1]}: {r}"] += 1
                        continue
                    _, _, cnt, told, put, take = r
                    if kp[0] == "voe":
                        if len(a_) != 2 or word.get(a_[0]) != cnt or abs(word.get(a_[1], 99) - cnt) != 1:
                            probs[f"{tag} voe: answers"] += 1
                        g = kp[2]
                    else:
                        if len(a_) != 10 or len(set(a_)) != 10 or set(a_) != set(word) or word.get(a_[0]) != cnt:
                            probs[f"{tag} {kp[1]}: answer / candidates"] += 1
                        g = kp[2]
                    if kp[1] == "count" and (take or told or cnt != int(g)):
                        probs[f"{tag} count: not put-only n"] += 1
                    if kp[1] == "next" and (take or put != 1 or told != int(g) or cnt != int(g) + 1):
                        probs[f"{tag} next: shape"] += 1
                    if kp[1] in ("box", "ten") or kp[0] == "voe":
                        if told:
                            probs[f"{tag} box: told count"] += 1
                        if g == "held" and (put, take) not in HELD:
                            probs[f"{tag} box held: not a held pair"] += 1
                        if g == "far" and not (6 <= put <= 10 and take < put):
                            probs[f"{tag} box far: shape"] += 1
                        if g == "taught" and ((put, take) in HELD or put > 5 or take >= put):
                            probs[f"{tag} box taught: shape"] += 1
                        spread[(tag, "box", g, put, take)] += 1
                    if kp[1] == "count":
                        spread[(tag, "count", cnt)] += 1
    print("items checked:", dict(n_items))
    print("problems:", dict(probs) if probs else "none")
    for tag in ("ten", "shuf"):
        print(f"  {tag}: heard answers right {spread[(tag, 'heard right')] / max(spread[(tag, 'heard n')], 1):.3f}"
              f" (expected {'1.000' if tag == 'ten' else 'about 0.1'})")
        print(f"  {tag}: heard counts put " + " ".join(f"{n}:{spread[(tag, 'heard count', n, 'put')]}" for n in range(1, 11)) +
              " | box " + " ".join(f"{n}:{spread[(tag, 'heard count', n, 'box')]}" for n in range(1, 11)))
        print(f"  {tag}: test count per n " + " ".join(f"{n}:{spread[(tag, 'count', n)]}" for n in range(1, 11)) +
              " | pick per n " + " ".join(f"{n}:{spread[(tag, 'pick', n)]}" for n in range(1, 11)))
        far = Counter()
        for k, v in spread.items():
            if k[:3] == (tag, "box", "far"):
                far[k[3] - k[4]] += v
        print(f"  {tag}: box far, count inside spread " + " ".join(f"{n}:{far[n]}" for n in range(1, 11)))
elif which == "reg":
    K = "both@ascii+rediscover+chain+wm+wmlast"
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    same_keys = all(set(a) == set(b) for a, b in zip(old, out)) and len(old) == len(out)
    nan_same = all((isinstance(a[k], float) and np.isnan(a[k])) == (isinstance(b[k], float) and np.isnan(b[k]))
                   for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float))
    print(which, "regression max diff", d, "| same records/keys", same_keys, "| nan pattern same", nan_same)
else:
    import run_ten
    out = RL.one((0, run_ten.CONDS[which]))
    keys = [f"ten {f}{x}" for f in ("count", "pick", "next") for x in ("", " all")] + \
        [f"ten count {n}" for n in range(1, 11)] + [f"ten pick {n}" for n in range(1, 11)] + \
        [f"ten box{x}" for x in ("", " new", " held", " far", " taught")] + \
        [f"voe ten{x}" for x in ("", " new", " held", " far", " taught")]
    keys = [k + t for k in keys for t in ("", " body")] + \
        ["contain in", "contain by", "multi", "chain", "never-heard", "count table right"]
    for k in keys:
        print(f"{which} {k:22}", " ".join("  -  " if np.isnan(st.get(k, np.nan)) else f"{st[k]:.3f}" for st in out))
print("time", round(time.time() - t0))
