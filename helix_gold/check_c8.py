"""Track A W7b checks for the criterion-8 world "@c8" (2026-10-16; SCORE_TABLE.md W7b, the user's "열쇠는 너의 생각에 따를게").
  world: every world without the new option is the same as before (data_ascii of the queue directory ../rie, unchanged):
         26 worlds x seeds 0, 3, 5 -- every attribute, every list of every category. And "@c8" CONTAINS "@multi": every
         "@multi" token keeps its id and features (one new token JEONE, last), every "@multi" category list is the same
         except "heard", whose "@multi" part is the same prefix (the extra heard moments are "c8-before-heard" only).
  items: every "@c8" / "@c8shuf" item, seeds 0-9, read back by an independent simulation: each clause's mover (the noun
         before GA), landmark (before ESEO) and motion word; a mover starts at its landmark's naming place (from the naming
         moments) the first time and goes on from where it is after that; one move per motion word (reversed in upside-down
         rooms); no two movers in one cell. "before": A moves at least twice, the answer = at(A's cell before its last
         move), different from A's end; "long": 4-6 clauses, the answer = to(A's end); heard "before" in "@c8" = the same
         rule, in "@c8shuf" a random cell (its share equal to the rule's cell reported, chance 1/25).
  reg:   the "@multi" body's life (both@ascii@multi+rediscover+chain+wm+bind+wmlast+track, seed 0) under the new code must
         equal its cache exactly (NaN-aware deep compare + pickle bytes).
    python3 check_c8.py world|items|reg"""
import importlib.util
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

which, t0 = sys.argv[1], time.time()


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


if which == "world":
    import data_ascii as new
    spec = importlib.util.spec_from_file_location("old_da", os.path.join("..", "rie", "data_ascii.py"))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    worlds = {"ascii": {}, "kor": dict(korean=True), "multi": dict(multi=True), "objs": dict(props=True),
              "grav": dict(grav=True), "tale": dict(tale=True), "frame": dict(frame=True), "contain": dict(contain=True),
              "count": dict(count=True), "succ": dict(succ=True), "numrow": dict(numline="row"),
              "numown": dict(numline="own"), "views": dict(views=True), "viewpt": dict(views=2), "goal": dict(goal=True),
              "self": dict(self_=True), "nw": dict(nw=True), "hold": dict(hold=True), "sp": dict(sp=True),
              "spshuf": dict(sp_shuffle=True), "when": dict(when=True), "whenshuf": dict(when_shuffle=True),
              "ten": dict(ten=True), "tenshuf": dict(ten_shuffle=True), "bi": dict(bi=True), "bishuf": dict(bi_shuffle=True)}
    bad = []
    for name, kw in worlds.items():
        for seed in (0, 3, 5):
            wo, wn = old.build(seed, **kw), new.build(seed, **kw)
            ka, kb = set(vars(wo)), set(vars(wn))
            diff = sorted(k for k in ka | kb if k not in ka or k not in kb or not same(vars(wo)[k], vars(wn)[k]))
            if diff:
                bad.append((name, seed, diff))
    print(f"world regression: {len(worlds)} worlds x 3 seeds, differing: {bad if bad else 'none'}", flush=True)
    bad2 = []
    for seed in (0, 3, 5):
        k, b = new.build(seed, multi=True), new.build(seed, c8=True)
        if b.V != k.V + 1 or not np.array_equal(b.vecs[:k.V], k.vecs) or list(b.cats[:k.V]) != list(k.cats) \
                or b.names[:k.V] != k.names or b.names[k.V] != "jeone":
            bad2.append((seed, "tokens"))
        for c in range(k.n_cat):
            ck, cb = k.categories[c], b.categories[c]
            if set(cb) - set(ck) - {"c8"}:
                bad2.append((seed, c, "extra keys", set(cb) - set(ck)))
            for key in ck:
                if key == "heard":
                    if not same(cb["heard"][:len(ck["heard"])], ck["heard"]):
                        bad2.append((seed, c, "heard prefix"))
                    extra = {x[0] for x in cb["heard"][len(ck["heard"]):]}
                    if not extra <= {"c8-before-heard"}:
                        bad2.append((seed, c, "heard extra", extra))
                elif not same(cb[key], ck[key]):
                    bad2.append((seed, c, key))
    print(f"'@c8' contains '@multi' (seeds 0, 3, 5): {bad2 if bad2 else 'yes -- tokens and lists the same, heard = prefix'}",
          flush=True)

elif which == "items":
    import data_ascii as D
    probs, n_items, spread = Counter(), Counter(), Counter()
    for world in ("c8", "c8shuf"):
        for seed in range(10):
            w = D.build(seed, **{("c8" if world == "c8" else "c8_shuffle"): True})
            to_of = {t: p for t, p in w.pos_of.items() if w.role_of[t] == "to"}
            at_of = {t: p for t, p in w.pos_of.items() if w.role_of[t] == "at"}
            P = w.particle_of
            GA, ESEO, GO, EODI, JEONE = P["GA"], P["ESEO"], P["GO"], P["EODI"], P["JEONE"]
            verbs = {t: k for t, k in w.verb_of.items() if k < 5}
            for c, cat in enumerate(w.categories):
                if c == 0:
                    continue
                names = [o[0] for o in cat["objects"]]
                place = {}
                for k_, s_, _ in cat["heard"]:
                    if k_ in ("name", "name-once", "name-conv"):
                        nm = [t for t in s_ if t in names]
                        ats = [t for t in s_ if t in at_of]
                        if len(nm) == 1 and len(ats) == 1:
                            place.setdefault(nm[0], set()).add(ats[0])
                if cat.get("placement"):
                    continue
                if any(len(v) != 1 for v in place.values()) or len(place) != 4:
                    probs["naming place not unique"] += 1
                    continue
                place = {k: at_of[next(iter(v))] for k, v in place.items()}
                sign = -1 if cat["flipped"] is True else 1
                if cat["flipped"] not in (True, False):
                    probs["unexpected frame"] += 1

                def run(story):
                    """-> (positions of each mover after each of its moves, None if a rule is broken)."""
                    clauses, cur = [], []
                    for t in story:
                        if t == GO:
                            clauses.append(cur)
                            cur = []
                        else:
                            cur.append(t)
                    if cur:
                        clauses.append(cur)
                    pos, hist = {}, {}
                    for cl in clauses:
                        m_ = [cl[i - 1] for i in range(1, len(cl)) if cl[i] == GA]
                        l_ = [cl[i - 1] for i in range(1, len(cl)) if cl[i] == ESEO]
                        v_ = [t for t in cl if t in verbs]
                        if len(m_) != 1 or len(v_) != 1 or len(l_) > 1:
                            return None, "clause form"
                        m_ = m_[0]
                        if m_ not in pos:
                            if len(l_) != 1:
                                return None, "first move without a landmark"
                            pos[m_] = place[l_[0]]
                        elif l_:
                            return None, "landmark on a later move"
                        dx, dy = D.VERBS[verbs[v_[0]]][1]
                        x, y = pos[m_]
                        pos[m_] = (x + sign * dx, y + sign * dy)
                        if max(abs(pos[m_][0]), abs(pos[m_][1])) > D.HALF:
                            return None, "off the grid"
                        if sum(1 for o, p in pos.items() if p == pos[m_]) > 1:
                            return None, "two in one cell"
                        hist.setdefault(m_, []).append(pos[m_])
                    return hist, None

                for k_, s_, a_ in cat["heard"]:
                    if k_ != "c8-before-heard":
                        continue
                    n_items[(world, k_)] += 1
                    story, q, ans = s_[:-5], s_[-5:-1], s_[-1]
                    if q[1:] != [GA, JEONE, EODI] or ans not in at_of:
                        probs[f"{world} heard form"] += 1
                        continue
                    hist, err = run(story)
                    if err:
                        probs[f"{world} heard: {err}"] += 1
                        continue
                    h = hist.get(q[0], [])
                    if len(h) < 2:
                        probs[f"{world} heard: A moves < 2"] += 1
                        continue
                    hit = at_of[ans] == h[-2]
                    spread[(world, "heard before = the rule's cell")] += hit
                    spread[(world, "heard before")] += 1
                    if world == "c8" and not hit:
                        probs["c8 heard before: wrong cell"] += 1
                for k_, s_, a_ in cat.get("c8", []):
                    n_items[(world, k_)] += 1
                    if k_.startswith("c8 before"):
                        story, q = s_[:-4], s_[-4:]
                        if story[-1] != GO or q[1:] != [GA, JEONE, EODI]:
                            probs[f"{k_} form"] += 1
                            continue
                        hist, err = run(story[:-1])
                        if err:
                            probs[f"{k_}: {err}"] += 1
                            continue
                        h = hist.get(q[0], [])
                        if len(h) < 2 or a_[0] not in at_of or at_of[a_[0]] != h[-2] or h[-2] == h[-1]:
                            probs[f"{k_}: answer"] += 1
                    else:
                        story, q = s_[:-3], s_[-3:]
                        if story[-1] != GO or q[1:] != [GA, EODI]:
                            probs[f"{k_} form"] += 1
                            continue
                        hist, err = run(story[:-1])
                        if err:
                            probs[f"{k_}: {err}"] += 1
                            continue
                        n_cl = story.count(GO)
                        h = hist.get(q[0], [])
                        if not 4 <= n_cl <= 6 or len(h) < 2 or a_[0] not in to_of or to_of[a_[0]] != h[-1]:
                            probs[f"{k_}: answer or length"] += 1
    print(f"items read back (seeds 0-9): {dict(sorted(n_items.items()))}")
    print(f"heard 'before' answers equal to the rule's cell: " +
          ", ".join(f"{w_} {spread[(w_, 'heard before = the rule' + chr(39) + 's cell')]}/{spread[(w_, 'heard before')]}"
                    for w_ in ("c8", "c8shuf")))
    print(f"problems: {dict(probs) if probs else 'none'}")

elif which == "reg":
    import torch
    import run_long as RL
    torch.set_num_threads(1)
    cond = "both@ascii@multi+rediscover+chain+wm+bind+wmlast+track"
    path = os.path.join(RL.CACHE, f"seed0_{cond}.pkl")
    cached = pickle.load(open(path, "rb"))
    fresh = RL.life(cond, 0)
    keys = sorted({k for st in cached + fresh for k in st})
    diff = [k for k in keys if not all(same(a.get(k), b.get(k)) for a, b in zip(cached, fresh))]
    print(f"life regression {cond} seed 0: sessions {len(fresh)} / {len(cached)}, keys differing "
          f"{diff if diff else 'none'}; pickle bytes equal to the cache: {pickle.dumps(fresh) == open(path, 'rb').read()}")
print(f"exit 0 ({time.time() - t0:.0f} s)")
