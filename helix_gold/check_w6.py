"""Track A W6 checks (2026-10-15; W6_DESIGN.md).
  world: every world without the new option is the same as before W6 (data_ascii before the change, kept at
         ../data_ascii_before_w6.py): 24 worlds x seeds 0, 3, 5 -- every attribute, every list of every category. And the
         "@bi" world CONTAINS the "@multi" world (edition 2; edition 1 contained "@kor"): every "@multi" token keeps its id
         and features, every "@multi" category list is the same except "heard", whose "@multi" part is the same prefix.
  items: every "@bi" / "@bishuf" item, seeds 0-9, read back by an independent simulation: the told story's mover and
         landmark (Korean: the nouns before GA / ESEO; English: the noun before the verb / after FROM; an English name is
         mapped to its object through w.en_name), the start = the landmark's naming place, the end = one move of the verb
         (reversed in upside-down rooms) -- the answer = to(end); "what": the candidates = the room's four names in the
         asked language, the mover's first; the question asks about the told mover; the distractor differs from both;
         heard English stories: from(start) / to(end) right; heard English naming moments: in "@bi" the name's own object
         (picture of the same kind as the Korean moment, place), in "@bishuf" a random object of the room (spread).
  reg:   the finding 125 Korean body's life (both@ascii@kor+rediscover+chain+wm+bind+wmlast, seed 0) under the new code
         must equal its cache exactly (run_long / integrated / data_ascii changed only for "@bi").
  P / S: seed-0 smoke of run_bi.CONDS: sane keys, no crash."""
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


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


if which == "world":
    import data_ascii as new
    spec = importlib.util.spec_from_file_location("old_da", os.path.join("..", "data_ascii_before_w6.py"))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    worlds = {"ascii": {}, "kor": dict(korean=True), "multi": dict(multi=True), "objs": dict(props=True),
              "grav": dict(grav=True), "tale": dict(tale=True), "frame": dict(frame=True), "contain": dict(contain=True),
              "count": dict(count=True), "succ": dict(succ=True), "numrow": dict(numline="row"),
              "numown": dict(numline="own"), "views": dict(views=True), "viewpt": dict(views=2), "goal": dict(goal=True),
              "self": dict(self_=True), "nw": dict(nw=True), "hold": dict(hold=True), "sp": dict(sp=True),
              "spshuf": dict(sp_shuffle=True), "when": dict(when=True), "whenshuf": dict(when_shuffle=True),
              "ten": dict(ten=True), "tenshuf": dict(ten_shuffle=True)}
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
        k, b = new.build(seed, multi=True), new.build(seed, bi=True)
        if not np.array_equal(b.vecs[:k.V], k.vecs) or list(b.cats[:k.V]) != list(k.cats) or b.names[:k.V] != k.names:
            bad2.append((seed, "tokens"))
        for c in range(k.n_cat):
            ck, cb = k.categories[c], b.categories[c]
            for key in ck:
                if key == "heard":
                    if not same(cb["heard"][:len(ck["heard"])], ck["heard"]):
                        bad2.append((seed, c, "heard prefix"))
                    extra = {x[0] for x in cb["heard"][len(ck["heard"]):]}
                    if not extra <= {"name-en", "name-en-once", "name-en-conv", "story-en-heard"}:
                        bad2.append((seed, c, "heard extra", extra))
                elif not same(cb[key], ck[key]):
                    bad2.append((seed, c, key))
    print(f"'@bi' contains '@multi' (edition 2; seeds 0, 3, 5): {bad2 if bad2 else 'yes -- tokens and lists the same, heard = prefix'}",
          flush=True)

elif which == "items":
    import data_ascii as D
    probs, spread, n_items = Counter(), Counter(), Counter()
    for world in ("bi", "bishuf"):
        for seed in range(10):
            w = D.build(seed, **{("bi" if world == "bi" else "bi_shuffle"): True})
            to_of = {t: p for t, p in w.pos_of.items() if w.role_of[t] == "to"}
            fr_of = {t: p for t, p in w.pos_of.items() if w.role_of[t] == "from"}
            at_of = {t: p for t, p in w.pos_of.items() if w.role_of[t] == "at"}
            ko_of_en = {v: k for k, v in w.en_name.items()}
            P = w.bi_words
            GA, ESEO, REUL, GO, EODI = P["GA"], P["ESEO"], P["REUL"], P["GO"], P["EODI"]
            FROM, WHERE, IS, WHAT, MUEOS = P["FROM"], P["WHERE"], P["IS"], P["WHAT"], P["MUEOS"]
            verbs = {t: k for t, k in w.verb_of.items() if k < 5}
            for c, cat in enumerate(w.categories):
                if c == 0:
                    continue
                objs = cat["objects"]
                names = [o[0] for o in objs]
                place = {}                              # each Korean name's naming place (from the Korean naming moments)
                for k_, s_, _ in cat["heard"]:
                    if k_ in ("name", "name-once", "name-conv"):
                        nm = [t for t in s_ if t in names]
                        ats = [t for t in s_ if t in at_of]
                        if len(nm) == 1 and len(ats) == 1:
                            place.setdefault(nm[0], set()).add(ats[0])
                if any(len(v) != 1 for v in place.values()) or len(place) != 4:
                    probs["naming place not unique"] += 1
                    continue
                place = {k: next(iter(v)) for k, v in place.items()}
                flipped = cat["flipped"]
                sign = -1 if flipped is True else 1
                if flipped not in (True, False):
                    probs["unexpected frame"] += 1

                def end_of(start_tok, verb_tok):
                    k = verbs[verb_tok]
                    dx, dy = D.VERBS[k][1]
                    x, y = at_of[start_tok]
                    return (x + sign * dx, y + sign * dy)

                def read_ko(sent):
                    m_ = [sent[i - 1] for i in range(1, len(sent)) if sent[i] == GA]
                    l_ = [sent[i - 1] for i in range(1, len(sent)) if sent[i] == ESEO]
                    v_ = [t for t in sent if t in verbs]
                    return m_, l_, v_

                def read_en(sent):
                    v_ = [i for i, t in enumerate(sent) if t in verbs]
                    f_ = [i for i, t in enumerate(sent) if t == FROM]
                    if len(v_) != 1 or len(f_) != 1 or v_[0] != 1:
                        return None
                    return ko_of_en.get(sent[0]), ko_of_en.get(sent[f_[0] + 1]), sent[v_[0]]
                # heard English naming: the shown object (picture kind as the Korean moment; place)
                for k_, s_, _ in cat["heard"]:
                    if not k_.startswith("name-en"):
                        continue
                    en = [t for t in s_ if t in ko_of_en]
                    ats = [t for t in s_ if t in at_of]
                    if len(en) != 1 or len(ats) != 1:
                        probs["en naming form"] += 1
                        continue
                    own = ko_of_en[en[0]]
                    shown = [n for n in names if place[n] == ats[0]]
                    spread[(world, "en naming shows its own object")] += own in shown
                    spread[(world, "en naming moments")] += 1
                    if world == "bi" and own not in shown:
                        probs["bi: en naming shows another object"] += 1
                for k_, s_, _ in cat["heard"]:
                    if k_ != "story-en-heard":
                        continue
                    r = read_en(s_[:4])
                    if r is None or r[0] is None or r[1] is None:
                        probs["heard en story form"] += 1
                        continue
                    m_, l_, v_ = r
                    e = end_of(place[l_], v_)
                    fr = [t for t in s_ if t in fr_of]
                    to = [t for t in s_ if t in to_of]
                    if len(fr) != 1 or len(to) != 1 or fr_of[fr[0]] != at_of[place[l_]] or to_of[to[0]] != e:
                        probs["heard en story scene"] += 1
                    if m_ == l_:
                        probs["heard en story mover = landmark"] += 1
                for k_, s_, a_ in cat.get("bi", []):
                    n_items[(world, k_)] += 1
                    told, ask = (s_[:s_.index(GO)], s_[s_.index(GO) + 1:]) if GO in s_ else (s_, None)
                    lang = "ko" if GA in told else "en"
                    if lang == "ko":
                        m_, l_, v_ = read_ko(told)
                        if len(m_) != 1 or len(l_) != 1 or len(v_) != 1 or m_[0] not in names or l_[0] not in names:
                            probs[f"{k_}: told ko form"] += 1
                            continue
                        m_, l_, v_ = m_[0], l_[0], v_[0]
                        d_ = [told[i - 1] for i in range(1, len(told)) if told[i] == REUL]
                    else:
                        if FROM not in told:
                            probs[f"{k_}: told en form"] += 1
                            continue
                        r = read_en([t for t in told if not (t in ko_of_en and told.index(t) == 2 and len(told) == 5)])
                        if r is None or r[0] is None or r[1] is None:
                            probs[f"{k_}: told en form"] += 1
                            continue
                        m_, l_, v_ = r
                        d_ = [ko_of_en[told[2]]] if len(told) == 5 else []
                    if m_ == l_ or (d_ and d_[0] in (m_, l_)):
                        probs[f"{k_}: nouns not distinct"] += 1
                    if (k_.endswith(" d")) != bool(d_):
                        probs[f"{k_}: distractor presence"] += 1
                    e = end_of(place[l_], v_)
                    if not (-2 <= e[0] <= 2 and -2 <= e[1] <= 2):
                        probs[f"{k_}: end off the grid"] += 1
                    x_lang = k_.split()[-1].split(">")[0] if ">" in k_ else k_.split()[2]
                    if x_lang != lang:
                        probs[f"{k_}: told language"] += 1
                    if k_.startswith("bi story ") or k_.startswith("bi where "):
                        if len(a_) != 1 or to_of.get(a_[0]) != e:
                            probs[f"{k_}: answer cell"] += 1
                    if k_.startswith("bi where "):
                        y_ = k_.split(">")[1]
                        if y_ == "ko" and not (len(ask) == 3 and ask[0] == m_ and ask[1] == GA and ask[2] == EODI):
                            probs[f"{k_}: question"] += 1
                        if y_ == "en" and not (len(ask) == 3 and ask[:2] == [WHERE, IS] and ko_of_en.get(ask[2]) == m_):
                            probs[f"{k_}: question"] += 1
                    if k_.startswith("bi what "):
                        y_ = k_.split(">")[1]
                        cand = names if y_ == "ko" else [w.en_name[n] for n in names]
                        right = m_ if y_ == "ko" else w.en_name[m_]
                        if sorted(a_) != sorted(cand) or a_[0] != right:
                            probs[f"{k_}: candidates"] += 1
                        if y_ == "ko":
                            q_m, q_l, q_v = read_ko(ask)
                            if not (q_m == [MUEOS] and q_l == [l_] and q_v == [v_]):
                                probs[f"{k_}: question"] += 1
                        elif not (ask[0] == WHAT and ask[1] == v_ and ask[2] == FROM and ko_of_en.get(ask[3]) == l_):
                            probs[f"{k_}: question"] += 1
    print(f"items checked: {sum(n_items.values())} (worlds x seeds 0-9); per key (bi, seeds 0-9): "
          f"{dict((k[1], v) for k, v in sorted(n_items.items()) if k[0] == 'bi')}", flush=True)
    for wn in ("bi", "bishuf"):
        print(f"  {wn}: English naming moments showing the name's own object "
              f"{spread[(wn, 'en naming shows its own object')]} / {spread[(wn, 'en naming moments')]}", flush=True)
    print(f"problems: {dict(probs) if probs else 'none'}", flush=True)

elif which in ("reg", "reg2"):
    # reg: the finding 125 Korean body, seed 0 (a 10-07 cache -- older than the 10-08 "describe" keys);
    # reg2: a recent cache -- the "@multi" world "@bi" is built on, seed 60, made 10-15 by the compute helper
    c, sd = (("both@ascii@kor+rediscover+chain+wm+bind+wmlast", 0) if which == "reg" else
             ("both@ascii@multi+rediscover+chain+wm+bind+wmlast", int(sys.argv[2]) if len(sys.argv) > 2 else 60))
    new = RL.life(c, sd)
    path = f"{RL.CACHE}/seed{sd}_{c}.pkl"
    old = pickle.load(open(path, "rb"))
    pickle.dump(new, open(f"reg_{which}_seed{sd}.pkl", "wb"))

    def deq(x, y):                                  # NaN-aware deep equality (a dict of NaNs is equal to itself)
        if isinstance(x, dict) and isinstance(y, dict):
            return set(x) == set(y) and all(deq(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return len(x) == len(y) and all(deq(a, b) for a, b in zip(x, y))
        try:
            if np.isnan(x) and np.isnan(y):
                return True
        except TypeError:
            pass
        return bool(x == y)
    bad = sorted({k for a, b in zip(new, old) for k in set(a) | set(b) if not deq(a.get(k, "MISSING"), b.get(k, "MISSING"))})
    print(f"life regression {c} seed {sd}: sessions {len(new)} / {len(old)}, keys differing {bad if bad else 'none'}; "
          f"pickle bytes equal to the cache: {pickle.dumps(new) == open(path, 'rb').read()}", flush=True)

else:                                               # P / S seed-0 smoke
    import run_bi
    c = run_bi.CONDS[which]
    steps = RL.life(c, 0)
    keys = sorted(k for k in steps[-1] if k.startswith("bi "))
    for k in keys:
        print(f"  {k:24} " + " ".join("  -  " if np.isnan(st.get(k, np.nan)) else f"{st[k]:.3f}" for st in steps),
              flush=True)
    print(f"{which} done in {time.time() - t0:.0f} s", flush=True)
