"""D72 subject 2 + the new control, world check (2026-10-18, Q14): (1) every existing world is unchanged -- old builder
(before this change) vs new, seeds 0-2, all flags, the D75 "@scene" worlds and BOTH subject-1 worlds ("cat", "verb");
(2) "chain" = the "verb" world exactly except the added "d72 chain" tests (the same elements, heard moments and "d72 verb"
tests -- so the cached "@d72verb" lives are this world's lives); (3) the shapes: "neg" words heard at their densities, the
negated moments' to-cell = the from-cell, the test items, the new elements last; chain items' pairs and candidates.
    python3 check_d72b_world.py OLD_data_ascii.py"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, ".")
import data_ascii as NEW

spec = importlib.util.spec_from_file_location("OLD", sys.argv[1])
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)


def sig(w, drop_chain=False):
    cats = []
    for cat in w.categories:
        items = {}
        for k, v in cat.items():
            if k == "d72" and drop_chain:
                v = [x for x in v if not x[0].startswith("d72 chain")]
            items[k] = v
        cats.append(repr(sorted((k, repr(v)) for k, v in items.items())))
    extra = repr(sorted((k, repr(v)) for k, v in vars(w).items() if k not in ("categories", "vecs", "cats", "d72")))
    return (w.vecs.tobytes(), w.cats.tobytes(), tuple(cats), extra)


flags = [{}, dict(korean=True), dict(multi=True), dict(props=True), dict(tale=True), dict(grav=True), dict(frame=True),
         dict(contain=True), dict(count=True), dict(succ=True), dict(numline="row"), dict(numline="own"), dict(views=True),
         dict(goal=True), dict(self_=True), dict(nw=True), dict(sp=True), dict(sp_shuffle=True), dict(when=True),
         dict(when_shuffle=True), dict(walk=True), dict(ten=True), dict(ten_shuffle=True), dict(bi=True),
         dict(bi_shuffle=True), dict(c8=True), dict(c8_shuffle=True), dict(hold=True), dict(scene=True),
         dict(scene_shuffle=True), dict(d72="cat"), dict(d72="verb")]
bad = 0
for seed in (0, 1, 2):
    for fl in flags:
        try:
            a, b = sig(OLD.build(seed, **fl)), sig(NEW.build(seed, **fl))
        except Exception as e:
            print("ERR", seed, fl, e)
            bad += 1
            continue
        if a != b:
            print("DIFF", seed, fl)
            bad += 1
print("unchanged worlds:", "PASS" if bad == 0 else f"FAIL {bad}", f"({3 * len(flags)} builds)")
bad2 = 0
for seed in (0, 1, 2):
    v, c = NEW.build(seed, d72="verb"), NEW.build(seed, d72="chain")
    if sig(v) != sig(c, drop_chain=True):
        print("chain != verb", seed)
        bad2 += 1
print("chain world = verb world + chain tests:", "PASS" if bad2 == 0 else f"FAIL {bad2}")
for seed in (0, 1, 2):
    base = NEW.build(seed)
    w = NEW.build(seed, d72="neg")
    k = [sum(t in s for c in range(1, w.n_cat) for kk, s, _ in w.categories[c]["heard"] if kk == "d72-heard")
         for t in w.d72_words]
    same_cell = all(w.pos_of[s_t[0]] == w.pos_of[[x for x in s if x in w.pos_from.values()][0]]
                    for c in range(1, w.n_cat) for kk, s, s_t in w.categories[c]["heard"] if kk == "d72-heard")
    items = [x for cat in w.categories for x in cat.get("d72", [])]
    right_stays = all(w.pos_of[a[0]] == w.pos_of[[t for t in s if t in w.pos_from.values()][0]] and len(set(a)) == 4
                      for _, s, a in items)
    print(f"neg seed {seed}: V {w.V} (+{w.V - base.V}), dens {w.d72_dens}, heard per word {k}, test items {len(items)} "
          f"(per word {[sum(1 for x in items if x[0] == 'd72 neg ' + str(j)) for j in range(4)]}), "
          f"heard to = from {same_cell}, right = stays & 4 distinct {right_stays}, "
          f"new elements last {all(t >= base.V for t in w.d72_words)}, kinds {[w.kinds[w.kind_of[t]] for t in w.d72_words]}")
    c = NEW.build(seed, d72="chain")
    ch = [x for cat in c.categories for x in cat.get("d72", []) if x[0].startswith("d72 chain")]
    keys = sorted({x[0] for x in ch})
    ok = all(len(set(a)) == 4 and sum(t in c.d72_words for t in s) == 2 for _, s, a in ch)
    print(f"chain seed {seed}: items {len(ch)}, pairs {keys}, two new words & 4 distinct candidates {ok}")
