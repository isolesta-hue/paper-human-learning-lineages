"""D72 subject 3 world check (2026-10-18, Q15): (1) every existing world is unchanged -- the builder before this change vs
now, seeds 0-2, all flags, the D75 scene worlds and every D72 world so far ("cat", "verb", "chain", "neg"); (2) the "ice"
shapes: floor moments move the thing twice the word's move in that room (the room's own moments of the word give its move),
tests' right = from + 2d and the lure = from + d, four distinct candidates, the new elements last, their kind.
    python3 check_d72c_world.py OLD_data_ascii.py"""
import importlib.util
import sys

sys.path.insert(0, ".")
import data_ascii as NEW
from check_d72b_world import sig

spec = importlib.util.spec_from_file_location("OLD", sys.argv[1])
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)

flags = [{}, dict(korean=True), dict(multi=True), dict(props=True), dict(tale=True), dict(grav=True), dict(frame=True),
         dict(contain=True), dict(count=True), dict(succ=True), dict(numline="row"), dict(numline="own"), dict(views=True),
         dict(goal=True), dict(self_=True), dict(nw=True), dict(sp=True), dict(sp_shuffle=True), dict(when=True),
         dict(when_shuffle=True), dict(walk=True), dict(ten=True), dict(ten_shuffle=True), dict(bi=True),
         dict(bi_shuffle=True), dict(c8=True), dict(c8_shuffle=True), dict(hold=True), dict(scene=True),
         dict(scene_shuffle=True), dict(d72="cat"), dict(d72="verb"), dict(d72="chain"), dict(d72="neg")]
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
for seed in (0, 1, 2):
    base = NEW.build(seed)
    w = NEW.build(seed, d72="ice")
    pos = w.pos_of
    froms, tos = set(w.pos_from.values()), set(w.pos_to.values())
    verbs = set(w.verbs)
    k = [sum(t in s for c in range(1, w.n_cat) for kk, s, _ in w.categories[c]["heard"] if kk == "d72-heard")
         for t in w.d72_words]
    ok_heard, n_heard = 0, 0
    for c in range(1, w.n_cat):
        usual = {}
        for kk, s, _ in w.categories[c]["heard"]:
            if kk[:1] == "R" and kk[1:].isdigit():
                f_ = [x for x in s if x in froms][0]
                t_ = [x for x in s if x in tos][0]
                v_ = [x for x in s if x in verbs][0]
                usual[v_] = (pos[t_][0] - pos[f_][0], pos[t_][1] - pos[f_][1])
        for kk, s, _ in w.categories[c]["heard"]:
            if kk != "d72-heard":
                continue
            n_heard += 1
            f_ = [x for x in s if x in froms][0]
            t_ = [x for x in s if x in tos][0]
            v_ = [x for x in s if x in verbs][0]
            d = usual[v_]
            ok_heard += (pos[t_][0] - pos[f_][0], pos[t_][1] - pos[f_][1]) == (2 * d[0], 2 * d[1])
    items = [x for cat in w.categories for x in cat.get("d72", [])]
    ok_items = 0
    for _, s, a in items:
        f_ = [x for x in s if x in froms][0]
        c_ = [cc for cc in range(1, w.n_cat) if any(x in s for x in [o[1] for o in w.categories[cc]["objects"]])][0]
        v_ = [x for x in s if x in verbs][0]
        usual = None
        for kk, s2, _ in w.categories[c_]["heard"]:
            if kk[:1] == "R" and kk[1:].isdigit() and v_ in s2:
                f2 = [x for x in s2 if x in froms][0]
                t2 = [x for x in s2 if x in tos][0]
                usual = (pos[t2][0] - pos[f2][0], pos[t2][1] - pos[f2][1])
                break
        p = pos[f_]
        ok_items += (usual is not None and pos[a[0]] == (p[0] + 2 * usual[0], p[1] + 2 * usual[1]) and
                     pos[a[1]] == (p[0] + usual[0], p[1] + usual[1]) and len(set(a)) == 4)
    print(f"ice seed {seed}: V {w.V} (+{w.V - base.V}), dens {w.d72_dens}, heard per word {k}, floor moments twice the move "
          f"{ok_heard}/{n_heard}, test items {len(items)} (per word {[sum(1 for x in items if x[0] == 'd72 ice ' + str(j)) for j in range(4)]}), "
          f"right = from + 2d & lure = from + d & 4 distinct {ok_items}/{len(items)}, new elements last "
          f"{all(t >= base.V for t in w.d72_words)}, kinds {sorted({w.kinds[w.kind_of[t]] for t in w.d72_words})}")
