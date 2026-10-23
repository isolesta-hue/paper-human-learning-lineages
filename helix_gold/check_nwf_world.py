"""Q14 frame test world check (2026-10-21): (1) every existing world is unchanged -- the builder before this change vs now,
seeds 0-2, all flags, the D75 scene worlds and every D72 world; (2) the "@nwf" shapes: its rooms are the "@nw" world's rooms
plus the frame moments and the "nwf" list (nothing else differs, the old elements and their features too); the new names are
heard exactly 1 / 2 / 4 / 8 times and never with a frame word; every frame moment = THERE, IS, one common word with its own
picture, one place, 1-3 of the room's pictures; every framed item = its bare twin + THERE, IS; the new elements last, their
kinds; common-word counts (Zipf-like).
    python3 check_nwf_world.py OLD_data_ascii.py"""
import importlib.util
import sys
from collections import Counter

sys.path.insert(0, ".")
import data_ascii as NEW


def sig(w):                                         # (as check_d72b_world.sig; copied -- importing that file runs its checks)
    cats = [repr(sorted((k, repr(v)) for k, v in cat.items())) for cat in w.categories]
    extra = repr(sorted((k, repr(v)) for k, v in vars(w).items() if k not in ("categories", "vecs", "cats", "d72")))
    return (w.vecs.tobytes(), w.cats.tobytes(), tuple(cats), extra)

spec = importlib.util.spec_from_file_location("OLD", sys.argv[1])
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)

flags = [{}, dict(korean=True), dict(multi=True), dict(props=True), dict(tale=True), dict(grav=True), dict(frame=True),
         dict(contain=True), dict(count=True), dict(succ=True), dict(numline="row"), dict(numline="own"), dict(views=True),
         dict(goal=True), dict(self_=True), dict(nw=True), dict(sp=True), dict(sp_shuffle=True), dict(when=True),
         dict(when_shuffle=True), dict(walk=True), dict(ten=True), dict(ten_shuffle=True), dict(bi=True),
         dict(bi_shuffle=True), dict(c8=True), dict(c8_shuffle=True), dict(hold=True), dict(scene=True),
         dict(scene_shuffle=True), dict(d72="cat"), dict(d72="verb"), dict(d72="chain"), dict(d72="neg"), dict(d72="ice")]
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
ok_all = True
for seed in (0, 1, 2):
    base, w = NEW.build(seed, nw=True), NEW.build(seed, nwf=True)
    fw, cw, cp = w.nwf_frame, w.nwf_common, w.nwf_common_pic
    new_el = fw + cw + cp
    checks = {}
    checks["old elements same"] = bool((w.vecs[:base.V] == base.vecs).all() and (w.cats[:base.V] == base.cats).all()
                                       and w.names[:base.V] == base.names)
    checks["new elements last"] = sorted(new_el) == list(range(base.V, w.V))
    same = True
    for c in range(w.n_cat):
        a, b = base.categories[c], w.categories[c]
        if c == 0:
            same &= repr(a) == repr(b)
            continue
        n_old = len(a["heard"])
        same &= repr(a["heard"]) == repr(b["heard"][:n_old])
        same &= all(k == "frame" for k, _, _ in b["heard"][n_old:]) and len(b["heard"]) - n_old == NEW.N_FRAME
        same &= repr({k: v for k, v in a.items() if k != "heard"}) == repr({k: v for k, v in b.items() if k not in ("heard", "nwf")})
    checks["rooms = @nw rooms + frame moments + nwf"] = same
    heard_n, with_frame = Counter(), 0
    shape_ok, n_frame, cw_count = True, 0, Counter()
    for c in range(1, w.n_cat):
        objs = w.categories[c]["objects"]
        names = [int(o[0]) for o in objs]
        room_pics = {int(x) for o in objs for x in o[1:]}
        for k, s, _ in w.categories[c]["heard"]:
            for t in s:
                if t in names:
                    heard_n[t] += 1
            if any(t in names for t in s) and any(t in fw for t in s):
                with_frame += 1
            if k == "frame":
                n_frame += 1
                ws = [t for t in s if t in cw]
                ps = [t for t in s if t in cp]
                at = [t for t in s if t in w.pos_at.values()]
                rest = [t for t in s if t not in fw and t not in cw and t not in cp and t not in at]
                shape_ok &= (sorted(t for t in s if t in fw) == sorted(fw) and len(ws) == 1 and len(ps) == 1 and
                             cw.index(ws[0]) == cp.index(ps[0]) and len(at) == 1 and 1 <= len(rest) <= 3 and
                             all(t in room_pics for t in rest))
                cw_count[cw.index(ws[0])] += 1
    expo = Counter()
    for c in range(1, w.n_cat):
        for k, s, a in w.categories[c]["nw"]:
            pass
        for j, o in enumerate(w.categories[c]["objects"]):
            expo[heard_n[int(o[0])]] += 1
    checks["names heard 1/2/4/8 only"] = set(expo) == {1, 2, 4, 8}
    checks["no name with a frame word"] = with_frame == 0
    checks["frame moments shape"] = shape_ok
    pairs_ok, n_pairs, n_common = True, Counter(), 0
    for c in range(1, w.n_cat):
        it = w.categories[c]["nwf"]
        names = {int(o[0]) for o in w.categories[c]["objects"]}
        bare_src = [(k[2:], s, a) for k, s, a in w.categories[c]["nw"] if int(a[0]) in names]
        pairs = [(it[i], it[i + 1]) for i in range(0, len(it), 2)]
        nw_pairs = [p for p in pairs if p[0][0] != "nwbc"]
        pairs_ok &= len(nw_pairs) == len(bare_src)
        for (kb, sb, ab), (kf, sf, af) in pairs:
            pairs_ok &= kb[:3] == "nwb" and kf[:3] == "nwf" and kb[3:] == kf[3:] and ab == af
            pairs_ok &= sorted(sf) == sorted(list(sb) + fw)
            if kb == "nwbc":
                n_common += 1
                pairs_ok &= ab[0] in cw and cp[cw.index(ab[0])] in sb
            else:
                n_pairs[kb] += 1
        for (b_, f_), src in zip(nw_pairs, bare_src):
            pairs_ok &= b_[0][3:] == src[0] and b_[1] == list(src[1]) and b_[2] == list(src[2])
    checks["pairs = bare + THERE, IS"] = pairs_ok
    checks["kinds"] = (sorted({w.kinds[w.kind_of[t]] for t in fw + cw}) == ["name"] and
                       sorted({w.kinds[w.kind_of[t]] for t in cp}) == ["drawn"])
    ok_all &= all(checks.values())
    print(f"nwf seed {seed}: V {w.V} (+{w.V - base.V}), frame moments {n_frame}, common-word counts "
          f"{[cw_count[k] for k in range(len(cw))]}, pairs per n {dict(sorted(n_pairs.items()))}, common pairs {n_common}; "
          + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in checks.items()))
print("nwf shapes:", "PASS" if ok_all else "FAIL")
