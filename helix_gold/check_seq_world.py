"""D80 v1 world check (2026-10-21): (1) every existing world is unchanged -- the builder before this change vs now, seeds
0-2, all flags incl. "@nwf" and every D72 world; (2) "@seq" = the "@scene" world plus, per walking room, the start scene
("seq start"), the question moments ("seq-heard") and cat["seq"] -- nothing else differs, the old elements and features
too; the two new words last, their kind; (3) the truth of every question moment and test FROM WHAT WAS SEEN: walking the
room's heard moments in order, a picture's time line = its cells in the seen moments [picture, at]; "now" = the line's last
value, "before" = its last value different from the last; every question moment and every test must equal these.
    python3 check_seq_world.py OLD_data_ascii.py"""
import importlib.util
import sys
from collections import Counter

sys.path.insert(0, ".")
import data_ascii as NEW

spec = importlib.util.spec_from_file_location("OLD", sys.argv[1])
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)


def sig(w):
    cats = [repr(sorted((k, repr(v)) for k, v in cat.items())) for cat in w.categories]
    extra = repr(sorted((k, repr(v)) for k, v in vars(w).items() if k not in ("categories", "vecs", "cats", "d72")))
    return (w.vecs.tobytes(), w.cats.tobytes(), tuple(cats), extra)


flags = [{}, dict(korean=True), dict(multi=True), dict(props=True), dict(tale=True), dict(grav=True), dict(frame=True),
         dict(contain=True), dict(count=True), dict(succ=True), dict(numline="row"), dict(numline="own"), dict(views=True),
         dict(goal=True), dict(self_=True), dict(nw=True), dict(sp=True), dict(sp_shuffle=True), dict(when=True),
         dict(when_shuffle=True), dict(walk=True), dict(ten=True), dict(ten_shuffle=True), dict(bi=True),
         dict(bi_shuffle=True), dict(c8=True), dict(c8_shuffle=True), dict(hold=True), dict(scene=True),
         dict(scene_shuffle=True), dict(d72="cat"), dict(d72="verb"), dict(d72="chain"), dict(d72="neg"), dict(d72="ice"),
         dict(nwf=True)]
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
    base, w = NEW.build(seed, scene=True), NEW.build(seed, seq=True)
    qw, qb = w.seq_words
    checks = {"old elements same": bool((w.vecs[:base.V] == base.vecs).all() and (w.cats[:base.V] == base.cats).all()
                                        and w.names[:base.V] == base.names),
              "new words last": sorted([qw, qb]) == list(range(base.V, w.V)),
              "kinds": sorted({w.kinds[w.kind_of[t]] for t in (qw, qb)}) == ["name"]}
    same, truth, n_q, n_t, kinds_t = True, True, 0, 0, Counter()
    at_set = set(w.pos_at.values())
    for c in range(w.n_cat):
        a, b = base.categories[c], w.categories[c]
        same &= repr(a["heard"]) == repr([x for x in b["heard"] if x[0] not in ("seq start", "seq-heard")])
        same &= repr({k: v for k, v in a.items() if k != "heard"}) == repr({k: v for k, v in b.items() if k not in ("heard", "seq")})
        if "seq" not in b:
            continue
        line = {}
        for kind, s_, _ in b["heard"]:
            if len(s_) == 2 and s_[1] in at_set and w.origin[s_[0]] == "drawn":
                line.setdefault(s_[0], []).append(s_[1])
            if kind == "seq-heard":
                n_q += 1
                pic = [t for t in s_ if w.origin[t] == "drawn"][0]
                ans = [t for t in s_ if t in at_set][0]
                vals = line.get(pic, [])
                if qb in s_:
                    prev = [v for v in vals if v != vals[-1]]
                    truth &= bool(prev) and prev[-1] == ans
                else:
                    truth &= bool(vals) and vals[-1] == ans
        for kind, s_, ans in b["seq"]:
            n_t += 1
            kinds_t[kind] += 1
            vals = line.get(s_[0], [])
            if kind == "seq before":
                prev = [v for v in vals if v != vals[-1]]
                truth &= bool(prev) and prev[-1] == ans[0]
            else:
                truth &= bool(vals) and vals[-1] == ans[0]
            if kind == "seq still":
                truth &= len(set(vals)) == 1
    checks["rooms = @scene rooms + start scene + question moments + seq"] = same
    checks["question moments and tests true from what was seen"] = truth
    ok_all &= all(checks.values())
    print(f"seq seed {seed}: V {w.V} (+{w.V - base.V}), question moments {n_q}, tests {dict(kinds_t)}; "
          + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in checks.items()))
print("seq shapes:", "PASS" if ok_all else "FAIL")
