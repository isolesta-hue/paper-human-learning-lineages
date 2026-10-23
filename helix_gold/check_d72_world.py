"""D72 world check: (1) every existing world is unchanged (old builder vs new, seeds 0-2, all flags), (2) the D72 worlds'
shape: the subject words heard at their densities, the test items, the new elements last."""
import sys, importlib.util, numpy as np
sys.path.insert(0, ".")
import data_ascii as NEW
spec = importlib.util.spec_from_file_location("OLD", sys.argv[1])
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)


def sig(w):
    cats = []
    for cat in w.categories:
        cats.append(repr(sorted((k, repr(v)) for k, v in cat.items())))
    extra = repr(sorted((k, repr(v)) for k, v in vars(w).items() if k not in ("categories", "vecs", "cats")))
    return (w.vecs.tobytes(), w.cats.tobytes(), tuple(cats), extra)


flags = [{}, dict(korean=True), dict(multi=True), dict(props=True), dict(tale=True), dict(grav=True), dict(frame=True),
         dict(contain=True), dict(count=True), dict(succ=True), dict(numline="row"), dict(numline="own"), dict(views=True),
         dict(goal=True), dict(self_=True), dict(nw=True), dict(sp=True), dict(sp_shuffle=True), dict(when=True),
         dict(when_shuffle=True), dict(walk=True), dict(ten=True), dict(ten_shuffle=True), dict(bi=True),
         dict(bi_shuffle=True), dict(c8=True), dict(c8_shuffle=True), dict(hold=True)]
bad = 0
for seed in (0, 1, 2):
    for fl in flags:
        try:
            a, b = sig(OLD.build(seed, **fl)), sig(NEW.build(seed, **fl))
        except Exception as e:
            print("ERR", seed, fl, e); bad += 1; continue
        if a != b:
            print("DIFF", seed, fl); bad += 1
print("unchanged worlds:", "PASS" if bad == 0 else f"FAIL {bad}", f"({3 * len(flags)} builds)")
base = NEW.build(0)
for d in ("cat", "verb"):
    for seed in (0, 1, 2):
        w = NEW.build(seed, d72=d)
        assert w.V > base.V or seed
        k = [sum(t in s for c in range(1, w.n_cat) for kk, s, _ in w.categories[c]["heard"] if kk == "d72-heard")
             for t in w.d72_words]
        n_rooms = len(w.d72_tests) + sum(len(c.get("d72", [])) for c in w.categories)
        new_last = all(t >= NEW.build(seed).V for t in w.d72_words)
        print(f"{d} seed {seed}: V {w.V} (+{w.V - NEW.build(seed).V}), dens {w.d72_dens}, heard per word {k}, "
              f"test items {n_rooms}, new elements last {new_last}")
