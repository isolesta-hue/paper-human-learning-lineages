import sys, pickle, importlib.util, numpy as np
sys.path.insert(0, ".")
import data_ascii as NEW
spec = importlib.util.spec_from_file_location("old_da", "../old_da/data_ascii.py"); OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
def sig(w):
    cats = [{k: v for k, v in c.items() if k not in ("tale", "tale_drop", "tale_pre", "pre_drop")} for c in w.categories]
    return pickle.dumps((w.vecs.tobytes(), w.cats.tolist(), w.names, w.origin, repr(cats), w.kind_of.tolist()))
worlds = [{}, dict(korean=True), dict(multi=True), dict(grav=True), dict(frame=True), dict(contain=True), dict(count=True),
          dict(succ=True), dict(numline="row"), dict(numline="own"), dict(views=True), dict(views=2), dict(goal=True),
          dict(self_=True), dict(props=True), dict(hold=True)]
bad = 0
for seed in (0, 3, 5):
    for kw in worlds:
        same = sig(OLD.build(seed, **kw)) == sig(NEW.build(seed, **kw))
        if not same:
            bad += 1; print("DIFF", seed, kw)
print("worlds checked", 3 * len(worlds), "differing", bad)
w = NEW.build(0, tale=True)
rooms = [c for c in w.categories[1:] if c.get("tale")]
print("tale rooms", len(rooms), "drop rooms", sum(c["tale_drop"] for c in rooms), "V", w.V, "tale things", len(w.tale_things))
from collections import Counter
print(Counter(k for c in w.categories[1:] for k, _, _ in c["heard"] if k.startswith("tale")), Counter(k for c in rooms for k, _, _ in c["tale"]))
c = rooms[0]; nm = lambda s: " ".join(w.names[t] for t in s)
for k, s, a in [x for x in c["heard"] if x[0].startswith("tale")] + c["tale"][:2] + c["tale"][-1:]:
    print(k, "|", nm(s), "|", nm(a))
print("flipped of tale rooms", set(c["flipped"] for c in rooms))
