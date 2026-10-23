"""D80 v2 world and device check (2026-10-22; D80_DESIGN.md v2), read BEFORE any life: (1) every existing world is unchanged
-- the builder before this change vs now, seeds 0-2, all flags; (2) "@seq2" = the "@seq" world with exactly the scene lines
after the walk's last SEQ_TOLD moves removed (nothing else differs), the told items are exactly the things whose last move
is told only, and every item's answer is true; (3) the scene part's observation rule now needs a moment without a word --
no observation of the "@scene" / "@seq" worlds is changed by it except the question-and-answer moments; (4) the device
logic without any body: the scene part reads each room's moments in order (as in a life), and the told-aware store's
time line (no model) gives "now" = last / "before" = before for the told items, against the seen-only store (v1.1).
    python3 check_seq2_world.py OLD_data_ascii.py"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, ".")
import data_ascii as NEW
import scene_part
from run_long import EpisodeStore

spec = importlib.util.spec_from_file_location("OLD", sys.argv[1])
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)


def sig(w):                                          # as check_d72b_world.sig (importing that file would run its checks)
    cats = [repr(sorted((k, repr(v)) for k, v in cat.items())) for cat in w.categories]
    extra = repr(sorted((k, repr(v)) for k, v in vars(w).items() if k not in ("categories", "vecs", "cats", "d72")))
    return (w.vecs.tobytes(), w.cats.tobytes(), tuple(cats), extra)


flags = [{}, dict(korean=True), dict(multi=True), dict(props=True), dict(tale=True), dict(grav=True), dict(frame=True),
         dict(contain=True), dict(count=True), dict(succ=True), dict(numline="row"), dict(numline="own"), dict(views=True),
         dict(goal=True), dict(self_=True), dict(nw=True), dict(sp=True), dict(sp_shuffle=True), dict(when=True),
         dict(when_shuffle=True), dict(walk=True), dict(ten=True), dict(ten_shuffle=True), dict(bi=True),
         dict(bi_shuffle=True), dict(c8=True), dict(c8_shuffle=True), dict(hold=True), dict(scene=True),
         dict(scene_shuffle=True), dict(nwf=True), dict(seq=True), dict(d72="cat"), dict(d72="verb"), dict(d72="chain"),
         dict(d72="neg"), dict(d72="ice"), dict(d72="next")]
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
print("unchanged worlds:", "PASS" if bad == 0 else f"FAIL {bad}", f"({3 * len(flags)} builds)", flush=True)

T = NEW.SEQ_TOLD
for seed in range(10):
    w1, w2 = NEW.build(seed, seq=True), NEW.build(seed, seq_told=T)
    ok_rest = sig_equal = True
    removed = told_items = good_items = n_items = n_rooms = 0
    for c in range(1, w1.n_cat):
        h1, h2 = w1.categories[c]["heard"], w2.categories[c]["heard"]
        if "seq" not in w1.categories[c]:
            ok_rest &= h1 == h2
            continue
        n_rooms += 1
        # the told part: the last T sentences of the walk; in "@seq" each is followed by its scene lines
        sent = [i for i, (k, _, _) in enumerate(h1) if k == "scene sentence"]
        drop = set()
        for i in sent[-T:]:
            j = i + 1
            while j < len(h1) and h1[j][0] == "scene line":
                drop.add(j)
                j += 1
        removed += len(drop)
        ok_rest &= [x for i, x in enumerate(h1) if i not in drop] == h2
        it1, it2 = w1.categories[c]["seq"], w2.categories[c]["seq"]
        ok_rest &= [(k.replace(" told", ""), s, a) for k, s, a in it2] == it1
        told_items += sum(k.endswith(" told") for k, _, _ in it2)
        n_items += len(it2)
    # every other attribute equal
    sig_equal = sig(w1)[0] == sig(w2)[0] and sig(w1)[3] == sig(w2)[3]
    print(f"seq2 seed {seed}: walking rooms {n_rooms}, scene lines removed {removed} (= the told moves' scenes), the rest of "
          f"every room equal to \"@seq\" {ok_rest}, elements and other attributes equal {sig_equal}, items {n_items} "
          f"(told {told_items})", flush=True)

# (3) the scene part's observation rule: moments with one picture, one cell AND a word (now not observations)
for name, kw in (("scene", dict(scene=True)), ("seq", dict(seq=True)), ("seq2", dict(seq_told=T))):
    cnt = {}
    for seed in range(10):
        w = NEW.build(seed, **kw)
        b = scene_part.SceneBooks(w)
        for c in range(1, w.n_cat):
            for k, s, _ in w.categories[c]["heard"]:
                ats = [x for x in s if x in b.pos_of]
                pics = [x for x in s if b.is_pic[x]]
                words = [x for x in s if b.is_word[x]]
                if len(ats) == 1 and len(pics) == 1 and words:
                    cnt[k] = cnt.get(k, 0) + 1
    print(f"world @{name} (10 seeds): one-picture one-cell moments WITH a word (no longer observations), by kind: "
          f"{cnt or 'none'}", flush=True)

# (4) the device logic, no body: per room in arrival order the scene part observes the room, both stores add it
tot = {}
for seed in range(10):
    w = NEW.build(seed, seq_told=T)
    books = scene_part.SceneBooks(w)
    e1, e2 = EpisodeStore(w.channel_of), EpisodeStore(w.channel_of, books=books)
    for c in range(1, w.n_cat):
        books.observe([s for _, s, _ in w.categories[c]["heard"]])
        e1.add(w.categories[c]["heard"])
        e2.add(w.categories[c]["heard"])
        for k, s, a in w.categories[c].get("seq", []):
            if k == "seq still":
                continue
            pic = [t for t in s if books.is_pic[t]][0]
            r = "last" if k.startswith("seq now") else "before"
            for nm, e in (("seen-only", e1), ("told-aware", e2)):
                v = EpisodeStore.rel(r, e.line(pic, 0))
                key = (k, nm)
                tot.setdefault(key, [0, 0])
                tot[key][0] += v == a[0]
                tot[key][1] += 1
for (k, nm), (h, n) in sorted(tot.items()):
    print(f"device logic (no body, 10 seeds): {k:16s} {nm:10s} right {h}/{n} = {h / n:.3f}", flush=True)
print("exit 0")
