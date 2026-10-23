"""D72 subject 4 world check (2026-10-22; D72_DESIGN.md subject 4, rule fixed 10-18): (1) every existing world is unchanged
-- the builder before this change vs now, seeds 0-2, all flags, the scene / frame / order worlds and every D72 world so far;
(2) the "next" shapes: every bell moment is followed at once by a verbless move of the same thing from the bell's cell one
up (turned with the room); every bell-less "seen" moment is followed at once by a verbless move to another neighbour, and
the verbless moves go to each of the four neighbours equally often; tests: right = one up, then stay, four distinct, a cell
never heard with that bell in the room; the new elements last, their kind; (3) the order device (D80 v1.1, EQ1) on every
D72 world: how many test items it could answer at all (it speaks only for a question with exactly one picture whose word set
has a learned meaning) -- read BEFORE any life, so that a life is not spent where the device is silent by construction.
    python3 check_d72d_world.py OLD_data_ascii.py"""
import importlib.util
import sys

sys.path.insert(0, ".")
import data_ascii as NEW


def sig(w):                                          # as check_d72b_world.sig (importing that file would run its checks)
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
         dict(scene_shuffle=True), dict(nwf=True), dict(seq=True), dict(d72="cat"), dict(d72="verb"), dict(d72="chain"),
         dict(d72="neg"), dict(d72="ice")]
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


def turned(d, flipped):
    r = 2 if flipped is True else 0 if flipped is False else int(flipped)
    for _ in range(r):
        d = (-d[1], d[0])
    return d


for seed in (0, 1, 2):
    base = NEW.build(seed)
    w = NEW.build(seed, d72="next")
    pos, bells = w.pos_of, set(w.d72_words)
    ats, froms, tos = set(w.pos_at.values()), set(w.pos_from.values()), set(w.pos_to.values())
    pics = {int(o[1]) for c in range(1, w.n_cat) for o in w.categories[c]["objects"]}
    n_bell = ok_bell = n_plain = ok_plain = 0
    dirs = {}
    k = [0] * 4
    for c in range(1, w.n_cat):
        cat = w.categories[c]
        up = turned((0, 1), cat["flipped"])
        hs = [(kk, s) for kk, s, _ in cat["heard"] if kk == "d72-heard"]
        for i, (_, s) in enumerate(hs):
            if any(x in froms for x in s):
                f_ = [x for x in s if x in froms][0]
                t_ = [x for x in s if x in tos][0]
                d = (pos[t_][0] - pos[f_][0], pos[t_][1] - pos[f_][1])
                dirs[d == up] = dirs.get(d == up, 0) + 1
                dirs[d] = dirs.get(d, 0) + 1
                continue
            nxt = hs[i + 1][1]
            a_ = [x for x in s if x in ats][0]
            f_ = [x for x in nxt if x in froms][0]
            t_ = [x for x in nxt if x in tos][0]
            same = set(s) & set(nxt) & pics
            d = (pos[t_][0] - pos[f_][0], pos[t_][1] - pos[f_][1])
            good = pos[f_] == pos[a_] and len(same) >= 1 and not any(x in w.verb_of for x in nxt)
            if any(x in bells for x in s):
                n_bell += 1
                ok_bell += good and d == up
                k[w.d72_words.index([x for x in s if x in bells][0])] += 1
            else:
                n_plain += 1
                ok_plain += good and d != up and abs(d[0]) + abs(d[1]) == 1
    items = [(c, x) for c in range(1, w.n_cat) for x in w.categories[c].get("d72", [])]
    ok_items = 0
    for c, (kk, s, a) in items:
        cat = w.categories[c]
        up = turned((0, 1), cat["flipped"])
        p = pos[[x for x in s if x in ats][0]]
        j = int(kk.split()[-1])
        heard_cells = {pos[[x for x in s2 if x in ats][0]] for kk2, s2, _ in cat["heard"]
                       if kk2 == "d72-heard" and w.d72_words[j] in s2}
        ok_items += (pos[a[0]] == (p[0] + up[0], p[1] + up[1]) and pos[a[1]] == p and len(set(a)) == 4 and
                     all(x in tos for x in a) and p not in heard_cells and w.d72_words[j] in s)
    print(f"next seed {seed}: V {w.V} (+{w.V - base.V}), dens {w.d72_dens}, bell pairs per word {k}, bell -> one up next "
          f"{ok_bell}/{n_bell}, plain -> another neighbour next {ok_plain}/{n_plain}, verbless moves up / not "
          f"{dirs.get(True, 0)} / {dirs.get(False, 0)} (by direction {sorted((d, n) for d, n in dirs.items() if not isinstance(d, bool))}), "
          f"test items {len(items)} (per word {[sum(1 for _, x in items if x[0] == 'd72 next ' + str(j)) for j in range(4)]}), "
          f"right = up & stay & 4 distinct & cell new {ok_items}/{len(items)}, new elements last "
          f"{all(t >= base.V for t in w.d72_words)}, kinds {sorted({w.kinds[w.kind_of[t]] for t in w.d72_words})}, "
          f"moments {sum(len(cat['heard']) for cat in w.categories[1:])} (base {sum(len(cat['heard']) for cat in base.categories[1:])})",
          flush=True)

from run_long import EpisodeStore
for sub in ("cat", "chain", "neg", "ice", "next"):
    speak, one_pic, n_all, entries = 0, 0, 0, set()
    for seed in range(10):
        w = NEW.build(seed, d72=sub)
        ch = [int(x) for x in w.channel_of]
        epi = EpisodeStore(w.channel_of)
        rooms = list(range(1, w.n_cat))
        for c in rooms:                              # the store after every room (the session's moments in the order heard)
            epi.add(w.categories[c]["heard"])
            epi.learn()
            entries |= {"+".join(sorted(w.names[t] for t in ws)) for ws in epi.table}
            for _, s, _ in list(w.d72_tests) + [x for cc in rooms[:rooms.index(c) + 1] for x in w.categories[cc].get("d72", [])]:
                n_all += 1
                q = [int(t) for t in s]
                if sum(ch[t] == 2 for t in q) == 1:
                    one_pic += 1
                    speak += frozenset(t for t in q if ch[t] == 1) in epi.table
    print(f"device on {sub} (10 seeds, after every room): item looks {n_all}, with exactly one picture {one_pic}, "
          f"with a learned meaning too (the device could speak) {speak}; word sets ever learned: {sorted(entries) or 'none'}",
          flush=True)
