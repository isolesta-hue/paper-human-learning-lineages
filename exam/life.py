"""One exam life (exam_spec/GENERATOR.md 5): its pseudo-words, its kinds of things and its people, and the exposure
stream of 16 sessions made of short stories (one sentence = one event; the scene record is the state after it).
The internal log (events, language, worlds) stays with the generator: the model-facing records carry only the sentence
and the scene."""
from .common import rng
from .words import Lexicon
from .world import World, W

SESSIONS = 16
STORIES = {"exist": 10, "count": 6, "push": 8, "put": 8, "box": 4, "long": 2, "rare": 4}   # per session (edition 2 sizes TBD)
N_KINDS, N_BOX_KINDS, N_PEOPLE = 10, 2, 4
NEW_COUNTS = (1, 1, 2, 2, 4, 4, 8, 8)            # criterion 2: new nouns heard this many times (half early, half late)
N_RARE = 4                                       # criterion 2: rare kinds, seen seldom and named whenever told about
P_BACKGROUND = 0.5                               # criterion 2: a story scene gets one unnamed new thing in the background
VERBS = ("go", "push", "put", "take")


class Life:
    def __init__(self, seed, idx, stories=None):
        self.seed, self.idx = seed, idx
        self.stories_per = dict(STORIES, **(stories or {}))
        L = Lexicon(seed, idx)
        self.people = [f"c{i}" for i in range(N_PEOPLE)]
        n_new = len(NEW_COUNTS)
        self.lex = {"names": {c: L.name() for c in self.people},
                    "nouns": {k: L.noun() for k in range(N_KINDS + n_new + N_RARE)},
                    "verbs": {v: L.verb() for v in VERBS}, "kind_of": {}}
        self.box_kinds = list(range(N_KINDS - N_BOX_KINDS, N_KINDS))
        self.plain_kinds = list(range(N_KINDS - N_BOX_KINDS))
        self.new_kinds = list(range(N_KINDS, N_KINDS + n_new))
        # criterion 2 (user decision 10-13): the new things are familiar before they are named -- they stand unnamed in the
        # background of other stories (no sentence ever tells about them except their naming) -- and each naming moment
        # shows the new thing beside one rare thing whose name is known (told whenever it is seen), so that "the rare
        # thing goes with the new word" fails and one-shot learning needs "a new word names the thing without a name"
        self.rare_kinds = list(range(N_KINDS + n_new, N_KINDS + n_new + N_RARE))
        self.naming_with = {k: [] for k in self.new_kinds}      # the rare kind beside each naming of a new kind
        r = rng(seed, "plan", idx)
        order = r.permutation(n_new)
        # criterion 2: kind -> (exposures, session); early half in sessions 1-7, late half in 8-14 (so that the test right
        # after the naming and the test at the end are never the same moment)
        self.new_plan = {}
        self.new_lang = {}                 # a new noun is named in one language only (so its test uses that language)
        for j, kk in enumerate(order):
            k = self.new_kinds[int(kk)]
            early = j < n_new // 2
            self.new_plan[k] = (NEW_COUNTS[int(kk)], int(r.integers(1, 8) if early else r.integers(8, 15)))
            self.new_lang[k] = "ko" if j % 2 == 0 else "en"
        # criterion 1 / 5: (person, verb, kind) never heard together; criterion 4: (put, take) box counts never heard
        self.held_triples = set()
        while len(self.held_triples) < 6:
            self.held_triples.add((self.people[int(r.integers(N_PEOPLE))], str(r.choice(["push", "put", "take"])),
                                   int(r.choice(self.plain_kinds))))
        self.held_box = {(3, 1), (4, 2), (2, 2), (5, 3)}
        self.inst = 0
        self.log = []                  # internal: one entry per story (kind, lang, moments [(event, World)])
        self.records = []              # model-facing exposure records
        self._build()

    # ---- helpers
    def new_id(self, kind):
        self.inst += 1
        k = f"t{kind}_{self.inst}"
        self.lex["kind_of"][k] = kind
        return k

    def layout(self, r, kinds, people=1, box_kind=None, stack=0.3):
        """a fresh scene: people and one thing of each kind on the floor, some things stacked (on a thing or on a box)."""
        w = World()
        cells = [int(x) for x in r.permutation(W)]
        cast = [self.people[int(i)] for i in r.choice(N_PEOPLE, people, replace=False)] if people else []
        ids = []
        for c in cast:
            w.place(c, cells.pop(), 0, person=True)
        if box_kind is not None:
            b = self.new_id(box_kind)
            w.place(b, cells.pop(), 0, box=True)
            ids.append(b)
        for k in kinds:
            t = self.new_id(k)
            floor = [x for x in range(W) if w.free(x, 0)]
            tops = [x for x in range(W) if w.at(x, 0) is not None and w.at(x, 0) not in w.people and w.free(x, 1)]
            if tops and (r.random() < stack or not floor):
                x = int(r.choice(tops))
                w.place(t, x, w.pos[w.top_of(w.at(x, 0))][1] + 1)
            elif floor:
                w.place(t, int(r.choice(floor)), 0)
            else:
                continue
            ids.append(t)
        return w, cast, ids

    def emit(self, session, kind, lang, moments):
        sid = len(self.log)
        self.log.append({"session": session, "kind": kind, "lang": lang, "moments": moments})
        from .render import sentence
        for m, (ev, w) in enumerate(moments):
            self.records.append({"session": session, "story": sid, "moment": m,
                                 "sentence": sentence(ev, self.lex, lang), "scene": w.scene(m)})

    def ok_triple(self, c, verb, t):
        return (c, verb, self.lex["kind_of"].get(t)) not in self.held_triples

    # ---- stories
    def setting(self, r, w, ids):
        """the moment that opens an event story: one thing of the scene told (with a relation when it has one) and the
        scene before the first event shown -- so every event is seen with the state before it and after it."""
        a = ids[int(r.integers(len(ids)))]
        rels = [(b, w.relation(a, b)) for b in ids if b != a and w.relation(a, b) in ("on", "under", "beside")]
        if rels and r.random() < 0.6:
            b, rel = rels[int(r.integers(len(rels)))]
            return ({"type": "exist", "a": a, "rel": rel, "b": b}, w.copy())
        return ({"type": "exist", "a": a}, w.copy())

    def background(self, r, w):
        """criterion 2: with probability P_BACKGROUND one new thing (any new kind) stands unnamed in the scene -- on the floor
        or on a plain thing; it is never in the story's ids, so no sentence tells about it (it may be moved along)."""
        if r.random() >= P_BACKGROUND:
            return
        k = int(r.choice(self.new_kinds))
        floor = [x for x in range(W) if w.free(x, 0)]
        tops = [x for x in range(W) if w.at(x, 0) is not None and w.at(x, 0) not in w.people and w.at(x, 0) not in w.box
                and w.free(x, 1)]
        spots = [(x, 0) for x in floor] + [(x, 1) for x in tops]
        if spots:
            x, y = spots[int(r.integers(len(spots)))]
            w.place(self.new_id(k), x, y)

    def s_exist(self, r):
        kinds = [int(k) for k in r.choice(self.plain_kinds, int(r.integers(2, 4)), replace=False)]
        w, _, ids = self.layout(r, kinds, people=0)
        self.background(r, w)
        a = ids[int(r.integers(len(ids)))]
        rels = [(b, w.relation(a, b)) for b in ids if b != a and w.relation(a, b)]
        if rels and r.random() < 0.6:
            b, rel = rels[int(r.integers(len(rels)))]
            return [({"type": "exist", "a": a, "rel": rel, "b": b}, w.copy())]
        return [({"type": "exist", "a": a}, w.copy())]

    def s_count(self, r):
        k = int(r.choice(self.plain_kinds))
        n = int(r.integers(1, 11))
        w = World()
        cols = [0] * W
        for _ in range(n):                      # stack into random columns (at most 3 high)
            x = int(r.choice([i for i in range(W) if cols[i] < 3]))
            w.place(self.new_id(k), x, cols[x])
            cols[x] += 1
        return [({"type": "count", "kind": k, "n": n}, w.copy())]

    def s_push(self, r, long=False):
        kinds = [int(k) for k in r.choice(self.plain_kinds, int(r.integers(1, 4)), replace=False)]
        w, cast, ids = self.layout(r, kinds, people=1)
        self.background(r, w)
        c = cast[0]
        out = [self.setting(r, w, ids)]
        for _ in range(int(r.integers(4, 9)) if long else 1):
            floor_things = [t for t in ids if t in w.pos and w.pos[t][1] == 0 and self.ok_triple(c, "push", t)]
            if not floor_things:
                break
            t = floor_things[int(r.integers(len(floor_things)))]
            if w.go(c, t):
                out.append(({"type": "go", "c": c, "t": t}, w.copy()))
                w.push(c, t)
                out.append(({"type": "push", "c": c, "t": t}, w.copy()))
        return out if len(out) > 1 else []

    def s_put(self, r):
        has_box = r.random() < 0.5
        kinds = [int(k) for k in r.choice(self.plain_kinds, int(r.integers(2, 4)), replace=False)]
        w, cast, ids = self.layout(r, kinds, people=1, box_kind=int(r.choice(self.box_kinds)) if has_box else None)
        self.background(r, w)
        c = cast[0]
        things = [t for t in ids if t not in w.box and self.ok_triple(c, "put", t)]
        if not things:
            return []
        t = things[int(r.integers(len(things)))]
        out = [self.setting(r, w, ids)]
        box = [x for x in ids if x in w.box]
        if box and r.random() < 0.6:                    # into the box
            b = box[0]
            if not w.put_in(c, t, b):
                return []
            out.append(({"type": "put", "c": c, "t": t, "rel": "in", "b": b}, w.copy()))
            if r.random() < 0.3:
                out.append(({"type": "exist", "a": t, "rel": "in", "b": b}, w.copy()))
            return out
        bases = [x for x in ids if x != t and x in w.pos and t not in w.column(x) and x not in w.column(t)
                 and w.top_of(x) in ids]             # the thing named under t is always one the story may tell about
        if box and box[0] in bases and r.random() < 0.5:   # on the box (a box also holds things on its top)
            bases = box
        if not bases:
            return []
        base = bases[int(r.integers(len(bases)))]
        if not w.put_on(c, t, base):
            return []
        under = w.at(w.pos[t][0], w.pos[t][1] - 1)
        out.append(({"type": "put", "c": c, "t": t, "rel": "on", "b": under}, w.copy()))
        if r.random() < 0.3:
            out.append(({"type": "exist", "a": under, "rel": "under", "b": t}, w.copy()))
        return out

    def s_box(self, r):
        """criterion 3 / 4: things of one kind go into a box and some come out; the last sentence tells the count inside."""
        while True:
            a = int(r.integers(1, 6))
            b = int(r.integers(0, a + 1))
            if (a, b) not in self.held_box:
                break
        w, cast, _ = self.layout(r, [], people=1, box_kind=int(r.choice(self.box_kinds)))
        c = cast[0]
        kinds = [k for k in self.plain_kinds if (c, "put", k) not in self.held_triples and (c, "take", k) not in self.held_triples]
        k = int(r.choice(kinds))
        box = [x for x in w.pos if x in w.box][0]
        out = [({"type": "exist", "a": box}, w.copy())]
        for _ in range(a):
            floor = [x for x in range(W) if w.free(x, 0)]
            t = self.new_id(k)
            w.place(t, int(r.choice(floor)), 0) if floor else None
            if t not in w.pos:
                break
            out.append(({"type": "exist", "a": t}, w.copy()))
            w.put_in(c, t, box)
            out.append(({"type": "put", "c": c, "t": t, "rel": "in", "b": box}, w.copy()))
        for _ in range(b):
            if not w.inside[box]:
                break
            t = w.inside[box][-1]
            if not w.take_out(c, t, box):
                break
            out.append(({"type": "take", "c": c, "t": t, "b": box}, w.copy()))
        n_in = len(w.inside[box])
        if n_in:
            out.append(({"type": "count", "kind": k, "n": n_in, "rel": "in", "b": box}, w.copy()))
        return out

    def s_rare(self, r):
        """criterion 2: a rare thing, told about whenever it is seen (with a relation when it has one), among 1-2 plain things."""
        rk = int(r.choice(self.rare_kinds))
        kinds = [rk] + [int(k) for k in r.choice(self.plain_kinds, int(r.integers(1, 3)), replace=False)]
        w, _, ids = self.layout(r, kinds, people=0)
        self.background(r, w)
        a = [t for t in ids if self.lex["kind_of"][t] == rk][0]
        rels = [(b, w.relation(a, b)) for b in ids if b != a and w.relation(a, b)]
        if rels and r.random() < 0.6:
            b, rel = rels[int(r.integers(len(rels)))]
            return [({"type": "exist", "a": a, "rel": rel, "b": b}, w.copy())]
        return [({"type": "exist", "a": a}, w.copy())]

    def s_naming(self, r, k):
        """criterion 2: the new noun k named once; the scene shows the new thing and one rare thing whose name is known --
        two things, so one moment alone leaves the word ambiguous unless the learner gives a new word to the thing without
        a name."""
        rk = int(r.choice(self.rare_kinds))
        self.naming_with[k].append(rk)
        w, _, ids = self.layout(r, [k, rk], people=0)
        a = [t for t in ids if self.lex["kind_of"][t] == k][0]
        return [({"type": "exist", "a": a}, w.copy())]

    def _build(self):
        for s in range(SESSIONS):
            r = rng(self.seed, "session", self.idx, s)
            plan = []
            for kind, n in self.stories_per.items():
                plan += [kind] * n
            plan += [("naming", k) for k, (cnt, ses) in self.new_plan.items() if ses == s for _ in range(cnt)]
            for j in r.permutation(len(plan)):
                item = plan[int(j)]
                lang = "ko" if r.random() < 0.5 else "en"
                if isinstance(item, tuple):
                    lang = self.new_lang[item[1]]
                    moments = self.s_naming(r, item[1])
                else:
                    moments = {"exist": self.s_exist, "count": self.s_count, "push": self.s_push, "put": self.s_put,
                               "box": self.s_box, "long": lambda r_: self.s_push(r_, long=True), "rare": self.s_rare}[item](r)
                if moments:
                    self.emit(s, item if isinstance(item, str) else "naming", lang, moments)
        self.check_plan()

    def check_plan(self):
        """every new noun is heard exactly as planned (1 / 2 / 4 / 8 times, in its one language), never otherwise."""
        for k in self.new_kinds:
            n = sum(1 for r in self.records for lang in ("ko", "en") if self.lex["nouns"][k][lang] in r["sentence"])
            assert n == self.new_plan[k][0], (k, n, self.new_plan[k])
