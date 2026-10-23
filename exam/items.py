"""Test items and answer keys of one life (exam_spec/GENERATOR.md 6; EXAM_DESIGN.md edition 2 section 3).

Model-facing item (lives/life_XXX/items.jsonl): {"id", "when", "format", "context", "question", "candidates", "options"?}
  when      "every" = answered after every session (the learning curve; the answer after the last session is the judged
            one), "after_session_k" = once after session k, "end" = once after the last session.
  format    "blank"  one word from the candidates;
            "slot"   one candidate: a person or thing id of the context scene, a cell [x, y], or a sentence number;
            "choice" one option label;
            "pair"   two options, a possible and an impossible outcome: the model gives each a probability.
  context   {"scene"?: the first scene (one line per visible person or thing, as in the exposure), "moments"?: [{"sentence",
            "scene"}, ...] told and shown as in the exposure (the scene after each sentence; things in a box are hidden),
            "sentences"?: [one sentence per event, in order, told only -- the model follows them], "scene_after"?: the
            scene after the told sentence (spatial-word items only)}.
  question  {"slot", "sentence"?, "target"?}: the asked part ("who", "what", "where", "when", "which", "where now",
            "where before", "blank", "choice", "pair"); the question sentence (the event's sentence with the asked part
            replaced in place by the closed-v1 wh-word, or a frame with "___"); the id whose cell is asked.
Answer key (keys/life_XXX.jsonl, opened only for grading): {"id", "criterion", "kind", "detail", "lang", "order", "when",
"answer"}. The criterion, the kind, the language and the order flag are in the key only: no label in the model-facing file
says which ability or which language an item tests, and items are shuffled within a life (the id is a running number).
order = a bag-of-words reading of the context and the question leaves more than one answer (RULES 6): computed by reading
each event's tokens in every role assignment that has the same tokens (Korean particles must still agree with their
noun) and is a possible event in that scene.
Every distractor comes from the same category; every impossible outcome involves a change; no context states its answer
(machine-checked again in checks.py, G3). Every part draws from its own random stream (seed, "items", life, part)."""
from collections import Counter

from .common import rng
from .render import sentence, NUM, SPATIAL
from .words import CLOSED, particle
from .world import World, W, H

WH = {e["slot"]: e for e in CLOSED["wh"]}
KO_PARTICLES = sorted({v for p in CLOSED["ko_particles"] for k, v in p.items() if k != "role"})
CELLS = [[x, y] for y in range(H) for x in range(W)]
LANGS = ("ko", "en")


def scene(w):
    """a scene record as in the exposure: one line per visible person or thing."""
    return {"lines": [{"id": k, "cells": [list(w.pos[k])]} for k in sorted(w.pos)]}


def bag(tokens):
    return tuple(sorted(tokens))


def apply(w, e):
    """one event on a world (False if the world rules do not allow it)."""
    t = e["type"]
    if t == "go":
        return w.go(e["c"], e["t"])
    if t == "push":
        return w.push(e["c"], e["t"])
    if t == "put":
        return w.put_on(e["c"], e["t"], e["b"]) if e["rel"] == "on" else w.put_in(e["c"], e["t"], e["b"])
    if t == "take":
        return w.take_out(e["c"], e["t"], e["b"])
    raise ValueError(e)


class Items:
    def __init__(self, life):
        self.L = life
        self.lex = life.lex
        self.rows = []

    def stream(self, part):
        return rng(self.L.seed, "items", self.L.idx, part)

    # ---- bookkeeping
    def add(self, crit, kind, lang, fmt, context, question, candidates, answer, when="every", options=None, order=False,
            detail=None):
        if fmt in ("blank", "slot", "choice"):
            assert answer in candidates, (crit, kind, answer)
        item = {"when": when, "format": fmt, "context": context, "question": question, "candidates": candidates}
        if options is not None:
            item["options"] = options
        key = {"criterion": crit, "kind": kind, "detail": detail or {}, "lang": lang, "order": bool(order), "when": when,
               "answer": answer}
        self.rows.append((item, key))

    def options(self, r, scenes):
        """shuffled option scenes -> (options, label of each original index)."""
        perm = [int(i) for i in r.permutation(len(scenes))]
        opts = [{"label": "ABCDEFGH"[j], "scene": scenes[i]} for j, i in enumerate(perm)]
        return opts, {i: "ABCDEFGH"[j] for j, i in enumerate(perm)}

    # ---- words
    def noun(self, k, lang):
        return self.lex["nouns"][k][lang]

    def word(self, eid, lang):
        return self.lex["names"][eid] if eid in self.lex["names"] else self.noun(self.lex["kind_of"][eid], lang)

    def name_cands(self, lang):
        """every word that can name a thing or a person in this life: all nouns of the language and all names."""
        return sorted([self.noun(k, lang) for k in self.lex["nouns"]] + list(self.lex["names"].values()))

    def num_cands(self, lang):
        return [NUM[n][lang] for n in range(1, 11)]

    def ask(self, slot, ev, lang):
        """the event's sentence with the asked part replaced in place by the closed-v1 wh-word."""
        s = sentence(ev, self.lex, lang)
        wh = WH[slot][lang]
        if slot == "who":
            return [wh] + (s[2:] if lang == "ko" else s[1:])
        if slot == "what":
            return s[:2] + [wh, particle(wh, "object")] + s[4:] if lang == "ko" else s[:2] + [wh] + s[3:]
        if slot == "when":
            return [wh] + s
        raise ValueError(slot)

    def ask_where(self, t, lang):
        x = self.word(t, lang)
        return [x, particle(x, "subject"), WH["where"]["ko"], "에", "있다"] if lang == "ko" else [WH["where"]["en"], "is", x]

    def how_many(self, k, lang, box=None):
        x = self.noun(k, lang)
        if lang == "ko":
            return [x] + WH["how many"]["ko"] + ["가"] + ([self.word(box, "ko"), SPATIAL["in"]["ko"], "에"] if box else []) + ["있다"]
        return WH["how many"]["en"] + [x, "are"] + (["in", self.word(box, "en")] if box else ["there"])

    # ---- the order flag (bag-of-words readings)
    def readings(self, w, e, lang):
        """the events a bag-of-words reader cannot tell from e in world w: e itself and, for an event with two things, the
        same tokens with the things' roles swapped, when that sentence has the same bag and is possible in w."""
        out = [e]
        if e["type"] in ("put", "take"):
            sw = dict(e, t=e["b"], b=e["t"])
            if bag(sentence(sw, self.lex, lang)) == bag(sentence(e, self.lex, lang)) and apply(w.copy(), sw):
                out.append(sw)
        return out

    def outcomes(self, w0, evs, lang):
        """every final world reachable by reading each event in any of its bag-of-words readings."""
        worlds = [w0.copy()]
        for e in evs:
            nxt = []
            for v in worlds:
                for alt in self.readings(v, e, lang):
                    u = v.copy()
                    if apply(u, alt):
                        nxt.append(u)
            worlds = nxt
        return worlds

    def where_order(self, w0, evs, t, lang):
        return len({u.pos.get(t) for u in self.outcomes(w0, evs, lang)}) > 1

    def who_order(self, evs, ev, lang):
        q = bag(x for x in self.ask("who", ev, lang) if x != WH["who"][lang])
        cut = 2 if lang == "ko" else 1
        return len({e["c"] for e in evs if e["type"] != "go" and bag(sentence(e, self.lex, lang)[cut:]) == q}) > 1

    def when_order(self, evs, ev, lang):
        q = bag(sentence(ev, self.lex, lang))
        return sum(1 for e in evs if bag(sentence(e, self.lex, lang)) == q) > 1

    # ---- scenes
    def layout(self, r, people, kinds, box_kind=None, stack=0.3, box_apart=False):
        """a fresh scene: the people and the box on the floor, one new thing of each kind on the floor or on a plain thing.
        Ids are made in a random order (no id number tells a role). Returns (World, {kind: id}, box id) or None."""
        L = self.L
        made = ([("box", box_kind)] if box_kind is not None else []) + [("thing", k) for k in kinds]
        ids, box = {}, None
        for j in r.permutation(len(made)):
            what, k = made[int(j)]
            i = L.new_id(k)
            if what == "box":
                box = i
            else:
                ids[k] = i
        w = World()
        floor = [int(x) for x in r.permutation(W)]
        for c in people:
            w.place(c, floor.pop(), 0, person=True)
        if box is not None:
            spots = [x for x in floor if not box_apart or all(abs(x - w.pos[c][0]) > 1 for c in people)]
            if not spots:
                return None
            floor.remove(spots[0])
            w.place(box, spots[0], 0, box=True)
        for j in r.permutation(len(kinds)):
            t = ids[kinds[int(j)]]
            free_floor = [x for x in range(W) if w.free(x, 0)]
            tops = [x for x in range(W) if w.at(x, 0) is not None and w.at(x, 0) not in w.people and w.at(x, 0) not in w.box
                    and w.free(x, w.pos[w.top_of(w.at(x, 0))][1] + 1)]
            if tops and (r.random() < stack or not free_floor):
                x = int(r.choice(tops))
                w.place(t, x, w.pos[w.top_of(w.at(x, 0))][1] + 1)
            elif free_floor:
                w.place(t, int(r.choice(free_floor)), 0)
            else:
                return None
        return w, ids, box

    def pick_free(self, r, verb):
        """a (person, kind) whose triple with verb is not held out of the exposure."""
        L = self.L
        while True:
            c = L.people[int(r.integers(len(L.people)))]
            k = int(r.choice(L.plain_kinds))
            if (c, verb, k) not in L.held_triples:
                return c, k

    def event(self, r, verb, c, kind, rel=None):
        """a fresh scene where person c does verb to a thing of kind (put: rel "on" or "in"), with 1-2 other people and 1-2
        other things; first another person d goes to one of the other things (so two people are named and a name alone
        never answers "who"); a push comes with its go (as in the exposure); before a take, d puts the thing in the box
        instead. Returns (World before, [events], World after, the thing) or None."""
        L = self.L
        boxed = verb == "take" or (verb == "put" and rel == "in")
        for _ in range(400):
            others = [p for p in L.people if p != c]
            people = [c] + [others[int(i)] for i in r.choice(len(others), 1 if boxed else int(r.integers(1, 3)), replace=False)]
            ok_kinds = [k for k in L.plain_kinds if k != kind]
            kinds = [kind] + [int(k) for k in r.choice(ok_kinds, int(r.integers(1, 3)), replace=False)]
            lay = self.layout(r, people, kinds, int(r.choice(L.box_kinds)) if boxed else None)
            if lay is None:
                continue
            w, ids, box = lay
            t = ids[kind]
            w0 = w.copy()
            d = people[1]
            first = []
            if verb != "take":
                us = [x for x in ids.values() if x != t]
                first = [{"type": "go", "c": d, "t": us[int(r.integers(len(us)))]}]
                if not apply(w, first[0]):
                    continue
            if verb == "push":
                evs = first + [{"type": "go", "c": c, "t": t}, {"type": "push", "c": c, "t": t}]
                if w.pos[t][1] != 0 or not apply(w, evs[-2]) or not apply(w, evs[-1]):
                    continue
            elif verb == "put" and rel == "on":
                bases = [x for x in ids.values() if x != t and t not in w.column(x) and x not in w.column(t)]
                if not bases or not w.put_on(c, t, bases[int(r.integers(len(bases)))]):
                    continue
                evs = first + [{"type": "put", "c": c, "t": t, "rel": "on", "b": w.at(w.pos[t][0], w.pos[t][1] - 1)}]
            elif verb == "put":
                evs = first + [{"type": "put", "c": c, "t": t, "rel": "in", "b": box}]
                if not apply(w, evs[-1]):
                    continue
            else:
                c2 = people[1]
                if (c2, "put", kind) in L.held_triples:
                    continue
                evs = [{"type": "put", "c": c2, "t": t, "rel": "in", "b": box}, {"type": "take", "c": c, "t": t, "b": box}]
                if not apply(w, evs[0]) or not apply(w, evs[1]):
                    continue
            if t in w.pos and w.pos[t] == w0.pos[t]:
                continue                                # the asked cell must not be in the context
            return w0, evs, w, t
        return None

    def event_items(self, crit, r, verb, c, kind, rel, lang, qlang, kind_name, detail):
        """who / what / where about one event told in lang and asked in qlang (criteria 1 and 5)."""
        res = self.event(r, verb, c, kind, rel)
        if res is None:
            return
        w0, evs, w1, t = res
        ev = evs[-1]
        ctx = {"scene": scene(w0), "sentences": [sentence(e, self.lex, lang) for e in evs]}
        people = sorted(p for p in w0.pos if p in w0.people)
        things = sorted(x for x in w0.pos if x not in w0.people)
        self.add(crit, f"{kind_name} who", lang, "slot", ctx, {"slot": "who", "sentence": self.ask("who", ev, qlang)}, people,
                 ev["c"], order=self.who_order(evs, ev, lang) or self.who_order(evs, ev, qlang), detail=detail)
        self.add(crit, f"{kind_name} what", lang, "slot", ctx, {"slot": "what", "sentence": self.ask("what", ev, qlang)},
                 things, t, order=False, detail=detail)
        if t in w1.pos:
            self.add(crit, f"{kind_name} where", lang, "slot", ctx,
                     {"slot": "where", "sentence": self.ask_where(t, qlang), "target": t}, CELLS, list(w1.pos[t]),
                     order=self.where_order(w0, evs, t, lang), detail=detail)

    # ---- criterion 0: element memory (every name; the Korean particle allomorphs)
    def c0(self):
        L, r = self.L, self.stream(0)
        ents = [("thing", k) for k in L.plain_kinds] + [("box", k) for k in L.box_kinds] + [("person", c) for c in L.people]
        for lang in LANGS:
            frame = ["___", "있다"] if lang == "ko" else ["there", "is", "___"]
            for typ, e in ents:
                for _ in range(4 if typ == "box" else 2):
                    w = World()
                    x = int(r.integers(W))
                    if typ == "person":
                        w.place(e, x, 0, person=True)
                        ans = self.lex["names"][e]
                    else:
                        w.place(L.new_id(e), x, 0, box=typ == "box")
                        ans = self.noun(e, lang)
                    self.add(0, f"name ({typ})", lang, "blank", {"scene": scene(w)}, {"slot": "blank", "sentence": frame},
                             self.name_cands(lang), ans)
        # particles: subject (이/가), object (을/를), direction (으로/로) -- each form four times, Korean only
        for role in ("subject", "object", "direction"):
            forms = {}
            for k in L.plain_kinds:
                forms.setdefault(particle(self.noun(k, "ko"), role), []).append(k)
            for form, ks in sorted(forms.items()):
                for _ in range(4):
                    k = int(r.choice(ks))
                    w = World()
                    if role == "subject":
                        w.place(L.new_id(k), int(r.integers(W)), 0)
                        q = [self.noun(k, "ko"), "___", "있다"]
                    else:
                        verb = "push" if role == "object" else "go"
                        ok = [p for p in L.people if (p, verb, k) not in L.held_triples]
                        c = ok[int(r.integers(len(ok)))]
                        x = int(r.integers(W - 1))
                        w.place(c, x, 0, person=True)
                        w.place(L.new_id(k), x + 1, 0)
                        a = self.lex["names"][c]
                        q = [a, particle(a, "subject"), self.noun(k, "ko"), "___", self.lex["verbs"][verb]["ko"]]
                    self.add(0, f"particle ({role})", "ko", "blank", {"scene": scene(w)}, {"slot": "blank", "sentence": q},
                             KO_PARTICLES, form, detail={"form": form})

    # ---- criterion 1: known words in a combination never heard (person, verb, kind) -> who / what / where
    def c1(self):
        L, r = self.L, self.stream(1)
        for (c, verb, kind) in sorted(L.held_triples):
            for lang in LANGS:
                for rep in range(2):
                    rel = ("on", "in")[rep] if verb == "put" else None
                    self.event_items(1, r, verb, c, kind, rel, lang, lang, "new combination",
                                     {"verb": verb, "rel": rel, "triple": [c, verb, kind]})

    # ---- criterion 2: new nouns heard 1 / 2 / 4 / 8 times (in one language), right after and at the end
    def c2(self):
        L, r = self.L, self.stream(2)
        for k, (cnt, ses) in sorted(L.new_plan.items()):
            lang = L.new_lang[k]
            x = self.noun(k, lang)
            exist = [x, particle(x, "subject"), "있다"] if lang == "ko" else ["there", "is", x]
            for when, timing in ((f"after_session_{ses}", "right after"), ("end", "end")):
                detail = {"exposures": cnt, "session": ses, "timing": timing}
                for _ in range(3):              # which thing it names: beside the rare thing seen at its naming (its name
                    # is known, and it is rarer than the new thing -- so rarity points the wrong way) and another new thing
                    with_ = Counter(L.naming_with[k]).most_common(1)[0][0]
                    other_new = int(r.choice([j for j in L.new_kinds if j != k]))
                    lay = None
                    while lay is None:
                        lay = self.layout(r, [], [k, with_, other_new])
                    w, ids, _ = lay
                    self.add(2, "new word which", lang, "slot", {"scene": scene(w), "sentences": [exist]},
                             {"slot": "which"}, sorted(w.pos), ids[k], when=when, detail=detail)
                for _ in range(2):              # its name, alone in the scene
                    w = World()
                    w.place(L.new_id(k), int(r.integers(W)), 0)
                    frame = ["___", "있다"] if lang == "ko" else ["there", "is", "___"]
                    self.add(2, "new word name", lang, "blank", {"scene": scene(w)}, {"slot": "blank", "sentence": frame},
                             self.name_cands(lang), x, when=when, detail=detail)

    # ---- criterion 3: counting 1-10 -- (a) scene -> number, (b) number -> scene, (c) the next number (report only)
    def count_world(self, r, k, n):
        w = World()
        cols = [0] * W
        for _ in range(n):
            x = int(r.choice([i for i in range(W) if cols[i] < 3]))
            w.place(self.L.new_id(k), x, cols[x])
            cols[x] += 1
        return w

    def c3(self):
        L, r = self.L, self.stream(3)
        for lang in LANGS:
            for n in range(1, 11):
                for _ in range(2):
                    k = int(r.choice(L.plain_kinds))
                    self.add(3, "scene -> number", lang, "blank", {"scene": scene(self.count_world(r, k, n))},
                             {"slot": "blank", "sentence": self.how_many(k, lang)}, self.num_cands(lang), NUM[n][lang],
                             detail={"n": n})
                k = int(r.choice(L.plain_kinds))
                ds = [int(m) for m in r.choice([m for m in range(1, 11) if m != n], 3, replace=False)]
                opts, where = self.options(r, [scene(self.count_world(r, k, m)) for m in [n] + ds])
                self.add(3, "number -> scene", lang, "choice",
                         {"sentences": [sentence({"type": "count", "kind": k, "n": n}, self.lex, lang)]}, {"slot": "choice"},
                         [o["label"] for o in opts], where[0], options=opts, detail={"n": n})
            for n in range(1, 10):              # told "n X are in the box", one more X goes in -> how many
                while True:
                    c, k = self.pick_free(r, "put")
                    lay = self.layout(r, [c], [k], box_kind=int(r.choice(L.box_kinds)), stack=0.0)
                    if lay is not None:
                        break
                w, ids, box = lay
                ctx = {"scene": scene(w),
                       "sentences": [sentence({"type": "count", "kind": k, "n": n, "rel": "in", "b": box}, self.lex, lang),
                                     sentence({"type": "put", "c": c, "t": ids[k], "rel": "in", "b": box}, self.lex, lang)]}
                self.add(3, "next number (report only)", lang, "blank", ctx,
                         {"slot": "blank", "sentence": self.how_many(k, lang, box)}, self.num_cands(lang), NUM[n + 1][lang],
                         detail={"n": n})

    # ---- criterion 4: things go into a box and some come out -> how many inside; the box emptied: possible vs impossible
    def box_story(self, r, a, b, lang):
        L = self.L
        for _ in range(300):
            c = L.people[int(r.integers(len(L.people)))]
            ks = [k for k in L.plain_kinds if (c, "put", k) not in L.held_triples and (c, "take", k) not in L.held_triples]
            k = int(r.choice(ks))
            lay = self.layout(r, [c], [], box_kind=int(r.choice(L.box_kinds)), box_apart=True)
            if lay is None:
                continue
            w, _, box = lay
            moments, ok = [], True
            for _i in range(a):
                floor = [x for x in range(W) if w.free(x, 0)]
                if not floor:
                    ok = False
                    break
                t = L.new_id(k)
                w.place(t, int(r.choice(floor)), 0)
                moments.append({"sentence": sentence({"type": "exist", "a": t}, self.lex, lang), "scene": scene(w)})
                ev = {"type": "put", "c": c, "t": t, "rel": "in", "b": box}
                apply(w, ev)
                moments.append({"sentence": sentence(ev, self.lex, lang), "scene": scene(w)})
            for _i in range(b if ok else 0):
                ev = {"type": "take", "c": c, "t": w.inside[box][-1], "b": box}
                if not apply(w, ev):
                    ok = False
                    break
                moments.append({"sentence": sentence(ev, self.lex, lang), "scene": scene(w)})
            if ok:
                return w, box, k, moments
        return None

    def reveal(self, w, box, show, extra=None):
        """the box emptied with no event told (the screen lowered): the shown things stand beside the box, nearest
        columns first, lowest cells first (never in a person's column)."""
        v = w.copy()
        v.inside[box] = []
        bx = v.pos[box][0]
        cols = [x for x in range(W) if x != bx and not any(v.pos[p][0] == x for p in v.people)]
        cells = []
        for x in cols:
            y = 0
            while y < H and v.at(x, y) is not None:
                y += 1
            cells += [(x, yy) for yy in range(y, H)]
        cells.sort(key=lambda p: (p[1], abs(p[0] - bx), p[0]))
        show = list(show) + ([extra] if extra else [])
        assert len(cells) >= len(show)
        for t, cell in zip(show, cells):
            v.pos[t] = cell
        return scene(v)

    def c4(self):
        L, r = self.L, self.stream(4)
        taught = sorted({(a, b) for a in range(1, 6) for b in range(0, a)} - L.held_box)
        far = [(a, b) for a in range(6, 11) for b in range(0, a)]
        for lang in LANGS:
            picks = [(p, False) for p in sorted(x for x in L.held_box if x[0] > x[1])]
            picks += [(far[int(i)], False) for i in r.choice(len(far), 9, replace=False)]
            picks += [(taught[int(i)], True) for i in r.choice(len(taught), 4, replace=False)]
            for (a, b), is_taught in picks:
                res = self.box_story(r, a, b, lang)
                if res is None:
                    continue
                w, box, k, moments = res
                n_in = a - b
                detail = {"put": a, "take": b, "inside": n_in, "taught": is_taught}
                self.add(4, "how many inside", lang, "blank", {"moments": moments},
                         {"slot": "blank", "sentence": self.how_many(k, lang, box)}, self.num_cands(lang), NUM[n_in][lang],
                         detail=detail)
                hidden = list(w.inside[box])
                if n_in > 1 and r.random() < 0.5:
                    bad, how = self.reveal(w, box, hidden[:-1]), "one fewer"         # one of them is gone
                else:
                    bad, how = self.reveal(w, box, hidden, extra=L.new_id(k)), "one more"   # one more than went in
                opts, where = self.options(r, [self.reveal(w, box, hidden), bad])
                self.add(4, "box emptied (possible vs impossible)", lang, "pair", {"moments": moments}, {"slot": "pair"},
                         [o["label"] for o in opts], {"impossible": where[1], "possible": where[0]}, options=opts,
                         detail=dict(detail, impossible=how))

    # ---- criterion 5: criterion 1 per language (graded from the criterion-1 items) + a fact told in one language only,
    # asked in the other
    def c5(self):
        r = self.stream(5)
        for lang in LANGS:
            other = "en" if lang == "ko" else "ko"
            for verb, rel in (("push", None), ("put", "on"), ("put", "in"), ("take", None)):
                for _ in range(2):
                    c, kind = self.pick_free(r, verb)
                    self.event_items(5, r, verb, c, kind, rel, lang, other, "cross-language",
                                     {"verb": verb, "rel": rel, "told": lang, "asked": other})

    # ---- criterion 6: support, solidity, contact, permanence: the next scene (3 options) and a possible/impossible pair
    def c6(self):
        r = self.stream(6)
        makers = [(a, m, False) for a, m in (("support", self.c6_support), ("solidity", self.c6_solid),
                                             ("contact", self.c6_contact), ("permanence", self.c6_permanence))]
        # configurations the exposure never shows (EXAM_DESIGN.md 5: reported as their own group; user decision 10-13):
        # a three-high column, a row of four, a pushed row carrying a three-high column, two kinds in one box
        makers += [(a, m, True) for a, m in (("support", self.c6n_support), ("solidity", self.c6n_solid),
                                            ("contact", self.c6n_contact), ("permanence", self.c6n_permanence))]
        for lang in LANGS:
            for area, maker, novel in makers:
                for _ in range(3):
                    res = maker(r)
                    if res is None:
                        continue
                    w0, evs, good, bad, bad2 = res
                    w, moments = w0.copy(), []             # every event before the asked one is told and shown
                    for e in evs[:-1]:
                        apply(w, e)
                        moments.append({"sentence": sentence(e, self.lex, lang), "scene": scene(w)})
                    ctx = {"scene": scene(w0), "sentences": [sentence(evs[-1], self.lex, lang)]}
                    if moments:
                        ctx["moments"] = moments
                    order = len({tuple(sorted(u.pos.items())) for u in self.outcomes(w, evs[-1:], lang)}) > 1
                    detail = {"area": area, "novel": novel}
                    opts, where = self.options(r, [scene(good), scene(bad), scene(bad2)])
                    self.add(6, f"{area} prediction", lang, "choice", ctx, {"slot": "choice"}, [o["label"] for o in opts],
                             where[0], options=opts, order=order, detail=detail)
                    opts, where = self.options(r, [scene(good), scene(bad)])
                    self.add(6, f"{area} pair", lang, "pair", ctx, {"slot": "pair"}, [o["label"] for o in opts],
                             {"impossible": where[1], "possible": where[0]}, options=opts, order=order, detail=detail)

    def c6_support(self, r):
        """a thing with another on top goes into a box: the top one falls straight down (possible) / stays up with nothing
        under it, or lands in another column (impossible)."""
        L = self.L
        for _ in range(300):
            c, _k = self.pick_free(r, "put")
            lay = self.layout(r, [c], [int(k) for k in r.choice(L.plain_kinds, 2, replace=False)],
                              box_kind=int(r.choice(L.box_kinds)), stack=1.0)
            if lay is None:
                continue
            w, ids, box = lay
            lows = [t for t in ids.values() if w.pos[t][1] == 0 and w.has_on_top(t) and L.ok_triple(c, "put", t)]
            if not lows:
                continue
            t = lows[0]
            top = w.at(w.pos[t][0], 1)
            w0 = w.copy()
            ev = {"type": "put", "c": c, "t": t, "rel": "in", "b": box}
            if not apply(w, ev):
                continue
            air = w.copy()
            air.pos[top] = w0.pos[top]
            side = w.copy()
            spots = [x for x in range(W) if side.free(x, 0) and x != w0.pos[t][0]]
            if not spots:
                continue
            side.pos[top] = (int(r.choice(spots)), 0)
            return w0, [ev], w, air, side
        return None

    def _row(self, r, n, room):
        """n things side by side on the floor and a person on the open side; room = a free cell after the row (else the
        row ends at a wall, and the person stands at least one cell away so that the go moves them and the possible
        outcome is never the scene before). The row runs to the right or to the left."""
        L = self.L
        for _ in range(300):
            c = L.people[int(r.integers(len(L.people)))]
            kinds = [int(k) for k in r.choice(L.plain_kinds, n, replace=False)]
            ids = [L.new_id(kinds[int(j)]) for j in r.permutation(n)]
            start = int(r.integers(1, W - n)) if room else W - n
            xs = [start + i for i in range(n)]
            open_side = list(range(0, start if room else start - 1))   # where the person may stand (before the row)
            if r.random() < 0.5:                        # mirror: the row runs to the left
                xs = [W - 1 - x for x in xs]
                open_side = [W - 1 - x for x in open_side]
            if not L.ok_triple(c, "push", ids[0]) or not open_side:
                continue
            w = World()
            for t, x in zip(ids, xs):
                w.place(t, x, 0)
            w.place(c, int(r.choice(open_side)), 0, person=True)
            return w, c, ids
        return None

    def c6_solid(self, r):
        """a push on a row that ends at the wall: nothing moves (possible) / the pushed thing moves into the next one's cell,
        or rises into the air (impossible)."""
        for _ in range(300):
            res = self._row(r, int(r.integers(2, 4)), room=False)
            if res is None:
                return None
            w, c, ts = res
            w0 = w.copy()
            evs = [{"type": "go", "c": c, "t": ts[0]}, {"type": "push", "c": c, "t": ts[0]}]
            if not apply(w, evs[0]) or apply(w.copy(), evs[1]):
                continue
            through = w.copy()
            through.pos[ts[0]] = w.pos[ts[1]]
            x0 = w.pos[ts[0]][0]
            if not w.free(x0, 1):
                continue
            up = w.copy()
            up.pos[ts[0]] = (x0, 1)
            return w0, evs, w, through, up
        return None

    def c6_contact(self, r):
        """a push on a row with room after it: every touching thing moves one cell (possible) / only the far thing moves
        (moved without contact), or the pushed thing moves into the next one's cell (impossible)."""
        for _ in range(300):
            res = self._row(r, int(r.integers(2, 4)), room=True)
            if res is None:
                return None
            w, c, ts = res
            w0 = w.copy()
            evs = [{"type": "go", "c": c, "t": ts[0]}, {"type": "push", "c": c, "t": ts[0]}]
            if not apply(w, evs[0]):
                continue
            mid = w.copy()
            if not apply(w, evs[1]):
                continue
            far = mid.copy()
            far.pos[ts[-1]] = w.pos[ts[-1]]
            first = mid.copy()
            first.pos[ts[0]] = mid.pos[ts[1]]
            return w0, evs, w, far, first
        return None

    def c6_permanence(self, r):
        """a thing is put in a box and taken out: it comes out beside the box (possible) / a thing of another kind comes out
        instead, or it comes out far from the box (impossible)."""
        L = self.L
        for _ in range(300):
            c, kind = self.pick_free(r, "take")
            if (c, "put", kind) in L.held_triples:
                continue
            lay = self.layout(r, [c], [kind, int(r.choice([k for k in L.plain_kinds if k != kind]))],
                              box_kind=int(r.choice(L.box_kinds)), stack=0.0)
            if lay is None:
                continue
            w, ids, box = lay
            t = ids[kind]
            w0 = w.copy()
            evs = [{"type": "put", "c": c, "t": t, "rel": "in", "b": box}, {"type": "take", "c": c, "t": t, "b": box}]
            if not apply(w, evs[0]) or not apply(w, evs[1]) or w.pos[t] == w0.pos[t]:
                continue
            fx = [x for x in range(W) if w.free(x, 0) and abs(x - w.pos[box][0]) > 1 and x != w0.pos[t][0]]
            if not fx:
                continue
            other = w.copy()
            other.pos[L.new_id(int(r.choice([k for k in L.plain_kinds if k != kind])))] = other.pos.pop(t)
            far = w.copy()
            far.pos[t] = (int(r.choice(fx)), 0)
            return w0, evs, w, other, far
        return None

    # ---- criterion 6, configurations never in the exposure (each returns w0, events, possible, impossible, impossible 2)
    def c6n_support(self, r):
        """a three-high column loses its bottom thing into a box: the two above fall one cell each and stay stacked
        (possible) / the top one stays where it was with a gap under it, or lands in another column (impossible)."""
        L = self.L
        for _ in range(300):
            c, _k = self.pick_free(r, "put")
            kinds = [int(k) for k in r.choice(L.plain_kinds, 3, replace=False)]
            ids = [L.new_id(kinds[int(j)]) for j in r.permutation(3)]
            box = L.new_id(int(r.choice(L.box_kinds)))
            cells = [int(x) for x in r.permutation(W)]
            w = World()
            x = cells.pop()
            for y, t in enumerate(ids):
                w.place(t, x, y)
            w.place(box, cells.pop(), 0, box=True)
            w.place(c, cells.pop(), 0, person=True)
            if not L.ok_triple(c, "put", ids[0]):
                continue
            w0 = w.copy()
            ev = {"type": "put", "c": c, "t": ids[0], "rel": "in", "b": box}
            if not apply(w, ev):
                continue
            gap = w.copy()
            gap.pos[ids[2]] = w0.pos[ids[2]]
            side = w.copy()
            spots = [z for z in range(W) if side.free(z, 0)]
            if not spots:
                continue
            side.pos[ids[2]] = (int(r.choice(spots)), 0)
            return w0, [ev], w, gap, side
        return None

    def c6n_solid(self, r):
        """a row of four against the wall is pushed: nothing moves (possible) / the pushed thing moves into the next one's
        cell, or rises into the air (impossible)."""
        L = self.L
        for _ in range(300):
            c = L.people[int(r.integers(len(L.people)))]
            kinds = [int(k) for k in r.choice(L.plain_kinds, 4, replace=False)]
            ids = [L.new_id(kinds[int(j)]) for j in r.permutation(4)]
            right = r.random() < 0.5
            xs = [1, 2, 3, 4] if right else [3, 2, 1, 0]
            if not L.ok_triple(c, "push", ids[0]):
                continue
            w = World()
            for t, x in zip(ids, xs):
                w.place(t, x, 0)
            w.place(c, 0 if right else W - 1, 0, person=True)
            w0 = w.copy()
            evs = [{"type": "go", "c": c, "t": ids[0]}, {"type": "push", "c": c, "t": ids[0]}]
            if not apply(w, evs[0]) or apply(w.copy(), evs[1]):
                continue
            through = w.copy()
            through.pos[ids[0]] = w.pos[ids[1]]
            up = w.copy()
            up.pos[ids[0]] = (w.pos[ids[0]][0], 1)
            return w0, evs, w, through, up
        return None

    def c6n_contact(self, r):
        """a pushed thing touches a three-high column: the thing and the whole column move one cell (possible) / only the
        column moves (no contact), or the pushed thing moves into the column's cell (impossible)."""
        L = self.L
        for _ in range(300):
            c = L.people[int(r.integers(len(L.people)))]
            kinds = [int(k) for k in r.choice(L.plain_kinds, 4, replace=False)]
            ids = [L.new_id(kinds[int(j)]) for j in r.permutation(4)]
            p, col = ids[0], ids[1:]
            start = int(r.integers(1, W - 2))             # p at start, the column at start + 1, room at start + 2
            xs = (start, start + 1)
            cx = start - 1
            if r.random() < 0.5:
                xs, cx = (W - 1 - xs[0], W - 1 - xs[1]), W - 1 - cx
            if not L.ok_triple(c, "push", p):
                continue
            w = World()
            w.place(p, xs[0], 0)
            for y, t in enumerate(col):
                w.place(t, xs[1], y)
            w.place(c, cx, 0, person=True)
            w0 = w.copy()
            evs = [{"type": "go", "c": c, "t": p}, {"type": "push", "c": c, "t": p}]
            if not apply(w, evs[0]):
                continue
            mid = w.copy()
            if not apply(w, evs[1]):
                continue
            only_col = mid.copy()
            for t in col:
                only_col.pos[t] = w.pos[t]
            into = mid.copy()
            into.pos[p] = mid.pos[col[0]]
            return w0, evs, w, only_col, into
        return None

    def c6n_permanence(self, r):
        """two things of different kinds go into a box, then one of them is taken out: that one comes out (possible) / the
        other one comes out instead, or both come out (impossible)."""
        L = self.L
        for _ in range(300):
            c = L.people[int(r.integers(len(L.people)))]
            k1, k2 = [int(k) for k in r.choice(L.plain_kinds, 2, replace=False)]
            if any((c, v, k) in L.held_triples for v in ("put", "take") for k in (k1, k2)):
                continue
            lay = self.layout(r, [c], [k1, k2], box_kind=int(r.choice(L.box_kinds)), stack=0.0, box_apart=True)
            if lay is None:
                continue
            w, ids, box = lay
            a, b = ids[k1], ids[k2]
            w0 = w.copy()
            evs = [{"type": "put", "c": c, "t": a, "rel": "in", "b": box}, {"type": "put", "c": c, "t": b, "rel": "in", "b": box},
                   {"type": "take", "c": c, "t": a, "b": box}]
            if not all(apply(w, e) for e in evs):
                continue
            other = w.copy()
            other.pos[b] = other.pos.pop(a)
            other.inside[box] = [a]
            both = w.copy()
            both.inside[box] = []
            spots = [x for x in range(W) if both.free(x, 0) and abs(x - both.pos[box][0]) == 1]
            if not spots:
                continue
            both.pos[b] = (spots[0], 0)
            return w0, evs, w, other, both
        return None

    # ---- criteria 7 and 8: stories told in sentences only (the scene before the first sentence is shown)
    def story(self, r, n_sent, focus_bias=0.0):
        """two people push and put three things of different kinds; a push is two sentences (go, push); no event is told
        twice in one story. Returns (World before, [events], World after, people) or None."""
        L = self.L
        for _ in range(300):
            people = [L.people[int(i)] for i in r.choice(len(L.people), 2, replace=False)]
            lay = self.layout(r, people, [int(k) for k in r.choice(L.plain_kinds, 3, replace=False)])
            if lay is None:
                continue
            w, ids, _ = lay
            w0 = w.copy()
            things = sorted(ids.values())
            focus = things[int(r.integers(len(things)))]
            evs = []
            for _try in range(100):
                if len(evs) >= n_sent:
                    break
                c = people[int(r.integers(2))]
                t = focus if r.random() < focus_bias else things[int(r.integers(len(things)))]
                told = {(e["type"], e["t"], e.get("b")) for e in evs}
                if r.random() < 0.5:
                    pair = [{"type": "go", "c": c, "t": t}, {"type": "push", "c": c, "t": t}]
                    v = w.copy()
                    if n_sent - len(evs) < 2 or w.pos[t][1] != 0 or not L.ok_triple(c, "push", t) \
                            or ("push", t, None) in told or not (apply(v, pair[0]) and apply(v, pair[1])):
                        continue
                    apply(w, pair[0])
                    apply(w, pair[1])
                    evs += pair
                else:
                    bases = [b for b in things if b != t and t not in w.column(b) and b not in w.column(t)]
                    if not L.ok_triple(c, "put", t) or not bases:
                        continue
                    v = w.copy()
                    if not v.put_on(c, t, bases[int(r.integers(len(bases)))]):
                        continue
                    ev = {"type": "put", "c": c, "t": t, "rel": "on", "b": v.at(v.pos[t][0], v.pos[t][1] - 1)}
                    if ("put", t, ev["b"]) in told:
                        continue
                    w = v
                    evs.append(ev)
            if len(evs) == n_sent:
                return w0, evs, w, people
        return None

    def track(self, w0, evs, t):
        """the cells t stood in, in order (a new entry each time it moved)."""
        w = w0.copy()
        out = [w.pos[t]]
        for e in evs:
            apply(w, e)
            if w.pos[t] != out[-1]:
                out.append(w.pos[t])
        return out

    def c7(self):
        r = self.stream(7)
        for lang in LANGS:
            for _ in range(5):
                n = int(r.integers(2, 5))
                res = self.story(r, n)
                if res is None:
                    continue
                w0, evs, w1, people = res
                ctx = {"scene": scene(w0), "sentences": [sentence(e, self.lex, lang) for e in evs]}
                main = [i for i, e in enumerate(evs) if e["type"] != "go"]
                i = main[int(r.integers(len(main)))]
                ev = evs[i]
                things = sorted(x for x in w0.pos if x not in w0.people)

                def same(e, keys):
                    return all(e.get(k) == ev.get(k) for k in keys)
                detail = {"sentences": n, "verb": ev["type"]}
                if not any(same(e, ("type", "t", "rel", "b")) and e["c"] != ev["c"] for e in evs):
                    self.add(7, "story who", lang, "slot", ctx, {"slot": "who", "sentence": self.ask("who", ev, lang)},
                             sorted(people), ev["c"], order=self.who_order(evs, ev, lang), detail=detail)
                if not any(same(e, ("type", "c", "rel", "b")) and e["t"] != ev["t"] for e in evs):
                    self.add(7, "story what", lang, "slot", ctx, {"slot": "what", "sentence": self.ask("what", ev, lang)},
                             things, ev["t"], order=False, detail=detail)
                if w1.pos[ev["t"]] != w0.pos[ev["t"]]:
                    self.add(7, "story where", lang, "slot", ctx,
                             {"slot": "where", "sentence": self.ask_where(ev["t"], lang), "target": ev["t"]}, CELLS,
                             list(w1.pos[ev["t"]]), order=self.where_order(w0, evs, ev["t"], lang), detail=detail)
                if sum(1 for e in evs if same(e, ("type", "c", "t", "rel", "b"))) == 1:
                    self.add(7, "story when", lang, "slot", ctx, {"slot": "when", "sentence": self.ask("when", ev, lang)},
                             list(range(n)), i, order=self.when_order(evs, ev, lang), detail=detail)

    def c8(self):
        r = self.stream(8)
        for lang in LANGS:
            for long in (False, True):
                made = 0
                for _ in range(60):
                    if made == 4:
                        break
                    n = int(r.integers(2, 5)) * (2 if long else 1)
                    res = self.story(r, n, focus_bias=0.5)
                    if res is None:
                        continue
                    w0, evs, w1, _ = res
                    things = sorted(x for x in w0.pos if x not in w0.people)
                    fits = [t for t in things if len(self.track(w0, evs, t)) >= 3
                            and self.track(w0, evs, t)[-1] != w0.pos[t] and self.track(w0, evs, t)[-2] != w0.pos[t]]
                    if not fits:
                        continue
                    t = fits[int(r.integers(len(fits)))]
                    tr = self.track(w0, evs, t)
                    made += 1
                    ctx = {"scene": scene(w0), "sentences": [sentence(e, self.lex, lang) for e in evs]}
                    detail = {"sentences": n, "long": long, "moves": len(tr) - 1}
                    order = self.where_order(w0, evs, t, lang)
                    self.add(8, "where now", lang, "slot", ctx, {"slot": "where now", "target": t}, CELLS, list(tr[-1]),
                             order=order, detail=detail)
                    self.add(8, "where before", lang, "slot", ctx, {"slot": "where before", "target": t}, CELLS,
                             list(tr[-2]), order=order, detail=detail)

    # ---- criterion 9: spatial words
    def c9(self):
        L, r = self.L, self.stream(9)
        for lang in LANGS:
            words = [SPATIAL[e][lang] for e in ("on", "under", "in", "beside")]
            for rel in ("on", "under", "beside"):
                for _ in range(3):                  # scene -> the spatial word
                    for _t in range(300):
                        kinds = [int(k) for k in r.choice(L.plain_kinds, 2, replace=False)]
                        lay = self.layout(r, [], kinds, stack=0.0 if rel == "beside" else 1.0)
                        if lay is None:
                            continue
                        w, ids, _ = lay
                        a, b = ids[kinds[0]], ids[kinds[1]]
                        if w.relation(a, b) == rel:
                            break
                    else:
                        continue
                    an, bn = self.word(a, lang), self.word(b, lang)
                    q = [an, particle(an, "subject"), bn, "___", "에", "있다"] if lang == "ko" else ["there", "is", an, "___", bn]
                    ev = {"type": "exist", "a": a, "rel": rel, "b": b}
                    order = rel != "beside" and bag(sentence(dict(ev, a=b, b=a), self.lex, lang)) == \
                        bag(sentence(ev, self.lex, lang))
                    self.add(9, "scene -> word", lang, "blank", {"scene": scene(w)}, {"slot": "blank", "sentence": q},
                             words, SPATIAL[rel][lang], order=order, detail={"rel": rel})
            for rel in ("in", "on", "in", "on"):    # a put with its spatial word blanked, always onto or into a box (so the
                for _t in range(300):               # base's noun does not tell the word); the scenes before and after
                    c, kind = self.pick_free(r, "put")
                    lay = self.layout(r, [c], [kind, int(r.choice([k for k in L.plain_kinds if k != kind]))],
                                      box_kind=int(r.choice(L.box_kinds)), stack=0.0)
                    if lay is None:
                        continue
                    w0, ids, box = lay
                    w1 = w0.copy()
                    ev = {"type": "put", "c": c, "t": ids[kind], "rel": rel, "b": box}
                    if apply(w1, ev):
                        break
                else:
                    continue
                s = ["___" if x == SPATIAL[rel][lang] else x for x in sentence(ev, self.lex, lang)]
                self.add(9, "event -> word", lang, "blank", {"scene": scene(w0), "scene_after": scene(w1)},
                         {"slot": "blank", "sentence": s}, words, SPATIAL[rel][lang], detail={"rel": rel})
            for rel in ("on", "under", "beside"):
                for _ in range(2):                  # the spatial word -> the scene: the same two things in each relation
                    kinds = [int(k) for k in r.choice(L.plain_kinds, 2, replace=False)]
                    made = [L.new_id(kinds[int(j)]) for j in r.permutation(2)]
                    a, b = made if r.random() < 0.5 else made[::-1]
                    x = int(r.integers(0, W - 1))
                    scenes = {}
                    for rr in ("on", "under", "beside"):
                        w = World()
                        if rr == "on":
                            w.place(b, x, 0)
                            w.place(a, x, 1)
                        elif rr == "under":
                            w.place(a, x, 0)
                            w.place(b, x, 1)
                        else:
                            w.place(b, x, 0)
                            w.place(a, x + 1, 0)
                        scenes[rr] = scene(w)
                    keys = [rel] + [rr for rr in ("on", "under", "beside") if rr != rel]
                    opts, where = self.options(r, [scenes[k] for k in keys])
                    ev = {"type": "exist", "a": a, "rel": rel, "b": b}
                    order = rel != "beside" and bag(sentence(dict(ev, a=b, b=a), self.lex, lang)) == \
                        bag(sentence(ev, self.lex, lang))
                    self.add(9, "word -> scene", lang, "choice", {"sentences": [sentence(ev, self.lex, lang)]},
                             {"slot": "choice"}, [o["label"] for o in opts], where[0], options=opts, order=order,
                             detail={"rel": rel})

    def build(self):
        """all items of the life, shuffled; returns (model-facing items, answer keys) with matching ids."""
        for part in (self.c0, self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9):
            part()
        perm = self.stream("order").permutation(len(self.rows))
        items, keys = [], []
        for n, j in enumerate(perm):
            item, key = self.rows[int(j)]
            iid = f"L{self.L.idx:03d}-{n:04d}"
            items.append(dict({"id": iid}, **item))
            keys.append(dict({"id": iid}, **key))
        return items, keys
