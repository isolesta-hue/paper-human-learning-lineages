"""W6b converter conv-v1: the shared sealed exam's records (gen-v1, exam_spec/GENERATOR.md 4-7) -> Dami's (Voyager's) input.

Fixed before building (W6_DESIGN.md 2 and 2-1, 2026-10-15; the user: "응, 그렇게 진행해줘" to the faithful converter; "테스트에
최적화된 로직을 설계하는 실수는 경계하고"). The converter only MOVES the records into Dami's form; it adds nothing and loses
nothing (a format check rebuilds every record from the converted moments):
  - a word (closed or open)       -> one element; features = the common rule exam.common.feature(word) (random, seeded by the
                                     word's letters; the same rule for closed and open words); no translation, no pairing
  - an object / person id         -> one picture element; features = exam.common.feature_of_kind(kind, id), kind = the id up
                                     to "_" ("t6_1" -> "t6": same shape number = same kind; a person "c2" is its own kind)
  - a cell [x, y]                 -> Dami's position element at(x - 2, y - 2) (her 5 x 5 "where" code, data_ascii's rule: x / y
                                     linear in the features + the "at" role code); the only rule that changes anything
  - one exposure line             -> two or more MOMENTS, in order: the sentence (its word elements as heard) + one moment per
                                     scene line (that thing's picture + its at); a session = one of Dami's categories
  - an item's context             -> its moments in time order (first scene, moments, sentences, scene after) -- joined by the
                                     boundary element when given as one sequence; the question sentence follows a boundary
  - a blank (___)                 -> removed from the question; one score per candidate word
  - a slot (who / what / which = id, where = cell, when = sentence number) -> one score per candidate: id -> its picture, cell
                                     -> its at, sentence number -> that context sentence's moment
  - a choice / a pair             -> the WHOLE-SCENE likelihood of each option: the product over every thing seen in the
                                     context or the option of the body's probability of that thing's state in the option (its
                                     at, or "not seen" when the option has no line for it). A per-line average cannot tell one
                                     thing more / less or two in one cell (Pioneer's trap, New Horizons 16). Pair: the two
                                     likelihoods as probabilities.
  - the answer file               -> {"id", "after_session", "scores"} (the shared grader reads it)
Not done (W6_DESIGN 2-1): pairing a scene with the next, from / to role markers, telling which word names which thing, any
language / item-kind / criterion mark, adding or removing things. "slot" and "target" pass through as the exam gives them to
every model. Format checks (check_conv.py) use the development seed's cases only (exam_dev/dev1000) -- never as training
material.
"""
import hashlib
import json
import os
import sys

import numpy as np

import data_ascii as DA

VERSION = "conv-v1"
EXAM_ROOT = os.environ.get("VOY_EXAM", "/home/user/shared-curriculum")
if EXAM_ROOT not in sys.path:
    sys.path.insert(0, EXAM_ROOT)
from exam import common as EC                      # noqa: E402  (the shared, read-only exam package)

GRID = 5                                           # exam/world.py W, H = Dami's 5 x 5
BOUNDARY = "<conv-v1 boundary>"                    # the boundary element's name (no exam word looks like it)
ABSENT = None                                      # a thing with no line in a scene ("not seen")


def kind_of_id(i):
    return i.split("_", 1)[0]


def at_features(tag):
    """Dami's 25 "at" position features (data_ascii.build's code with roles: x and y linear in the first WHERE_DIMS -
    ROLE_DIMS features, the "at" role code in the next ROLE_DIMS, the rest 0.5), from the converter's own random stream."""
    rng = EC.rng(VERSION, tag, "where")
    xy = DA.WHERE_DIMS - DA.ROLE_DIMS
    su, sv = rng.choice([-1, 1], xy // 2), rng.choice([-1, 1], xy // 2)
    H = np.array([[1]])
    while len(H) < DA.ROLE_DIMS:
        H = np.block([[H, H], [H, -H]])
    flip = rng.choice([-1, 1], DA.ROLE_DIMS)
    code_at = 0.5 + DA.ROLE_STEP * H[DA.ROLES.index("at") + 1] * flip
    out = {}
    for x in range(-DA.HALF, DA.HALF + 1):
        for y in range(-DA.HALF, DA.HALF + 1):
            v = np.full(DA.RAW_DIM, 0.5)
            v[:xy] = np.concatenate([0.5 + DA.STEP * x * su, 0.5 + DA.STEP * y * sv]) + rng.normal(0, DA.POS_NOISE, xy)
            v[xy:DA.WHERE_DIMS] = code_at
            out[(x, y)] = np.clip(v, 0, 1)
    return out


class World:
    """Dami's input for one exam life (data_ascii.World's interface where it applies): vecs (V x RAW_DIM), names, origin,
    cats, channel_of (0 where, 1 word, 2 picture), pos_of / pos_at / role_of, categories (0 = the elements, 1..16 = the
    sessions 0..15 with their heard moments in the exam's order), phases, plus the converter's index: moment_of (per heard
    moment: session, story, moment, part) for the rebuild check, id_of / word_of / cell_of, item_only_from (elements first met
    in the items, appended last)."""


class Elements:
    def __init__(self, tag):
        self.vecs, self.names, self.origin, self.cats = [], [], [], []
        self.word, self.thing = {}, {}
        self.pos_at, self.pos_of = {}, {}
        for (x, y), v in at_features(tag).items():
            t = self._add(v, f"at({x},{y})", "element", 0)
            self.pos_at[(x, y)], self.pos_of[t] = t, (x, y)
        self.boundary = self._add(EC.feature(BOUNDARY, DA.RAW_DIM), BOUNDARY, "element", 0)

    def _add(self, v, name, origin, cat):
        self.vecs.append(np.asarray(v, dtype=np.float32))
        self.names.append(name)
        self.origin.append(origin)
        self.cats.append(cat)
        return len(self.vecs) - 1

    def w(self, word, cat):
        if word not in self.word:
            self.word[word] = self._add(EC.feature(word, DA.RAW_DIM), word, "word", cat)
        return self.word[word]

    def p(self, i, cat):
        if i not in self.thing:
            self.thing[i] = self._add(EC.feature_of_kind(kind_of_id(i), i, DA.RAW_DIM), i, "picture", cat)
        return self.thing[i]

    def at(self, cell):
        x, y = cell
        assert 0 <= x < GRID and 0 <= y < GRID, cell
        return self.pos_at[(x - DA.HALF, y - DA.HALF)]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def scene_moments(E, scene, cat):
    """one moment per scene line: [picture, at] (one cell per line in gen-v1; a line with more cells keeps them all)."""
    return [[E.p(ln["id"], cat)] + [E.at(c) for c in ln["cells"]] for ln in scene["lines"]]


def convert_life(life_dir, tag=None):
    """-> (World, [Query]). tag seeds the position code (default: the life folder's name)."""
    tag = tag or os.path.basename(os.path.normpath(life_dir))
    expo = load_jsonl(os.path.join(life_dir, "exposure.jsonl"))
    items = load_jsonl(os.path.join(life_dir, "items.jsonl"))
    E = Elements(tag)
    n_sess = 1 + max(r["session"] for r in expo)
    cats = [{"heard": [], "test": [], "second": False, "words": [], "room": None}] + \
           [{"heard": [], "test": [], "second": False, "words": [], "room": s + 1, "session": s} for s in range(n_sess)]
    moment_of = []
    for r in expo:
        c = r["session"] + 1
        before = len(E.vecs)
        heard = cats[c]["heard"]
        heard.append(("conv sentence", [E.w(x, c) for x in r["sentence"]], []))
        moment_of.append((r["session"], r["story"], r["moment"], "sentence", r["scene"]["moment"]))
        for j, m in enumerate(scene_moments(E, r["scene"], c)):
            heard.append(("conv line", m, []))
            moment_of.append((r["session"], r["story"], r["moment"], "line", j))
        cats[c]["words"] += list(range(before, len(E.vecs)))
    item_only_from = len(E.vecs)
    queries = [convert_item(E, it) for it in items]
    w = World()
    w.version, w.tag = VERSION, tag
    w.vecs = np.array(E.vecs, dtype=np.float32)
    w.names, w.origin, w.cats = E.names, E.origin, np.array(E.cats)
    w.V, w.n_cat, w.categories = len(E.vecs), len(cats), cats
    w.pos_at, w.pos_of = E.pos_at, E.pos_of
    w.role_of = {t: "at" for t in E.pos_of}
    w.where_dims = DA.WHERE_DIMS
    w.boundary = E.boundary
    w.channels = ["where", "word", "picture"]
    w.channel_of = np.array([0 if o == "element" and t in E.pos_of else 2 if o == "picture" else 1
                             for t, o in enumerate(E.origin)])
    w.kinds = ["at", "word", "picture", "boundary"]
    w.kind_of = np.array([0 if t in E.pos_of else 3 if t == E.boundary else 2 if o == "picture" else 1
                          for t, o in enumerate(E.origin)])
    w.word_of = {t: x for x, t in E.word.items()}
    w.id_of = {t: i for i, t in E.thing.items()}
    w.cell_of = {t: (x + DA.HALF, y + DA.HALF) for t, (x, y) in E.pos_of.items()}
    w.moment_of = moment_of
    w.item_only_from = item_only_from
    w.sessions = list(range(1, len(cats)))
    w.phases = [[0] + w.sessions]
    return w, queries


class Query:
    """one item in Dami's terms. moments: [(part, k, tokens)] in time order (part: "scene" / "moment sentence" / "moment
    line" / "sentence" / "scene after"; k = the moment / sentence / line number); question: the asked sentence's tokens with
    the blank removed (blank_at = where it was, or None); target: a picture element or None; slot / format / when / id as
    the exam gives them; cands: [(key, token)] -- key = the answer file's score key, token = an element (word, picture or
    at), or ("moment", index into moments) for a sentence number; options: [(label, [(picture, at or None)] over every thing
    seen in the context or any option -- the whole-scene states)]."""

    def sequence(self, boundary):
        """the context and the question as ONE sequence: moments joined by the boundary element, then the question."""
        out = []
        for _, _, toks in self.moments:
            out += toks + [boundary]
        return out + list(self.question)


def convert_item(E, it):
    q = Query()
    q.id, q.when, q.format = it["id"], it["when"], it["format"]
    q.slot = it["question"]["slot"]
    ctx = it["context"]
    q.moments = []
    if "scene" in ctx:
        q.moments += [("scene", j, m) for j, m in enumerate(scene_moments(E, ctx["scene"], -1))]
    for k, mo in enumerate(ctx.get("moments", [])):
        q.moments.append(("moment sentence", k, [E.w(x, -1) for x in mo["sentence"]]))
        q.moments += [("moment line", k, m) for m in scene_moments(E, mo["scene"], -1)]
    for k, s in enumerate(ctx.get("sentences", [])):
        q.moments.append(("sentence", k, [E.w(x, -1) for x in s]))
    if "scene_after" in ctx:
        q.moments += [("scene after", j, m) for j, m in enumerate(scene_moments(E, ctx["scene_after"], -1))]
    sent = it["question"].get("sentence")
    q.has_sentence = sent is not None
    q.blank_at = sent.index("___") if sent is not None and "___" in sent else None
    q.question = [E.w(x, -1) for x in (sent or []) if x != "___"]
    q.target = E.p(it["question"]["target"], -1) if "target" in it["question"] else None
    q.cands, q.options = [], []
    if q.format in ("choice", "pair"):
        seen = []
        for _, _, toks in q.moments:
            if toks and toks[0] in E.thing.values() and toks[0] not in seen and len(toks) >= 2 and toks[1] in E.pos_of:
                seen.append(toks[0])
        opt = [(o["label"], {E.p(ln["id"], -1): [E.at(c) for c in ln["cells"]] for ln in o["scene"]["lines"]})
               for o in it["options"]]
        for _, d in opt:
            seen += [t for t in d if t not in seen]
        q.things = seen
        q.option_order = {lab: [E.thing[ln["id"]] for ln in o["scene"]["lines"]] for (lab, _), o in zip(opt, it["options"])}
        q.options = [(lab, [(t, d[t][0] if t in d else ABSENT) for t in seen]) for lab, d in opt]
        assert all(len(v) == 1 for _, d in opt for v in d.values()), "gen-v1: one cell per line"
        q.cands = [(lab, lab) for lab, _ in opt]
    else:
        for c in it["candidates"]:                 # the candidate's type from the record itself (gen-v1, GENERATOR 6-7)
            if isinstance(c, list):                # a cell
                q.cands.append((f"{c[0]},{c[1]}", E.at(c)))
            elif isinstance(c, int):               # a sentence number of the context
                k = [j for j, (part, kk, _) in enumerate(q.moments) if part == "sentence" and kk == c]
                assert len(k) == 1, (it["id"], c)
                q.cands.append((str(c), ("moment", k[0])))
            elif q.format == "blank":              # a word
                q.cands.append((c, E.w(c, -1)))
            else:                                  # an id of the scene
                q.cands.append((c, E.p(c, -1)))
    return q


def answer(body, q, after_session):
    """one answer-file row. The body answers WITHOUT learning (the runner checks its fingerprint)."""
    if q.format in ("choice", "pair"):
        ll = {}
        cache = {}
        for lab, states in q.options:
            s = 0.0
            for t, a in states:
                if t not in cache:
                    cache[t] = body.state_probs(q, t)
                s += float(np.log(max(float(cache[t].get(a, 0.0)), 1e-12)))
            ll[lab] = s
        if q.format == "choice":
            return {"id": q.id, "after_session": after_session, "scores": ll}
        m = max(ll.values())
        z = sum(np.exp(v - m) for v in ll.values())
        return {"id": q.id, "after_session": after_session,
                "scores": {lab: float(np.exp(v - m) / z) for lab, v in ll.items()}}
    sc = body.score(q)
    assert len(sc) == len(q.cands)
    return {"id": q.id, "after_session": after_session, "scores": {k: float(s) for (k, _), s in zip(q.cands, sc)}}


def sessions_to_answer(when, last=15):
    """the sessions after which an item is answered: "every" -> all, "after_session_k" -> k, "end" -> the last."""
    if when == "every":
        return list(range(last + 1))
    if when.startswith("after_session_"):
        return [int(when.rsplit("_", 1)[1])]
    return [last]


# ---------- the rebuild (format check: nothing lost, nothing added) ----------
def rebuild_exposure(w):
    out, cur = [], None
    for c in w.sessions:
        for (kind_, toks, _), (s, st, mo, part, extra) in zip(w.categories[c]["heard"],
                                                               [m for m in w.moment_of if m[0] == c - 1]):
            if part == "sentence":
                cur = {"session": s, "story": st, "moment": mo, "sentence": [w.word_of[t] for t in toks],
                       "scene": {"moment": extra, "lines": []}}
                out.append(cur)
            else:
                cur["scene"]["lines"].append({"id": w.id_of[toks[0]], "cells": [list(w.cell_of[t]) for t in toks[1:]]})
    return out


def rebuild_item(w, q):
    def scene(ms):
        return {"lines": [{"id": w.id_of[m[0]], "cells": [list(w.cell_of[t]) for t in m[1:]]} for m in ms]}
    ctx = {}
    parts = [p for p, _, _ in q.moments]
    if "scene" in parts:
        ctx["scene"] = scene([m for p, _, m in q.moments if p == "scene"])
    if "moment sentence" in parts:
        n = 1 + max(k for p, k, _ in q.moments if p == "moment sentence")
        ctx["moments"] = [{"sentence": [w.word_of[t] for p, kk, m in q.moments if p == "moment sentence" and kk == k
                                        for t in m],
                           "scene": scene([m for p, kk, m in q.moments if p == "moment line" and kk == k])}
                          for k in range(n)]
    if "sentence" in parts:
        ctx["sentences"] = [[w.word_of[t] for t in m] for p, _, m in q.moments if p == "sentence"]
    if "scene after" in parts:
        ctx["scene_after"] = scene([m for p, _, m in q.moments if p == "scene after"])
    question = {"slot": q.slot}
    if q.has_sentence:
        s = [w.word_of[t] for t in q.question]
        if q.blank_at is not None:
            s.insert(q.blank_at, "___")
        question["sentence"] = s
    if q.target is not None:
        question["target"] = w.id_of[q.target]
    it = {"id": q.id, "when": q.when, "format": q.format, "context": ctx, "question": question}
    if q.format in ("choice", "pair"):
        it["candidates"] = [k for k, _ in q.cands]
        it["options"] = [{"label": lab, "scene": {"lines": [{"id": w.id_of[t], "cells": [list(w.cell_of[a])]}
                                                            for t in q.option_order[lab]
                                                            for a in [dict(states)[t]]]}}
                         for lab, states in q.options]
    else:
        cs = []
        for k, tok in q.cands:
            if isinstance(tok, tuple):
                cs.append(int(k))
            elif tok in w.cell_of:
                cs.append(list(w.cell_of[tok]))
            elif tok in w.id_of:
                cs.append(w.id_of[tok])
            else:
                cs.append(w.word_of[tok])
        it["candidates"] = cs
    return it


def digest(obj):
    """a stable digest of a converted life (for the twice-the-same check across processes)."""
    w, qs = obj
    h = hashlib.sha256()
    h.update(w.vecs.tobytes())
    h.update(json.dumps([w.names, w.origin, w.cats.tolist(), w.channel_of.tolist(), w.kind_of.tolist(),
                         [[(k, t, a) for k, t, a in c["heard"]] for c in w.categories], w.moment_of],
                        ensure_ascii=False).encode())
    for q in qs:
        h.update(json.dumps([q.id, q.when, q.format, q.slot, [(p, k, m) for p, k, m in q.moments], q.question, q.blank_at,
                             q.target, [(k, list(t) if isinstance(t, tuple) else t) for k, t in q.cands],
                             [(lab, [(t, a) for t, a in st]) for lab, st in q.options]], ensure_ascii=False).encode())
    return h.hexdigest()
