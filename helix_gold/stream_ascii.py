"""Stream form of Voyager's glyph-picture worlds (2026-10-13; asked by the Pioneer session for its transformer redesign --
a "stream body" that takes life as a time-ordered flow and learns from a window of recent moments; user decision 10-13
"응, 지금 해줘").

The SAME world (data_ascii.build: elements, rules, rooms, test items unchanged); only the way the heard moments are
presented differs:
  * the life order of run_long (phases, rooms permuted with default_rng(70_000 + seed), PER_SESSION rooms a session);
  * inside a room the heard moments are put in TIME ORDER: every object has a cell; a moment that moves an object starts
    where the object is (its result is where the next moment of that object starts); when the order cannot chain, a
    SILENT PLACEMENT is inserted first: no sentence, only the scene changes (the thing was moved while nobody spoke) --
    so the heard words are exactly the world's; counted and reported (the world's motion moments start at random cells,
    so a share of them cannot chain);
  * multi-moment stories ("go"-joined) are cut into their clauses, one moment each, the question part last;
  * every moment carries the scene AFTER it: one line per object of the room {"id": name, "cell": [x, y]} (the shared
    exam's exposure form);
  * boundaries: session, room, story (an id shared by the clauses of one story).
Every heard moment of the world appears exactly once (whole, or all its clauses in order) -- machine-checked by
check(). Naming moments say where a thing is: in the stream they put it there (it is shown at its place). Moments whose
meaning is not an object's move in this version (noise, the unseen pair's positions heard apart, and the special stories
of objs / grav / tale / contain / count / sp / views / numline / goal / self worlds) are "static": sentence + the
unchanged scene, flagged "static" (the coverage per world is reported). Upside-down rooms move every verb the other way.

    python3 stream_ascii.py <world name, e.g. ascii-multi> <seed> <out.jsonl>     (write one life)
    python3 stream_ascii.py check                                                  (machine check, 15+ worlds x seeds)
"""
import json
import sys

import numpy as np

import data_ascii as D
from run_schema import PER_SESSION

MOVE_KINDS = ("R0", "R1", "R2", "R3", "R4", "once")
SPLIT = ("multi-heard", "when-heard")
STORY = "story-heard"


def world_kw(name):
    """the build options of run_long / integrated world names (ascii-<x>)."""
    x = name[len("ascii"):].lstrip("-")
    table = {"": {}, "kor": dict(korean=True), "multi": dict(multi=True), "objs": dict(props=True), "grav": dict(grav=True),
             "tale": dict(tale=True), "frame": dict(frame=True), "contain": dict(contain=True), "count": dict(count=True),
             "succ": dict(succ=True), "numrow": dict(numline="row"), "numown": dict(numline="own"), "views": dict(views=True),
             "viewpt": dict(views=2), "goal": dict(goal=True), "self": dict(self_=True), "nw": dict(nw=True),
             "hold": dict(hold=True), "sp": dict(sp=True), "spshuf": dict(sp_shuffle=True), "when": dict(when=True),
             "whenshuf": dict(when_shuffle=True), "walk": dict(walk=True)}
    return table[x]


def move(p, k, flipped):
    d = D.VERBS[k][1]
    r = 2 if flipped is True else 0 if flipped is False else int(flipped)
    for _ in range(r):
        d = (-d[1], d[0])
    return (p[0] + d[0], p[1] + d[1])


def build_traced(name, seed):
    D.TRACE = []
    try:
        w = D.build(seed, **world_kw(name))
        trace = D.TRACE
    finally:
        D.TRACE = None
    return w, trace


def room_events(w, c, cat, trace_c, kw):
    """the room's heard moments as events: (heard index, kind, sentence, list of parts); a part = (pre, post, sentence)
    with pre / post {object: cell} (pre = what must hold before, post = what holds after) -- static parts have neither."""
    objs = cat["objects"]
    obj_of = {int(t): j for j, o in enumerate(objs) for t in o}
    P = getattr(w, "particle_of", {})
    GA, ESEO = P.get("GA"), P.get("ESEO")
    GO = int(w.go) if getattr(w, "go", None) is not None else None   # (the @multi branch does not rebuild particle_of after GO)
    home = {}
    for _, _, kind, core in trace_c.values():
        if kind.startswith("name"):
            home[obj_of[core[0]]] = w.pos_of[core[2]]
    flipped = cat["flipped"]
    grav = bool(kw.get("grav") or kw.get("tale"))
    events = []
    for h, (kind, s, ans) in enumerate(cat["heard"]):
        s = [int(x) for x in s]
        tr = trace_c.get(h)
        if tr is not None and kind.startswith("name"):
            j = obj_of[tr[3][0]]
            events.append((h, kind, [({}, {j: w.pos_of[tr[3][2]]}, s)]))
        elif tr is not None and kind in MOVE_KINDS:
            j = obj_of[tr[3][0]]
            events.append((h, kind, [({j: w.pos_of[tr[3][2]]}, {j: w.pos_of[tr[3][3]]}, s)]))
        elif kind == STORY and GA is not None:
            mover, lm = s[s.index(GA) - 1], s[s.index(ESEO) - 1]
            fr = [x for x in s if w.role_of.get(x) == "from"][0]
            to = [x for x in s if w.role_of.get(x) == "to"][0]
            m, l_ = obj_of[mover], obj_of[lm]
            events.append((h, kind, [({m: w.pos_of[fr], l_: home[l_]}, {m: w.pos_of[to]}, s)]))
        elif kind in SPLIT and GO is not None and not grav:
            segs, cur = [], []
            for x in s:
                if x == GO:
                    segs.append(cur)
                    cur = []
                else:
                    cur.append(x)
            parts, where = [], {}
            for cl in segs:
                m = obj_of[cl[cl.index(GA) - 1]]
                pre = {}
                if ESEO in cl:
                    l_ = obj_of[cl[cl.index(ESEO) - 1]]
                    pre = {m: home[l_], l_: home[l_]}
                    where[m] = home[l_]
                v = [x for x in cl if x in w.verb_of][0]
                where[m] = move(where[m], w.verb_of[v], flipped)
                parts.append((pre, {m: where[m]}, cl))
            parts.append(({}, {}, cur))                 # the question (and its heard answer) after the story
            events.append((h, kind, parts))
        else:
            events.append((h, kind, [({}, {}, s)]))
    return events, home, objs


def life_stream(name, seed):
    """the whole life as a list of moment dicts (token ids; see write() for names)."""
    w, trace = build_traced(name, seed)
    kw = world_kw(name)
    by_room = {}
    for c, h, kind, core in trace:
        by_room.setdefault(c, {})[h] = (c, h, kind, core)
    orng = np.random.default_rng(70_000 + seed)               # run_long's life order
    order = [int(c) for phase in w.phases for c in orng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    rng = np.random.default_rng(seed * 1319 + 97_000_241)
    out, n_place, story_id = [], 0, 0
    for si, ses in enumerate(sessions):
        for ci, c in enumerate(ses):
            cat = w.categories[c]
            first_room = True
            if c == 0:                                         # the textbook: formulas, no objects
                for h, (kind, s, _) in enumerate(cat["heard"]):
                    out.append(dict(session=si, room=c, room_start=first_room, session_start=(ci == 0 and first_room),
                                    src=h, part=None, kind=kind, event="heard", static=True, story=None,
                                    sentence=[int(x) for x in s], scene={}))
                    first_room = False
                continue
            events, home, objs = room_events(w, c, cat, by_room.get(c, {}), kw)
            state = dict(home)                                 # things start where they are named
            if "order" in cat:                                 # "@walk" (B form): the world's own time order
                by_h = {ev[0]: ev for ev in events}
                pending = [by_h[h] for h in cat["order"]]
            else:
                pending = [events[int(i)] for i in rng.permutation(len(events))]
            in_order = "order" in cat

            def ready(ev):
                return all(state.get(o) == p for o, p in ev[2][0][0].items())

            def emit(**kw_):
                nonlocal first_room
                out.append(dict(session=si, room=c, room_start=first_room, session_start=(ci == 0 and first_room),
                                scene=dict(state), **kw_))
                first_room = False
            while pending:
                # 1) a moving moment that can start now; 2) a naming moment that puts a thing where one of its waiting
                # moves starts; 3) any other moment that needs nothing (spread through the room); 4) a moving moment
                # after a silent placement
                movers = [i for i, ev in enumerate(pending) if ev[2][0][0] and ready(ev)]
                starts = {(o, p) for ev in pending if ev[2][0][0] for o, p in ev[2][0][0].items()}
                enable = [i for i, ev in enumerate(pending) if not ev[2][0][0] and ev[2][0][1] and
                          any(x in starts and state.get(x[0]) != x[1] for x in ev[2][0][1].items())]
                free = [i for i, ev in enumerate(pending) if not ev[2][0][0]]
                n_move = len(pending) - len(free)
                if in_order:
                    i = 0
                elif movers:
                    i = movers[int(rng.integers(len(movers)))]
                elif enable:
                    i = enable[int(rng.integers(len(enable)))]
                elif free and rng.random() < len(free) / len(pending):
                    i = free[int(rng.integers(len(free)))]
                else:
                    i = [j for j, ev in enumerate(pending) if ev[2][0][0]][0] if n_move else 0
                h, kind, parts = pending.pop(i)
                sid = None
                if len(parts) > 1:
                    story_id += 1
                    sid = story_id
                for pi, (pre, post, sent) in enumerate(parts):
                    for o, p in pre.items():                  # placements so that the moment can start
                        if state.get(o) != p:
                            state[o] = p
                            n_place += 1
                            emit(src=None, part=None, kind="place", event="place", static=False, story=None,
                                 sentence=[])
                    state.update(post)
                    emit(src=h, part=([pi, len(parts)] if len(parts) > 1 else None), kind=kind, event="heard",
                         static=not pre and not post, story=sid, sentence=[int(x) for x in sent])
    return w, out, n_place


def write(name, seed, path):
    w, out, n_place = life_stream(name, seed)
    with open(path, "w", encoding="utf-8") as f:
        for t, m in enumerate(out):
            objs = w.categories[m["room"]].get("objects", [])
            row = {"t": t, "session": m["session"], "room": m["room"], "session_start": m["session_start"],
                   "room_start": m["room_start"], "story": m["story"], "event": m["event"], "kind": m["kind"],
                   "static": m["static"], "src": m["src"], "part": m["part"],
                   "sentence": [w.names[x] for x in m["sentence"]],
                   "scene": [{"id": w.names[int(objs[o][0])], "cell": list(p)} for o, p in sorted(m["scene"].items())]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name} seed {seed}: {len(out)} moments, {n_place} placements -> {path}")


def check(names, seeds):
    """every heard moment exactly once (whole or all parts in order); every moving moment starts where its things are and
    leaves them where it says; static moments change nothing; coverage and placements per world."""
    for name in names:
        tot = {"moments": 0, "place": 0, "static": 0, "heard": 0, "bad": 0}
        for seed in seeds:
            w, out, n_place = life_stream(name, seed)
            seen = {}
            prev_room, scene = None, None
            last_place = False
            for m in out:
                if m["event"] == "heard" and not m["static"] and m["kind"] not in ("place",) and not m["kind"].startswith("name"):
                    tot["moving"] = tot.get("moving", 0) + 1
                    tot["chained"] = tot.get("chained", 0) + (not last_place)
                last_place = m["event"] == "place"
                if m["room"] != prev_room:
                    prev_room, scene = m["room"], None
                if m["event"] == "heard":
                    seen.setdefault((m["room"], m["src"]), []).append(m["part"])
                tot["moments"] += 1
                tot["place"] += m["event"] == "place"
                tot["static"] += bool(m["static"])
                if scene is not None and m["static"] and m["scene"] != scene:
                    tot["bad"] += 1
                scene = m["scene"]
            go = getattr(w, "go", None)
            grav_w = bool(world_kw(name).get("grav") or world_kw(name).get("tale"))
            for m in out:
                if m["event"] == "heard" and m["kind"] in SPLIT and m["part"] is None and not grav_w and go is not None:
                    tot["bad"] += 1                       # a multi-moment story left whole
            for c, cat in enumerate(w.categories):
                for h in range(len(cat["heard"])):
                    parts = seen.get((c, h))
                    tot["heard"] += 1
                    if parts is None or not (parts == [None] or parts == [[i, len(parts)] for i in range(len(parts))]):
                        tot["bad"] += 1
            if sum(len(v) if v != [None] else 1 for v in seen.values()) < len(seen):
                tot["bad"] += 1
        print(f"{name:14} seeds {seeds}: moments {tot['moments']}, heard {tot['heard']}, placements {tot['place']} "
              f"({tot['place'] / max(tot['heard'], 1):.1%} of heard), static {tot['static'] / tot['moments']:.1%}, "
              f"moving moments that start where the thing already was {tot.get('chained', 0) / max(tot.get('moving', 1), 1):.1%}, "
              f"problems {tot['bad']}")


WALK_STORIES = {"same": "AA", "two": "AB", "three": "ABA"}


def walk_multi(seed, n_heard=2, n_test=D.TEST):
    """v2 (2026-10-13, the Pioneer session's first wish): the walking world with multi-moment stories, stream-native.
    The elements are the "@multi" world's (data_ascii.build(seed, multi=True)); the life order and the naming moments are
    the world's; after the naming every room WALKS (each move starts where the thing is): D.WALK_PER_VERB motion moments a
    word ("[picture] VERB from(a) to(b)" + background pictures, as v1) mixed with STORIES told as the walk goes on --
    kinds as the world's multi stories, "same" (A moves twice), "two" (A, then B), "three" (A, B, A) -- each clause
    "[name ga] VERB" with NO landmark and NO cell: where the thing starts is only in the scenes BEFORE the story (the
    window's job). A heard story: its clauses each with the scene after, then "[A ga] EODI to(x)" (the answer heard). A
    test story (PROBE): its clauses told WITHOUT scenes (heard, not seen), then "[A ga] EODI" without the answer and without
    a scene; the key (A's cell) goes to the probe list; the next moment's scene shows the world again. n_heard heard and
    n_test test stories per kind and room. Returns (w, moments, probes)."""
    w, trace = build_traced("ascii-multi", seed)
    P = w.particle_of
    GA, GO, EODI = P["GA"], int(w.go), int(w.eodi)
    orng = np.random.default_rng(70_000 + seed)
    order = [int(c) for phase in w.phases for c in orng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    rng = np.random.default_rng(seed * 1361 + 107_000_253)
    by_room = {}
    for c, h, kind, core in trace:
        by_room.setdefault(c, []).append((h, kind, core))
    out, probes, story_id = [], [], 0
    for si, ses in enumerate(sessions):
        for ci, c in enumerate(ses):
            cat = w.categories[c]
            first = [True]

            def emit(state, **kw_):
                out.append(dict(session=si, room=c, room_start=first[0], session_start=(ci == 0 and first[0]),
                                scene=None if state is None else dict(state), **kw_))
                first[0] = False
            if c == 0:
                for h, (kind, s_, _) in enumerate(cat["heard"]):
                    emit({}, src=h, kind=kind, event="heard", story=None, sentence=[int(x) for x in s_])
                continue
            objs, flipped = cat["objects"], cat["flipped"]
            obj_of = {int(t): j for j, o in enumerate(objs) for t in o}
            pics = [o[1] for o in objs] + [o[2] for o in objs]
            state = {}
            naming = [(h, core) for h, kind, core in by_room.get(c, []) if kind.startswith("name")]
            for h, core in naming:                                  # the world's naming moments: things where they are named
                state[obj_of[core[0]]] = w.pos_of[core[2]]
            for h, core in naming:
                emit(state, src=h, kind=cat["heard"][h][0], event="heard", story=None,
                     sentence=[int(x) for x in cat["heard"][h][1]])
            noise = [h for h, x in enumerate(cat["heard"]) if x[0] == "noise"]
            if cat["placement"]:
                for h in noise:
                    emit(state, src=h, kind="noise", event="heard", story=None, sentence=[int(x) for x in cat["heard"][h][1]])
                continue
            ks = [k for k in range(D.N_KINDS) if any(k == w.verb_of[v] for v in w.verbs)
                  and (k < D.NEW_KIND or cat["phase"] >= 1) and k != cat.get("held")]
            plan = ["move"] * (D.WALK_PER_VERB * len(ks)) + [("story", kd, b) for kd in WALK_STORIES
                                                             for b, n in (("heard", n_heard), ("test", n_test))
                                                             for _ in range(n)] + [("noise", h) for h in noise]
            plan = [plan[int(i)] for i in rng.permutation(len(plan))]

            def step(j):
                for _ in range(100):
                    k = int(ks[int(rng.integers(len(ks)))])
                    nxt = move(state[j], k, flipped)
                    if max(abs(nxt[0]), abs(nxt[1])) <= D.HALF:
                        return k, nxt
                return None
            for item in plan:
                if item == "move":
                    j = int(rng.integers(len(objs)))
                    r = step(j)
                    if r is None:
                        continue
                    k, nxt = r
                    a = state[j]
                    pool = [q for q in pics if q not in objs[j]]
                    fill = [int(x) for x in rng.choice(pool, int(rng.integers(D.FILL[0], D.FILL[1] + 1)), replace=False)]
                    s_ = [int(objs[j][1]), int(w.verbs[k]), int(w.pos_from[a]), int(w.pos_to[nxt])] + fill
                    state[j] = nxt
                    emit(state, src=None, kind=f"walk R{k}", event="heard", story=None,
                         sentence=[s_[i] for i in rng.permutation(len(s_))])
                elif item[0] == "noise":
                    emit(state, src=item[1], kind="noise", event="heard", story=None,
                         sentence=[int(x) for x in cat["heard"][item[1]][1]])
                else:
                    _, kd, bucket = item
                    for _try in range(20):                          # a story whose moves all stay on the grid
                        a_, b_ = [int(x) for x in rng.permutation(len(objs))[:2]]
                        who = {"A": a_, "B": b_}
                        sim, clauses = dict(state), []
                        for m in WALK_STORIES[kd]:
                            j = who[m]
                            ks_ok = [k for k in ks if max(abs(v) for v in move(sim[j], k, flipped)) <= D.HALF]
                            if not ks_ok:
                                break
                            k = int(ks_ok[int(rng.integers(len(ks_ok)))])
                            sim[j] = move(sim[j], k, flipped)
                            clauses.append((j, k, dict(sim)))
                        if len(clauses) == len(WALK_STORIES[kd]):
                            break
                    else:
                        continue
                    story_id += 1
                    seen = bucket == "heard"
                    for j, k, after in clauses:
                        state = after
                        emit(state if seen else None, src=None, kind=f"walk story {kd}", event="heard", story=story_id,
                             sentence=[int(objs[j][0]), GA, int(w.verbs[k])])
                    ans = int(w.pos_to[state[a_]])
                    q = [int(objs[a_][0]), GA, EODI]
                    if seen:
                        emit(state, src=None, kind=f"walk question {kd}", event="heard", story=story_id, sentence=q + [ans])
                    else:
                        emit(None, src=None, kind=f"walk probe {kd}", event="probe", story=story_id, sentence=q)
                        probes.append({"t": len(out) - 1, "kind": f"multi {kd}", "answer": ans,
                                       "candidates": "every 'to' cell", "story": story_id})
    return w, out, probes


def walk_contain(seed, n_heard=2, n_test=D.TEST):
    """v3 (2026-10-13, the Pioneer session's second wish): the walking world with CONTAINMENT, stream-native. Elements are
    the "@contain" world's (ANE inside, YEOPE beside). After the naming the room walks as v1 / v2 (D.WALK_PER_VERB motion
    moments a word, each starting where the thing is); mixed in:
      put in      "[A ga] [C ane]"   -- A goes to C's cell and is INSIDE C: from then on A is NOT in the scenes (hidden);
      put beside  "[A ga] [C yeope]" -- A goes to a cell next to C (shown in the scene after);
      a container that moves carries what is inside it (unseen); a thing inside that moves by itself comes out (seen again);
      stories "in" (A inside C; C moves once or twice; "[A ga] EODI") and "by" (A beside C; C moves; "[A ga] EODI"),
      the setting (the put) told and shown just before the story, the story clauses "[C ga] VERB" with no cell.
    Heard stories: clauses with scenes, then the question with A's cell. Test stories (PROBES): clauses without scenes,
    then the question without answer; keys in the probe list. Only things that hold nothing are put; nothing is put into
    a thing that is itself inside another. Returns (w, moments, probes)."""
    w, trace = build_traced("ascii-contain", seed)
    P = w.particle_of
    GA, ANE, YEOPE = P["GA"], P["ANE"], P["YEOPE"]
    EODI = int(w.eodi)
    orng = np.random.default_rng(70_000 + seed)
    order = [int(c) for phase in w.phases for c in orng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    rng = np.random.default_rng(seed * 1373 + 109_000_259)
    by_room = {}
    for c, h, kind, core in trace:
        by_room.setdefault(c, []).append((h, kind, core))
    out, probes, story_id = [], [], 0
    on = lambda p: max(abs(p[0]), abs(p[1])) <= D.HALF
    for si, ses in enumerate(sessions):
        for ci, c in enumerate(ses):
            cat = w.categories[c]
            first = [True]
            inside = {}                                        # thing -> its container

            def visible(state):
                return {j: p for j, p in state.items() if j not in inside}

            def emit(state, **kw_):
                out.append(dict(session=si, room=c, room_start=first[0], session_start=(ci == 0 and first[0]),
                                scene=None if state is None else visible(state), **kw_))
                first[0] = False
            if c == 0:
                for h, (kind, s_, _) in enumerate(cat["heard"]):
                    emit({}, src=h, kind=kind, event="heard", story=None, sentence=[int(x) for x in s_])
                continue
            objs, flipped = cat["objects"], cat["flipped"]
            obj_of = {int(t): j for j, o in enumerate(objs) for t in o}
            pics = [o[1] for o in objs] + [o[2] for o in objs]
            state = {}
            naming = [(h, core) for h, kind, core in by_room.get(c, []) if kind.startswith("name")]
            for h, core in naming:
                state[obj_of[core[0]]] = w.pos_of[core[2]]
            for h, core in naming:
                emit(state, src=h, kind=cat["heard"][h][0], event="heard", story=None,
                     sentence=[int(x) for x in cat["heard"][h][1]])
            noise = [h for h, x in enumerate(cat["heard"]) if x[0] == "noise"]
            if cat["placement"]:
                for h in noise:
                    emit(state, src=h, kind="noise", event="heard", story=None, sentence=[int(x) for x in cat["heard"][h][1]])
                continue
            ks = [k for k in range(D.N_KINDS) if (k < D.NEW_KIND or cat["phase"] >= 1) and k != cat.get("held")]
            plan = ["move"] * (D.WALK_PER_VERB * len(ks)) + [("story", kd, b) for kd in ("in", "by")
                                                             for b, n in (("heard", n_heard), ("test", n_test))
                                                             for _ in range(n)] + [("noise", h) for h in noise]
            plan = [plan[int(i)] for i in rng.permutation(len(plan))]

            def go(st, j, k):                                  # j moves by k; what is inside j comes along; j leaves its container
                st = dict(st)
                st[j] = move(st[j], k, flipped)
                for a, cc in inside.items():
                    if cc == j:
                        st[a] = st[j]
                return st
            for item in plan:
                if item == "move":
                    j = int(rng.integers(len(objs)))
                    ok = [k for k in ks if on(move(state[j], k, flipped))]
                    if not ok:
                        continue
                    k = int(ok[int(rng.integers(len(ok)))])
                    a = state[j]
                    inside.pop(j, None)
                    state = go(state, j, k)
                    pool = [q for q in pics if q not in objs[j]]
                    fill = [int(x) for x in rng.choice(pool, int(rng.integers(D.FILL[0], D.FILL[1] + 1)), replace=False)]
                    s_ = [int(objs[j][1]), int(w.verbs[k]), int(w.pos_from[a]), int(w.pos_to[state[j]])] + fill
                    emit(state, src=None, kind=f"walk R{k}", event="heard", story=None,
                         sentence=[s_[i] for i in rng.permutation(len(s_))])
                    continue
                if item[0] == "noise":
                    emit(state, src=item[1], kind="noise", event="heard", story=None,
                         sentence=[int(x) for x in cat["heard"][item[1]][1]])
                    continue
                _, kd, bucket = item
                done = False
                for _try in range(30):
                    a_, c_ = [int(x) for x in rng.permutation(len(objs))[:2]]
                    if a_ in inside.values() or c_ in inside:      # A holds nothing; C is not inside another
                        continue
                    if kd == "by":
                        sides = [move(state[c_], k, flipped) for k in range(4)]
                        sides = [p for p in sides if on(p)]
                        if not sides:
                            continue
                        put_to = sides[int(rng.integers(len(sides)))]
                    n_cl = 1 + int(rng.integers(2))
                    sim, ok_cl = dict(state), []
                    saved = dict(inside)
                    inside.pop(a_, None)
                    if kd == "in":
                        inside[a_] = c_
                        sim[a_] = sim[c_]
                    else:
                        sim[a_] = put_to
                    for _ in range(n_cl):
                        ok = [k for k in ks if on(move(sim[c_], k, flipped))]
                        if not ok:
                            break
                        k = int(ok[int(rng.integers(len(ok)))])
                        sim = go(sim, c_, k)
                        ok_cl.append((k, dict(sim)))
                    if len(ok_cl) != n_cl:
                        inside.clear()
                        inside.update(saved)
                        continue
                    done = True
                    break
                if not done:
                    continue
                story_id += 1
                put_state = dict(state)
                put_state[a_] = sim[a_] if kd == "by" else state[c_]
                if kd == "by":
                    put_state[a_] = put_to
                state = put_state
                emit(state, src=None, kind=f"walk put {kd}", event="heard", story=story_id,
                     sentence=[int(objs[a_][0]), GA, int(objs[c_][0]), ANE if kd == "in" else YEOPE])
                seen = bucket == "heard"
                for k, after in ok_cl:
                    state = after
                    emit(state if seen else None, src=None, kind=f"walk contain story {kd}", event="heard", story=story_id,
                         sentence=[int(objs[c_][0]), GA, int(w.verbs[k])])
                ans = int(w.pos_to[state[a_]])
                q = [int(objs[a_][0]), GA, EODI]
                if seen:
                    emit(state, src=None, kind=f"walk contain question {kd}", event="heard", story=story_id,
                         sentence=q + [ans])
                else:
                    emit(None, src=None, kind=f"walk contain probe {kd}", event="probe", story=story_id, sentence=q)
                    probes.append({"t": len(out) - 1, "kind": f"contain {kd}", "answer": ans,
                                   "candidates": "every 'to' cell", "story": story_id})
    return w, out, probes


def check_walk_contain(seeds):
    """independent replay from the sentences: a put moves A to C's cell (inside, hidden) or next to C (as the scene shows);
    a story clause moves C and what is inside it; a motion moment moves its picture's thing (it comes out of a container)
    and what is inside it; every shown scene = the visible part of the replayed state; hidden things never shown; every
    heard answer and probe key = the asked thing's replayed cell; story clauses carry no cell; plus the no-tracking
    baseline (the asked thing's last SHOWN cell)."""
    from collections import Counter
    for seed in seeds:
        w, out, probes = walk_contain(seed)
        P = w.particle_of
        ANE = P["ANE"]
        keys = {p["t"]: p["answer"] for p in probes}
        bad, room = 0, None
        base_hit, base_n = Counter(), Counter()
        for t, m in enumerate(out):
            cat = w.categories[m["room"]]
            if m["room"] != room:
                room, st, ins, last_seen = m["room"], dict(m["scene"] or {}), {}, dict(m["scene"] or {})
            objs = cat.get("objects", [])
            name_of = {int(o[0]): j for j, o in enumerate(objs)}
            k_ = m["kind"]
            sent = m["sentence"]
            if k_.startswith("name") or m["room"] == 0 or k_ == "noise":
                pass
            elif k_.startswith("walk R"):
                fr = [x for x in sent if w.role_of.get(x) == "from"][0]
                to = [x for x in sent if w.role_of.get(x) == "to"][0]
                cand = [j for j, o in enumerate(objs) if int(o[1]) in sent and st.get(j) == w.pos_of[fr]]
                good = None
                for j in cand:
                    st2, ins2 = dict(st), dict(ins)
                    ins2.pop(j, None)
                    st2[j] = w.pos_of[to]
                    for a, cc in ins2.items():
                        if cc == j:
                            st2[a] = st2[j]
                    if {x: p for x, p in st2.items() if x not in ins2} == m["scene"]:
                        good = (st2, ins2)
                        break
                if good is None:
                    bad += 1
                else:
                    st, ins = good
            elif k_.startswith("walk put"):
                a, cc = name_of[sent[0]], name_of[sent[2]]
                ins.pop(a, None)
                if sent[3] == ANE:
                    ins[a] = cc
                    st[a] = st[cc]
                else:
                    st[a] = tuple(m["scene"][a])
                    bad += max(abs(st[a][0] - st[cc][0]), abs(st[a][1] - st[cc][1])) != 1
            elif k_.startswith("walk contain story"):
                cc = name_of[sent[0]]
                st[cc] = move(st[cc], w.verb_of[sent[2]], cat["flipped"])
                for a, h in ins.items():
                    if h == cc:
                        st[a] = st[cc]
                bad += any(x in w.pos_of for x in sent)
            elif k_.startswith("walk contain question") or m["event"] == "probe":
                a = name_of[sent[0]]
                ans = sent[-1] if m["event"] != "probe" else keys.get(t)
                bad += ans is None or w.pos_of[ans] != st[a]
                if m["event"] == "probe":
                    kd = k_.split()[-1]
                    base_n[kd] += 1
                    base_hit[kd] += last_seen.get(a) == st[a]
            if m["scene"] is not None:
                vis = {x: p for x, p in st.items() if x not in ins}
                bad += m["scene"] != vis
                bad += any(x in m["scene"] for x in ins)
                last_seen.update(m["scene"])
        print(f"ascii-walk-contain seed {seed}: moments {len(out)}, probes {len(probes)} "
              f"({dict(Counter(p['kind'] for p in probes))}), problems {bad}; no-tracking baseline (last shown cell): " +
              ", ".join(f"{k} {base_hit[k] / base_n[k]:.2f}" for k in sorted(base_n)))


PHYS_KINDS = {"grav": ("floor", "stack", "step"), "objs": ("light", "heavy", "free")}


def phys_result(mode, state, j, k, flipped, heavy, objs):
    """the physics of one move of thing j by word k, with every thing of the room present: returns the new state or None
    (not allowed: off the grid, or a push that would leave the grid / run into a third thing)."""
    st = dict(state)
    on = lambda p: max(abs(p[0]), abs(p[1])) <= D.HALF
    occ = lambda p: [o for o, q in st.items() if o != j and q == p]
    nxt = move(st[j], k, flipped)
    if not on(nxt):
        return None
    if mode == "grav" and k == 0:                          # "falls": until the edge below or a thing in the way (support)
        p = st[j]
        while on(move(p, k, flipped)) and not occ(move(p, k, flipped)):
            p = move(p, k, flipped)
        if p == st[j]:
            return None
        st[j] = p
        return st
    hit = occ(nxt)
    if mode == "objs" and hit:
        if len(hit) > 1:
            return None
        b = hit[0]
        if heavy[int(objs[b][0])]:
            return st                                       # a heavy thing stops the mover: both stay
        pb = move(nxt, k, flipped)
        if not on(pb) or occ(pb):
            return None
        st[b], st[j] = pb, nxt                              # a light thing is pushed one cell the same way
        return st
    st[j] = nxt
    return st


def walk_phys(seed, mode, n_heard=2, n_test=D.TEST):
    """v4 (2026-10-13, the Pioneer session's third wish): the walking world with PHYSICS, stream-native -- mode "grav"
    (elements of "@grav": "falls" goes on until the edge below or until a thing is in the way -- support; upside-down rooms
    fall up) or "objs" (elements of "@objs": every thing is HEAVY or LIGHT for the life, as the world's table; a mover that
    runs into a light thing pushes it one cell the same way and takes its cell; a heavy thing stops it, both stay). Every
    move obeys the physics with every thing of the room present. After the naming the room walks (D.WALK_PER_VERB motion
    moments a word that run into nothing, "[picture] VERB from(a) to(b)" + background, b by the physics); mixed in, STORIES
    of one clause "[A ga] VERB" (no cell) whose outcome depends on the scene before -- grav: "floor" (falls to the edge),
    "stack" (falls onto a thing), "step" (another word, one cell); objs: "light" (runs into a light thing; "[B ga] EODI"
    asks the pushed thing), "heavy" (runs into a heavy thing; "[A ga] EODI"), "free" (runs into nothing). When the room
    offers no such situation, a SETTING moment "[A ga] at(p)" (shown) puts A where the story can happen. Heard stories: the
    clause with its scene, then the question with the answer; test stories (PROBES): the clause without a scene, then the
    question without answer (key in the probe list). Returns (w, moments, probes)."""
    w, trace = build_traced("ascii-" + mode, seed)
    GA = w.particle_of["GA"]
    EODI = int(w.eodi)
    heavy = getattr(w, "heavy", {})
    orng = np.random.default_rng(70_000 + seed)
    order = [int(c) for phase in w.phases for c in orng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    rng = np.random.default_rng(seed * 1381 + (113_000_261 if mode == "grav" else 127_000_271))
    by_room = {}
    for c, h, kind, core in trace:
        by_room.setdefault(c, []).append((h, kind, core))
    out, probes, story_id = [], [], 0
    on = lambda p: max(abs(p[0]), abs(p[1])) <= D.HALF
    for si, ses in enumerate(sessions):
        for ci, c in enumerate(ses):
            cat = w.categories[c]
            first = [True]

            def emit(state, **kw_):
                out.append(dict(session=si, room=c, room_start=first[0], session_start=(ci == 0 and first[0]),
                                scene=None if state is None else dict(state), **kw_))
                first[0] = False
            if c == 0:
                for h, (kind, s_, _) in enumerate(cat["heard"]):
                    emit({}, src=h, kind=kind, event="heard", story=None, sentence=[int(x) for x in s_])
                continue
            objs, flipped = cat["objects"], cat["flipped"]
            obj_of = {int(t): j for j, o in enumerate(objs) for t in o}
            pics = [o[1] for o in objs] + [o[2] for o in objs]
            state = {}
            naming = [(h, core) for h, kind, core in by_room.get(c, []) if kind.startswith("name")]
            for h, core in naming:
                state[obj_of[core[0]]] = w.pos_of[core[2]]
            for h, core in naming:
                emit(state, src=h, kind=cat["heard"][h][0], event="heard", story=None,
                     sentence=[int(x) for x in cat["heard"][h][1]])
            noise = [h for h, x in enumerate(cat["heard"]) if x[0] == "noise"]
            if cat["placement"]:
                for h in noise:
                    emit(state, src=h, kind="noise", event="heard", story=None, sentence=[int(x) for x in cat["heard"][h][1]])
                continue
            ks = [k for k in range(D.N_KINDS) if (k < D.NEW_KIND or cat["phase"] >= 1) and k != cat.get("held")]
            plan = ["move"] * (D.WALK_PER_VERB * len(ks)) + [("story", kd, b) for kd in PHYS_KINDS[mode]
                                                             for b, n in (("heard", n_heard), ("test", n_test))
                                                             for _ in range(n)] + [("noise", h) for h in noise]
            plan = [plan[int(i)] for i in rng.permutation(len(plan))]

            def kind_of(st, j, k):
                """the story kind this move would be (and the asked thing), or None."""
                new = phys_result(mode, st, j, k, flipped, heavy, objs)
                if new is None:
                    return None
                if mode == "grav":
                    if k != 0:
                        return ("step", j, new)
                    below = move(new[j], 0, flipped)
                    return ("stack" if on(below) else "floor", j, new)
                nxt = move(st[j], k, flipped)
                hit = [o for o, q in st.items() if o != j and q == nxt]
                if not hit:
                    return ("free", j, new)
                return ("heavy", j, new) if heavy[int(objs[hit[0]][0])] else ("light", hit[0], new)
            for item in plan:
                if item == "move":
                    for _t in range(30):
                        j = int(rng.integers(len(objs)))
                        k = int(ks[int(rng.integers(len(ks)))])
                        new = phys_result(mode, state, j, k, flipped, heavy, objs)
                        if new is None or (mode == "objs" and any(new[o] != state[o] for o in new if o != j)) or \
                                (mode == "objs" and new[j] == state[j]):
                            continue
                        break
                    else:
                        continue
                    a = state[j]
                    state = new
                    pool = [q for q in pics if q not in objs[j]]
                    fill = [int(x) for x in rng.choice(pool, int(rng.integers(D.FILL[0], D.FILL[1] + 1)), replace=False)]
                    s_ = [int(objs[j][1]), int(w.verbs[k]), int(w.pos_from[a]), int(w.pos_to[state[j]])] + fill
                    emit(state, src=None, kind=f"walk R{k}", event="heard", story=None,
                         sentence=[s_[i] for i in rng.permutation(len(s_))])
                    continue
                if item[0] == "noise":
                    emit(state, src=item[1], kind="noise", event="heard", story=None,
                         sentence=[int(x) for x in cat["heard"][item[1]][1]])
                    continue
                _, kd, bucket = item
                found = [(j, k) for j in range(len(objs)) for k in ks
                         if (r := kind_of(state, j, k)) is not None and r[0] == kd]
                setting = None
                if not found:                                  # put a thing where the story can happen (shown)
                    for _t in range(200):
                        j = int(rng.integers(len(objs)))
                        p = (int(rng.integers(-D.HALF, D.HALF + 1)), int(rng.integers(-D.HALF, D.HALF + 1)))
                        if any(q == p for o, q in state.items() if o != j):
                            continue
                        st2 = dict(state)
                        st2[j] = p
                        ok = [k for k in ks if (r := kind_of(st2, j, k)) is not None and r[0] == kd]
                        if ok:
                            setting, state = (j, p), st2
                            found = [(j, k) for k in ok]
                            break
                if not found:
                    continue
                j, k = found[int(rng.integers(len(found)))]
                _, asked, new = kind_of(state, j, k)
                story_id += 1
                if setting is not None:
                    emit(state, src=None, kind=f"walk setting {kd}", event="heard", story=story_id,
                         sentence=[int(objs[setting[0]][0]), GA, int(w.pos_at[setting[1]])])
                state = new
                seen = bucket == "heard"
                emit(state if seen else None, src=None, kind=f"walk {mode} story {kd}", event="heard", story=story_id,
                     sentence=[int(objs[j][0]), GA, int(w.verbs[k])])
                ans = int(w.pos_to[state[asked]])
                q = [int(objs[asked][0]), GA, EODI]
                if seen:
                    emit(state, src=None, kind=f"walk {mode} question {kd}", event="heard", story=story_id, sentence=q + [ans])
                else:
                    emit(None, src=None, kind=f"walk {mode} probe {kd}", event="probe", story=story_id, sentence=q)
                    probes.append({"t": len(out) - 1, "kind": f"{mode} {kd}", "answer": ans,
                                   "candidates": "every 'to' cell", "story": story_id})
    return w, out, probes


def check_walk_phys(mode, seeds):
    """independent replay: a setting puts its thing at its cell; a story clause / a motion moment follows the physics
    (re-implemented here from the rule text: grav "falls" until the edge or a thing below; objs push / stop by the world's
    heavy table); every shown scene = the replayed state; every answer / probe key = the asked thing's replayed cell; story
    clauses carry no cell; no-tracking baseline = the asked thing's cell in the last shown scene."""
    from collections import Counter
    for seed in seeds:
        w, out, probes = walk_phys(seed, mode)
        heavy = getattr(w, "heavy", {})
        keys = {p["t"]: p["answer"] for p in probes}
        bad, room = 0, None
        base_hit, base_n, kinds = Counter(), Counter(), Counter()
        on = lambda p: max(abs(p[0]), abs(p[1])) <= D.HALF
        for t, m in enumerate(out):
            cat = w.categories[m["room"]]
            if m["room"] != room:
                room, st, last = m["room"], dict(m["scene"] or {}), dict(m["scene"] or {})
            objs = cat.get("objects", [])
            name_of = {int(o[0]): j for j, o in enumerate(objs)}
            k_, sent, fl = m["kind"], m["sentence"], cat.get("flipped")

            def physics(st, j, k):
                st = dict(st)
                d = move((0, 0), k, fl)
                if mode == "grav" and k == 0:
                    while on((st[j][0] + d[0], st[j][1] + d[1])) and (st[j][0] + d[0], st[j][1] + d[1]) not in \
                            [q for o, q in st.items() if o != j]:
                        st[j] = (st[j][0] + d[0], st[j][1] + d[1])
                    return st
                nxt = (st[j][0] + d[0], st[j][1] + d[1])
                hit = [o for o, q in st.items() if o != j and q == nxt]
                if mode == "objs" and hit:
                    if heavy[int(objs[hit[0]][0])]:
                        return st
                    st[hit[0]] = (nxt[0] + d[0], nxt[1] + d[1])
                st[j] = nxt
                return st
            if k_.startswith("walk R"):
                fr = [x for x in sent if w.role_of.get(x) == "from"][0]
                v = [x for x in sent if x in w.verb_of][0]
                cand = [j for j, o in enumerate(objs) if int(o[1]) in sent and st.get(j) == w.pos_of[fr]]
                res = [physics(st, j, w.verb_of[v]) for j in cand]
                res = [r for r in res if r == m["scene"]]
                bad += not res
                st = res[0] if res else dict(m["scene"])
            elif k_.startswith("walk setting"):
                st[name_of[sent[0]]] = w.pos_of[sent[2]]
            elif " story " in k_:
                kinds[k_.split()[-1]] += 1
                bad += any(x in w.pos_of for x in sent)
                st = physics(st, name_of[sent[0]], w.verb_of[sent[2]])
            elif " question " in k_ or m["event"] == "probe":
                a = name_of[sent[0]]
                ans = sent[-1] if m["event"] != "probe" else keys.get(t)
                bad += ans is None or w.pos_of[ans] != st[a]
                if m["event"] == "probe":
                    kd = k_.split()[-1]
                    base_n[kd] += 1
                    base_hit[kd] += last.get(a) == st[a]
            if m["scene"] is not None:
                bad += m["scene"] != st
                last = dict(m["scene"])
        print(f"ascii-walk-{mode} seed {seed}: moments {len(out)}, story clauses {dict(kinds)}, probes {len(probes)}, "
              f"problems {bad}; no-tracking baseline (last shown cell): " +
              ", ".join(f"{k} {base_hit[k] / base_n[k]:.2f}" for k in sorted(base_n)))


def write_walk(fn, name, seed, path):
    w, out, probes = fn(seed)
    with open(path, "w", encoding="utf-8") as f:
        for t, m in enumerate(out):
            objs = w.categories[m["room"]].get("objects", [])
            row = {"t": t, "session": m["session"], "room": m["room"], "session_start": m["session_start"],
                   "room_start": m["room_start"], "story": m["story"], "event": m["event"], "kind": m["kind"],
                   "sentence": [w.names[x] for x in m["sentence"]],
                   "scene": None if m["scene"] is None else
                   [{"id": w.names[int(objs[o][0])], "cell": list(p)} for o, p in sorted(m["scene"].items())]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(path.replace(".jsonl", "") + ".probes.jsonl", "w", encoding="utf-8") as f:
        for p_ in probes:
            f.write(json.dumps(dict(p_, answer=w.names[p_["answer"]]), ensure_ascii=False) + "\n")
    print(f"{name} seed {seed}: {len(out)} moments, {len(probes)} probes -> {path}")


def write_walk_multi(seed, path):
    w, out, probes = walk_multi(seed)
    with open(path, "w", encoding="utf-8") as f:
        prev_objs = None
        for t, m in enumerate(out):
            objs = w.categories[m["room"]].get("objects", [])
            row = {"t": t, "session": m["session"], "room": m["room"], "session_start": m["session_start"],
                   "room_start": m["room_start"], "story": m["story"], "event": m["event"], "kind": m["kind"],
                   "sentence": [w.names[x] for x in m["sentence"]],
                   "scene": None if m["scene"] is None else
                   [{"id": w.names[int(objs[o][0])], "cell": list(p)} for o, p in sorted(m["scene"].items())]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(path.replace(".jsonl", "") + ".probes.jsonl", "w", encoding="utf-8") as f:
        for p_ in probes:
            f.write(json.dumps(dict(p_, answer=w.names[p_["answer"]]), ensure_ascii=False) + "\n")
    print(f"ascii-walk-multi seed {seed}: {len(out)} moments, {len(probes)} probes -> {path}")


def check_walk_multi(seeds):
    """replay: every moving moment / story clause moves exactly its thing by its word (upside-down rooms reversed); every
    heard answer and every probe key = the asked thing's cell; probes and test-story clauses carry no scene; a story's
    start is never said in it (no position token in a story clause)."""
    for seed in seeds:
        w, out, probes = walk_multi(seed)
        bad, last, n_story = 0, None, 0
        keys = {p["t"]: p["answer"] for p in probes}
        hidden = None
        for t, m in enumerate(out):
            cat = w.categories[m["room"]]
            if m["room_start"]:
                last, hidden = (m["scene"] or {}), None
            k_ = m["kind"]
            if k_.startswith("walk R") or k_.startswith("walk story"):
                base = hidden if hidden is not None else last
                objs = cat["objects"]
                if k_.startswith("walk R"):
                    pic = [x for x in m["sentence"] if x in (int(o[1]) for o in objs)]
                    fr = [x for x in m["sentence"] if w.role_of.get(x) == "from"][0]
                    to = [x for x in m["sentence"] if w.role_of.get(x) == "to"][0]
                    v = [x for x in m["sentence"] if x in w.verb_of][0]
                    j = [jj for jj, o in enumerate(objs) if int(o[1]) in pic and base.get(jj) == w.pos_of[fr]]
                    ok = move(w.pos_of[fr], w.verb_of[v], cat["flipped"]) == w.pos_of[to] and \
                        any(m["scene"] == {**base, jj: w.pos_of[to]} for jj in j)   # (a background thing may share the cell)
                    bad += not ok
                else:
                    n_story += 1
                    j = [jj for jj, o in enumerate(objs) if int(o[0]) == m["sentence"][0]][0]
                    v = m["sentence"][2]
                    new = dict(base)
                    new[j] = move(base[j], w.verb_of[v], cat["flipped"])
                    bad += any(x in w.pos_of for x in m["sentence"])
                    if m["scene"] is None:
                        hidden = new
                    else:
                        bad += m["scene"] != new
                        hidden = None
                    if m["scene"] is None:
                        continue
            if m["event"] == "probe":
                j = [jj for jj, o in enumerate(cat["objects"]) if int(o[0]) == m["sentence"][0]][0]
                bad += keys.get(t) is None or w.pos_of[keys[t]] != hidden[j] or m["scene"] is not None
                last, hidden = hidden, None
                continue
            if k_.startswith("walk question"):
                j = [jj for jj, o in enumerate(cat["objects"]) if int(o[0]) == m["sentence"][0]][0]
                bad += w.pos_of[m["sentence"][-1]] != m["scene"][j]
            if m["scene"] is not None:
                last = m["scene"]
        print(f"ascii-walk-multi seed {seed}: moments {len(out)}, story clauses {n_story}, probes {len(probes)}, problems {bad}")


if __name__ == "__main__":
    if sys.argv[1] in ("walk-grav", "walk-objs"):
        md = sys.argv[1][5:]
        write_walk(lambda sd: walk_phys(sd, md), "ascii-walk-" + md, int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] in ("check-walk-grav", "check-walk-objs"):
        check_walk_phys(sys.argv[1][11:], [0, 3, 5])
    elif sys.argv[1] == "walk-contain":
        write_walk(walk_contain, "ascii-walk-contain", int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "check-walk-contain":
        check_walk_contain([0, 3, 5])
    elif sys.argv[1] == "walk-multi":
        write_walk_multi(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "check-walk-multi":
        check_walk_multi([0, 3, 5])
    elif sys.argv[1] == "check":
        names = ["ascii", "ascii-kor", "ascii-multi", "ascii-when", "ascii-objs", "ascii-grav", "ascii-frame", "ascii-contain",
                 "ascii-count", "ascii-succ", "ascii-numrow", "ascii-views", "ascii-goal", "ascii-self", "ascii-nw",
                 "ascii-hold", "ascii-sp", "ascii-tale"]
        check(names, [0, 3, 5])
    else:
        write(sys.argv[1], int(sys.argv[2]), sys.argv[3])
