"""B stage: the glyph-picture world model (the user's ASCII-art world, 2026-09-23 / 09-29; design approved 2026-09-30).

The same interface as data_v3 (vecs, cats, categories with heard / test sentences, phases), so Voyager runs on it unchanged.
Every token is an ELEMENT with a RAW_DIM feature vector that Module A memorises first (memory-first rule); a sentence is one
MOMENT -- the tokens that arrived at the same time (binding by time).

Egocentric coding ("na" is the origin, the user's idea): every position is a token pos(x, y) RELATIVE TO "na" on a 5 x 5 grid
(x, y in -2..2). Its features are linear in (x, y) -- the first half of the features carries x, the second half y, each as
0.5 + STEP * x * (a fixed +-1 pattern) + tiny noise -- so a move of one cell is a constant offset in feature space: a motion verb IS a relation template (data_v3's DELTA), which Voyager already knows how to
find and carry over. Relations between two objects are compositions of their relations to "na" (finding 71's chaining) --
so a bag of tokens needs no role binding here (each moment binds ONE object to its positions; the bag-of-words limit,
README "(b)", is left for later).

Pictures: every object has a 5 x 5 glyph sprite. Its picture token's features come from the sprite itself through a fixed
random "retina" (one-hot glyph per cell -> fixed projection -> sigmoid), so sprites that look alike have close features
(within a family objects share an outline and differ in a few cells: the apple / ball kind of discrimination). Two
ORIGINS of a picture (input source tags, memory-first rule 3): "drawn" (the family sprite, iconic) and "converted" (the
object's own feature vector rendered cell by cell into glyphs, literal) -- separate tokens, tagged in w.origin.

Categories:
  0          -- the ELEMENTS / the textbook: "na", the 81 position tokens and the motion verbs; its sentences are the verbs'
                formulas given from outside (the user's rule: formulas are injected and memorised, the learner's part is to
                USE them): (verb, pos(p), pos(p + d_verb)) for a few positions. Presented first.
  1..n_cat-1 -- ROOMS, each with 4 new objects (one per family; name word + drawn picture + converted picture):
    naming moments (picture, name, where it is): "name" units (picture, name) heard HEARD times for two objects, "name-once"
                for the third (fast mapping from one moment, Carey 1978), the fourth is named with its CONVERTED picture;
    motion moments (picture, verb, pos(p), pos(p + d)): template pairs (pos(p), pos(p + d)) of the verb's kind -- "R{k}"
                heard HEARD times, "once" heard in one moment, "unseen" (the two positions heard only apart) -- the same
                test kinds as data_v3, so the long-life measures carry over;
    other objects of the room are the background (fillers).
  The life (shift2, like life32): phase 1 the four basic verbs; phase 2 the NEW verb "rolls" (diagonal) arrives; phase 3
  the rooms are upside down -- every verb moves the other way (the formula in the textbook no longer holds there).
Test items: a unit member hidden among never-heard background combinations (as data_v3).
Representative values (no tuning): positions 5 x 5, SPRITE 5, 8 glyphs, 4 families, STEP 0.2, POS_NOISE 0.005.
Element check first (B0, memory-first rule 2): the first design had a 9 x 9 grid with Gaussian directions (STEP 0.1) --
pictures and words were recalled and recognised 100%, positions only 35-67% (neighbouring cells 0.1 apart, Module A's
recall error ~0.5): the learning test was NOT started. Per-axis +-1 code: 9 x 9 60-98%, 7 x 7 67-100%, 5 x 5 100% (10 seeds,
STEP 0.175 and 0.2) -> the 5 x 5 grid = the coarse rung of the resolution ladder (the user's "coarse first"); finer grids wait
for a finer memory (a structure question, parked).
"""
import numpy as np

RAW_DIM = 40
GLYPHS = ".#o|-^~*"
SPRITE, HALF = 5, 2                              # 5 x 5 sprites; positions x, y in -2..2 (a 5 x 5 grid around "na")
N_FAMILY, OBJ_PER_ROOM = 4, 4
STEP, POS_NOISE = 0.2, 0.005
WHERE_DIMS = 32                                  # = Module A's LATENT (run_seed_transfer), see w.where_dims
ROLES = ["at", "from", "to"]                     # role markers on positions (finding 90: like Korean particles -e / -eseo / -ro)
ROLE_STEP, ROLE_DIMS = 0.3, 8                    # role code in ROLE_DIMS of the where features: 0.5 +- ROLE_STEP
VERBS = [("falls", (0, -1)), ("rises", (0, 1)), ("pushed", (1, 0)), ("pulled", (-1, 0)), ("rolls", (1, -1))]
N_KINDS, NEW_KIND = 5, 4                         # "rolls" is the new kind (phase 2 on)
HEARD, TEST, NOISE, FILL = 6, 3, 0.1, (1, 3)
TEXTBOOK = 4                                     # formula examples per verb in the textbook category
WALK_PER_VERB = HEARD + 1                        # "@walk": moves a verb per room (as the template pair's HEARD + once)
NOVEL_PER = 4                                    # D72 "cat": novel test things per family (pictures only)
N_COMMON, N_FRAME = 8, 6                         # Q14 frame test "@nwf": common things; frame moments per room
D72_MOVES = [(1, 1), (-1, -1), (-1, 1), (2, 0)]
SEQ_TOLD = 4                                     # D80 v2 "@seq2": a walk's last moves told only in words (C8: 2-4 sentences)  # D72 "verb": the four new motion words' moves (no other word has them)
TRACE = None                                     # stream form (stream_ascii.py): when a list, the heard naming / motion
                                                 # moments' cores are recorded here (c, heard index, kind, core); off = nothing


class World:
    """vecs (V x RAW_DIM), cats, n_cat, V, categories, phases; names (token -> text), origin (token -> source tag),
    sprites (picture token -> 5 x 5 glyph array), pos_of (token -> (x, y)), verb_of (token -> verb index)."""


def retina(rng):
    """a fixed random projection from a sprite's one-hot glyph cells to RAW_DIM features (the sense organ)."""
    return rng.normal(0, 1 / np.sqrt(SPRITE * SPRITE), (SPRITE * SPRITE * len(GLYPHS), RAW_DIM))


def see(sprite, P):
    """features of a 5 x 5 glyph sprite through the retina."""
    one_hot = np.zeros((SPRITE * SPRITE, len(GLYPHS)))
    one_hot[np.arange(SPRITE * SPRITE), sprite.ravel()] = 1
    return 1 / (1 + np.exp(-(one_hot.ravel() @ P) * 2))


def render(vec):
    """the converted picture: an object's own features drawn cell by cell (brightness -> glyph)."""
    order = np.array([GLYPHS.index(g) for g in ".-~o*|^#"])          # dark to bright
    lv = np.clip((vec[:SPRITE * SPRITE] * len(GLYPHS)).astype(int), 0, len(GLYPHS) - 1)
    return order[lv].reshape(SPRITE, SPRITE)


def show(sprite):
    return "\n".join("".join(GLYPHS[g] for g in row) for row in sprite)


def _walk_now(moves, j, b, objs, w, obj_at, told_len):
    """D75 "@scene": every room thing's cell right after walk move number told_len (the mover j now at b): each thing
    starts where it is named and is where its last move left it."""
    now = [w.pos_of[obj_at[o[0]]] for o in objs]
    for jj, _, _, bb in moves[:told_len + 1]:
        now[jj] = bb
    return now


def build(seed, n_cat=32, shift2=True, roles=True, curriculum=False, hold=False, korean=False, multi=False,
          props=False, grav=False, frame=False, contain=False, count=False, succ=False, numline=None, views=False,
          goal=False, self_=False, tale=False, nw=False, sp=False, sp_shuffle=False, when=False, when_shuffle=False,
          walk=False, ten=False, ten_shuffle=False, bi=False, bi_shuffle=False, c8=False, c8_shuffle=False,
          d72=None, scene=False, scene_shuffle=False, nwf=False, seq=False, seq_told=0):
    grav = grav or tale                              # (D22 "@tale" = the "@grav" world plus told tales)
    nw = nw or nwf                                   # (Q14 frame test "@nwf": the "@nw" world plus a sentence frame)
    bi = bi or bi_shuffle                            # (W6 "@bi": the "@multi" world plus English-like names and stories;
    multi = multi or bi                              #  edition 2 -- the tracking part earns its trust on heard multi-moment
                                                     #  stories, which the "@kor" base of edition 1 did not have)
    c8 = c8 or c8_shuffle                            # (W7b "@c8": the "@multi" world plus criterion 8's "where before" and
    multi = multi or c8                              #  long stories -- test keys of the first score table)
    ten = ten or ten_shuffle                         # (W5 "@ten": the "@count" world with the numbers 1-10 instead of 1-3)
    count = count or succ or ten
    sp = sp or sp_shuffle                            # (W3 "@sp": the "@contain" world plus the four spatial words)
    when = when or when_shuffle                      # (W4 "@when": the "@multi" world plus "when" questions)
    korean = korean or bool(numline)                 # (step 4+ "@numrow" / "@numown": the number line is told in Korean)
    korean = korean or views                         # (step 4+ "@views": the view labels are particle-like elements)
    korean = korean or goal                          # (D2 "@goal": the question word EODIRO is particle-like)
    korean = korean or self_                         # (D2 "@self": the question word NUGU is particle-like)
    contain = contain or count or sp
    multi = multi or props or grav or contain or when
    hold = hold or frame
    korean = korean or multi
    seq = seq or seq_told > 0                        # (D80 v2 "@seq2": the "@seq" world, a walk's last moves told only)
    scene = scene or seq                             # (D80 "@seq": the "@scene" world plus question moments and tests)
    scene_form = scene or scene_shuffle               # (D75 "@scene": the "@walk" world told as sentence + the scene after;
    walk = walk or scene_form                         #  its own name -- the goal world's code uses a local "scene")
    assert not (walk and korean), "@walk v1: the base motion world only (stories later)"
    rng = np.random.default_rng(seed)
    w = World()
    w.schema = True
    P = retina(np.random.default_rng(20_000 + seed))
    vecs, cats, names, origin = [], [], [], []
    w.sprites, w.pos_of, w.verb_of = {}, {}, {}

    def token(v, c, name, src):
        vecs.append(np.asarray(v, dtype=np.float32))
        cats.append(c)
        names.append(name)
        origin.append(src)
        return len(vecs) - 1

    # category 0: the elements
    # the "where" channel's features: x in the first WHERE_DIMS / 2, y in the next WHERE_DIMS / 2, the rest the centre 0.5
    # (so that a channel memory of LATENT dims can hold them exactly -- finding 90's structure part)
    xy = WHERE_DIMS - ROLE_DIMS if roles else WHERE_DIMS          # with roles the role code lives inside the where features
    su, sv = rng.choice([-1, 1], xy // 2), rng.choice([-1, 1], xy // 2)
    w.where_dims = WHERE_DIMS
    na = token(np.clip(rng.uniform(0.3, 0.7, RAW_DIM), 0, 1), 0, "na", "element")
    # roles (finding 90, 2026-09-30): a bag of words cannot tell whether (0,0) in "falls + (0,0)" is where it starts or where
    # it ends. Languages mark roles by order or by case; Korean by particles (namu-EseO "from the tree", ttang-EUro "to the
    # ground"). So a position arrives with its role marker as one element (as an inflected word): at (naming: where it is),
    # from, to. Without roles (roles=False) every position is "at" (the first B-stage world).
    # role codes: rows of an 8 x 8 Hadamard matrix (any two differ in exactly half the features -- random +-1 rows could be
    # nearly equal: in the first try "at" and "to" were confused in one seed of ten), columns' signs random per seed
    H = np.array([[1]])
    while len(H) < ROLE_DIMS:
        H = np.block([[H, H], [H, -H]])
    flip = rng.choice([-1, 1], ROLE_DIMS)
    role_code = {r: 0.5 + ROLE_STEP * H[i + 1] * flip for i, r in enumerate(ROLES)}
    w.role_of = {}
    pos = {}
    for r in (ROLES if roles else ["at"]):
        for x in range(-HALF, HALF + 1):
            for y in range(-HALF, HALF + 1):
                v = np.full(RAW_DIM, 0.5)
                v[:xy] = np.concatenate([0.5 + STEP * x * su, 0.5 + STEP * y * sv]) + rng.normal(0, POS_NOISE, xy)
                if roles:
                    v[xy:WHERE_DIMS] = role_code[r]
                v = np.clip(v, 0, 1)
                pos[(r, (x, y))] = token(v, 0, f"{r}({x},{y})", "element")
                w.pos_of[pos[(r, (x, y))]] = (x, y)
                w.role_of[pos[(r, (x, y))]] = r
    R_AT, R_FROM, R_TO = ("at", "from", "to") if roles else ("at", "at", "at")
    w.pos_at = {p: pos[(R_AT, p)] for (r, p) in pos if r == R_AT}
    w.pos_from = {p: pos[(R_FROM, p)] for (r, p) in pos if r == R_FROM}
    w.pos_to = {p: pos[(R_TO, p)] for (r, p) in pos if r == R_TO}
    verbs = []
    for k, (vn, _) in enumerate(VERBS):
        verbs.append(token(np.clip(rng.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, vn, "element"))
        w.verb_of[verbs[-1]] = k
    w.na, w.pos, w.verbs = na, pos, verbs

    def inside(p):
        return max(abs(p[0]), abs(p[1])) <= HALF

    def move(p, k, flipped):
        # flipped: False / True (the upside-down rooms: half a turn) or, in the "@frame" world (stage-1 step 4+, reference
        # frames, 2026-10-08), the room's frame as quarter turns 1 / 2 / 3 -- the ground on a side wall, the ceiling, the
        # other side wall; every motion word turns with it
        d = VERBS[k][1]
        r = 2 if flipped is True else 0 if flipped is False else int(flipped)
        for _ in range(r):
            d = (-d[1], d[0])
        return (p[0] + d[0], p[1] + d[1])

    def starts(k, flipped, n):
        """n distinct starting cells from which verb k stays on the grid."""
        cells = [p for p in w.pos_at if inside(move(p, k, flipped))]
        return [cells[i] for i in rng.choice(len(cells), n, replace=False)]

    textbook = []
    for k in range(N_KINDS):
        for p in starts(k, False, TEXTBOOK):
            q = move(p, k, False)
            textbook.append(("formula", [verbs[k], w.pos_from[p], w.pos_to[q]], [w.pos_to[q]]))
    w.categories = [{"heard": textbook, "test": [], "second": False, "words": [na] + list(pos.values()) + verbs,
                     "room": None}]

    # families: a shared outline; objects differ in a few cells
    fam_outline = [rng.choice(len(GLYPHS), (SPRITE, SPRITE), p=[0.5] + [0.5 / 7] * 7) for _ in range(N_FAMILY)]
    fam_centre = rng.uniform(0.3, 0.7, (N_FAMILY, RAW_DIM))
    prev_names = []                                 # (step 4+ "@succ": a fourth thing borrowed from the previous room)
    seq_rooms = {}                                  # (D80 "@seq": each walking room's told walk, filled only with seq)
    for c in range(1, n_cat):
        phase = 0 if c % 2 == 1 else (1 if c % 4 == 2 else 2) if shift2 else 0
        flipped = phase == 2
        if frame and flipped:                       # "@frame": the room's quarter turns (a separate random stream)
            flipped = int(np.random.default_rng(seed * 1087 + c + 15_000_061).integers(1, 4))
        kinds = list(range(NEW_KIND)) + ([NEW_KIND] if phase >= 1 else [])
        objs = []
        for f in range(N_FAMILY):
            sprite = fam_outline[f].copy()
            cells = rng.choice(SPRITE * SPRITE, 4, replace=False)
            sprite.ravel()[cells] = rng.integers(0, len(GLYPHS), 4)
            nv = np.clip(fam_centre[f] + rng.normal(0, 0.15, RAW_DIM), 0, 1)
            name = token(nv, c, f"obj{c}.{f}", "word")
            drawn = token(see(sprite, P), c, f"pic{c}.{f}", "drawn")
            conv_sprite = render(nv)
            conv = token(see(conv_sprite, P), c, f"cpic{c}.{f}", "converted")
            w.sprites[drawn], w.sprites[conv] = sprite, conv_sprite
            objs.append((name, drawn, conv))
        pics = [o[1] for o in objs] + [o[2] for o in objs]
        heard, test = [], []

        def moments(kind, core, answer, n_heard, n_test, bg_pool, seen=None):
            seen = set() if seen is None else seen
            for bucket, n in ((heard, n_heard), (test, n_test)):
                made = 0
                while made < n:
                    k_ = int(rng.integers(FILL[0], FILL[1] + 1))
                    pool = [b for b in bg_pool if not any(b in o and x in o for o in objs for x in core)]
                    fill = tuple(sorted(int(x) for x in rng.choice(pool, min(k_, len(pool)), replace=False)))
                    if fill in seen:
                        continue
                    seen.add(fill)
                    s = list(core) + list(fill)
                    bucket.append((kind, [s[i] for i in rng.permutation(len(s))], list(answer)))
                    if TRACE is not None and bucket is heard:
                        TRACE.append((c, len(heard) - 1, kind, [int(x) for x in core]))
                    made += 1

        # placement rooms (the user's proposal, 2026-09-30: before motion words, practise putting a thing where it belongs --
        # the name, its picture and its place, as nouns come before verbs): every eighth room pair of the first phase has no
        # motion at all. With curriculum=True they come first in the life (w.phases); otherwise they are mixed in.
        placement = phase == 0 and c % 8 in (1, 3)
        # naming moments: (picture, name, where); background = other pictures of the room. "place": where is it? (the place
        # hidden, given the picture and the name -- tested in every room)
        obj_at = {}
        # stage-1 track A W2 (2026-10-13, criterion 2 new words; the "@nw" world): every object is named with its DRAWN picture
        # and heard 1 / 2 / 4 / 8 times (which family gets which count: a separate random stream per room); the names are
        # left out of the noise sentences, so a name is heard exactly that many times. Test list "nw" (NOT in "test"): the
        # name in a DIFFERENT sentence -- another place (never the one it was named at) and fresh background pictures --
        # asking its picture, and the picture asking its name.
        nwrng = np.random.default_rng(seed * 1231 + c + 59_000_191) if nw else None
        expo = [int(x) for x in nwrng.permutation([1, 2, 4, 8])] if nw else None
        for j, (name, drawn, conv) in enumerate(objs):
            p = w.pos_at[tuple(int(v) for v in rng.integers(-HALF, HALF + 1, 2))]
            obj_at[name] = p
            if nw:
                pic, kind_, n_h = drawn, f"name{expo[j]}", expo[j]
            else:
                pic, kind_, n_h = ((drawn, "name", HEARD) if j < 2 else (drawn, "name-once", 1) if j == 2 else
                                   (conv, "name-conv", HEARD))
            seen_fill = set()
            moments(kind_, [pic, name, p], [pic, name], n_h, TEST, pics, seen_fill)
            moments("place", [pic, name, p], [p], 0, TEST, pics, seen_fill)
        nw_items = []
        if nw:
            for j, (name, drawn, conv) in enumerate(objs):
                elsewhere = [int(q_) for q_ in w.pos_at.values() if q_ != obj_at[name]]
                pool = [int(x) for x in pics if x not in (drawn, conv)]
                for _ in range(TEST):
                    q = int(nwrng.choice(elsewhere))
                    fill = [int(x) for x in nwrng.choice(pool, int(nwrng.integers(FILL[0], FILL[1] + 1)), replace=False)]
                    for cue, ans in ((name, drawn), (drawn, name)):
                        s_ = [int(cue), q] + fill
                        nw_items.append((f"nw{expo[j]}", [s_[i] for i in nwrng.permutation(len(s_))], [int(ans)]))
        # world-model stage B (2026-10-07, hold=True; the "@hold" world): in every upside-down room one motion word is
        # never heard (a separate random stream); the room's chain questions all use it -- the learner must carry the
        # room's rule change, learned from its OTHER motion words, over to a word it never met there
        held = None
        if hold and flipped and not placement:
            held = int(np.random.default_rng(seed * 1013 + c + 4_000_007).choice(kinds))
        # "@walk" (B form for the stream body, 2026-10-13, Pioneer's request, user decision "둘 다 진행해줘"): instead of
        # template moments from random start cells, the room's things WALK: each move starts where the thing is now (they
        # start where they are named), WALK_PER_VERB moves per motion word, the mover and the word drawn at random among
        # the moves that stay on the grid; the background pictures as in the other moments. The tests come from the walk by
        # the same rules: "R{k}" = the (word, start) pair heard most often (at least twice), "once" = a pair heard exactly
        # once, "unseen" = a word and a start never heard together whose two cells were both heard (from / to) in the
        # room. A separate random stream; cat["order"] keeps the time order (naming first, then the walk; noise spread).
        walk_idx, walk_moves = [], []
        if walk and not placement:
            wkrng = np.random.default_rng(seed * 1321 + c + 101_000_243)
            ks_w = [k_ for k_ in kinds if k_ != held]
            now = {j: w.pos_of[obj_at[o[0]]] for j, o in enumerate(objs)}
            pairs = {}
            for _ in range(WALK_PER_VERB * len(ks_w)):
                for _t in range(100):
                    j = int(wkrng.integers(OBJ_PER_ROOM))
                    k = int(ks_w[int(wkrng.integers(len(ks_w)))])
                    if inside(move(now[j], k, flipped)):
                        break
                a, b = now[j], move(now[j], k, flipped)
                pool = [q for q in pics if q not in objs[j]]
                fill = [int(x) for x in wkrng.choice(pool, int(wkrng.integers(FILL[0], FILL[1] + 1)), replace=False)]
                core = [objs[j][1], verbs[k], w.pos_from[a], w.pos_to[b]]
                s_ = core + fill
                heard.append((f"R{k}", [s_[i] for i in wkrng.permutation(len(s_))], [w.pos_to[b]]))
                walk_idx.append(len(heard) - 1)
                if TRACE is not None:
                    TRACE.append((c, len(heard) - 1, f"R{k}", [int(x) for x in core]))
                pairs.setdefault((k, a), []).append(j)
                now[j] = b
                walk_moves.append((j, k, a, b))

            def walk_tests(kind_t, k, a, j):
                b = move(a, k, flipped)
                pool = [q for q in pics if q not in objs[j]]
                for _ in range(TEST):
                    fill = [int(x) for x in wkrng.choice(pool, int(wkrng.integers(FILL[0], FILL[1] + 1)), replace=False)]
                    s_ = [objs[j][1], verbs[k], w.pos_from[a], w.pos_to[b]] + fill
                    test.append((kind_t, [s_[i] for i in wkrng.permutation(len(s_))], [w.pos_to[b]]))
            for k in ks_w:
                mine = {a: js for (k2, a), js in pairs.items() if k2 == k}
                if not mine:
                    continue
                many = max(mine, key=lambda a: (len(mine[a]), float(wkrng.random())))
                if len(mine[many]) >= 2:
                    walk_tests(f"R{k}", k, many, max(set(mine[many]), key=mine[many].count))
                ones = sorted(a for a in mine if len(mine[a]) == 1)
                if ones:
                    a = ones[int(wkrng.integers(len(ones)))]
                    walk_tests("once", k, a, mine[a][0])
            froms = {a for (_, a) in pairs}
            tos = {move(a, k, flipped) for (k, a) in pairs}
            cand = [(k, a) for k in ks_w for a in sorted(w.pos_at) if inside(move(a, k, flipped)) and (k, a) not in pairs
                    and a in froms and move(a, k, flipped) in tos]
            if cand:
                k, a = cand[int(wkrng.integers(len(cand)))]
                walk_tests("unseen", k, a, 0)
        # motion moments: (picture, verb, from, to); the template pair is (from, to)
        for k in ([] if placement or walk else [k_ for k_ in kinds if k_ != held]):
            a, b_ = starts(k, flipped, 2)
            obj = objs[int(rng.integers(0, OBJ_PER_ROOM))][1]
            ta, tb = w.pos_to[move(a, k, flipped)], w.pos_to[move(b_, k, flipped)]
            moments(f"R{k}", [obj, verbs[k], w.pos_from[a], ta], [ta], HEARD, TEST, pics)
            moments("once", [obj, verbs[k], w.pos_from[b_], tb], [tb], 1, TEST, pics)
        # the unseen pair: a basic verb's two positions heard only apart (each with another verb's moment), tested together
        k = None if placement or walk else int(rng.integers(0, NEW_KIND))
        if k is not None:
            (a,) = starts(k, flipped, 1)
            pa, pb = w.pos_from[a], w.pos_to[move(a, k, flipped)]
            for x in (pa, pb):
                for _ in range(2):
                    s = [x] + [int(y) for y in rng.choice(pics, int(rng.integers(FILL[0], FILL[1] + 1)), replace=False)]
                    heard.append(("apart", [s[i] for i in rng.permutation(len(s))], []))
            obj = objs[0][1]
            for _ in range(TEST):
                s = [obj, verbs[k], pa, pb] + [int(y) for y in rng.choice([q for q in pics if q != obj], 1, replace=False)]
                test.append(("unseen", [s[i] for i in rng.permutation(len(s))], [pb]))
        # world-model stage A (2026-10-06, the user's formula-library design): CHAIN items -- one object moved by two
        # different verbs, the start given, the end asked; the two verbs were never heard together (every heard moment has
        # one verb), so a bag-of-words body must chain two one-step rules it only ever met apart. Two verbs' moves add up in
        # either order (no edge on the way: every cell of both orders stays on the grid). A separate random stream, kept in
        # the room's "chain" list (NOT in "test"): every existing item, heard or tested, is unchanged.
        crng = np.random.default_rng(seed * 1009 + c + 3_000_003)
        chain = []
        if not placement:
            for _ in range(TEST):
                while True:                         # two moves that do not cancel (the end differs from the start)
                    k1, k2 = [int(x) for x in crng.choice(kinds, 2, replace=False)]
                    if held is not None and held not in (k1, k2):
                        k1 = held if k2 != held else k1
                    if k1 == k2:
                        continue
                    if (VERBS[k1][1][0] + VERBS[k2][1][0], VERBS[k1][1][1] + VERBS[k2][1][1]) != (0, 0):
                        break
                cells = [p for p in w.pos_at if all(inside(q) for q in (move(p, k1, flipped), move(p, k2, flipped),
                                                                          move(move(p, k1, flipped), k2, flipped)))]
                a = cells[int(crng.integers(len(cells)))]
                end = w.pos_to[move(move(a, k1, flipped), k2, flipped)]
                obj = objs[int(crng.integers(0, OBJ_PER_ROOM))][1]
                s = [obj, verbs[k1], verbs[k2], w.pos_from[a]] + [int(crng.choice([q for q in pics if q != obj]))]
                chain.append(("chain", [s[i] for i in crng.permutation(len(s))], [end]))
        # stage-1 path step 3a (2026-10-08, the user: run the logic BACKWARDS -- scene -> words): DESCRIBE items -- an object
        # seen at a start cell and an end cell (from(a), to(b)), the motion word asked. Start cells drawn anew (they may
        # coincide with a heard moment's); in a room no two motion words make the same move, so the word is decided. A
        # separate random stream, kept in the room's "describe" list (NOT in "test"): every existing item is unchanged.
        drng = np.random.default_rng(seed * 1039 + c + 9_000_031)
        describe = []
        if not placement:
            for k in [k_ for k_ in kinds if k_ != held]:
                cells = [p for p in w.pos_at if inside(move(p, k, flipped))]
                for i in drng.permutation(len(cells))[:TEST]:
                    a = cells[int(i)]
                    obj = objs[int(drng.integers(0, OBJ_PER_ROOM))][1]
                    s_ = [obj, w.pos_from[a], w.pos_to[move(a, k, flipped)]] + \
                        [int(drng.choice([q for q in pics if q != obj]))]
                    describe.append(("describe", [s_[j] for j in drng.permutation(len(s_))], [verbs[k]]))
        # Korean stories (2026-10-07, finding 123; the user: Korean first): ORDERED sentences of particle phrases -- [noun
        # particle] -- in a scrambled phrase order with the motion word last (SOV, as Korean): "OBJ ga  LM eseo  VERB" = the
        # object, starting where the landmark is, moves by the verb. Heard stories end with the scene the learner sees
        # (from(start), to(end)); test stories add a distractor landmark phrase "LM2 reul" and no scene -- the end is asked.
        # The particle tokens are made after all rooms (below), so nothing above changes; a separate random stream.
        story = []
        if korean and not placement:
            srng = np.random.default_rng(seed * 1019 + c + 5_000_011)
            room_names = [o[0] for o in objs]
            for bucket, n_per in (("heard", 2), ("test", TEST)):
                for k in [k_ for k_ in kinds if k_ != held]:
                    made, tries = 0, 0
                    while made < n_per and tries < 200:
                        tries += 1
                        mover, lm, lm2 = [room_names[int(i)] for i in srng.permutation(len(room_names))[:3]]
                        start = w.pos_of[obj_at[lm]]
                        end = move(start, k, flipped)
                        if not inside(end):
                            continue
                        phrases = [[mover, "GA"], [lm, "ESEO"]] + ([[lm2, "REUL"]] if bucket == "test" else [])
                        phrases = [phrases[int(i)] for i in srng.permutation(len(phrases))]
                        sent = [x for ph in phrases for x in ph] + [verbs[k]]
                        if bucket == "heard":
                            story.append(("story-heard", sent + [w.pos_from[start], w.pos_to[end]], []))
                        else:
                            story.append(("story", sent, [w.pos_to[end]]))
                        made += 1
        # stage-1 track A W6 (2026-10-15, criterion 5 two languages; W6_DESIGN.md; the user: "응, 너의 의견대로 하자" and
        # "테스트에 신경쓰다가 범용 로직이 아닌 테스트에 최적화된 로직을 설계하는 실수는 경계하고" -- W6 widens the world and
        # the tests only, no part changes). Every object gets a SECOND name, English-like (made last, "EN:<name>" here);
        # heard, each story in ONE language (the exam's exposure form -- no paired translations): English naming moments of
        # the same form and counts as the Korean ones (picture, name, where), and English stories "m VERB FROM lm" + the
        # scene (2 per motion word, as the Korean "story-heard"). "@bishuf" (control): every English naming moment shows a
        # RANDOM object of the room (picture and place drawn per moment) -- an English name points at no object. Tests (one
        # event per motion word; no scene): "bi story ko / ko d / en / en d" (the end cell; "d" = with a distractor noun --
        # Korean "[lm2 REUL]", English the object "m VERB lm2 FROM lm", which only word order resolves: report only), and
        # "bi where X>Y" / "bi what X>Y": the story told in X, then GO, then a question in Y (ko "[m GA] EODI" / en "WHERE IS
        # m"; ko "[MUEOS GA] [lm ESEO] VERB" / en "WHAT VERB FROM lm") -- the end cell / m's name in Y among the room's four
        # names in Y. A separate random stream; the "@multi" world is unchanged (the English heard moments are appended last).
        # EDITION 2 (2026-10-15, after the seed-0 smoke, no part changed): built on "@multi" instead of "@kor" -- in edition 1
        # the tracking part (+track, finding 126) never spoke (its trust is earned on heard GO-joined stories, none there).
        bi_items = []
        if bi:
            brng = np.random.default_rng(seed * 1361 + c + 113_000_271)
            EN = lambda t_: f"EN:{int(t_)}"
            for j, (name, drawn, conv) in enumerate(objs):
                pic, kind_, n_h = ((drawn, "name-en", HEARD) if j < 2 else (drawn, "name-en-once", 1) if j == 2 else
                                   (conv, "name-en-conv", HEARD))
                for _ in range(n_h):
                    jj = int(brng.integers(OBJ_PER_ROOM)) if bi_shuffle else j   # "@bishuf": a random object of the room
                    pic_ = (objs[jj][2] if j == 3 else objs[jj][1]) if bi_shuffle else pic
                    p_ = obj_at[objs[jj][0]]
                    pool = [int(b) for b in pics if b not in objs[j][1:] and b not in objs[jj][1:]]   # as the Korean moments
                    fill = [int(x) for x in brng.choice(pool, int(brng.integers(FILL[0], FILL[1] + 1)), replace=False)]
                    s_ = [int(pic_), EN(name), int(p_)] + fill
                    bi_items.append((kind_, [s_[i] for i in brng.permutation(len(s_))], []))
            if not placement:
                room_names = [o[0] for o in objs]
                ks_b = [k_ for k_ in kinds if k_ != held]
                for bucket, n_per in (("heard", 2), ("test", 1)):
                    for k in ks_b:
                        made, tries = 0, 0
                        while made < n_per and tries < 200:
                            tries += 1
                            m_, lm, lm2 = [room_names[int(i)] for i in brng.permutation(len(room_names))[:3]]
                            start = w.pos_of[obj_at[lm]]
                            end = move(start, k, flipped)
                            if not inside(end):
                                continue
                            en_plain = [EN(m_), verbs[k], "FROM", EN(lm)]
                            if bucket == "heard":
                                bi_items.append(("story-en-heard", en_plain + [w.pos_from[start], w.pos_to[end]], []))
                                made += 1
                                continue
                            ph = [[m_, "GA"], [lm, "ESEO"]]
                            ko_plain = [x for i in brng.permutation(2) for x in ph[int(i)]] + [verbs[k]]
                            ph3 = ph + [[lm2, "REUL"]]
                            ko_d = [x for i in brng.permutation(3) for x in ph3[int(i)]] + [verbs[k]]
                            en_d = [EN(m_), verbs[k], EN(lm2), "FROM", EN(lm)]
                            to_ = [w.pos_to[end]]
                            bi_items += [("bi story ko", ko_plain, to_), ("bi story ko d", ko_d, to_),
                                         ("bi story en", en_plain, to_), ("bi story en d", en_d, to_)]
                            told = {"ko": ko_plain, "en": en_plain}
                            ask_where = {"ko": [m_, "GA", "EODI"], "en": ["WHERE", "IS", EN(m_)]}
                            phq = [["MUEOS", "GA"], [lm, "ESEO"]]
                            ask_what = {"ko": [x for i in brng.permutation(2) for x in phq[int(i)]] + [verbs[k]],
                                        "en": ["WHAT", verbs[k], "FROM", EN(lm)]}
                            cands = {"ko": [m_] + [n_ for n_ in room_names if n_ != m_],
                                     "en": [EN(m_)] + [EN(n_) for n_ in room_names if n_ != m_]}
                            for x_ in ("ko", "en"):
                                for y_ in ("ko", "en"):
                                    bi_items.append((f"bi where {x_}>{y_}", told[x_] + ["GO"] + ask_where[y_], to_))
                                    bi_items.append((f"bi what {x_}>{y_}", told[x_] + ["GO"] + ask_what[y_], cands[y_]))
                            made += 1
        # multi-moment stories (2026-10-08, stage-1 path step 2 -- state over time; multi=True, the "@multi" world): clauses
        # joined by the connective GO ("-go", and then), the last clause asks "[OBJ ga] EODI" (where is OBJ now?). Kinds:
        # "same" -- one object, two moves (the second clause has no landmark: it goes on from where the object is now);
        # "two" -- the asked object moves once, then ANOTHER object moves from its own landmark (a distractor motion);
        # "three" -- A moves, B moves, A moves again. Heard (2 per kind): the story, the question and the scene of the asked
        # object's end (to(end)); test (TEST per kind): no scene, the end is asked. A separate random stream; the GO / EODI
        # tokens are made after the particles -- the "@kor" world is unchanged.
        multi_items = []
        if multi and not placement:
            mrng = np.random.default_rng(seed * 1031 + c + 7_000_019)
            room_names = [o[0] for o in objs]
            for bucket, n_per in (("heard", 2), ("test", TEST)):
                for kind_m in ("same", "two", "three"):
                    made, tries = 0, 0
                    while made < n_per and tries < 400:
                        tries += 1
                        a_, la, b_, lb = [room_names[int(i)] for i in mrng.permutation(len(room_names))[:4]]
                        ks_ = [k_ for k_ in kinds if k_ != held]
                        v1, v2, v3 = [int(ks_[int(i)]) for i in mrng.integers(0, len(ks_), 3)]
                        sa, sb = w.pos_of[obj_at[la]], w.pos_of[obj_at[lb]]

                        def step(p, k, block=None):
                            # the "@grav" world (step 4+): "falls" goes on until the edge or another object already in
                            # the story (every story of that world follows the same physics); elsewhere one move
                            if not (grav and k == 0):
                                return move(p, k, flipped)
                            while inside(move(p, k, flipped)) and move(p, k, flipped) != block:
                                p = move(p, k, flipped)
                            return p
                        ea = step(sa, v1)
                        if kind_m == "same":
                            ea2 = step(ea, v2)
                            path = [ea, ea2]
                            clauses = [[a_, "GA", la, "ESEO", verbs[v1]], [a_, "GA", verbs[v2]]]
                        else:
                            eb = step(sb, v2, ea)
                            path = [ea, eb]
                            clauses = [[a_, "GA", la, "ESEO", verbs[v1]], [b_, "GA", lb, "ESEO", verbs[v2]]]
                            if kind_m == "three":
                                ea = step(ea, v3, eb)
                                path.append(ea)
                                clauses.append([a_, "GA", verbs[v3]])
                            ea2 = ea
                        if not all(inside(p) for p in path):
                            continue
                        for cl in clauses:              # the phrase order inside a clause is free (as in finding 123)
                            if "ESEO" in cl and mrng.integers(2):
                                cl[:4] = cl[2:4] + cl[:2]
                        sent = [x for cl in clauses for x in cl + ["GO"]] + [a_, "GA", "EODI"]
                        if bucket == "heard":
                            multi_items.append(("multi-heard", sent + [w.pos_to[ea2]], []))
                        else:
                            multi_items.append((f"multi {kind_m}", sent, [w.pos_to[ea2]]))
                        made += 1
        # stage-1 track A W7b (2026-10-16, criterion 8 keys for the first score table; the user: "열쇠는 너의 생각에 따를게";
        # the caution 10-15 "테스트에 최적화된 로직을 설계하는 실수는 경계" -- tests and the question's teaching examples only, no
        # part changes). "@c8" = the "@multi" world plus (a) "where before": a story in which A moves twice ("same": A, A;
        # "three": A, B, A), then GO and "[A ga] JEONE EODI" (where was A before?) -> A's cell BEFORE its last move, an "at"
        # cell (a place where A was, not the end of a motion -- and so the tracking part's trust record, kept on heard stories
        # that end in a "to" cell, does not count these); heard 2 per kind (the question word has to be learned, as EONJE in
        # W4), test TEST per kind; (b) long stories (the exam's "2x long", test only): 4-6 clauses of two movers A and B
        # (A at least twice), then "[A ga] EODI" -> A's end ("to", as every where-now). "@c8shuf" (control): each heard
        # "before" answer is a random cell of the grid (drawn always, used only there, so the stories are the same).
        # A separate random stream; the "@multi" world is unchanged (the new heard moments and the token JEONE come last).
        c8_items = []
        if c8 and not placement:
            r8 = np.random.default_rng(seed * 1381 + c + 137_000_313)
            names8 = [o[0] for o in objs]
            ks8 = [k_ for k_ in kinds if k_ != held]

            def step8(p, k, block=None):
                if not (grav and k == 0):           # (as the "@multi" stories: "falls" goes on only in the "@grav" world)
                    return move(p, k, flipped)
                while inside(move(p, k, flipped)) and move(p, k, flipped) != block:
                    p = move(p, k, flipped)
                return p
            for bucket, n_per in (("heard", 2), ("test", TEST)):
                for kind_m in ("same", "three"):
                    made, tries = 0, 0
                    while made < n_per and tries < 400:
                        tries += 1
                        a_, la, b_, lb = [names8[int(i)] for i in r8.permutation(len(names8))[:4]]
                        v1, v2, v3 = [int(ks8[int(i)]) for i in r8.integers(0, len(ks8), 3)]
                        rand_cell = (int(r8.integers(-HALF, HALF + 1)), int(r8.integers(-HALF, HALF + 1)))
                        sa, sb = w.pos_of[obj_at[la]], w.pos_of[obj_at[lb]]
                        ea = step8(sa, v1)
                        if kind_m == "same":
                            prev, ea2 = ea, step8(ea, v2)
                            path = [ea, ea2]
                            clauses = [[a_, "GA", la, "ESEO", verbs[v1]], [a_, "GA", verbs[v2]]]
                        else:
                            eb = step8(sb, v2, ea)
                            prev, ea2 = ea, step8(ea, v3, eb)
                            path = [ea, eb, ea2]
                            clauses = [[a_, "GA", la, "ESEO", verbs[v1]], [b_, "GA", lb, "ESEO", verbs[v2]], [a_, "GA", verbs[v3]]]
                        if not all(inside(p) for p in path) or ea2 == prev or (kind_m == "three" and eb in (ea, ea2)):
                            continue                # (no two movers in one cell, as in the long stories)
                        for cl in clauses:
                            if "ESEO" in cl and r8.integers(2):
                                cl[:4] = cl[2:4] + cl[:2]
                        sent = [x for cl in clauses for x in cl + ["GO"]] + [a_, "GA", "JEONE", "EODI"]
                        if bucket == "heard":
                            c8_items.append(("c8-before-heard", sent + [w.pos_at[rand_cell if c8_shuffle else prev]], []))
                        else:
                            c8_items.append((f"c8 before {kind_m}", sent, [w.pos_at[prev]]))
                        made += 1
            made, tries = 0, 0
            while made < TEST and tries < 400:      # (b) long stories, test only
                tries += 1
                a_, la, b_, lb = [names8[int(i)] for i in r8.permutation(len(names8))[:4]]
                n_cl = int(r8.integers(4, 7))
                order = ["A", "A", "B"] + ["A" if r8.integers(2) else "B" for _ in range(n_cl - 3)]
                order = [order[int(i)] for i in r8.permutation(len(order))]
                vs = [int(ks8[int(i)]) for i in r8.integers(0, len(ks8), n_cl)]
                pos_, said, clauses, ok = {"A": w.pos_of[obj_at[la]], "B": w.pos_of[obj_at[lb]]}, set(), [], True
                for who, k in zip(order, vs):
                    other = pos_["B" if who == "A" else "A"]
                    pos_[who] = step8(pos_[who], k, other)
                    if not inside(pos_[who]) or pos_[who] == other:
                        ok = False
                        break
                    nm, lm = (a_, la) if who == "A" else (b_, lb)
                    cl = [nm, "GA", verbs[k]] if who in said else [nm, "GA", lm, "ESEO", verbs[k]]
                    if who not in said and r8.integers(2):
                        cl = cl[2:4] + cl[:2] + cl[4:]
                    said.add(who)
                    clauses.append(cl)
                if not ok:
                    continue
                sent = [x for cl in clauses for x in cl + ["GO"]] + [a_, "GA", "EODI"]
                c8_items.append(("c8 long", sent, [w.pos_to[pos_["A"]]]))
                made += 1
        # stage-1 path step 3b (2026-10-08): SPEAK items -- a scene (an object's name, from(a), to(b)) to be told as one
        # Korean sentence "[mover ga] [landmark eseo] VERB" (any phrase order). The truth kept with the item: the mover, the
        # nouns named at the start cell (any of them is a right landmark) and the motion word. Kept in "speak" (not in
        # "test"); a separate random stream; nothing else changes.
        speak = []
        if korean and not placement:
            krng = np.random.default_rng(seed * 1049 + c + 10_000_037)
            room_names = [o[0] for o in objs]
            for k in [k_ for k_ in kinds if k_ != held]:
                made, tries = 0, 0
                while made < TEST and tries < 200:
                    tries += 1
                    mover, lm = [room_names[int(i)] for i in krng.permutation(len(room_names))[:2]]
                    a = w.pos_of[obj_at[lm]]
                    b = move(a, k, flipped)
                    if not inside(b):
                        continue
                    lms = [n for n in room_names if n != mover and w.pos_of[obj_at[n]] == a]
                    speak.append(("speak", [mover, w.pos_from[a], w.pos_to[b]], {"mover": mover, "lms": lms,
                                                                                 "verb": verbs[k]}))
                    made += 1
        # step 4+ (2026-10-08, core knowledge: object permanence; record only): violation-of-expectation pairs on "three"
        # stories (A moves, B moves while A is not mentioned, A moves again; A asked) -- possible: A's true end; impossible:
        # "permanence" (A as if it had vanished and started again from its landmark: the landmark's place + its last move),
        # "identity" (A ends where B is). Kept in "voeperm" (a separate random stream; nothing else changes).
        voeperm = []
        if multi and not placement and not grav:   # (one-cell moves; the @grav world's falling is not one cell)
            prng_ = np.random.default_rng(seed * 1091 + c + 16_000_067)
            room_names = [o[0] for o in objs]
            ks_ = [k_ for k_ in kinds if k_ != held]
            for kind_p in ("permanence", "identity"):
                made, tries = 0, 0
                while made < TEST and tries < 400:
                    tries += 1
                    a_, la, b_, lb = [room_names[int(i)] for i in prng_.permutation(len(room_names))[:4]]
                    v1, v2, v3 = [int(ks_[int(i)]) for i in prng_.integers(0, len(ks_), 3)]
                    sa, sb = w.pos_of[obj_at[la]], w.pos_of[obj_at[lb]]
                    ea1 = move(sa, v1, flipped)
                    eb = move(sb, v2, flipped)
                    ea = move(ea1, v3, flipped)
                    bad = move(sa, v3, flipped) if kind_p == "permanence" else eb
                    if not all(inside(p) for p in (ea1, eb, ea, bad)) or bad == ea or eb in (ea1, ea):
                        continue
                    sent = [a_, "GA", la, "ESEO", verbs[v1], "GO", b_, "GA", lb, "ESEO", verbs[v2], "GO",
                            a_, "GA", verbs[v3], "GO", a_, "GA", "EODI"]
                    voeperm.append((f"voe {kind_p}", sent, [w.pos_to[ea], w.pos_to[bad]]))
                    made += 1
        # stage-1 path step 3c (2026-10-08): TELL items -- a sequence of pictures (frames joined by GO; a frame = each moving
        # object's name followed by the cell it is at now, at(p)) to be told as a multi-moment story. Kinds as the stories
        # above (same / two / three); the truth kept: the moments in order (mover, motion word, the nouns named at the start
        # cell for a mover's first move). Kept in "tell" (not in "test"); a separate random stream; nothing else changes.
        tell = []
        if multi and not placement:
            trng = np.random.default_rng(seed * 1051 + c + 11_000_041)
            room_names = [o[0] for o in objs]
            ks_ = [k_ for k_ in kinds if k_ != held]
            for kind_m in ("same", "two", "three"):
                made, tries = 0, 0
                while made < TEST and tries < 400:
                    tries += 1
                    a_, la, b_, lb = [room_names[int(i)] for i in trng.permutation(len(room_names))[:4]]
                    v1, v2, v3 = [int(ks_[int(i)]) for i in trng.integers(0, len(ks_), 3)]
                    plan = {"same": [(a_, v1), (a_, v2)], "two": [(a_, v1), (b_, v2)],
                            "three": [(a_, v1), (b_, v2), (a_, v3)]}[kind_m]
                    where = {a_: w.pos_of[obj_at[la]], b_: w.pos_of[obj_at[lb]]}
                    movers = [a_] if kind_m == "same" else [a_, b_]
                    frames, truth, first = [[x for m in movers for x in (m, w.pos_at[where[m]])]], [], set()
                    ok = True
                    for m, v in plan:
                        lms = None
                        if m not in first:
                            first.add(m)
                            lms = [n for n in room_names if n not in movers and w.pos_of[obj_at[n]] == where[m]]
                        where[m] = move(where[m], v, flipped)
                        if not inside(where[m]):
                            ok = False
                            break
                        truth.append((m, verbs[v], lms))
                        frames.append([x for mm in movers for x in (mm, w.pos_at[where[mm]])])
                    if not ok:
                        continue
                    sent = [x for f in frames for x in f + ["GO"]][:-1]
                    tell.append((f"tell {kind_m}", sent, truth))
                    made += 1
        # stage-1 path step 4 (2026-10-08, props=True, the "@objs" world; principle borrowed from New Horizons NH 56 -- what is
        # shared is not learned per object, what is learned per object is not shared): two objects; each object of the room is
        # HEAVY or LIGHT (fixed for the life; a separate random stream, at least one of each per room). A story: "[B ga] at(q)"
        # (B is at q) GO "[A ga] from(p) VERB" (A moves from p by the motion word) GO "[X ga] EODI". When A's move lands on B's
        # cell, a LIGHT B is pushed one cell the same way (A takes B's old cell); a HEAVY B stops A (both stay). Heard (2 per
        # object as the obstacle): with the asked object's end scene; test: the collisions with every object as the obstacle,
        # both objects asked ("objs hit"), and moves that miss B ("objs free"). Kept in "objs"; nothing else changes.
        objs_items = []
        if props and not placement:
            orng = np.random.default_rng(seed * 1061 + c + 12_000_043)
            room_names = [o[0] for o in objs]
            heavy = {n: bool(orng.integers(2)) for n in room_names}
            if len(set(heavy.values())) < 2:
                heavy[room_names[int(orng.integers(len(room_names)))]] = not heavy[room_names[0]]
            ks_ = [k_ for k_ in kinds if k_ != held]
            for bucket, n_per in (("heard", 2), ("test", TEST)):
                for b_ in room_names:
                    made, tries = 0, 0
                    while made < n_per and tries < 400:
                        tries += 1
                        a_ = [n for n in room_names if n != b_][int(orng.integers(len(room_names) - 1))]
                        k = int(ks_[int(orng.integers(len(ks_)))])
                        q = tuple(int(v) for v in orng.integers(-HALF, HALF + 1, 2))
                        d = VERBS[k][1]
                        d = (-d[0], -d[1]) if flipped else d
                        p = (q[0] - d[0], q[1] - d[1])
                        if not inside(p) or (not heavy[b_] and not inside(move(q, k, flipped))):
                            continue
                        end = {a_: p, b_: q} if heavy[b_] else {a_: q, b_: move(q, k, flipped)}
                        sent0 = [b_, "GA", w.pos_at[q], "GO", a_, "GA", w.pos_from[p], verbs[k], "GO"]
                        for x in ([a_, b_] if bucket == "test" else [(a_, b_)[int(orng.integers(2))]]):
                            sent = sent0 + [x, "GA", "EODI"]
                            if bucket == "heard":
                                objs_items.append(("objs-heard", sent + [w.pos_to[end[x]]], []))
                            else:
                                objs_items.append(("objs hit", sent, [w.pos_to[end[x]]]))
                        if bucket == "test":            # the same A and motion word from a start that misses B
                            free = [r for r in w.pos_at if inside(move(r, k, flipped)) and move(r, k, flipped) != q and r != q]
                            r = free[int(orng.integers(len(free)))]
                            objs_items.append(("objs free", [b_, "GA", w.pos_at[q], "GO", a_, "GA", w.pos_from[r], verbs[k],
                                                             "GO", a_, "GA", "EODI"], [w.pos_to[move(r, k, flipped)]]))
                        made += 1
            # step 4+ (2026-10-08): VIOLATION-OF-EXPECTATION pairs (as infant studies: a possible and an impossible outcome of
            # the same story -- does the learner's own surprise rise for the impossible one?). Kinds: "solid" (a heavy B: A
            # stays / A ends INSIDE B's cell), "push" (a light B asked: pushed / B stays while A takes its cell), "causal" (A
            # misses B, B asked: B stays / B moves without being touched), "teleport" (A moves freely: one step / two steps).
            # A separate random stream; kept in "voe"; nothing else changes.
            vrng = np.random.default_rng(seed * 1063 + c + 13_000_051)
            voe = []
            for kind_v in ("solid", "push", "causal", "teleport"):
                made, tries = 0, 0
                while made < TEST and tries < 400:
                    tries += 1
                    cands = [n for n in room_names if heavy[n] == (kind_v == "solid")] if kind_v in ("solid", "push") \
                        else room_names
                    b_ = cands[int(vrng.integers(len(cands)))]
                    a_ = [n for n in room_names if n != b_][int(vrng.integers(len(room_names) - 1))]
                    k = int(ks_[int(vrng.integers(len(ks_)))])
                    q = tuple(int(v) for v in vrng.integers(-HALF, HALF + 1, 2))
                    d = VERBS[k][1]
                    d = (-d[0], -d[1]) if flipped else d
                    if kind_v in ("solid", "push"):
                        p = (q[0] - d[0], q[1] - d[1])
                        if not inside(p) or not inside(move(q, k, flipped)):
                            continue
                        asked, good, bad = (a_, p, q) if kind_v == "solid" else (b_, move(q, k, flipped), q)
                    else:
                        free = [r for r in w.pos_at if inside(move(move(r, k, flipped), k, flipped)) and
                                move(r, k, flipped) != q and r != q and move(q, k, flipped) != r and inside(move(q, k, flipped))]
                        if not free:
                            continue
                        p = free[int(vrng.integers(len(free)))]
                        asked, good, bad = (b_, q, move(q, k, flipped)) if kind_v == "causal" else \
                            (a_, move(p, k, flipped), move(move(p, k, flipped), k, flipped))
                    sent = [b_, "GA", w.pos_at[q], "GO", a_, "GA", w.pos_from[p], verbs[k], "GO", asked, "GA", "EODI"]
                    voe.append((f"voe {kind_v}", sent, [w.pos_to[good], w.pos_to[bad]]))
                    made += 1
            objs_items += voe
            w.heavy = getattr(w, "heavy", {})
            w.heavy.update(heavy)
        # stage-1 path step 4+ (2026-10-08, grav=True, the "@grav" world; core knowledge: gravity / support): "falls" does not
        # stop after one cell -- the object falls until it rests (on the edge of the world below it, or on top of another
        # object in its way); the other motion words still move one cell. Stories "[B ga] at(q) GO [A ga] from(p) VERB GO
        # [A ga] EODI". Kinds: "grav floor" (nothing in the way), "grav stack" (B in the way: A rests on B), "grav step"
        # (another motion word: one cell). Heard 2 per kind with the end scene; test TEST per kind; violation-of-expectation
        # pairs "voe float" (rests at the edge / stops one cell down, in mid-air) and "voe through" (rests on B / ends beyond
        # B). In upside-down rooms "falls" goes up (the room's book) and so does the falling. A separate random stream.
        grav_items = []
        if grav and not placement:
            grng = np.random.default_rng(seed * 1069 + c + 14_000_057)
            room_names = [o[0] for o in objs]
            k_f = 0                                     # "falls"
            others = [k_ for k_ in kinds if k_ not in (k_f, held)]

            def fall(p, block):
                while inside(move(p, k_f, flipped)) and move(p, k_f, flipped) != block:
                    p = move(p, k_f, flipped)
                return p
            for bucket, n_per in (("heard", 2), ("test", TEST)):
                for kind_g in ("floor", "stack", "step", "float", "through"):
                    if bucket == "heard" and kind_g in ("float", "through"):
                        continue
                    made, tries = 0, 0
                    while made < n_per and tries < 400:
                        tries += 1
                        a_, b_ = [room_names[int(i)] for i in grng.permutation(len(room_names))[:2]]
                        p = tuple(int(v) for v in grng.integers(-HALF, HALF + 1, 2))
                        k = k_f if kind_g != "step" else int(others[int(grng.integers(len(others)))])
                        if kind_g in ("stack", "through"):
                            # B somewhere below A (along the fall) with at least one free cell between
                            path, r = [], p
                            while inside(move(r, k_f, flipped)):
                                r = move(r, k_f, flipped)
                                path.append(r)
                            if len(path) < 2:
                                continue
                            q = path[int(grng.integers(1, len(path)))]
                        else:
                            q = tuple(int(v) for v in grng.integers(-HALF, HALF + 1, 2))
                            if q == p:
                                continue
                        end = fall(p, q) if k == k_f else move(p, k, flipped)
                        if not inside(end) or end == q or end == p:
                            continue
                        if kind_g in ("floor", "float"):
                            if fall(p, None) != end:            # B must not be in the way
                                continue
                            if kind_g == "float" and move(p, k_f, flipped) == end:
                                continue                        # at least two cells to fall
                        sent = [b_, "GA", w.pos_at[q], "GO", a_, "GA", w.pos_from[p], verbs[k], "GO", a_, "GA", "EODI"]
                        if bucket == "heard":
                            grav_items.append(("grav-heard", sent + [w.pos_to[end]], []))
                        elif kind_g == "float":
                            grav_items.append(("voe float", sent, [w.pos_to[end], w.pos_to[move(p, k_f, flipped)]]))
                        elif kind_g == "through":
                            beyond = move(q, k_f, flipped)
                            if not inside(beyond):
                                continue
                            grav_items.append(("voe through", sent, [w.pos_to[end], w.pos_to[beyond]]))
                        else:
                            grav_items.append((f"grav {kind_g}", sent, [w.pos_to[end]]))
                        made += 1
        # D22 (2026-10-10, tale=True, the "@tale" world = "@grav" plus told tales; frames vs reality): in every room that is not
        # upside down, tales are told in the SAME room with the same objects: "YENNAL [T ga] at(q) GO [A ga] from(p) falls GO
        # [A ga] EODI to(end)" -- inside a tale only "falls" goes UP (one rule broken, the room not turned; the real upside-down
        # rooms turn every word). T is the room's tale-only thing (first heard in tales; made last). Heard 3 per room (floor,
        # floor, stack); in half of the rooms ("drop" rooms, drawn per room) one of the three is told WITHOUT the marker (the
        # child missed "once upon a time"). Tests (TEST each): "tale in" (marked; the answer goes up), "tale transfer" (a real,
        # unmarked stack question with T in the way: T learned in tales still blocks in reality). The real "grav" tests of the
        # room measure contamination. A separate random stream; YENNAL and the T things are made last.
        tale_items, tale_drop = [], None
        if tale and not placement and flipped is False:
            trng = np.random.default_rng(seed * 1213 + c + 43_000_151)
            tale_drop = bool(trng.random() < 0.5)
            t_names = [o[0] for o in objs]
            t_f = 0                                     # "falls"

            def t_move(p, up):
                return move(p, t_f, up)

            def t_fall(p, block, up):
                while inside(t_move(p, up)) and t_move(p, up) != block:
                    p = t_move(p, up)
                return p
            plan = [("heard", "floor"), ("heard", "floor"), ("heard", "stack")] + \
                   [("in", ("floor", "stack")[i % 2]) for i in range(TEST)] + [("transfer", "stack")] * TEST
            unmarked = int(trng.integers(3)) if tale_drop else -1
            n_heard = 0
            for bucket, kind_t in plan:
                up = bucket != "transfer"                   # tales: up; the transfer question is real: down
                for _ in range(400):
                    a_ = t_names[int(trng.integers(len(t_names)))]
                    b_ = "TALEOBJ" if bucket != "in" or trng.random() < 0.5 else \
                        [x for x in t_names if x != a_][int(trng.integers(len(t_names) - 1))]
                    p = tuple(int(v) for v in trng.integers(-HALF, HALF + 1, 2))
                    if kind_t == "stack":
                        path, r = [], p
                        while inside(t_move(r, up)):
                            r = t_move(r, up)
                            path.append(r)
                        if len(path) < 2:
                            continue
                        q = path[int(trng.integers(1, len(path)))]
                    else:
                        q = tuple(int(v) for v in trng.integers(-HALF, HALF + 1, 2))
                        if q == p or t_fall(p, None, up) != t_fall(p, q, up):
                            continue
                    end = t_fall(p, q, up)
                    if end == p or end == q:
                        continue
                    sent = [b_, "GA", w.pos_at[q], "GO", a_, "GA", w.pos_from[p], verbs[t_f], "GO", a_, "GA", "EODI"]
                    if bucket == "heard":
                        mark = [] if n_heard == unmarked else ["YENNAL"]
                        tale_items.append(("tale-heard" if mark else "tale-heard-unmarked", mark + sent + [w.pos_to[end]], []))
                        n_heard += 1
                    elif bucket == "in":
                        tale_items.append(("tale in", ["YENNAL"] + sent, [w.pos_to[end]]))
                    else:
                        tale_items.append(("tale transfer", sent, [w.pos_to[end]]))
                    break
        # D22 v1b (2026-10-10, "rooms known only from tales"): the late rooms of phase 1 are first TOLD about -- 2 tales of
        # room c (its things, "falls" up) are heard in the session of room c - 1 (phase 0, always earlier); in half of these
        # rooms (drawn per room) the preview tales have no marker. Before c is seen, its real gravity questions ask whether a
        # room known only from tales is taken as real. A separate random stream (the v1 tales above are unchanged).
        tale_pre, pre_drop = [], None
        if tale and not placement and flipped is False and phase == 1 and c > 1:
            prng_t = np.random.default_rng(seed * 1229 + c + 53_000_171)
            pre_drop = bool(prng_t.random() < 0.5)
            for _ in range(2):
                for _ in range(400):
                    a_ = t_names[int(prng_t.integers(len(t_names)))]
                    p = tuple(int(v) for v in prng_t.integers(-HALF, HALF + 1, 2))
                    q = tuple(int(v) for v in prng_t.integers(-HALF, HALF + 1, 2))
                    if q == p or t_fall(p, None, True) != t_fall(p, q, True):
                        continue
                    end = t_fall(p, q, True)
                    if end == p:
                        continue
                    sent = ["TALEOBJ", "GA", w.pos_at[q], "GO", a_, "GA", w.pos_from[p], verbs[t_f], "GO", a_, "GA", "EODI"]
                    tale_pre.append(("tale-pre" if not pre_drop else "tale-pre-unmarked",
                                     ([] if pre_drop else ["YENNAL"]) + sent + [w.pos_to[end]], []))
                    break
        # stage-1 path step 4+ (2026-10-08, contain=True, the "@contain" world; core knowledge: containment): two relation
        # particles -- ANE ("an-e", inside) and YEOPE ("yeop-e", beside). Stories "[C ga] at(q) GO [A ga] at(r) [C REL] GO
        # [C ga] VERB GO [A ga] EODI": inside (r = q), A moves with C; beside (r a neighbouring cell), A stays. Heard 2 per
        # relation with A's end scene; test TEST per relation ("contain in" / "contain by"); violation-of-expectation pairs
        # "voe inside" (A carried along / A left behind) and "voe beside" (A stays / A dragged along). A separate random
        # stream; the relation tokens are made after GO / EODI.
        contain_items = []
        if contain and not placement:
            crng2 = np.random.default_rng(seed * 1097 + c + 18_000_073)
            room_names = [o[0] for o in objs]
            ks_ = [k_ for k_ in kinds if k_ != held]
            for bucket, n_per in (("heard", 2), ("test", TEST), ("voe", TEST)):
                for rel in ("ANE", "YEOPE"):
                    made, tries = 0, 0
                    while made < n_per and tries < 400:
                        tries += 1
                        a_, c_ = [room_names[int(i)] for i in crng2.permutation(len(room_names))[:2]]
                        k = int(ks_[int(crng2.integers(len(ks_)))])
                        q = tuple(int(v) for v in crng2.integers(-HALF, HALF + 1, 2))
                        qe = move(q, k, flipped)
                        if not inside(qe):
                            continue
                        if rel == "ANE":
                            r, end, bad = q, qe, q
                        else:
                            nb = [(q[0] + dx, q[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                            nb = [x for x in nb if inside(x) and x != qe and inside((x[0] + qe[0] - q[0], x[1] + qe[1] - q[1]))]
                            if not nb:
                                continue
                            r = nb[int(crng2.integers(len(nb)))]
                            end, bad = r, (r[0] + qe[0] - q[0], r[1] + qe[1] - q[1])
                        sent = [c_, "GA", w.pos_at[q], "GO", a_, "GA", w.pos_at[r], c_, rel, "GO", c_, "GA", verbs[k], "GO",
                                a_, "GA", "EODI"]
                        if bucket == "heard":
                            contain_items.append(("contain-heard", sent + [w.pos_to[end]], []))
                        elif bucket == "test":
                            contain_items.append((f"contain {'in' if rel == 'ANE' else 'by'}", sent, [w.pos_to[end]]))
                        else:
                            contain_items.append((f"voe {'inside' if rel == 'ANE' else 'beside'}", sent,
                                                  [w.pos_to[end], w.pos_to[bad]]))
                        made += 1
        # stage-1 track A W3 (2026-10-13, criterion 9 spatial words; the "@sp" world = "@contain" plus): the four spatial words
        # WI (on), ARAE (under), ANE (in), YEOPE (beside) told as static statements "[B ga] at(q) GO [A ga] at(r) B REL" --
        # r = q one cell against the room's fall (WI), one cell along it (ARAE), the same cell (ANE, as "@contain"), one
        # cell to a side (YEOPE); upside-down rooms turn WI / ARAE with their gravity. Heard 2 per relation; tests TEST per
        # relation, never at a (B's cell, relation) heard in the room: "sp word" (the statement without REL: scene -> word)
        # and "sp scene" (without at(r): word -> scene); B's cell always leaves all four relations' cells on the grid, so both
        # are answered among the four (the same category: words / the four cells -- New Horizons' lesson). "@spshuf" (the
        # shuffle control): the heard statements carry a relation word drawn at random (a separate stream); the tests are
        # the same items. A separate random stream; WI / ARAE are made last -- every other world is unchanged.
        sp_items = []
        if sp and not placement:
            sprng = np.random.default_rng(seed * 1249 + c + 67_000_213)
            shrng = np.random.default_rng(seed * 1259 + c + 71_000_219)
            room_names = [o[0] for o in objs]
            rels = ("WI", "ARAE", "ANE", "YEOPE")

            def rel_cells(q, side):
                return {"WI": move(q, 1, flipped), "ARAE": move(q, 0, flipped), "ANE": q, "YEOPE": move(q, side, flipped)}
            heard_q = set()
            for bucket, n_per in (("heard", 2), ("test", TEST)):
                for rel in rels:
                    made, tries = 0, 0
                    while made < n_per and tries < 400:
                        tries += 1
                        a_, b_ = [room_names[int(i)] for i in sprng.permutation(len(room_names))[:2]]
                        q = tuple(int(v) for v in sprng.integers(-HALF, HALF + 1, 2))
                        side = int(sprng.choice([2, 3]))
                        cells = rel_cells(q, side)
                        if not all(inside(x) for x in cells.values()) or len(set(cells.values())) < 4:
                            continue
                        if bucket == "test" and (q, rel) in heard_q:
                            continue
                        r = cells[rel]
                        if bucket == "heard":
                            heard_q.add((q, rel))
                            told = rels[int(shrng.integers(4))] if sp_shuffle else rel
                            sp_items.append(("sp-heard", [b_, "GA", w.pos_at[q], "GO", a_, "GA", w.pos_at[r], b_, told], []))
                        else:
                            sp_items.append(("sp word", [b_, "GA", w.pos_at[q], "GO", a_, "GA", w.pos_at[r], b_],
                                             [rel] + [x for x in rels if x != rel]))
                            sp_items.append(("sp scene", [b_, "GA", w.pos_at[q], "GO", a_, "GA", b_, rel],
                                             [w.pos_at[r]] + [w.pos_at[cells[x]] for x in rels if x != rel]))
                        made += 1
        # stage-1 track A W4 (2026-10-13, criterion 7 "when"; the "@when" world = "@multi" plus): a "when" question that word
        # overlap cannot answer (the sealed exam v1's "when" says the story sentence again -- word overlap alone gives 1.00, so
        # it is report-only there). Stories of n = 2 / 3 / 4 moments (clauses joined by GO, as "@multi"); plans "AA" (n 2),
        # "AAB" / "ABA" / "BAA" (n 3), "AABB" / "ABAB" / "ABBA" (n 4): the asked object always moves TWICE, so its name does
        # not tell the moment (B moves once in n 3 and is never asked). An object's first clause names its landmark ("[A ga]
        # [LA eseo] VERB", the phrase order free), a later one goes on from where it is ("[A ga] VERB"); one-cell moves. The
        # question "[X ga] to(p) EONJE" (when did X go to p?): p is never said in the story (only landmarks and motion words
        # are), X's two arrival cells and its start all differ, so only following the moves decides the moment; the answer
        # is that clause's moment word (CHEOTJJAE / DULJJAE / SETJJAE / NETJJAE = first .. fourth); plan, asked object and
        # arrival are drawn uniformly, so within each n every moment is the answer equally often. Report-only "when echo"
        # (the exam v1 form): "EONJE" + one clause said again (drawn uniformly; clauses are unique in a story). Heard 2 per n
        # and form with the answer word last; tests TEST per n and form, answered AMONG THE STORY'S n MOMENT WORDS (the
        # same category). "@whenshuf" (the shuffle control): the heard items' answer word is drawn at random among the
        # story's n moment words; the tests are the same items. Separate random streams; EONJE and the moment words are made
        # last -- every other world is unchanged.
        when_items = []
        if when and not placement:
            wrng = np.random.default_rng(seed * 1297 + c + 79_000_231)
            wsrng = np.random.default_rng(seed * 1301 + c + 83_000_233)
            room_names = [o[0] for o in objs]
            ks_ = [k_ for k_ in kinds if k_ != held]
            moments_w = ("CHEOTJJAE", "DULJJAE", "SETJJAE", "NETJJAE")
            plans_w = {"two": ("AA",), "three": ("AAB", "ABA", "BAA"), "four": ("AABB", "ABAB", "ABBA")}
            for bucket, n_per in (("heard", 2), ("test", TEST)):
                for form in ("to", "echo"):
                    for kind_w, plans in plans_w.items():
                        made, tries = 0, 0
                        while made < n_per and tries < 400:
                            tries += 1
                            plan = plans[int(wrng.integers(len(plans)))]
                            a_, la, b_, lb = [room_names[int(i)] for i in wrng.permutation(len(room_names))[:4]]
                            who, lm_of = {"A": a_, "B": b_}, {"A": la, "B": lb}
                            where = {"A": w.pos_of[obj_at[la]], "B": w.pos_of[obj_at[lb]]}
                            start, arrive, clauses, ok = dict(where), {"A": [], "B": []}, [], True
                            for j, m in enumerate(plan):
                                k = int(ks_[int(wrng.integers(len(ks_)))])
                                if plan.index(m) == j:
                                    cl = [who[m], "GA", lm_of[m], "ESEO", verbs[k]]
                                    if wrng.integers(2):
                                        cl[:4] = cl[2:4] + cl[:2]
                                else:
                                    cl = [who[m], "GA", verbs[k]]
                                where[m] = move(where[m], k, flipped)
                                if not inside(where[m]):
                                    ok = False
                                    break
                                arrive[m].append((where[m], j))
                                clauses.append(cl)
                            if not ok:
                                continue
                            asked = [m for m in "AB" if len(arrive[m]) == 2]
                            m = asked[int(wrng.integers(len(asked)))]
                            if len({start[m]} | {p_ for p_, _ in arrive[m]}) < 3:
                                continue
                            p_, j = arrive[m][int(wrng.integers(2))]
                            if form == "echo":
                                j = int(wrng.integers(len(plan)))
                                q = ["EONJE"] + clauses[j]
                            else:
                                q = [who[m], "GA", w.pos_to[p_], "EONJE"]
                            n = len(plan)
                            sent = [x for cl in clauses for x in cl + ["GO"]] + q
                            if bucket == "heard":
                                told = moments_w[int(wsrng.integers(n))] if when_shuffle else moments_w[j]
                                when_items.append(("when-heard", sent + [told], []))
                            else:
                                when_items.append((f"when {form} {kind_w}", sent,
                                                   [moments_w[j]] + [moments_w[i] for i in range(n) if i != j]))
                            made += 1
        # stage-1 path step 4+ (2026-10-09, count=True, the "@count" world; core knowledge: small numbers -- as Wynn 1992,
        # 1 + 1 behind a screen): things are put into a container one by one ("[X ga] at(q) [C ane]") and sometimes one is
        # taken out ("[X ga] at(r)", elsewhere); the question "[C ane] MYEOT" (how many in C?) is answered by a number word
        # HANA / DUL / SET (1 / 2 / 3). Heard 2 per final count with the number word; test "count put" (no taking out) and
        # "count take"; violation-of-expectation pairs "voe number" (the right count / one more or one less). A separate
        # random stream; the number words and MYEOT are made last.
        count_items = []
        if count and not placement and not ten:     # ("@ten" makes its own number items after the rooms, below)
            nrng = np.random.default_rng(seed * 1103 + c + 20_000_081)
            room_names = [o[0] for o in objs]
            nums = {1: "HANA", 2: "DUL", 3: "SET"}
            for bucket, n_per in (("heard", 2), ("test", TEST), ("voe", TEST)):
                for final in (1, 2, 3):
                    for take in ((False, True) if final < 3 else (False,)):
                        if bucket == "heard" and take:
                            continue
                        for _ in range(n_per):
                            c_ = room_names[int(nrng.integers(len(room_names)))]
                            others = [n for n in room_names if n != c_]
                            put = final + int(take)
                            things = [others[int(i)] for i in nrng.permutation(len(others))[:put]]
                            q = tuple(int(v) for v in nrng.integers(-HALF, HALF + 1, 2))
                            sent = [c_, "GA", w.pos_at[q], "GO"]
                            for x in things:
                                sent += [x, "GA", w.pos_at[q], c_, "ANE", "GO"]
                            if take:
                                r = [p for p in w.pos_at if p != q][int(nrng.integers(len(w.pos_at) - 1))]
                                sent += [things[int(nrng.integers(put))], "GA", w.pos_at[r], "GO"]
                            sent += [c_, "ANE", "MYEOT"]
                            if bucket == "heard":
                                count_items.append(("count-heard", sent + [nums[final]], []))
                            elif bucket == "test":
                                count_items.append((f"count {'take' if take else 'put'}", sent, [nums[final]]))
                            else:
                                wrong = final + 1 if final < 3 and (final == 1 or nrng.integers(2)) else final - 1
                                count_items.append(("voe number", sent, [nums[final], nums[wrong]]))
        # step 4+ (2026-10-09, succ=True, the "@succ" world; ledger D1a -- the successor rule): the counting list is heard
        # as a chant, "HANA DUL SET NET" (children recite it before they know what the later words mean); counts are heard
        # with their words only up to 3 (as in "@count"); test "count four": four things put into a container (the fourth
        # borrowed from the previous room -- a room has four objects) -- the word NET was never heard with a count;
        # violation-of-expectation pairs "voe four" (NET / SET). A separate random stream.
        if succ and not placement:
            srng4 = np.random.default_rng(seed * 1117 + c + 22_000_087)
            count_items.append(("count-heard", ["HANA", "DUL", "SET", "NET"], []))
            if prev_names:
                room_names = [o[0] for o in objs]
                for bucket in ("test", "voe"):
                    for _ in range(TEST):
                        c_ = room_names[int(srng4.integers(len(room_names)))]
                        things = [n for n in room_names if n != c_] + [prev_names[int(srng4.integers(len(prev_names)))]]
                        things = [things[int(i)] for i in srng4.permutation(len(things))]
                        q = tuple(int(v) for v in srng4.integers(-HALF, HALF + 1, 2))
                        sent = [c_, "GA", w.pos_at[q], "GO"]
                        for x in things:
                            sent += [x, "GA", w.pos_at[q], c_, "ANE", "GO"]
                        sent += [c_, "ANE", "MYEOT"]
                        if bucket == "test":
                            count_items.append(("count four", sent, ["NET"]))
                        else:
                            count_items.append(("voe four", sent, ["NET", "SET"]))
            prev_names = [o[0] for o in objs]
        # stage-1 path step 4+ (2026-10-09; ledger D13 + D15 -- the number-line room, the user's design C): in the placement
        # rooms (no motion moments at all) a NUMBER LINE: the number words "N1".."N5" are named at the line's five cells (as
        # objects are named at their places); heard: one-step stories in Korean on the line, "[OBJ ga] [N eseo] pushed" with
        # the scene (from(n), to(n + 1)), n = 1, 2 (results up to 3); test "numline add": pushed two or three times, or from
        # 3 / 4 (results 4 and 5, never heard as results); test "numline sub": "pulled" one to three times -- a word NEVER
        # heard on the line (heard only in the rooms). numline="row": the line IS the rooms' middle row (pos(x, 0), the same
        # coordinates and spacing as every room); numline="own": the line's own cells, half the rooms' spacing (0.1), in a
        # plane of its own (made last, below). Placeholders "N*", "LA*", "LF*", "LT*" are replaced after the particles; a
        # separate random stream -- every other item of the room and the world is unchanged.
        numline_items = []
        if numline and placement:
            lrng = np.random.default_rng(seed * 1123 + c + 23_000_089)
            room_names = [o[0] for o in objs]
            for n in range(1, 6):
                for _ in range(HEARD):
                    s = [f"N{n}", f"LA{n}", int(lrng.choice(pics))]
                    numline_items.append(("numline-name", [s[i] for i in lrng.permutation(len(s))], []))

            def nl_story(start, k, steps, bucket):
                mover, lm2 = [room_names[int(i)] for i in lrng.permutation(len(room_names))[:2]]
                phrases = [[mover, "GA"], [f"N{start}", "ESEO"]] + ([[lm2, "REUL"]] if bucket != "heard" else [])
                phrases = [phrases[int(i)] for i in lrng.permutation(len(phrases))]
                return [x for ph in phrases for x in ph] + [verbs[k]] * steps
            for start in (1, 2):
                for _ in range(2):
                    numline_items.append(("numline-heard", nl_story(start, 2, 1, "heard") + [f"LF{start}", f"LT{start + 1}"],
                                          []))
            for start, steps in ((2, 2), (3, 1), (1, 3), (3, 2), (4, 1), (2, 3)):
                numline_items.append(("numline add", nl_story(start, 2, steps, "test"), [f"LT{start + steps}"]))
            for start, steps in ((5, 1), (4, 1), (5, 2), (4, 2), (3, 2), (5, 3)):
                numline_items.append(("numline sub", nl_story(start, 3, steps, "test"), [f"LT{start - steps}"]))
        # stage-1 path step 4+ (2026-10-09; ledger D17 v1 -- the three-view world, the user's idea of 2026-09-23 / 10-09):
        # in the placement rooms the objects stand in a 3-D room (x, y, z in -2..2) and are SEEN in three pictures at once,
        # each picture a 5 x 5 image on the SAME image cells as every room (a retina codes image places the same way for every
        # picture): front (x, y), top (x, z), side (z, y); which picture a cell belongs to is a view label right after it
        # ("VFRONT", "VTOP", "VSIDE", made last). ONE 3-D state drawn three ways -- never three pictures moved apart.
        # Heard: an object's three views (HEARD per object); depth moves "goes" (z + 1) / "comes" (z - 1) -- new motion
        # words, made last -- seen in all three views before and after (2 per word). Test "views fill": two views of a
        # point never seen, the third view asked; test "views depth": the three views before a depth move, one view after
        # asked (front: unchanged; top, side: moved). A separate random stream; every other item is unchanged.
        views_items = []
        if views and placement:
            vrng = np.random.default_rng(seed * 1151 + c + 27_000_103)
            room_names = [o[0] for o in objs]
            proj = {"VFRONT": lambda p: (p[0], p[1]), "VTOP": lambda p: (p[0], p[2]), "VSIDE": lambda p: (p[2], p[1])}

            def pt():
                return tuple(int(v) for v in vrng.integers(-HALF, HALF + 1, 3))

            def seen(p, role, labels):
                pairs = [[role[proj[l](p)], l] for l in labels]
                return [x for i in vrng.permutation(len(pairs)) for x in pairs[int(i)]]
            for name in room_names:
                p = pt()
                for _ in range(HEARD):
                    views_items.append(("views-heard", [name] + seen(p, w.pos_at, list(proj)), []))
            for vname, dz in (("GOES", 1), ("COMES", -1)):
                for _ in range(2):
                    while True:
                        p = pt()
                        if abs(p[2] + dz) <= HALF:
                            break
                    q = (p[0], p[1], p[2] + dz)
                    views_items.append(("views-heard", [room_names[int(vrng.integers(len(room_names)))]] +
                                        seen(p, w.pos_from, list(proj)) + [vname] + seen(q, w.pos_to, list(proj)), []))
            labels = list(proj)
            for miss in labels:
                for _ in range(TEST):
                    p = pt()
                    given = [l for l in labels if l != miss]
                    s = [room_names[int(vrng.integers(len(room_names)))]] + seen(p, w.pos_at, given) + [miss]
                    views_items.append(("views fill", s, [w.pos_at[proj[miss](p)]]))
            for ask in labels:
                for _ in range(TEST):
                    vname, dz = (("GOES", 1), ("COMES", -1))[int(vrng.integers(2))]
                    while True:
                        p = pt()
                        if abs(p[2] + dz) <= HALF:
                            break
                    q = (p[0], p[1], p[2] + dz)
                    s = [room_names[int(vrng.integers(len(room_names)))]] + seen(p, w.pos_from, labels) + [vname, ask]
                    views_items.append(("views depth", s, [w.pos_to[proj[ask](q)]]))
            # stage-1 path step 4+ (2026-10-09; ledger D2's viewpoint tests, carried over from D17; views=2, the "@viewpt"
            # world -- everything above unchanged, then): the BACK view (the front seen from behind: (-x, y)) heard only in
            # some sightings, and TURNS -- a quarter turn of the object about the vertical axis through the room's centre,
            # (x, y, z) -> (z, y, -x), seen in three views before and after (4 per room). Test "viewpt back": front and top
            # of a point never seen, the back asked (Piaget's three mountains: what does the one on the other side see?);
            # "viewpt turn": three views before a turn, one view after asked (mental rotation -- a turn is not a shift: the
            # axes trade places). A separate random stream.
            if views == 2:
                prng = np.random.default_rng(seed * 1163 + c + 31_000_109)
                proj["VBACK"] = lambda p: (-p[0], p[1])
                pt_ = lambda: tuple(int(v) for v in prng.integers(-HALF, HALF + 1, 3))

                def seen2(p, role, labels):
                    pairs = [[role[proj[l](p)], l] for l in labels]
                    return [x for i in prng.permutation(len(pairs)) for x in pairs[int(i)]]
                for name in room_names:
                    p = pt_()
                    for _ in range(HEARD // 2):
                        views_items.append(("views-heard", [name] + seen2(p, w.pos_at, ["VFRONT", "VBACK", "VTOP"]), []))
                three = ["VFRONT", "VTOP", "VSIDE"]
                turn = lambda p: (p[2], p[1], -p[0])
                for _ in range(4):
                    p = pt_()
                    views_items.append(("views-heard", [room_names[int(prng.integers(len(room_names)))]] +
                                        seen2(p, w.pos_from, three) + ["TURNS"] + seen2(turn(p), w.pos_to, three), []))
                for _ in range(TEST * 2):
                    p = pt_()
                    s = [room_names[int(prng.integers(len(room_names)))]] + seen2(p, w.pos_at, ["VFRONT", "VTOP"]) + ["VBACK"]
                    views_items.append(("viewpt back", s, [w.pos_at[proj["VBACK"](p)]]))
                for ask in three:
                    for _ in range(TEST):
                        p = pt_()
                        s = [room_names[int(prng.integers(len(room_names)))]] + seen2(p, w.pos_from, three) + ["TURNS", ask]
                        views_items.append(("viewpt turn", s, [w.pos_to[proj[ask](turn(p))]]))
        # stage-1 path step 4+ (2026-10-09; ledger D2, its second part -- goal-directed motion: infants expect an agent to
        # move efficiently toward its goal, Gergely & Csibra 1995 / 2003; goal=True, the "@goal" world): in the placement
        # rooms three objects -- the AGENT (named first), its goal and another thing -- each named with its cell, then the
        # agent's one-cell steps (the basic motion words). Heard: whole shortest paths that end at the goal (2 per room).
        # Test "goal which": the first steps (never reaching either thing), then EODIRO (where to?) -- the goal's name is
        # asked; "goal next": the first steps, the next motion word asked (made unique: what is left lies on one axis).
        # The steps approach the goal and never approach the other thing. A separate random stream.
        goal_items = []
        if goal and placement:
            grng = np.random.default_rng(seed * 1181 + c + 35_000_117)
            room_names = [o[0] for o in objs]
            step = {(1, 0): 2, (-1, 0): 3, (0, 1): 1, (0, -1): 0}     # pushed, pulled, rises, falls (VERBS order)
            dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])

            def scene():
                while True:
                    cells = [tuple(int(v) for v in grng.integers(-HALF, HALF + 1, 2)) for _ in range(3)]
                    pa, pg, pd = cells
                    if len(set(cells)) == 3 and dist(pa, pg) >= 3:
                        break
                moves = [(int(np.sign(pg[0] - pa[0])), 0)] * abs(pg[0] - pa[0]) + [(0, int(np.sign(pg[1] - pa[1])))] * abs(pg[1] - pa[1])
                moves = [moves[int(i)] for i in grng.permutation(len(moves))]
                path, p = [pa], pa
                for m in moves:
                    p = (p[0] + m[0], p[1] + m[1])
                    path.append(p)
                return pa, pg, pd, moves, path
            agent, others = room_names[0], room_names[1:]
            for bucket, n in (("heard", 2), ("which", TEST * 2), ("next", TEST * 2)):
                made = 0
                while made < n:
                    pa, pg, pd, moves, path = scene()
                    g_, d_ = [others[int(i)] for i in grng.permutation(len(others))[:2]]
                    head = [agent, w.pos_at[pa]] + [x for nm, p in [(g_, pg), (d_, pd)][::int(grng.choice([1, -1]))]
                                                     for x in (nm, w.pos_at[p])]
                    if bucket == "heard":
                        goal_items.append(("goal-heard", head + [verbs[step[m]] for m in moves], []))
                        made += 1
                        continue
                    k = len(moves) - 1 if bucket == "next" else int(grng.integers(1, len(moves)))
                    first = moves[:k]
                    if any(dist(q, pd) < dist(path[i], pd) for i, q in enumerate(path[1:k + 1])) or pd in path[:k + 1]:
                        continue                        # the steps must not approach the other thing
                    if bucket == "next":
                        rest = moves[k:]
                        if len(set(rest)) != 1:
                            continue
                        goal_items.append(("goal next", head + [verbs[step[m]] for m in first], [verbs[step[rest[0]]]]))
                    else:
                        goal_items.append(("goal which", head + [verbs[step[m]] for m in first] + ["EODIRO"], [g_]))
                    made += 1
        # stage-1 path step 4+ (2026-10-10; ledger D2, third part -- the self by contingency: infants find what they move
        # by the timing match between their own action and a seen motion, Watson 1972, Bahrick & Watson 1985; self_=True, the
        # "@self" world): Voyager is given ACTION words for the first time ("ACT0".."ACT3", made last: its own doings).
        # In the placement rooms one object is "me" (a different one per room -- as a body seen anew in a mirror or a
        # video). Heard: an action and the move it makes -- only "me" moves (6 per room). Test "self which": two actions,
        # and in each, three things move (me + two others); NUGU (who is me?) -- the one that moved with the action both
        # times; "self mirror": the same, but "me" is seen mirrored (left and right swapped, as in a mirror). The others'
        # moves are drawn so that no symmetry of the square makes them follow the actions. A separate random stream.
        self_items = []
        if self_ and placement:
            srng5 = np.random.default_rng(seed * 1193 + c + 39_000_127)
            room_names = [o[0] for o in objs]
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            syms = [lambda d: d, lambda d: (-d[0], d[1]), lambda d: (d[0], -d[1]), lambda d: (-d[0], -d[1]),
                    lambda d: (d[1], d[0]), lambda d: (-d[1], d[0]), lambda d: (d[1], -d[0]), lambda d: (-d[1], -d[0])]
            me = room_names[int(srng5.integers(len(room_names)))]
            others = [n for n in room_names if n != me]

            def mv(p, d):
                q = (p[0] + d[0], p[1] + d[1])
                return q if max(abs(q[0]), abs(q[1])) <= HALF else None
            for _ in range(6):
                while True:
                    k = int(srng5.integers(4))
                    p = tuple(int(v) for v in srng5.integers(-HALF, HALF + 1, 2))
                    if mv(p, dirs[k]) is not None:
                        break
                self_items.append(("self-heard", [f"ACT{k}", me, w.pos_from[p], w.pos_to[mv(p, dirs[k])]], []))
            for kind_, mirror in (("self which", False), ("self mirror", True)):
                made = 0
                while made < TEST * 2:
                    acts = [int(x) for x in srng5.integers(4, size=2)]
                    if mirror and all(k >= 2 for k in acts):
                        continue                        # (a mirror only shows on a left / right action)
                    ds = {n: [] for n in [me] + others[:2]}
                    for k in acts:
                        d = dirs[k]
                        ds[me].append((-d[0], d[1]) if mirror else d)
                        for n in others[:2]:
                            ds[n].append(dirs[int(srng5.integers(4))])
                    if any(all(f(dirs[k]) == dd for k, dd in zip(acts, ds[n])) for n in others[:2] for f in syms):
                        continue                        # an other would follow the actions under some symmetry
                    segs, ok = [], True
                    for i, k in enumerate(acts):
                        seg = [f"ACT{k}"]
                        for n in [me] + others[:2]:
                            for _ in range(20):
                                p = tuple(int(v) for v in srng5.integers(-HALF, HALF + 1, 2))
                                if mv(p, ds[n][i]) is not None:
                                    break
                            seg += [n, w.pos_from[p], w.pos_to[mv(p, ds[n][i])]]
                        body = seg[1:]
                        trip = [body[j:j + 3] for j in range(0, len(body), 3)]
                        trip = [trip[int(t)] for t in srng5.permutation(len(trip))]
                        segs += [seg[0]] + [x for t in trip for x in t]
                    self_items.append((kind_, segs + ["NUGU"], [me]))
                    made += 1
        # D75 step 1 (2026-10-17, "@scene" / "@sceneshuf"; D75_DESIGN.md): the same walk told WITHOUT role markers, as a
        # child sees it -- every move becomes a sentence [the mover's name, the motion word] followed by the scene after it,
        # one moment [picture, at(cell)] per thing of the room (no background, no from / to). Where a thing was before is
        # only in the earlier moments (its naming moment, the scenes before). "@sceneshuf": the sentence's motion word is
        # drawn at random among the room's (a separate stream) while the scene shows the true move -- nothing to learn from
        # the pairing. Tests "scene" (cat["scene"], NOT in "test"): in every walking room, per motion word, a start never
        # heard with that word there: [picture, at(start), name, word] -> at(arrival) among the room's motion words'
        # arrival cells from that start (the order kept: the scene line first, then the sentence). The walk itself, its
        # streams and every other world are unchanged.
        scene_items = []
        if scene_form and walk_moves:
            scrng = np.random.default_rng(seed * 1423 + c + 151_000_303)
            shrng7 = np.random.default_rng(seed * 1427 + c + 157_000_307)
            told = []
            for j, k, a, b in walk_moves:
                kk = int(ks_w[int(shrng7.integers(len(ks_w)))]) if scene_shuffle else k
                told.append([("scene sentence", [objs[j][0], verbs[kk]], [])] +
                            [("scene line", [objs[jj][1], w.pos_at[p_]], []) for jj, p_ in
                             enumerate(_walk_now(walk_moves, j, b, objs, w, obj_at, told_len=len(told)))])
            first = walk_idx[0]
            if seq:
                seq_rooms[c] = (first, [len(g) for g in told], list(walk_moves), [w.pos_of[obj_at[o[0]]] for o in objs])
            heard[:] = heard[:first] + [m for ms in told for m in ms] + heard[walk_idx[-1] + 1:]
            heard_k = {(k, a) for _, k, a, _ in walk_moves}
            for k in ks_w:
                made, tries = 0, 0
                while made < TEST and tries < 400:
                    tries += 1
                    a = tuple(int(v) for v in scrng.integers(-HALF, HALF + 1, 2))
                    ends = [move(a, kx, flipped) for kx in ks_w]
                    if (k, a) in heard_k or not all(inside(e) for e in ends) or len(set(ends)) < len(ends):
                        continue
                    j = int(scrng.integers(OBJ_PER_ROOM))
                    scene_items.append((f"scene {k}", [objs[j][1], w.pos_at[a], objs[j][0], verbs[k]],
                                        [w.pos_at[move(a, k, flipped)]] + [w.pos_at[e] for kx, e in zip(ks_w, ends) if kx != k]))
                    made += 1
        room_words = [t for o in objs for t in o]
        noise_pool = [t for o in objs for t in o[1:]] if nw else room_words     # "@nw": no name in the noise
        for _ in range(int(round(len(heard) * NOISE))):
            heard.append(("noise", [int(x) for x in rng.choice(noise_pool, 3, replace=False)], []))
        w.categories.append({"heard": heard, "test": test, "second": phase > 0, "phase": phase, "flipped": flipped,
                             "words": room_words, "objects": objs, "room": c, "placement": placement,
                             "chain": chain, "held": held, "story": story, **({"bi_raw": bi_items} if bi else {}),
                             **({"c8_raw": c8_items} if c8 else {}),
                             "multi": multi_items, "describe": describe,
                             "speak": speak, "tell": tell, "objs": objs_items,
                             "grav": grav_items, "voeperm": voeperm,
                             "contain": contain_items, "count": count_items, "numline": numline_items,
                             "views": views_items, "goal": goal_items, "self": self_items,
                             "tale": tale_items, "tale_drop": tale_drop, "tale_pre": tale_pre, "pre_drop": pre_drop})
        if sp:
            w.categories[-1]["sp_raw"] = sp_items
        if when:
            w.categories[-1]["when_raw"] = when_items
        if nw:
            w.categories[-1]["nw"] = nw_items
        if scene_form:
            w.categories[-1]["scene"] = scene_items
        if walk and not scene_form:                  # the time order: naming first, the walk, the noise spread through it
            wo = np.random.default_rng(seed * 1327 + c + 103_000_249)
            noise_i = [i for i, x in enumerate(heard) if x[0] == "noise"]
            order = [i for i in range(len(heard)) if i not in set(walk_idx) and i not in set(noise_i)] + walk_idx
            for i in noise_i:
                order.insert(int(wo.integers(len(order) + 1)), i)
            w.categories[-1]["order"] = order
    # stage-1 track A W5 (2026-10-14, ten=True, the "@ten" world = the "@count" world with the numbers 1-10 and putting in /
    # taking out instead of 1-3; criteria 3 and 4 of STAGE1 edition 2, in the forms of the sealed exam's items, not its
    # content): things are put into a container one by one ("[X ga] at(q) [C ane]") and some taken out ("[X ga] at(r)",
    # elsewhere); the question "[C ane] MYEOT" is answered by a number word 1-10 (HANA .. YEOL). A room has four objects, so
    # the things are drawn uniformly from every room's objects (so that a large count does not mean "borrowed"). Heard per
    # motion room: one story per count 1-10 (the exam's count scenes) and 4 box stories, put a <= 5 / take b < a, never the
    # held pairs (3, 1) (4, 2) (5, 3) (the exam's box stories). Tests: "ten count n" (put n; 2 per n), "ten pick n" (the
    # number word -> which of four stories with n / three other counts; 1 per n), "ten next n" (told "[C ane] NUM_n", one
    # more put in; n = 1-9; report only), "ten box held / far / taught" (the held pairs; a 6-10; heard pairs: 3 / 3 / 2) and
    # "voe ten held / far / taught" (the right count / one more or one fewer; the same mix, other stories). "@tenshuf": the
    # heard answers drawn at random among the ten words. Separate random streams; NET .. YEOL are made last of all.
    if ten:
        NUMS = ("HANA", "DUL", "SET", "NET", "DASEOT", "YEOSEOT", "ILGOP", "YEODEOL", "AHOP", "YEOL")
        HELD = ((3, 1), (4, 2), (5, 3))
        TAUGHT = [(a, b) for a in range(1, 6) for b in range(a) if (a, b) not in HELD]
        FAR = [(a, b) for a in range(6, 11) for b in range(a)]
        # (edition 2 of the W5 world, 2026-10-14, BEFORE any W5 result: a W5 life did not fit the 2-hour job -- the things
        # drawn from EVERY room's objects put ~120 names into every room's stories; now the things come from a fixed
        # world-wide pool of 12 counting things (objects of the placement rooms, drawn once) plus the room's own three,
        # uniformly; and 5 count scenes a room with distinct n instead of all ten. Edition 3, 2026-10-14 ~14:00Z, still
        # before any W5 result: the seed-0 smoke of edition 2 also died at the 2-hour limit -- 3 count scenes and 2 box
        # stories a room)
        prng_t = np.random.default_rng(seed * 1343 + 111_000_269)
        place_names = [o[0] for cat in w.categories[1:] if cat["placement"] for o in cat["objects"]]
        thing_pool = [place_names[int(i)] for i in prng_t.permutation(len(place_names))[:12]]
        cells = sorted(w.pos_at)
        for cat in w.categories[1:]:
            if cat["placement"]:
                continue
            c = cat["room"]
            trng = np.random.default_rng(seed * 1331 + c + 105_000_251)
            tsrng = np.random.default_rng(seed * 1337 + c + 107_000_257)
            room_names = [o[0] for o in cat["objects"]]

            def setting():
                return room_names[int(trng.integers(len(room_names)))], cells[int(trng.integers(len(cells)))]

            def story(C, q, put, take):
                pool = [n_ for n_ in thing_pool + room_names if n_ != C]
                things = [pool[int(i)] for i in trng.permutation(len(pool))[:put]]
                sent = [C, "GA", w.pos_at[q], "GO"]
                for x in things:
                    sent += [x, "GA", w.pos_at[q], C, "ANE", "GO"]
                inside_ = list(things)
                for _ in range(take):
                    x = inside_.pop(int(trng.integers(len(inside_))))
                    r = [p_ for p_ in cells if p_ != q][int(trng.integers(len(cells) - 1))]
                    sent += [x, "GA", w.pos_at[r], "GO"]
                return sent + [C, "ANE", "MYEOT"]

            def cands(n):
                return [NUMS[n - 1]] + [NUMS[m - 1] for m in range(1, 11) if m != n]

            def told(n):
                return NUMS[int(tsrng.integers(10))] if ten_shuffle else NUMS[n - 1]

            def mix():
                return [("held", list(HELD)), ("far", [FAR[int(i)] for i in trng.choice(len(FAR), 3, replace=False)]),
                        ("taught", [TAUGHT[int(i)] for i in trng.choice(len(TAUGHT), 2, replace=False)])]

            ten_items = []
            for n in sorted(int(x) + 1 for x in trng.permutation(10)[:3]):    # heard: 3 count scenes, distinct n (ed. 3)
                C, q = setting()
                ten_items.append(("ten-heard", story(C, q, n, 0) + [told(n)], []))
            for _ in range(2):                      # heard: 2 box stories (edition 3)
                a, b = TAUGHT[int(trng.integers(len(TAUGHT)))]
                C, q = setting()
                ten_items.append(("ten-heard", story(C, q, a, b) + [told(a - b)], []))
            for n in range(1, 11):
                for _ in range(2):
                    C, q = setting()
                    ten_items.append((f"ten count {n}", story(C, q, n, 0), cands(n)))
            for n in range(1, 11):
                ds = [int(m) + 1 for m in trng.permutation(10) if int(m) + 1 != n][:3]
                C, q = setting()
                ten_items.append((f"ten pick {n}", [story(C, q, m, 0) for m in [n] + ds], [NUMS[n - 1]]))
            for n in range(1, 10):
                C, q = setting()
                pool = [n_ for n_ in thing_pool + room_names if n_ != C]
                x = pool[int(trng.integers(len(pool)))]
                ten_items.append((f"ten next {n}", [C, "GA", w.pos_at[q], "GO", C, "ANE", NUMS[n - 1], "GO",
                                                    x, "GA", w.pos_at[q], C, "ANE", "GO", C, "ANE", "MYEOT"], cands(n + 1)))
            for g, pairs in mix():
                for a, b in pairs:
                    C, q = setting()
                    ten_items.append((f"ten box {g}", story(C, q, a, b), cands(a - b)))
            for g, pairs in mix():
                for a, b in pairs:
                    C, q = setting()
                    n_in = a - b
                    wrong = n_in - 1 if n_in == 10 or (n_in > 1 and trng.integers(2)) else n_in + 1
                    ten_items.append((f"voe ten {g}", story(C, q, a, b), [NUMS[n_in - 1], NUMS[wrong - 1]]))
            cat["ten_raw"] = ten_items
    if korean:                                      # the particles: three elements of the textbook category, made last
        prng = np.random.default_rng(seed * 1021 + 6_000_013)
        parts = {name: token(np.clip(prng.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                 for name in ("GA", "ESEO", "REUL")}
        w.particles = list(parts.values())
        w.particle_of = dict(parts)                 # name -> token (step 3b grades spoken sentences with it)
        for cat in w.categories[1:]:
            st = []
            for kind_, sent, ans in cat.get("story", []):
                st.append((kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent], ans))
            cat["story"] = st
            cat["heard"] = cat["heard"] + [x for x in st if x[0] == "story-heard"]
            cat["story"] = [x for x in st if x[0] == "story"]
        if multi:                                   # GO (and then) and EODI (where?), made after the particles
            mrng2 = np.random.default_rng(seed * 1033 + 8_000_029)
            parts.update({name: token(np.clip(mrng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                          for name in ("GO", "EODI")})
            w.particles = list(parts.values())
            w.go, w.eodi = parts["GO"], parts["EODI"]
            if contain:                             # ANE (inside) and YEOPE (beside), made after GO / EODI
                crng3 = np.random.default_rng(seed * 1099 + 19_000_079)
                parts.update({name: token(np.clip(crng3.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                              for name in ("ANE", "YEOPE")})
                w.particles = list(parts.values())
                w.particle_of = dict(parts)
            if count:                               # MYEOT (how many?) and the number words, made last
                nrng2 = np.random.default_rng(seed * 1109 + 21_000_083)
                parts["MYEOT"] = token(np.clip(nrng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, "myeot", "element")
                w.particles = list(parts.values())
                w.particle_of = dict(parts)
                w.numbers = {name: token(np.clip(nrng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                             for name in ("HANA", "DUL", "SET") + (("NET",) if succ else ())}
                parts.update(w.numbers)
            for cat in w.categories[1:]:
                mi = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent], ans)
                      for kind_, sent, ans in cat.get("multi", [])]
                cat["heard"] = cat["heard"] + [x for x in mi if x[0] == "multi-heard"]
                cat["multi"] = [x for x in mi if x[0] != "multi-heard"]
                cat["tell"] = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent], truth)
                               for kind_, sent, truth in cat.get("tell", [])]
                cat["voeperm"] = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent], ans)
                                  for kind_, sent, ans in cat.get("voeperm", [])]
                ci = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent], ans)
                      for kind_, sent, ans in cat.get("contain", [])]
                cat["heard"] = cat["heard"] + [x for x in ci if x[0] == "contain-heard"]
                cat["contain"] = [x for x in ci if x[0].startswith("contain ")]
                cat["voeperm"] = cat["voeperm"] + [x for x in ci if x[0].startswith("voe ")]
                ni = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent],
                       [parts.get(x, x) if isinstance(x, str) else x for x in ans]) for kind_, sent, ans in cat.get("count", [])]
                cat["heard"] = cat["heard"] + [x for x in ni if x[0] == "count-heard"]
                cat["count"] = [x for x in ni if x[0].startswith("count ")]
                cat["voeperm"] = cat["voeperm"] + [x for x in ni if x[0] in ("voe number", "voe four")]
                oi = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent], ans)
                      for kind_, sent, ans in cat.get("objs", [])]
                cat["heard"] = cat["heard"] + [x for x in oi if x[0] == "objs-heard"]
                cat["objs"] = [x for x in oi if x[0] in ("objs hit", "objs free")]
                cat["voe"] = [x for x in oi if x[0].startswith("voe ")]
                gi = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent], ans)
                      for kind_, sent, ans in cat.get("grav", [])]
                cat["heard"] = cat["heard"] + [x for x in gi if x[0] == "grav-heard"]
                cat["grav"] = [x for x in gi if x[0].startswith("grav ")]
                cat["voe"] = cat["voe"] + [x for x in gi if x[0].startswith("voe ")]
        if self_:                                   # D2 self: the action words (verbs: Voyager's doings) and NUGU, last
            srng6 = np.random.default_rng(seed * 1201 + 41_000_131)
            parts["NUGU"] = token(np.clip(srng6.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, "nugu", "element")
            w.particles = list(parts.values())
            w.particle_of = dict(parts)
            w.self_q = parts["NUGU"]
            acts = {}
            for k in range(4):
                acts[f"ACT{k}"] = token(np.clip(srng6.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, f"act{k}", "element")
                w.verb_of[acts[f"ACT{k}"]] = 110 + k
            w.actions = list(acts.values())
            smap = lambda x: parts.get(x, acts.get(x, x)) if isinstance(x, str) else x
            for cat in w.categories[1:]:
                si = [(kind_, [smap(x) for x in sent], ans) for kind_, sent, ans in cat.get("self", [])]
                cat["heard"] = cat["heard"] + [x for x in si if x[0] == "self-heard"]
                cat["self"] = [x for x in si if x[0].startswith("self ")]
        if goal:                                    # D2 goal: the question word EODIRO, last
            grng2 = np.random.default_rng(seed * 1187 + 37_000_119)
            parts["EODIRO"] = token(np.clip(grng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, "eodiro", "element")
            w.particles = list(parts.values())
            w.particle_of = dict(parts)
            w.goal_q = parts["EODIRO"]
            for cat in w.categories[1:]:
                gi = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent], ans)
                      for kind_, sent, ans in cat.get("goal", [])]
                cat["heard"] = cat["heard"] + [x for x in gi if x[0] == "goal-heard"]
                cat["goal"] = [x for x in gi if x[0].startswith("goal ")]
        if views:                                   # step 4+ (D17): the view labels (particles) and the depth words, last
            vrng2 = np.random.default_rng(seed * 1153 + 29_000_107)
            w.view_labels = {name: token(np.clip(vrng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                             for name in ("VFRONT", "VTOP", "VSIDE")}
            if views == 2:                          # (D2 viewpoint: the back label and TURNS, a separate stream, made after)
                vrng3 = np.random.default_rng(seed * 1171 + 33_000_113)
                w.view_labels["VBACK"] = token(np.clip(vrng3.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, "vback", "element")
            parts.update(w.view_labels)
            w.particles = list(parts.values())
            w.particle_of = dict(parts)
            depth = {}
            for i, name in enumerate(("GOES", "COMES")):
                depth[name] = token(np.clip(vrng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                w.verb_of[depth[name]] = 100 + i
            if views == 2:
                depth["TURNS"] = token(np.clip(vrng3.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, "turns", "element")
                w.verb_of[depth["TURNS"]] = 102
            vmap = lambda x: parts.get(x, depth.get(x, x)) if isinstance(x, str) else x
            for cat in w.categories[1:]:
                vi = [(kind_, [vmap(x) for x in sent], [vmap(x) for x in ans]) for kind_, sent, ans in cat.get("views", [])]
                cat["heard"] = cat["heard"] + [x for x in vi if x[0] == "views-heard"]
                cat["views"] = [x for x in vi if x[0].startswith(("views ", "viewpt "))]
        if numline:                                 # step 4+ (D13): the line's cells (own: made here) and the number words, last
            lrng2 = np.random.default_rng(seed * 1129 + 25_000_093)
            if numline == "own":
                plane = lrng2.choice([-1, 1], xy // 2)
                line = {}
                for r in ROLES:
                    for x in range(-HALF, HALF + 1):
                        v = np.full(RAW_DIM, 0.5)
                        v[:xy] = np.concatenate([0.5 + 0.1 * x * su, 0.5 + 0.3 * plane]) + lrng2.normal(0, POS_NOISE, xy)
                        v[xy:WHERE_DIMS] = role_code[r]
                        t = token(np.clip(v, 0, 1), 0, f"{r}line({x})", "element")
                        w.pos_of[t], w.role_of[t] = (x, 100), r          # (100: never next to a room cell)
                        line[(r, x)] = t
                w.line_ids = sorted(line.values())
                cell = {"LA": lambda x: line[("at", x)], "LF": lambda x: line[("from", x)], "LT": lambda x: line[("to", x)]}
            else:
                cell = {"LA": lambda x: w.pos_at[(x, 0)], "LF": lambda x: w.pos_from[(x, 0)], "LT": lambda x: w.pos_to[(x, 0)]}
            w.number_words = [token(np.clip(lrng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, nm, "word")
                              for nm in ("hana", "dul", "set", "net", "daseot")]

            def nl_map(x):
                if not isinstance(x, str):
                    return x
                if x in parts:
                    return parts[x]
                if x[0] == "N":
                    return w.number_words[int(x[1:]) - 1]
                return cell[x[:2]](int(x[2:]) - 3)
            for cat in w.categories[1:]:
                nl = [(kind_, [nl_map(x) for x in sent], [nl_map(x) for x in ans]) for kind_, sent, ans in cat.get("numline", [])]
                cat["heard"] = [(kind_, [nl_map(x) for x in sent], ans) for kind_, sent, ans in cat["heard"]] + \
                    [x for x in nl if x[0] in ("numline-name", "numline-heard")]
                cat["numline"] = [x for x in nl if x[0].startswith("numline ")]
        if tale:                                    # D22: YENNAL (the tale marker, a particle) and each room's tale thing, last
            trng2 = np.random.default_rng(seed * 1223 + 47_000_161)
            parts["YENNAL"] = token(np.clip(trng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, "yennal", "element")
            w.particles = list(parts.values())
            w.particle_of = dict(parts)
            w.tale_mark = parts["YENNAL"]
            w.tale_things = {}
            for cat in w.categories[1:]:
                if not cat.get("tale"):
                    continue
                c_ = cat["room"]
                w.tale_things[c_] = token(np.clip(trng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), c_, f"tale{c_}", "word")
                tmap = lambda x: w.tale_things[c_] if x == "TALEOBJ" else parts.get(x, x) if isinstance(x, str) else x
                ti = [(kind_, [tmap(x) for x in sent], ans) for kind_, sent, ans in cat["tale"]]
                cat["heard"] = cat["heard"] + [x for x in ti if x[0].startswith("tale-heard")]
                cat["tale"] = [x for x in ti if x[0].startswith("tale ")]
            for cat in w.categories[1:]:            # v1b: the preview tales go to the earlier room c - 1 (after the tale things)
                if cat.get("tale_pre"):
                    c_ = cat["room"]
                    tmap = lambda x: w.tale_things[c_] if x == "TALEOBJ" else parts.get(x, x) if isinstance(x, str) else x
                    w.categories[c_ - 1]["heard"] = w.categories[c_ - 1]["heard"] + \
                        [(kind_, [tmap(x) for x in sent], ans) for kind_, sent, ans in cat["tale_pre"]]
    if sp:                                          # W3: WI (on) and ARAE (under), the last elements of all (other worlds unchanged)
        sprng2 = np.random.default_rng(seed * 1277 + 73_000_223)
        parts.update({name: token(np.clip(sprng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                      for name in ("WI", "ARAE")})
        w.particles = list(parts.values())
        w.particle_of = dict(parts)
        w.sp_words = [parts[x] for x in ("WI", "ARAE", "ANE", "YEOPE")]
        for cat in w.categories[1:]:
            si = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent],
                   [parts.get(x, x) if isinstance(x, str) else x for x in ans]) for kind_, sent, ans in cat.pop("sp_raw", [])]
            cat["heard"] = cat["heard"] + [x for x in si if x[0] == "sp-heard"]
            cat["sp"] = [x for x in si if x[0].startswith("sp ")]
    if when:                                        # W4: EONJE (when?) and the moment words, the last elements of all
        wrng2 = np.random.default_rng(seed * 1307 + 89_000_237)
        parts.update({name: token(np.clip(wrng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                      for name in ("EONJE", "CHEOTJJAE", "DULJJAE", "SETJJAE", "NETJJAE")})
        w.particles = list(parts.values())
        w.particle_of = dict(parts)
        w.when_q = parts["EONJE"]
        w.moment_words = [parts[x] for x in ("CHEOTJJAE", "DULJJAE", "SETJJAE", "NETJJAE")]
        for cat in w.categories[1:]:
            wi = [(kind_, [parts.get(x, x) if isinstance(x, str) else x for x in sent],
                   [parts.get(x, x) if isinstance(x, str) else x for x in ans]) for kind_, sent, ans in cat.pop("when_raw", [])]
            cat["heard"] = cat["heard"] + [x for x in wi if x[0] == "when-heard"]
            cat["when"] = [x for x in wi if x[0].startswith("when ")]
    if ten:                                         # W5: the number words NET .. YEOL, the last elements of all
        trng2 = np.random.default_rng(seed * 1339 + 109_000_263)
        more = {name: token(np.clip(trng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element") for name in NUMS[3:]}
        parts.update(more)
        w.numbers = dict(w.numbers, **more)
        w.ten_words = [parts[x] for x in NUMS]

        def conv(s_):
            return [parts.get(x, x) if isinstance(x, str) else x for x in s_]
        for cat in w.categories[1:]:
            ti = [(kind_, [conv(x) for x in sent] if kind_.startswith("ten pick") else conv(sent), conv(ans))
                  for kind_, sent, ans in cat.pop("ten_raw", [])]
            cat["heard"] = cat["heard"] + [x for x in ti if x[0] == "ten-heard"]
            cat["ten"] = [x for x in ti if x[0] != "ten-heard"]
    if bi:                                          # W6: the English-like names and words, the last elements of all
        brng2 = np.random.default_rng(seed * 1367 + 127_000_283)
        w.en_name = {}
        for c in range(1, n_cat):                   # a second name per object: a new word (random features), its room's
            for name, _, _ in w.categories[c]["objects"]:
                w.en_name[int(name)] = token(np.clip(brng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), c,
                                             f"en.{names[name]}", "word")
        parts.update({name: token(np.clip(brng2.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name.lower(), "element")
                      for name in ("FROM", "WHERE", "IS", "WHAT", "MUEOS")})
        w.particles = list(parts.values())
        w.particle_of = dict(parts)
        w.bi_words = {x: parts[x] for x in ("FROM", "WHERE", "IS", "WHAT", "MUEOS", "GO", "EODI", "GA", "ESEO", "REUL")}

        def conv_b(x):
            if isinstance(x, str):
                return w.en_name[int(x[3:])] if x.startswith("EN:") else parts[x]
            return int(x)
        for cat in w.categories[1:]:
            bi_ = [(kind_, [conv_b(x) for x in sent], [conv_b(x) for x in ans]) for kind_, sent, ans in cat.pop("bi_raw", [])]
            cat["heard"] = cat["heard"] + [x for x in bi_ if not x[0].startswith("bi ")]
            cat["bi"] = [x for x in bi_ if x[0].startswith("bi ")]
    if c8:                                          # W7b: the question word JEONE ("before"), the last element of all
        r8b = np.random.default_rng(seed * 1399 + 139_000_291)
        jeone = token(np.clip(r8b.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, "jeone", "element")
        parts["JEONE"] = jeone
        w.particles = list(parts.values())
        w.particle_of = dict(parts)
        w.jeone = jeone
        for cat in w.categories[1:]:
            c8_ = [(kind_, [parts[x] if isinstance(x, str) else int(x) for x in sent], [int(a) for a in ans])
                   for kind_, sent, ans in cat.pop("c8_raw", [])]
            cat["heard"] = cat["heard"] + [x for x in c8_ if x[0] == "c8-before-heard"]
            cat["c8"] = [x for x in c8_ if x[0] != "c8-before-heard"]
    # D72 (2026-10-17, the learning-power test, D72_DESIGN.md; user: every subject, one by one in order): a SUBJECT no part
    # was designed for, put into the base world and heard at four densities, so the learning curve over the times heard is
    # read inside one life (the tests are records only). "cat" (subject 1, a new category rule): four family words, one per
    # sprite family -- one word for every thing of that outline (the shape bias, Landau 1988) -- heard as [drawn picture of
    # the room's thing of that family, WORD] + the room's other pictures (the naming moments' ambiguity), never with a
    # position; tested on NOVEL things of each family (drawn pictures made only for the test, memorised at birth like every
    # element, never in a sentence): picture -> word among the four words, word -> picture among four novel pictures (one
    # per family). "verb" (the designed control -- the world model's motion books): four new motion words with moves no
    # other word has, heard as motion moments [picture, WORD, from(p), to(p + d)] + the room's other pictures (turned with
    # the room, as every motion word); tested in every room on starts never heard with that word there: to(?) among the four
    # new words' arrival cells. Densities: the four words are heard in every room / every 2nd / 4th / 8th room, spread over
    # the life -- in each phase, that share of its rooms drawn at random (a fixed stride would follow the room numbers, and
    # the phases follow them: a stride of 2 put a word in one half of the life only -- found by the seed-0 smoke before any
    # score was read). Which word gets which density and its rooms: a separate stream; "verb" skips the placement rooms,
    # which have no motion. The new elements are the last of all -- without d72 every world is unchanged.
    # Subject 2 and the new control (2026-10-18, the user's decision Q14 "세가지 다 너의 추천대로 해보자." -- finding 162:
    # the "verb" control's premise was wrong, the world model's books roll only questions with two motion words or more):
    # "chain" (the designed control from subject 2 on) = the "verb" world EXACTLY (the same draws, the same heard moments,
    # the same "d72 verb" tests) plus, from a separate stream, CHAIN tests: an object moved by TWO of the new words (never
    # heard together; a pair whose moves cancel is left out), the start given, the end asked among the ends of four pairs
    # from that start -- the world model's books chain the new words' moves learned from experience (the design's part).
    # "neg" (subject 2, a new grammar marker -- negation): four negators (one meaning, four words, at the four densities),
    # heard in the motion rooms as [picture, a motion word of the room, from(p), NEGATOR, to(p)] + the room's other pictures
    # -- "it did not move": the thing stays where it was; tested on starts never heard with that word and negator in the
    # room: to(?) among to(p) (stays, right), the motion word's own end (its usual meaning) and two other motion words' ends.
    # No part reads a negator (the world model's books take every one-verb moment with a from and a to at face value -- a
    # negated moment enters the motion word's book as a move of zero: written before any life as the subject's expected
    # interference, read in R5).
    # Subject 3 (2026-10-18, the user's decision Q15 "그럼 너의 추천대로 하자." -- subjects 3 and 4 before the first edition):
    # "ice" (a new causal rule -- contact with a slippery floor): four floor pictures (one meaning, four looks, at the four
    # densities; drawn sprites made like a new thing's), heard in the motion rooms as [picture, a motion word of the room,
    # from(p), FLOOR, to(p + 2d)] + the room's other pictures -- on the floor the thing slides one step further than the word
    # says (twice its move, turned with the room); tested on starts never heard with that word and floor in the room: to(?)
    # among p + 2d (right), the word's own end p + d (its usual meaning) and two other motion words' ends. No part reads a
    # floor (object properties "+props" and the room scale "+scale" are off in the frozen brain); the world model's books
    # take a floor moment as a move of 2d for the word -- the expected interference, read in R5.
    # Subject 4 (rule fixed 2026-10-18, Q15; built 2026-10-22, the user's decision "응, 너의 의견대로 (가)와 (나)를 순서대로
    # 진행해줘." -- after the order device D80): "next" (a new order rule): four bell words (one meaning, four words, at the
    # four densities); in a room, the bell moment [BELL, picture, at(p)] + the room's other pictures is followed in the very
    # next moment by [picture, from(p), to(p + up)] + other pictures -- no motion word; up turns with the room. The contrast
    # (fixed with the world, before any life and any device for this subject; so that "up" is told only by the bell before
    # it): for every bell pair, three pairs without the bell, [picture, at(p)] + others then [picture, from(p), to(p + d)]
    # with d each of the three other neighbours -- every verbless move goes to each neighbour equally often. A pair's two
    # moments stay together and in order in the room's list (the episode store's order); the pairs are shuffled. Tests: the
    # bell moment as heard, at a cell never heard with that bell in the room, asks to(?) among one up (right), stay, and two
    # other neighbours.
    if d72:
        assert d72 in ("cat", "verb", "chain", "neg", "ice", "next"), d72
        d7 = np.random.default_rng(seed * 1409 + 149_000_301)
        dens = [int(x) for x in d7.permutation([1, 2, 4, 8])]
        rooms7 = [c for c in range(1, n_cat) if not (d72 != "cat" and w.categories[c]["placement"])]
        active = []                                 # per word: its rooms, the share 1 / density of every phase's rooms
        for n_ in dens:
            act = set()
            for ph in sorted({w.categories[c]["phase"] for c in rooms7}):
                rs = [c for c in rooms7 if w.categories[c]["phase"] == ph]
                act |= {int(x) for x in d7.choice(rs, max(1, int(round(len(rs) / n_))), replace=False)}
            active.append(act)
        w.d72, w.d72_dens, w.d72_tests = d72, dens, []
        for cat in w.categories[1:]:
            cat["d72"] = []

        def fill7(objs_, j):
            pool = [int(x) for i_, o in enumerate(objs_) if i_ != j for x in o[1:]]
            return [int(x) for x in d7.choice(pool, int(d7.integers(FILL[0], FILL[1] + 1)), replace=False)]

        def shuffled(s_):
            return [s_[i] for i in d7.permutation(len(s_))]
        if d72 == "cat":
            w.d72_words = [token(np.clip(d7.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, f"catw{f}", "word")
                           for f in range(N_FAMILY)]
            novel = []                              # NOVEL_PER novel things per family: drawn pictures only
            for m in range(NOVEL_PER):
                row = []
                for f in range(N_FAMILY):
                    sprite = fam_outline[f].copy()
                    cells = d7.choice(SPRITE * SPRITE, 4, replace=False)
                    sprite.ravel()[cells] = d7.integers(0, len(GLYPHS), 4)
                    row.append(token(see(sprite, P), 0, f"npic{m}.{f}", "drawn"))
                    w.sprites[row[-1]] = sprite
                novel.append(row)
            w.d72_novel = novel
            for f in range(N_FAMILY):
                for c in rooms7:
                    if c in active[f]:
                        objs_ = w.categories[c]["objects"]
                        w.categories[c]["heard"] = w.categories[c]["heard"] + \
                            [("d72-heard", shuffled([int(objs_[f][1]), w.d72_words[f]] + fill7(objs_, f)), [])]
            for row in novel:
                for f in range(N_FAMILY):
                    others = [g for g in range(N_FAMILY) if g != f]
                    w.d72_tests.append((f"d72 word {f}", [row[f]], [w.d72_words[f]] + [w.d72_words[g] for g in others]))
                    w.d72_tests.append((f"d72 pic {f}", [w.d72_words[f]], [row[f]] + [row[g] for g in others]))
        elif d72 == "neg":
            w.d72_words = [token(np.clip(d7.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, f"neg{j}", "word") for j in range(4)]

            def cell_n():
                return tuple(int(v) for v in d7.integers(-HALF, HALF + 1, 2))
            for c in rooms7:
                cat = w.categories[c]
                fl, objs_ = cat["flipped"], cat["objects"]
                room_v = sorted({int(k_[1:]) for k_, _, _ in cat["heard"] if k_[:1] == "R" and k_[1:].isdigit()})
                heard_n = set()
                for j in range(4):
                    if c not in active[j] or not room_v:
                        continue
                    v = room_v[int(d7.integers(len(room_v)))]
                    p = cell_n()
                    while not inside(move(p, v, fl)):
                        p = cell_n()
                    o = int(d7.integers(N_FAMILY))
                    cat["heard"] = cat["heard"] + \
                        [("d72-heard", shuffled([int(objs_[o][1]), w.verbs[v], w.pos_from[p], w.d72_words[j],
                                                 w.pos_to[p]] + fill7(objs_, o)), [w.pos_to[p]])]
                    heard_n.add((j, v, p))
                for j in range(4):
                    made = 0
                    while made < TEST and room_v:
                        v = room_v[int(d7.integers(len(room_v)))]
                        p = cell_n()
                        others = [u for u in range(len(VERBS)) if u != v and inside(move(p, u, fl))]
                        if not inside(move(p, v, fl)) or len(others) < 2 or (j, v, p) in heard_n:
                            continue
                        two = [others[int(i)] for i in d7.permutation(len(others))[:2]]
                        o = int(d7.integers(N_FAMILY))
                        cat["d72"].append((f"d72 neg {j}", shuffled([int(objs_[o][1]), w.verbs[v], w.pos_from[p],
                                                                     w.d72_words[j]] + fill7(objs_, o)),
                                           [w.pos_to[p], w.pos_to[move(p, v, fl)]] + [w.pos_to[move(p, u, fl)] for u in two]))
                        made += 1
        elif d72 == "ice":
            w.d72_words = []
            for j in range(4):
                sprite = d7.choice(len(GLYPHS), (SPRITE, SPRITE), p=[0.5] + [0.5 / 7] * 7)
                w.d72_words.append(token(see(sprite, P), 0, f"ice{j}", "drawn"))
                w.sprites[w.d72_words[-1]] = sprite

            def twice(p, v, fl):
                return move(move(p, v, fl), v, fl)

            def cell_i():
                return tuple(int(v) for v in d7.integers(-HALF, HALF + 1, 2))
            for c in rooms7:
                cat = w.categories[c]
                fl, objs_ = cat["flipped"], cat["objects"]
                room_v = sorted({int(k_[1:]) for k_, _, _ in cat["heard"] if k_[:1] == "R" and k_[1:].isdigit()})
                heard_i = set()
                for j in range(4):
                    if c not in active[j] or not room_v:
                        continue
                    v = room_v[int(d7.integers(len(room_v)))]
                    p = cell_i()
                    while not inside(twice(p, v, fl)):
                        p = cell_i()
                    o = int(d7.integers(N_FAMILY))
                    cat["heard"] = cat["heard"] + \
                        [("d72-heard", shuffled([int(objs_[o][1]), w.verbs[v], w.pos_from[p], w.d72_words[j],
                                                 w.pos_to[twice(p, v, fl)]] + fill7(objs_, o)), [w.pos_to[twice(p, v, fl)]])]
                    heard_i.add((j, v, p))
                for j in range(4):
                    made = 0
                    while made < TEST and room_v:
                        v = room_v[int(d7.integers(len(room_v)))]
                        p = cell_i()
                        others = [u for u in range(len(VERBS)) if u != v and inside(move(p, u, fl))]
                        if not inside(twice(p, v, fl)) or len(others) < 2 or (j, v, p) in heard_i:
                            continue
                        two = [others[int(i)] for i in d7.permutation(len(others))[:2]]
                        o = int(d7.integers(N_FAMILY))
                        cat["d72"].append((f"d72 ice {j}", shuffled([int(objs_[o][1]), w.verbs[v], w.pos_from[p],
                                                                     w.d72_words[j]] + fill7(objs_, o)),
                                           [w.pos_to[twice(p, v, fl)], w.pos_to[move(p, v, fl)]] +
                                           [w.pos_to[move(p, u, fl)] for u in two]))
                        made += 1
        elif d72 == "next":
            w.d72_words = [token(np.clip(d7.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, f"bell{j}", "word") for j in range(4)]
            UP, OTHER = 1, (0, 2, 3)                # "rises" (one up, turned with the room) and the three other neighbours

            def cell_x():
                return tuple(int(v) for v in d7.integers(-HALF, HALF + 1, 2))

            def inner():
                p = cell_x()
                while not all(inside(move(p, k, fl)) for k in range(4)):
                    p = cell_x()
                return p

            def pair(bell, o, p, k):
                first = ([bell] if bell is not None else []) + [int(objs_[o][1]), w.pos_at[p]]
                q = w.pos_to[move(p, k, fl)]
                return [("d72-heard", shuffled(first + fill7(objs_, o)), []),
                        ("d72-heard", shuffled([int(objs_[o][1]), w.pos_from[p], q] + fill7(objs_, o)), [q])]
            for c in rooms7:
                cat = w.categories[c]
                fl, objs_ = cat["flipped"], cat["objects"]
                pairs, heard_x = [], set()
                for j in range(4):
                    if c not in active[j]:
                        continue
                    p = inner()
                    pairs.append(pair(w.d72_words[j], int(d7.integers(N_FAMILY)), p, UP))
                    heard_x.add((j, p))
                    for k in OTHER:                 # the contrast: no bell before, the next move to each other neighbour
                        pairs.append(pair(None, int(d7.integers(N_FAMILY)), inner(), k))
                for i in d7.permutation(len(pairs)):
                    cat["heard"] = cat["heard"] + pairs[int(i)]
                for j in range(4):
                    made = 0
                    while made < TEST:
                        p = cell_x()
                        others = [k for k in OTHER if inside(move(p, k, fl))]
                        if not inside(move(p, UP, fl)) or len(others) < 2 or (j, p) in heard_x:
                            continue
                        two = [others[int(i)] for i in d7.permutation(len(others))[:2]]
                        o = int(d7.integers(N_FAMILY))
                        cat["d72"].append((f"d72 next {j}", shuffled([w.d72_words[j], int(objs_[o][1]), w.pos_at[p]] +
                                                                     fill7(objs_, o)),
                                           [w.pos_to[move(p, UP, fl)], w.pos_to[p]] + [w.pos_to[move(p, k, fl)] for k in two]))
                        made += 1
        else:
            w.d72_words = []
            for j in range(len(D72_MOVES)):
                w.d72_words.append(token(np.clip(d7.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, f"newv{j}", "element"))
                w.verb_of[w.d72_words[-1]] = 120 + j

            def mv(p, d, fl):
                for _ in range(2 if fl is True else 0 if fl is False else int(fl)):
                    d = (-d[1], d[0])
                return (p[0] + d[0], p[1] + d[1])

            def cell():
                return tuple(int(v) for v in d7.integers(-HALF, HALF + 1, 2))
            for c in rooms7:
                cat = w.categories[c]
                fl, objs_ = cat["flipped"], cat["objects"]
                heard_p = set()
                for j, d in enumerate(D72_MOVES):
                    if c not in active[j]:
                        continue
                    p = cell()
                    while not inside(mv(p, d, fl)):
                        p = cell()
                    o = int(d7.integers(N_FAMILY))
                    q = w.pos_to[mv(p, d, fl)]
                    cat["heard"] = cat["heard"] + \
                        [("d72-heard", shuffled([int(objs_[o][1]), w.d72_words[j], w.pos_from[p], q] + fill7(objs_, o)), [q])]
                    heard_p.add((j, p))
                for j in range(len(D72_MOVES)):
                    made = 0
                    while made < TEST:
                        p = cell()
                        ends = [mv(p, d, fl) for d in D72_MOVES]
                        if not all(inside(e) for e in ends) or (j, p) in heard_p:
                            continue
                        o = int(d7.integers(N_FAMILY))
                        cat["d72"].append((f"d72 verb {j}", shuffled([int(objs_[o][1]), w.d72_words[j], w.pos_from[p]] +
                                                                     fill7(objs_, o)),
                                           [w.pos_to[ends[j]]] + [w.pos_to[ends[g]] for g in range(len(D72_MOVES)) if g != j]))
                        made += 1
            if d72 == "chain":                     # the chain tests: a separate stream, after everything above
                ch7 = np.random.default_rng(seed * 1423 + 151_000_303)
                pairs = [(a, b) for a in range(len(D72_MOVES)) for b in range(a + 1, len(D72_MOVES))
                         if (D72_MOVES[a][0] + D72_MOVES[b][0], D72_MOVES[a][1] + D72_MOVES[b][1]) != (0, 0)]

                def end2(p, a, b, fl):
                    return mv(mv(p, D72_MOVES[a], fl), D72_MOVES[b], fl)
                for c in rooms7:
                    cat = w.categories[c]
                    fl, objs_ = cat["flipped"], cat["objects"]
                    for a, b in pairs:
                        made = 0
                        while made < TEST:
                            p = tuple(int(v) for v in ch7.integers(-HALF, HALF + 1, 2))
                            if not all(inside(q) for q in (mv(p, D72_MOVES[a], fl), mv(p, D72_MOVES[b], fl), end2(p, a, b, fl))):
                                continue
                            alt = [e for e in dict.fromkeys(end2(p, x, y, fl) for x, y in pairs if (x, y) != (a, b))
                                   if inside(e) and e != end2(p, a, b, fl)]
                            if len(alt) < 3:
                                continue
                            alt = [alt[int(i)] for i in ch7.permutation(len(alt))[:3]]
                            o = int(ch7.integers(N_FAMILY))
                            pool = [int(x) for i_, ob in enumerate(objs_) if i_ != o for x in ob[1:]]
                            s_ = [int(objs_[o][1]), w.d72_words[a], w.d72_words[b], w.pos_from[p]] + \
                                [int(x) for x in ch7.choice(pool, int(ch7.integers(FILL[0], FILL[1] + 1)), replace=False)]
                            cat["d72"].append((f"d72 chain {a}{b}", [s_[i] for i in ch7.permutation(len(s_))],
                                               [w.pos_to[end2(p, a, b, fl)]] + [w.pos_to[e] for e in alt]))
                            made += 1
    # Q14 frame test (2026-10-21, the "@nwf" world = the "@nw" world plus a sentence frame; finding 167's correction: in the
    # sealed exam a name is asked INSIDE a sentence frame ("there is ___") never heard with that kind of name, while this
    # world's name tests are frameless -- the guess "the frame's word statistics outvote the scene" is measured here first).
    # Two frame words THERE, IS and N_COMMON common things (a word and its own drawn picture, shared by every room) are made;
    # every room hears N_FRAME frame moments [THERE, IS, common word, its picture, a place] + 1-3 of the room's pictures, the
    # common word drawn Zipf-like (p ~ 1 / rank). The frame is never heard with a room's new names (as person names were never
    # heard after "there is" in the exam). Tests (the room's list "nwf", pairs in this order): for every "@nw" test item that
    # gives the picture and asks the name, "nwb{n}" = that item unchanged and "nwf{n}" = the same item plus THERE, IS; and
    # TEST items per room of the frame's own use, "nwfc" = [THERE, IS, a common picture, a place, fill] -> its word, with
    # "nwbc" = the same without the frame. A separate random stream, after everything above: with the option off every
    # world is unchanged; the room streams are the "@nw" world's.
    # D80 v1 (2026-10-21, the "@seq" world = the "@scene" world plus question moments and tests; D80_DESIGN.md): two
    # question words QWHERE ("where") and QBEFORE ("before"); in every walking room, between the told moves at random
    # times, 2 "now" moments [picture, QWHERE, at(its cell then)] and 2 "before" moments [picture, QBEFORE, QWHERE, at(its
    # cell before its last move then)] -- true at their time, so the question words must be learned from them; the walk
    # begins with the room's scene before any move ("seq start" [picture, at(start)] per thing); tests
    # (cat["seq"]) after the room's whole walk: per thing that moved, "seq now" [picture, QWHERE] -> at(its cell now) and
    # "seq before" [picture, QBEFORE, QWHERE] -> at(its cell before its last move); a thing that never moved: "seq still"
    # (report). A separate random stream after everything else: with the option off every world is unchanged; the rooms
    # are the "@scene" world's plus these moments.
    # D80 v2 (2026-10-22, the "@seq2" world, seq_told = SEQ_TOLD): the same world, but the walk's last SEQ_TOLD moves are
    # told ONLY by their sentence [name, motion word] -- no scene after them (the exam's C8: moves told in words); a thing
    # whose last move was told only is tested as "seq now told" / "seq before told". No new random draw.
    if seq:
        sq = np.random.default_rng(seed * 1447 + 167_000_341)
        qwhere, qbefore = [token(np.clip(sq.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, nm, "word") for nm in ("qwhere", "qbefore")]
        w.seq_words = (qwhere, qbefore)

        def states(moves, starts, upto):
            now, before = list(starts), [None] * len(starts)
            for j, _, a, b in moves[:upto + 1]:
                before[j], now[j] = a, b
            return now, before
        for c, (first, glen, moves, starts) in seq_rooms.items():
            cat = w.categories[c]
            objs_ = cat["objects"]
            teach = []                              # (after move i, moment)
            for kind_ in ("now", "now", "before", "before"):
                for _ in range(100):
                    i = int(sq.integers(len(moves)))
                    j = int(sq.integers(len(objs_)))
                    now, before = states(moves, starts, i)
                    if kind_ == "before" and before[j] is None:
                        continue
                    if kind_ == "now":
                        m = [int(objs_[j][1]), qwhere, w.pos_at[now[j]]]
                    else:
                        m = [int(objs_[j][1]), qbefore, qwhere, w.pos_at[before[j]]]
                    teach.append((i, [m[t] for t in sq.permutation(len(m))]))
                    break
            heard = cat["heard"]
            ends = [first + sum(glen[:i + 1]) for i in range(len(glen))]     # the index after move i's scene
            # the walk begins with the room's scene before any move (every thing at its start cell, as the scenes after)
            new, at_ = heard[:first] + [("seq start", [int(o[1]), w.pos_at[p_]], []) for o, p_ in zip(objs_, starts)], first
            for i, end in enumerate(ends):
                # D80 v2: the walk's last seq_told moves are told only -- the sentence, no scene after it
                new += heard[at_:end] if i < len(moves) - seq_told else heard[at_:at_ + 1]
                at_ = end
                new += [("seq-heard", m, []) for ii, m in teach if ii == i]
            cat["heard"] = new + heard[at_:]
            now, before = states(moves, starts, len(moves) - 1)
            items = []
            for j in range(len(objs_)):
                pic = int(objs_[j][1])
                if before[j] is None:
                    items.append(("seq still", [pic, qwhere], [w.pos_at[now[j]]]))
                    continue
                last_j = max(i for i, mv_ in enumerate(moves) if mv_[0] == j)
                tl = " told" if last_j >= len(moves) - seq_told else ""         # its last move told only (D80 v2)
                items.append(("seq now" + tl, [pic, qwhere], [w.pos_at[now[j]]]))
                items.append(("seq before" + tl, [pic, qbefore, qwhere], [w.pos_at[before[j]]]))
            cat["seq"] = items
    if nwf:
        fr = np.random.default_rng(seed * 1439 + 163_000_327)
        frame_w = [token(np.clip(fr.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, name, "word") for name in ("there", "is")]
        common, common_pic = [], []
        for k in range(N_COMMON):
            common.append(token(np.clip(fr.uniform(0.2, 0.8, RAW_DIM), 0, 1), 0, f"cw{k}", "word"))
            sprite = fr.choice(len(GLYPHS), (SPRITE, SPRITE), p=[0.5] + [0.5 / 7] * 7)
            common_pic.append(token(see(sprite, P), 0, f"cwpic{k}", "drawn"))
            w.sprites[common_pic[-1]] = sprite
        zipf = np.array([1.0 / (k + 1) for k in range(N_COMMON)])
        zipf /= zipf.sum()
        w.nwf_frame, w.nwf_common, w.nwf_common_pic = frame_w, common, common_pic
        cells = sorted(w.pos_at)

        def fill_f(pool):
            return [int(x) for x in fr.choice(pool, int(fr.integers(FILL[0], FILL[1] + 1)), replace=False)]

        def shuffled_f(s_):
            return [int(s_[i]) for i in fr.permutation(len(s_))]
        for c in range(1, n_cat):
            cat = w.categories[c]
            pool = [int(x) for o in cat["objects"] for x in o[1:]]
            frame_heard = []
            for _ in range(N_FRAME):
                k = int(fr.choice(N_COMMON, p=zipf))
                p = w.pos_at[cells[int(fr.integers(len(cells)))]]
                frame_heard.append(("frame", shuffled_f(frame_w + [common[k], common_pic[k], int(p)] + fill_f(pool)), []))
            cat["heard"] = cat["heard"] + frame_heard
            room_names = {int(o[0]) for o in cat["objects"]}
            items = []
            for k_, s_, ans in cat["nw"]:
                if int(ans[0]) in room_names:            # the picture given, its name asked
                    items.append(("nwb" + k_[2:], list(s_), list(ans)))
                    items.append(("nwf" + k_[2:], shuffled_f(list(s_) + frame_w), list(ans)))
            for _ in range(TEST):
                k = int(fr.choice(N_COMMON, p=zipf))
                p = w.pos_at[cells[int(fr.integers(len(cells)))]]
                bare = shuffled_f([common_pic[k], int(p)] + fill_f(pool))
                items.append(("nwbc", bare, [common[k]]))
                items.append(("nwfc", shuffled_f(bare + frame_w), [common[k]]))
            cat["nwf"] = items
    w.vecs = np.array(vecs, dtype=np.float32)
    w.cats = np.array(cats)
    w.names, w.origin = names, origin
    # the channel tag each element arrives with (memory-first rule 3: tag inputs at arrival): where (position), a motion word,
    # "na", an object's name, a drawn picture, a converted picture -- the brain's "where" and "what" streams split at the senses
    kinds = ["at", "from", "to", "verb", "na", "name", "drawn", "converted"] + (["particle"] if korean else []) + \
        (["number"] if count else [])
    w.kinds = kinds
    # the sense channel of every element: "where" (positions, any role), "word" (names, motion words, na), "picture"
    w.channels = ["where", "word", "picture"]
    part_set = set(getattr(w, "particles", []))
    num_set = set(getattr(w, "numbers", {}).values())
    w.kind_of = np.array([kinds.index(w.role_of[t]) if t in w.pos_of else kinds.index("verb") if t in w.verb_of else
                          kinds.index("particle") if t in part_set else kinds.index("number") if t in num_set else
                          kinds.index("na") if t == na else kinds.index({"word": "name"}.get(o, o))
                          for t, o in enumerate(origin)])
    w.channel_of = np.array([0 if t in w.pos_of else 2 if o in ("drawn", "converted") else 1 for t, o in enumerate(origin)])
    w.V, w.n_cat = len(vecs), n_cat
    rooms = range(1, n_cat)
    w.first_half = [0] + [c for c in rooms if w.categories[c]["phase"] == 0]
    w.second_half = [c for c in rooms if w.categories[c]["phase"] > 0]
    w.phases = [w.first_half, [c for c in rooms if w.categories[c]["phase"] == 1],
                [c for c in rooms if w.categories[c]["phase"] == 2]]
    if curriculum:                                  # placement rooms first, then the textbook and the motion rooms
        place = [c for c in rooms if w.categories[c]["placement"]]
        w.phases = [place, [c for c in w.first_half if c not in place]] + w.phases[1:]
    return w


def element_check(w, memory, seed, linear_pos=None):
    """memory-first rule 2: every element recalled from memory (Module A's frozen decoder) and recognised -- the nearest
    element to the recalled features is the element itself. Reported by element type."""
    import torch
    from run_seed_transfer import LATENT
    g = torch.Generator().manual_seed(10_000 + seed)
    Wd = torch.randn(LATENT, RAW_DIM, generator=g) / np.sqrt(LATENT)
    rec = torch.sigmoid(memory @ Wd).numpy()
    if linear_pos is not None and linear_pos is not False:   # the "where" channel through its own decoder (pass it)
        pi = np.array(sorted(w.pos_of))
        rec[pi] = 0.5
        rec[pi, :w.where_dims] = (0.5 + memory[pi] @ linear_pos).numpy()
    d = ((rec[:, None, :] - w.vecs[None, :, :]) ** 2).sum(-1)
    ok = d.argmin(1) == np.arange(w.V)
    out = {}
    for src in sorted(set(w.origin)):
        m = np.array([o == src for o in w.origin])
        out[src] = float(ok[m].mean())
    out["all"] = float(ok.mean())
    return out


if __name__ == "__main__":
    w = build(0)
    n_heard = sum(len(c["heard"]) for c in w.categories)
    print(f"{w.V} elements, {w.n_cat} categories (textbook + {w.n_cat - 1} rooms), {n_heard} moments; phases "
          f"{[len(p) for p in w.phases]}")
    objs = w.categories[1]["objects"]
    for name, drawn, conv in objs[:2]:
        print(f"\n{w.names[name]}  drawn / converted")
        for a, b in zip(show(w.sprites[drawn]).split("\n"), show(w.sprites[conv]).split("\n")):
            print(f"  {a}   {b}")
    for kind, s, ans in w.categories[1]["heard"][:3] + [x for x in w.categories[1]["heard"] if x[0] == "R0"][:1]:
        print(kind, [w.names[t] for t in s], "->", [w.names[t] for t in ans])
