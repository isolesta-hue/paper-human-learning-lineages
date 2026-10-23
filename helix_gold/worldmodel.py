"""World-model layer, stage A (2026-10-06, the user's design: a world model built from mathematics as its raw material -- a
library of formulas / procedures, README "④ 월드모델 층 설계 v1").

What the bag-of-words body cannot do: chain two one-step rules it only ever met apart ("pulled, then rises: where does it
end?"). A world model keeps an explicit STATE and rolls it forward one book at a time.

  books       -- per motion word (cue), the moves the learner has met first-hand: from each remembered first-hand moment
                 with exactly one motion word, one "from" position and one "to" position (its whole-sentence memory, source
                 "world"), the move = memory[to] - memory[from] in the learner's ORIGINAL memory (fixed, linear for
                 positions -- finding 90). Moves that coincide (closer than half the smallest gap between two "to" cells)
                 are one book entry with a count and the set of rooms it was heard in (as the cued templates, finding 91).
  role shift  -- a move from "from" to "to" also carries the role marker; the pure move is move - shift, shift = mean of the
                 "to" positions' memory - mean of the "from" positions' memory (the same grid under both markers).
  rollout     -- a question with one "from" position and two or more motion words: state = memory[from]; for each word, add
                 its book's pure move (the entry confirmed in the question's room if any, else the most counted); then the
                 role shift. The state lands on a "to" cell or between cells.
  answer part -- the nearest "to" cell gets evidence log(n_known) -- the template units of finding 55 (log(1 + r (n - 1)),
                 n = the words the learner knows, as the cued templates) for one fully decided candidate (r = 1) -- only
                 when the state lands within half a cell gap of it (decided); otherwise nothing. (First smoke: log(n_to),
                 the 25 "to" cells, was outvoted -- chain 0.10 -- although the rollout alone is right on 69 of 69.)
                 Each book applied carries its own unit (k decided steps = k log(n_known), as agreeing pieces add in the
                 consensus memory, finding 64): second smoke with one unit -- chain 0.33 at the end -- the cued templates
                 give the same unit to each ONE-step end (the intermediate states), so one unit only ties them.
                 Added to the answer score as its own part (the split of finding 112 stays exact).
Learned from the learner's own record (the books) and its own memory; the channel tags (from / to / motion word) arrive with
the elements (memory-first rule 3). No constant beyond the declared ones above.
"""
import copy
import math

import numpy as np
import torch



class _FrameIdent:
    """the identity frame (D11: picklable, unlike a lambda)."""

    def __call__(self, m):
        return m


class _FrameMap:
    """a frame as a map on moves: V @ (pinv @ m) -- exactly the lambda it replaces (D11: picklable)."""

    def __init__(self, V, pinv):
        self.V, self.pinv = V, pinv

    def __call__(self, m):
        return self.V @ (self.pinv @ m)

# D69 (2026-10-16, the user after finding 159: "다 켜기는 폐기하자. 대신 너가 낸 새 연구 후보로 한번 실험을 해보는게 좋을거
# 같아"): the parts a self-trust gate may quiet -- (condition flag, attribute). The tracker's trust rule (trusted(): a recipe
# speaks only when it has predicted the heard stories right more often than not) widened to every part. "+wmlast" is not
# here: it orders the world model against the body, it does not change what the world model predicts.
GATED = (("wmctx", "context"), ("wmrev", "reverse"), ("bind", "bind"), ("track", "track"), ("props", "props"),
         ("grav", "grav"), ("wmframe", "frame"), ("contain", "contain"), ("count", "count"), ("succ", "succ"),
         ("scale", "scale"), ("views", "views"), ("vturn", "vturn"), ("goal", "goal"), ("self", "self_part"))


class WorldBooks:
    def __init__(self, learner):
        w = learner.w
        self.learner = learner
        self.mem = learner.memory.detach().clone()          # the original memory (never changes; checksum-guarded)
        kinds = list(w.kinds)
        self.kind_of = np.asarray(w.kind_of)
        self.k_from, self.k_to, self.k_verb = kinds.index("from"), kinds.index("to"), kinds.index("verb")
        self.to_ids = torch.tensor(np.flatnonzero(self.kind_of == self.k_to))
        self.from_ids = torch.tensor(np.flatnonzero(self.kind_of == self.k_from))
        self.place = np.where(np.asarray(w.cats) > 0, np.asarray(w.cats), -1)
        # step 4+ (D13, "@numown"): a number line with its own (finer) spacing -- its cells get their own half gap; the
        # rooms' half gap is taken over the rooms' cells only (identical in every world without such a line)
        self.line_set = set(int(t) for t in getattr(w, "line_ids", []))
        room_to = torch.tensor([int(t) for t in self.to_ids if int(t) not in self.line_set])
        to_mem = self.mem[room_to]
        d = torch.cdist(to_mem, to_mem)
        d.fill_diagonal_(float("inf"))
        self.half_gap = float(d.min()) / 2                  # half the smallest gap between two "to" cells
        self.line_to = set(int(t) for t in self.to_ids if int(t) in self.line_set)
        self.line_half = None
        if self.line_to:
            lm = self.mem[torch.tensor(sorted(self.line_to))]
            dl = torch.cdist(lm, lm)
            dl.fill_diagonal_(float("inf"))
            self.line_half = float(dl.min()) / 2
        self.shift = self.mem[self.to_ids].mean(0) - self.mem[self.from_ids].mean(0)
        self.n_rows, self.books = -1, {}
        # D22 (2026-10-10, frames vs reality; ledger D22, D22_DESIGN.md): tale_mode None = as before. "one" ("+tk1"): a heard
        # moment with the tale marker goes to the TALE book, not the reality book; a marked question is answered in the tale
        # frame (the tale book's move where it has one -- the exception -- else the reality book: anchor + exception).
        # "anchor" ("+tkA"): as "one", and an unmarked moment is ALSO taken as a tale when it runs against the reality
        # anchor -- its move points against the usual move of its word in the other rooms while the room itself is not
        # turned (the room's other words keep their usual direction; a room with no other words is not taken as turned):
        # several tags at once (the marker, the anchor, the room's frame). "ignore" ("+tkig"): marked moments are dropped
        # and marked questions answered as real.
        self.tale_mode, self.tale_mark, self.tale_books, self.tale_flagged = None, None, {}, 0
        # stage B (2026-10-07, "+wmctx"): a room's CONTEXT -- when the room's own books for the motion words heard there run
        # against the learner's usual books (the most counted entry: the surprise of meeting "falls" going up), the room is
        # taken to transform every move the same way, and a word never heard in the room gets its usual move transformed
        # likewise. The transformations tried: keep or reverse (the sign of the agreement, the majority over the room's
        # words) -- the smallest library of rules about rules; off (stage A) unless set.
        self.context = False
        # finding 123 (2026-10-07, the user: Korean first): sentences -> scenes through PARTICLES. "+bind": each particle is
        # bound to the element right before it (a Korean noun phrase is noun + particle; the phrases themselves may come in
        # any order) -- the smallest piece of word order. Which particle marks the START is learned from the learner's own
        # ordered record of heard stories (the bound noun's place, from its naming moments, is where the scene starts);
        # the start state is that place, then the motion books roll as in stage A. "+bindrand": the same, but each particle
        # is bound to a random noun of the sentence (control: the bag of words knows the particles, not their neighbours).
        self.bind, self.bind_random = False, False
        self.k_particle = kinds.index("particle") if "particle" in kinds else None
        self.k_name, self.k_at = kinds.index("name"), kinds.index("at")
        self.at_ids = torch.tensor(np.flatnonzero(self.kind_of == self.k_at))
        self.shift_at = self.mem[self.from_ids].mean(0) - self.mem[self.at_ids].mean(0)
        self.n_log, self.start_particle, self.noun_at = -1, None, {}
        # stage-1 path step 2 (2026-10-08, "+track"): multi-moment stories -- STATE OVER TIME, one state per object. The
        # story is cut into moments at the connective (GO); in each moment the noun bound to the start particle (learned
        # above) is the landmark, the noun bound to any other particle is the mover; a landmark puts the mover at the
        # landmark's place, a moment without one goes on from where the mover is now; the moment's motion words roll the
        # MOVER's state only. The question moment (with EODI) names the asked object. The procedure is given (a recipe),
        # its TRUST is earned: it speaks only when it has predicted the heard stories' scenes right more often than not
        # (Laplace; predicted before the scene is used -- the user's "predict first, compare after").
        # "+trackone" (control): one state for the whole story -- a landmark resets it, every motion word moves it.
        self.track, self.track_one = False, False
        self.go_tok, self.eodi_tok = getattr(w, "go", None), getattr(w, "eodi", None)
        self.track_key, self.track_trust = None, (0, 0)
        # stage-1 path step 3a (2026-10-08, "+wmrev"): the books read BACKWARDS -- a question with one "from" and one "to"
        # position and no motion word: the seen move (to - from - role shift) is looked up among the motion words' books
        # (the room's own entry if any, else the most counted -- as forwards); the one word whose move lands within half a
        # cell gap gets evidence log(n_known) (one decided book, the stage-A unit); none or several: nothing. No trust
        # check: the books were built from the very moments a check could use (it would grade itself).
        self.reverse = False
        # stage-1 path step 4 (2026-10-08, "+props"; principle from New Horizons NH 56: what is shared is not learned per
        # object, what is learned per object is not shared): the motion books stay shared; each OBJECT gets its own entry in a
        # property table -- does it get pushed (light) or does it stop the mover (heavy) -- learned one-shot from heard stories
        # with a collision: the story is read twice, the collided object taken as light, then as heavy; when exactly one
        # reading lands on the scene, that reading is the object's entry (majority over its stories). While tracking, a mover
        # that lands on another object's cell meets that object's entry: light -- the object moves by the same move; heavy --
        # the mover goes back; no entry -- the story is not answered. The rule's shape (pushed = the same move; heavy = stay)
        # is given, as NH 56 (NH 70 learns it from a library -- later). "+propsrand" (control): each object reads ANOTHER
        # object's entry (a fixed rotation of the table's names). Stories may also place a mover by a position token ("[B ga]
        # at(q)": B is at q; "[A ga] from(p) VERB": A moves from p) -- multi-moment stories without them are read as before.
        self.props, self.props_rand, self._force, self._hits = False, False, None, []
        self.props_key, self.prop_table = None, {}
        # stage-1 path step 4+ (2026-10-08, "+grav"; core knowledge: gravity / support): some motion words do not stop after
        # one book move -- they repeat it UNTIL the next move would leave the world or enter another object's cell (falling
        # until it rests). Which words do so is learned per word from heard stories: each read twice, the word moving once,
        # then until blocked; the reading that lands on the scene is the word's vote (majority). "+gravall" (control): every
        # motion word repeats (the over-generalisation a child goes through -- "goed"). Without either: one move, as before.
        self.grav, self.grav_all, self._until_force = False, False, None
        self.until_key, self.until_of = None, {}
        # stage-1 path step 4+ (2026-10-08, "+wmframe"; the user: the centre of a baby's world rules is the GROUND -- a room's
        # frame explains every motion word at once): stage B's keep / reverse becomes the room's FRAME -- the symmetries of
        # the square (quarter turns and mirrorings; 8). The learner's two most nearly orthogonal usual books are its axes;
        # a frame sends each axis to plus or minus one of them; the room's frame is the one that best maps the usual books of
        # the words heard in the room onto the room's own books (keep on ties); a word never heard there gets its usual move
        # through the frame. The library of frames is given (as stage B's keep / reverse); which one a room has is learned.
        self.frame, self.frame_cache = False, {}
        # stage-1 path step 4+ (2026-10-08, "+contain"; core knowledge: containment): particles other than the start
        # particle and the mover particle (the one most often bound to a story's names) are RELATION particles -- the noun
        # bound to one is the relatum, not a mover ("[A ga] at(r) [C ane]": A is at r, in relation ANE to C). Whether a relation
        # CARRIES (when C moves, A moves the same way) is learned per particle from heard stories, each read twice (carried /
        # not); the reading that lands on the scene votes. Without "+contain" no relation carries; "+containall" (control):
        # every relation carries.
        self.contain, self.contain_all, self._rel_force = False, False, None
        self.rel_key, self.rel_parts, self.rel_effect, self.mover_particle = None, set(), {}, None
        # stage-1 path step 4+ (2026-10-09, "+count"; core knowledge: small numbers): a story may end "[C REL] Q" (Q a particle
        # no noun is bound to -- "how many in C?"); the tracking keeps, for every object, its latest relation (C, REL) -- set
        # when it is put in relation to C, cleared when it is placed elsewhere -- and the count is the number of objects in
        # that relation to C and at C's cell. Which NUMBER WORD names each count is learned from heard stories (their last
        # element; majority). "+countrand" (control): the table's words rotated (each count read with another count's word).
        self.count, self.count_rand = False, False
        self.k_number = kinds.index("number") if "number" in kinds else None
        self.num_key, self.num_table = None, {}
        # step 4+ (2026-10-09, "+succ"; ledger D1a -- the successor rule): the counting chant (heard sentences made only of
        # number words, in order) gives each word its NEXT word; a count with no word heard for it is named by the next
        # word after the word for one fewer ("one more = the next word") -- Gelman & Gallistel's cardinal and successor
        # principles. Without it a count never heard has no word.
        self.succ, self.num_next = False, {}
        self.succ_rev = False                       # control "+succrev": the chant read backwards (the order should carry it)
        # step 4+ (2026-10-09, "+scale"; ledger D15 -- the user: frames of different scale must be matched, and matching them
        # is itself to be learned): a room whose own books for the words heard there are a SCALED copy of the usual books
        # (the same word moves half as far on a finer number line) gets the ratio from the learner's own record -- per word
        # heard in the room, (local . usual) / (usual . usual), the median over the words -- and a word never heard in the
        # room gets its usual move times that ratio (stage B's keep / reverse is the ratio +1 / -1; this is its continuous
        # form). Control "+scalerev": the inverse ratio (a wrong alignment of the same size of change).
        self.scale, self.scale_rev, self.scale_cache = False, False, {}
        # step 4+ (2026-10-09, "+views"; ledger D17 v1 -- the three-view world): every picture is an image on the same image
        # cells; a view label follows each cell. Image coordinates of a cell = its memory against the two most orthogonal
        # usual moves (the learner's own books; per role, centred). From the heard three-view moments the learner finds, for
        # each axis of each view, which axis of which OTHER view always shows the same number (and with which sign) -- the
        # axis matching between frames (D15's scale alignment, one step on); from the heard depth moves, each view's
        # displacement per motion word. Fill: the asked view's coordinates from the two given views through the matched axes;
        # depth: the asked view's start plus that view's displacement. Control "+viewsrand": each axis matched to the OTHER
        # axis of the same source view (a wrong correspondence of the same kind). Assumes (places to change when extending):
        # the view labels are given as tags; three orthogonal views; image grid = the rooms' 5 x 5.
        self.views, self.views_rand, self.view_key, self.view_tables = False, False, None, None
        # D2 viewpoint (2026-10-09, "+vturn"): a motion word's effect on each view axis learned as "after = sign x (some axis
        # of some view before) + offset" -- the best-agreeing source over the heard moves, the offset their most common one;
        # a shift (GOES) is the same axis with sign + and an offset, a quarter turn (TURNS) trades axes between views. Off:
        # the per-view displacement of "+views" alone (a turn cannot be written as a shift).
        self.vturn = False
        # D2 goal (2026-10-09, "+goal"; Gergely & Csibra's teleological stance -- a core-knowledge principle, so it is a
        # HUMAN-GIVEN part: agents move efficiently toward a goal): in a story where things are named with their cells and
        # the first-named thing (the agent) then takes steps, the agent's place is rolled with the learner's own books (image
        # coordinates as in "+views"); the goal = the other thing whose distance shrank the most (unique, > 0); "EODIRO" asks
        # it; otherwise the next step asked = the motion word that brings the agent closest to that goal (unique). Control
        # "+goalrand": the other thing taken as the goal. Assumes: one agent named first, one-cell steps, city-block distance,
        # no obstacles (the detour test is v2).
        self.goal, self.goal_rand = False, False
        self.goal_q = getattr(w, "goal_q", None)
        # D2 self (2026-10-10, "+self"; contingency -- Watson 1972, Bahrick & Watson 1985): the learner's own action words
        # get their moves from its books (heard: an action and the move it made); in a question, the thing whose seen move
        # follows the action EVERY time -- through one and the same symmetry of the square (the room-frame library of
        # finding 131: a mirror is one of them) -- is "me". Control "+selfid": the identity only (no mirror). Assumes: one
        # "me" per room, one-cell moves, the 8 square symmetries as the library (HUMAN-GIVEN, as in "+wmframe").
        self.self_part, self.self_id = False, False
        self.self_q = getattr(w, "self_q", None)
        self.actions = set(int(t) for t in getattr(w, "actions", []))
        self.view_tok = set(int(t) for t in getattr(w, "view_labels", {}).values())
        # D69 "+gate": each part's record on the heard material -- every sentence of a session is predicted (context = all
        # but its last element, answer = the last element) once the session has been taken in (edition 3; the tracker's
        # rule), by the world model as it speaks now and by the same with that part added (a quiet part) or taken out (a
        # speaking part); where the two predictions differ, the record counts whether the part's prediction was the heard
        # one (edition 2, see grade_parts). A part speaks when (right + 1) / (differed + 2) > 1/2 (Laplace, the tracker's
        # rule); no record = quiet (a part that never changes a prediction on this life's own experience has not shown it belongs here). Off
        # unless set: nothing is graded and nothing is quieted.
        self.gate, self.part_trust = False, {}
        # D83 (1) "+useful" (2026-10-23; Toolformer's rule, Schick et al. 2023 -- a call is kept where it made the heard
        # continuation more likely than the same context without it): each part's usefulness is counted PER QUESTION FORM
        # (which element kinds the question holds) on all heard material, against the world model as it speaks now in that
        # form, scored +1 right / 0 silent / -1 wrong over the cells the kind gate lets answer (grade_use, edition 3); when
        # answering, a part speaks in a form where it has helped there (net > 0); a form where it never changed anything
        # falls back to its all-forms decision; no decision at all = quiet.
        # use_mode "quiet" (every part quiet) and "shuf" (each part reads another part's record) are the reference gates.
        self.useful, self.use_mode, self.use_rec, self.use_all, self.use_views, self.use_count = False, None, {}, {}, {}, {}
        self.use_perms, self.use_dec, self.use_dec_all, self.use_heard = {}, {}, {}, []

    def __deepcopy__(self, memo):
        """a branch (Learner3.clone deep-copies the model) shares the fixed memory and tags; the clone rebinds .learner to
        the branch and the books are rebuilt from the branch's own whole-sentence memory."""
        c = object.__new__(WorldBooks)
        c.__dict__.update(self.__dict__)
        c.learner, c.n_rows, c.books, c.n_log = None, -1, {}, -1
        c.track_key, c.props_key, c.prop_table = None, None, {}
        c.until_key, c.until_of = None, {}
        c.frame_cache = {}
        c.rel_key, c.rel_parts, c.rel_effect = None, set(), {}
        c.num_key, c.num_table, c.num_next = None, {}, {}
        c.scale_cache = {}
        c.view_key, c.view_tables = None, None
        c.tale_books = {}
        c.part_trust = dict(self.part_trust)            # D69: a branch grades into its own copy of the record
        c.use_rec = {k: list(v) for k, v in self.use_rec.items()}     # D83 (1): the same for the usefulness record
        c.use_all = {k: list(v) for k, v in self.use_all.items()}
        c.use_count, c.use_views, c.use_perms = dict(self.use_count), {}, {}
        c.use_dec, c.use_dec_all, c.use_heard = dict(self.use_dec), dict(self.use_dec_all), list(self.use_heard)
        return c

    def refresh(self):
        """rebuild the books when the whole-sentence memory has grown (first-hand rows only)."""
        ch = self.learner.chunks
        if len(ch.rows) == self.n_rows:
            return
        self.n_rows = len(ch.rows)
        books, rows = {}, []
        for key, i in ch.index.items():
            if ch.source[i] != "world":
                continue
            ks = self.kind_of[list(key)]
            vs, fs, ts = [x for x, k in zip(key, ks) if k == self.k_verb], [x for x, k in zip(key, ks) if k == self.k_from], \
                [x for x, k in zip(key, ks) if k == self.k_to]
            if len(vs) != 1 or len(fs) != 1 or len(ts) != 1:
                continue
            move = self.mem[ts[0]] - self.mem[fs[0]] - self.shift
            rooms = {int(self.place[x]) for x in key if self.place[x] >= 0}
            gap = self.line_half if self.line_set and any(int(x) in self.line_set for x in key) else self.half_gap
            if self.tale_mode is not None:
                rows.append((vs[0], move, rooms, gap, self.tale_mark in key))
                continue
            self._book_add(books, vs[0], move, rooms, gap)
        if self.tale_mode is not None:
            books, self.tale_books = self._frame_books(rows)
        self.books = books

    @staticmethod
    def _book_add(books, v, move, rooms, gap):
        entries = books.setdefault(v, [])
        for e in entries:
            if float((e[0] - move).norm()) < gap:
                e[1] += 1
                e[2] |= rooms
                break
        else:
            entries.append([move, 1, set(rooms)])

    def _frame_books(self, rows):
        """D22: (reality books, tale books) from the first-hand moments (verb, move, rooms, gap, marked)."""
        real, tale = {}, {}
        for v, move, rooms, gap, marked in rows:
            if marked and self.tale_mode in ("one", "anchor"):
                self._book_add(tale, v, move, rooms, gap)
            elif not marked:
                self._book_add(real, v, move, rooms, gap)
        if self.tale_mode != "anchor":
            return real, tale

        def usual(books, v, rooms):                 # the anchor: the most counted move of v (reality's majority; an entry
            es = books.get(v, [])                   # is shared by every room that heard it, so none is left out)
            return max(es, key=lambda e_: e_[1])[0] if es else None

        def turned(books, v, rooms):                # the room's other words against their usual moves (majority)
            votes = 0
            for u, es in books.items():
                here = [e for e in es if e[2] & rooms]
                if u == v or not here:
                    continue
                a = usual(books, u, rooms)
                if a is not None:
                    votes += 1 if float(max(here, key=lambda e_: e_[1])[0] @ a) < 0 else -1
            return votes > 0
        flagged = set()
        for j, (v, move, rooms, gap, marked) in enumerate(rows):
            if marked or not rooms:
                continue
            a = usual(real, v, rooms)
            if a is not None and float(move @ a) < 0 and not turned(real, v, rooms):
                flagged.add(j)
        self.tale_flagged = len(flagged)
        if not flagged:
            return real, tale
        real = {}
        for j, (v, move, rooms, gap, marked) in enumerate(rows):
            if marked or j in flagged:
                self._book_add(tale, v, move, rooms, gap)
            else:
                self._book_add(real, v, move, rooms, gap)
        return real, tale

    def tale_frame(self):
        """D22: the books a marked question is answered with -- ANCHOR + EXCEPTION: a word the tales tell about keeps its
        reality move (the anchor's one step) times the tales' agreement with it (keep or reverse, the count-weighted sign;
        the tale moments only hold whole falls of any length, so their raw moves are not one step); every other word as
        in reality."""
        frame = dict(self.books)
        for v, es in self.tale_books.items():
            real = self.books.get(v)
            if not es or not real:
                continue
            anchor = max(real, key=lambda e_: e_[1])[0]
            agree = sum(e[1] * (1 if float(e[0] @ anchor) >= 0 else -1) for e in es)
            frame[v] = [[anchor * (1.0 if agree >= 0 else -1.0), sum(e[1] for e in es), set()]]
        return frame

    def learn_particles(self):
        """finding 123: from the learner's ordered first-hand log -- where each noun was named (its "at" place, the most
        heard), and which particle's bound noun is where heard stories start (nearest "from" cell to the noun's place)."""
        log = getattr(self.learner, "order_log", [])
        if len(log) == self.n_log:
            return
        self.n_log = len(log)
        at_counts = {}
        for sent in log:
            ks = self.kind_of[sent]
            ats = [x for x, k in zip(sent, ks) if k == self.k_at]
            # naming moments only: one "at" and no particle (step 4's first smoke: "[B ga] at(q)" -- where B is NOW, not where
            # it was named -- was counted as naming and moved the landmarks: one-moment stories 1.00 -> 0.79)
            if len(ats) == 1 and (self.k_particle is None or self.k_particle not in ks):
                for x, k in zip(sent, ks):
                    if k == self.k_name:
                        d = at_counts.setdefault(x, {})
                        d[ats[0]] = d.get(ats[0], 0) + 1
        self.noun_at = {n: max(d, key=d.get) for n, d in at_counts.items()}
        hits = {}
        from_mem = self.mem[self.from_ids]
        for sent in log:
            ks = list(self.kind_of[sent])
            if self.k_particle not in ks or self.k_from not in ks:
                continue
            frm = sent[ks.index(self.k_from)]
            for i in range(1, len(sent)):
                if ks[i] != self.k_particle or ks[i - 1] != self.k_name or sent[i - 1] not in self.noun_at:
                    continue
                guess = self.from_ids[int((from_mem - (self.mem[self.noun_at[sent[i - 1]]] + self.shift_at)).norm(dim=1).argmin())]
                h = hits.setdefault(sent[i], [0, 0])
                h[0] += int(int(guess) == frm)
                h[1] += 1
        best = max(hits.items(), key=lambda kv: (kv[1][0] + 1) / (kv[1][1] + 2), default=None)
        self.start_particle = best[0] if best and best[1][0] > 0 else None

    def story_state(self, ws, ks):
        """the start state of an ordered story (in "from" space), or None."""
        self.learn_particles()
        if self.start_particle is None or self.start_particle not in ws:
            return None
        nouns = [x for x, k in zip(ws, ks) if k == self.k_name]
        i = ws.index(self.start_particle)
        if self.bind_random:
            noun = nouns[sum(ws) % len(nouns)] if nouns else None
        else:
            noun = ws[i - 1] if i > 0 and ks[i - 1] == self.k_name else None
        if noun is None or noun not in self.noun_at:
            return None
        return self.mem[self.noun_at[noun]] + self.shift_at

    def book_move(self, v, rooms):
        """the pure move of motion word v in these rooms (the room's own entry if any, else the most counted; stage B's
        context sign when set), or None."""
        entries = self.books.get(v)
        if not entries:
            return None
        here = [e for e in entries if e[2] & rooms]
        move = max(here or entries, key=lambda e_: e_[1])[0]
        if not here and self.context:
            move = move * self.room_sign(rooms)
        if not here and self.frame:
            move = self.room_frame(rooms)(move)
        if not here and self.scale:
            move = move * self.room_scale(rooms)
        return move

    def track_state(self, ws, rooms):
        """step 2: (the asked object's state in "from" space, the moves applied to it) for a multi-moment story, or None."""
        self.learn_particles()
        if self.start_particle is None:
            return None
        moments, cur = [], []
        for x in ws:
            if x == self.go_tok:
                moments.append(cur)
                cur = []
            else:
                cur.append(x)
        moments.append(cur)
        q = moments[-1]
        names_q = [x for x in q if self.kind_of[x] == self.k_name]
        rel_parts = self.relation_particles()
        count_q = None                              # step 4+: "[C REL] Q" -- how many in relation REL to C?
        if self.eodi_tok not in q and len(names_q) == 1 and len(q) == 3 and q[0] == names_q[0] \
                and q[1] in rel_parts and self.kind_of[q[2]] == self.k_particle:
            count_q = (q[0], q[1])
        elif self.eodi_tok not in q or len(names_q) != 1:
            return None
        asked, state, steps, one, placed = (None if count_q else names_q[0]), {}, 0, None, set()
        attach = {}                                 # step 4+: object -> the object it is carried by
        rel_of = {}                                 # step 4+: object -> its latest (relatum, relation particle)
        for m in moments[:-1]:
            ks = self.kind_of[m]
            bound = {m[i]: m[i - 1] for i in range(1, len(m)) if ks[i] == self.k_particle and ks[i - 1] == self.k_name}
            lm = bound.get(self.start_particle)
            movers = [n for p, n in bound.items() if p != self.start_particle and p not in rel_parts]
            if len(movers) == 1:
                rels = rel_parts & set(bound)
                for p in rels:
                    rel_of[movers[0]] = (bound[p], p)
                    if self.carries(p):
                        attach[movers[0]] = bound[p]
                if not rels and any(k in (self.k_at, self.k_from) for k in ks):
                    rel_of.pop(movers[0], None)     # placed elsewhere: out of its relation
                    attach.pop(movers[0], None)
            if len(movers) != 1:
                return None
            mv = movers[0]
            if lm is not None:
                if lm not in self.noun_at:
                    return None
                if self.track_one:
                    one = self.mem[self.noun_at[lm]] + self.shift_at
                else:
                    state[mv] = self.mem[self.noun_at[lm]] + self.shift_at
            elif not self.track_one:            # step 4: a position token places the mover ("at" = where it is)
                fr = [x for x, k in zip(m, ks) if k == self.k_from]
                at = [x for x, k in zip(m, ks) if k == self.k_at]
                if fr:
                    state[mv] = self.mem[fr[0]].clone()
                    placed.add(mv)
                elif at:
                    state[mv] = self.mem[at[0]] + self.shift_at
                    placed.add(mv)
            for v in [x for x, k in zip(m, ks) if k == self.k_verb]:
                move = self.book_move(v, rooms)
                if move is None:
                    return None
                if self.track_one:
                    if one is None:
                        return None
                    one = one + move
                    steps += 1
                else:
                    if mv not in state:
                        return None
                    if self.repeats(v):                 # step 4+: until it rests
                        for _ in range(4 * len(self.to_ids)):
                            nxt = state[mv] + move
                            if self.land(nxt) is None or any(float((nxt - state[o]).norm()) < self.half_gap
                                                             for o in state if o != mv):
                                break
                            state[mv] = nxt
                        steps += int(mv == asked)
                        continue
                    state[mv] = state[mv] + move
                    steps += int(mv == asked)
                    for o in [o for o, h in attach.items() if h == mv and o in state]:   # step 4+: carried along
                        state[o] = state[o] + move
                        steps += int(o == asked)
                    if self.props or self._force is not None:   # step 4: did the mover land on another object?
                        for o in [o for o in state if o != mv]:
                            if float((state[mv] - state[o]).norm()) < self.half_gap:
                                h = self.prop_of(o)
                                if h is None:
                                    return None
                                if h == "heavy":
                                    state[mv] = state[mv] - move
                                else:
                                    state[o] = state[o] + move
                                    steps += int(o == asked)
        if count_q is not None:
            cq, cp = count_q
            if cq not in state:
                return None
            n = sum(1 for o, rc in rel_of.items() if rc == count_q and o in state and
                    float((state[o] - state[cq]).norm()) < self.half_gap)
            return "count", n
        end = one if self.track_one else state.get(asked)
        if end is None or (steps == 0 and asked not in placed):
            return None
        return end, max(steps, 1)

    def learn_numbers(self):
        """step 4+: count -> number word, from heard stories ending with a number word (majority)."""
        log = getattr(self.learner, "order_log", [])
        key = (len(log), self.n_rows)
        if key == self.num_key or self.k_number is None:
            return
        self.num_key = key
        votes = {}
        for sent in log:
            if self.go_tok not in sent or self.kind_of[sent[-1]] != self.k_number:
                continue
            rooms = {int(self.place[x]) for x in sent if self.place[x] >= 0}
            res = self.track_state(list(sent[:-1]), rooms)
            if res is not None and res[0] == "count":
                d = votes.setdefault(res[1], {})
                d[sent[-1]] = d.get(sent[-1], 0) + 1
        self.num_table = {n: max(d, key=d.get) for n, d in votes.items()}
        nxt = {}
        for sent in log:                            # the chant: a sentence of number words only, in order
            if len(sent) >= 2 and all(self.kind_of[x] == self.k_number for x in sent):
                for a, b in (zip(sent[1:], sent) if self.succ_rev else zip(sent, sent[1:])):
                    d = nxt.setdefault(a, {})
                    d[b] = d.get(b, 0) + 1
        self.num_next = {a: max(d, key=d.get) for a, d in nxt.items()}

    def number_word(self, n):
        self.learn_numbers()
        if self.count_rand:
            ns = sorted(self.num_table)
            if n not in ns or len(ns) < 2:
                return None
            return self.num_table[ns[(ns.index(n) + 1) % len(ns)]]
        if n in self.num_table:
            return self.num_table[n]
        if self.succ and n > 1:                     # step 4+: one more = the next word in the chant
            prev = self.number_word(n - 1)
            return self.num_next.get(prev) if prev is not None else None
        return None

    def repeats(self, v):
        """step 4+: does motion word v repeat until blocked?"""
        if self._until_force is not None:
            return self._until_force == "until"
        if self.grav_all:
            return True
        if not self.grav:
            return False
        self.learn_until()
        return self.until_of.get(v, False)

    def learn_until(self):
        """step 4+: per motion word, does it repeat? From the heard stories, each read twice (once / until blocked)."""
        log = getattr(self.learner, "order_log", [])
        key = (len(log), self.n_rows)
        if key == self.until_key:
            return
        self.until_key = key
        votes = {}
        for sent in log:
            if self.go_tok not in sent or self.kind_of[sent[-1]] != self.k_to:
                continue
            vs = [x for x in sent if self.kind_of[x] == self.k_verb]
            if len(vs) != 1:
                continue
            rooms = {int(self.place[x]) for x in sent if self.place[x] >= 0}
            got = {}
            for f in ("once", "until"):
                self._until_force = f
                res = self.track_state(list(sent[:-1]), rooms)
                got[f] = res is not None and self.land(res[0]) == sent[-1]
            self._until_force = None
            if got["once"] != got["until"]:
                d = votes.setdefault(vs[0], [0, 0])
                d[int(got["until"])] += 1
        self.until_of = {v: d[1] > d[0] for v, d in votes.items()}

    def relation_particles(self):
        """step 4+: the relation particles (see __init__); learned from the ordered log, cached."""
        self.learn_rel()
        return self.rel_parts

    def carries(self, p):
        if self._rel_force is not None:
            return self._rel_force == "carried"
        if self.contain_all:
            return True
        if not self.contain:
            return False
        return self.rel_effect.get(p, False)

    def learn_rel(self):
        log = getattr(self.learner, "order_log", [])
        key = (len(log), self.n_rows)
        if key == self.rel_key:
            return
        self.rel_key = key
        if self.start_particle is None or self.go_tok is None:
            self.rel_parts, self.rel_effect = set(), {}
            return
        cnt = {}
        for sent in log:
            if self.go_tok not in sent:
                continue
            ks = self.kind_of[sent]
            for i in range(1, len(sent)):
                if ks[i] == self.k_particle and ks[i - 1] == self.k_name and sent[i] not in (self.start_particle,
                                                                                            self.go_tok, self.eodi_tok):
                    cnt[sent[i]] = cnt.get(sent[i], 0) + 1
        self.mover_particle = max(cnt, key=cnt.get) if cnt else None
        self.rel_parts = {p for p in cnt if p != self.mover_particle}
        votes = {}
        for sent in log:
            if self.go_tok not in sent or self.kind_of[sent[-1]] != self.k_to:
                continue
            ps = [p for p in self.rel_parts if p in sent]
            if len(ps) != 1:
                continue
            rooms = {int(self.place[x]) for x in sent if self.place[x] >= 0}
            got = {}
            for f in ("carried", "not"):
                self._rel_force = f
                res = self.track_state(list(sent[:-1]), rooms)
                got[f] = res is not None and self.land(res[0]) == sent[-1]
            self._rel_force = None
            if got["carried"] != got["not"]:
                d = votes.setdefault(ps[0], [0, 0])
                d[int(got["carried"])] += 1
        self.rel_effect = {p: d[1] > d[0] for p, d in votes.items()}

    def prop_of(self, o):
        """step 4: the property-table entry ("light" / "heavy") the learner reads for object o, or None."""
        if self._force is not None:
            self._hits.append(o)
            return self._force
        self.learn_props()
        if self.props_rand:
            names = sorted(self.prop_table)
            if o not in names or len(names) < 2:
                return None
            o = names[(names.index(o) + 1) % len(names)]
        return self.prop_table.get(o)

    def learn_props(self):
        """step 4: the property table from the heard stories with a collision (read twice: light / heavy)."""
        log = getattr(self.learner, "order_log", [])
        key = (len(log), self.n_rows)
        if key == self.props_key:
            return
        self.props_key = key
        votes = {}
        for sent in log:
            if self.go_tok not in sent or self.kind_of[sent[-1]] != self.k_to:
                continue
            rooms = {int(self.place[x]) for x in sent if self.place[x] >= 0}
            got = {}
            for f in ("light", "heavy"):
                self._force, self._hits = f, []
                res = self.track_state(list(sent[:-1]), rooms)
                got[f] = (res is not None and self.land(res[0]) == sent[-1], list(self._hits))
            self._force = None
            hits = set(got["light"][1]) | set(got["heavy"][1])
            if len(hits) == 1 and got["light"][0] != got["heavy"][0]:
                o = hits.pop()
                v = votes.setdefault(o, {"light": 0, "heavy": 0})
                v["light" if got["light"][0] else "heavy"] += 1
        self.prop_table = {o: max(v, key=v.get) for o, v in votes.items() if v["light"] != v["heavy"]}

    def land(self, state):
        """the "to" cell the state lands on within half a gap, or None."""
        dist = (self.mem[self.to_ids] - (state + self.shift)).norm(dim=1)
        j = int(dist.argmin())
        return int(self.to_ids[j]) if float(dist[j]) < self.gap_at(int(self.to_ids[j])) else None

    def gap_at(self, t):
        """half the gap around "to" cell t (the number line's own when t is on it)."""
        return self.line_half if t in self.line_to else self.half_gap

    def room_scale(self, rooms):
        """step 4+ (D15): the room's scale against the usual books (1.0 when nothing can be compared)."""
        key = (tuple(sorted(rooms)), self.n_rows)
        if key not in self.scale_cache:
            rs = []
            for v, entries in self.books.items():
                here = [e for e in entries if e[2] & rooms]
                if not here:
                    continue
                usual = max(entries, key=lambda e_: e_[1])[0]
                local = max(here, key=lambda e_: e_[1])[0]
                rs.append(float(local @ usual) / (float(usual @ usual) + 1e-12))
            r = float(np.median(rs)) if rs else 1.0
            if self.scale_rev and abs(r) > 1e-9:
                r = 1.0 / r
            self.scale_cache[key] = r
        return self.scale_cache[key]

    def trusted(self):
        """step 2: the tracking recipe's record on the heard multi-moment stories (scene = the last element, a "to" cell):
        predicted from the story alone, then compared; trusted when right more often than not (Laplace)."""
        log = getattr(self.learner, "order_log", [])
        key = (len(log), self.n_rows)
        if key != self.track_key:
            self.track_key = key
            hit = n = 0
            for sent in log:
                if self.go_tok not in sent or self.kind_of[sent[-1]] != self.k_to:
                    continue
                rooms = {int(self.place[x]) for x in sent if self.place[x] >= 0}
                res = self.track_state(list(sent[:-1]), rooms)
                n += 1
                hit += int(res is not None and self.land(res[0]) == sent[-1])
            self.track_trust = (hit, n)
        hit, n = self.track_trust
        return (hit + 1) / (n + 2) > 0.5

    def room_frame(self, rooms):
        """step 4+: the room's frame as a function on moves (identity when nothing can be compared)."""
        key = (tuple(sorted(rooms)), self.n_rows)
        if key in self.frame_cache:
            return self.frame_cache[key]
        ident = _FrameIdent()                       # (D11: frames are small objects, not lambdas, so a life's state can be
        usual = {v: max(es, key=lambda e_: e_[1])[0] for v, es in self.books.items() if es}   # written; same arithmetic)
        vs = list(usual)
        best_pair, best_cos = None, 1.0
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                a, b = usual[vs[i]], usual[vs[j]]
                cs = abs(float(a @ b) / (float(a.norm() * b.norm()) + 1e-9))
                if cs < best_cos:
                    best_pair, best_cos = (a, b), cs
        if best_pair is None:
            self.frame_cache[key] = ident
            return ident
        u1, u2 = best_pair
        U = torch.stack([u1, u2], 1)                                   # D x 2
        pinv = torch.linalg.pinv(U)                                    # 2 x D

        def make(img1, img2):
            V = torch.stack([img1, img2], 1)
            return _FrameMap(V, pinv)
        frames = [ident] + [make(s1 * x, s2 * y) for x, y in ((u1, u2), (u2, u1)) for s1 in (1, -1) for s2 in (1, -1)
                            if not (x is u1 and s1 == 1 and s2 == 1)]
        pairs = []
        for v, es in self.books.items():
            here = [e for e in es if e[2] & rooms]
            if here and v in usual:
                pairs.append((usual[v], max(here, key=lambda e_: e_[1])[0]))
        if not pairs:
            self.frame_cache[key] = ident
            return ident
        errs = [sum(float((f(u) - l_).norm()) for u, l_ in pairs) for f in frames]
        best = frames[int(np.argmin(errs))]
        self.frame_cache[key] = best
        return best

    def room_sign(self, rooms):
        """stage B: +1 if the room's own books agree with the usual ones (or nothing to compare), -1 if they mostly run
        against them -- the majority over the motion words heard in the room."""
        votes = 0
        for v, entries in self.books.items():
            here = [e for e in entries if e[2] & rooms]
            if not here:
                continue
            usual = max(entries, key=lambda e_: e_[1])[0]
            local = max(here, key=lambda e_: e_[1])[0]
            votes += 1 if float(local @ usual) >= 0 else -1
        return -1.0 if votes < 0 else 1.0

    def view_coords(self):
        """image coordinates (2 numbers) of every at / from / to cell, from the two most orthogonal usual moves; or None."""
        usual = [max(es, key=lambda e_: e_[1])[0] for es in self.books.values() if es]
        best, best_cos = None, 1.0
        for i in range(len(usual)):
            for j in range(i + 1, len(usual)):
                a_, b_ = usual[i], usual[j]
                cs = abs(float(a_ @ b_) / (float(a_.norm() * b_.norm()) + 1e-9))
                if cs < best_cos:
                    best, best_cos = (a_, b_), cs
        if best is None:
            return None
        pinv = torch.linalg.pinv(torch.stack(best, 1))
        self.vc_pinv = pinv
        coords = {}
        for k in (self.k_at, self.k_from, self.k_to):
            ids = [int(t) for t in np.flatnonzero(self.kind_of == k) if int(t) not in self.line_set]
            centre = self.mem[ids].mean(0)
            for t in ids:
                coords[t] = (pinv @ (self.mem[t] - centre)).numpy()
        return coords

    def learn_views(self):
        """step 4+ (D17): the axis matching and the per-view depth moves, from the heard three-view moments."""
        log = getattr(self.learner, "order_log", [])
        key = (len(log), self.n_rows)
        if key == self.view_key:
            return self.view_tables
        self.view_key, self.view_tables = key, None
        coords = self.view_coords()
        if coords is None:
            return None
        labels = sorted(self.view_tok)
        sights, moves = [], {}
        for sent in log:
            if not self.view_tok & set(sent):
                continue
            ks = self.kind_of[sent]
            vi = [i for i, k in enumerate(ks) if k == self.k_verb]
            pairs = [(i, sent[i - 1], sent[i]) for i in range(1, len(sent)) if sent[i] in self.view_tok and sent[i - 1] in coords]
            if not vi:
                sights.append({lab: np.round(coords[c]) for _, c, lab in pairs})
            else:
                before = {lab: coords[c] for i, c, lab in pairs if i < vi[0]}
                after = {lab: coords[c] for i, c, lab in pairs if i > vi[0]}
                for lab in before:
                    if lab in after:
                        d = tuple(np.round(after[lab] - before[lab]).astype(int))
                        tab = moves.setdefault((sent[vi[0]], lab), {})
                        tab[d] = tab.get(d, 0) + 1
        match = {}
        for t in labels:
            for ax in (0, 1):
                best, best_n = None, -1
                for s in labels:
                    if s == t:
                        continue
                    for j in (0, 1):
                        for sg in (1, -1):
                            n = sum(1 for sg_ in sights if t in sg_ and s in sg_ and sg_[t][ax] == sg * sg_[s][j])
                            if n > best_n:
                                best, best_n = (s, j, sg), n
                if best is not None and self.views_rand:
                    best = (best[0], 1 - best[1], best[2])
                match[(t, ax)] = best
        disp = {k_: np.array(max(v_, key=v_.get), dtype=float) for k_, v_ in moves.items()}
        if self.vturn:                                  # D2: verb maps (after axis <- sign x before axis + offset)
            pairs_by_verb = {}
            for sent in log:
                if not self.view_tok & set(sent):
                    continue
                ks = self.kind_of[sent]
                vi = [i for i, k in enumerate(ks) if k == self.k_verb]
                if not vi:
                    continue
                pr = [(i, sent[i - 1], sent[i]) for i in range(1, len(sent)) if sent[i] in self.view_tok and sent[i - 1] in coords]
                before = {lab: np.round(coords[c]) for i, c, lab in pr if i < vi[0]}
                after = {lab: np.round(coords[c]) for i, c, lab in pr if i > vi[0]}
                pairs_by_verb.setdefault(sent[vi[0]], []).append((before, after))
            vmap = {}
            for v, obs in pairs_by_verb.items():
                for t in labels:
                    for ax in (0, 1):
                        best, best_n = None, -1
                        for s in labels:
                            for j in (0, 1):
                                for sg in (1, -1):
                                    offs = {}
                                    for b_, a_ in obs:
                                        if t in a_ and s in b_:
                                            o = int(a_[t][ax] - sg * b_[s][j])
                                            offs[o] = offs.get(o, 0) + 1
                                    if offs:
                                        o, n = max(offs.items(), key=lambda kv: kv[1])
                                        if n > best_n:
                                            best, best_n = (s, j, sg, o), n
                        if best is not None:
                            vmap[(v, t, ax)] = best
            disp = (disp, vmap)
        self.view_tables = (coords, match, disp)
        return self.view_tables

    def view_answer(self, ws):
        """step 4+ (D17): the answer cell of a three-view question, or None."""
        tabs = self.learn_views()
        if tabs is None:
            return None
        coords, match, disp = tabs
        given = {ws[i]: coords[ws[i - 1]] for i in range(1, len(ws)) if ws[i] in self.view_tok and ws[i - 1] in coords}
        asked = [x for i, x in enumerate(ws) if x in self.view_tok and (i == 0 or ws[i - 1] not in coords)]
        if len(asked) != 1:
            return None
        t = asked[0]
        verbs = [x for x in ws if self.kind_of[x] == self.k_verb]
        if not verbs:                                   # fill the missing view
            tgt = []
            for ax in (0, 1):
                m = match.get((t, ax))
                if m is None or m[0] not in given:
                    return None
                tgt.append(m[2] * given[m[0]][m[1]])
            kind = self.k_at
        elif self.vturn:                                # D2: the verb's map from the views before
            vmap = disp[1]
            tgt = []
            for ax in (0, 1):
                m = vmap.get((verbs[0], t, ax))
                if m is None or m[0] not in given:
                    return None
                tgt.append(m[2] * given[m[0]][m[1]] + m[3])
            kind = self.k_to
        else:                                           # a depth move: this view's start plus its displacement
            if t not in given or (verbs[0], t) not in disp:
                return None
            tgt = list(given[t] + disp[(verbs[0], t)])
            kind = self.k_to
        tgt = np.array(tgt)
        best, best_d = None, 0.5
        for c, v in coords.items():
            if self.kind_of[c] == kind and float(np.abs(v - tgt).max()) < best_d:
                best, best_d = c, float(np.abs(v - tgt).max())
        return best

    def goal_answer(self, ws):
        """D2 goal: the goal's name (EODIRO) or the next motion word, or None."""
        coords = self.view_coords()
        if coords is None:
            return None
        pairs = [(ws[i], ws[i + 1]) for i in range(len(ws) - 1)
                 if self.kind_of[ws[i]] == self.k_name and ws[i + 1] in coords and self.kind_of[ws[i + 1]] == self.k_at]
        if len(pairs) < 3:
            return None
        agent, others = pairs[0], pairs[1:]
        usual = {v: max(es, key=lambda e_: e_[1])[0] for v, es in self.books.items() if es}
        step = lambda v: np.round((self.vc_pinv @ usual[v]).numpy())
        verbs = [x for x in ws if self.kind_of[x] == self.k_verb]
        if not verbs or any(v not in usual for v in verbs):
            return None
        start = np.round(coords[agent[1]])
        now = start + sum(step(v) for v in verbs)
        dist = lambda a, b: float(np.abs(a - b).sum())
        gains = [dist(start, np.round(coords[c])) - dist(now, np.round(coords[c])) for _, c in others]
        order = np.argsort(gains)[::-1]
        if gains[order[0]] <= 0 or (len(gains) > 1 and gains[order[0]] == gains[order[1]]):
            return None
        g = order[1] if self.goal_rand else order[0]
        if self.goal_q is not None and self.goal_q in ws:
            return others[g][0]
        tgt = np.round(coords[others[g][1]])
        best = sorted(((dist(now, tgt) - dist(now + step(v), tgt), v) for v in usual), reverse=True)
        if best[0][0] <= 0 or (len(best) > 1 and best[0][0] == best[1][0]):
            return None
        return best[0][1]

    def self_answer(self, ws):
        """D2 self: the thing that moves with my actions, or None."""
        coords = self.view_coords()
        if coords is None:
            return None
        usual = {v: max(es, key=lambda e_: e_[1])[0] for v, es in self.books.items() if es}
        segs, cur = [], None
        for x in ws:
            if x in self.actions:
                cur = [x, []]
                segs.append(cur)
            elif cur is not None:
                cur[1].append(x)
        if not segs or any(a not in usual for a, _ in segs):
            return None
        syms = [np.array(m) for m in ([[1, 0], [0, 1]], [[-1, 0], [0, 1]], [[1, 0], [0, -1]], [[-1, 0], [0, -1]],
                                      [[0, 1], [1, 0]], [[0, -1], [1, 0]], [[0, 1], [-1, 0]], [[0, -1], [-1, 0]])]
        if self.self_id:
            syms = syms[:1]
        moves = {}
        for a, body in segs:
            act = np.round((self.vc_pinv @ usual[a]).numpy())
            for i in range(len(body) - 2):
                n, f, t = body[i], body[i + 1], body[i + 2]
                if self.kind_of[n] == self.k_name and self.kind_of[f] == self.k_from and self.kind_of[t] == self.k_to \
                        and f in coords and t in coords:
                    moves.setdefault(n, []).append((act, np.round(coords[t] - coords[f])))
        found = [n for n, ms in moves.items() if len(ms) == len(segs) and
                 any(all(np.array_equal(S @ a_, m_) for a_, m_ in ms) for S in syms)]
        return found[0] if len(found) == 1 else None

    def part_ok(self, p):
        """D69: has part p earned its say (Laplace over the part's record on fresh heard material)?"""
        hit, n = self.part_trust.get(p, (0, 0))
        return (hit + 1) / (n + 2) > 0.5

    def quiet_parts(self):
        """D69: the attributes of the parts that are on but have not earned their say (none unless "+gate")."""
        if not self.gate:
            return []
        return [a for p, a in GATED if getattr(self, a, False) and not self.part_ok(p)]

    def term(self, ctx):
        """the world model's part of the answer score; with "+gate" (D69) the parts that have not earned their say are
        quiet for this call (their flags off, then back on)."""
        if self.useful:
            return self._use_term(ctx)
        quiet = self.quiet_parts()
        if not quiet:
            return self._term(ctx)
        for a in quiet:
            setattr(self, a, False)
        try:
            return self._term(ctx)
        finally:
            for a in quiet:
                setattr(self, a, True)

    def use_sig(self, ws):
        """D83 (1): a question's form -- which element kinds it holds (presence, not counts)."""
        return tuple(sorted(set(int(k) for k in self.kind_of[ws]))) if len(ws) else ()

    def _use_perm(self, sig):
        """D83 (1) reference gate "shuf": in each form, each part reads the record of another part (a fixed permutation of
        the parts that are on, seeded by the form)."""
        m = self.use_perms.get(sig)
        if m is None:
            names = [p for p, a in GATED if getattr(self, a, False)]
            s_ = sum((k + 1) * 31 ** i for i, k in enumerate(sig)) % (2 ** 31)
            perm = np.random.RandomState(s_).permutation(len(names))
            m = self.use_perms[sig] = {names[i]: names[int(j)] for i, j in enumerate(perm)}
        return m

    def use_say(self, p, sig, tally=False):
        """D83 (1): does part p speak in question form sig? The decision made at the last grading for that form; a form where
        it has never made a difference: its all-forms decision; neither: quiet."""
        if self.use_mode == "quiet":
            return False
        if self.use_mode == "shuf":
            p = self._use_perm(sig).get(p, p)
        d = self.use_dec.get((p, sig))
        if d is not None:
            if tally:
                self.use_count["in form"] = self.use_count.get("in form", 0) + 1
            return d
        d = self.use_dec_all.get(p)
        if d is not None and tally:
            self.use_count["backoff"] = self.use_count.get("backoff", 0) + 1
        return bool(d)

    def _use_view(self, on):
        """D83 (1): the world model with exactly the parts in `on` (attribute names) acting, with its OWN learned caches (the
        tracker's trust, the until / carried / property / number tables are read off the heard stories THROUGH the parts
        that act, so a brain with other parts has other tables) -- rebuilt from the learner as a branch's are."""
        v = self.use_views.get(on)
        if v is None:
            v = object.__new__(WorldBooks)
            v.__dict__.update(self.__dict__)
            v.useful, v.gate, v.use_views, v.use_rec, v.use_all, v.use_count = False, False, {}, {}, {}, {}
            for p_, a in GATED:
                setattr(v, a, a in on)
            v.n_rows, v.books, v.n_log = -1, {}, -1
            v.track_key, v.props_key, v.prop_table = None, None, {}
            v.until_key, v.until_of = None, {}
            v.frame_cache = {}
            v.rel_key, v.rel_parts, v.rel_effect = None, set(), {}
            v.num_key, v.num_table, v.num_next = None, {}, {}
            v.scale_cache = {}
            v.view_key, v.view_tables = None, None
            v.tale_books = {}
            self.use_views[on] = v
        return v

    def _use_term(self, ctx):
        """D83 (1): every question answered by the world model with the parts that speak in its form."""
        out = torch.zeros(len(ctx), self.mem.shape[0])
        if ctx.is_sparse:
            return out
        on = [(p, a) for p, a in GATED if getattr(self, a, False)]
        groups = {}
        for i, row in enumerate(ctx.tolist()):
            sig = self.use_sig([x for x in row if x >= 0])
            speak = frozenset(a for p, a in on if self.use_say(p, sig, tally=True))
            groups.setdefault(speak, []).append(i)
            self.use_count["rows"] = self.use_count.get("rows", 0) + 1
            self.use_count["decisions"] = self.use_count.get("decisions", 0) + len(on)
            for p, a in on:
                if a in speak:
                    self.use_count[f"say {p}"] = self.use_count.get(f"say {p}", 0) + 1
        for speak, rows in groups.items():
            idx = torch.tensor(rows)
            out[idx] = self._use_view(speak)._term(ctx[idx])
        return out

    def _use_allowed(self, cx):
        """D83 (1) edition 2: the cells the learner's own kind gate lets answer each question (finding 90's table; all cells
        when the gate is off) -- a world-model cell the gate masks never reaches the answer, so it is neither right nor wrong."""
        m = getattr(self.learner, "model", None)
        v0 = torch.zeros(len(cx), self.mem.shape[0])
        if m is None or not hasattr(m, "kind_first"):
            return v0 == 0
        return m.kind_first(cx, v0) > -1e8

    def grade_use(self, sents):
        """D83 (1): grade every part that is on, right after a session is taken in (D69 edition 3's moment: a story's things
        are named in its own session), on EVERY sentence heard so far (the tracker's own trust rule: its record is read again
        over all heard stories whenever the record grows). Context = all but the last element, answer = the last element. A
        part is judged in each sentence's question form against the world model AS IT SPEAKS NOW IN THAT FORM (Toolformer:
        the call inserted into the actual context, against the same context without it): with the part's say added (a quiet
        part) and taken out (a speaking part). Score per sentence: +1 right, 0 silent, -1 wrong, counting only cells the
        learner's kind gate lets answer (as the answer does). Per (part, form): net = sum of (with - without), changed = how
        many sentences differ. The decision for a form is made where changed > 0 (speak when net > 0); where the part made no
        difference the last decision stands (no evidence, no change); the all-forms decision is made the same way from the
        sums over forms.
        Edition 2 (2026-10-23, BEFORE ANY LIFE; peek_d83_ed1_ten4.out, records only): edition 1 judged each part ALONE on the
        bare world model and scored every firing. Two flaws of that measurement, not of the principle: the story binder alone
        answered the multi-moment stories that the tracker answers first whenever it speaks (the answer's own order), and
        firings of a kind the question never asks counted as wrong although the kind gate keeps them out of every answer.
        Edition 3 (same day, BEFORE ANY LIFE; peek_d83_ed2_ten4.out, records only): edition 2 added each session's judgement
        to a running sum, so a judgement made against an earlier brain (before the tracker spoke, the binder was charged with
        the multi-moment stories) stayed in the record after that brain had changed. Edition 3 reads all heard sentences
        again against the brain as it speaks now, as the tracker's trust does; "no difference -> the decision stands" keeps
        two parts that give the same answer from switching each other off and on."""
        import lab2 as L_
        self.use_heard.extend(list(x) for x in sents if len(x) >= 2)
        items = [(list(x[:-1]), int(x[-1])) for x in self.use_heard]
        if not items:
            return
        cx, truth = L_.items_tensor(items)
        sigs = [self.use_sig([x for x in row if x >= 0]) for row in cx.tolist()]
        allowed = self._use_allowed(cx)
        on = [(p, a) for p, a in GATED if getattr(self, a, False)]
        now = {sg: frozenset(a for p, a in on if self.use_say(p, sg)) for sg in set(sigs)}
        by_form = {}
        for i, sg in enumerate(sigs):
            by_form.setdefault(sg, []).append(i)
        rec = {}
        for sg, rows in by_form.items():
            idx = torch.tensor(rows)
            t, ok = truth[idx], allowed[idx]
            memo = {}

            def score(acting):
                if acting not in memo:
                    with torch.no_grad():
                        o = self._use_view(acting)._term(cx[idx])
                    o = torch.where(ok, o, torch.zeros_like(o))
                    best = o.max(1)
                    pred = torch.where(best.values > 0, best.indices, torch.full_like(best.indices, -1))
                    memo[acting] = (pred == t).long() - ((pred >= 0) & (pred != t)).long()
                return memo[acting]

            for p, a in on:
                d = score(now[sg] | {a}) - score(now[sg] - {a})
                ch = int((d != 0).sum())
                if ch:
                    rec[(p, sg)] = [int(d.sum()), ch]
        self.use_rec = rec
        self.use_all = {}
        for (p, sg), (net, ch) in rec.items():
            g = self.use_all.setdefault(p, [0, 0])
            g[0] += net
            g[1] += ch
        for key, (net, ch) in rec.items():
            self.use_dec[key] = net > 0
        for p, (net, ch) in self.use_all.items():
            self.use_dec_all[p] = net > 0

    def _shadow(self):
        """D69: a copy for grading -- the same books and tables, its own caches (one level copied), so grading never
        changes what the world model has learned or cached."""
        c = object.__new__(WorldBooks)
        c.__dict__.update(self.__dict__)
        for k, v in list(c.__dict__.items()):
            if isinstance(v, (dict, list, set)):
                c.__dict__[k] = copy.copy(v)
        return c

    def grade_parts(self, sents):
        """D69: grade every part that is on against this session's new sentences (call from the learner's prepare, right
        after the whole-sentence memory and the ordered record take them in). A record only.
        Edition 3 (2026-10-16, before any gated life; diag_d69.py): graded AFTER the session's sentences are taken in, as
        the tracker's own trust is (trusted(): heard stories predicted from the story alone -- all but the last element --
        then compared). Predicting before the session can never answer its stories: their things are named in the same
        session (a story room's naming moments come with its stories), so in editions 1-2 no story was ever answered and
        only the describe part got a record.
        Edition 2 (2026-10-16, before any gated life; peek_d69.out): a part is judged by what it changes in the world model
        AS IT NOW SPEAKS -- a speaking part against that brain without it, a quiet part against that brain with it added.
        (Edition 1 judged each part against every other part on: parts that give the same answer -- particles and the
        tracker -- then never differ, and all stay quiet.) A part that needs another (counting needs the tracker) is judged
        once the other speaks."""
        import lab2 as L_
        items = [(list(x[:-1]), int(x[-1])) for x in sents if len(x) >= 2]
        if not items:
            return
        cx, truth = L_.items_tensor(items)
        on_parts = [(p, a) for p, a in GATED if getattr(self, a, False)]
        quiet = {a for p, a in on_parts if not self.part_ok(p)}

        def pred(extra_on=None, extra_off=None):
            sh = self._shadow()
            for a in quiet:
                setattr(sh, a, False)
            if extra_on is not None:
                setattr(sh, extra_on, True)
            if extra_off is not None:
                setattr(sh, extra_off, False)
            with torch.no_grad():
                out = sh._term(cx)
            best = out.max(1)
            return torch.where(best.values > 0, best.indices, torch.full_like(best.indices, -1))

        now = pred()                                # the brain as it speaks now
        for p, a in on_parts:
            if a in quiet:
                on, off = pred(extra_on=a), now
            else:
                on, off = now, pred(extra_off=a)
            diff = on != off
            hit, n = self.part_trust.get(p, (0, 0))
            self.part_trust[p] = (hit + int((diff & (on == truth)).sum()), n + int(diff.sum()))

    def _term(self, ctx):
        """the world model's part of the answer score (V-dim per question); zero for questions it does not roll out."""
        out = torch.zeros(len(ctx), self.mem.shape[0])
        if ctx.is_sparse:
            return out
        self.refresh()
        log_n = math.log(max(len(self.learner.known), 2))
        framed = self.tale_mode in ("one", "anchor") and self.tale_mark is not None
        if framed and getattr(self, "grav", False):
            self.learn_until()                      # (cached with the reality books before any tale frame is put on)
        real_books = self.books
        for i, row in enumerate(ctx.tolist()):
            ws = [x for x in row if x >= 0]
            self.books = self.tale_frame() if framed and self.tale_mark in ws else real_books
            ks = self.kind_of[ws]
            verbs = [x for x, k in zip(ws, ks) if k == self.k_verb]
            froms = [x for x, k in zip(ws, ks) if k == self.k_from]
            rooms = {int(self.place[x]) for x in ws if self.place[x] >= 0}
            if self.self_part and self.self_q is not None and self.self_q in ws:
                j = self.self_answer(ws)                    # D2 self: who is me?
                if j is not None:
                    out[i, j] = log_n
                continue
            if self.goal and sum(1 for x in ks if x == self.k_at) >= 3 and not any(k in (self.k_from, self.k_to) for k in ks):
                j = self.goal_answer(ws)                    # D2 goal: where is the agent going / its next step
                if j is not None:
                    out[i, j] = log_n
                continue
            if self.views and self.view_tok & set(ws):     # step 4+ (D17): three-view questions
                j = self.view_answer(ws)
                if j is not None:
                    out[i, j] = log_n
                continue
            if self.reverse and not verbs and len(froms) == 1:
                tos = [x for x, k in zip(ws, ks) if k == self.k_to]
                if len(tos) == 1 and not any(k == self.k_at for k in ks):
                    seen = self.mem[tos[0]] - self.mem[froms[0]] - self.shift
                    hits = [v for v in self.books if (m_ := self.book_move(v, rooms)) is not None and
                            float((m_ - seen).norm()) < self.half_gap]
                    if len(hits) == 1:
                        out[i, hits[0]] = log_n
                    continue
            if (self.track or self.track_one) and self.go_tok is not None and self.go_tok in ws:
                if any(k == self.k_to for k in ks):
                    continue
                res = self.track_state(ws, rooms)
                if res is None or not self.trusted():
                    continue
                if res[0] == "count":                   # step 4+: a count question
                    if self.count or self.count_rand:
                        word = self.number_word(res[1])
                        if word is not None:
                            out[i, word] = log_n
                    continue
                j = self.land(res[0])
                if j is not None:
                    out[i, j] = log_n * res[1]
                continue
            if self.bind and self.k_particle is not None and verbs and not froms and \
                    not any(k in (self.k_to, self.k_at) for k in ks) and any(k == self.k_particle for k in ks):
                state = self.story_state(ws, list(ks))
                if state is None:
                    continue
            elif len(verbs) < 2 or len(froms) != 1 or any(k == self.k_to for k in ks):
                continue
            else:
                state = self.mem[froms[0]].clone()
            ok = True
            for v in verbs:
                entries = self.books.get(v)
                if not entries:
                    ok = False
                    break
                here = [e for e in entries if e[2] & rooms]
                e = max(here or entries, key=lambda e_: e_[1])
                move = e[0]
                if not here and self.context:
                    move = move * self.room_sign(rooms)
                if not here and self.frame:
                    move = self.room_frame(rooms)(move)
                if not here and self.scale:
                    move = move * self.room_scale(rooms)
                state = state + move
            if not ok:
                continue
            state = state + self.shift
            dist = (self.mem[self.to_ids] - state).norm(dim=1)
            j = int(dist.argmin())
            if float(dist[j]) < self.gap_at(int(self.to_ids[j])):
                out[i, int(self.to_ids[j])] = log_n * len(verbs)
        self.books = real_books
        return out
