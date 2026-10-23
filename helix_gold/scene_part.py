"""D75 step 1 (2026-10-17; D75_DESIGN.md, user decision "너의 추천대로 (가)부터 해보자"): learning movement from the scenes
before and after, WITHOUT role markers -- a part that reads only three kinds of element (a cell "at", a picture, a word)
and the order of the moments, so it runs on Dami's own worlds and on the exam form alike.

The rule (the same in every world; no answer is written in it):
  1. scene memory: for every picture, the cell it was last seen at -- a moment with exactly one picture and one cell and
     no word is an observation (a naming moment with background pictures is not: which picture is at the cell is not said;
     D80 v2, 2026-10-22: nor a question-and-answer moment such as [picture, "before", "where", cell] -- what is heard with
     words is said, not seen; no observation of the "@scene" worlds holds a word, so their lives are unchanged);
  2. change = event: when an observation puts a picture on a NEW cell, the words of the last sentence (the last moment
     with words and no cell) are counted with that change -- the move (cell now - cell before) per word, in the room (the
     element's category) and over all rooms, and the picture that moved per word;
  3. the word that explains the change: when answering, the sentence's word whose counted moves are the most constant
     (the largest share on one move; more counts break a tie) gives the move -- the room's own counts first (a room may be
     turned), else all rooms';
  4. roll out: the question's last seen cell of its picture + that move = the answer cell (only when the move lands on
     the grid); otherwise the part is silent.
Step 1 has the own-cell frame only (D75 step 2 adds the other-thing frame). Off unless "+scene" (the model has no
`scene` attribute): every other life is unchanged."""
import math
from collections import Counter, defaultdict

import numpy as np
import torch


class SceneBooks:
    def __init__(self, w, learner=None):
        self.w, self.learner = w, learner
        kinds = list(w.kinds)
        self.kind_of = np.asarray(w.kind_of)
        self.k_at = kinds.index("at")
        pic_kinds = {kinds.index(k) for k in ("drawn", "converted", "picture") if k in kinds}
        self.is_pic = np.isin(self.kind_of, list(pic_kinds))
        is_pos = np.zeros(len(self.kind_of), dtype=bool)
        is_pos[list(w.pos_of)] = True                     # every cell element (any role) is not a word
        self.is_word = ~self.is_pic & ~is_pos
        bnd = getattr(w, "boundary", None)
        if bnd is not None:
            self.is_word[bnd] = False
        self.boundary = bnd
        self.pos_of = {int(t): tuple(p) for t, p in w.pos_of.items() if w.role_of.get(t, "at") == "at"}
        self.at_of = {p: t for t, p in self.pos_of.items()}
        self.cats = np.asarray(w.cats)
        self.seen = {}                                   # picture -> cell last seen (the scene memory)
        self.moves = defaultdict(Counter)                # (room, word) -> Counter(move)
        self.moves_all = defaultdict(Counter)            # word -> Counter(move)
        self.movers = defaultdict(Counter)               # word -> Counter(picture)
        self.last_words = []
        self.events = 0
        self.silent = False

    # ---------- learning
    def observe(self, moments):
        """read heard moments in time order (one room's, or one exam session's)."""
        for m in moments:
            m = [int(x) for x in m]
            ats = [x for x in m if x in self.pos_of]
            pics = [x for x in m if self.is_pic[x]]
            words = [x for x in m if self.is_word[x]]
            if not ats:
                if words:
                    self.last_words = words
                continue
            if len(ats) == 1 and len(pics) == 1 and not words:
                p, cell = pics[0], self.pos_of[ats[0]]
                before = self.seen.get(p)
                if before is not None and before != cell and self.last_words:
                    d = (cell[0] - before[0], cell[1] - before[1])
                    room = int(self.cats[p])
                    for x in self.last_words:
                        self.moves[(room, x)][d] += 1
                        self.moves_all[x][d] += 1
                        self.movers[x][p] += 1
                    self.events += 1
                self.seen[p] = cell

    # ---------- answering
    def _best_move(self, words, room):
        best = None
        for x in words:
            c = self.moves.get((room, x)) or self.moves_all.get(x)
            if not c:
                continue
            n = sum(c.values())
            d, top = c.most_common(1)[0]
            key = (top / n, n)
            if best is None or key > best[0]:
                best = (key, d)
        return None if best is None else best[1]

    def told(self, words, room_of=None):
        """D80 v2 (2026-10-22): a sentence heard with no scene after it -> (the picture it moves, its move), from the counted
        events only, and only what is believed (Laplace (top + 1) / (n + 2) > 1/2, as the D80 equals sign):
          the mover = the top picture of the sentence word whose movers are the most constant (the largest share on one
          picture; more counts break a tie);
          the move = rule 3 in that picture's room among the OTHER words that were heard with changes of two or more
          different things (a word that tells a motion, not a name -- a moment of names only tells no move).
        None otherwise."""
        def believed(c):
            n = sum(c.values())
            return (c.most_common(1)[0][1] + 1) / (n + 2) > 0.5
        best = None
        for x in words:
            c = self.movers.get(x)
            if not c or not believed(c):
                continue
            n = sum(c.values())
            p, top = c.most_common(1)[0]
            key = (top / n, n)
            if best is None or key > best[0]:
                best = (key, p, x)
        if best is None:
            return None
        room = int(self.cats[best[1]])
        pick = None                                      # rule 3 (as _best_move) over the motion words, with its counts
        for x in words:
            if x == best[2] or len(self.movers.get(x, ())) < 2:
                continue
            c = self.moves.get((room, x)) or self.moves_all.get(x)
            if not c:
                continue
            n = sum(c.values())
            d, top = c.most_common(1)[0]
            key = (top / n, n)
            if pick is None or key > pick[0]:
                pick = (key, d, c)
        if pick is None or not believed(pick[2]):
            return None
        return (int(best[1]), pick[1])

    def term(self, ctx):
        out = torch.zeros(len(ctx), len(self.kind_of))
        if self.silent or ctx.is_sparse:
            return out
        log_n = math.log(max(len(getattr(self.learner, "known", [])) if self.learner is not None else 2, 2))
        for i, row in enumerate(ctx.tolist()):
            ws = [x for x in row if x >= 0]
            pics = [x for x in ws if self.is_pic[x]]
            ats = [x for x in ws if x in self.pos_of]
            words = [x for x in ws if self.is_word[x]]
            if len(pics) != 1 or len(ats) != 1 or not words:
                continue
            cell = self.pos_of[ats[0]]
            d = self._best_move(words, int(self.cats[pics[0]]))
            if d is None:
                continue
            new = (cell[0] + d[0], cell[1] + d[1])
            if new in self.at_of:
                out[i, self.at_of[new]] = log_n
        return out

    def record(self):
        return {"scene events": float(self.events), "scene words": float(len(self.moves_all))}
