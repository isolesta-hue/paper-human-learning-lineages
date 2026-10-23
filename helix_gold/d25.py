"""D25 edition 1 (2026-10-13, user-approved design D25_DESIGN.md): a filter, a temporary store and a transfer of the parts'
confident answers into the body's practice -- the "sleep" between sessions.

Candidates (made from heard material only, never from test items): per room, once the room is met, a fixed pool of NEW
two-verb questions -- one of the room's things, two motion words heard in the room, a start cell heard in the room, one
background picture -- the shape of the chain questions, which the world model rolls out and the body alone does not answer
(finding 145: body 0.03-0.06). A pool question whose (two words, start cell) equals a chain test question of any room is never
made. In a world with told tales the same questions are also made with the tale marker in front ("tale" frame tag: the world
model answers them in the tale frame).
Sleep (after every session): the world model answers every pool question of every room met so far; a question it answers
(it fires = it is sure, finding 125) is a candidate with that answer. Counters per question: in how many sleeps in a row it
gave the SAME answer (repeat -- also "still there after time"); its frame tag (reality / tale).
Selection (when a session or a method-layer branch prepares its practice; the store is copied into every branch):
  U  every candidate confirmed once, any frame, no usefulness check (the unfiltered control);
  F  repeat >= 2, reality frame, and useful = the body alone (world model silent) gets it wrong now (fixed, automatic);
  K  as F with the repeat threshold a method-layer knob ("d25_k" in 1 / 2 / 3, start 2);
  M  as K plus the method layer's direct marks ("d25_mark": "keep" = transfer every confirmed reality candidate now, ignoring
     the threshold; "forget" = clear the store's unconfirmed counts; one-shot, reset after the session);
  E  practice volume control: as many items as U would have transferred, drawn from the session's own heard items instead;
  N  the control: the store sleeps and records (candidates, body-alone keys), nothing is transferred.
Transferred questions stay in the body's practice for the rest of the life (weight 1, as the self-made practice items),
with their source ("inferred by the world model") and frame kept in the store's record.
"""
import numpy as np
import torch

import lab2 as L

POOL = 12                   # new questions per room (a representative value)


class Store:
    def __init__(self, w, seed, mode):
        self.w, self.seed, self.mode = w, seed, mode
        self.pools = {}                     # room -> list of (context, frame)
        self.state = {}                     # (room, j) -> [repeat, answer]
        self.transferred = {}               # (room, j) -> (context, answer, frame)
        self.e_count = 0                    # E: how many U would have transferred so far
        self.u_seen = set()
        chains = set()
        for c in range(w.n_cat):
            for _, s_, _ in w.categories[c].get("chain", []):
                vs = frozenset(int(x) for x in s_ if int(x) in w.verb_of)
                fr = [int(x) for x in s_ if int(x) in self.froms()]
                if len(vs) == 2 and len(fr) == 1:
                    chains.add((vs, fr[0]))
        self.test_cores = chains
        self.last = {"cand": 0, "new": 0}

    def copy(self):
        """a method-layer branch's own store: counters and the transferred set copied; the world and the pools shared."""
        c = Store.__new__(Store)
        c.__dict__.update(self.__dict__)
        c.state = {k: list(v) for k, v in self.state.items()}
        c.transferred = dict(self.transferred)
        c.u_seen = set(self.u_seen)
        c.last = dict(self.last)
        return c

    def froms(self):
        if not hasattr(self, "_froms"):
            self._froms = {int(t) for t in self.w.pos_from.values()}
        return self._froms

    def pool(self, c):
        """the room's fixed pool of new two-verb questions (a separate random stream per room)."""
        if c in self.pools:
            return self.pools[c]
        w = self.w
        rng = np.random.default_rng(self.seed * 1237 + c + 61_000_193)
        heard = [[int(x) for x in s] for _, s, _ in w.categories[c]["heard"]]
        verbs = sorted({x for s in heard for x in s if x in w.verb_of})
        froms = sorted({x for s in heard for x in s if x in self.froms()})
        objs = [int(o[1]) for o in w.categories[c].get("objects", [])]
        pics = [int(x) for o in w.categories[c].get("objects", []) for x in o[1:]]
        combos = [(a, b, f) for i, a in enumerate(verbs) for b in verbs[i + 1:] for f in froms
                  if (frozenset((a, b)), f) not in self.test_cores]
        out = []
        if combos and objs:
            for k in rng.permutation(len(combos))[:POOL]:
                a, b, f = combos[int(k)]
                obj = int(rng.choice(objs))
                fill = int(rng.choice([p for p in pics if p != obj]))
                s_ = [obj, a, b, f, fill]
                out.append(([s_[i] for i in rng.permutation(len(s_))], "reality"))
            mark = getattr(w, "tale_mark", None)
            if mark is not None:            # the same questions told as a tale (the tale frame)
                out += [([int(mark)] + list(ctx), "tale") for ctx, _ in list(out)]
        self.pools[c] = out
        return out

    def sleep(self, learner, seen):
        """after a session: the world model answers every pool question of the rooms met so far."""
        wm = getattr(learner.model, "wm", None)
        if wm is None:
            return
        keys, ctxs = [], []
        for c in seen:
            for j, (ctx, _) in enumerate(self.pool(c)):
                keys.append((c, j))
                ctxs.append(ctx)
        if not ctxs:
            return
        cx, _ = L.items_tensor([(x, 0) for x in ctxs])
        with torch.no_grad():
            t = wm.term(cx)
        fired = (t.sum(1) > 0).tolist()
        ans = t.argmax(1).tolist()
        n = 0
        for key, f, a in zip(keys, fired, ans):
            st = self.state.setdefault(key, [0, None])
            if f:
                n += 1
                st[0] = st[0] + 1 if st[1] == a else 1
                st[1] = a
            else:
                st[0], st[1] = 0, None
        self.last["cand"] = n

    def _useful(self, learner, keys):
        """the body alone (world model silent) gets the question wrong now."""
        if not keys:
            return []
        wm = learner.model.wm
        was = getattr(wm, "silent", False)
        wm.silent = True
        try:
            cx, tg = L.items_tensor([(self.pools[c][j][0], self.state[(c, j)][1]) for c, j in keys])
            with torch.no_grad():
                wrong = (learner.model.complete(cx).argmax(1) != tg).tolist()
        finally:
            wm.silent = was
        return [k for k, bad in zip(keys, wrong) if bad]

    def items(self, learner, s, heard_items):
        """the practice items this session (or branch) adds: everything transferred so far plus what passes now."""
        mode = self.mode
        if mode == "N":                     # the control: the store sleeps and records, nothing is transferred
            self.last["new"] = 0
            return []
        if mode == "M" and s.get("d25_mark") == "forget":
            for st in self.state.values():
                st[0], st[1] = 0, None
        conf = [k for k, st in self.state.items() if st[0] >= 1 and k not in self.transferred]
        if mode in ("U", "E"):
            new = conf
        else:
            thr = 2 if mode == "F" else int(s.get("d25_k", 2))
            if mode == "M" and s.get("d25_mark") == "keep":
                thr = 1
            new = [k for k in conf if self.state[k][0] >= thr and self.pools[k[0]][k[1]][1] == "reality"]
            new = self._useful(learner, new)
        if mode == "E":
            self.u_seen.update(new)
            self.e_count = len(self.u_seen)
            self.last["new"] = 0
            if not heard_items or not self.e_count:
                return []
            rng = np.random.default_rng(97_000 + self.seed * 131 + len(self.u_seen))
            pick = rng.choice(len(heard_items), self.e_count, replace=True)
            return [heard_items[int(i)] for i in pick]
        for k in new:
            c, j = k
            self.transferred[k] = (list(self.pools[c][j][0]), int(self.state[k][1]), self.pools[c][j][1])
        self.last["new"] = len(new)
        return [(ctx, a) for ctx, a, _ in self.transferred.values()]

    def record(self):
        tr = list(self.transferred.values())
        return {"d25 candidates": float(self.last["cand"]), "d25 transferred": float(len(tr) if self.mode != "E" else self.e_count),
                "d25 transferred tale": float(sum(1 for *_, f in tr if f == "tale")), "d25 new": float(self.last["new"])}
