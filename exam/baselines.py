"""Simple baselines for the sealed exam (EXAM_DESIGN.md 5 and 4 (b); RULES 7). Each one answers every item of a life from
that life's exposure only -- never from the answer keys -- the same way a body would (it hears the exposure, then answers).
Fixed by hash with the generator before an opening day; run on the opening day's items with no change.

Tokens of an exposure moment: the sentence's words + one token per visible thing kind ("K3": the kind is visible) and per
visible person ("Pc1").
  M  majority: the candidate whose token was heard / seen most often.
  C  co-occurrence: sum over the item's context tokens of how often the candidate's token and that context token were in one
     moment (Voyager Q5's C, with scene tokens added).
  N  normalised co-occurrence: the same sum of count(a, b) / sqrt(freq(a) freq(b)) (Voyager Q5's N).
  B  bigram (blank items): the candidate maximising P(c | the token before the blank) P(the token after | c), add-one.
  L  lexical overlap ("when" items): the sentence sharing the most tokens with the question.
  T  transition-count table (NH's proposal, made sentence-aware): per person or thing, key = (the sentence's verb, its
     mention position in the sentence -- words mapped to persons / kinds by N --, the mention position of the thing right
     under it, whether it stands on the floor in one touching row with the named object and on which side, on which side
     the named person stands, its 4 neighbours) -> its displacement or "gone", with back-off to coarser keys; a thing that
     appears: (verb, its kind's mention position) -> rate. Counted over the exposure's consecutive moments (the scene
     before a sentence -> the scene after it). An option's score = the log-probability of the change from the last shown
     scene; choice = the best option, pair = the worse one is impossible. "where": the most likely change of every
     person and thing, applied sentence by sentence from the scene shown.
Runner (the "baseline runner" of EXAM_DESIGN.md 5): for every life of an edition and every session s, each baseline is
built from the exposure up to s and answers the items due after s ("every" items after each session, "after_session_k"
after k, "end" after 15), written as answer files the grader reads like a body's:
    python3 -m exam.baselines --edition DIR --out DIR      (-> DIR/<baseline>/life_XXX.jsonl)
Rules (Q5): candidates are the item's own; ties -> the first candidate in the item's order; a zero score -> no answer
(wrong); a pair with equal scores -> wrong."""
import math
from collections import Counter, defaultdict

import numpy as np

from .world import W, H

WH_TOKENS = {"누가", "무엇", "어디", "언제", "어떻게", "왜", "몇", "개", "who", "what", "where", "when", "how", "why", "many"}


def elem(i):
    """the visible token of an id: a person's own id, or a thing's kind (the shape is visible; the id number is not)."""
    return "P" + i if i.startswith("c") else "K" + i[1:].split("_")[0]


def scene_tokens(sc):
    return {elem(l["id"]) for l in sc["lines"]}


def cells_of(sc):
    return {l["id"]: tuple(l["cells"][0]) for l in sc["lines"]}


def code(cells, x, y, skip=None):
    """what fills cell (x, y): F floor (below the grid), W wall, E empty, P person, T thing."""
    if y < 0:
        return "F"
    if not (0 <= x < W) or y >= H:
        return "W"
    for i, p in cells.items():
        if p == (x, y) and i != skip:
            return "P" if i.startswith("c") else "T"
    return "E"


def nb4(cells, p, skip=None):
    x, y = p
    return "".join(code(cells, a, b, skip) for a, b in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)))


class Backoff:
    """counts of outcomes under keys of decreasing detail; P(o | key) with add-alpha back-off to the coarser key."""

    def __init__(self, levels):
        self.levels = levels
        self.n = [defaultdict(Counter) for _ in range(levels)]
        self.base = Counter()

    def add(self, keys, o):
        for lv, k in enumerate(keys):
            self.n[lv][k][o] += 1
        self.base[o] += 1

    def p(self, keys, o, alpha=1.0):
        tot = sum(self.base.values())
        pr = (self.base[o] + 1) / (tot + len(self.base) + 1)
        for lv in range(self.levels - 1, -1, -1):
            c = self.n[lv].get(keys[lv])
            if c:
                pr = (c[o] + alpha * pr) / (sum(c.values()) + alpha)
        return pr

    def best(self, keys):
        for lv in range(self.levels):
            c = self.n[lv].get(keys[lv])
            if c:
                return c.most_common(1)[0][0]
        return self.base.most_common(1)[0][0] if self.base else (0, 0)


def verb_of(sent):
    """the verb of a sentence of the exam grammar (Korean-like: the last word; English-like: the word after the name), or
    None for an existence sentence. (A reference baseline may know the public grammar.)"""
    if not sent:
        return None
    if any(t.isascii() and t.isalpha() for t in sent):
        return None if sent[0] == "there" or len(sent) < 2 else sent[1]
    return None if sent[-1] == "있다" else sent[-1]


class Baselines:
    def __init__(self, exposure):
        moments = [set(r["sentence"]) | scene_tokens(r["scene"]) for r in exposure]
        vocab = sorted(set().union(*moments))
        self.ix = {t: i for i, t in enumerate(vocab)}
        n = len(vocab)
        self.freq = np.zeros(n)
        self.co = np.zeros((n, n))
        for toks in moments:
            u = [self.ix[t] for t in toks]
            self.freq[u] += 1
            self.co[np.ix_(u, u)] += 1
        np.fill_diagonal(self.co, 0)
        self.norm = self.co / np.sqrt(np.outer(self.freq, self.freq) + 1e-12)
        self.big, self.uni = Counter(), Counter()
        for r in exposure:
            s = ["<s>"] + r["sentence"] + ["</s>"]
            for a, b in zip(s, s[1:]):
                self.big[(a, b)] += 1
                self.uni[a] += 1
        # word -> person / kind (for the where-table): the element with the highest N, if clearly above the next one
        elems = [t for t in vocab if t[0] in "PK" and len(t) > 1 and (t[1:].isdigit() or t[1] == "c")]
        self.ref = {}
        for w in vocab:
            if w in elems:
                continue
            sc = sorted(((self.norm[self.ix[w], self.ix[e]], e) for e in elems), reverse=True)
            if sc and sc[0][0] > 0 and self.co[self.ix[w], self.ix[sc[0][1]]] >= 2 and (len(sc) < 2 or sc[0][0] >= 2 * sc[1][0]):
                self.ref[w] = sc[0][1]
        # the transition table: the scene before a sentence -> the scene after it (consecutive moments of a story)
        self.move = Backoff(5)
        self.new_rate, self.verb_n = Counter(), Counter()
        stories = defaultdict(list)
        for r in exposure:
            stories[(r["session"], r["story"])].append(r)
        for rs in stories.values():
            rs.sort(key=lambda r: r["moment"])
            for a, b in zip(rs, rs[1:]):
                ca, cb = cells_of(a["scene"]), cells_of(b["scene"])
                ks = self.move_keys(ca, b["sentence"])
                for i, p in ca.items():
                    self.move.add(ks[i], (cb[i][0] - p[0], cb[i][1] - p[1]) if i in cb else "gone")
                v, men = verb_of(b["sentence"]), self.mentions(b["sentence"])
                self.verb_n[v] += 1
                for i in set(cb) - set(ca):
                    self.new_rate[(v, men.get(elem(i), 0))] += 1

    # ---- helpers
    def mentions(self, sent):
        """element -> its mention position in the sentence (1 = first referring word)."""
        out = {}
        for w in sent:
            e = self.ref.get(w)
            if e and e not in out:
                out[e] = len(out) + 1
        return out

    def move_keys(self, cells, sent):
        """the back-off keys of every person and thing of a scene for the sentence told next."""
        v, men = verb_of(sent), self.mentions(sent)
        obj = [i for i in cells if men.get(elem(i)) == 2]
        agent = [i for i in cells if men.get(elem(i)) == 1]
        out = {}
        for i, (x, y) in cells.items():
            typ = "P" if i.startswith("c") else "T"
            m = men.get(elem(i), 0)
            under = [j for j, q in cells.items() if q == (x, y - 1)]
            bm = men.get(elem(under[0]), 0) if under else -1
            row = 0
            if y == 0 and obj and cells[obj[0]][1] == 0 and obj[0] != i and typ == "T":
                ox = cells[obj[0]][0]
                step = 1 if x > ox else -1
                if all(any(q == (z, 0) and not j.startswith("c") for j, q in cells.items()) for z in range(ox + step, x, step)):
                    row = step
            ps = 0
            if agent and cells[agent[0]][1] == 0 and agent[0] != i:
                ps = 1 if x > cells[agent[0]][0] else -1
            nb = nb4(cells, (x, y), i)
            out[i] = ((typ, v, m, bm, row, ps, nb), (typ, v, m, bm, row, ps), (typ, v, m, bm), (typ, v, m), (typ,))
        return out

    def p_new(self, v, m):
        n = self.verb_n.get(v, 0)
        if n:
            return min(0.99, (self.new_rate[(v, m)] + 0.5) / (n + 1))
        tot = sum(self.verb_n.values())
        return min(0.99, (sum(c for (_, mm), c in self.new_rate.items() if mm == m) + 0.5) / (tot + 1))

    def score_cn(self, cand_tok, ctx, which):
        if cand_tok not in self.ix:
            return 0.0
        m = self.co if which == "C" else self.norm
        ci = self.ix[cand_tok]
        return float(sum(m[ci, self.ix[t]] for t in ctx if t in self.ix and t != cand_tok))

    def pick(self, cands, scores):
        best, arg = 0.0, None
        for c, s in zip(cands, scores):
            if s > best + 1e-12:
                best, arg = s, c
        return arg

    # ---- answering
    def context_words(self, item):
        ctx = item["context"]
        out = [w for s in ctx.get("sentences", []) for w in s]
        out += [w for m in ctx.get("moments", []) for w in m["sentence"]]
        out += [w for w in item["question"].get("sentence", []) if w != "___"]
        return out

    def context_scene_tokens(self, item):
        ctx = item["context"]
        out = set()
        for k in ("scene", "scene_after"):
            if ctx.get(k):
                out |= scene_tokens(ctx[k])
        for m in ctx.get("moments", []):
            out |= scene_tokens(m["scene"])
        return out

    def answer(self, item, b):
        """baseline b's answer to an item (None = no answer)."""
        fmt, q = item["format"], item["question"]
        cands = item["candidates"]
        if fmt == "blank":
            if b in "CN":
                ctx = self.context_words(item) + sorted(self.context_scene_tokens(item))
                return self.pick(cands, [self.score_cn(c, ctx, b) for c in cands])
            if b == "M":
                return self.pick(cands, [self.freq[self.ix[c]] if c in self.ix else 0 for c in cands])
            if b == "B":
                return self.bigram(item)
            return None
        if fmt == "slot":
            slot = q["slot"]
            if slot in ("who", "what", "which") and b in "CNM":
                if b == "M":
                    return self.pick(cands, [self.freq[self.ix[elem(c)]] if elem(c) in self.ix else 0 for c in cands])
                ctx = self.context_words(item)
                return self.pick(cands, [self.score_cn(elem(c), ctx, b) for c in cands])
            if slot == "when" and b == "L":
                qs = Counter(w for w in q["sentence"] if w not in WH_TOKENS)
                sents = item["context"]["sentences"]
                return self.pick(cands, [sum((Counter(sents[c]) & qs).values()) for c in cands])
            if slot in ("where", "where now", "where before") and b == "T":
                return self.where_answer(item)
            return None
        if fmt in ("choice", "pair"):
            opts = item["options"]
            if b == "T":
                sc = self.option_scores_t(item)
            elif b in "CN":
                ctx = self.context_words(item)
                sc = [np.mean([self.score_cn(t, ctx, b) for t in scene_tokens(o["scene"])] or [0]) for o in opts]
            elif b == "M":
                sc = [np.mean([self.freq[self.ix[t]] if t in self.ix else 0 for t in scene_tokens(o["scene"])] or [0])
                      for o in opts]
            else:
                return None
            if sc is None:
                return None
            if fmt == "choice":
                return self.pick([o["label"] for o in opts], [s - min(sc) + 1e-6 for s in sc]) if len(set(sc)) > 1 else None
            if sc[0] == sc[1]:
                return None
            return {"impossible": opts[int(np.argmin(sc))]["label"]}
        return None

    def bigram(self, item):
        s = item["question"].get("sentence", [])
        cands = item["candidates"]
        if "___" in s:
            j = s.index("___")
            before, after = (s[j - 1] if j > 0 else "<s>"), (s[j + 1] if j + 1 < len(s) else "</s>")
        else:                                       # a how-many question: the number stands where the wh-word is
            js = [j for j, w in enumerate(s) if w in WH_TOKENS]
            if not js:
                return None
            before = s[js[0] - 1] if js[0] > 0 else "<s>"
            after = s[js[-1] + 1] if js[-1] + 1 < len(s) else "</s>"
        V = len(self.uni) + 1
        sc = [math.log((self.big[(before, c)] + 1) / (self.uni[before] + V)) +
              math.log((self.big[(c, after)] + 1) / (self.uni[c] + V)) for c in cands]
        best = max(sc)
        return cands[sc.index(best)] if sc.count(best) < len(sc) else None

    def option_scores_t(self, item):
        ctx = item["context"]
        if ctx.get("moments"):
            before = cells_of(ctx["moments"][-1]["scene"])
        elif ctx.get("scene") is not None:
            before = cells_of(ctx["scene"])
        else:
            return None
        sent = ctx["sentences"][-1] if ctx.get("sentences") else []
        ks = self.move_keys(before, sent)
        v, men = verb_of(sent), self.mentions(sent)
        out = []
        for o in item["options"]:
            after = cells_of(o["scene"])
            s = 0.0
            for i, p in before.items():
                oc = (after[i][0] - p[0], after[i][1] - p[1]) if i in after else "gone"
                s += math.log(self.move.p(ks[i], oc))
            for i in set(after) - set(before):
                s += math.log(self.p_new(v, men.get(elem(i), 0)))
            out.append(s)
        return out

    def where_answer(self, item):
        ctx, q = item["context"], item["question"]
        cells = cells_of(ctx["moments"][-1]["scene"] if ctx.get("moments") else ctx["scene"])
        t = q["target"]
        hist = [cells.get(t)]
        for s in ctx.get("sentences", []):
            ks = self.move_keys(cells, s)
            new = {}
            for i, p in cells.items():
                o = self.move.best(ks[i])
                if o != "gone":
                    new[i] = (min(max(p[0] + o[0], 0), W - 1), min(max(p[1] + o[1], 0), H - 1))
            cells = new
            if cells.get(t) != hist[-1]:
                hist.append(cells.get(t))
        p = (hist[-2] if len(hist) >= 2 else None) if q["slot"] == "where before" else hist[-1]
        return list(p) if p is not None else None


BASELINES = ("M", "C", "N", "B", "L", "T")


def correct(item, key, ans):
    """is an answer right (pair: the impossible option named)."""
    if ans is None:
        return False
    if item["format"] == "pair":
        return isinstance(ans, dict) and ans.get("impossible") == key["answer"]["impossible"]
    return ans == key["answer"]


def run(edition, out_dir):
    import json
    import os
    from .grade import load, LAST
    lives = sorted(d for d in os.listdir(os.path.join(edition, "lives")) if d.startswith("life_"))
    for life in lives:
        items = load(os.path.join(edition, "lives", life, "items.jsonl"))
        expo = load(os.path.join(edition, "lives", life, "exposure.jsonl"))
        rows = {b: [] for b in BASELINES}
        for s in range(LAST + 1):
            due = [it for it in items if it["when"] == "every" or it["when"] == f"after_session_{s}"
                   or (it["when"] == "end" and s == LAST)]
            if not due:
                continue
            bl = Baselines([r for r in expo if r["session"] <= s])
            for it in due:
                for b in BASELINES:
                    a = bl.answer(it, b)
                    if a is not None:
                        rows[b].append({"id": it["id"], "after_session": s, "answer": a})
        for b, rs in rows.items():
            os.makedirs(os.path.join(out_dir, b), exist_ok=True)
            with open(os.path.join(out_dir, b, f"{life}.jsonl"), "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.edition, a.out)
