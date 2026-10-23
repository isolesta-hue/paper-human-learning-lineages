"""Dami's exam life (readiness step 3, 2026-10-17; user decision Q12 (가) -- the first sealed edition opens 10-21 and Dami
takes it as she stands). The shared exam's records go through conv-v1 (unchanged; its fingerprint is printed) into Dami's
BODY: Learner3 with Module A rehearsing until every element is recalled (+rehearse, readiness step 2), learning each exam
session as one of her sessions in the exam's order with the same session learning as her own lives (Learner3.session,
START + the glyph-world settings kind_first / cue_templates). After session s the body answers every item due then
(conv_v1.sessions_to_answer) WITHOUT learning -- a fingerprint of the whole learner before and after answering.

What v1 leaves out (known gaps, written before any exam score is seen):
  - her PARTS are not attached: each was built for her own world's input (motion moments with from / to role markers,
    verb and particle elements -- data_ascii), which conv-v1 does not make (it only moves the records; W6_DESIGN 2-1 "the
    body's known gap"). The fix belongs in her own world first, as a general principle (track B), not in the converter.
  - her METHOD LAYER (branches, judging, knobs -- run_long.life) is not run: that loop is built around her own world's
    rooms and phases. v1 runs the content learner's session directly.
How the body answers (fixed here before any exam score is seen):
  - blank / slot: the item read as ONE sequence (conv_v1.Query.sequence: the context moments joined by the boundary
    element, then the question with the blank removed) -> the body's completion (model.complete: the missing element given
    the context); a candidate's score = its log-probability. A sentence-number candidate ("when") is not supported: every
    such candidate gets the same score (a tie is wrong under the shared grader) -- as Ara's v1.
  - choice / pair: conv_v1's whole-scene likelihood; state_probs(t) = the completion of (sequence + t's picture) over the
    25 cells, and "not seen" = the probability left over (1 - the cells' total): the body's belief that t is at no cell.
python3 run_exam.py LIFE_DIR OUT_DIR [--smoke] [--sessions N]
  --smoke: a copy of the life with every word (exposure and items) replaced by a token drawn at random for each
  occurrence from a pool as large as the life's distinct words -- the words' identities removed, nothing learnable (the development cases are never training
  material). Checks only: no error, an answer row for every item due, the learner unchanged while answering."""
import copy
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

import conv_v1 as CV
from integrated import Learner3, START

torch.set_num_threads(1)


def learner_fp(me):
    """the whole learner's learnable state: the model's tensors, Module A's memory, the connection record, the chunks."""
    h = hashlib.sha256()
    for k, v in sorted(me.model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    h.update(me.memory.numpy().tobytes())
    for obj in (me.hebb, me.chunks, me.pattern):
        try:
            h.update(pickle.dumps(obj, protocol=4))
        except Exception:
            h.update(repr(type(obj)).encode())
    return h.hexdigest()


# Q20 (user decision 2026-10-21, after finding 169): from the next edition the query also carries the item's TARGET (the
# thing a "where" item asks about -- the exam gives it to every model; v1's reading dropped it, so C8 "where now / before",
# which have no sentence, never told the body which thing was asked). Items without a target are read exactly as in v1.
# The v1 answers were made with READ_TARGET = False.
READ_TARGET = True


class DamiBody:
    def __init__(self, me, w, read_target=None):
        self.me, self.w = me, w
        self.cells = sorted(w.pos_of)
        self.read_target = READ_TARGET if read_target is None else read_target

    def _logp(self, seq):
        from lab2 import pad
        with torch.no_grad():
            return torch.log_softmax(self.me.model.complete(pad([seq])), dim=1)[0]

    def query(self, q):
        seq = q.sequence(self.w.boundary)
        return seq + [q.target] if self.read_target and q.target is not None else seq

    def score(self, q):
        lp = self._logp(self.query(q))
        return [0.0 if isinstance(tok, tuple) else float(lp[tok]) for _, tok in q.cands]

    def state_probs(self, q, t):
        p = self._logp(self.query(q) + [t]).exp()
        out = {c: float(p[c]) for c in self.cells}
        out[CV.ABSENT] = max(0.0, 1.0 - sum(out.values()))
        return out


def scrambled_copy(life_dir, dst, seed=0):
    """the smoke life's input: every word replaced by a random token (fresh per occurrence)."""
    rng = np.random.default_rng(seed)
    os.makedirs(dst, exist_ok=True)
    words = set()                                   # the pool: as many random tokens as the life has distinct words, so the
    for fn in ("exposure.jsonl", "items.jsonl"):    # element table keeps its size (a fresh token per occurrence made 26,559
        for l in open(os.path.join(life_dir, fn)):  # elements instead of about 4,000 and ran out of memory)
            r = json.loads(l)
            words.update(r.get("sentence", []))
            ctx = r.get("context", {})
            for mo in ctx.get("moments", []):
                words.update(mo["sentence"])
            for s in ctx.get("sentences", []):
                words.update(s)
            words.update(x for x in (r.get("question", {}).get("sentence") or []) if x != "___")
    n_pool = max(1, len(words))
    tok = lambda: f"z{int(rng.integers(n_pool))}"
    with open(os.path.join(life_dir, "exposure.jsonl")) as f, open(os.path.join(dst, "exposure.jsonl"), "w") as g:
        for l in f:
            r = json.loads(l)
            r["sentence"] = [tok() for _ in r["sentence"]]
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(life_dir, "items.jsonl")) as f, open(os.path.join(dst, "items.jsonl"), "w") as g:
        for l in f:
            it = json.loads(l)
            ctx = it["context"]
            for mo in ctx.get("moments", []):
                mo["sentence"] = [tok() for _ in mo["sentence"]]
            if "sentences" in ctx:
                ctx["sentences"] = [[tok() for _ in s] for s in ctx["sentences"]]
            qs = it["question"].get("sentence")
            if qs is not None:
                it["question"]["sentence"] = [x if x == "___" else tok() for x in qs]
            if it["format"] == "blank":
                it["candidates"] = [tok() for _ in it["candidates"]]
            g.write(json.dumps(it, ensure_ascii=False) + "\n")
    return dst


def life_seed(life_dir):
    """the body's seed for an exam life = the life's number in the edition (life_000 -> 0, life_029 -> 29): an edition's
    30 lives are 30 seeds of Dami, each on its own world instance (2026-10-17, before any edition is opened)"""
    name = os.path.basename(os.path.normpath(life_dir))
    return int(name.rsplit("_", 1)[1]) if name.startswith("life_") else 0


def exam_life(life_dir, out_path, seed=0, n_sessions=None):
    t0 = time.process_time()
    w, qs = CV.convert_life(life_dir)
    me = Learner3(w, seed, content=True, where="linear", rehearse=True)
    rng = np.random.default_rng(70_000 + seed)
    settings = dict(START, kind_first=True, cue_templates=True)
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    rows, unchanged, seen = [], True, [0]
    n = len(w.sessions) if n_sessions is None else n_sessions
    for s in range(n):
        c = w.sessions[s]
        me.session(sents[c], {o: sents[o] for o in seen if sents[o]}, settings, rng)
        seen.append(c)
        body, before = DamiBody(me, w), learner_fp(me)
        for q in qs:
            if s in CV.sessions_to_answer(q.when):
                rows.append(CV.answer(body, q, s))
        unchanged = unchanged and learner_fp(me) == before
        print(f"session {s}: rows {len(rows)}, cpu {time.process_time() - t0:.0f}s, unchanged {unchanged}", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    due = sum(len([x for x in CV.sessions_to_answer(q.when) if x < n]) for q in qs)
    keys_ok = all(set(r["scores"]) == {k for k, _ in q.cands} for q in qs for r in rows if r["id"] == q.id)
    return {"rows": len(rows), "due": due, "keys_match": keys_ok, "unchanged_while_answering": unchanged,
            "rehearse_rounds": me.rehearse_rounds, "rehearse_recall": me.rehearse_recall,
            "cpu_s": round(time.process_time() - t0, 1), "V": w.V, "items": len(qs)}


if __name__ == "__main__":
    life_dir, out_dir = sys.argv[1], sys.argv[2]
    smoke = "--smoke" in sys.argv
    n = int(sys.argv[sys.argv.index("--sessions") + 1]) if "--sessions" in sys.argv else None
    os.makedirs(out_dir, exist_ok=True)
    src = scrambled_copy(life_dir, os.path.join(out_dir, "smoke_input")) if smoke else life_dir
    print(f"conv-v1 version {CV.VERSION}; smoke {smoke}; sessions {n or 'all'}", flush=True)
    res = exam_life(src, os.path.join(out_dir, os.path.basename(os.path.normpath(life_dir)) + ".jsonl"),
                    seed=life_seed(life_dir), n_sessions=n)
    print(json.dumps(res))
    print("exit 0")
