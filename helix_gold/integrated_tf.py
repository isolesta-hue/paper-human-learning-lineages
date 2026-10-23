"""A transformer control for the integrated learner (2026-09-26; the user's question: we built from the baby's behaviour
up; engineering can also start from an existing structure and fit it to the goal -- what if the logic and structure were a
transformer with attention, keeping the fixed centre and the fixed original memory?).

TLearner has Learner3's interface (prepare / train_to / finish / clone / judge_score / recombined_score), so the method
layer (run_integrated.py) runs on it unchanged. What stays: the same world, sessions, review, outside test; the original
memory (Module A, grown from the origin -- the "self" centre) as the word embeddings' fixed part, never trained (outer
layer, checked by the fingerprint); the category labels. What changes: instead of the standardised habit, templates,
whole-sentence memory and question-format procedures, a small transformer reads the context words (a set: no positions,
the sentences are bags) plus a query token and names the hidden word -- attention has to learn by itself which words
matter, what goes together, and the conjunctions.
  embedding(word) = memory(word) (frozen) + delta(word) (trained, starts at 0) -- like z_work starting from the memory
  2 layers, width 32 (the memory's size), 4 heads, feed-forward 64; output = query state . embedding + bias
Representative values: the sizes above (the smallest standard shape at this width), no dropout (the data is tiny).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import lab2 as L
from run_schema import D_positions, EPOCHS, CHECKS, acc_items
from integrated import Learner3


class TinyTF(nn.Module):
    def __init__(self, V, n_cat, memory, seed):
        super().__init__()
        torch.manual_seed(seed)
        d = memory.shape[1]
        self.register_buffer("mem", memory.clone())
        self.delta = nn.Parameter(torch.zeros_like(memory))
        self.query = nn.Parameter(torch.randn(d) * 0.1)
        layer = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.bias = nn.Parameter(torch.zeros(V))
        self.cls_head = nn.Sequential(nn.Linear(d, L.HIDDEN), nn.Tanh(), nn.Linear(L.HIDDEN, n_cat))

    def emb(self):
        return self.mem + self.delta

    def classify(self, ids):
        return self.cls_head(self.emb()[ids])

    def complete(self, ctx, **_):
        E = self.emb()
        x = E[ctx.clamp(min=0)]
        q = self.query.expand(len(ctx), 1, -1)
        h = self.enc(torch.cat([q, x], dim=1),
                     src_key_padding_mask=torch.cat([torch.zeros(len(ctx), 1, dtype=torch.bool), ctx < 0], dim=1))
        return h[:, 0] @ E.T + self.bias


class TLearner(Learner3):
    def __init__(self, w, seed, **_):
        super().__init__(w, seed, content=False)
        self.model = TinyTF(w.V, w.n_cat, self.memory, seed)

    def fingerprint(self):
        import hashlib
        return hashlib.sha256(self.memory.numpy().tobytes()).hexdigest() + (
            hashlib.sha256(self.model.mem.numpy().tobytes()).hexdigest() if hasattr(self, "model") and isinstance(self.model, TinyTF) else "")

    def clone(self):
        import copy
        c = copy.copy(self)
        c.model = copy.deepcopy(self.model)
        c.hebb = copy.deepcopy(self.hebb)
        c.known = set(self.known)
        c.FS, c.policy = dict(self.FS), dict(self.policy)
        c.st = None
        return c

    def prepare(self, new_sents, old_by_cat, s, rng, weights=None):
        import numpy as np
        import run_schema3 as S3
        new_words = {x for x in new_sents for x in x}
        self.known |= new_words
        review = self.pick_review(old_by_cat, s, rng, weights or {})
        heard = new_sents + review
        self.hebb.observe(heard)                    # the learner's own record (for its self-made questions), not its answer
        known_by_cat = {}
        for x in sorted(self.known):
            known_by_cat.setdefault(int(self.cats[x]), []).append(x)
        self.known_by_cat = known_by_cat
        exam = S3.procedures(self.model, S3.formatted(D_positions(new_sents), self.cats, known_by_cat,
                                                       np.random.default_rng(80_000 + self.seed + self.i)))
        for shape, procs in exam.items():
            for p, ok in procs.items():
                self.FS[(shape, p)] = self.FS.get((shape, p), 0) + float(ok.sum())
        for shape in ("blank", "options", "candidate"):
            self.policy[shape] = max((p for (sh, p) in self.FS if sh == shape), key=lambda p: self.FS[(shape, p)])
        phases = [heard] if s["order"] == "mixed" else [new_sents, heard]
        n_ep = [EPOCHS] if s["order"] == "mixed" else [EPOCHS // 2, EPOCHS - EPOCHS // 2]
        prepared = []
        for sents_p, n in zip(phases, n_ep):
            ctx, tgt = L.items_tensor(D_positions(sents_p))
            prepared.append((ctx, tgt, torch.tensor(sorted({x for x in sents_p for x in x})), n))
        self.st = {"phases": prepared, "opt": torch.optim.Adam(self.model.parameters(), lr=s["lr"]), "done": 0}

    def train_to(self, total, checks=None):
        model, cats, st = self.model, self.cats, self.st
        curves = {k: {} for k in (checks or {})}
        start = 0
        for ctx, tgt, ids, n in st["phases"]:
            for e in range(start, start + n):
                if st["done"] <= e < total:
                    if checks is not None and e in CHECKS:
                        for k, items in checks.items():
                            curves[k][e] = acc_items(model, items)
                    st["opt"].zero_grad()
                    loss = F.cross_entropy(model.complete(ctx), tgt) + F.cross_entropy(model.classify(ids), cats[ids])
                    loss.backward()
                    st["opt"].step()
            start += n
        st["done"] = max(st["done"], min(total, EPOCHS))
        if checks is not None and st["done"] == EPOCHS and EPOCHS in CHECKS:
            for k, items in checks.items():
                curves[k][EPOCHS] = acc_items(model, items)
        return curves

    def finish(self):
        self.st = None
        self.i += 1
        self.intact = self.intact and self.fingerprint() == self.memory_hash

    def judge_score(self, sentences):
        items = [it for s_ in sentences for it in D_positions([s_])]
        return acc_items(self.model, items) if items else float("nan")
