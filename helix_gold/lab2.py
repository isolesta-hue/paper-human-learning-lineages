"""Shared pieces for experiments on the harder data (data_v2.py): variable
sentence length, any vocabulary size.

Models
  ValueOnly2 -- design A of finding 27: the sentence answer is made only from
                the value tables (cosine of word vectors + connection table).
  Vanilla    -- an ordinary MLP: word embeddings (random start) -> mean of the
                context -> hidden layer -> one score per word; plus a category
                head. This is the "plain MLP" reference.
Internal logic 2 (connection seeds), for any vocabulary: Hebbian (PPMI) and
DirectionalHebbian (log P(b|a) + log(#known), finding 35).
Contexts are padded with -1 and averaged with a mask.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT, HIDDEN = 32, 64


def pad(contexts):
    L = max(len(c) for c in contexts)
    out = torch.full((len(contexts), L), -1, dtype=torch.long)
    for i, c in enumerate(contexts):
        out[i, :len(c)] = torch.tensor(c)
    return out


def items_tensor(items):
    """items: list of (context, target) -> (padded ctx, targets)."""
    return pad([c for c, _ in items]), torch.tensor([t for _, t in items])


def bag(ctx, V):
    """sparse (n x V) matrix: row i = the context words of item i, each weighted 1 / (number of words); padding -1 ignored."""
    mask = ctx >= 0
    rows = torch.arange(len(ctx)).unsqueeze(1).expand_as(ctx)[mask]
    w = (1.0 / mask.sum(1).clamp(min=1).float()).unsqueeze(1).expand_as(ctx)[mask]
    return torch.sparse_coo_tensor(torch.stack([rows, ctx[mask]]), w, (len(ctx), V), check_invariants=False).coalesce()


def masked_mean(table, ctx):
    """mean of the table rows of the context words, as bag @ table. ctx: padded word ids, or a bag made once
    with bag() (same result; reusing the bag over many epochs is faster)."""
    b = ctx if ctx.is_sparse else bag(ctx, table.shape[0])
    return torch.sparse.mm(b, table)


class ValueOnly2(nn.Module):
    def __init__(self, V, n_cat, start_vectors, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.z_work = nn.Parameter(start_vectors.clone())
        self.assoc = nn.Parameter(torch.zeros(V, V))
        self.cls_head = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, n_cat))

    def classify(self, ids):
        return self.cls_head(self.z_work[ids])

    def complete(self, ctx):
        if getattr(self, "attn", None) is not None:
            b = attention_bag(self, ctx)
            sim = F.normalize(torch.sparse.mm(b, self.z_work), dim=-1) @ F.normalize(self.z_work, dim=-1).T
            return sim + torch.sparse.mm(b, self.assoc)
        sim = F.normalize(masked_mean(self.z_work, ctx), dim=-1) @ F.normalize(self.z_work, dim=-1).T
        return sim + masked_mean(self.assoc, ctx)


def attention_bag(model, ctx):
    """the context mean with learned attention (2026-10-01, finding 100's next step; README "(5)(d)", Net2Net-style function
    preservation): the habit answers from the MEAN of the context words -- every word gets the same say, so in the glyph
    world the background pictures dilute the verb and the start position (finding 100: early in life the knobs barely change
    the outcome). Here each context word c gets the weight bag_c x exp(m^T A z_c), normalised per question, where m is the
    plain context mean and A (latent x latent) starts at ZERO: at birth every exp(...) is 1 and the weights are exactly the
    mean (the same answers); A learns with the rest of the habit, and attention moves to the words that tell. One matrix,
    not a query x key product: two zero factors would never get a gradient."""
    b = ctx if ctx.is_sparse else bag(ctx, model.z_work.shape[0])
    b = b.coalesce()
    idx, val = b.indices(), b.values()
    r, c = idx[0], idx[1]
    m = torch.sparse.mm(b, model.z_work)
    s = ((m[r] @ model.attn) * model.z_work[c]).sum(-1)
    n = b.shape[0]
    smax = torch.full((n,), float("-inf")).scatter_reduce(0, r, s, reduce="amax", include_self=True)
    e = val * torch.exp(s - smax[r])
    w = e / torch.zeros(n).index_add(0, r, e)[r]
    return torch.sparse_coo_tensor(idx, w, b.shape, check_invariants=False)


class Vanilla(nn.Module):
    def __init__(self, V, n_cat, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Parameter(torch.randn(V, LATENT) * 0.5)
        self.sent_head = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, V))
        self.cls_head = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, n_cat))

    def classify(self, ids):
        return self.cls_head(self.emb[ids])

    def complete(self, ctx):
        return self.sent_head(masked_mean(self.emb, ctx))


class Hebbian2:
    def __init__(self, V):
        self.V = V
        self.counts = torch.zeros(V, V)
        self.with_ = torch.zeros(V)

    def observe(self, sentences):
        for s in sentences:
            u = sorted(set(s))
            for a in u:
                self.with_[a] += 1
                for b in u:
                    if a != b:
                        self.counts[a, b] += 1

    def _mask(self, seen):
        m = torch.zeros(self.V, dtype=torch.bool)
        m[seen] = True
        return m[:, None] & m[None, :]

    def connections(self, seen):
        c = self.counts
        total, row, col = c.sum(), c.sum(1, keepdim=True), c.sum(0, keepdim=True)
        pmi = torch.nan_to_num(torch.log(c * total / (row * col)), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
        return pmi * self._mask(seen)


class DirectionalHebbian2(Hebbian2):
    def connections(self, seen):
        cond = self.counts / self.with_.clamp(min=1)[:, None]
        conn = torch.where(cond > 0, torch.log(cond.clamp(min=1e-12)) + math.log(len(seen)), torch.zeros_like(cond))
        return conn.clamp(min=0) * self._mask(seen)


def make_internal2(kind, V):
    return {"hebb": Hebbian2, "dhebb": DirectionalHebbian2}.get(kind, lambda V: None)(V)


def acc(logits, y):
    return (logits.argmax(-1) == y).float().mean().item() if len(y) else float("nan")


def train(model, items, word_ids, cats, epochs, lr=0.01):
    ctx, tgt = items_tensor(items)
    ctx = bag(ctx, model.assoc.shape[0] if hasattr(model, "assoc") else model.emb.shape[0])
    ids = torch.tensor(word_ids)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model.complete(ctx), tgt) + F.cross_entropy(model.classify(ids), cats[ids])
        loss.backward()
        opt.step()


def evaluate_units(model, unit_items_list):
    """accuracy per relation kind on (kind, context, hidden) items."""
    out = {}
    kinds = sorted({k for k, _, _ in unit_items_list})
    with torch.no_grad():
        for k in kinds + ["all"]:
            sel = [(c, t) for kk, c, t in unit_items_list if k == "all" or kk == k]
            if sel:
                ctx, tgt = items_tensor(sel)
                out[k] = acc(model.complete(ctx), tgt)
    return out


class KnowDo2(ValueOnly2):
    """Design D of finding 34: output = per-word production skill x memory value."""

    def __init__(self, V, n_cat, start_vectors, seed, a0=0.1):
        super().__init__(V, n_cat, start_vectors, seed)
        self.skill = nn.Parameter(torch.full((V,), math.log(math.expm1(a0))))

    def complete(self, ctx):
        return F.softplus(self.skill) * super().complete(ctx)
