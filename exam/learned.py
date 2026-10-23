"""Learned baselines for the sealed exam (G5 part 2; exam_spec/BASELINES_LEARNED.md). Each one hears only a life's exposure,
learns after every session from what it has heard so far, and answers the items due then -- never reading the answer keys --
in the simple baselines' answer files (DIR/<name>/life_XXX.jsonl), graded by the same grader.

Tokens of a moment (the networks): the sentence's words; the scene's bare tokens (baselines.elem: "Pc0", "K3"); position
tokens ("K3@2,0": the line's first cell); the same story's previous scene as "b:K3@1,0" (none at a story's first moment).
A token gets its slot when first heard -- no future token is known in advance.
  P   Pursuit (Stevens, Gleitman, Trueswell & Yang 2017), word -> bare scene token (its original word -> object use);
      gamma 0.02, lam 0.001 (Voyager Q5 part 2). Blank: score(c) = sum over the context scene's bare tokens o of P(o | c).
      Who / what / which: score(c) = sum over the item's words w of P(elem(c) | w).
  V   propose-but-verify (Trueswell, Medina, Hafri & Gleitman 2013), word -> bare scene token; a0 0.6, a1 0.8 (Q5 part 2).
      The same two item kinds, score = how many of the words' hypotheses are the candidate's token.
  MLP, TF  masked-token networks: from every heard moment, one word / bare / position token hidden, the rest of the moment
      (with its "b:" tokens) as a bag -> the hidden token (cross-entropy). TF = Pioneer n0's shape (Voyager's TinyTF: 2
      layers, width 32, 4 heads, feed-forward 64, bag input + a query token) with n0's standard training (AdamW weight decay
      0.01, dropout 0.1, minibatch 32, gradient clipping 1); MLP = the same width, bag mean -> 64 ReLU -> tokens. Embeddings
      learned from scratch (normal, sd 0.1). After every session: E x (the session's new items) samples drawn evenly from
      everything heard so far (replay on a fixed budget: every session costs the same; full replay of E epochs would be
      ~8x dearer -- ~100 CPU h for an edition and its machine check), one optimizer for the whole life, torch seed 90_000 +
      the life's number. lr and E from GRID, chosen on development seed
      1001 lives 0-2 (CHOSEN below; --grid prints the table).
      Blank: P(c | item words + context scene bare and position tokens). Who / what / which: P(elem(c) | item words).
      Where (now / before): P("K@x,y" | context scene as "b:" tokens + item words + K), K = the target's bare token.
      Choice / pair: an option's score = the mean over its position tokens t of log P(t | context scene as "b:" tokens +
      item words + the option's bare tokens); choice = the best, pair = the worse one is impossible.
Ties and zeros as the simple baselines (pick: the first of equal best, a zero score -> no answer; equal option scores -> no
answer). A position token never heard counts as probability 1e-12 in an option's mean.
    python3 -m exam.learned --edition DIR --out DIR [--workers 4]
    python3 -m exam.learned --grid [--dev 1001] [--lives 3] [--workers 4]"""
import argparse
import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import elem

LAST = 15
CAP = 4096                                          # token slots
D, FF, HEADS, LAYERS, HIDDEN = 32, 64, 4, 2, 64
NATIVE = {"batch": 32, "wd": 0.01, "dropout": 0.1, "clip": 1.0}
GRID = [(lr, ep) for lr in (1e-3, 3e-3) for ep in (10, 40)]
CHOSEN = {"MLP": (3e-3, 40), "TF": (3e-3, 40)}      # (lr, E), from --grid on dev seed 1001 lives 0-2 (2026-10-17:
                                                    # MLP 0.386, TF 0.398 -- both at the grid's corner; exam_dev/learned_grid_dev1001.txt)
LEARNED = ("P", "V", "MLP", "TF")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def bare_tokens(sc):
    return [elem(l["id"]) for l in sc["lines"]]


def pos_tokens(sc, prefix=""):
    return [f"{prefix}{elem(l['id'])}@{l['cells'][0][0]},{l['cells'][0][1]}" for l in sc["lines"]]


def moments(exposure):
    """per exposure record (file order): session, words, bare, pos, before (the same story's previous moment, 'b:')."""
    scenes = {(r["session"], r["story"], r["moment"]): r["scene"] for r in exposure}
    out = []
    for r in exposure:
        prev = scenes.get((r["session"], r["story"], r["moment"] - 1))
        out.append({"session": r["session"], "words": list(r["sentence"]), "bare": bare_tokens(r["scene"]),
                    "pos": pos_tokens(r["scene"]), "before": pos_tokens(prev, "b:") if prev else []})
    return out


def pick(cands, scores):
    best, arg = 0.0, None
    for c, s in zip(cands, scores):
        if s > best + 1e-12:
            best, arg = s, c
    return arg


def item_words(item):
    """baselines.Baselines.context_words: the context's and the question's words (the blank removed)."""
    ctx = item["context"]
    out = [w for s in ctx.get("sentences", []) for w in s]
    out += [w for m in ctx.get("moments", []) for w in m["sentence"]]
    out += [w for w in item["question"].get("sentence", []) if w != "___"]
    return out


def item_scenes(item):
    ctx = item["context"]
    return [ctx[k] for k in ("scene", "scene_after") if ctx.get(k)] + [m["scene"] for m in ctx.get("moments", [])]


def before_tokens(item):
    sc = item["context"].get("scene")
    return pos_tokens(sc, "b:") if sc else []


def is_due(it, s):
    return it["when"] == "every" or it["when"] == f"after_session_{s}" or (it["when"] == "end" and s == LAST)


# ---------------------------------------------------------------- word -> object learners
class Pursuit:
    def __init__(self, life, gamma=0.02, lam=0.001):
        self.rng = np.random.default_rng(91_000 + life)
        self.gamma, self.lam = gamma, lam
        self.A, self.objs = defaultdict(dict), set()

    def hear(self, moms):
        g = self.gamma
        for m in moms:
            objs = sorted(set(m["bare"]))
            self.objs.update(objs)
            if not objs:
                continue
            for w in sorted(set(m["words"])):
                row = self.A[w]
                if not row:                         # a new word: the object least claimed by any word (mutual exclusivity)
                    strongest = [max((r.get(o, 0.0) for r in self.A.values()), default=0.0) for o in objs]
                    mn = min(strongest)
                    cand = [o for o, v in zip(objs, strongest) if v == mn]
                    row[cand[int(self.rng.integers(len(cand)))]] = g
                    continue
                h = max(sorted(row), key=lambda o: row[o])
                if h in objs:
                    row[h] += g * (1 - row[h])
                else:
                    row[h] *= 1 - g
                    h2 = objs[int(self.rng.integers(len(objs)))]
                    row[h2] = row.get(h2, 0.0) + g * (1 - row.get(h2, 0.0))

    def p(self, o, w):
        row = self.A.get(w)
        if not row:
            return 0.0
        return (row.get(o, 0.0) + self.lam) / (sum(row.values()) + len(self.objs) * self.lam)

    def answer(self, item):
        fmt, q, cands = item["format"], item["question"], item["candidates"]
        if fmt == "blank":
            objs = {o for sc in item_scenes(item) for o in bare_tokens(sc)}
            return pick(cands, [sum(self.p(o, c) for o in objs) for c in cands])
        if fmt == "slot" and q["slot"] in ("who", "what", "which"):
            ws = item_words(item)
            return pick(cands, [sum(self.p(elem(c), w) for w in ws) for c in cands])
        return None


class PbV:
    def __init__(self, life, a0=0.6, a1=0.8):
        self.rng = np.random.default_rng(92_000 + life)
        self.a0, self.a1 = a0, a1
        self.hyp, self.conf = {}, {}

    def hear(self, moms):
        for m in moms:
            objs = sorted(set(m["bare"]))
            if not objs:
                continue
            for w in sorted(set(m["words"])):
                recalled = w in self.hyp and self.rng.random() < (self.a1 if self.conf[w] else self.a0)
                if recalled and self.hyp[w] in objs:
                    self.conf[w] = True
                else:
                    self.hyp[w] = objs[int(self.rng.integers(len(objs)))]
                    self.conf[w] = False

    def answer(self, item):
        fmt, q, cands = item["format"], item["question"], item["candidates"]
        if fmt == "blank":
            objs = {o for sc in item_scenes(item) for o in bare_tokens(sc)}
            return pick(cands, [float(self.hyp.get(c) in objs) for c in cands])
        if fmt == "slot" and q["slot"] in ("who", "what", "which"):
            ws = item_words(item)
            return pick(cands, [float(sum(self.hyp.get(w) == elem(c) for w in ws)) for c in cands])
        return None


# ---------------------------------------------------------------- masked-token networks
class TF(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(CAP, D) * 0.1)
        self.query = nn.Parameter(torch.randn(D) * 0.1)
        layer = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=NATIVE["dropout"], batch_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.bias = nn.Parameter(torch.zeros(CAP))

    def forward(self, ctx):
        x = self.emb[ctx.clamp(min=0)]
        q = self.query.expand(len(ctx), 1, -1)
        pad = torch.cat([torch.zeros(len(ctx), 1, dtype=torch.bool), ctx < 0], dim=1)
        h = self.enc(torch.cat([q, x], dim=1), src_key_padding_mask=pad)
        return h[:, 0] @ self.emb.T + self.bias


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(CAP, D) * 0.1)
        self.f = nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, CAP))

    def forward(self, ctx):
        m = (ctx >= 0).float().unsqueeze(-1)
        x = (self.emb[ctx.clamp(min=0)] * m).sum(1) / m.sum(1).clamp(min=1)
        return self.f(x)


class Net:
    def __init__(self, kind, life, lr, epochs):
        torch.manual_seed(90_000 + life)
        self.model = TF() if kind == "TF" else MLP()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=NATIVE["wd"])
        self.g = torch.Generator().manual_seed(90_000 + life)
        self.epochs, self.ix, self.ctx, self.tgt, self.n_new = epochs, {}, [], [], 0

    def hear(self, moms):
        n0 = len(self.tgt)
        for m in moms:
            toks = m["words"] + m["bare"] + m["pos"] + m["before"]
            for t in toks:
                if t not in self.ix:
                    if len(self.ix) >= CAP:
                        raise RuntimeError("token slots full")
                    self.ix[t] = len(self.ix)
            ids = [self.ix[t] for t in toks]
            for j in range(len(m["words"]) + len(m["bare"]) + len(m["pos"])):
                self.ctx.append(ids[:j] + ids[j + 1:])
                self.tgt.append(ids[j])
        self.n_new = len(self.tgt) - n0

    def practise(self):
        L = max(len(c) for c in self.ctx)
        X = torch.full((len(self.ctx), max(L, 1)), -1, dtype=torch.long)
        for i, c in enumerate(self.ctx):
            X[i, :len(c)] = torch.tensor(c, dtype=torch.long)
        Y = torch.tensor(self.tgt, dtype=torch.long)
        mask = torch.full((CAP,), float("-inf"))
        mask[:len(self.ix)] = 0.0
        need, order = self.epochs * self.n_new, []        # the budget: E x the session's new items, drawn from all heard
        while sum(len(o) for o in order) < need:
            order.append(torch.randperm(len(Y), generator=self.g))
        self.model.train()
        if need:
            for b in torch.cat(order)[:need].split(NATIVE["batch"]):
                self.opt.zero_grad()
                loss = F.cross_entropy(self.model(X[b]) + mask, Y[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), NATIVE["clip"])
                self.opt.step()
        self.model.eval()

    def probs(self, toks):
        ids = [self.ix[t] for t in toks if t in self.ix] or [-1]
        with torch.no_grad():
            lg = self.model(torch.tensor([ids], dtype=torch.long))[0, :len(self.ix)]
        return torch.softmax(lg, 0).numpy()

    def answer(self, item):
        fmt, q, cands, ix = item["format"], item["question"], item["candidates"], self.ix
        if fmt == "blank":
            ctx = item_words(item) + [t for sc in item_scenes(item) for t in bare_tokens(sc) + pos_tokens(sc)]
            p = self.probs(ctx)
            return pick(cands, [float(p[ix[c]]) if c in ix else 0.0 for c in cands])
        if fmt == "slot":
            slot = q["slot"]
            if slot in ("who", "what", "which"):
                p = self.probs(item_words(item))
                return pick(cands, [float(p[ix[elem(c)]]) if elem(c) in ix else 0.0 for c in cands])
            if slot in ("where", "where now", "where before"):
                K = elem(q["target"])
                p = self.probs(before_tokens(item) + item_words(item) + [K])
                toks = [f"{K}@{x},{y}" for x, y in cands]
                return pick(cands, [float(p[ix[t]]) if t in ix else 0.0 for t in toks])
            return None
        if fmt in ("choice", "pair"):
            opts, base = item["options"], before_tokens(item) + item_words(item)
            sc = []
            for o in opts:
                p = self.probs(base + bare_tokens(o["scene"]))
                ts = pos_tokens(o["scene"])
                sc.append(float(np.mean([math.log(max(float(p[ix[t]]), 1e-12)) if t in ix else math.log(1e-12)
                                         for t in ts])) if ts else math.log(1e-12))
            if fmt == "choice":
                return pick([o["label"] for o in opts], [s - min(sc) + 1e-6 for s in sc]) if len(set(sc)) > 1 else None
            if sc[0] == sc[1]:
                return None
            return {"impossible": opts[int(np.argmin(sc))]["label"]}
        return None


# ---------------------------------------------------------------- one life
def learners(life, names, cfg):
    out = {}
    for b in names:
        if b == "P":
            out[b] = Pursuit(life)
        elif b == "V":
            out[b] = PbV(life)
        else:
            lr, ep = cfg[b]
            out[b] = Net(b, life, lr, ep)
    return out


def live(life, exposure, items, names, cfg, when=None):
    """-> {name: [(item index, session, answer)]}: after each session, every item due then (or only at `when(item)`)."""
    torch.set_num_threads(1)
    moms = moments(exposure)
    ls = learners(life, names, cfg)
    out = {b: [] for b in names}
    for s in range(LAST + 1):
        new = [m for m in moms if m["session"] == s]
        for b, l in ls.items():
            l.hear(new)
            if isinstance(l, Net) and l.ctx:
                l.practise()
        for i, it in enumerate(items):
            on = (when(it) == s) if when else is_due(it, s)
            if not on:
                continue
            for b, l in ls.items():
                out[b].append((i, s, l.answer(it)))
    return out


def _edition_life(args):
    edition, life_dir, out_dir, cfg = args
    life = int(life_dir.rsplit("_", 1)[1])
    expo = load(os.path.join(edition, "lives", life_dir, "exposure.jsonl"))
    items = load(os.path.join(edition, "lives", life_dir, "items.jsonl"))
    res = live(life, expo, items, LEARNED, cfg)
    for b, rows in res.items():
        os.makedirs(os.path.join(out_dir, b), exist_ok=True)
        with open(os.path.join(out_dir, b, f"{life_dir}.jsonl"), "w", encoding="utf-8") as f:
            for i, s, a in rows:
                if a is not None:
                    f.write(json.dumps({"id": items[i]["id"], "after_session": s, "answer": a}, ensure_ascii=False) + "\n")
    return life_dir


def run(edition, out_dir, workers=4, cfg=None):
    cfg = cfg or CHOSEN
    if any(v is None for v in cfg.values()):
        raise SystemExit("CHOSEN is not fixed yet: run --grid on a development seed first")
    lives = sorted(d for d in os.listdir(os.path.join(edition, "lives")) if d.startswith("life_"))
    with Pool(workers) as pool:
        for d in pool.imap_unordered(_edition_life, [(edition, d, out_dir, cfg) for d in lives]):
            print("done", d, flush=True)


def _end_task(args):
    from .items import Items
    from .life import Life
    seed, life, cfg, cache = args
    L = Life(seed, life)
    items, _ = Items(L).build()
    res = live(life, L.records, items, LEARNED, cfg, when=lambda it: LAST)
    out = {b: [a for _, _, a in sorted(rows, key=lambda r: r[0])] for b, rows in res.items()}
    if cache:                                       # one file per life, written whole (a cut run loses at most the lives running)
        path = os.path.join(cache, f"seed{seed}_life{life:03d}.json")
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump({"cfg": {k: list(v) for k, v in cfg.items()}, "answers": out}, f, ensure_ascii=False)
        os.replace(path + ".part", path)
    return life, out


def end_answers(seed, n_lives, workers=4, cfg=None, cache=None):
    """the machine check's view (checks.py (b)): every item answered at the life's end; -> {life: {name: [answer]}}.
    cache: a folder for one file per life; lives already there (same CHOSEN) are read, not recomputed."""
    cfg = cfg or CHOSEN
    if any(v is None for v in cfg.values()):
        raise SystemExit("CHOSEN is not fixed yet: run --grid on a development seed first")
    done = {}
    if cache:
        os.makedirs(cache, exist_ok=True)
        for life in range(n_lives):
            path = os.path.join(cache, f"seed{seed}_life{life:03d}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
                if d["cfg"] == {k: list(v) for k, v in cfg.items()}:
                    done[life] = d["answers"]
    todo = [(seed, life, cfg, cache) for life in range(n_lives) if life not in done]
    if todo:
        with Pool(workers) as pool:
            for life, out in pool.imap_unordered(_end_task, todo):
                done[life] = out
                print(f"learned baselines: life {life} done ({len(done)}/{n_lives})", flush=True)
    return done


def judged_session(item):
    return LAST if item["when"] in ("every", "end") else int(item["when"].rsplit("_", 1)[1])


def _grid_task(args):
    from .checks import group, pass_line
    from .baselines import correct
    from .items import Items
    from .life import Life
    seed, life, net, lr, ep = args
    L = Life(seed, life)
    items, keys = Items(L).build()
    res = live(life, L.records, items, (net,), {net: (lr, ep)}, when=judged_session)[net]
    ok = [correct(items[i], keys[i], a) for i, s, a in res if pass_line(group(keys[i])) is not None]
    return net, lr, ep, life, float(np.mean(ok)), len(ok)


def grid(seed, n_lives, workers):
    tasks = [(seed, life, net, lr, ep) for net in ("MLP", "TF") for lr, ep in GRID for life in range(n_lives)]
    with Pool(workers) as pool:
        rows = pool.map(_grid_task, tasks, chunksize=1)
    table = defaultdict(list)
    for net, lr, ep, life, acc, n in rows:
        table[(net, lr, ep)].append(acc)
        print(f"{net} lr {lr} epochs {ep} life {life}: judged items {n}, accuracy {acc:.4f}", flush=True)
    chosen = {}
    for net in ("MLP", "TF"):
        best = max(GRID, key=lambda g: (np.mean(table[(net,) + g]), -GRID.index(g)))
        chosen[net] = best
        for g in GRID:
            print(f"  {net} {g}: mean {np.mean(table[(net,) + g]):.4f}" + ("  <- chosen" if g == best else ""))
    print("CHOSEN =", chosen)
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition")
    ap.add_argument("--out")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--dev", type=int, default=1001)
    ap.add_argument("--lives", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.grid:
        if not 1000 <= a.dev <= 1999:
            raise SystemExit("development seeds are 1000-1999 only")
        grid(a.dev, a.lives, a.workers)
    else:
        run(a.edition, a.out, a.workers)
