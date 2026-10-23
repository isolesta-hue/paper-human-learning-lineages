"""Content layer, first experiment: can knowledge of older categories speed up learning new ones? (roadmap step 5,
2026-09-23; the user's hypothesis that accumulated knowledge accelerates learning, and their design for it).

World (`data_v3`): schema (the same relation kinds recur in every category and show in the features; a new kind
appears in the second half of life) or random (identical, but no pair follows a template).
Pattern logic -- internal logic 2 given two roles, working without feedback:
  pattern memory   -- pairs that the connection seed finds strongly tied (P(b|a) >= STRONG, a heard >= MIN_SEEN
                      times) give a direction in memory space, m_b - m_a (Module A's memory vectors). Directions
                      are grouped: a direction joins the template it matches best if |cos| >= VIGILANCE and the
                      length ratio is within [0.5, 2] (assimilation: the template is refined as a running mean);
                      otherwise it starts a new template (accommodation). Templates are sign-free (relations here
                      go both ways). A template seen at least twice is established.
  pattern channel  -- understanding a sentence first with the patterns already known: for every known word pair
                      P[a, b] = best over established templates of conf x match x log(#known) (conf = n/(n+1),
                      match = 1 - |m_a +/- t - m_b| / |t|, kept if > 0.5), and the answer becomes
                      skill x (value + max over the context words of P[a, .]) -- the MAX, since one anchor word is
                      enough for a relation. The channel is not trained; it changes only when the templates do.
                      (First version: the same values were written into the trainable connection table as seeds.
                      That did nothing -- one-shot pairs identical with and without it -- because the answer
                      AVERAGES the context words' rows, so the one anchor is diluted by the fillers, and training
                      pushes non-target connections down, so the seeds were trained away.)
Learners
  D           -- design D + vector seed + directional connection seed (the base learner so far)
  D+pattern   -- the same + pattern logic
  MLP-feat    -- a plain MLP that is given the word features (input and output embeddings are linear maps of the
                 features), so that it too could in principle carry a relation over to new words
Life: first half of the categories (relation kinds R0-R2) then the second half (R0-R3), 2 categories per session,
8 sessions; review 0.5 of older sentences mixed in, lr 0.01, 200 epochs per session (no imagination, no cycles).
Measured: the newest categories' test accuracy at epochs 0/10/25/50/100/200 of their own session, in particular
for the ONE-SHOT pairs (heard in a single sentence; acceleration = their accuracy rising session after session --
design D cannot yet produce brand-new words at epoch 0, so early means 10-50 epochs), the "unseen" pairs (the two words were only ever heard apart: only a
carried-over template can link them), never-heard accuracy over everything met so far, and -- developer view only --
how well the learner's templates line up with the true relation kinds (|cos| to the true mean direction).
Connection seed for D and D+pattern alike: the directional seed x n/(n+1) (n = times the word was heard) -- the
same confidence rule as the templates, so that a pair seen together twice by chance does not outrank a template
confirmed many times (first version: the unweighted seed gave such chance pairs 5.4 vs a template seed of 3.5).
Representative values: STRONG 0.8, MIN_SEEN 3, VIGILANCE 0.8.
"""
import math
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_v3 as W
import lab2 as L
from run_seed_transfer import memorize
from run_unified_continual import sign_p

STRONG, MIN_SEEN, VIGILANCE = 0.8, 3, 0.8
EPOCHS, REVIEW, LR, PER_SESSION = 200, 0.5, 0.01, 2
CHECKS = [0, 10, 25, 50, 100, 200]
LEARNERS = ["D", "D+pattern", "MLP-feat"]
WORLDS = ["schema", "random"]


class ConfDirectionalHebbian(L.DirectionalHebbian2):
    """directional connection seed (finding 35) weighted by the same confidence rule the templates use:
    n / (n + 1) for a word heard n times -- two sightings are not treated as a law. Used by BOTH D and D+pattern."""

    def connections(self, seen):
        conf = self.with_ / (self.with_ + 1)
        return super().connections(seen) * conf[:, None]


class PatternMemory:
    def __init__(self):
        self.templates = []            # [mean direction, count]
        self.done = set()

    def absorb(self, hebb, known, memory):
        """assimilate / accommodate the directions of strongly tied pairs not seen before."""
        cond = hebb.counts / hebb.with_.clamp(min=1)[:, None]
        for a in known:
            if hebb.with_[a] < MIN_SEEN:
                continue
            for b in torch.nonzero(cond[a] >= STRONG).flatten().tolist():
                key = (min(a, b), max(a, b))
                if b not in known or key in self.done:
                    continue
                self.done.add(key)
                d = memory[key[1]] - memory[key[0]]
                best, best_cos = None, 0.0
                for t in self.templates:
                    c = float(F.cosine_similarity(d, t[0], dim=0))
                    ratio = float(d.norm() / t[0].norm().clamp(min=1e-9))
                    if abs(c) > abs(best_cos) and 0.5 <= ratio <= 2.0:
                        best, best_cos = t, c
                if best is not None and abs(best_cos) >= VIGILANCE:
                    aligned = d if best_cos > 0 else -d
                    best[1] += 1
                    best[0] = best[0] + (aligned - best[0]) / best[1]
                else:
                    self.templates.append([d.clone(), 1])

    def established(self):
        return [t for t in self.templates if t[1] >= 2]


def pattern_matrix(pattern, memory, known, V):
    """P[a, b] = best over established templates (and both signs) of conf x match x log(#known), for known words;
    match = 1 - |m_a +/- t - m_b| / |t|, kept if > 0.5 (the same rule as the first version's seeds)."""
    P = torch.zeros(V, V)
    words = torch.tensor(sorted(known))
    m = memory[words]
    for t, n in pattern.established():
        conf = n / (n + 1)
        for sign in (1, -1):
            match = 1 - torch.cdist(m + sign * t, m) / t.norm()
            match.fill_diagonal_(0)
            val = torch.where(match > 0.5, conf * match * math.log(len(known)), torch.zeros_like(match))
            sub = P[words][:, words]
            P[words.unsqueeze(1), words.unsqueeze(0)] = torch.maximum(sub, val)
    return P


class PatternKnowDo(L.KnowDo2):
    """design D with a pattern channel: answer = skill x (value + best template match from ANY context word).
    The channel is not trained (internal logic 2 works without feedback); the templates change only by
    assimilation / accommodation between sessions."""

    def __init__(self, V, n_cat, start_vectors, seed):
        super().__init__(V, n_cat, start_vectors, seed)
        self.register_buffer("pattern", torch.zeros(V, V))

    def pattern_term(self, ctx):
        mask = (ctx >= 0).unsqueeze(-1)
        rows = self.pattern[ctx.clamp(min=0)] * mask
        return rows.max(dim=1).values

    def complete(self, ctx, pterm=None):
        if pterm is None:
            pterm = self.pattern_term(ctx)
        return F.softplus(self.skill) * (L.ValueOnly2.complete(self, ctx) + pterm)


class FeatMLP(nn.Module):
    """plain MLP whose word embeddings are linear maps of the word features (input and output side)."""

    def __init__(self, vecs, n_cat, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.vecs = vecs
        self.enc = nn.Linear(vecs.shape[1], L.LATENT)
        self.out = nn.Linear(vecs.shape[1], L.LATENT)
        self.hidden = nn.Sequential(nn.Linear(L.LATENT, L.HIDDEN), nn.Tanh(), nn.Linear(L.HIDDEN, L.LATENT))
        self.cls_head = nn.Sequential(nn.Linear(L.LATENT, L.HIDDEN), nn.Tanh(), nn.Linear(L.HIDDEN, n_cat))

    def classify(self, ids):
        return self.cls_head(self.enc(self.vecs[ids]))

    def complete(self, ctx):
        return self.hidden(L.masked_mean(self.enc(self.vecs), ctx)) @ self.out(self.vecs).T


def acc_items(model, items):
    if not items:
        return float("nan")
    ctx, tgt = L.items_tensor(items)
    with torch.no_grad():
        return L.acc(model.complete(ctx), tgt)


def life(world_name, learner, seed):
    w = W.build(seed, schema=world_name == "schema")
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    vecs = torch.tensor(w.vecs)
    cats = torch.tensor(w.cats)
    memory, _ = memorize(vecs, seed)
    if learner == "MLP-feat":
        model = FeatMLP(vecs, w.n_cat, seed)
    elif learner == "D+pattern":
        model = PatternKnowDo(w.V, w.n_cat, memory, seed)
    else:
        model = L.KnowDo2(w.V, w.n_cat, memory, seed)
    hebb = ConfDirectionalHebbian(w.V)
    pattern = PatternMemory() if learner == "D+pattern" else None
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for c in rng.permutation(w.first_half)] + [int(c) for c in rng.permutation(w.second_half)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    true = W.true_pairs(w)
    known, seen, steps = set(), [], []
    for i, new_cats in enumerate(sessions):
        old = list(seen)
        seen += new_cats
        new_sents = [s for c in new_cats for s in sents[c]]
        new_words = {x for s in new_sents for x in s}
        known |= new_words
        old_sents = [s for c in old for s in sents[c]]
        review = [old_sents[j] for j in rng.choice(len(old_sents), int(round(len(old_sents) * REVIEW)), replace=False)] if old_sents else []
        heard = new_sents + review
        if learner != "MLP-feat":
            hebb.observe(heard)
            conn = hebb.connections(sorted(known))
            nw = torch.tensor(sorted(new_words))
            with torch.no_grad():
                model.assoc[nw, :] = conn[nw, :]
                model.assoc[:, nw] = conn[:, nw]
            if pattern is not None:
                model.pattern.copy_(pattern_matrix(pattern, memory, known, w.V))
        newest = [(ctx, t) for c in new_cats for kind, ctx, t in tests[c]]
        newest_unseen = [(ctx, t) for c in new_cats for kind, ctx, t in tests[c] if kind == "unseen"]
        newest_once = [(ctx, t) for c in new_cats for kind, ctx, t in tests[c] if kind == "once"]
        curve, curve_unseen, curve_once = {}, {}, {}
        ctx, tgt = L.items_tensor(D_positions(heard))
        bag = L.bag(ctx, w.V)
        extra = {"pterm": model.pattern_term(ctx)} if pattern is not None else {}
        ids = torch.tensor(sorted({x for s in heard for x in s}))
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for e in range(EPOCHS + 1):
            if e in CHECKS:
                curve[e] = acc_items(model, newest)
                curve_unseen[e] = acc_items(model, newest_unseen)
                curve_once[e] = acc_items(model, newest_once)
            if e == EPOCHS:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(bag, **extra), tgt) + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()
        if pattern is not None:                      # learn patterns from what was heard (after the session)
            pattern.absorb(hebb, known, memory)
        align = {}
        if pattern is not None:
            for k, pairs in true.items():
                T = torch.stack([memory[b] - memory[a] for a, b in pairs]).mean(0)
                est = pattern.established()
                align[k] = max((abs(float(F.cosine_similarity(T, t, dim=0))) for t, _ in est), default=0.0)
        all_items = [(ctx, t) for c in seen for kind, ctx, t in tests[c]]
        unseen_all = [(ctx, t) for c in seen for kind, ctx, t in tests[c] if kind == "unseen"]
        once_all = [(ctx, t) for c in seen for kind, ctx, t in tests[c] if kind == "once"]
        steps.append({"curve": curve, "curve unseen": curve_unseen, "curve once": curve_once,
                      "once all": acc_items(model, once_all), "never-heard": acc_items(model, all_items),
                      "unseen all": acc_items(model, unseen_all),
                      "older": acc_items(model, [(ctx, t) for c in old for kind, ctx, t in tests[c]]) if old else np.nan,
                      "templates": len(pattern.established()) if pattern is not None else np.nan,
                      "align": align})
    return steps


def D_positions(sentences):
    return [([x for j, x in enumerate(s) if j != i], y) for s in sentences for i, y in enumerate(s)]


def one_seed(seed):
    torch.set_num_threads(1)
    return {(wn, ln): life(wn, ln, seed) for wn in WORLDS for ln in LEARNERS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_sess = len(results[0][(WORLDS[0], LEARNERS[0])])
    print(f"=== {n_seeds} seeds; data_v3, {n_sess} sessions x {PER_SESSION} categories (first half R0-R2, second half + R3); "
          f"STRONG={STRONG} MIN_SEEN={MIN_SEEN} VIGILANCE={VIGILANCE} ===")
    for key, label in [("curve", "newest categories, accuracy at epoch"), ("curve once", "newest ONE-SHOT pairs, accuracy at epoch"),
                       ("curve unseen", "newest UNSEEN pairs, accuracy at epoch")]:
        print(f"\n--- {label} {CHECKS} (rows: session 1..{n_sess}) ---")
        for wn in WORLDS:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:9}")
                for i in range(n_sess):
                    row = [np.nanmean([r[(wn, ln)][i][key][e] for r in results]) for e in CHECKS]
                    print(f"      s{i+1} " + " ".join(f"{v:.3f}" for v in row))
    for key in ["never-heard", "once all", "unseen all", "older", "templates"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                print(f"{wn:6} {ln:9} " + " ".join(f"{v:6.3f}" for v in row))
    print("\n--- developer view: |cos| of the best learned template to each true relation kind (schema world, D+pattern) ---")
    for k in range(W.N_KINDS):
        row = [np.mean([r[("schema", "D+pattern")][i]["align"].get(k, 0.0) for r in results]) for i in range(n_sess)]
        print(f"R{k} " + " ".join(f"{v:.3f}" for v in row))

    print("\n--- acceleration: newest ONE-SHOT pairs at epoch 50, sessions 2-4 vs 6-8 minus session 1 / 5 ---")
    for wn in WORLDS:
        for ln in LEARNERS:
            e10 = np.array([[r[(wn, ln)][i]["curve once"][50] for i in range(n_sess)] for r in results])
            first = e10[:, 1:4].mean(1) - e10[:, 0]
            second = e10[:, 5:8].mean(1) - e10[:, 4]
            cells = []
            for lab, dd in [("first half gain", first), ("second half gain", second)]:
                n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
                cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
            print(f"{wn:6} {ln:9} " + "  ".join(cells))

    print("\n--- paired (schema world): D+pattern vs D, MLP-feat vs D; schema vs random for each learner ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", lambda r, g: np.mean([st["never-heard"] for st in r[g]])),
                         ("once@50", lambda r, g: np.mean([st["curve once"][50] for st in r[g]])),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("unseen end", lambda r, g: r[g][-1]["unseen all"]),
                         ("newest@10", lambda r, g: np.mean([st["curve"][10] for st in r[g]])),
                         ("newest@0", lambda r, g: np.mean([st["curve"][0] for st in r[g]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:34} " + "  ".join(cells))
    cmp(("schema", "D+pattern"), ("schema", "D"), "schema: D+pattern vs D")
    cmp(("schema", "MLP-feat"), ("schema", "D"), "schema: MLP-feat vs D")
    cmp(("schema", "D+pattern"), ("schema", "MLP-feat"), "schema: D+pattern vs MLP-feat")
    cmp(("random", "D+pattern"), ("random", "D"), "random: D+pattern vs D")
    for ln in LEARNERS:
        cmp(("schema", ln), ("random", ln), f"{ln}: schema vs random world")


if __name__ == "__main__":
    main()
