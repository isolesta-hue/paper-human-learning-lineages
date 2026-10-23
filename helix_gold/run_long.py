"""The long test (roadmap step 6, 2026-09-26; the user's order after finding 69, step 3): a life twice as long with two
changes, the integrated learner with and without the method layer, and one option added per condition.

World life32 (data_v3 n_cat=32, shift2): 32 categories, 16 sessions -- sessions 1-8 the first half (rule set A, kinds R0-R2),
9-12 the new kind R3 arrives, 13-16 EVERY relation kind changes direction (rule set B). Finding 70's question: does a gap that
went flat widen again after a second change?
Conditions (tier 1, agreed with the user):
  content          -- the content layer with fixed start settings (the completion criterion's "fixed version")
  both             -- + the method layer (finding 69)
  both+wake        -- + the dormant parts that passed the screening (run_screen.py) as branch candidates: the method layer may
                      wake a part (and put it back to sleep) when its judge says so, like any knob (the dormant-parts rule)
  both+recomb      -- + self-made recombination practice from birth (finding 71: the setter's gain was the practice itself)
  both-low         -- low budget (finding 72's cliff: 0.475 < fixed 0.497)
  both-low+aware   -- low budget + budget-aware caution: the current settings' branch is never pruned early, and a change is
                      made only if the challenger beats the current settings on more judged categories than it loses to
                      them (finding 61's stakes rule as a sign test; finding 72: the less budget, the stricter the judgement)
Key comparisons, fixed before running (two-stage seed rule: 30 cached seeds; 100 only for an undecided key comparison):
  K1 completion: both - content, end (s16) and widening (mean gap s13-16 - mean gap s2-4)
  K2 both+wake vs both, K3 both+recomb vs both: life mean and end
  K4 both-low+aware vs both-low: end
"""
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import data_v3 as W
import lab2 as L
from run_unified_continual import sign_p
from run_schema import D_positions, PER_SESSION, acc_items
from integrated import Learner3, START, build_world, measure
import run_integrated as RI

# cache_long: results of the code before the optimisation pass (2026-09-28); cache_long_v2: after it -- never mixed in one
# comparison (float rounding changes individual lives)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_long_v2")
WORLD = "life32"
CONDITIONS = ["content", "both", "both+wake", "both+recomb", "both-low", "both-low+aware", "content+recomb"]
# content+recomb (added after the first 30-seed result): recombination practice WITHOUT the method layer -- is the late
# flattening of both+recomb / both+wake the practice's own property, or its interference with the judge (the judge's
# old-category questions, recombined_score, are made by the same generator as the practice: teaching to the test)?
RUNGS = {"low": [25, 50, 100, 200], "mid": [50, 100, 200], "high": [200]}
LEVELS = ["low", "mid", "high"]
# the dormant parts that may be woken (run_screen.py's verdict); each value is the part's alternatives to "asleep"
# screening (run_screen.py, 10 seeds): arbiter, ladder (both built on other matching than idf: life -0.023 to -0.031, 0-1/9-10)
# and versions (-0.002 to -0.003, 2/8) stay asleep -- not woken here, not removed
WAKE = {"chunk_mode": ["consensus"], "compose": [True], "setter": ["hard"], "fmt_practice": [True]}
ASLEEP = {"chunk_mode": "idf", "compose": False, "setter": None, "fmt_practice": False}


SIDE_KNOBS = {"side_w": [0.0, 0.5, 1.0, 2.0], "side_decay": [1.0, 0.5]}   # finding 86 (a): representative values
# finding 95 (route weights, 2026-09-30): how much of each route is added to the habit when answering -- the templates' term
# and the whole-sentence memory's evidence (finding 92: templates alone 0.96, the whole learner 0.87). Representative values.
ROUTE_KNOBS = {"pattern_w": [0.5, 1.0, 2.0, 4.0], "chunk_w": [0.5, 1.0, 2.0, 4.0]}


# the rediscovery test (2026-10-02, the user's question after finding 100: "is the misfit deeper than where the problem shows?"):
# the improvements WE found by hand and wrote into the fixed body, given back to the method layer as two-state knobs, starting
# from the old body: self-made recombination practice (finding 71/84), its own side part (finding 85/86), connections from
# first-hand evidence only (finding 82), and in the glyph world the four template repairs (finding 92/93). Does the method layer
# find them itself? If yes, its part was taken by us; if not, the method layer itself is the misfit.
REDISCOVER = {"setter": [None, "random"], "self_side": [False, True], "hebb_source": ["all", "world"]}
PLASTIC_KNOB = {"plastic": [False, True]}
FORGET_KNOB = {"forget_pick": [False, True]}     # finding 108: review the categories being forgotten
D25_KNOB = {"d25_k": [1, 2, 3]}                  # D25 K / M (2026-10-13): the transfer's repeat threshold, a method-layer knob
REDISCOVER_GLYPH = {"cue_kinds": [False, True], "cue_specific": [False, True], "cue_members_vote": [False, True],
                    "cue_pure": [False, True]}


def moves(settings, wake):
    """the method layer's knob moves, plus (wake) waking or putting back to sleep each dormant part, plus (finding 86 (a))
    the side part's knobs when the settings carry them: its weight when answering (0 = asleep) and how fast it forgets."""
    out = []
    for m in RI.MOVES:
        try:
            t = RI.apply(settings, m)
        except ValueError:                          # a rung the method layer added itself (finding 109): see life()
            t = None
        if t is not None:
            out.append((m, t))
    for k, vals in {**SIDE_KNOBS, **ROUTE_KNOBS, **PLASTIC_KNOB, **FORGET_KNOB, **D25_KNOB}.items():
        if k in settings:
            i = vals.index(settings[k])
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    out.append((f"{k}={vals[j]}", dict(settings, **{k: vals[j]})))
    if "d25_mark" in settings:                      # D25 M: the method layer's direct marks (keep now / forget)
        for v in ("none", "keep", "forget"):
            if v != settings["d25_mark"]:
                out.append((f"d25_mark={v}", dict(settings, d25_mark=v)))
    if settings.get("rediscover"):
        for k, vals in {**REDISCOVER, **(REDISCOVER_GLYPH if settings.get("cue_templates") else {})}.items():
            cur = settings.get(k, vals[0])
            for v in vals:
                if v != cur:
                    out.append((f"{k}={v}", dict(settings, **{k: v})))
    if wake:
        for k, alts in WAKE.items():
            cur = settings.get(k, ASLEEP[k])
            for v in [ASLEEP[k]] + alts:
                if v != cur:
                    out.append((f"{k}={v}", dict(settings, **{k: v})))
    return out


SQUEEZE_BUDGETS = (8, 32, 128)          # finding 110: the compressed table keeps 1/8, 1/32, 1/128 of the V x V numbers


def squeeze_table(A, how, frac):
    """finding 110 (record only): the connection table A (V x V) squeezed into frac of its numbers. "rank": a low-rank
    product U S V^T (2 V r numbers); "top": per row the k entries farthest from the row mean, the rest the row mean (a value
    and an index each); "both": half the budget low-rank, half top-k on what the low rank leaves."""
    V = A.shape[0]

    def rank(M, r):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return (U[:, :r] * S[:r]) @ Vh[:r]

    def top(M, k):
        mu = M.mean(1, keepdim=True)
        idx = (M - mu).abs().topk(k, dim=1).indices
        out = mu.expand_as(M).clone()
        out.scatter_(1, idx, M.gather(1, idx))
        return out

    n = max(1, int(V * frac / 2))
    if how == "rank":
        return rank(A, n)
    if how == "top":
        return top(A, n)
    low = rank(A, max(1, n // 2))
    return low + top(A - low, max(1, n // 2))


def squeeze_probe(me, tests, seen):
    """finding 110: how much of the outside test survives when the connection table is squeezed (record only -- the table
    is put back exactly); also the table's shape: how many of its numbers carry the energy, and how sparse the record is."""
    A = me.model.assoc
    keep = A.detach().clone()
    rec = {}
    items = {"never-heard": [(cx, t) for c in seen for kind, cx, t in tests[c]],
             "once all": [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "once"]}
    with torch.no_grad():
        S = torch.linalg.svdvals(keep)
        e = (S ** 2).cumsum(0) / (S ** 2).sum().clamp(min=1e-12)
        for d in SQUEEZE_BUDGETS:
            r = max(1, int(keep.shape[0] / d / 2))
            rec[f"sq energy r{d}"] = float(e[r - 1])
        rec["sq record density"] = float((me.hebb.counts != 0).float().mean()) if hasattr(me.hebb, "counts") else np.nan
        for how in ("rank", "top", "both"):
            for d in SQUEEZE_BUDGETS:
                A.copy_(squeeze_table(keep, how, 1.0 / d))
                for k, it in items.items():
                    rec[f"sq {how}{d} {k}"] = acc_items(me.model, it)
        A.copy_(keep)
    return rec


CF_W = (0.0, 0.5, 1.0, 1.5)             # finding 113: the part weights tried in the self-experiment (representative values)


def counterfactual(learner, items, ep=None):
    """finding 113, the self-experiment: because the answer's score splits exactly into its parts (finding 112), the learner
    can compute without relearning how many items it would get right if a part weighed w times as much -- for the
    whole-sentence memory (chunk) and the habit's connection table (assoc). ep: per item, the whole-sentence memory entry to
    leave out (the judge's leave-one-out, so held-out own sentences are not recalled verbatim); None = as answering."""
    if not items:
        return {}
    model = learner.model
    ctx, tgt = L.items_tensor(items)
    with torch.no_grad():
        on = model.chunk_on
        model.chunk_on = False
        z0 = model.complete(ctx)
        model.chunk_on = on
        if ep is None:
            z = model.complete(ctx)
        else:                                       # judge_score's answer, written out (finding 56 / 62 leave-one-out)
            z = z0
            if learner.content and on:
                ev = learner.chunks.evidence(ctx, exclude=torch.tensor(ep))
                if model.route == "rare":
                    ev = ev * model.chunk_gate(ctx, z0)
                elif model.route == "trust":
                    ev = ev * learner.chunks.route_gate(ctx, z0, ev, exclude=torch.tensor(ep))
                z = z0 + getattr(model, "chunk_w", 1.0) * ev
        chunk = z - z0
        assoc = answer_parts(model, ctx)[0]["assoc"]
        out = {}
        for wt in CF_W:
            out[f"chunk {wt}"] = L.acc(z0 + wt * chunk, tgt)
            out[f"assoc {wt}"] = L.acc(z - (1.0 - wt) * assoc, tgt)
    return out


def selfsim_self(learner, judged):
    """the self-experiment on the learner's OWN held-out sentences (the judge's), with the judge's leave-one-out."""
    items, ep = [], []
    for ss in judged.values():
        for s_ in ss:
            for cx, t in D_positions([s_]):
                items.append((cx, t))
                ep.append(learner.chunks.index.get(tuple(sorted(set(s_))), -1))
    cf = counterfactual(learner, items, ep)
    rec = {f"cf self {k}": v for k, v in cf.items()}
    rec["cf self judge"] = float(np.mean([learner.judge_score(ss) for ss in judged.values()])) if judged else np.nan
    rec["cf self pooled"] = cf.get("chunk 1.0", np.nan)
    return rec


def tune_weights(learner, judged, rng):
    """finding 115: the method layer sets the answering weights of the whole-sentence memory and the connection table by
    the world-shaped self-experiment (finding 114) -- for each, the weight in CF_W with the best accuracy on the learner's
    own world-shaped questions (others held at 1); the current weight is kept when it ties the best, otherwise ties go
    to 1. Nothing is relearned and the outside test is never seen."""
    items = [it for ss in judged.values() for it in learner.world_items(ss, rng)]
    if not items:
        return None
    m = learner.model
    keep = (getattr(m, "chunk_w", 1.0), getattr(m, "assoc_w", 1.0))
    m.chunk_w, m.assoc_w = 1.0, 1.0
    cf = counterfactual(learner, items)
    m.chunk_w, m.assoc_w = keep

    def pick(part, cur):
        accs = {wt: cf[f"{part} {wt}"] for wt in CF_W}
        best = max(accs.values())
        if cur in accs and accs[cur] >= best - 1e-12:
            return cur
        return max(CF_W, key=lambda wt: (accs[wt], wt == 1.0))
    return pick("chunk", keep[0]), pick("assoc", keep[1])


def grade_pick(learner, judged, pick):
    """finding 119 (the user's order, 2026-10-05: the learner scores its own self-experiment's predictions to set how far
    to trust them): the weights the eye chose LAST session are graded on THIS session's held-out sentences of the new
    categories -- material the eye never saw when it chose (the judge's leave-one-out, as finding 113). Per part, the
    realised gain of the chosen weight over weight 1 (others at 1); a part the eye left at 1 made no prediction.
    Nothing is relearned and the outside test is never seen."""
    items, ep = [], []
    for ss in judged.values():
        for s_ in ss:
            for cx, t in D_positions([s_]):
                items.append((cx, t))
                ep.append(learner.chunks.index.get(tuple(sorted(set(s_))), -1))
    if not items:
        return {}
    m = learner.model
    keep = (getattr(m, "chunk_w", 1.0), getattr(m, "assoc_w", 1.0))
    m.chunk_w, m.assoc_w = 1.0, 1.0
    cf = counterfactual(learner, items, ep)
    m.chunk_w, m.assoc_w = keep
    return {part: cf[f"{part} {w}"] - cf[f"{part} 1.0"] for part, w in zip(("chunk", "assoc"), pick) if w != 1.0}


def trust_of(grades):
    """finding 119: trust in the eye for one part = the share of its graded choices that were not worse than weight 1 on
    the fresh sentences, with the uniform prior (Laplace: (hits + 1) / (n + 2) -- the declared representative value, 0.5
    before any evidence)."""
    return (sum(g >= 0 for g in grades) + 1) / (len(grades) + 2)


PARTS = ("sim", "assoc", "pattern", "chunk", "side")


def answer_parts(model, ctx):
    """finding 112 (the user, 2026-10-03: the learner should see every number of its own answering): the answer's score
    split into the parts that add up to it -- habit similarity, habit connection table (the habit's standardisation is
    linear once its row mean and spread are fixed, so the split is exact), templates, whole-sentence memory (the rest), the
    self-made side part -- and the kind gate as a mask. Returns ({part: V-scores}, gate mask, the largest split error)."""
    from lab2 import attention_bag
    with torch.no_grad():
        if getattr(model, "attn", None) is not None:
            b = attention_bag(model, ctx)
            zb, am = torch.sparse.mm(b, model.z_work), torch.sparse.mm(b, model.assoc)
        else:
            zb, am = L.masked_mean(model.z_work, ctx), L.masked_mean(model.assoc, ctx)
        k = F.softplus(model.skill)
        a = k * (F.normalize(zb, dim=-1) @ F.normalize(model.z_work, dim=-1).T)
        c = k * am
        v = a + c
        sd = ((v - v.mean(1, keepdim=True)).pow(2).sum(1, keepdim=True) / (v.shape[1] - 1)).sqrt().clamp(min=1e-6)
        T = model.temperature(ctx).unsqueeze(1)
        parts = {"sim": (a - a.mean(1, keepdim=True)) / sd / T, "assoc": (c - c.mean(1, keepdim=True)) / sd / T}
        parts["pattern"] = model.pattern_term(ctx) if model.use_pattern else torch.zeros_like(v)
        parts["side"] = (model.side_w * L.masked_mean(model.side, ctx) if getattr(model, "side_on", False)
                         else torch.zeros_like(v))
        gate = getattr(model, "kind_gate", False)
        model.kind_gate = False
        total = model.complete(ctx)
        model.kind_gate = gate
        if getattr(model, "wm", None) is not None:  # world-model stage A: its own part (the split stays exact)
            parts["wm"] = model.wm.last_out if getattr(model.wm, "last", False) else model.wm.term(ctx)
        parts["chunk"] = total - parts["sim"] - parts["assoc"] - parts["pattern"] - parts["side"] - parts.get("wm", 0)
        blocked = model.kind_first(ctx, torch.zeros_like(v)) < -1e8
        # check: without whole-sentence memory the remainder must be ~0 (the split is exact)
        on = model.chunk_on
        model.chunk_on = False
        rest = (model.complete(ctx) if not gate else model.kind_first(ctx, model.complete(ctx)))
        model.chunk_on = on
        err = float((model.kind_first(ctx, parts["sim"] + parts["assoc"] + parts["pattern"] + parts["side"]) - rest)
                    .masked_fill(blocked, 0).abs().max())
    return parts, blocked, err


VOTE_T = (0.25, 0.5, 1.0, 2.0, 4.0)       # finding 124: temperatures tried, in units of the part's own score spread (declared)


def vote_probe(me, w, tests, seen, heard, rng):
    """finding 124 (record only; the user, 2026-10-07: the voting logic itself -- how the parts' answers are combined).
    Today the answer is the plain SUM of the parts' scores (finding 112's split), so every new part needed its units set
    by hand (findings 120, 123). Here each part's scores become a probability through its own temperature, chosen on the
    learner's OWN heard items of this session (the hidden word of each heard sentence; the scale tried is the part's own
    score spread x VOTE_T), and the parts vote as a product of experts (sum of log-probabilities). Compared with the sum on
    the outside test, the Korean stories and the chain questions. Caveat (declared): heard items flatter the
    whole-sentence memory (it remembers them) -- a held-out calibration is the next step if this shows headroom."""
    model = me.model
    calib = D_positions(heard)
    if not calib:
        return {}
    idx = rng.permutation(len(calib))[:400]
    ctx_c, tgt_c = L.items_tensor([calib[i] for i in idx])
    pc, bc, _ = answer_parts(model, ctx_c)
    temps = {}
    for p, sc in pc.items():
        spread = float(sc.std(dim=1).mean()) + 1e-6
        best, bt = -np.inf, 1.0
        for t in VOTE_T:
            lp = F.log_softmax((sc / (t * spread)).masked_fill(bc, -1e9), dim=1)
            ll = float(lp.gather(1, tgt_c.unsqueeze(1)).mean())
            if ll > best:
                best, bt = ll, t * spread
        temps[p] = bt
    rec = {}
    sets = {"never-heard": [(cx, t) for c in seen for kind, cx, t in tests[c]],
            "story": [(s_, a[0]) for c in seen for _, s_, a in w.categories[c].get("story", [])],
            "chain": [(s_, a[0]) for c in seen for _, s_, a in w.categories[c].get("chain", [])]}
    for name, items in sets.items():
        if not items:
            continue
        ctx, tgt = L.items_tensor(items)
        pt, bt_, _ = answer_parts(model, ctx)
        vote = sum(F.log_softmax((sc / temps[p]).masked_fill(bt_, -1e9), dim=1) for p, sc in pt.items())
        with torch.no_grad():
            plain = model.complete(ctx)
        rec[f"vote {name}"] = float((vote.argmax(1) == tgt).float().mean())
        rec[f"sum {name}"] = float((plain.argmax(1) == tgt).float().mean())
        # the user's follow-up (same day): vote by VECTORS and dot products, as language models do. Each part's
        # probability over the elements becomes a proposal vector = the expected element vector in the learner's original
        # memory (a sure part gives a long vector pointing at one element; an unsure part a short, blurred one). "vec": the
        # proposals are added and the answer is the element whose vector has the largest dot product with the sum.
        # "agree" (routing by agreement, Sabour et al. 2017): three rounds -- each part's weight = softmax over parts of its
        # proposal's dot product with the current consensus (scaled by the consensus' length), consensus = weighted sum.
        Z = me.memory
        props = torch.stack([F.softmax((sc / temps[p]).masked_fill(bt_, -1e9), dim=1) @ Z for p, sc in pt.items()], 1)
        allowed = ~bt_
        def pick(vsum):
            sim = vsum @ Z.T
            return sim.masked_fill(~allowed, -1e9).argmax(1)
        cons = props.sum(1)
        rec[f"vec {name}"] = float((pick(cons) == tgt).float().mean())
        for _ in range(3):
            logits = (props * cons.unsqueeze(1)).sum(-1) / (cons.norm(dim=1, keepdim=True) + 1e-9)
            wts = F.softmax(logits, dim=1)
            cons = (wts.unsqueeze(-1) * props).sum(1)
        rec[f"agree {name}"] = float((pick(cons) == tgt).float().mean())
    rec.update({f"vote T {p}": v for p, v in temps.items()})
    return rec


def speak_probe(me, w, seen):
    """stage-1 path step 3b (2026-10-08, record only; the user: run understanding backwards -- scene -> words). For each
    scene (an object's name, from(a), to(b)) the learner says one Korean sentence "[mover P1] [landmark P2] VERB" with what
    it has learned: P2 = the particle it learned marks the start (finding 123), P1 = the other particle it heard bound to
    names in heard stories; the landmark = a noun it remembers named at the start cell (its naming moments); the motion
    word = (body) the body's best motion word for [mover, from, to], or (book) the books read backwards (step 3a). Self-
    monitoring ("mon"; Levelt 1989: speakers check their own words with the comprehension system): the sentence is read back
    with the learner's own understanding (finding 123's path: start from the bound landmark, roll the motion book); if it
    does not land on the seen end cell, the next candidate (motion word x landmark) is tried; the first that reads back
    right is said (none: the first). Graded by the world: the noun with GA is the mover, the noun with ESEO is named at the
    start cell, the motion word is right (phrase order free). Control "swap": the two particles exchanged."""
    import lab2 as L_
    model, wm = me.model, me.model.wm
    wm.refresh()
    wm.learn_particles()
    items = [it for c in seen for it in w.categories[c].get("speak", [])]
    if not items or wm.start_particle is None:
        return {}
    cnt = {}
    for sent in getattr(me, "order_log", []):
        ks = wm.kind_of[sent]
        for i in range(1, len(sent)):
            if ks[i] == wm.k_particle and ks[i - 1] == wm.k_name and sent[i] != wm.start_particle:
                cnt[sent[i]] = cnt.get(sent[i], 0) + 1
    if not cnt:
        return {}
    p_start, p_mover = wm.start_particle, max(cnt, key=cnt.get)
    at_ids = wm.at_ids
    nouns_at = {}
    for n, t in wm.noun_at.items():
        nouns_at.setdefault(int(t), []).append(n)
    GA, ESEO = w.particle_of["GA"], w.particle_of["ESEO"]
    verb_ids = [int(x) for x in np.flatnonzero(wm.kind_of == wm.k_verb)]
    rec = {k: [] for k in ("body", "body mon", "book", "book mon", "swap", "readback", "lm found")}

    def read_back(sent, rooms, end):
        st_ = wm.story_state(sent, list(wm.kind_of[sent]))
        if st_ is None:
            return False
        mv = wm.book_move(sent[-1], rooms)
        return mv is not None and wm.land(st_ + mv) == end

    def right(bound, verb, truth):
        return bound.get(GA) == truth["mover"] and bound.get(ESEO) in truth["lms"] and verb == truth["verb"]

    for _, (mover, frm, to), truth in items:
        rooms = {int(wm.place[mover])} if wm.place[mover] >= 0 else set()
        at = int(at_ids[int((wm.mem[at_ids] + wm.shift_at - wm.mem[frm]).norm(dim=1).argmin())])
        lms = [n for n in nouns_at.get(at, []) if n != mover]
        # the nouns named at that cell IN THE MOVER'S ROOM first (positions are shared by every room; first smoke: a noun of
        # another room named at the same cell was said -- the read-back cannot see it, the cell is the same -- 0.75 -> 0.11)
        lms = [n for n in lms if wm.place[n] == wm.place[mover]] or lms
        rec["lm found"].append(float(bool(lms)))
        with torch.no_grad():
            sc = model.complete(torch.tensor([[mover, frm, to]]))[0]
        body_rank = sorted(verb_ids, key=lambda v: -float(sc[v]))
        seen_mv = wm.mem[to] - wm.mem[frm] - wm.shift
        book_rank = sorted([v for v in wm.books if wm.book_move(v, rooms) is not None],
                           key=lambda v: float((wm.book_move(v, rooms) - seen_mv).norm())) or body_rank
        if not lms:
            for k in ("body", "body mon", "book", "book mon", "swap", "readback"):
                rec[k].append(0.0)
            continue
        for name, rank in (("body", body_rank), ("book", book_rank)):
            rec[name].append(float(right({p_mover: mover, p_start: lms[0]}, rank[0], truth)))
            said = (rank[0], lms[0])
            for v in rank:
                hit = next((lm for lm in lms if read_back([mover, p_mover, lm, p_start, v], rooms, to)), None)
                if hit is not None:
                    said = (v, hit)
                    break
            rec[name + " mon"].append(float(right({p_mover: mover, p_start: said[1]}, said[0], truth)))
        rec["swap"].append(float(right({p_start: mover, p_mover: lms[0]}, book_rank[0], truth)))
        rec["readback"].append(float(read_back([mover, p_mover, lms[0], p_start, book_rank[0]], rooms, to)))
    out = {f"speak {k}": float(np.mean(v)) for k, v in rec.items() if v}
    out["speak particles ok"] = float(p_start == ESEO and p_mover == GA)
    return out


def tell_probe(me, w, seen):
    """stage-1 path step 3c (2026-10-08, record only): a sequence of pictures -> a multi-moment story. Each frame lists the
    moving objects' names, each followed by the cell it is at now. The learner tells it with what it has learned: between
    two frames the object whose cell changed is the moment's mover (event segmentation -- the backwards of step 2's
    tracking); the motion word by (book) the books read backwards on the seen move (step 3a) or (body) its body's best
    motion word for [mover, from, to]; a mover's first moment names a landmark (a noun of its room remembered at the start
    cell) with the start particle, later moments go on from where it is (step 2's form); moments joined by GO. Read-back:
    the told story is read with step 2's own tracking -- does each object land where the last frame shows it? Graded by
    the world: every moment's mover, motion word and (first moments) landmark right. Control "first": the mover taken to be
    the first object listed, not the one that changed."""
    model, wm = me.model, me.model.wm
    if wm.go_tok is None:
        return {}
    wm.refresh()
    wm.learn_particles()
    items = [it for c in seen for it in w.categories[c].get("tell", [])]
    if not items or wm.start_particle is None:
        return {}
    cnt = {}
    for sent in getattr(me, "order_log", []):
        ks = wm.kind_of[sent]
        for i in range(1, len(sent)):
            if ks[i] == wm.k_particle and ks[i - 1] == wm.k_name and sent[i] != wm.start_particle \
                    and sent[i] not in (wm.go_tok, wm.eodi_tok):
                cnt[sent[i]] = cnt.get(sent[i], 0) + 1
    if not cnt:
        return {}
    p_start, p_mover = wm.start_particle, max(cnt, key=cnt.get)
    nouns_at = {}
    for n, t in wm.noun_at.items():
        nouns_at.setdefault(int(t), []).append(n)
    verb_ids = [int(x) for x in np.flatnonzero(wm.kind_of == wm.k_verb)]
    near = lambda ids, v: int(ids[int((wm.mem[ids] - v).norm(dim=1).argmin())])
    rec = {k: [] for k in ("book", "body", "first", "readback", "segmented")}
    by_kind = {}

    def tell(frames, movers, rooms, how):
        clauses, started = [], set()
        for f0, f1 in zip(frames, frames[1:]):
            changed = [m for m in movers if f0[m] != f1[m]]
            if how == "first":
                changed = movers[:1]
            if len(changed) != 1:
                return None
            m = changed[0]
            if how == "body":
                frm = near(wm.from_ids, wm.mem[f0[m]] + wm.shift_at)
                to = near(wm.to_ids, wm.mem[f1[m]] + wm.shift_at + wm.shift)
                with torch.no_grad():
                    sc = model.complete(torch.tensor([[m, frm, to]]))[0]
                v = max(verb_ids, key=lambda x: float(sc[x]))
            else:
                seen_mv = wm.mem[f1[m]] - wm.mem[f0[m]]
                cand = [x for x in wm.books if wm.book_move(x, rooms) is not None]
                if not cand:
                    return None
                v = min(cand, key=lambda x: float((wm.book_move(x, rooms) - seen_mv).norm()))
            lm = None
            if m not in started:
                started.add(m)
                lms = [n for n in nouns_at.get(int(f0[m]), []) if n not in movers]
                lms = [n for n in lms if wm.place[n] == wm.place[m]] or lms
                if not lms:
                    return None
                lm = lms[0]
            clauses.append((m, v, lm))
        return clauses

    for kind_, sent, truth in items:
        go = [i for i, x in enumerate(sent) if x == wm.go_tok]
        parts_ = [sent[i + 1:j] for i, j in zip([-1] + go, go + [len(sent)])]
        frames = [{f[i]: f[i + 1] for i in range(0, len(f), 2)} for f in parts_]
        movers = [f for f in parts_[0][::2]]
        rooms = {int(wm.place[movers[0]])} if wm.place[movers[0]] >= 0 else set()
        res = {}
        for how in ("book", "body", "first"):
            cl = tell(frames, movers, rooms, how)
            res[how] = cl
            ok = cl is not None and len(cl) == len(truth) and all(
                m == tm and v == tv and ((tl is None and lm is None) or (tl is not None and lm in tl))
                for (m, v, lm), (tm, tv, tl) in zip(cl, truth))
            rec[how].append(float(ok))
            if how == "book":
                by_kind.setdefault(kind_, []).append(float(ok))
        rec["segmented"].append(float(res["book"] is not None))
        cl = res["book"]
        rb = False
        if cl is not None:
            story = []
            for m, v, lm in cl:
                story += [m, p_mover] + ([lm, p_start] if lm is not None else []) + [v, wm.go_tok]
            rb = True
            for m in movers:
                r_ = wm.track_state(story + [m, p_mover, wm.eodi_tok], rooms)
                end_to = near(wm.to_ids, wm.mem[frames[-1][m]] + wm.shift_at + wm.shift)
                rb = rb and r_ is not None and wm.land(r_[0]) == end_to
        rec["readback"].append(float(rb))
    out = {f"tell {k}": float(np.mean(v)) for k, v in rec.items() if v}
    out.update({f"{k} book": float(np.mean(v)) for k, v in by_kind.items()})
    return out


ERR_OWN = ("kind", "topic", "pulled", "near", "other")


def error_probe(me, w, tests, seen):
    """finding 112 (record only): the error-type eye. Every wrong answer on the outside test items is labelled twice:
    OWN -- with what the learner itself knows: kind (its input channel tag differs from the hidden one's; glyph world),
    topic (the answer's category -- taught label for heard words, its own classifier otherwise -- differs from the hidden
    one's), pulled (its own Hebbian record ties the context MORE to the answer than to the hidden one: the association
    pulled it away; first version "heard together at all" was true for ~every wrong answer, smoke test), near (the answer is among the hidden one's 5 nearest in its
    own memory space), other; the first that applies. DEVELOPER truth -- with the world's labels: topic (w.cats differ),
    background (the answer is a filler / room word of a context word's category), near (glyph: both positions, one cell apart).
    Returns shares of all items per own label, and per (own, developer) pair the counts for agreement."""
    model = me.model
    items = [(cx, t) for c in seen for kind, cx, t in tests[c]]
    ctx, tgt = L.items_tensor(items)
    parts, blocked, split_err = answer_parts(model, ctx)
    with torch.no_grad():
        pred = model.complete(ctx).argmax(-1).tolist()
        z = F.normalize(model.z_work, dim=-1)
        # the category labels the learner was TAUGHT (the textbook's names for heard words); its own classifier for the rest
        guess = model.classify(torch.arange(w.V)).argmax(-1).tolist()
        own_cat = [int(me.cats[x]) if x in me.known else guess[x] for x in range(w.V)]
    counts = me.hebb.counts
    ko = getattr(model, "kind_of", None)
    fill = {}
    for c in range(w.n_cat):
        for x in w.categories[c].get("fillers", w.categories[c].get("words", [])):
            fill.setdefault(x, set()).add(c)
    rec = {f"err {k}": 0.0 for k in ERR_OWN}
    pairs = {f"errv {o} {d}": 0.0 for o, d in (("topic", "topic"), ("pulled", "background"), ("near", "near"))}
    pairs.update({f"errv dev {d}": 0.0 for d in ("topic", "background", "near")})
    pairs.update({f"errv own {o}": 0.0 for o in ("topic", "pulled", "near")})
    rec.update({f"cause {q}": 0.0 for q in PARTS + ("gate",)})
    rec.update({f"margin {q}": 0.0 for q in PARTS})
    n_wrong = 0
    row = []
    for i_, ((cx, t), p) in enumerate(zip(items, pred)):
        if p == t:
            continue
        row.append(i_)
        n_wrong += 1
        near_t = set(torch.topk(z @ z[t], 6).indices.tolist()) - {t}
        own = {"kind": ko is not None and int(ko[p]) != int(ko[t]),
               "topic": own_cat[p] != own_cat[t],
               "pulled": float(counts[cx, p].sum()) > float(counts[cx, t].sum()),
               "near": p in near_t}
        lab = next((k for k in ERR_OWN[:-1] if own[k]), "other")
        rec[f"err {lab}"] += 1
        # the part that made the wrong answer win: the largest share of score(answer) - score(hidden), or the kind gate
        j = n_wrong - 1
        if bool(blocked[row[j], t]):
            cause = "gate"
        else:
            m = {q: float(parts[q][row[j], p] - parts[q][row[j], t]) for q in PARTS}
            cause = max(m, key=m.get)
            for q in PARTS:
                rec[f"margin {q}"] += m[q]
        rec[f"cause {cause}"] += 1
        tc = int(w.cats[t])
        dev = {"topic": int(w.cats[p]) != tc,
               "background": bool({int(w.cats[c_]) for c_ in cx} & fill.get(p, set())),
               "near": (p in getattr(w, "pos_of", {}) and t in getattr(w, "pos_of", {}) and
                        max(abs(w.pos_of[p][0] - w.pos_of[t][0]), abs(w.pos_of[p][1] - w.pos_of[t][1])) == 1)}
        for o, d in (("topic", "topic"), ("pulled", "background"), ("near", "near")):
            pairs[f"errv own {o}"] += own[o]
            pairs[f"errv dev {d}"] += dev[d]
            pairs[f"errv {o} {d}"] += own[o] and dev[d]
    n = max(len(items), 1)
    rec = {k: v / (max(n_wrong, 1) if k.startswith(("cause", "margin")) else n) for k, v in rec.items()}
    rec["err wrong"] = n_wrong / n
    rec["parts split error"] = split_err
    rec.update(pairs)
    return rec


def sp_keys(model, items, w, tag):
    """W3: scene -> word ("sp word") and word -> scene ("sp scene"); "4" = answered among the item's four candidates (the
    four spatial words / the four relations' cells: right when the answer scores strictly highest), plain = among every
    element; per relation for the four-way keys."""
    import lab2 as L_
    out = {}
    names = {t: w.names[t].upper() for t in getattr(w, "sp_words", [])}
    for kind in ("word", "scene"):
        sub = [(s_, a_) for k_, s_, a_ in items if k_ == f"sp {kind}"]
        if not sub:
            continue
        cx_, _ = L_.items_tensor([(s_, a_[0]) for s_, a_ in sub])
        with torch.no_grad():
            lp_ = F.log_softmax(model.complete(cx_), dim=1)
        cand = torch.tensor([a_ for _, a_ in sub])
        sc = lp_.gather(1, cand)
        right4 = (sc[:, 0] > sc[:, 1:].max(1).values).float()
        out[f"sp {kind}{tag}"] = float((lp_.argmax(1) == cand[:, 0]).float().mean())
        out[f"sp {kind}4{tag}"] = float(right4.mean())
        rel_of = [names.get(a_[0]) if kind == "word" else names.get(s_[-1]) for s_, a_ in sub]
        for rel in ("WI", "ARAE", "ANE", "YEOPE"):
            m_ = torch.tensor([r_ == rel for r_ in rel_of])
            if m_.any():
                out[f"sp {kind}4 {rel}{tag}"] = float(right4[m_].mean())
    return out


def four_way_keys(model, items, tag, prefix):
    """four-way keys (right when the answer scores strictly highest among the item's candidates): the mean over all
    items ("{prefix}{tag}") and per item kind ("{kind}{tag}")."""
    import lab2 as L_
    out = {}
    cx_, _ = L_.items_tensor([(s_, a_[0]) for _, s_, a_ in items])
    with torch.no_grad():
        lp_ = F.log_softmax(model.complete(cx_), dim=1)
    right = torch.tensor([float(lp_[i, a_[0]] > lp_[i, a_[1:]].max()) for i, (_, _, a_) in enumerate(items)])
    out[f"{prefix}{tag}"] = float(right.mean())
    kinds_ = [k_ for k_, _, _ in items]
    for key in sorted(set(kinds_)):
        out[f"{key}{tag}"] = float(right[torch.tensor([k_ == key for k_ in kinds_])].mean())
    return out


def d79_logits(model, cx, channel_of):
    """D79 v1, the relational equals sign (the user's idea 10-21; Falkner, Levi & Carpenter 1999, McNeil et al. 2006, Knuth et
    al. 2006), at ANSWER time only (learning untouched). The body's usual answer is "operational": the question's elements
    in, the element that usually goes with them out (finding 168: inside a frame the frame's usual words win). Here, for an
    item whose usual top answer is a WORD and which shows at least one PICTURE: left = a candidate word x (the unknown),
    right = what is seen (the item's pictures -- the "what" stream; places are the "where" stream and do not name a thing);
    x is chosen so that the two sides are equal -- the body's own memory of x (its completion of the one-element context
    [x]) must account for every picture seen: R(x) = sum over the pictures e of log p(e | x). The candidates are the words the
    usual answer allows (its kind gate and known mask: the frame decides the KIND of x, Q14 requirement (2)); among them only
    R decides WHICH. Other items: the usual answer, unchanged."""
    with torch.no_grad():
        base = model.complete(cx)
        ch = torch.as_tensor(channel_of)
        allowed = base > -1e8
        cand = torch.nonzero((ch == 1) & allowed.any(0)).squeeze(1)
        if len(cand) == 0:
            return base
        mem = torch.log_softmax(model.complete(cand.view(-1, 1)), dim=1)       # |cand| x V: what each word goes with
        pos = {int(t): i for i, t in enumerate(cand.tolist())}
        out = base.clone()
        top = base.argmax(-1)
        for i in range(len(cx)):
            if int(ch[top[i]]) != 1:
                continue
            pics = [int(t) for t in cx[i].tolist() if t >= 0 and int(ch[t]) == 2]
            if not pics:
                continue
            ok = [int(t) for t in cand.tolist() if bool(allowed[i, t])]
            row = torch.full_like(base[i], -1e9)
            r = mem[[pos[t] for t in ok]][:, pics].sum(1)
            row[ok] = r
            out[i] = row
    return out


def acc_scored(model, items, scorer):
    """acc_items with another answer rule (scorer(model, ctx) -> V-scores)."""
    import lab2 as L_
    if not items:
        return float("nan")
    ctx, tgt = L_.items_tensor(items)
    with torch.no_grad():
        return L_.acc(scorer(model, ctx), tgt)


class EpisodeStore:
    """D80 v1 (D80_DESIGN.md; the user's question and decision 10-21), at answer time only -- the habits learn as before.
    (가) every moment the world says, with its stamp = the order heard in the life (rooms in arrival order, a room's moments
    in the world's order); (나) kept apart from the habits; a picture's TIME LINE = the elements of one channel in the SEEN
    moments (no word in them) that hold the picture and exactly one element of that channel, in stamp order; three time
    relations (a small designed vocabulary): "last" (now), "before" (the last value different from the last), "first";
    (다) the D79 equals sign: in every heard moment that holds words, exactly one picture and exactly one other element (an
    answer), each relation applied to the picture's time line up to that moment is checked against the answer heard; per
    word set the relation with the best Laplace belief (hits + 1) / (seen + 2) is used if that belief > 1/2.
    v1.1 (2026-10-22, the user's decision "응, 너의 추천대로 진행해줘."): the table also keeps the CHANNEL of the answers
    heard with the word set (where / word / picture), and mode "EQ1" reads the time line in that learned channel instead
    of the channel of the usual top answer (finding 171: "where now" questions often had a picture on top, so v1 never
    opened the store for them). Modes "E" and "EQ" are unchanged."""
    RELS = ("last", "before", "first")

    def __init__(self, channel_of, books=None):
        self.ch = [int(x) for x in channel_of]
        self.moments = []
        self.table = {}
        self.books = books              # D80 v2: the scene part's books -- a sentence heard alone adds an expected cell

    def add(self, heard):
        self.moments += [tuple(int(x) for x in s_) for _, s_, _ in heard]

    def line(self, pic, chan, upto=None):
        if self.books is not None and chan == 0:
            return self.told_line(pic, upto)
        out = []
        for m in self.moments[:upto]:
            if pic in m and not any(self.ch[t] == 1 for t in m):
                vals = [t for t in m if t != pic and self.ch[t] == chan]
                if len(vals) == 1:
                    out.append(vals[0])
        return out

    def told_line(self, pic, upto=None):
        """D80 v2 (2026-10-22): the where time line with what was TOLD: a moment of words only (a sentence) whose counted
        mover is the picture adds an expected cell = the line's last cell + the counted move (on the grid only); the cell
        seen right after it (before the next moment with words) takes its place -- what is seen replaces what was said."""
        out, pending = [], None
        for m in self.moments[:upto]:
            if any(self.ch[t] == 1 for t in m):
                if pending is not None:
                    out.append(pending)
                    pending = None
                if out and out[-1] in self.books.pos_of and all(self.ch[t] == 1 for t in m):
                    told = self.books.told([t for t in m if self.books.is_word[t]])
                    if told is not None and told[0] == pic:
                        x, y = self.books.pos_of[out[-1]]
                        pending = self.books.at_of.get((x + told[1][0], y + told[1][1]))
                continue
            if pic in m:
                vals = [t for t in m if t != pic and self.ch[t] == 0]
                if len(vals) == 1:
                    out.append(vals[0])
                    pending = None
        if pending is not None:
            out.append(pending)
        return out

    @staticmethod
    def rel(name, vals):
        if not vals:
            return None
        if name == "last":
            return vals[-1]
        if name == "first":
            return vals[0]
        prev = [v for v in vals if v != vals[-1]]
        return prev[-1] if prev else None

    def learn(self):
        counts = {}
        for i, m in enumerate(self.moments):
            words = frozenset(t for t in m if self.ch[t] == 1)
            pics = [t for t in m if self.ch[t] == 2]
            rest = [t for t in m if self.ch[t] != 1 and t not in pics]
            if not words or len(pics) != 1 or len(rest) != 1:
                continue
            chan = self.ch[rest[0]]
            vals = self.line(pics[0], chan, upto=i)
            for r in self.RELS:
                c = counts.setdefault((words, r, chan), [0, 0])
                c[0] += self.rel(r, vals) == rest[0]
                c[1] += 1
        self.table = {}
        for (words, r, chan), (h, n) in counts.items():
            b = (h + 1) / (n + 2)
            if b > 0.5 and b > self.table.get(words, (None, 0.5))[1]:
                self.table[words] = (r, b, chan)

    def scores(self, model, cx, mode):
        """mode "E": the last value of the question's picture's time line; "EQ": the relation learned for the question's
        word set (none learned: the usual answer). Only for a question with exactly one picture, in the channel of the
        usual top answer, and only an element the usual answer allows; otherwise the usual answer."""
        with torch.no_grad():
            base = model.complete(cx)
        out = base.clone()
        top = base.argmax(-1)
        for i in range(len(cx)):
            q = [int(t) for t in cx[i].tolist() if t >= 0]
            pics = [t for t in q if self.ch[t] == 2]
            if len(pics) != 1:
                continue
            chan = self.ch[int(top[i])]
            if mode == "E":
                r = "last"
            else:
                r, _, chan1 = self.table.get(frozenset(t for t in q if self.ch[t] == 1), (None, None, None))
                if r is None:
                    continue
                if mode == "EQ1":
                    chan = chan1
            v = self.rel(r, self.line(pics[0], chan))
            if v is None or not bool(base[i, v] > -1e8):
                continue
            out[i, v] = base[i].max() + 1.0
        return out


def seq_keys(model, w, epi, rooms, tag, tests, epi2=None):
    """D80 v1 keys: per answer rule (U usual, E, EQ) -- "seq now / before / still" (the room's cat["seq"]; whole vocabulary,
    top answer), and the old keys: motion tests (R*, once, unseen) and names (name*) from the rooms' test lists, and the
    scene tests ("scene", right when the answer scores strictly highest among the item's candidates)."""
    import lab2 as L_
    rules = {"U": lambda m_, c_: m_.complete(c_), "E": lambda m_, c_: epi.scores(m_, c_, "E"),
             "EQ": lambda m_, c_: epi.scores(m_, c_, "EQ"), "EQ1": lambda m_, c_: epi.scores(m_, c_, "EQ1")}
    if epi2 is not None:                            # D80 v2: rule EQ1 on the store whose time lines hold what was told
        rules["EQ2"] = lambda m_, c_: epi2.scores(m_, c_, "EQ1")
    out = {}
    groups = {}
    for c in rooms:
        for k_, s_, ans in w.categories[c].get("seq", []):
            groups.setdefault(k_, []).append((s_, ans[0]))
        for kind, cx_, t in tests.get(c, []):
            g = "old motion" if (kind[:1] == "R" or kind in ("once", "unseen")) else "old name" if kind.startswith("name") else None
            if g:
                groups.setdefault(g, []).append((cx_, t))
    sc_items = [x for c in rooms for x in w.categories[c].get("scene", [])]
    for nm, rule in rules.items():
        for g, items in groups.items():
            out[f"{g} {nm}{tag}"] = acc_scored(model, items, rule)
            out[f"{g} n{tag}"] = float(len(items))
        if sc_items:
            cx_, _ = L_.items_tensor([(s_, a_[0]) for _, s_, a_ in sc_items])
            lp_ = rule(model, cx_)
            right = [float(lp_[i, a_[0]] > lp_[i, a_[1:]].max()) for i, (_, _, a_) in enumerate(sc_items)]
            out[f"old scene {nm}{tag}"] = float(np.mean(right))
    if not tag.endswith("all"):
        for nm_, e_ in (("seq", epi), ("seq2", epi2)):
            for words, (r, b, chan) in (e_.table.items() if e_ is not None else ()):
                out[f"{nm_} rel {'+'.join(sorted(w.names[t] for t in words))}"] = float(EpisodeStore.RELS.index(r))
                out[f"{nm_} chan {'+'.join(sorted(w.names[t] for t in words))}"] = float(chan)
    return out


def nwf_keys(model, w, rooms, tag, scorer=None):
    """Q14 frame test ("@nwf", data_ascii nwf=True): the room's "nwf" pairs (bare, framed), the same item without and with
    the frame THERE, IS. Counts (floats, so that a reader pools items exactly over sessions and seeds), per n and pooled
    ("nwfb{n}" / "nwfb"): items, bare right, framed right, bare right & framed wrong ("b+f-"), the reverse ("b-f+"), and what
    the top answer was (whole vocabulary, as acc_items -- ties go to the lower index) on the framed items the bare got right
    but the framed missed: a common word, a frame word, another name, or anything else. Also the room-names reading (right
    when the answer scores strictly highest among the room's four new names -- the exam's candidate form, report only) and
    the frame's own use ("nwbc" / "nwfc": a common picture given, its word asked, without / with the frame)."""
    import lab2 as L_
    out = {}
    items = [(c, x) for c in rooms for x in w.categories[c].get("nwf", [])]
    if not items:
        return out
    cx_, tg_ = L_.items_tensor([(s_, a_[0]) for _, (_, s_, a_) in items])
    with torch.no_grad():
        sc = model.complete(cx_) if scorer is None else scorer(model, cx_)
    top = sc.argmax(-1)
    right = (top == tg_).tolist()
    common, frame = set(w.nwf_common), set(w.nwf_frame)
    names_all = {int(o[0]) for c in range(1, w.n_cat) for o in w.categories[c]["objects"]}

    def what(t):
        return "common" if t in common else "frame" if t in frame else "name" if t in names_all else "other"
    acc = {}
    for i in range(0, len(items), 2):
        (c, (kb, _, ab)), (_, (kf, _, _)) = items[i], items[i + 1]
        if kb in ("nwbc",):
            for key, j in (("nwbc", i), ("nwfc", i + 1)):
                acc.setdefault(f"{key} n", 0.0)
                acc.setdefault(f"{key} right", 0.0)
                acc[f"{key} n"] += 1
                acc[f"{key} right"] += right[j]
            continue
        rn = [int(o[0]) for o in w.categories[c]["objects"]]
        others = [x for x in rn if x != int(ab[0])]
        four = [float(sc[j, ab[0]] > sc[j, others].max()) for j in (i, i + 1)]
        for key in (f"nwfb{kb[3:]}", "nwfb"):
            for part, v in (("n", 1), ("bare right", right[i]), ("framed right", right[i + 1]),
                            ("b+f-", right[i] and not right[i + 1]), ("b-f+", right[i + 1] and not right[i]),
                            ("bare four", four[0]), ("framed four", four[1])):
                acc[f"{key} {part}"] = acc.get(f"{key} {part}", 0.0) + float(v)
            if right[i] and not right[i + 1]:
                k_ = f"{key} flip {what(int(top[i + 1]))}"
                acc[k_] = acc.get(k_, 0.0) + 1.0
            if not right[i + 1]:
                k_ = f"{key} framed err {what(int(top[i + 1]))}"
                acc[k_] = acc.get(k_, 0.0) + 1.0
    for k_, v in acc.items():
        out[f"{k_}{tag}"] = v
    return out


def d72_keys(model, w, seen, tag, scorer=None, conf=False):
    """D72 (the learning-power test, D72_DESIGN.md): per new word j, the times the world has said it so far in the rooms
    met ("d72 k{j}" -- the world's moments, not the learner's rehearsals; "d72 dens{j}" its density) and its four-way keys
    (right when the answer scores strictly highest among the item's four candidates): "cat" -- the novel things' picture
    -> word ("d72 word {j}") and word -> picture ("d72 pic {j}"), asked from birth on (k = 0 is the first point); "verb"
    -- the arrival cell among the four new words' ("d72 verb {j}"), in the rooms met. scorer: another answer rule (as
    acc_scored), the usual answer when None. conf (D82 first step, record only): per kind also "d82 <kind> conf" -- the
    mean confidence, the top candidate's share when the item's four candidates' scores are normalised among themselves
    (0.25 = all four tie) -- and "d82 <kind> pairs", every item's [confidence, right]."""
    import lab2 as L_
    out = {}
    if not tag:
        for j, t_ in enumerate(w.d72_words):
            out[f"d72 k{j}"] = float(sum(t_ in s_ for c in seen for k_, s_, _ in w.categories[c]["heard"] if k_ == "d72-heard"))
            out[f"d72 dens{j}"] = float(w.d72_dens[j])
    items = list(w.d72_tests) + [x for c in seen for x in w.categories[c].get("d72", [])]
    if not items:
        return out
    cx_, _ = L_.items_tensor([(s_, a_[0]) for _, s_, a_ in items])
    with torch.no_grad():
        lp_ = F.log_softmax(model.complete(cx_) if scorer is None else scorer(model, cx_), dim=1)
    sc = lp_.gather(1, torch.tensor([a_ for _, _, a_ in items]))
    right = (sc[:, 0] > sc[:, 1:].max(1).values).float()
    # (2026-10-18, the D72 diagnosis -- diag_d72cat.py: subject 1's picture -> word items were all four candidates
    # blocked by the kind gate, a tie scored wrong) the share of items whose four candidates all tie: the body abstains
    tied = (sc.max(1).values == sc.min(1).values).float()
    kinds_ = [k_ for k_, _, _ in items]
    for key in sorted(set(kinds_)):
        sel = torch.tensor([k_ == key for k_ in kinds_])
        out[f"{key}{tag}"] = float(right[sel].mean())
        if getattr(w, "d72", None) not in (None, "cat", "verb"):     # every subject from 2 on
            out[f"{key}{tag} abstain"] = float(tied[sel].mean())
    if conf:                                        # D82: the answer's confidence among the item's own candidates
        cf = torch.softmax(sc, dim=1).max(1).values
        for key in sorted(set(kinds_)):
            sel = torch.tensor([k_ == key for k_ in kinds_])
            out[f"d82 {key}{tag} conf"] = float(cf[sel].mean())
            out[f"d82 {key}{tag} pairs"] = [[round(float(a), 5), float(b)] for a, b in zip(cf[sel], right[sel])]
    return out


def when_keys(model, items, tag):
    """W4: "when" asked by an arrival cell never said in the story ("when to") and in the exam v1 form ("when echo"); right
    when the answer scores strictly highest AMONG THE STORY'S MOMENT WORDS (plain), "all" = among every element; by story
    length (two / three / four moments)."""
    import lab2 as L_
    out = {}
    for form in ("to", "echo"):
        sub = [(k_, s_, a_) for k_, s_, a_ in items if k_.startswith(f"when {form} ")]
        if not sub:
            continue
        cx_, _ = L_.items_tensor([(s_, a_[0]) for _, s_, a_ in sub])
        with torch.no_grad():
            lp_ = F.log_softmax(model.complete(cx_), dim=1)
        right = []
        for r_, (_, _, a_) in enumerate(sub):
            sc = lp_[r_, torch.tensor(a_)]
            right.append(float(sc[0] > sc[1:].max()))
        right = torch.tensor(right)
        out[f"when {form}{tag}"] = float(right.mean())
        out[f"when {form} all{tag}"] = float((lp_.argmax(1) == torch.tensor([a_[0] for _, _, a_ in sub])).float().mean())
        for kw in ("two", "three", "four"):
            m_ = torch.tensor([k_ == f"when {form} {kw}" for k_, _, _ in sub])
            if m_.any():
                out[f"when {form} {kw}{tag}"] = float(right[m_].mean())
    return out


def ten_keys(model, items, tag):
    """W5: numbers 1-10 (criteria 3 and 4). "ten count n" / "ten next n" / "ten box g": right when the answer scores strictly
    highest among the ten number words (plain), "all" = among every element; "ten pick n": the told number word scored after
    each of the item's four stories (n and three other counts) -- right when it scores strictly highest after the story with
    n (the exam's number -> scene choice); "voe ten g": the wrong count more surprising than the right one (ties half), as
    "voe number". "new" = the held and far pairs pooled (combinations never heard)."""
    import lab2 as L_
    out = {}

    def scores(sub):
        cx_, _ = L_.items_tensor([(s_, a_[0]) for s_, a_ in sub])
        with torch.no_grad():
            return F.log_softmax(model.complete(cx_), dim=1)
    for form in ("count", "next", "box"):
        sub = [(k_, s_, a_) for k_, s_, a_ in items if k_.startswith(f"ten {form} ")]
        if not sub:
            continue
        lp_ = scores([(s_, a_) for _, s_, a_ in sub])
        cand = torch.tensor([a_ for _, _, a_ in sub])
        sc = lp_.gather(1, cand)
        right = (sc[:, 0] > sc[:, 1:].max(1).values).float()
        out[f"ten {form}{tag}"] = float(right.mean())
        out[f"ten {form} all{tag}"] = float((lp_.argmax(1) == cand[:, 0]).float().mean())
        grp = [k_.split()[-1] for k_, _, _ in sub]
        for g in sorted(set(grp)):
            out[f"ten {form} {g}{tag}"] = float(right[torch.tensor([x == g for x in grp])].mean())
        if form == "box":
            out[f"ten box new{tag}"] = float(right[torch.tensor([x in ("held", "far") for x in grp])].mean())
    sub = [(k_, s_, a_) for k_, s_, a_ in items if k_.startswith("ten pick ")]
    if sub:
        lp_ = scores([(st_, a_) for _, s_, a_ in sub for st_ in s_])
        word = torch.tensor([a_[0] for _, s_, a_ in sub for _ in s_])
        sc = lp_.gather(1, word.unsqueeze(1)).squeeze(1).view(len(sub), -1)
        right = (sc[:, 0] > sc[:, 1:].max(1).values).float()
        out[f"ten pick{tag}"] = float(right.mean())
        grp = [k_.split()[-1] for k_, _, _ in sub]
        for g in sorted(set(grp)):
            out[f"ten pick {g}{tag}"] = float(right[torch.tensor([x == g for x in grp])].mean())
    sub = [(k_, s_, a_) for k_, s_, a_ in items if k_.startswith("voe ten ")]
    if sub:
        lp_ = scores([(s_, a_) for _, s_, a_ in sub])
        s_good = -lp_.gather(1, torch.tensor([a_[0] for _, _, a_ in sub]).unsqueeze(1)).squeeze(1)
        s_bad = -lp_.gather(1, torch.tensor([a_[1] for _, _, a_ in sub]).unsqueeze(1)).squeeze(1)
        longer = (s_bad > s_good).float() + 0.5 * (s_bad == s_good).float()
        out[f"voe ten{tag}"] = float(longer.mean())
        grp = [k_.split()[-1] for k_, _, _ in sub]
        for g in sorted(set(grp)):
            out[f"voe ten {g}{tag}"] = float(longer[torch.tensor([x == g for x in grp])].mean())
        out[f"voe ten new{tag}"] = float(longer[torch.tensor([x in ("held", "far") for x in grp])].mean())
    return out


def bi_keys(model, items, w, tag):
    """W6 (criterion 5, two languages): "bi story ..." / "bi where X>Y": the end cell -- right when it scores strictly highest
    among the 25 "to" cells (plain; the exam's cell slot), "all" = among every element; "bi what X>Y": m's name in Y --
    right when it scores strictly highest among the room's four names in Y (the item's candidates, the right one first).
    Pooled: "bi cross where" / "bi cross what" / "bi cross" (X != Y), "bi same ..." (X = Y), "bi story ko all" = ko + ko d."""
    import lab2 as L_
    out = {}
    if not items:
        return out
    to_cells = torch.tensor(sorted(int(t) for t, r in w.role_of.items() if r == "to"))
    cx_, _ = L_.items_tensor([(s_, a_[0]) for _, s_, a_ in items])
    with torch.no_grad():
        lp_ = F.log_softmax(model.complete(cx_), dim=1)
    right, right_all = [], []
    for r_, (k_, _, a_) in enumerate(items):
        if k_.startswith("bi what "):
            sc = lp_[r_, torch.tensor(a_)]
            right.append(float(sc[0] > sc[1:].max()))
        else:
            sc = lp_[r_, to_cells]
            a_i = int((to_cells == int(a_[0])).nonzero()[0, 0])
            right.append(float(sc[a_i] > torch.cat([sc[:a_i], sc[a_i + 1:]]).max()))
        right_all.append(float(int(lp_[r_].argmax()) == int(a_[0])))
    groups = {}
    for k_, r_, ra_ in zip([k_ for k_, _, _ in items], right, right_all):
        keys = [k_]
        if k_ in ("bi story ko", "bi story ko d"):
            keys.append("bi story ko all")
        if k_.startswith(("bi where ", "bi what ")):
            x_, y_ = k_.split()[-1].split(">")
            form = k_.split()[1]
            side = "cross" if x_ != y_ else "same"
            keys += [f"bi {side} {form}", f"bi {side}"]
        for key in keys:
            groups.setdefault(key, []).append(r_)
        groups.setdefault(k_ + " all", []).append(ra_)
    for key, v in groups.items():
        out[f"{key}{tag}"] = float(np.mean(v))
    return out


# D11 (the user, 10-16: "재시작을 이어도는 1번으로 해줘" -- the container restarts often and every restart threw away the
# running lives): a life resumes from its last finished session. After each session the life's whole local state (the
# learner, the world, the random streams, the method layer's records ...) and the global random states are written at once
# (one pickle keeps shared objects shared), atomically; a restarted life rebuilds what cannot be written (functions, made
# again by the same start code) and then puts the written state back into its own frame and goes on with the next session.
# Never part of the result: a resumed life must equal the uninterrupted one exactly (check_resume.py). CPython 3.11 / 3.12
# only (frame locals written back); otherwise, or with RIE_NO_RESUME set, nothing is written and a life starts over.
RESUME = CACHE + "_resume"
RESUME_VERSION = 1


def _resume_ok():
    return not os.environ.get("RIE_NO_RESUME") and sys.version_info[:2] in ((3, 11), (3, 12))


def resume_path(cond, seed):
    return os.path.join(RESUME, f"seed{seed}_{cond}.pkl")


def resume_save(cond, seed, i, n, loc):
    """D11: the state after session i of n, written atomically (a record only)."""
    if not _resume_ok() or i + 1 >= n:           # (after the last session the life is cached at once: nothing to save)
        return
    import random
    import types
    skip_t = (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.ModuleType, types.GeneratorType)
    vars_ = {k: v for k, v in loc.items() if not k.startswith("_") and k not in ("cond", "seed")
             and not isinstance(v, skip_t)}
    state = {"version": RESUME_VERSION, "i": i, "n": n, "vars": vars_, "skipped": [],
             "torch": torch.get_rng_state(), "np": np.random.get_state(), "py": random.getstate(),
             "cum": list(LIFE_CPU["cum"]), "ref": LIFE_CPU["ref"]}
    os.makedirs(RESUME, exist_ok=True)
    path = resume_path(cond, seed)
    try:                                            # streamed into the file (the same bytes as dumps; no copy held)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        bad = []
        for k, v in vars_.items():
            try:
                pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                bad.append(k)
        for k in bad:
            vars_.pop(k)
        state["skipped"] = bad
        print(f"D11 resume: not written (made again on resume): {bad}", file=sys.stderr, flush=True)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    stop = os.environ.get("RIE_RESUME_STOP")          # check_resume.py only: end the process right after session i
    if stop and i + 1 == int(stop):
        os._exit(17)


def resume_load(cond, seed):
    if not _resume_ok() or not os.path.exists(resume_path(cond, seed)):
        return None
    with open(resume_path(cond, seed), "rb") as f:
        state = pickle.load(f)
    return state if state.get("version") == RESUME_VERSION else None


def resume_restore(frame, state):
    """put the written state back into the life's own frame (CPython: frame locals written back to fast locals and cells)."""
    import ctypes
    import random
    frame.f_locals.update(state["vars"])
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["np"])
    random.setstate(state["py"])
    LIFE_CPU["cum"], LIFE_CPU["ref"] = list(state["cum"]), state["ref"]
    LIFE_CPU["resumed"] = True                      # its peak memory is a resumed life's (the gate keeps those apart)
    print(f"D11 resume: session {state['i'] + 2}/{state['n']} on", file=sys.stderr, flush=True)


def c8_keys(model, items, w, tag):
    """W7b (criterion 8 keys): "c8 before same / three" -- right when A's earlier cell scores strictly highest among the 25
    "at" cells (the exam's cell slot); "c8 long" -- A's end among the 25 "to" cells; " all" = among every element. Pooled:
    "c8 before" (both kinds)."""
    import lab2 as L_
    out = {}
    if not items:
        return out
    at_cells = torch.tensor(sorted(int(t) for t, r in w.role_of.items() if r == "at"))
    to_cells = torch.tensor(sorted(int(t) for t, r in w.role_of.items() if r == "to"))
    cx_, _ = L_.items_tensor([(s_, a_[0]) for _, s_, a_ in items])
    with torch.no_grad():
        lp_ = F.log_softmax(model.complete(cx_), dim=1)
    groups = {}
    for r_, (k_, _, a_) in enumerate(items):
        cells = at_cells if k_.startswith("c8 before") else to_cells
        sc = lp_[r_, cells]
        a_i = int((cells == int(a_[0])).nonzero()[0, 0])
        ok = float(sc[a_i] > torch.cat([sc[:a_i], sc[a_i + 1:]]).max())
        ok_all = float(int(lp_[r_].argmax()) == int(a_[0]))
        for key in ([k_, "c8 before"] if k_.startswith("c8 before") else [k_]):
            groups.setdefault(key, []).append(ok)
            groups.setdefault(key + " all", []).append(ok_all)
    for key, v in groups.items():
        out[f"{key}{tag}"] = float(np.mean(v))
    return out


def life(cond, seed):
    global _cpu0
    LIFE_CPU["ref"], LIFE_CPU["cum"], LIFE_CPU["resumed"] = ref_job(), [], False
    _cpu0 = time.process_time()
    # B stage (finding 90): "@ascii" in the condition = the glyph-picture world with its parts on -- the where channel's own
    # linear memory, kind-first answers, templates with a cue (all learned from the record, no constants)
    ascii_world = "@ascii" in cond
    w = build_world(("ascii-seq2" if "@seq2" in cond else "ascii-seq" if "@seq" in cond else "ascii-sceneshuf" if "@sceneshuf" in cond else
                     "ascii-scene" if "@scene" in cond else
                     "ascii-d72cat" if "@d72cat" in cond else "ascii-d72verb" if "@d72verb" in cond else
                     "ascii-d72chain" if "@d72chain" in cond else "ascii-d72neg" if "@d72neg" in cond else
                     "ascii-d72ice" if "@d72ice" in cond else "ascii-d72next" if "@d72next" in cond else
                     "ascii-hold" if "@hold" in cond else "ascii-nwf" if "@nwf" in cond else "ascii-nw" if "@nw" in cond else "ascii-kor" if "@kor" in cond else
                     "ascii-spshuf" if "@spshuf" in cond else "ascii-sp" if "@sp" in cond else
                     "ascii-whenshuf" if "@whenshuf" in cond else "ascii-when" if "@when" in cond else
                     "ascii-tenshuf" if "@tenshuf" in cond else "ascii-ten" if "@ten" in cond else
                     "ascii-bishuf" if "@bishuf" in cond else "ascii-bi" if "@bi" in cond else
                     "ascii-c8shuf" if "@c8shuf" in cond else "ascii-c8" if "@c8" in cond else
                     "ascii-multi" if "@multi" in cond else "ascii-objs" if "@objs" in cond else
                     "ascii-grav" if "@grav" in cond else "ascii-tale" if "@tale" in cond else "ascii-frame" if "@frame" in cond else
                     "ascii-contain" if "@contain" in cond else "ascii-count" if "@count" in cond else
                     "ascii-succ" if "@succ" in cond else "ascii-numrow" if "@numrow" in cond else
                     "ascii-numown" if "@numown" in cond else "ascii-views" if "@views" in cond else
                     "ascii-viewpt" if "@viewpt" in cond else "ascii-goal" if "@goal" in cond else "ascii-self" if "@self" in cond else
                     "ascii") if ascii_world else "life32blind" if "@blind" in cond else
                    "life32share" if "@share" in cond else WORLD, seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    mrng = np.random.default_rng(60_000 + seed)
    if cond.startswith("tf"):                       # the transformer control (finding 73) in the long life
        from integrated_tf import TLearner
        me = TLearner(w, seed)
    else:
        # finding 101 ("+attend"): the habit's context mean becomes attention starting as the mean (function-preserving)
        me = Learner3(w, seed, content=True, where="linear" if ascii_world else None, attend="+attend" in cond,
                      rehearse="+rehearse" in cond)   # readiness step 2 (2026-10-17): Module A rehearses until recalled
    me.model.known_only = "+knownonly" in cond       # D26 (1) S1: only words heard so far are answer candidates
    me.bg_pool = "+bgpool" in cond                  # finding 118: the world-shaped eye's background from partnerless words
    d25_mode = next((m for m in "NUFKME" if f"+d25{m}" in cond), None)   # D25 (2026-10-13): parts -> body, edition 1
    if d25_mode:
        import d25
        me.d25 = d25.Store(w, seed, d25_mode)
    # world-model stage A (2026-10-06): "+wm" adds the rolled-out world-model part to the answer; "+chain" (and "+wm")
    # records the accuracy on the rooms' chain questions (two motion words never heard together; record only)
    chain_probe = ascii_world and ("+chain" in cond or "+wm" in cond)
    if "+scene" in cond:                            # D75 step 1: the scene part (before / after scenes, no role markers)
        import scene_part
        me.model.scene = scene_part.SceneBooks(w, me)
    if ascii_world and "+wm" in cond:
        import worldmodel
        me.model.wm = worldmodel.WorldBooks(me)
        me.model.wm.context = "+wmctx" in cond      # stage B: context books from the room's surprises
        me.model.wm.bind = "+bind" in cond          # finding 123: particles bound to their neighbours (Korean stories)
        me.model.wm.bind_random = "+bindrand" in cond
        me.model.wm.last = "+wmlast" in cond          # finding 125: the decided world model speaks last
        me.model.wm.track = "+track" in cond and "+trackone" not in cond   # stage-1 step 2: one state per object
        me.model.wm.track_one = "+trackone" in cond                          # control: one state for the whole story
        me.model.wm.reverse = "+wmrev" in cond                               # stage-1 step 3a: scene -> motion word
        me.model.wm.props = "+props" in cond                                 # stage-1 step 4: object property table
        me.model.wm.props_rand = "+propsrand" in cond                        # control: another object's entry
        me.model.wm.tale_mode = ("one" if "+tk1" in cond else "anchor" if "+tkA" in cond else "ignore" if "+tkig" in cond
                                 else None)                         # D22: frames vs reality (tale books)
        me.model.wm.tale_mark = getattr(w, "tale_mark", None)
        me.model.wm.grav = "+grav" in cond and "+gravall" not in cond        # stage-1 step 4+: words that repeat until resting
        me.model.wm.grav_all = "+gravall" in cond                            # control: every word repeats
        me.model.wm.frame = "+wmframe" in cond                               # stage-1 step 4+: the room's frame (8 symmetries)
        me.model.wm.contain = "+contain" in cond and "+containall" not in cond   # step 4+: relations that carry
        me.model.wm.contain_all = "+containall" in cond                            # control: every relation carries
        me.model.wm.count = "+count" in cond and "+countrand" not in cond        # step 4+: small numbers
        me.model.wm.count_rand = "+countrand" in cond                              # control: number words rotated
        me.model.wm.succ = "+succ" in cond                                         # step 4+: the successor rule
        me.model.wm.succ_rev = "+succrev" in cond                                  # control: the chant read backwards
        me.model.wm.scale = "+scale" in cond                                       # step 4+ (D15): scale alignment
        me.model.wm.scale_rev = "+scalerev" in cond                                # control: the inverse ratio
        me.model.wm.views = "+views" in cond                                       # step 4+ (D17): three-view axis matching
        me.model.wm.views_rand = "+viewsrand" in cond                              # control: the other axis of the same view
        me.model.wm.vturn = "+vturn" in cond                                       # D2 viewpoint: verb maps (turns)
        me.model.wm.goal = "+goal" in cond                                         # D2 goal: the teleological stance
        me.model.wm.goal_rand = "+goalrand" in cond                                # control: the other thing as the goal
        me.model.wm.self_part = "+self" in cond                                    # D2 self: contingency (8 symmetries)
        me.model.wm.self_id = "+selfid" in cond                                    # control: the identity only (no mirror)
        me.model.wm.test_only = "+wmtest" in cond                                  # the leaning test: no answer in practice
        me.model.wm.measure_only = "+wmmeasure" in cond                            # the leaning test 2: silent in sessions
        me.model.wm.gate = "+gate" in cond                                         # D69: each part earns its say (self-trust)
        me.model.wm.useful = "+useful" in cond                                     # D83 (1): a part speaks where it helped
        me.model.wm.use_mode = ("quiet" if "+usequiet" in cond else "shuf" if "+useshuf" in cond else None)  # references
        me.keep_order, me.order_log = True, []
    method = cond.startswith("both") or cond == "tf+method"
    wake, aware = "wake" in cond, "aware" in cond
    clean = "clean" in cond or "sleep" in cond
    # clean part judgement (finding 75's follow-up, 2026-09-28): waking or putting to sleep a PART is not decided by the knob
    # judge (whose old-category questions have the same shape as the self-made practice) but by a clean check -- each part
    # branch against the stay branch, both practised the full session, on the new categories' held-out sentences only
    # (never practised, never recombined; leave-one-out whole-sentence memory); a part changes only if strictly better
    # part_mode: "level" (the clean score at the end), "slope" (the user's proposal, 2026-09-28: which branch is still
    # climbing -- clean score gain over the second half of the session's practice), "trend" (the part's advantage over sessions)
    part_mode = "slope" if "slope" in cond else "trend" if ("trend" in cond or "fit" in cond) else "level"
    fit = "fit" in cond
    clean = clean or part_mode != "level"
    ledger = {}
    parts = ({"setter": [None, "random"]} if any(x in cond for x in ("recomb-sleep", "recomb-slope", "recomb-trend", "recomb-fit")) else
             {k: [ASLEEP[k]] + v for k, v in WAKE.items()} if clean else {})
    budget = "low" if "-low" in cond else "high" if "-high" in cond else "mid"   # ("-high" was missing: finding 78's bug)
    # self-budget (the method ledger's first use, 2026-09-28, the user's proposal): the method layer chooses how hard to think
    # each session. It keeps a ledger of what its last search was worth -- did the chosen branch beat the stay branch on more
    # judged categories than it lost to it (finding 61's stakes rule)? -- and moves one budget level up after a search that
    # found a clear win, one level down after one that did not (levels low / mid / high = findings 72's rungs; starts low).
    # No exchange-rate constant between accuracy and compute. Uses the budget-aware caution (finding 75) for its decisions.
    # review fix (2026-10-12): the bare "self" also matched the world name "@self" (finding 140 ran with the self-budget on);
    # "-self" is the one condition that meant it ("both-self", finding 77) -- every other condition is unchanged
    self_budget = "-self" in cond or "surprise" in cond
    aware = aware or self_budget
    level = 0
    # surprise (self-budget v2, finding 78): before thinking, how well does the learner already answer the new material (the
    # new categories' held-out sentences, with its knowledge set up but before practice)? Lower than it has come to expect (the mean of its past sessions) = something changed
    # -> one level up; otherwise one level down. The expectation is its own record, no constant (as finding 41's surprise).
    seen_before = []
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    aside = {}
    for c in range(w.n_cat):
        idx = mrng.permutation(len(sents[c]))
        aside[c] = set(idx[:int(round(len(sents[c]) * RI.HOLD))].tolist())
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    early_rooms = {c for s_ in sessions[:len(sessions) // 2] for c in s_}     # W2 ("@nw"): rooms met in the first half
    settings = dict(START, **({"setter": "random"} if "recomb" in cond else {}))   # recomb-sleep too: awake from birth
    if ascii_world:
        settings.update(kind_first=True, cue_templates=True)
    if d25_mode in ("K", "M"):
        settings["d25_k"] = 2
    if d25_mode == "M":
        settings["d25_mark"] = "none"
        if "+fix" in cond:                          # finding 92: the four template repairs
            settings.update(cue_kinds=True, cue_specific=True, cue_members_vote=True, cue_pure=True)
    if "+world" in cond:                            # finding 82: connections and templates from first-hand evidence only
        settings["hebb_source"] = "world"
    if "-side" in cond:                             # self-made practice through its own side part (source axis in learning)
        settings["self_side"] = True
    if "-routeknobs" in cond:                       # finding 95: the method layer gets the routes' weights
        settings.update(pattern_w=1.0, chunk_w=1.0)
    if "-sideknobs" in cond:                        # finding 86 (a): the method layer gets the side part's knobs
        settings.update(side_w=1.0, side_decay=1.0)
    if "+plastic3" in cond:                         # finding 106's follow-up: slow only the new material's share
        settings["plastic"] = "pairnew"
    elif "+plastic2" in cond:                       # finding 105's follow-up: per connection
        settings["plastic"] = "pair"
    elif "+plastic" in cond:                        # Pioneer's idea as part of the body (fixed on)
        settings["plastic"] = True
    if "-plasticknob" in cond:                      # ... and as a method-layer knob, off at birth (the new principle)
        settings["plastic"] = False
    if "-forgetknob" in cond:                       # finding 108: the forgetting-targeted review as a knob, off at birth
        settings["forget_pick"] = False
    if "+rediscover" in cond:                       # the rediscovery test: the hand-found improvements as knobs, off at birth
        settings["rediscover"] = True
    if "+disagree" in cond:                         # tier 2 (finding 83): practise once more where the routes disagree
        settings["disagree"] = True
    # two kinds of trust (the user's proposal, 2026-09-30, after finding 87): the judge's two sources are kept apart --
    # first-hand feedback (the new categories' held-out sentences, never practised) and imagination (the old categories'
    # self-made recombinations, the same generator the self-made practice uses). First-hand is the top judge (weight 1);
    # imagination counts only as much as it has EARNED: its trust is the learner's own record of how well imagination ranked
    # the branches the same way first-hand feedback did (Kendall's tau over the branches after the first rung, mean over past
    # sessions, floored at 0 -- chance agreement earns nothing, and it can never outrank first-hand feedback). No constant.
    trust2 = "+trust2" in cond
    # the long eye (finding 93's follow-up, 2026-09-30, the user's principle of trying choices in parallel spaces, stretched
    # over time): when the method layer changes a setting, a SHADOW learner -- the state before the change, keeping the old
    # settings -- lives on beside it and meets the same experience. Every later session the shadow enters the branch race
    # as one more branch, judged on the same fresh material; if it wins (with the budget-aware caution, when on), the learner
    # goes back to it. A change is judged by everything it led to, not by the one session after it. At most one shadow (a new
    # change replaces it). No constant; about twice the compute while a shadow lives.
    longeye = "+longeye" in cond
    # finding 96's follow-up (two eyes): the judge's old-category score is the recombined score (imagination -- in the glyph
    # world a question of another shape, blind to the late collapse) AND the old heard sentences' score (holding on --
    # finding 96: follows the outside test within a life, +0.54), averaged with equal weight (no constant chosen)
    twoeyes = "+twoeyes" in cond
    # finding 101's follow-up ("+worldeye"): the judge's old-category eye is the world-shaped imagination (world_score)
    # instead of the recombined one -- with attention in the habit the world eye follows the outside test +0.33 and the
    # recombined eye +0.06 (attend_probe_10.out); the same rule in every world (no world-specific choice)
    worldeye = "+worldeye" in cond
    # the mode layer, first step (2026-10-01, the user's emotions-as-modes idea; finding 97: the holding-on eye mixed into the
    # judge ALL the time made the text world choose slow learning, -0.12): a mode is chosen from the dashboard and re-weights
    # what the judge values, with persistence. "protect" (a loss mode -- like noticing one is forgetting and turning to
    # review) is entered when the old heard indicator fell THREE sessions in a row (a trend, not noise -- the user's choice:
    # with two, pure noise enters protect in 91% of 15-session lives and the text world, whose indicator rises, in 25 of 30;
    # with three, 36% and 12 of 30), left when it rose three sessions in a row; otherwise the mode stays. In protect the judge sees old categories with two eyes
    # (imagination + holding on); in explore (the default) with imagination only, as before. Decided from past sessions only
    use_modes = "+mode" in cond
    # the eye probe (2026-10-01, finding 98's (a); recorded only): after the first rung every branch is scored by each eye on
    # the old categories -- imagination (recombined), holding on (old heard), the world-shaped imagination (world_score) --
    # and by the outside test on the same categories (which the learner never sees); the rank correlation over branches says
    # whether an eye points where the outside test points (finding 95: the judge picked the wrong direction in the glyph world)
    probe = "+probe" in cond and "+probe2" not in cond
    # the late probe (finding 99's follow-up, 2026-10-01; recorded only): every branch practised to the LAST rung, then raw
    # per-branch scores are kept -- each eye twice (another draw / the other half: the eye's own reliability), the outside
    # test on two halves of its items (how reliable the ranking target itself is), the fresh eye and the outside test on the
    # new categories. Asks: is the judge blind because it looks too early (eyes sharpen with practice), because the eyes are
    # noisy (small held-out sets), or because the knob choices barely change the outcome (the body gives no lever)?
    probe2 = "+probe2" in cond
    probe_rec = {}
    selfsim = "+whatif" in cond                     # finding 113 (also on with "+whatif2")
    worldsim = "+whatif2" in cond                   # finding 114
    tune, tuned = "+tune" in cond, None             # finding 115
    tune_answer_only = "+tunex" in cond             # finding 117 (includes "+tune")
    # finding 119: grade the eye's last choice on fresh sentences and answer with weight 1 for a part whose trust is below
    # one half (the eye keeps choosing and being graded, so trust can come back)
    graded, grades, last_pick, trust_now = "+graded" in cond, {"chunk": [], "assoc": []}, None, {}
    mode, held = "explore", []
    # the mode layer, second step ("+bundle", 2026-10-02, the user's emotions-as-modes idea as first stated: a mode sets a
    # BUNDLE of knobs coherently and persists; finding 98: switching only the judge's eye was too little). Three modes from the
    # dashboard, each entered and left by the same three-sessions-in-a-row rule (the user's choice; finding 98's analysis):
    #   protect (loss)      -- the old heard indicator fell 3 sessions in a row; left when it rose 3 in a row. Bundle: review one
    #                          step up on entry, no move that learns slower or reviews less (finding 97: the holding-on eye
    #                          drove lr to 0.004), the judge looks with two eyes.
    #   settle (calm)       -- the options indicator (best branch - stay) fell 3 in a row: there is less and less to gain by
    #                          changing; left when it rose 3 in a row. Bundle: change only on a clear win (the budget-aware
    #                          caution: the challenger must beat the current settings on more categories than it loses) --
    #                          finding 94's "changes almost every session" restlessness.
    #   explore (default)   -- everything as before.
    # protect outranks settle. Decided from past sessions only. No constant.
    bundle = "+bundle" in cond
    # the method layer makes new options itself (finding 109, the user's choice (A), 2026-10-03; finding 108: the gap stops
    # growing once the fixed menu of options is used up). Two first steps, both from the method layer's own record:
    #   range extension -- a numeric knob (review, lr) that a winning move pushed to the END of its ladder gets one new rung
    #                      beyond it, spaced by the ladder's own last ratio (review capped at 1: it is a fraction); a new
    #                      rung that wins can be extended again;
    #   composite moves -- when no single move beat the current settings in a session, the next session also tries the two
    #                      single moves the judge ranked best, together (if they touch different knobs).
    # No constant: the spacing is the ladder's, the pair is the record's.
    openended = "+open" in cond
    ladders = {k: list(RI.KNOBS[k]) for k in ("review", "lr")}
    pushed, last_pair, n_ext, n_pair_won = {}, None, 0, 0
    opts = []
    shadow, shadow_settings, swaps = None, None, 0
    tau_hist, tau = [], 0.0
    seen, steps = [], []
    acc_now, expect, rpe = {}, {}, {}
    # D35 (d), the first curiosity experiment (2026-10-12): "+lpcur" feeds the curious review pick the learning progress
    # itself (|accuracy change| over the last session, Oudeyer & Kaplan 2007) instead of the surprise in it (progress minus
    # its running expectation, finding 41). Without "+lpcur" the pick reads rpe exactly as before (same dict object).
    prog = {}
    curw = prog if "+lpcur" in cond else rpe
    _sd = os.environ.get("VOY_STATEDIFF")              # hidden-state audit (statediff_voy.py), record only
    if _sd:
        import statediff_voy as SD
        _sd_x = lambda: {"rpe": rpe, "expect": expect, "acc_now": acc_now, "settings": settings, "rng": rng, "mrng": mrng}
    _start, _rs = 0, resume_load(cond, seed)      # D11: a restarted life goes on after its last finished session
    if _rs is not None:
        resume_restore(sys._getframe(), _rs)
        _start = _rs["i"] + 1
        _cpu0 = time.process_time() - (_rs["cum"][-1] if _rs["cum"] else 0.0)
    # D80 v1: the episode store; "+epi" (D72 subject 4, 2026-10-22): the same store in any world, its EQ1 answers kept beside
    epi = EpisodeStore(w.channel_of) if "@seq" in cond or "+epi" in cond else None
    # D80 v2: with the scene part, a second store whose where lines also hold the cells told in words (expected scenes)
    epi2 = EpisodeStore(w.channel_of, books=me.model.scene) if epi is not None and \
        getattr(me.model, "scene", None) is not None else None
    for i, new_cats in enumerate(sessions):
        if i < _start:
            continue
        if _sd:
            SD.mark(i, "top", me, _sd_x())
        if getattr(getattr(me.model, "wm", None), "measure_only", False):
            me.model.wm.silent = True               # "+wmmeasure": the part keeps silent through the session
        if "recomb-alt" in cond:
            # diagnostic (finding 79): the trend signal's partial gain may come from its flicker = practice every other
            # session (spacing), not from detecting anything -- practice on even sessions, off on odd, told from outside
            settings = dict(settings, setter="random" if i % 2 == 0 else None)
        if "until" in cond and i == int(cond.split("until")[1]):
            # diagnostic (finding 77): recombination practice put to sleep at a FIXED session, told from outside -- would a
            # timely sleep help at all, if the learner knew when?
            settings = dict(settings, setter=None)
        old = list(seen)
        seen += new_cats
        if getattr(me.model, "scene", None) is not None:   # D75: the scene part reads the new rooms' moments in time order
            for c in new_cats:
                me.model.scene.observe([s_ for _, s_, _ in w.categories[c]["heard"]])
        cost, n_br, changed = np.nan, np.nan, np.nan
        if use_modes and len(held) >= 4:
            d = np.diff(held[-4:])
            if (d < 0).all():
                mode = "protect"
            elif (d > 0).all():
                mode = "explore"
        if bundle:
            def trend(h):
                if len(h) < 4:
                    return 0
                d = np.diff(h[-4:])
                return -1 if (d < 0).all() else 1 if (d > 0).all() else 0
            prev_mode = mode
            th, to = trend(held), trend(opts)
            if mode == "protect":
                mode = "explore" if th > 0 else "protect"
            elif mode == "settle":
                mode = "protect" if th < 0 else "explore" if to > 0 else "settle"
            if mode == "explore":
                mode = "protect" if th < 0 else "settle" if to < 0 else "explore"
            if mode == "protect" and prev_mode != "protect" and RI.apply(settings, "review+") is not None:
                settings = RI.apply(settings, "review+")
        two_now = twoeyes or (use_modes and mode == "protect") or (bundle and mode == "protect")
        aware_now = aware or (bundle and mode == "settle")
        # the prediction dashboard, first step (2026-09-30, the user's emotions-as-modes idea): indicators only RECORDED, the
        # learner's behaviour is unchanged. fresh before = the stay branch on the new categories' held-out sentences, set up
        # but not practised (how predictable the new material already is -- freshness); fresh after = the same after the
        # session's trial practice (what this session's learning reached); old = the stay branch's old-category score
        # (holding on -- loss); options = the best branch's judge minus the stay branch's after the first rung (is there
        # anything better to do)
        # finding 95's follow-up (the dashboard's eye): "dash old" is the recombined score, which in the glyph world asks a
        # different question from the world's own (flat 0.12 while the outside test fell 0.87 -> 0.76). old heard = the stay
        # branch on the old categories' held-out HEARD sentences (leave-one-out whole-sentence memory) -- holding on to what
        # was heard, where proactive interference shows. Recorded only ("+eye" names a fresh cache; behaviour unchanged)
        dash = {"dash fresh before": np.nan, "dash fresh after": np.nan, "dash old": np.nan, "dash options": np.nan,
                "dash old heard": np.nan}
        if method and i > 0:
            cost = 0
            trial_new = [s for c in new_cats for j, s in enumerate(sents[c]) if j not in aside[c]]
            judged = {c: [s for j, s in enumerate(sents[c]) if j in aside[c]] for c in new_cats}
            trial_old = {}
            for c in old:
                idx = mrng.permutation(len(sents[c]))
                k = int(round(len(sents[c]) * RI.HOLD))
                judged[c] = [sents[c][j] for j in idx[:k]]
                trial_old[c] = [sents[c][j] for j in idx[k:]]
            branches = []
            prev_settings = dict(settings)
            extra_moves = []
            if openended:
                for k, lad in ladders.items():
                    v = settings[k]
                    if v not in RI.KNOBS[k]:        # on a self-made rung: its neighbours on the ladder
                        j = lad.index(v)
                        for jj in (j - 1, j + 1):
                            if 0 <= jj < len(lad):
                                extra_moves.append((f"{k}={lad[jj]:.4g}", dict(settings, **{k: lad[jj]})))
                    if pushed.get(k) == "+" and v == lad[-1] and len(lad) > 1:
                        nv = lad[-1] * (lad[-1] / lad[-2])
                        nv = min(nv, 1.0) if k == "review" else nv
                        if nv > lad[-1]:
                            extra_moves.append((f"{k}={nv:.4g}", dict(settings, **{k: nv})))
                    if pushed.get(k) == "-" and v == lad[0] and len(lad) > 1:
                        nv = lad[0] * (lad[0] / lad[1])
                        extra_moves.append((f"{k}={nv:.4g}", dict(settings, **{k: nv})))
                if last_pair is not None:
                    (m1, s1), (m2, s2) = last_pair
                    k1 = {k for k in s1 if s1[k] != settings.get(k)}
                    k2 = {k for k in s2 if s2[k] != settings.get(k)}
                    if k1 and k2 and not (k1 & k2):
                        extra_moves.append((f"{m1}&{m2}", dict(settings, **{k: s1[k] for k in k1}, **{k: s2[k] for k in k2})))
            for m, s in [(None, settings)] + [(m_, s_) for m_, s_ in moves(settings, wake and not clean)
                                              if not (bundle and mode == "protect" and m_ in ("review-", "lr-"))] + extra_moves:
                b = me.clone()
                b.prepare(trial_new, trial_old, s, np.random.default_rng(mrng.integers(1 << 31)), curw)
                branches.append((m, s, b))
            if longeye and shadow is not None:
                b = shadow.clone()
                b.prepare(trial_new, trial_old, shadow_settings, np.random.default_rng(mrng.integers(1 << 31)), curw)
                branches.append(("shadow", shadow_settings, b))
            n_br = len(branches)
            jseed = int(mrng.integers(1 << 31))

            def per_cat(b):
                return {c: (b.judge_score(ss) if c in new_cats else
                            (b.recombined_score(ss, np.random.default_rng(jseed + c)) + b.judge_score(ss)) / 2 if two_now else
                            b.world_score(ss, np.random.default_rng(jseed + c)) if worldeye else
                            b.recombined_score(ss, np.random.default_rng(jseed + c))) for c, ss in judged.items()}

            def judge(b):
                pc = per_cat(b)
                if not trust2:
                    return float(np.mean(list(pc.values())))
                e = float(np.nanmean([pc[c] for c in new_cats]))
                o = [pc[c] for c in old]
                if not o or tau <= 0:
                    return e
                return (e + tau * float(np.nanmean(o))) / (1 + tau)

            def weight(c):
                return 1.0 if (not trust2 or c in new_cats) else tau
            stay = branches[0]
            alive = branches
            dash["dash fresh before"] = float(np.nanmean([stay[2].judge_score(judged[c]) for c in new_cats]))
            if "surprise" in cond:
                # the stay branch, prepared (connections, templates, whole-sentence memory set up) but not yet practised,
                # on the new categories' held-out sentences: what the learner's knowledge already gives it
                a_now = float(np.mean([branches[0][2].judge_score(judged[c]) for c in new_cats]))
                if seen_before:
                    level = min(level + 1, len(LEVELS) - 1) if a_now < np.mean(seen_before) else max(level - 1, 0)
                seen_before.append(a_now)
            if self_budget:
                budget = LEVELS[level]
            for rung in RUNGS[budget]:
                for _, _, b in alive:
                    before = b.st["done"]
                    b.train_to(rung)
                    cost += b.st["done"] - before
                if selfsim and rung == RUNGS[budget][0]:     # finding 113: record only
                    probe_rec.update(selfsim_self(branches[0][2], judged))
                if tune and rung == RUNGS[budget][0]:        # finding 115: choose the answering weights by self-experiment
                    if graded and last_pick is not None:     # finding 119: first grade last session's choice
                        for part, g in grade_pick(branches[0][2], judged, last_pick).items():
                            grades[part].append(g)
                            probe_rec[f"grade {part}"] = g
                    tuned = tune_weights(branches[0][2], judged, np.random.default_rng(2_000_000 + 100 * seed + i))
                    if graded and tuned is not None:
                        last_pick = tuned
                        trust_now = {p: trust_of(grades[p]) for p in ("chunk", "assoc")}
                        probe_rec.update({f"eye {p}": w for p, w in zip(("chunk", "assoc"), tuned)})
                        probe_rec.update({f"trust {p}": v for p, v in trust_now.items()})
                        tuned = tuple(w if trust_now[p] >= 0.5 else 1.0 for p, w in zip(("chunk", "assoc"), tuned))
                if worldsim and rung == RUNGS[budget][0]:    # finding 114: the world-shaped self-test (own generator)
                    wr = np.random.default_rng(1_000_000 + 100 * seed + i)
                    wit = [it for ss in judged.values() for it in branches[0][2].world_items(ss, wr)]
                    probe_rec.update({f"cf world {k}": v for k, v in counterfactual(branches[0][2], wit).items()})
                if rung == RUNGS[budget][0] and len(alive) > 1:
                    js = [judge(b) for _, _, b in alive]
                    dash["dash options"] = float(max(js) - js[0])
                    if openended:
                        ranked = sorted([(js[j], alive[j][0], alive[j][1]) for j in range(1, len(alive))
                                         if alive[j][0] is not None and "&" not in str(alive[j][0])],
                                        key=lambda x: -x[0])
                        top2 = [(m_, s_) for _, m_, s_ in ranked[:2]]
                    if probe and old:
                        def rank(v):
                            return np.argsort(np.argsort(v)).astype(float)

                        def rcorr(a_, b_):
                            a_, b_ = rank(np.asarray(a_)), rank(np.asarray(b_))
                            if a_.std() == 0 or b_.std() == 0:
                                return np.nan
                            return float(np.corrcoef(a_, b_)[0, 1])
                        true = [acc_items(b.model, [(cx, t) for c in old for kind, cx, t in tests[c]]) for _, _, b in alive]
                        eyes = {
                            "recomb": [np.nanmean([b.recombined_score(judged[c], np.random.default_rng(jseed + c)) for c in old])
                                       for _, _, b in alive],
                            "heard": [np.nanmean([b.judge_score(judged[c]) for c in old]) for _, _, b in alive],
                            "world": [np.nanmean([b.world_score(judged[c], np.random.default_rng(jseed + c)) for c in old])
                                      for _, _, b in alive]}
                        probe_rec = {f"probe {k}": rcorr(v, true) for k, v in eyes.items()}
                        probe_rec["probe spread"] = float(np.max(true) - np.min(true))
                if trust2 and rung == RUNGS[budget][0] and old and len(alive) > 1:
                    pcs = [per_cat(b) for _, _, b in alive]
                    ext = [float(np.nanmean([p[c] for c in new_cats])) for p in pcs]
                    imag = [float(np.nanmean([p[c] for c in old])) for p in pcs]
                    conc = [np.sign(ext[x] - ext[y]) * np.sign(imag[x] - imag[y])
                            for x in range(len(pcs)) for y in range(x + 1, len(pcs))]
                    tau_now = float(np.mean(conc))
                if rung == RUNGS[budget][-1]:
                    break
                alive = sorted(alive, key=lambda x: -judge(x[2]))[:math.ceil(len(alive) / 2)]
                if aware_now and stay not in alive:     # the current settings are never pruned early
                    alive.append(stay)
            if probe2 and old:
                rng_state = torch.get_rng_state()
                raw = {k: [] for k in ("true", "true A", "true B", "recomb", "recomb 2", "world", "world 2", "heard",
                                       "heard A", "heard B", "fresh", "true new")}
                t_old = [(cx, t) for c in old for kind, cx, t in tests[c]]
                t_new = [(cx, t) for c in new_cats for kind, cx, t in tests[c]]
                for _, _, b in branches:
                    b.train_to(RUNGS[budget][-1])
                    raw["true"].append(acc_items(b.model, t_old))
                    raw["true A"].append(acc_items(b.model, t_old[0::2]))
                    raw["true B"].append(acc_items(b.model, t_old[1::2]))
                    raw["recomb"].append(np.nanmean([b.recombined_score(judged[c], np.random.default_rng(jseed + c))
                                                     for c in old]))
                    raw["recomb 2"].append(np.nanmean([b.recombined_score(judged[c], np.random.default_rng(jseed + c + 7919))
                                                       for c in old]))
                    raw["world"].append(np.nanmean([b.world_score(judged[c], np.random.default_rng(jseed + c)) for c in old]))
                    raw["world 2"].append(np.nanmean([b.world_score(judged[c], np.random.default_rng(jseed + c + 7919))
                                                      for c in old]))
                    raw["heard"].append(np.nanmean([b.judge_score(judged[c]) for c in old]))
                    raw["heard A"].append(np.nanmean([b.judge_score(judged[c][0::2]) for c in old]))
                    raw["heard B"].append(np.nanmean([b.judge_score(judged[c][1::2]) for c in old]))
                    raw["fresh"].append(np.nanmean([b.judge_score(judged[c]) for c in new_cats]))
                    raw["true new"].append(acc_items(b.model, t_new))
                torch.set_rng_state(rng_state)
                probe_rec = {"probe2": raw}
            pcs = per_cat(stay[2])
            dash["dash fresh after"] = float(np.nanmean([pcs[c] for c in new_cats]))
            dash["dash old"] = float(np.nanmean([pcs[c] for c in old])) if old else np.nan
            if old:
                dash["dash old heard"] = float(np.nanmean([stay[2].judge_score(judged[c]) for c in old]))
            best = max(alive, key=lambda x: judge(x[2]))
            if aware_now and best is not stay:      # change only if the challenger wins on more categories than it loses
                a, b_ = per_cat(best[2]), per_cat(stay[2])
                up = sum(weight(c) * (a[c] > b_[c]) for c in a)
                down = sum(weight(c) * (a[c] < b_[c]) for c in a)
                if up <= down:
                    best = stay
            if self_budget and "surprise" not in cond:
                if best is not stay:
                    level = min(level + 1, len(LEVELS) - 1)
                else:
                    level = max(level - 1, 0)
            changed = float(best[0] is not None)
            if openended:
                # the record the next session's new options come from
                if best[0] is None:
                    last_pair = tuple(top2) if len(top2) == 2 else None
                else:
                    last_pair = None
                    if "&" in str(best[0]):
                        n_pair_won += 1
                for k, lad in ladders.items():
                    nv, ov = best[1][k], settings[k]
                    if nv != ov:
                        if nv not in lad:
                            lad.append(nv)
                            lad.sort()
                            n_ext += 1
                        pushed[k] = "+" if nv > ov else "-"
                    elif best[0] is not None:
                        pushed.pop(k, None)
            if longeye:
                if best[0] == "shadow":             # the shadow lineage did better: go back to it, keep the other as shadow
                    me, shadow = shadow, me
                    shadow_settings = prev_settings
                    swaps += 1
                elif best[0] is not None:           # a new change: the state before it becomes the shadow
                    shadow, shadow_settings = me.clone(), prev_settings
            new_settings = dict(best[1])
            if trust2 and old and len(branches) > 1:  # this session's agreement counts from the NEXT session on
                tau_hist.append(tau_now)
                tau = max(0.0, float(np.mean(tau_hist)))
            if parts:
                def clean_score(b):
                    return float(np.mean([b.judge_score(judged[c]) for c in new_cats]))
                top = RUNGS[budget][-1]
                mid = RUNGS[budget][-2] if len(RUNGS[budget]) > 1 else top // 2

                def practise(s_):
                    """a branch practised the full session with settings s_: (clean score at mid, at the end)."""
                    b = me.clone()
                    b.prepare(trial_new, trial_old, s_, np.random.default_rng(mrng.integers(1 << 31)), curw)
                    b.train_to(mid)
                    m_ = clean_score(b)
                    b.train_to(top)
                    return m_, clean_score(b)
                if part_mode == "slope":
                    s_mid, s_end = practise(settings)          # a fresh stay branch, scored half-way too
                    cost += top
                    base = s_end - s_mid
                else:
                    before = stay[2].st["done"]
                    stay[2].train_to(top)                      # the stay branch may have been pruned early: practise it fully
                    cost += stay[2].st["done"] - before
                    base = clean_score(stay[2])
                for k, vals in parts.items():
                    cur = settings.get(k, ASLEEP.get(k))
                    tried = []
                    for v in vals:
                        if v == cur:
                            continue
                        m_, e_ = practise(dict(settings, **{k: v}))
                        cost += top
                        tried.append((e_ - m_ if part_mode == "slope" else e_, v))
                    if part_mode == "trend":
                        # finding 77's follow-up (the user, 2026-09-28: judge by how the progress goes, not by the level): the
                        # part's advantage this session (on - off, clean check) goes into its ledger; it stays awake while the
                        # advantage is positive and has not shrunk more often than it grew -- "this is giving less and less"
                        on_v = [v for v in vals if v != ASLEEP.get(k)][0]
                        e_other = tried[0][0]
                        adv = (base - e_other) if cur == on_v else (e_other - base)
                        hist = ledger.setdefault(k, [])
                        hist.append(adv)
                        grew = sum(b_ > a_ for a_, b_ in zip(hist, hist[1:]))
                        shrank = sum(b_ < a_ for a_, b_ in zip(hist, hist[1:]))
                        if fit:
                            # finding 79 (the per-session sign flickered): a straight line through the whole ledger (least
                            # squares, advantage vs session) -- the part stays awake while the line's value now is positive
                            # (learning-curve extrapolation; all the record smooths one noisy session)
                            xs = np.arange(len(hist), dtype=float)
                            now = (np.polyval(np.polyfit(xs, hist, 1), xs[-1]) if len(hist) >= 2 else adv)
                            new_settings[k] = on_v if now > 0 else ASLEEP.get(k)
                        else:
                            new_settings[k] = on_v if (adv > 0 and shrank <= grew) else ASLEEP.get(k)
                        changed = float(changed == 1.0 or new_settings[k] != cur)
                    elif tried and max(tried)[0] > base:
                        new_settings[k] = max(tried, key=lambda x: x[0])[1]
                        changed = 1.0
                    else:
                        new_settings[k] = cur
            settings = new_settings
        sess_settings = settings
        if tune and tuned is not None:              # finding 115: this session and onwards answer with the chosen weights
            # own keys (not the route-knob "chunk_w", whose ladder the knob moves walk): the chosen absolute weight
            chosen = {"tune_chunk": tuned[0] / settings.get("chunk_w", 1.0), "assoc_w": tuned[1]}
            if tune_answer_only:
                # finding 117: the chosen weights only for answering -- the method layer's branches keep weight 1, so its
                # judge of the learning settings (review, lr, ...) is not looking through the tuned weights
                sess_settings = {**settings, **chosen}
            else:
                settings = sess_settings = {**settings, **chosen}
        checks = {"once": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "once"]}
        if _sd:
            SD.mark(i, "pre-session", me, {**_sd_x(), "args": ([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old},
                                                             sess_settings, checks)})
        curves = me.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, sess_settings, rng, curw,
                            checks=checks)
        if d25_mode == "M" and settings.get("d25_mark", "none") != "none":
            settings = dict(settings, d25_mark="none")      # D25 M: a mark is one-shot
        if longeye and shadow is not None:          # the shadow meets the same experience, with its own settings
            shadow.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, shadow_settings,
                           np.random.default_rng(mrng.integers(1 << 31)), curw)
        if method:
            me.drop = {}
            for c in seen:
                a = acc_items(me.model, D_positions(sents[c]))
                if c in acc_now:
                    me.drop[c] = acc_now[c] - a     # finding 108: how much the category dropped this session
                    lp = a - acc_now[c]
                    rpe[c] = lp - expect.get(c, 0.0)
                    prog[c] = lp
                    expect[c] = expect.get(c, 0.0) + RI.EXPECT_RATE * (lp - expect.get(c, 0.0))
                acc_now[c] = a
        if _sd:
            SD.mark(i, "post-method", me, _sd_x())
        if getattr(getattr(me.model, "wm", None), "measure_only", False):
            me.model.wm.silent = False              # "+wmmeasure": it answers when measured (after the method layer's own
                                                    # forgetting check above -- that check is one of the learner's decisions)
        st = measure(me, tests, seen, new_cats, curves, seed)
        st["older"] = acc_items(me.model, [(cx, t) for c in old for kind, cx, t in tests[c]]) if old else np.nan
        if d25_mode:                                # D25: the body alone (world model silent), then the sleep
            wm_d = getattr(me.model, "wm", None)
            if wm_d is not None:
                was_d = getattr(wm_d, "silent", False)
                wm_d.silent = True
                chs_d = [(s_, ans[0]) for c in seen for _, s_, ans in w.categories[c].get("chain", [])]
                st["d25 chain body"] = acc_items(me.model, chs_d) if chs_d else np.nan
                for k_, kinds_ in (("never-heard", None), ("once", ("once",)), ("unseen", ("unseen",))):
                    sub_d = [(cx, t) for c in seen for kind, cx, t in tests[c] if kinds_ is None or kind in kinds_]
                    st[f"d25 {k_} body"] = acc_items(me.model, sub_d) if sub_d else np.nan
                wm_d.silent = was_d
            st["d25 mark keep"] = float(sess_settings.get("d25_mark") == "keep")
            st["d25 mark forget"] = float(sess_settings.get("d25_mark") == "forget")
            st["d25 k"] = float(sess_settings.get("d25_k", np.nan))
            me.d25.sleep(me, seen)
            st.update(me.d25.record())
        if "@nw" in cond:                           # W2 (stage-1 criterion 2): new words heard 1 / 2 / 4 / 8 times
            for n_ in (1, 2, 4, 8):
                for tag_, rooms_ in (("new", new_cats), ("early", [c for c in seen if c in early_rooms]), ("all", seen)):
                    sub = [(s_, ans[0]) for c in rooms_ for k_, s_, ans in w.categories[c].get("nw", []) if k_ == f"nw{n_}"]
                    st[f"nw{n_} {tag_}"] = acc_items(me.model, sub) if sub else np.nan
                sub = [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == f"name{n_}"]
                st[f"name{n_} same"] = acc_items(me.model, sub) if sub else np.nan
        if epi is not None:                         # D80 v1: the rooms' moments in the order heard, then the keys
            for c in new_cats:
                epi.add(w.categories[c]["heard"])
                if epi2 is not None:
                    epi2.add(w.categories[c]["heard"])
            epi.learn()
            if epi2 is not None:
                epi2.learn()
                st.update(me.model.scene.record())  # the scene part's counted events (the told cells come from them)
            if "@seq" in cond:
                st.update(seq_keys(me.model, w, epi, new_cats, " new", tests, epi2))
                st.update(seq_keys(me.model, w, epi, seen, " all", tests, epi2))
        if "@nwf" in cond:                          # Q14 frame test: the same new-name items without / with the frame
            st.update(nwf_keys(me.model, w, new_cats, " new"))
            st.update(nwf_keys(me.model, w, seen, " all"))
            if "+eq" in cond:                       # D79 v1: the same items answered by the relational equals sign
                eq = lambda m_, c_: d79_logits(m_, c_, w.channel_of)
                st.update(nwf_keys(me.model, w, new_cats, " new eq", scorer=eq))
                st.update(nwf_keys(me.model, w, seen, " all eq", scorer=eq))
                for n_ in (1, 2, 4, 8):
                    for tag_, rooms_ in (("new", new_cats), ("all", seen)):
                        sub = [(s_, ans[0]) for c in rooms_ for k_, s_, ans in w.categories[c].get("nw", []) if k_ == f"nw{n_}"]
                        st[f"nw{n_} {tag_} eq"] = acc_scored(me.model, sub, eq) if sub else np.nan
                    sub = [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == f"name{n_}"]
                    st[f"name{n_} same eq"] = acc_scored(me.model, sub, eq) if sub else np.nan
        if "@scene" in cond:                        # D75 step 1: "where now" after a sentence, from a start never heard there
            sc_items = [x for c in seen for x in w.categories[c].get("scene", [])]
            if sc_items:
                st.update(four_way_keys(me.model, sc_items, "", "scene"))
                for m_ in (getattr(me.model, "scene", None), getattr(me.model, "wm", None)):
                    if m_ is not None:
                        m_.silent_was = getattr(m_, "silent", False)
                        m_.silent = True
                st.update(four_way_keys(me.model, sc_items, " body", "scene"))
                for m_ in (getattr(me.model, "scene", None), getattr(me.model, "wm", None)):
                    if m_ is not None:
                        m_.silent = m_.silent_was
                if getattr(me.model, "scene", None) is not None:
                    st.update(me.model.scene.record())
        if "@d72" in cond:                          # D72 (learning-power test): times heard so far + the four-way keys
            st.update(d72_keys(me.model, w, seen, "", conf="+d82" in cond))
            if epi is not None:                     # the same items answered with the order device (D80 v1.1 rule EQ1)
                st.update(d72_keys(me.model, w, seen, " eq1", scorer=lambda m_, c_: epi.scores(m_, c_, "EQ1")))
            wm_7 = getattr(me.model, "wm", None)
            if wm_7 is not None:                    # the body alone (the world model silent; D21)
                was_7 = getattr(wm_7, "silent", False)
                wm_7.silent = True
                st.update(d72_keys(me.model, w, seen, " body"))
                wm_7.silent = was_7
        if "@sp" in cond:                           # W3 (stage-1 criterion 9): the four spatial words, asked as words
            sp_ = [(k_, s_, ans) for c in seen for k_, s_, ans in w.categories[c].get("sp", [])]
            if sp_:
                st.update(sp_keys(me.model, sp_, w, ""))
                wm_s = getattr(me.model, "wm", None)
                if wm_s is not None:                # the body alone (the world model silent; D21)
                    was_s = getattr(wm_s, "silent", False)
                    wm_s.silent = True
                    st.update(sp_keys(me.model, sp_, w, " body"))
                    wm_s.silent = was_s
        if "@when" in cond:                         # W4 (stage-1 criterion 7): "when", asked by a cell the story never says
            wh_ = [(k_, s_, ans) for c in seen for k_, s_, ans in w.categories[c].get("when", [])]
            if wh_:
                st.update(when_keys(me.model, wh_, ""))
                wm_w = getattr(me.model, "wm", None)
                if wm_w is not None:                # the body alone (the world model silent; D21)
                    was_w = getattr(wm_w, "silent", False)
                    wm_w.silent = True
                    st.update(when_keys(me.model, wh_, " body"))
                    wm_w.silent = was_w
        if "@ten" in cond:                          # W5 (stage-1 criteria 3 and 4): numbers 1-10, put in / take out
            te_ = [(k_, s_, ans) for c in seen for k_, s_, ans in w.categories[c].get("ten", [])]
            if te_:
                st.update(ten_keys(me.model, te_, ""))
                wm_t = getattr(me.model, "wm", None)
                if wm_t is not None:                # the body alone (the world model silent; D21)
                    was_t = getattr(wm_t, "silent", False)
                    wm_t.silent = True
                    st.update(ten_keys(me.model, te_, " body"))
                    wm_t.silent = was_t
        if "@bi" in cond:                           # W6 (stage-1 criterion 5): two languages, told in one and asked in either
            bi_ = [(k_, s_, ans) for c in seen for k_, s_, ans in w.categories[c].get("bi", [])]
            if bi_:
                st.update(bi_keys(me.model, bi_, w, ""))
                wm_b = getattr(me.model, "wm", None)
                if wm_b is not None:                # the body alone (the world model silent; D21)
                    was_b = getattr(wm_b, "silent", False)
                    wm_b.silent = True
                    st.update(bi_keys(me.model, bi_, w, " body"))
                    wm_b.silent = was_b
        if "@c8" in cond:                           # W7b (stage-1 criterion 8): "where before" and long stories
            c8_ = [(k_, s_, ans) for c in seen for k_, s_, ans in w.categories[c].get("c8", [])]
            if c8_:
                st.update(c8_keys(me.model, c8_, w, ""))
                wm_c = getattr(me.model, "wm", None)
                if wm_c is not None:                # the body alone (the world model silent; D21)
                    was_c = getattr(wm_c, "silent", False)
                    wm_c.silent = True
                    st.update(c8_keys(me.model, c8_, w, " body"))
                    wm_c.silent = was_c
        if "+lpcur" in cond:                      # D35 (d): did the method layer use the curious pick this session
            st["pick curious"] = float(sess_settings.get("pick") == "curious")
        if "+squeeze" in cond:                      # finding 110: record only
            st.update(squeeze_probe(me, tests, seen))
        if "@multi" in cond and getattr(me.model, "wm", None) is not None:   # stage-1 step 3c: record only
            st.update(tell_probe(me, w, seen))
        if "+speak" in cond and getattr(me.model, "wm", None) is not None:   # stage-1 step 3b: record only
            st.update(speak_probe(me, w, seen))
        if "+vote" in cond:                         # finding 124: record only
            st.update(vote_probe(me, w, tests, seen, [x for c in new_cats for x in sents[c]],
                                 np.random.default_rng(7_000_000 + 100 * seed + i)))
        if "+errors" in cond:                       # finding 112: record only
            st.update(error_probe(me, w, tests, seen))
        if selfsim:                                 # finding 113: the same self-experiment on the outside test (checking only)
            st.update({f"cf out {k}": v for k, v in
                       counterfactual(me, [(cx, t) for c in seen for kind, cx, t in tests[c]]).items()})
        if getattr(me.model, "known_only", False) or "+knownrec" in cond:
            # D26 (1) S1, record only ("+knownrec": the same records in a life without the mask): the test items whose answer
            # word was never heard, and the accuracy on the items whose answer WAS heard (the fair comparison)
            nh = [(cx, t) for c in seen for kind, cx, t in tests[c]]
            st["unheard answer never-heard"] = float(np.mean([t not in me.known for _, t in nh])) if nh else np.nan
            ok = [(cx, t) for cx, t in nh if t in me.known]
            st["never-heard answerable"] = acc_items(me.model, ok) if ok else np.nan
            chs = [(s_, ans[0]) for c in seen for _, s_, ans in w.categories[c].get("chain", [])]
            st["unheard answer chain"] = float(np.mean([t not in me.known for _, t in chs])) if chs else np.nan
            okc = [(s_, t) for s_, t in chs if t in me.known]
            st["chain answerable"] = acc_items(me.model, okc) if okc else np.nan
        if chain_probe:                             # world-model stage A, record only: chain questions of the rooms met so far
            ch_items = [(s_, ans[0]) for c in seen for _, s_, ans in w.categories[c].get("chain", [])]
            st["chain"] = acc_items(me.model, ch_items) if ch_items else np.nan
            fl = [(s_, ans[0]) for c in seen if w.categories[c].get("flipped") for _, s_, ans in w.categories[c].get("chain", [])]
            st["chain flipped"] = acc_items(me.model, fl) if fl else np.nan
            if "@frame" in cond:                        # step 4+: by the room's frame (quarter turns / half turn)
                qt = lambda f_: 2 if f_ is True else 0 if f_ is False else int(f_)
                for tag_, turns in (("quarter", (1, 3)), ("half", (2,))):
                    it_ = [(s_, ans[0]) for c in seen if qt(w.categories[c].get("flipped", False)) in turns
                           for _, s_, ans in w.categories[c].get("chain", [])]
                    st[f"chain {tag_}"] = acc_items(me.model, it_) if it_ else np.nan
            sto = [(s_, ans[0]) for c in seen for _, s_, ans in w.categories[c].get("story", [])]
            st["story"] = acc_items(me.model, sto) if sto else np.nan
            if sto and getattr(me.model, "wm", None) is not None:   # diagnostic: the world model's own answer alone
                import lab2 as L_
                cx_, tg_ = L_.items_tensor(sto)
                wt_ = me.model.wm.term(cx_)
                st["story wm fired"] = float((wt_.sum(1) > 0).float().mean())
                st["story wm right"] = float(((wt_.argmax(1) == tg_) & (wt_.sum(1) > 0)).float().mean())
            de = [(s_, ans[0]) for c in seen for _, s_, ans in w.categories[c].get("describe", [])]
            if de:                                  # stage-1 step 3a: scene -> motion word (record only unless +wmrev)
                st["describe"] = acc_items(me.model, de)
                fl = [(s_, ans[0]) for c in seen if w.categories[c].get("flipped") for _, s_, ans in w.categories[c].get("describe", [])]
                st["describe flipped"] = acc_items(me.model, fl) if fl else np.nan
                if getattr(me.model, "wm", None) is not None:
                    import lab2 as L_
                    cx_, tg_ = L_.items_tensor(de)
                    wt_ = me.model.wm.term(cx_)
                    st["describe wm fired"] = float((wt_.sum(1) > 0).float().mean())
                    st["describe wm right"] = float(((wt_.argmax(1) == tg_) & (wt_.sum(1) > 0)).float().mean())
            sf_ = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("self", [])]
            for kn in ("which", "mirror"):          # D2: the self by contingency ("@self")
                sub = [(s_, a_) for k_, s_, a_ in sf_ if k_ == f"self {kn}"]
                if sub:
                    st[f"self {kn}"] = acc_items(me.model, sub)
            gl_ = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("goal", [])]
            for kn in ("which", "next"):            # D2: goal-directed motion ("@goal")
                sub = [(s_, a_) for k_, s_, a_ in gl_ if k_ == f"goal {kn}"]
                if sub:
                    st[f"goal {kn}"] = acc_items(me.model, sub)
            vw_ = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("views", [])]
            if vw_:                                 # stage-1 step 4+ (D17 v1): the three-view rooms
                lab_ = getattr(w, "view_labels", {})
                for kn in ("fill", "depth"):
                    sub = [(s_, a_) for k_, s_, a_ in vw_ if k_ == f"views {kn}"]
                    st[f"views {kn}"] = acc_items(me.model, sub) if sub else np.nan
                for kn in ("back", "turn"):             # D2 viewpoint ("@viewpt")
                    sub = [(s_, a_) for k_, s_, a_ in vw_ if k_ == f"viewpt {kn}"]
                    if sub:
                        st[f"viewpt {kn}"] = acc_items(me.model, sub)
                for tag_, keep in (("front", lambda s_: s_[-1] == lab_.get("VFRONT")),
                                   ("moved", lambda s_: s_[-1] != lab_.get("VFRONT"))):
                    sub = [(s_, a_) for k_, s_, a_ in vw_ if k_ == "views depth" and keep(s_)]
                    st[f"views depth {tag_}"] = acc_items(me.model, sub) if sub else np.nan
            nl_ = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("numline", [])]
            if nl_:                                 # stage-1 step 4+ (D13 / D15): the number-line room
                for kn in ("add", "sub"):
                    sub = [(s_, a_) for k_, s_, a_ in nl_ if k_ == f"numline {kn}"]
                    st[f"numline {kn}"] = acc_items(me.model, sub) if sub else np.nan
                wm_ = getattr(me.model, "wm", None)
                if wm_ is not None:                 # diagnostic: the world model's own answers on "sub" (fired / right)
                    import lab2 as L_
                    sb_ = [(s_, a_) for k_, s_, a_ in nl_ if k_ == "numline sub"]
                    if sb_:
                        cx_, tg_ = L_.items_tensor(sb_)
                        wt_ = wm_.term(cx_)
                        st["numline sub wm fired"] = float((wt_.sum(1) > 0).float().mean())
                        st["numline sub wm right"] = float(((wt_.argmax(1) == tg_) & (wt_.sum(1) > 0)).float().mean())
                    rooms_ = [c for c in seen if w.categories[c].get("numline")]
                    if rooms_:
                        wm_.refresh()
                        st["numline scale"] = float(np.mean([wm_.room_scale({c}) for c in rooms_]))
            cn = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("count", [])]
            if cn:                                  # stage-1 step 4+: small numbers
                for kn in ("put", "take", "four"):
                    sub = [(s_, a_) for k_, s_, a_ in cn if k_ == f"count {kn}"]
                    st[f"count {kn}"] = acc_items(me.model, sub) if sub else np.nan
                wm_ = getattr(me.model, "wm", None)
                nums_ = getattr(w, "numbers", {})
                if wm_ is not None and nums_:
                    wm_.learn_numbers()
                    st["count table right"] = float(all(wm_.num_table.get(n) == nums_[nm] for n, nm in
                                                        ((1, "HANA"), (2, "DUL"), (3, "SET"))))
            ct = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("contain", [])]
            if ct:                                  # stage-1 step 4+: containment
                for kc in ("in", "by"):
                    sub = [(s_, a_) for k_, s_, a_ in ct if k_ == f"contain {kc}"]
                    st[f"contain {kc}"] = acc_items(me.model, sub) if sub else np.nan
                wm_ = getattr(me.model, "wm", None)
                if wm_ is not None:
                    wm_.learn_rel()
                    pof = getattr(w, "particle_of", {})
                    st["contain table right"] = float(wm_.rel_effect.get(pof.get("ANE"), None) is True and
                                                      wm_.rel_effect.get(pof.get("YEOPE"), None) is False)
            gr = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("grav", [])]
            if gr:                                  # stage-1 step 4+: gravity / support
                for kg in ("floor", "stack", "step"):
                    sub = [(s_, a_) for k_, s_, a_ in gr if k_ == f"grav {kg}"]
                    st[f"grav {kg}"] = acc_items(me.model, sub) if sub else np.nan
                wm_ = getattr(me.model, "wm", None)
                if wm_ is not None:                 # D21 / D22: the body alone on the real gravity questions (record only)
                    was = getattr(wm_, "silent", False)
                    wm_.silent = True
                    for gname, flip in (("unturned", False), ("turned", True)):
                        sub = [(s_, ans[0]) for c in seen if w.categories[c].get("flipped") is flip
                               for k_, s_, ans in w.categories[c].get("grav", []) if k_ in ("grav floor", "grav stack")]
                        st[f"grav body {gname}"] = acc_items(me.model, sub) if sub else np.nan
                    tl = [(s_, ans[0]) for c in seen for k_, s_, ans in w.categories[c].get("tale", []) if k_ == "tale in"]
                    if tl:
                        st["tale in body"] = acc_items(me.model, tl)
                    wm_.silent = was
                if wm_ is not None:
                    wm_.learn_until()
                    st["grav falls repeats"] = float(wm_.until_of.get(int(w.verbs[0]), False))
                    st["grav others repeat"] = float(np.mean([wm_.until_of.get(int(v), False) for v in w.verbs[1:]]))
            if any(w.categories[c].get("tale") for c in seen):     # D22: frames vs reality (record only)
                for kt in ("in", "transfer"):
                    sub = [(s_, ans[0]) for c in seen for k_, s_, ans in w.categories[c].get("tale", []) if k_ == f"tale {kt}"]
                    st[f"tale {kt}"] = acc_items(me.model, sub) if sub else np.nan
                groups = {"clean": [c for c in seen if w.categories[c].get("tale_drop") is False],
                          "drop": [c for c in seen if w.categories[c].get("tale_drop") is True],
                          "turned": [c for c in seen if w.categories[c].get("flipped") is True]}
                for gname, cs in groups.items():    # the rooms' real gravity questions (floor + stack)
                    sub = [(s_, ans[0]) for c in cs for k_, s_, ans in w.categories[c].get("grav", [])
                           if k_ in ("grav floor", "grav stack")]
                    st[f"tale real {gname}"] = acc_items(me.model, sub) if sub else np.nan
                # v1b: rooms known only from tales -- c told about in c - 1's session, c itself not yet seen: its real gravity
                # questions (combined answer and the body alone, D21)
                wm_t = getattr(me.model, "wm", None)
                if wm_t is not None and wm_t.tale_mode == "anchor":
                    st["tale flagged"] = float(wm_t.tale_flagged)   # unmarked moments taken as tales by the anchor
                for gname, dropv in (("marked", False), ("unmarked", True)):
                    cs = [c for c in range(1, w.n_cat) if w.categories[c].get("pre_drop") is dropv and c - 1 in seen and c not in seen]
                    sub = [(s_, ans[0]) for c in cs for k_, s_, ans in w.categories[c].get("grav", [])
                           if k_ in ("grav floor", "grav stack")]
                    st[f"tale preview {gname}"] = acc_items(me.model, sub) if sub else np.nan
                    if sub and wm_t is not None:
                        was = getattr(wm_t, "silent", False)
                        wm_t.silent = True
                        st[f"tale preview {gname} body"] = acc_items(me.model, sub)
                        wm_t.silent = was
            ob = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("objs", [])]
            if ob:                                  # stage-1 step 4: objects with properties
                for ko in ("hit", "free"):
                    sub = [(s_, a_) for k_, s_, a_ in ob if k_ == f"objs {ko}"]
                    st[f"objs {ko}"] = acc_items(me.model, sub) if sub else np.nan
                wm_ = getattr(me.model, "wm", None)
                if wm_ is not None:
                    wm_.learn_props()
                    names_ = [n for n in wm_.prop_table if n in getattr(w, "heavy", {})]
                    st["objs table right"] = (float(np.mean([(wm_.prop_table[n] == "heavy") == w.heavy[n] for n in names_]))
                                              if names_ else np.nan)
                    st["objs table size"] = float(len(wm_.prop_table))
            vo = [(kind_, s_, ans) for c in seen for kind_, s_, ans in
                  w.categories[c].get("voe", []) + w.categories[c].get("voeperm", [])]
            if vo:                                  # step 4+: violation of expectation, as infant looking-time studies
                import lab2 as L_
                cx_, _ = L_.items_tensor([(s_, ans[0]) for _, s_, ans in vo])
                with torch.no_grad():
                    lp_ = F.log_softmax(me.model.complete(cx_), dim=1)
                good_ = torch.tensor([ans[0] for _, _, ans in vo])
                bad_ = torch.tensor([ans[1] for _, _, ans in vo])
                s_good = -lp_.gather(1, good_.unsqueeze(1)).squeeze(1)      # surprise = -log p(outcome shown)
                s_bad = -lp_.gather(1, bad_.unsqueeze(1)).squeeze(1)
                longer = (s_bad > s_good).float() + 0.5 * (s_bad == s_good).float()
                for kv in ("solid", "push", "causal", "teleport", "float", "through", "permanence", "identity",
                           "inside", "beside", "number", "four"):
                    m_ = torch.tensor([k_ == f"voe {kv}" for k_, _, _ in vo])
                    if m_.any():
                        st[f"voe {kv}"] = float(longer[m_].mean())
                        st[f"voe {kv} gap"] = float((s_bad - s_good)[m_].clamp(-50, 50).mean())
            mu = [(kind_, s_, ans[0]) for c in seen for kind_, s_, ans in w.categories[c].get("multi", [])]
            if mu:                                  # stage-1 step 2: multi-moment stories, by kind
                st["multi"] = acc_items(me.model, [(s_, a_) for _, s_, a_ in mu])
                for km in ("same", "two", "three"):
                    sub = [(s_, a_) for k_, s_, a_ in mu if k_ == f"multi {km}"]
                    st[f"multi {km}"] = acc_items(me.model, sub) if sub else np.nan
                if getattr(me.model, "wm", None) is not None:
                    import lab2 as L_
                    cx_, tg_ = L_.items_tensor([(s_, a_) for _, s_, a_ in mu])
                    wt_ = me.model.wm.term(cx_)
                    st["multi wm fired"] = float((wt_.sum(1) > 0).float().mean())
                    st["multi wm right"] = float(((wt_.argmax(1) == tg_) & (wt_.sum(1) > 0)).float().mean())
                    st["multi trust"] = me.model.wm.track_trust[0] / max(me.model.wm.track_trust[1], 1)
        if graded and last_pick is not None:        # finding 119, checking only: the eye's latest choice on the outside test
            m_ = me.model
            keep_ = (getattr(m_, "chunk_w", 1.0), getattr(m_, "assoc_w", 1.0))
            m_.chunk_w, m_.assoc_w = 1.0, 1.0
            cf_ = counterfactual(me, [(cx, t) for c in seen for kind, cx, t in tests[c]])
            m_.chunk_w, m_.assoc_w = keep_
            st.update({f"out gain {p_}": cf_[f"{p_} {w_}"] - cf_[f"{p_} 1.0"] for p_, w_ in zip(("chunk", "assoc"), last_pick)})
        st.update({"review": settings["review"], "lr": settings["lr"], "trial epochs": cost, "branches": n_br,
                   "changed": changed, "intact": float(me.intact),
                   **{f"awake {k}": float(settings.get(k, ASLEEP[k]) != ASLEEP[k]) for k in WAKE},
                   "recomb awake": float(settings.get("setter") == "random"),
                   "budget level": float(LEVELS.index(budget)) if method and i > 0 else np.nan,
                   "side w": settings.get("side_w", np.nan), "side decay": settings.get("side_decay", np.nan),
                   "chunk mode": settings.get("chunk_mode", "idf"), "trust imagination": tau if trust2 else np.nan,
                   "swaps": float(swaps) if longeye else np.nan,
                   "pattern w": settings.get("pattern_w", np.nan), "chunk w": settings.get("chunk_w", np.nan), **dash,
                   "assoc w": sess_settings.get("assoc_w", np.nan), "tune chunk": sess_settings.get("tune_chunk", np.nan),
                   "protect": float(mode == "protect") if (use_modes or bundle) else np.nan,
                   "settle": float(mode == "settle") if bundle else np.nan,
                   "plastic": float(bool(settings.get("plastic"))) if "plastic" in settings else np.nan,
                   "forget pick": float(bool(settings.get("forget_pick"))) if "forget_pick" in settings else np.nan,
                   "rungs added": float(n_ext) if openended else np.nan,
                   "pairs won": float(n_pair_won) if openended else np.nan, **probe_rec,
                   **({f"on {k}": float(settings.get(k, v[0]) != v[0])
                       for k, v in {**REDISCOVER, **(REDISCOVER_GLYPH if ascii_world else {})}.items()}
                      if settings.get("rediscover") else {})})
        probe_rec = {}
        if not np.isnan(dash["dash old heard"]):
            held.append(dash["dash old heard"])
        if not np.isnan(dash["dash options"]):
            opts.append(dash["dash options"])
        wm_g = getattr(me.model, "wm", None)
        if wm_g is not None and getattr(wm_g, "gate", False):     # D69: each part's record and whether it speaks now
            import worldmodel as WM_
            for p_, a_ in WM_.GATED:
                if getattr(wm_g, a_, False):
                    h_, n_ = wm_g.part_trust.get(p_, (0, 0))
                    st[f"gate {p_} speaks"] = float(wm_g.part_ok(p_))
                    st[f"gate {p_} differed"] = float(n_)
                    st[f"gate {p_} right"] = float(h_)
        if wm_g is not None and getattr(wm_g, "useful", False):   # D83 (1): each part's usefulness record and where it speaks
            import worldmodel as WM_
            for p_, a_ in WM_.GATED:
                if getattr(wm_g, a_, False):
                    net_, n_ = wm_g.use_all.get(p_, (0, 0))
                    sg_ = [s_ for (q_, s_) in wm_g.use_rec if q_ == p_]
                    st[f"use {p_} net"] = float(net_)
                    st[f"use {p_} changed"] = float(n_)
                    st[f"use {p_} forms"] = float(len(sg_))
                    st[f"use {p_} says in"] = float(sum(wm_g.use_say(p_, s_) for s_ in sg_))
                    st[f"use {p_} said rows"] = float(wm_g.use_count.get(f"say {p_}", 0))
            st["use rows"] = float(wm_g.use_count.get("rows", 0))
            dec_ = wm_g.use_count.get("backoff", 0) + wm_g.use_count.get("in form", 0)
            st["use backoff share"] = float(wm_g.use_count.get("backoff", 0) / max(dec_, 1))  # of decisions made by a record
        steps.append(st)
        LIFE_CPU["cum"].append(time.process_time() - _cpu0)
        progress_line(cond, seed, i, len(sessions))                # D57: a record only, never part of the result
        resume_save(cond, seed, i, len(sessions), locals())        # D11: a record only, never part of the result
        if _sd:
            SD.mark(i, "end", me, _sd_x())
            if i + 1 >= int(os.environ.get("VOY_STATEDIFF_STOP", "99")):
                return steps
    return steps


# D51 (2026-10-14, the user: compare the cost of equal performance / the performance at equal cost): the life's process CPU
# time after each session, kept OUTSIDE the life's records (a side file per cached life, cache_long_v2_cpu/), so no record
# changes. The CPU time is divided by a fixed reference job timed in the same process at the life's start (machines and
# their load differ; wall time doubles on a busy machine).
LIFE_CPU = {"ref": None, "cum": []}


def ref_job():
    """the fixed reference job (D51): a small float matrix product loop and a small Python loop; returns its process CPU
    seconds."""
    t = time.process_time()
    g = torch.Generator().manual_seed(0)
    a = torch.rand(256, 256, generator=g)
    for _ in range(400):
        a = torch.tanh(a @ a / 256)
    x = 0
    for i in range(2_000_000):
        x = (x * 31 + i) % 1_000_003
    return time.process_time() - t


# D57 (2026-10-15, the user: "결과가 나올때까지 기다리기만 하면 안돼고 중간중간 체크하는것도 중요한거군"; the runner change
# approved 10-15 "응, 너의 의견대로 하자"). RECORDS ONLY -- nothing here changes a life or a report:
# (1) a progress line per session in a side file, cache_long_v2_progress/seed<s>_<cond>.log (UTC time, session k/n, the
#     session's CPU seconds, the cumulative cost in reference-job units, the alarm line);
# (2) an over-expectation alarm whose line comes from the records, not a constant: the recorded lives of the same condition
#     (D51 side files, other seeds) give the cumulative units after each session; the line at session k = their max plus
#     their range (max - min); fewer than 3 recorded lives = no line. A crossing is written to the life's log AND to
#     cache_long_v2_progress/ALARMS.log (the check-in reads both);
# (3) map_lives(): one process per life (= chunksize 1); a dead worker (e.g. the kernel's out-of-memory kill, exit -9) is
#     named on stderr at once and the step FAILS at the end (the other lives finish and are cached first) instead of
#     waiting forever as Pool.map did (10-14: three lives lost silently); a new life starts only when the free memory
#     covers the running lives' expected peaks (recorded "peak_rss_gb" of the condition, else RIE_LIFE_GB, else 2 GB)
#     plus its own -- at least one life always runs.
PROGRESS = CACHE + "_progress"
_ALARM = {}


def _side_files(cond, seed=None):
    import glob
    import json
    import re
    pat = re.compile(r"seed(\d+)_" + re.escape(cond) + r"\.json")
    out = []
    for f in glob.glob(os.path.join(CACHE + "_cpu", "seed*.json")):
        m = pat.fullmatch(os.path.basename(f))
        if m and (seed is None or int(m.group(1)) != seed):
            try:
                with open(f) as fh:
                    out.append(json.load(fh))
            except Exception:
                pass
    return out


def alarm_line(cond, seed):
    """per session: the max of the recorded lives' cumulative units plus their range (None where < 3 lives)"""
    rows = [[x / d["ref_cpu_s"] for x in d["cpu_cum_s"]] for d in _side_files(cond, seed) if d.get("ref_cpu_s")]
    n = max((len(r) for r in rows), default=0)
    line = []
    for k in range(n):
        v = [r[k] for r in rows if len(r) > k]
        line.append(max(v) + (max(v) - min(v)) if len(v) >= 3 else None)
    return line


def progress_line(cond, seed, i, n):
    try:
        if i == 0 or (cond, seed) not in _ALARM:
            _ALARM[(cond, seed)] = alarm_line(cond, seed)
        line = _ALARM[(cond, seed)]
        cum, ref = LIFE_CPU["cum"], LIFE_CPU["ref"]
        units = cum[-1] / ref
        lim = line[i] if i < len(line) else None
        over = lim is not None and units > lim
        txt = (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} session {i + 1}/{n} cpu {cum[-1] - (cum[-2] if len(cum) > 1 else 0):.0f}s "
               f"units {units:.0f} line {'-' if lim is None else f'{lim:.0f}'}{' ALARM' if over else ''}")
        os.makedirs(PROGRESS, exist_ok=True)
        with open(os.path.join(PROGRESS, f"seed{seed}_{cond}.log"), "a") as f:
            f.write(txt + "\n")
        if over:
            with open(os.path.join(PROGRESS, "ALARMS.log"), "a") as f:
                f.write(f"{txt} | pid {os.getpid()} seed {seed} {cond}\n")
    except Exception as e_:
        print(f"progress line not written: {e_}", file=sys.stderr)


def _mem_free_gb():
    """the smaller of MemAvailable and the cgroup's room (memory.max - memory.current); None if unreadable"""
    free = None
    try:
        with open("/proc/meminfo") as f:
            for l in f:
                if l.startswith("MemAvailable:"):
                    free = int(l.split()[1]) / 2 ** 20
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            mx = f.read().strip()
        if mx != "max":
            with open("/sys/fs/cgroup/memory.current") as f:
                room = (int(mx) - int(f.read())) / 2 ** 30
            free = room if free is None else min(free, room)
    except Exception:
        pass
    return free


def _rss_gb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for l in f:
                if l.startswith("VmRSS:"):
                    return int(l.split()[1]) / 2 ** 20
    except Exception:
        pass
    return 0.0


def life_peak_gb(task):
    """the expected peak memory of a task: the recorded peaks of its condition, else RIE_LIFE_GB, else 2 GB; a life that
    will go on from a written D11 state needs more -- x1.6 (check_resume: the @ten life with every part on, resumed after
    session 11, held 6.8 GB late in life against a 4.3 GB peak uninterrupted; a gate only, never part of a result)"""
    base = None
    try:
        sides = [d for d in _side_files(task[1]) if d.get("peak_rss_gb")]
        whole = [d["peak_rss_gb"] for d in sides if not d.get("resumed")]
        peaks = whole or [d["peak_rss_gb"] for d in sides]   # a resumed life's peak is not a whole life's (x1.6 below)
        if peaks:
            base = 1.1 * max(peaks)
    except Exception:
        pass
    if base is None:
        base = float(os.environ.get("RIE_LIFE_GB", 2.0))
    try:
        if _resume_ok() and os.path.exists(resume_path(task[1], task[0])):
            return 1.6 * base
    except Exception:
        pass
    return base


def map_lives(tasks, workers, fn=None, peak=None):
    """D57: like Pool(workers).map(fn, tasks, chunksize=1), but a dead worker fails loudly and memory gates new starts."""
    import multiprocessing as mp
    from multiprocessing.connection import wait
    fn = fn or one
    peak = peak or life_peak_gb
    ctx = mp.get_context("fork")
    tmp = CACHE + "_tmp"
    os.makedirs(tmp, exist_ok=True)
    tag = f"{os.getpid()}_{time.time_ns()}"
    path = lambda j: os.path.join(tmp, f"{tag}_{j}.pkl")

    def child(j):
        res = fn(tasks[j])
        with open(path(j) + ".part", "wb") as f:
            pickle.dump(res, f)
        os.replace(path(j) + ".part", path(j))

    todo, running, results, lost = list(range(len(tasks))), {}, {}, []
    need = {}

    def room(j):
        if not running:
            return True
        free = _mem_free_gb()
        if free is None:
            return True
        if j not in need:
            need[j] = peak(tasks[j])
        owed = sum(max(0.0, need[k] - _rss_gb(p.pid)) for k, p in running.items())
        return free - owed >= need[j]

    while todo or running:
        while todo and len(running) < workers and room(todo[0]):
            j = todo.pop(0)
            if j not in need:
                need[j] = peak(tasks[j])
            p = ctx.Process(target=child, args=(j,))
            p.start()
            running[j] = p
        if running:
            wait([p.sentinel for p in running.values()], timeout=20)
        for j, p in list(running.items()):
            if p.exitcode is None:
                continue
            p.join()
            del running[j]
            if p.exitcode == 0 and os.path.exists(path(j)):
                with open(path(j), "rb") as f:
                    results[j] = pickle.load(f)
                os.remove(path(j))
            else:
                lost.append(tasks[j])
                why = " (killed by a signal -- e.g. the kernel's out-of-memory kill)" if p.exitcode and p.exitcode < 0 else ""
                print(f"LIFE LOST (D57): task {tasks[j]} -- its process ended with exit code {p.exitcode}{why}",
                      file=sys.stderr, flush=True)
    if lost:
        raise RuntimeError(f"D57: {len(lost)} task(s) lost by dead workers: {lost}; the finished lives are cached -- "
                           "rerun the step (it skips them)")
    return [results[j] for j in range(len(tasks))]


class PieceDeadline(Exception):
    """2026-10-13 (user decision): the queue runs in PIECES that finish within the background-task limit. After a piece's
    deadline (environment RIE_DEADLINE, epoch seconds) no new life starts; a life already running finishes and is cached."""


def one(task):
    """one (seed, condition) life, cached on its own (a container restart loses at most one life per worker)."""
    seed, c = task
    path = os.path.join(CACHE, f"seed{seed}_{c}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    dl = os.environ.get("RIE_DEADLINE")
    if dl and time.time() >= float(dl):
        # no new life after the piece's deadline; returning (not raising) lets the other workers' running lives finish
        # and be cached -- an exception would end the pool at once and kill them. The marker goes to the step's output
        # file (stderr), so run_detached.sh's step() ends the piece and reruns this step next piece.
        print(f"PIECE DEADLINE: no new life after the piece's deadline (seed {seed}, {c})", file=sys.stderr, flush=True)
        return None
    torch.set_num_threads(1)
    t_wall = time.time()
    out = life(c, seed)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    if os.path.exists(resume_path(c, seed)):        # D11: the finished life needs no resume state
        os.remove(resume_path(c, seed))
    try:                                            # D51: the cost side file (never part of the result)
        import json
        os.makedirs(CACHE + "_cpu", exist_ok=True)
        with open(os.path.join(CACHE + "_cpu", f"seed{seed}_{c}.json"), "w") as f:
            import resource
            json.dump({"ref_cpu_s": LIFE_CPU["ref"], "cpu_cum_s": LIFE_CPU["cum"], "wall_s": time.time() - t_wall,
                       "host": os.uname().nodename,
                       "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                       "resumed": bool(LIFE_CPU.get("resumed"))}, f)
    except Exception as e_:
        print(f"cost side file not written: {e_}", file=sys.stderr)
    return out


def load_seed(seed, done):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    out = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            out = pickle.load(f)
    if any(c not in out for c in CONDITIONS):
        out.update({c: done[(seed, c)] for c in CONDITIONS if c not in out})
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def have(seed):
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as f:
        return set(pickle.load(f))


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(CACHE, exist_ok=True)
    tasks = [(sd, c) for sd in range(n_seeds) for c in CONDITIONS if c not in have(sd)]
    tasks.sort(key=lambda t: (t[1] == "content", t[0]))   # long (method) lives first, cheap ones fill the gaps
    with Pool(workers) as pool:
        done = dict(zip(tasks, pool.map(one, tasks, chunksize=1)))
    results = [load_seed(sd, done) for sd in range(n_seeds)]
    n = len(results[0]["content"])

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    def sgn(dd):
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        return f"{dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    def v(r, c, key, sl=slice(None)):
        return np.nanmean([s[key] for s in r[c][sl]])

    print(f"=== {n_seeds} seeds; long test (life32: s1-8 rule A, s9-12 new kind, s13-16 all rules change) ===")
    keys = ["never-heard", "newest", "older", "once all", "unseen all", "fmt options", "review", "lr", "trial epochs",
            "branches", "changed", "intact"] + [f"awake {k}" for k in WAKE]
    for key in keys:
        print(f"\n--- {key} (sessions 1..{n}) ---")
        for c in CONDITIONS:
            print(f"{c:15} " + " ".join(f"{m([r[c][i].get(key, np.nan) for r in results]):>6}" for i in range(n)))
    print("\n--- paired (never-heard) ---")
    for g, b in [("both", "content"), ("both+wake", "both"), ("both+recomb", "both"), ("both-low", "both"),
                 ("both-low+aware", "both-low"), ("both-low+aware", "content"), ("content+recomb", "content"),
                 ("both+recomb", "content+recomb")]:
        dl = np.array([v(r, g, "never-heard") - v(r, b, "never-heard") for r in results])
        de = np.array([r[g][-1]["never-heard"] - r[b][-1]["never-heard"] for r in results])
        dc = np.array([v(r, g, "newest", slice(12, 16)) - v(r, b, "newest", slice(12, 16)) for r in results])
        print(f"{g:15} vs {b:9} life {sgn(dl)}  end {sgn(de)}  newest after the rule change (s13-16) {sgn(dc)}")
    for g in CONDITIONS[1:]:
        gaps = np.array([[r[g][i]["never-heard"] - r["content"][i]["never-heard"] for i in range(n)] for r in results])
        print(f"\n{g}: gap to content by session " + " ".join(f"{x:+.3f}" for x in gaps.mean(0)))
        print(f"{g}: end {sgn(gaps[:, -1])} | widening s13-16 - s2-4 {sgn(gaps[:, 12:16].mean(1) - gaps[:, 1:4].mean(1))}"
              f" | s9-12 - s2-4 {sgn(gaps[:, 8:12].mean(1) - gaps[:, 1:4].mean(1))}")
    modes = {}
    for r in results:
        for s in r["both+wake"]:
            modes[s["chunk mode"]] = modes.get(s["chunk mode"], 0) + 1
    print(f"\nboth+wake chunk modes (session count): {modes}")
    print("budget: trial epochs per life " + "  ".join(
        f"{c} {np.mean([np.nansum([s['trial epochs'] for s in r[c]]) for r in results]):.0f}" for c in CONDITIONS[1:]))


if __name__ == "__main__":
    main()
