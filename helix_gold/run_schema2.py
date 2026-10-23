"""Content layer, second experiment: keep practice from overwriting what the templates understood, and let unused
templates rest instead of cluttering (finding 52 follow-up, 2026-09-23).

Finding 52: templates gave first-sight understanding that grew session after session, but practising a pair heard
in a single sentence overwrote half of it (the fillers of that sentence were learned as if they were partners too),
and chance templates piled up (33 confirmed templates for 4 true relation kinds).
  explaining away -- when a sentence contains a pair that an active template explains (a and b both in it, with a
                     template match), the other words of that sentence are taken as incidental: practice items that
                     predict one of those other words from a context holding the explained pair are weighted by
                     1 - (the template's confidence x match). Items that predict the explained partner itself keep
                     full weight. (Like not tying a stranger to a friend just because they stood together.)
  resting         -- a template is ACTIVE (used in the pattern channel) if it is confirmed at least twice and was
                     confirmed within the last REST sessions or at least ESTABLISHED times; otherwise it RESTS -- kept,
                     never deleted (user principle 6), and woken up again as soon as a new direction matches it
                     (resting templates still take part in assimilation). Of two active templates that are nearly
                     the same (|cos| >= VIGILANCE), only the more confirmed one is used (user, 2026-09-23: people keep
                     both, use the more reliable one, and may bring the other back when things change).
Learners (same seeds, both worlds): D / D+pattern (finding 52) / +explain / +rest / +both.
Representative values: REST 2 sessions, ESTABLISHED 4 confirmations.
"""
import math
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import data_v3 as W
import lab2 as L
from run_seed_transfer import memorize
from run_unified_continual import sign_p
from run_schema import (PatternMemory, PatternKnowDo, ConfDirectionalHebbian, acc_items, D_positions,
                        EPOCHS, REVIEW, LR, PER_SESSION, CHECKS, VIGILANCE, WORLDS)

REST, ESTABLISHED = 2, 4
LEARNERS = ["D", "D+pattern", "D+pattern+explain", "D+pattern+rest", "D+pattern+both"]


class RestingPatternMemory(PatternMemory):
    """templates carry [direction, count, last session confirmed]; absorb() matches resting ones too."""

    def __init__(self, resting):
        super().__init__()
        self.resting = resting
        self.session = 0

    def absorb(self, hebb, known, memory):
        before = [t[1] for t in self.templates]
        super().absorb(hebb, known, memory)
        for k, t in enumerate(self.templates):
            if len(t) == 2:
                t.append(self.session)                        # new template
            elif k < len(before) and t[1] > before[k]:
                t[2] = self.session                           # confirmed again (wakes up if resting)

    def established(self):
        cands = [t for t in self.templates if t[1] >= 2]
        if not self.resting:
            return [(t[0], t[1]) for t in cands]
        active = [t for t in cands if self.session - t[2] <= REST or t[1] >= ESTABLISHED]
        active.sort(key=lambda t: -t[1])
        chosen = []
        for t in active:                                      # of near-duplicates keep the more confirmed one
            if all(abs(float(F.cosine_similarity(t[0], u[0], dim=0))) < VIGILANCE for u in chosen):
                chosen.append(t)
        return [(t[0], t[1]) for t in chosen]


def explain_weights(sentences, model, P_conf):
    """per training item weight: 1 - explained strength if the target is an incidental word of a sentence whose
    context holds a template-explained pair."""
    weights = []
    for s in sentences:
        ex = 0.0
        explained = set()
        for i, a in enumerate(s):
            for b in s[i + 1:]:
                c = max(float(P_conf[a, b]), float(P_conf[b, a]))
                if c > 0:
                    explained |= {a, b}
                    ex = max(ex, c)
        for i, y in enumerate(s):
            if explained and y not in explained and (explained - {y}):
                weights.append(1.0 - ex)
            else:
                weights.append(1.0)
    return torch.tensor(weights)


def conf_matrix(pattern, memory, known, V):
    """template confidence x match (0..1) for known word pairs, from active templates."""
    C = torch.zeros(V, V)
    words = torch.tensor(sorted(known))
    m = memory[words]
    for t, n in pattern.established():
        conf = n / (n + 1)
        for sign in (1, -1):
            match = 1 - torch.cdist(m + sign * t, m) / t.norm()
            match.fill_diagonal_(0)
            val = torch.where(match > 0.5, conf * match, torch.zeros_like(match))
            C[words.unsqueeze(1), words.unsqueeze(0)] = torch.maximum(C[words][:, words], val)
    return C


def life(world_name, learner, seed):
    w = W.build(seed, schema=world_name == "schema")
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    vecs = torch.tensor(w.vecs)
    cats = torch.tensor(w.cats)
    memory, _ = memorize(vecs, seed)
    use_pattern = learner != "D"
    explain = learner.endswith("explain") or learner.endswith("both")
    resting = learner.endswith("rest") or learner.endswith("both")
    model = PatternKnowDo(w.V, w.n_cat, memory, seed) if use_pattern else L.KnowDo2(w.V, w.n_cat, memory, seed)
    hebb = ConfDirectionalHebbian(w.V)
    pattern = RestingPatternMemory(resting) if use_pattern else None
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for c in rng.permutation(w.first_half)] + [int(c) for c in rng.permutation(w.second_half)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    true = W.true_pairs(w)
    T = {k: torch.stack([memory[b] - memory[a] for a, b in pairs]).mean(0) for k, pairs in true.items()}
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
        hebb.observe(heard)
        conn = hebb.connections(sorted(known))
        nw = torch.tensor(sorted(new_words))
        with torch.no_grad():
            model.assoc[nw, :] = conn[nw, :]
            model.assoc[:, nw] = conn[:, nw]
        if use_pattern:
            pattern.session = i
            C = conf_matrix(pattern, memory, known, w.V)
            with torch.no_grad():
                model.pattern.copy_(C * math.log(len(known)))
        ctx, tgt = L.items_tensor(D_positions(heard))
        bag = L.bag(ctx, w.V)
        extra = {"pterm": model.pattern_term(ctx)} if use_pattern else {}
        wts = explain_weights(heard, model, C) if explain else torch.ones(len(tgt))
        ids = torch.tensor(sorted({x for s in heard for x in s}))
        newest_once = [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "once"]
        newest_unseen = [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "unseen"]
        newest = [(cx, t) for c in new_cats for kind, cx, t in tests[c]]
        curve, curve_once, curve_unseen = {}, {}, {}
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for e in range(EPOCHS + 1):
            if e in CHECKS:
                curve[e] = acc_items(model, newest)
                curve_once[e] = acc_items(model, newest_once)
                curve_unseen[e] = acc_items(model, newest_unseen)
            if e == EPOCHS:
                break
            opt.zero_grad()
            ce = F.cross_entropy(model.complete(bag, **extra), tgt, reduction="none")
            loss = (ce * wts).sum() / wts.sum() + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()
        if use_pattern:
            pattern.absorb(hebb, known, memory)
        active = pattern.established() if use_pattern else []
        junk = sum(1 for t, _ in active if max(abs(float(F.cosine_similarity(t, T[k], dim=0))) for k in T) < 0.8)
        align3 = max((abs(float(F.cosine_similarity(T[W.NEW_KIND], t, dim=0))) for t, _ in active), default=0.0)
        steps.append({"curve": curve, "curve once": curve_once, "curve unseen": curve_unseen,
                      "never-heard": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c]]),
                      "once all": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "once"]),
                      "unseen all": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "unseen"]),
                      "pairs all": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind.startswith("R")]),
                      "active": len(active) if use_pattern else np.nan,
                      "stored": len([t for t in pattern.templates if t[1] >= 2]) if use_pattern else np.nan,
                      "junk active": junk if use_pattern else np.nan, "R3 align": align3 if use_pattern else np.nan})
    return steps


def one_seed(seed):
    torch.set_num_threads(1)
    return {(wn, ln): life(wn, ln, seed) for wn in WORLDS for ln in LEARNERS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_sess = len(results[0][(WORLDS[0], LEARNERS[0])])
    print(f"=== {n_seeds} seeds; data_v3; REST={REST} ESTABLISHED={ESTABLISHED}; rows per learner: sessions 1..{n_sess} ===")
    for key, label in [("curve once", "newest ONE-SHOT pairs at epoch"), ("curve unseen", "newest UNSEEN pairs at epoch"),
                       ("curve", "newest categories (all kinds) at epoch")]:
        print(f"\n--- {label} {CHECKS} ---")
        for wn in WORLDS:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:18} " + " | ".join(
                    " ".join(f"{np.nanmean([r[(wn, ln)][i][key][e] for r in results]):.3f}" for e in (0, 50, 200))
                    for i in range(n_sess)))
    for key in ["never-heard", "pairs all", "once all", "unseen all", "active", "stored", "junk active", "R3 align"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                print(f"{wn:6} {ln:18} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", lambda r, g: np.mean([st["never-heard"] for st in r[g]])),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("pairs end", lambda r, g: r[g][-1]["pairs all"]),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("unseen end", lambda r, g: r[g][-1]["unseen all"]),
                         ("once@200 s2-8", lambda r, g: np.mean([st["curve once"][200] for st in r[g][1:]])),
                         ("once kept", lambda r, g: np.mean([st["curve once"][200] - st["curve once"][0] for st in r[g][1:]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:40} " + "  ".join(cells))
    for wn in WORLDS:
        cmp((wn, "D+pattern"), (wn, "D"), f"{wn}: pattern vs D")
        for ln in LEARNERS[2:]:
            cmp((wn, ln), (wn, "D+pattern"), f"{wn}: {ln} vs pattern")
        cmp((wn, "D+pattern+both"), (wn, "D"), f"{wn}: both vs D")


if __name__ == "__main__":
    main()
