"""Separating "knowing" from "doing".

Findings 29-30: in design A the output IS the memory values, so knowing a
word's connections and being able to produce it are the same thing. The
external logic starts from the internal values, the internal goal is met at
epoch 0, a "satisfied" learner never practises, and "knew it, but couldn't do
it at first" never happens. In designs B/C a separate output head existed,
but it buried the memory (finding 27).

Design D keeps the output going THROUGH the memory values (so memory keeps
its influence) but adds a per-word production skill:
    knowing  v_w = cosine(context, word) + connection table   (as design A)
    doing    logit_w = a_w * v_w,   a_w = softplus(p_w)
Every word starts with a low skill (a = A0 = 0.1: "barely says anything");
a word's skill grows only by being produced as an answer in practice. So a
new word may be known (high v) but not yet sayable (low a), while practised
words come out first -- like a toddler who understands a word long before
saying it and keeps reusing the familiar ones. The skill is part of doing,
not memory: the compare step (update / restore, judged on everything heard
so far) acts on the value tables only.

Compared with design A (a fixed at 1) under three judgement styles
(finding 30): persist, satisfied (stop once doing reaches the internal
goal), pre (skip practice if the internal prediction already knows >= 0.9).
Conditions: none / vec / hebb / vec+hebb / vec+bert.
Measured per session on the new heard partner items: goal (internal
prediction on the stored tables), knowing curve (v on the working tables),
doing curve (the actual output), the gap at the start and when doing
reaches the goal; effort; never-heard accuracy on everything after the
compare step; the internal logic's influence (vs none).
"""
import math
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_unified_continual import STAGES, V, sign_p
from run_seed_transfer import memorize, EPOCHS, CHECK_EVERY, REPLAY
from run_connection_seed import Hebbian, MiniBert, masked_items, partner_items, acc
from run_ratchet_seed import build_data, score_with
from run_three_designs import ValueOnly
from run_role2_target import predict

DESIGNS = ["A", "D"]
STYLES = ["persist", "satisfied", "pre"]
CONDITIONS = ["none", "vec", "hebb", "vec+hebb", "vec+bert"]
A0 = 0.1
PRE_ALREADY_KNOWN = 0.9


class KnowDo(ValueOnly):
    def __init__(self, start_vectors, seed):
        super().__init__(start_vectors, seed)
        self.skill = nn.Parameter(torch.full((V,), math.log(math.expm1(A0))))

    def know(self, ctx):
        return super().complete(ctx)

    def complete(self, ctx):
        return F.softplus(self.skill) * self.know(ctx)


def first_reach(curve, target):
    return next((i for i, v in enumerate(curve) if v >= target - 1e-9), float("nan"))


def run(design, style, condition, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = (KnowDo if design == "D" else ValueOnly)(start, seed)
    know = model.know if design == "D" else model.complete
    internal2 = Hebbian() if "hebb" in condition else MiniBert(memory, seed) if "bert" in condition else None
    stored = {"z_work": start.clone(), "assoc": torch.zeros_like(model.assoc)}
    rng = np.random.default_rng(20_000 + seed)
    sessions, seen_cats, seen_words = [], [], []

    for new_cats in STAGES:
        old_cats = list(seen_cats)
        seen_cats += new_cats
        new_words = [w for c in new_cats for w in split[c]["train"]]
        seen_words += new_words
        old_train = [w for c in old_cats for w in split[c]["train"]]
        replay_words = [int(w) for w in rng.choice(old_train, int(len(old_train) * REPLAY), replace=False)] if old_train else []
        new_sents = [s for c in new_cats for s in split[c]["sentences"]]
        old_sents = [s for c in old_cats for s in split[c]["sentences"]]
        replay_sents = [old_sents[i] for i in rng.choice(len(old_sents), int(len(old_sents) * REPLAY), replace=False)] if old_sents else []
        heard = new_sents + replay_sents

        if internal2 is not None:
            internal2.observe(heard)
            conn = internal2.connections(seen_words)
            nw = torch.tensor(new_words)
            stored["assoc"][nw, :] = conn[nw, :]
            stored["assoc"][:, nw] = conn[:, nw]
            with torch.no_grad():
                model.assoc[nw, :] = conn[nw, :]
                model.assoc[:, nw] = conn[:, nw]

        px, py = partner_items(new_sents, partner)
        with torch.no_grad():
            goal = acc(predict(stored["z_work"], stored["assoc"], px), py)
        budget = 0 if (style == "pre" and goal >= PRE_ALREADY_KNOWN) else EPOCHS

        train_ids = torch.tensor(new_words + replay_words)
        ctx, tgt, _, _ = masked_items(heard)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        know_curve, do_curve = [], []
        effort = 0
        for epoch in range(budget + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    know_curve.append(acc(know(px), py))
                    do_curve.append(acc(model.complete(px), py))
                if style == "satisfied" and goal > 0 and do_curve[-1] >= goal - 1e-9:
                    break
            if epoch == budget:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()
            effort += 1

        judge = masked_items([s for c in seen_cats for s in split[c]["sentences"]])[:2]
        for name in ["z_work", "assoc"]:
            param = getattr(model, name)
            working = score_with(model, name, param.detach().clone(), judge)
            kept = score_with(model, name, stored[name], judge)
            if working > kept:
                stored[name] = param.detach().clone()
            elif working < kept:
                with torch.no_grad():
                    param.copy_(stored[name])

        test = partner_items([s for c in seen_cats for s in split[c]["test_sents"]], partner)
        with torch.no_grad():
            sessions.append({
                "goal": goal, "effort": effort,
                "know0": know_curve[0], "do0": do_curve[0], "do10": do_curve[min(2, len(do_curve) - 1)],
                "know_end": know_curve[-1], "do_end": do_curve[-1],
                "reach": first_reach(do_curve, goal) if goal > 0 else float("nan"),
                "never_heard": acc(model.complete(test[0]), test[1]),
            })
    return sessions


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(d, st, c): run(d, st, c, data, memory, seed) for d in DESIGNS for st in STYLES for c in CONDITIONS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    print(f"=== {n_seeds} seeds, A0={A0} ===")
    print("\n--- persist: goal / knowing at start / doing at start / doing at 10 epochs / checks until doing reaches goal ---")
    for s in range(len(STAGES)):
        for d in DESIGNS:
            cells = []
            for c in CONDITIONS:
                x = [r[(d, "persist", c)][s] for r in results]
                cells.append(f"{c}: {m([y['goal'] for y in x])}/{m([y['know0'] for y in x])}/{m([y['do0'] for y in x])}/"
                             f"{m([y['do10'] for y in x])}/{m([y['reach'] for y in x])}")
            print(f"s{s+1} {d} " + " | ".join(cells))

    print("\n--- effort (epochs practised) and never-heard after the compare step, by style ---")
    for s in range(len(STAGES)):
        for d in DESIGNS:
            for st in STYLES:
                cells = [f"{c}: {m([r[(d, st, c)][s]['effort'] for r in results])}ep "
                         f"{m([r[(d, st, c)][s]['never_heard'] for r in results])}" for c in CONDITIONS]
                print(f"s{s+1} {d} {st:9} " + " | ".join(cells))

    print("\n--- internal logic's influence: condition vs none, never-heard after compare (mean, better/worse) ---")
    for d in DESIGNS:
        for st in STYLES:
            for c in CONDITIONS[1:]:
                cells = []
                for s in range(len(STAGES)):
                    diff = np.array([r[(d, st, c)][s]["never_heard"] - r[(d, st, "none")][s]["never_heard"] for r in results])
                    n_pos, n_neg = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
                    cells.append(f"s{s+1} {diff.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
                print(f"{d} {st:9} {c:9} " + "  ".join(cells))

    print("\n--- influence on the doing curve (persist): condition vs none, doing at 10 epochs ---")
    for d in DESIGNS:
        for c in CONDITIONS[1:]:
            cells = []
            for s in range(len(STAGES)):
                diff = np.array([r[(d, "persist", c)][s]["do10"] - r[(d, "persist", "none")][s]["do10"] for r in results])
                n_pos, n_neg = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
                cells.append(f"s{s+1} {diff.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
            print(f"{d} {c:9} " + "  ".join(cells))


if __name__ == "__main__":
    main()
