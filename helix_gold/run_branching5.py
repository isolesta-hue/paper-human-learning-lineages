"""Cutting losing branches early (Hyperband / successive halving), and incubating the ones cut (roadmap step 4,
2026-09-25; the user's proposals: "AI thinking falls into loops or pushes a wrong thought to the end -- our learning has
counts too", and "the brain keeps working on a problem it gave up, and the answer flashes up later").

Finding 51: with five knobs, trying every one-knob move each session ("full") was best (+0.073 over fixed settings), but
costs one whole practice session per move; choosing moves from the record did no better than random (too few labels).
  sh          -- successive halving (Jamieson & Talwalkar 2016; Hyperband, Li et al. 2018): every branch (the current
                 settings + every available move) practises a quarter of a session (50 epochs); the better half by
                 finding 50's judge practises on to half a session; the better half of those to a full session; the best
                 is chosen. Each branch keeps its own optimizer through its rungs (finding 47: restarting Adam when
                 learning is about to take off costs a lot). Cost: about half of "full".
  half        -- control at the same cost: the current settings + a random half of the moves, each practised fully.
  sh+incubate -- sh, and the best branch cut at the first rung is practised on to a full session "in the background"
                 (+3/4 of a session of compute). If it ends better than the chosen branch, that is a flash of insight:
                 its settings are adopted (checked by the same judge), and its move type is remembered as a late bloomer
                 -- from then on that move type skips the first cut (takes-off-late knowledge, finding 47). Wallas 1926
                 (incubation -> illumination -> verification); freeze-thaw (Swersky et al. 2014).
  full / fixed -- finding 51's references.
Platform as run_branching4.py (hard level, design D + both seeds, five knobs, 8 sessions x 2 categories + 2 cycles).
Measured: the outside test, and the practice epochs spent on trials (compute). Representative values: rungs 50/100/200
epochs, keep the better half (rounded up).
"""
import math
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import data_v2 as D
import lab2 as L
from run_unified_continual import sign_p
from run_society import imagine, PER_SESSION, CYCLES, EPOCHS, CYCLE_EPOCHS
from run_self_test import positions, score, HOLD
from run_branching4 import Learner2, KNOBS, START, MOVES, apply, EXPECT_RATE, FLOOR

RUNGS = [50, 100, 200]
CONDITIONS = ["fixed", "full", "half", "sh", "sh+incubate"]


class Trial:
    """one branch's trial session, run in stages with its own optimizer (same steps as Learner2.session)."""

    def __init__(self, learner, new_sents, old_by_cat, s, rng, weights):
        self.b, self.s = learner.clone(), s
        b = self.b
        if s["imag"] > 0 and len(b.known) > 5:
            b.practise([(imagine(b.model, sorted(b.known), b.known, rng), s["imag"])], s["lr"])
        new_words = {w for x in new_sents for w in x}
        b.known |= new_words
        cats = [c for c in old_by_cat if old_by_cat[c]]
        n_rev = int(round(sum(len(old_by_cat[c]) for c in cats) * s["review"]))
        review = []
        if cats and n_rev:
            w = np.array([(abs(weights.get(c, 0.0)) + FLOOR) if s["pick"] == "curious" else 1.0 for c in cats])
            pools = {c: list(rng.permutation(len(old_by_cat[c]))) for c in cats}
            for _ in range(n_rev):
                live = [j for j, c in enumerate(cats) if pools[c]]
                c = cats[live[int(rng.choice(len(live), p=w[live] / w[live].sum()))]]
                review.append(old_by_cat[c][pools[c].pop()])
        b.seed_connections(new_sents + review, new_words)
        phases = [(new_sents + review, EPOCHS)] if s["order"] == "mixed" else \
            [(new_sents, EPOCHS // 2), (new_sents + review, EPOCHS - EPOCHS // 2)]
        self.phases = []
        for sents, n in phases:
            ctx, tgt = L.items_tensor(positions(sents))
            ids = torch.tensor(sorted({x for x in sents for x in x}))
            self.phases.append((L.bag(ctx, b.d.V), tgt, ids, n))
        self.opt = torch.optim.Adam(b.model.parameters(), lr=s["lr"])
        self.done = 0

    def run_to(self, total):
        """practise on until `total` epochs of this session are done; returns the epochs spent."""
        spent, start = 0, 0
        for bag, tgt, ids, n in self.phases:
            for e in range(start, start + n):
                if self.done <= e < total:
                    self.opt.zero_grad()
                    loss = F.cross_entropy(self.b.model.complete(bag), tgt) + \
                        F.cross_entropy(self.b.model.classify(ids), self.b.cats[ids])
                    loss.backward()
                    self.opt.step()
                    spent += 1
            start += n
        self.done = max(self.done, min(total, EPOCHS))
        return spent


JUDGE_MODE = "mean"


def judge(trial, judged, new_cats=None):
    """finding 50's judge: mean over categories (old categories outnumber new ones as the life goes on). "balanced"
    (finding 68): new learning and keeping the old count half each."""
    if JUDGE_MODE == "balanced" and new_cats is not None:
        new = [score(trial.b.model, items) for c, items in judged.items() if c in new_cats]
        old = [score(trial.b.model, items) for c, items in judged.items() if c not in new_cats]
        if old:
            return 0.5 * float(np.mean(new)) + 0.5 * float(np.mean(old))
    return float(np.mean([score(trial.b.model, items) for items in judged.values()]))


def life(cond, d, seed, split=None):
    """split (finding 68, the user's "AI can think in parts"): besides the main (stable) learner, a plastic copy is forked
    from it at the start of every session and learns the session with the start settings (fast learning of the new);
    questions are routed: "age" -- categories new this session go to the plastic copy, the rest to the stable one;
    "judged" -- per bucket (new / old categories) the plastic copy is used while, on the trial's held-out sentences, it
    has beaten the chosen branch at least as often as it lost (finding 61's stakes rule)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(60_000 + seed)
    sents, aside = {}, {}
    for c, cat in enumerate(d.categories):
        ss = [x for _, x, _ in cat["heard"]]
        idx = rng.permutation(len(ss))
        n_hold = int(round(len(ss) * HOLD))
        aside[c] = [ss[i] for i in idx[:n_hold]]
        sents[c] = [ss[i] for i in idx[n_hold:]] + aside[c]
    hidden = {c: [(ctx, t) for _, ctx, t in D.unit_items(d.categories[c]["test"])] for c in range(d.n_cat)}
    order = [int(c) for c in rng.permutation(d.n_cat)]
    schedule = [("session", order[i:i + PER_SESSION]) for i in range(0, d.n_cat, PER_SESSION)] + [("cycle", [])] * CYCLES
    me = Learner2(d, seed)
    settings = dict(START)
    seen_cats, steps = [], []
    acc_now, expect, rpe = {}, {}, {}
    late = set()                                 # move types that bloomed late (sh+incubate)
    wins = {"new": [0, 0], "old": [0, 0]}         # split="judged": plastic copy beat / lost to the chosen branch
    plastic = None

    for i, (kind, new_cats) in enumerate(schedule):
        old_cats = list(seen_cats)
        seen_cats += new_cats
        cost, flash, cut_early = 0, 0, 0
        if kind == "session":
            old_by_cat = {c: sents[c] for c in old_cats}
            if cond != "fixed" and i > 0:
                trial_new = [x for c in new_cats for x in sents[c][:len(sents[c]) - len(aside[c])]]
                judged = {c: positions(aside[c]) for c in new_cats}
                trial_old = {}
                for c in old_cats:
                    idx = rng.permutation(len(sents[c]))
                    k = int(round(len(sents[c]) * HOLD))
                    judged[c] = positions([sents[c][j] for j in idx[:k]])
                    trial_old[c] = [sents[c][j] for j in idx[k:]]
                available = [m for m in MOVES if apply(settings, m) is not None]
                if cond == "half":
                    available = [available[j] for j in rng.permutation(len(available))[:math.ceil((len(available) + 1) / 2) - 1]]
                branches = [(m, settings if m is None else apply(settings, m)) for m in [None] + available]
                trials = [(m, s, Trial(me, trial_new, trial_old, s, np.random.default_rng(rng.integers(1 << 31)), rpe))
                          for m, s in branches]
                if cond in ("full", "half"):
                    for m, s, t in trials:
                        cost += t.run_to(EPOCHS)
                    best = max(trials, key=lambda x: judge(x[2], judged, new_cats))
                else:
                    alive = trials
                    first_cut = []
                    for r, rung in enumerate(RUNGS):
                        for m, s, t in alive:
                            cost += t.run_to(rung)
                        if rung == RUNGS[-1]:
                            break
                        ranked = sorted(alive, key=lambda x: -judge(x[2], judged, new_cats))
                        keep = ranked[:math.ceil(len(ranked) / 2)]
                        if r == 0 and cond == "sh+incubate":  # late bloomers skip the first cut
                            keep += [x for x in ranked[len(keep):] if x[0] in late]
                            first_cut = [x for x in ranked if x not in keep]
                            cut_early = len(first_cut)
                        alive = keep
                    best = max(alive, key=lambda x: judge(x[2], judged, new_cats))
                    if cond == "sh+incubate" and first_cut:
                        m, s, t = first_cut[0]            # the best of those cut first, practised on in the background
                        cost += t.run_to(EPOCHS)
                        if judge(t, judged, new_cats) > judge(best[2], judged, new_cats):
                            flash = 1                     # insight: adopted after the same check, remembered as late bloomer
                            best = (m, s, t)
                            if m is not None:
                                late.add(m)
                if split == "judged":
                    pt = Trial(me, trial_new, trial_old, dict(START), np.random.default_rng(rng.integers(1 << 31)), rpe)
                    cost += pt.run_to(EPOCHS)
                    for bucket, cs in (("new", [c for c in judged if c in new_cats]), ("old", [c for c in judged if c not in new_cats])):
                        if cs:
                            a = np.mean([score(pt.b.model, judged[c]) for c in cs])
                            b_ = np.mean([score(best[2].b.model, judged[c]) for c in cs])
                            wins[bucket][0] += a > b_
                            wins[bucket][1] += a < b_
                settings = dict(best[1])
            if split:
                plastic = me.clone()
                plastic.session([x for c in new_cats for x in sents[c]], old_by_cat, dict(START),
                                np.random.default_rng(rng.integers(1 << 31)), rpe)
            me.session([x for c in new_cats for x in sents[c]], old_by_cat, settings, rng, rpe)
        else:
            plastic = None
            if settings["imag"] > 0:
                me.practise([(imagine(me.model, sorted(me.known), me.known, rng), settings["imag"])], settings["lr"])
            allsents = [x for c in seen_cats for x in sents[c]]
            me.hebb.observe(allsents)
            me.practise([(allsents, CYCLE_EPOCHS)], settings["lr"])

        for c in seen_cats:
            a = score(me.model, positions(sents[c]))
            if c in acc_now:
                lp = a - acc_now[c]
                rpe[c] = lp - expect.get(c, 0.0)
                expect[c] = expect.get(c, 0.0) + EXPECT_RATE * (lp - expect.get(c, 0.0))
            acc_now[c] = a
        def route(c):
            if plastic is None:
                return me
            bucket = "new" if c in new_cats else "old"
            if split == "age":
                return plastic if bucket == "new" else me
            return plastic if (wins[bucket][0] >= wins[bucket][1] and sum(wins[bucket]) > 0) else me
        per = {c: score(route(c).model, hidden[c]) for c in seen_cats}
        n_items = {c: len(hidden[c]) for c in seen_cats}
        steps.append({"never-heard": sum(per[c] * n_items[c] for c in seen_cats) / sum(n_items.values()),
                      "older": float(np.mean([per[c] for c in old_cats])) if (old_cats and kind == "session") else np.nan,
                      "plastic new": float(wins["new"][0] >= wins["new"][1] and sum(wins["new"]) > 0) if split == "judged" else np.nan,
                      "plastic old": float(wins["old"][0] >= wins["old"][1] and sum(wins["old"]) > 0) if split == "judged" else np.nan,
                      "newest": float(np.mean([per[c] for c in new_cats])) if new_cats else np.nan,
                      "review": settings["review"], "lr": settings["lr"], "imag": settings["imag"],
                      "trial epochs": cost if kind == "session" and i > 0 and cond != "fixed" else np.nan,
                      "flash": flash if cond == "sh+incubate" and kind == "session" and i > 0 else np.nan,
                      "late": len(late) if cond == "sh+incubate" else np.nan})
    return steps


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_b5")


def one_seed(args):
    seed, conds = args
    path = os.path.join(CACHE, f"seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    torch.set_num_threads(1)
    d = D.build("hard", seed)
    out = {c: life(c, d, seed) for c in conds}
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    conds = CONDITIONS
    os.makedirs(CACHE, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(one_seed, [(s, conds) for s in range(n_seeds)])
    n_steps = len(results[0][conds[0]])
    n_sess = n_steps - CYCLES

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    print(f"=== {n_seeds} seeds; hard level; five knobs; rungs {RUNGS} ===")
    for key in ["never-heard", "newest", "review", "lr", "imag", "trial epochs", "flash", "late"]:
        print(f"\n--- {key} ---")
        for c in conds:
            print(f"{c:12} " + " ".join(f"{m([r[c][i][key] for r in results]):>7}" for i in range(n_steps)))
    print("\n--- trial epochs per life (compute) ---")
    for c in conds:
        print(f"{c:12} {np.mean([np.nansum([st['trial epochs'] for st in r[c]]) for r in results]):.0f}")
    print("\n--- paired ---")
    for g, base in [("full", "fixed"), ("sh", "fixed"), ("sh", "full"), ("sh", "half"), ("half", "fixed"),
                    ("sh+incubate", "sh"), ("sh+incubate", "full")]:
        cells = []
        for label, get in [("life", lambda r, gg: np.mean([st["never-heard"] for st in r[gg]])),
                           ("end", lambda r, gg: r[gg][-1]["never-heard"]),
                           ("newest", lambda r, gg: np.mean([st["newest"] for st in r[gg][1:n_sess]]))]:
            dd = np.array([get(r, g) - get(r, base) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{label} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{g:12} vs {base:12} " + "  ".join(cells))


if __name__ == "__main__":
    main()
