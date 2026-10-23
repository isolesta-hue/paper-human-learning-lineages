"""Branching with a less fluency-biased judge (finding 48 follow-up, 2026-09-23).

Finding 48: the self-evaluation judged older categories on sentences already
practised, so it took fluency for skill and pushed the learning rate up while
the outside test wanted it down (-0.020 against choosing by the outside test).
Fix tried here -- give the set-aside quarter back LATER: a new category's
quarter stays unpractised (not practised, not observed by the connection seed)
for DELAY more sessions after its own, so while it is a recent older category
it can still be tested on sentences the learner has never practised. The new
judge uses ONLY such never-practised sentences: the new categories' quarter and
the quarters still held by recent older categories (older categories whose
quarter has been given back are not judged). All held quarters are given back
before the everyday cycles.

Giving back later changes two things at once -- the judge AND less practice
material for a while -- so a control keeps the late return with the old judge:
  fixed               -- start settings throughout (quarters given back as in finding 48)
  branch:self         -- finding 48 as it was (quarter held only during its own session, old judge)
  branch:self:d1:old  -- quarter held one more session, old judge (practised sentences for older categories)
  branch:self:d1      -- quarter held one more session, new judge (never-practised sentences only)
judge effect = d1 vs d1:old (same material); holding cost = d1:old vs branch:self (same judge).
Body, branches, commit and platform exactly as run_branching.py.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch

import data_v2 as D
from run_unified_continual import sign_p
from run_society import imagine, PER_SESSION, CYCLES, CYCLE_EPOCHS, PHASE_EPOCHS
from run_self_test import positions, score, HOLD
from run_branching import Learner, neighbours, KNOBS, START

CONDITIONS = ["fixed", "branch:self", "branch:self:d1:old", "branch:self:d1"]


def life(cond, d, seed):
    delay = 1 if ":d1" in cond else 0
    new_judge = cond == "branch:self:d1"
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
    me = Learner(d, seed)
    settings = dict(START)
    seen_cats, steps, held, cat_step = [], [], set(), {}

    def practised(c):
        return sents[c][:len(sents[c]) - len(aside[c])] if c in held else sents[c]

    for i, (kind, new_cats) in enumerate(schedule):
        # give back quarters whose holding time is over (all of them before the everyday cycles)
        back = [c for c in sorted(held) if kind == "cycle" or i - cat_step[c] > delay]
        if back:
            returned = [x for c in back for x in aside[c]]
            held -= set(back)
            words = {w for x in returned for w in x}
            me.known |= words
            me.seed_connections(returned, words)
        if i > 0:
            me.practise([(imagine(me.model, sorted(me.known), me.known, rng), PHASE_EPOCHS)], settings["lr"])
        old_cats = list(seen_cats)
        seen_cats += new_cats
        picked_best = np.nan
        if kind == "session":
            for c in new_cats:
                cat_step[c] = i
            old_sents = [x for c in old_cats for x in practised(c)]
            trial_new = [x for c in new_cats for x in sents[c][:len(sents[c]) - len(aside[c])]]
            if cond.startswith("branch"):
                if new_judge:
                    judged = {c: positions(aside[c]) for c in new_cats}
                    judged.update({c: positions(aside[c]) for c in old_cats if c in held})
                else:
                    judged = {c: positions(aside[c]) for c in new_cats}
                    judged.update({c: positions(practised(c)) for c in old_cats})
                outside_items = [it for c in seen_cats for it in hidden[c]]
                trials = []
                for s in neighbours(settings):
                    b = me.clone()
                    b.session(trial_new, old_sents, s, np.random.default_rng(rng.integers(1 << 31)))
                    own = float(np.mean([score(b.model, items) for items in judged.values()]))
                    out = score(b.model, outside_items)           # developer diagnostic only
                    trials.append((s, own, out))
                best = max(trials, key=lambda t: t[1])
                picked_best = float(best[2] >= max(t[2] for t in trials) - 1e-9)
                settings = dict(best[0])
            if delay > 0:
                held |= set(new_cats)
                me.session(trial_new, old_sents, settings, rng)
            else:
                me.session([x for c in new_cats for x in sents[c]], old_sents, settings, rng)
        else:
            allsents = [x for c in seen_cats for x in sents[c]]
            me.hebb.observe(allsents)
            me.practise([(allsents, CYCLE_EPOCHS)], settings["lr"])

        per = {c: score(me.model, hidden[c]) for c in seen_cats}
        steps.append({"never-heard": score(me.model, [it for c in seen_cats for it in hidden[c]]),
                      "newest": float(np.mean([per[c] for c in new_cats])) if new_cats else np.nan,
                      "older": float(np.mean([per[c] for c in old_cats])) if (old_cats and kind == "session") else np.nan,
                      "review": settings["review"], "lr": settings["lr"],
                      "new-first": float(settings["order"] == "new-first"), "picked best": picked_best})
    return steps


def one_seed(args):
    seed, conds = args
    torch.set_num_threads(1)
    d = D.build("hard", seed)
    return {c: life(c, d, seed) for c in conds}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    conds = sys.argv[3].split(",") if len(sys.argv) > 3 else CONDITIONS
    with Pool(workers) as pool:
        results = pool.map(one_seed, [(s, conds) for s in range(n_seeds)])
    n_steps = len(results[0][conds[0]])
    n_sess = n_steps - CYCLES
    labels = [f"s{i+1}" if i < n_sess else f"c{i - n_sess + 1}" for i in range(n_steps)]

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    print(f"=== {n_seeds} seeds; hard level, design D + both seeds; knobs {KNOBS}; start {START}; "
          f"columns: " + " ".join(labels) + " ===")
    for key in ["never-heard", "newest", "older", "review", "lr", "new-first", "picked best"]:
        print(f"\n--- {key} ---")
        for c in conds:
            print(f"{c:19} " + " ".join(f"{m([r[c][i][key] for r in results]):>6}" for i in range(n_steps)))

    print("\n--- paired: never-heard (life mean / end / newest in sessions 2-8 / older in sessions 2-8) ---")
    for g, base in [("branch:self", "fixed"), ("branch:self:d1:old", "branch:self"),
                    ("branch:self:d1", "branch:self:d1:old"), ("branch:self:d1", "branch:self"),
                    ("branch:self:d1", "fixed")]:
        if g not in conds or base not in conds:
            continue
        cells = []
        for label, get in [("life", lambda r, gg: np.mean([st["never-heard"] for st in r[gg]])),
                           ("end", lambda r, gg: r[gg][-1]["never-heard"]),
                           ("newest", lambda r, gg: np.mean([st["newest"] for st in r[gg][1:n_sess]])),
                           ("older", lambda r, gg: np.mean([st["older"] for st in r[gg][1:n_sess]]))]:
            dd = np.array([get(r, g) - get(r, base) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{label} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{g:19} vs {base:19} " + "  ".join(cells))


if __name__ == "__main__":
    main()
