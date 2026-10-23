"""Branching with a judge that is cross-validated inside the trial only (finding 49 follow-up, 2026-09-23).

Finding 49: judging on never-practised sentences fixed the direction of the choices but was noisy (few
sentences) and holding sentences back from the real session was costly (-0.056). The fluency bias comes from
judging on sentences practised IN THAT SESSION (a high learning rate only re-fits them faster). So:
  trial      -- while the branches are tried, a fresh random quarter of the older categories' sentences is
                taken out of the session's review material and used, with the new categories' set-aside
                quarter, as the judge (a new quarter every session: cross-validation inside the trial)
  commit     -- the winning settings live the session for real on ALL sentences (nothing is held back)
The older sentences were practised in earlier sessions; that familiarity is the same in every branch, so it
does not tilt the comparison. Every category counts equally in the judge.
Conditions: fixed / branch:self (finding 48) / branch:self:cv (this judge). Body, branches, commit and platform
as run_branching.py.
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

CONDITIONS = ["fixed", "branch:self", "branch:self:cv"]


def life(cond, d, seed):
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
    seen_cats, steps = [], []

    for i, (kind, new_cats) in enumerate(schedule):
        if i > 0:
            me.practise([(imagine(me.model, sorted(me.known), me.known, rng), PHASE_EPOCHS)], settings["lr"])
        old_cats = list(seen_cats)
        seen_cats += new_cats
        picked_best = np.nan
        if kind == "session":
            old_sents = [x for c in old_cats for x in sents[c]]
            if cond.startswith("branch"):
                trial_new = [x for c in new_cats for x in sents[c][:len(sents[c]) - len(aside[c])]]
                judged = {c: positions(aside[c]) for c in new_cats}
                trial_old = old_sents
                if cond == "branch:self:cv":
                    trial_old = []
                    for c in old_cats:
                        idx = rng.permutation(len(sents[c]))
                        k = int(round(len(sents[c]) * HOLD))
                        judged[c] = positions([sents[c][j] for j in idx[:k]])
                        trial_old += [sents[c][j] for j in idx[k:]]
                else:
                    judged.update({c: positions(sents[c]) for c in old_cats})
                outside_items = [it for c in seen_cats for it in hidden[c]]
                trials = []
                for s in neighbours(settings):
                    b = me.clone()
                    b.session(trial_new, trial_old, s, np.random.default_rng(rng.integers(1 << 31)))
                    own = float(np.mean([score(b.model, items) for items in judged.values()]))
                    out = score(b.model, outside_items)           # developer diagnostic only
                    trials.append((s, own, out))
                best = max(trials, key=lambda t: t[1])
                picked_best = float(best[2] >= max(t[2] for t in trials) - 1e-9)
                settings = dict(best[0])
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
            print(f"{c:15} " + " ".join(f"{m([r[c][i][key] for r in results]):>6}" for i in range(n_steps)))

    print("\n--- paired: never-heard (life mean / end / newest in sessions 2-8 / older in sessions 2-8) ---")
    for g, base in [("branch:self", "fixed"), ("branch:self:cv", "fixed"), ("branch:self:cv", "branch:self")]:
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
        print(f"{g:15} vs {base:12} " + "  ".join(cells))


if __name__ == "__main__":
    main()
