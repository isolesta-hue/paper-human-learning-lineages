"""Branching: the learner tries its own ways of learning in parallel copies and
keeps the one its self-evaluation prefers (roadmap step 4, 2026-09-23).

Findings 46-47: the large gains on this platform came from HOW to learn (more
review of older sentences; new first, then new + old mixed) and every one of
them was found by the developer's control conditions, not by the system. The
branching design agreed with the user:
  body         -- a list of knobs the developer gives (outer layer), like limbs:
                    review share  {0.1, 0.25, 0.5, 0.75}   (start 0.25)
                    learning rate {0.003, 0.01, 0.03}      (start 0.01)
                    order         {mixed, new-first}       (start mixed)
                  mixed     = 200 epochs on the new sentences + review share x older sentences
                  new-first = 100 epochs on the new sentences only, then 100 on new + review share x older
                  (one Adam through the whole session: no restart inside a session, finding 47)
  choice point -- the start of every session (fixed for now)
  branches     -- the current settings, and every setting with ONE knob moved one step (others unchanged,
                  so each comparison is fair by construction); each branch is a copy of the learner that
                  lives through the whole session
  judge        -- self-evaluation from the learner's own experience only: new categories on a quarter of
                  their sentences set aside during the trial; older categories on how well their heard
                  sentences are still completed (forgetting); every category counts equally
  commit       -- the winning settings are then used to live the session for real, from the state before
                  branching, on ALL sentences (the set-aside quarter is given back)
Conditions
  fixed           -- the start settings throughout
  fixed:found     -- the settings the developer's controls found (review 0.5, new-first) -- reference
  branch:self     -- branching judged by the self-evaluation
  branch:outside  -- branching judged by the hidden outside test -- DEVELOPER REFERENCE ONLY (the system can
                     never see this); the ceiling of what one-knob branching could reach
Recorded: settings chosen at every session; in branch:self, whether the self-evaluation picked the branch that
was best on the outside test (developer diagnostic, never used by the learner).
Platform: hard level, design D + vector seed + directional connection seed, 8 sessions x 2 categories + 2
everyday cycles, imagination slot 50 epochs before every step (as run_self_test2.py).
"""
import copy
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import data_v2 as D
import lab2 as L
from run_seed_transfer import memorize
from run_unified_continual import sign_p
from run_society import imagine, PER_SESSION, CYCLES, EPOCHS, CYCLE_EPOCHS, PHASE_EPOCHS
from run_self_test import positions, score, HOLD

KNOBS = {"review": [0.1, 0.25, 0.5, 0.75], "lr": [0.003, 0.01, 0.03], "order": ["mixed", "new-first"]}
START = {"review": 0.25, "lr": 0.01, "order": "mixed"}
FOUND = {"review": 0.5, "lr": 0.01, "order": "new-first"}
CONDITIONS = ["fixed", "fixed:found", "branch:self", "branch:outside"]


def neighbours(s):
    out = [dict(s)]
    for k, vals in KNOBS.items():
        i = vals.index(s[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                t = dict(s)
                t[k] = vals[j]
                out.append(t)
    return out


class Learner:
    def __init__(self, d, seed):
        self.d = d
        self.cats = torch.tensor(d.cats)
        memory, _ = memorize(torch.tensor(d.vecs), seed)
        self.model = L.KnowDo2(d.V, d.n_cat, memory, seed)
        self.hebb = L.make_internal2("dhebb", d.V)
        self.known = set()

    def clone(self):
        c = copy.copy(self)
        c.model = copy.deepcopy(self.model)
        c.hebb = copy.deepcopy(self.hebb)
        c.known = set(self.known)
        return c

    def practise(self, phases, lr):
        """phases: list of (sentences, epochs); one Adam through all phases."""
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for sents, epochs in phases:
            if epochs <= 0 or not sents:
                continue
            ctx, tgt = L.items_tensor(positions(sents))
            ctx = L.bag(ctx, self.d.V)
            ids = torch.tensor(sorted({x for s in sents for x in s}))
            for _ in range(epochs):
                opt.zero_grad()
                loss = F.cross_entropy(self.model.complete(ctx), tgt) + F.cross_entropy(self.model.classify(ids), self.cats[ids])
                loss.backward()
                opt.step()

    def seed_connections(self, observed, words):
        self.hebb.observe(observed)
        conn = self.hebb.connections(sorted(self.known))
        nw = torch.tensor(sorted(words))
        with torch.no_grad():
            self.model.assoc[nw, :] = conn[nw, :]
            self.model.assoc[:, nw] = conn[:, nw]

    def session(self, new_sents, old_sents, s, rng):
        """live one session with settings s on new_sents (list) + older sentences."""
        new_words = {w for x in new_sents for w in x}
        self.known |= new_words
        n_rev = int(round(len(old_sents) * s["review"]))
        review = [old_sents[j] for j in rng.choice(len(old_sents), n_rev, replace=False)] if (old_sents and n_rev) else []
        self.seed_connections(new_sents + review, new_words)
        if s["order"] == "mixed":
            self.practise([(new_sents + review, EPOCHS)], s["lr"])
        else:
            self.practise([(new_sents, EPOCHS // 2), (new_sents + review, EPOCHS - EPOCHS // 2)], s["lr"])


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
    settings = dict(FOUND if cond == "fixed:found" else START)
    seen_cats, steps = [], []

    for i, (kind, new_cats) in enumerate(schedule):
        if i > 0:
            sents_im = imagine(me.model, sorted(me.known), me.known, rng)
            me.practise([(sents_im, PHASE_EPOCHS)], settings["lr"])
        old_cats = list(seen_cats)
        seen_cats += new_cats
        picked_best = np.nan
        if kind == "session":
            old_sents = [x for c in old_cats for x in sents[c]]
            if cond.startswith("branch"):
                trial_new = [x for c in new_cats for x in sents[c][:len(sents[c]) - len(aside[c])]]
                self_items = {c: positions(aside[c]) for c in new_cats}
                self_items.update({c: positions(sents[c]) for c in old_cats})
                outside_items = [it for c in seen_cats for it in hidden[c]]
                trials = []
                for s in neighbours(settings):
                    b = me.clone()
                    b.session(trial_new, old_sents, s, np.random.default_rng(rng.integers(1 << 31)))
                    own = float(np.mean([score(b.model, self_items[c]) for c in seen_cats]))
                    out = score(b.model, outside_items)           # developer diagnostic / oracle only
                    trials.append((s, own, out))
                key = 1 if cond == "branch:self" else 2
                best = max(trials, key=lambda t: t[key])
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
    for g, base in [("branch:self", "fixed"), ("fixed:found", "fixed"), ("branch:self", "fixed:found"),
                    ("branch:outside", "fixed"), ("branch:outside", "branch:self")]:
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
