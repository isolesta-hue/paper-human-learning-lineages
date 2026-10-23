"""Learning WHERE to look: which knob moves pay off (roadmap step 4, part 2, 2026-09-23).

User's proposal: use the categorising ability to learn which things are worth changing, so the learner no
longer has to try every branch. Finding 50 gave a judge that agrees with the outside test (cross-validation
inside the trial only); its verdicts are the labels here.
  body     -- five knobs (the developer's outer layer); two added to findings 48-50's three so that trying
              every one-knob move becomes expensive:
                review share   {0.1, 0.25, 0.5, 0.75}      start 0.25
                learning rate  {0.003, 0.01, 0.03}         start 0.01
                order          {mixed, new-first}          start mixed
                imagination    {0, 25, 50, 100} epochs     start 50   (finding 36; practised at the start of the step)
                review picking {uniform, curious}          start uniform (curious = older categories drawn by
                                                           |surprise| + 0.01 of their own accuracy, finding 41)
  moves    -- one knob one step up or down (8 move types: review+/-, lr+/-, order flip, imagination+/-, picking flip)
  budget   -- 3 trials per session: the current settings + 2 moves
  labels   -- for every tried move: gain = judge(move) - judge(current) (finding 50's judge); effective if > 0
  choosing the 2 moves
    random   -- 2 of the available moves at random
    guided   -- a record per move type of its mean gain so far: untried move types first, otherwise the two
                with the highest mean gain; each slot is replaced by a random move with probability 0.2 so
                every move type keeps getting labels
    full     -- every available move (no budget) -- reference
    fixed    -- start settings throughout
The sign of learning where to look: the share of tried moves that win, session by session -- it should rise
for guided and stay flat for random.
Platform as run_branching3.py (hard level, design D + both seeds, 8 sessions x 2 categories + 2 cycles).
Representative values: budget 3, exploration 0.2, surprise rate 0.5, floor 0.01.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch

import data_v2 as D
from run_unified_continual import sign_p
from run_society import imagine, PER_SESSION, CYCLES, EPOCHS, CYCLE_EPOCHS
from run_self_test import positions, score, HOLD
from run_branching import Learner

KNOBS = {"review": [0.1, 0.25, 0.5, 0.75], "lr": [0.003, 0.01, 0.03], "order": ["mixed", "new-first"],
         "imag": [0, 25, 50, 100], "pick": ["uniform", "curious"]}
START = {"review": 0.25, "lr": 0.01, "order": "mixed", "imag": 50, "pick": "uniform"}
MOVES = ["review+", "review-", "lr+", "lr-", "order", "imag+", "imag-", "pick"]
BUDGET, EXPLORE, EXPECT_RATE, FLOOR = 3, 0.2, 0.5, 0.01
CONDITIONS = ["fixed", "random", "guided", "full"]


def apply(s, move):
    """settings after a move, or None if the move is not possible."""
    k, sign = (move[:-1], move[-1]) if move[-1] in "+-" else (move, None)
    vals = KNOBS[k]
    i = vals.index(s[k])
    j = (i + 1) % len(vals) if sign is None else i + (1 if sign == "+" else -1)
    if not 0 <= j < len(vals):
        return None
    t = dict(s)
    t[k] = vals[j]
    return t


class Learner2(Learner):
    def session(self, new_sents, old_by_cat, s, rng, weights):
        """imagination (s['imag'] epochs), then the session with settings s; old_by_cat: category -> sentences."""
        if s["imag"] > 0 and len(self.known) > 5:
            self.practise([(imagine(self.model, sorted(self.known), self.known, rng), s["imag"])], s["lr"])
        new_words = {w for x in new_sents for w in x}
        self.known |= new_words
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
        self.seed_connections(new_sents + review, new_words)
        if s["order"] == "mixed":
            self.practise([(new_sents + review, EPOCHS)], s["lr"])
        else:
            self.practise([(new_sents, EPOCHS // 2), (new_sents + review, EPOCHS - EPOCHS // 2)], s["lr"])


def choose_moves(cond, available, record, rng):
    if cond == "full":
        return list(available)
    if cond == "random":
        return [available[j] for j in rng.permutation(len(available))[:BUDGET - 1]]
    untried = [m for m in available if not record.get(m)]
    ranked = [m for m in rng.permutation(untried)] + sorted(
        [m for m in available if record.get(m)], key=lambda m: -np.mean(record[m]))
    picks = []
    for m in ranked:
        if len(picks) == BUDGET - 1:
            break
        if m not in picks:
            picks.append(m)
    for k in range(len(picks)):
        if rng.random() < EXPLORE:
            rest = [m for m in available if m not in picks]
            if rest:
                picks[k] = rest[int(rng.integers(len(rest)))]
    return picks


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
    me = Learner2(d, seed)
    settings = dict(START)
    seen_cats, steps = [], []
    record = {}                                  # move type -> list of gains (the learner's own labels)
    acc_now, expect, rpe = {}, {}, {}

    for i, (kind, new_cats) in enumerate(schedule):
        old_cats = list(seen_cats)
        seen_cats += new_cats
        tried = won = 0
        tried_moves = []
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
                moves = choose_moves(cond, available, record, rng)
                results = []
                for m in [None] + moves:
                    s = settings if m is None else apply(settings, m)
                    b = me.clone()
                    b.session(trial_new, trial_old, s, np.random.default_rng(rng.integers(1 << 31)), rpe)
                    results.append((m, s, float(np.mean([score(b.model, items) for items in judged.values()]))))
                base = results[0][2]
                for m, s, own in results[1:]:
                    record.setdefault(m, []).append(own - base)
                    tried += 1
                    won += own > base
                    tried_moves.append(m)
                settings = dict(max(results, key=lambda t: t[2])[1])
            me.session([x for c in new_cats for x in sents[c]], old_by_cat, settings, rng, rpe)
        else:
            if settings["imag"] > 0:
                me.practise([(imagine(me.model, sorted(me.known), me.known, rng), settings["imag"])], settings["lr"])
            allsents = [x for c in seen_cats for x in sents[c]]
            me.hebb.observe(allsents)
            me.practise([(allsents, CYCLE_EPOCHS)], settings["lr"])

        for c in seen_cats:                      # surprise signal from the learner's own heard sentences (finding 41)
            a = score(me.model, positions(sents[c]))
            if c in acc_now:
                lp = a - acc_now[c]
                rpe[c] = lp - expect.get(c, 0.0)
                expect[c] = expect.get(c, 0.0) + EXPECT_RATE * (lp - expect.get(c, 0.0))
            acc_now[c] = a
        per = {c: score(me.model, hidden[c]) for c in seen_cats}
        steps.append({"never-heard": score(me.model, [it for c in seen_cats for it in hidden[c]]),
                      "newest": float(np.mean([per[c] for c in new_cats])) if new_cats else np.nan,
                      "older": float(np.mean([per[c] for c in old_cats])) if (old_cats and kind == "session") else np.nan,
                      "review": settings["review"], "lr": settings["lr"], "new-first": float(settings["order"] == "new-first"),
                      "imag": settings["imag"], "curious": float(settings["pick"] == "curious"),
                      "tried": tried if tried else np.nan, "win share": won / tried if tried else np.nan,
                      "moves": tried_moves})
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

    print(f"=== {n_seeds} seeds; hard level, design D + both seeds; knobs {KNOBS}; start {START}; budget {BUDGET}, "
          f"explore {EXPLORE}; columns: " + " ".join(labels) + " ===")
    for key in ["never-heard", "newest", "older", "review", "lr", "new-first", "imag", "curious", "tried", "win share"]:
        print(f"\n--- {key} ---")
        for c in conds:
            print(f"{c:8} " + " ".join(f"{m([r[c][i][key] for r in results]):>6}" for i in range(n_steps)))

    print("\n--- which moves were tried (share of tries) and how often each won, sessions 2-8 ---")
    for c in conds:
        if c == "fixed":
            continue
        cnt = {mv: 0 for mv in MOVES}
        for r in results:
            for st in r[c][1:n_sess]:
                for mv in st["moves"]:
                    cnt[mv] += 1
        total = max(1, sum(cnt.values()))
        print(f"{c:8} " + "  ".join(f"{mv} {cnt[mv] / total:.2f}" for mv in MOVES))

    print("\n--- win share early (sessions 2-4) vs late (sessions 6-8) ---")
    for c in conds:
        if c == "fixed":
            continue
        early = np.array([np.nanmean([st["win share"] for st in r[c][1:4]]) for r in results])
        late = np.array([np.nanmean([st["win share"] for st in r[c][5:8]]) for r in results])
        dd = late - early
        n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
        print(f"{c:8} early {early.mean():.3f} late {late.mean():.3f} change {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")

    print("\n--- paired: never-heard (life mean / end / newest in sessions 2-8 / older in sessions 2-8) ---")
    for g, base in [("random", "fixed"), ("guided", "fixed"), ("full", "fixed"), ("guided", "random"), ("full", "guided")]:
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
        print(f"{g:8} vs {base:7} " + "  ".join(cells))


if __name__ == "__main__":
    main()
