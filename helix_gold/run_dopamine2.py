"""Signals, second attempt: set points against habituation, and dopamine on a
strong knob (what to review).

Finding 40: (1) punishment judged against the EXPECTED change habituated --
steady drift became "expected" and stopped triggering it; (2) the knob
dopamine moved (the exploration rate of imagination) barely changed results.
Here:
  set point  -- a category is restored (finding 39) only when its accuracy on
                its heard sentences is DELTA below the best it has ever
                reached. The reference does not move with the drift, like a
                body temperature set point.
  dopamine on review -- the replay of older sentences keeps its size (25%)
                but WHICH older categories are replayed is no longer random:
                  repair  -- in proportion to how far each category is below
                             its set point (deficit)
                  curious -- in proportion to how surprising its latest
                             change was (|RPE|, finding 40's signal)
                (a small floor keeps every category possible).
All signals come only from real experience (accuracy on heard sentences
measured after each step), never from imagined sentences.
Regimes (same life and platform as run_dopamine.py; imagination EPS 0.3):
  fixed             -- random replay, restore (finding 39)
  setpoint          -- random replay, restore only below the set point
  norestore         -- random replay, no restore
  norestore-repair  -- repair replay, no restore
  norestore-curious -- curious replay, no restore
(A first version also crossed repair/curious replay with restoring. It came
out identical to "fixed": restoring already brings every category back to
its best level after each step, so there is no deficit left to steer the
replay by. The "restore" of finding 39 itself only acts when some past
state did better -- i.e. below the best level reached -- so it already is a
set-point rule; see finding 41.)
Conditions: none / vec / hebb.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from run_unified_continual import CATEGORIES, sign_p
from run_seed_transfer import memorize, EPOCHS, REPLAY
from run_connection_seed import masked_items, partner_items, acc
from run_ratchet_seed import build_data, score_with
from run_three_designs import ValueOnly
from run_role1_anchor import make_internal2
from run_explore_imagine import imagine
from run_role3_imagine import PHASE_EPOCHS
from run_dopamine import restore_category, CYCLES, CYCLE_EPOCHS, EPS0, DELTA, EXPECT_RATE

CONDITIONS = ["none", "vec", "hebb"]
REGIMES = ["fixed", "setpoint", "norestore", "norestore-repair", "norestore-curious"]
FLOOR = 0.01


def pick_replay(rng, split, old_cats, weights, n):
    """n older sentences; categories drawn by weight, sentences without replacement inside a category."""
    pools = {c: list(rng.permutation(len(split[c]["sentences"]))) for c in old_cats}
    w = np.array([weights[c] for c in old_cats], dtype=float)
    out = []
    for _ in range(n):
        live = [k for k, c in enumerate(old_cats) if pools[c]]
        p = w[live] / w[live].sum()
        c = old_cats[live[int(rng.choice(len(live), p=p))]]
        out.append(split[c]["sentences"][pools[c].pop()])
    return out


def run(condition, regime, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = ValueOnly(start, seed)
    internal2 = make_internal2(condition, memory, seed)
    stored = {"z_work": start.clone(), "assoc": torch.zeros_like(model.assoc)}
    rng = np.random.default_rng(20_000 + seed)
    phase_rng = np.random.default_rng(40_000 + seed)
    order = [CATEGORIES[i] for i in rng.permutation(len(CATEGORIES))]
    schedule = [("session", c) for c in order] + [("cycle", None)] * CYCLES
    history, cat_step, seen_cats, seen_words = [], {}, [], []
    acc_now, best, expect, rpe = {}, {}, {}, {}
    steps = []

    def practise(ctx, tgt, ids, epochs):
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(ctx), tgt)
            if ids is not None:
                loss = loss + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()

    for i, (kind, cat) in enumerate(schedule):
        if i > 0:
            sents = imagine(stored["z_work"], stored["assoc"], seen_words, phase_rng, EPS0)
            pctx, ptgt, _, _ = masked_items(sents)
            practise(pctx, ptgt, None, PHASE_EPOCHS)

        if kind == "session":
            old_cats = list(seen_cats)
            seen_cats.append(cat)
            cat_step[cat] = i
            new_words = list(split[cat]["train"])
            seen_words += new_words
            old_train = [w for c in old_cats for w in split[c]["train"]]
            replay_words = [int(w) for w in rng.choice(old_train, int(len(old_train) * REPLAY), replace=False)] if old_train else []
            n_replay = int(sum(len(split[c]["sentences"]) for c in old_cats) * REPLAY)
            if regime.endswith("repair"):
                weights = {c: max(best[c] - acc_now[c], 0.0) + FLOOR for c in old_cats}
            elif regime.endswith("curious"):
                weights = {c: abs(rpe.get(c, 0.0)) + FLOOR for c in old_cats}
            else:
                weights = {c: 1.0 for c in old_cats}
            replay_sents = pick_replay(rng, split, old_cats, weights, n_replay) if old_cats else []
            heard = split[cat]["sentences"] + replay_sents
            if internal2 is not None:
                internal2.observe(heard)
                conn = internal2.connections(seen_words)
                nw = torch.tensor(new_words)
                stored["assoc"][nw, :] = conn[nw, :]
                stored["assoc"][:, nw] = conn[:, nw]
                with torch.no_grad():
                    model.assoc[nw, :] = conn[nw, :]
                    model.assoc[:, nw] = conn[:, nw]
            ctx, tgt, _, _ = masked_items(heard)
            practise(ctx, tgt, torch.tensor(new_words + replay_words), EPOCHS)
        else:
            heard = [s for c in seen_cats for s in split[c]["sentences"]]
            if internal2 is not None:
                internal2.observe(heard)
            ctx, tgt, _, _ = masked_items(heard)
            practise(ctx, tgt, torch.tensor(seen_words), CYCLE_EPOCHS)

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

        # restore: always, or only below the set point
        restored = 0
        if history and not regime.startswith("norestore"):
            for c in seen_cats:
                if c == cat and kind == "session":
                    continue
                hx, hy = partner_items(split[c]["sentences"], partner)
                if regime == "setpoint":
                    with torch.no_grad():
                        a = acc(model.complete(hx), hy)
                    if a >= best.get(c, 0.0) - DELTA:
                        continue
                if restore_category(model, stored, history, torch.tensor(split[c]["train"]), hx, hy, cat_step[c]):
                    restored += 1
        history.append((i, stored["z_work"].clone(), stored["assoc"].clone()))

        # signals from real experience: level, set point (best ever), surprise
        for c in seen_cats:
            hx, hy = partner_items(split[c]["sentences"], partner)
            with torch.no_grad():
                a = acc(model.complete(hx), hy)
            if c in acc_now:
                lp = a - acc_now[c]
                rpe[c] = lp - expect.get(c, 0.0)
                expect[c] = expect.get(c, 0.0) + EXPECT_RATE * (lp - expect.get(c, 0.0))
            acc_now[c] = a
            best[c] = max(best.get(c, 0.0), a)

        per_age = {}
        for c in seen_cats:
            x, y = partner_items(split[c]["test_sents"], partner)
            with torch.no_grad():
                per_age.setdefault(i - cat_step[c], []).append(acc(model.complete(x), y))
        older = [v for a, vs in per_age.items() if a > 0 for v in vs]
        test = partner_items([s for c in seen_cats for s in split[c]["test_sents"]], partner)
        with torch.no_grad():
            steps.append({"never-heard": acc(model.complete(test[0]), test[1]),
                          "newest": per_age[0][0] if 0 in per_age else np.nan,
                          "older": float(np.mean(older)) if older else np.nan,
                          "restored": restored})
    return steps


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(c, g): run(c, g, data, memory, seed) for c in CONDITIONS for g in REGIMES}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_steps = len(results[0][(CONDITIONS[0], REGIMES[0])])
    labels = [f"s{i+1}" if i < 6 else f"c{i - 5}" for i in range(n_steps)]

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    print(f"=== {n_seeds} seeds; DELTA={DELTA} FLOOR={FLOOR}; columns: " + " ".join(labels) + " ===")
    for key in ["never-heard", "newest", "older", "restored"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            for g in REGIMES:
                row = [m([r[(c, g)][i][key] for r in results]) for i in range(n_steps)]
                print(f"{c:5} {g:16} " + " ".join(f"{v:>6}" for v in row))

    print("\n--- paired: never-heard (life mean / end / newest in sessions 2-6 / older in sessions 2-6) ---")
    for c in CONDITIONS:
        for g, base in [("setpoint", "fixed"), ("norestore", "fixed"), ("norestore-repair", "norestore"),
                        ("norestore-curious", "norestore")]:
            cells = []
            for label, get in [("life", lambda r, gg: np.mean([st["never-heard"] for st in r[(c, gg)]])),
                               ("end", lambda r, gg: r[(c, gg)][-1]["never-heard"]),
                               ("newest", lambda r, gg: np.mean([st["newest"] for st in r[(c, gg)][1:6]])),
                               ("older", lambda r, gg: np.mean([st["older"] for st in r[(c, gg)][1:6]]))]:
                d = np.array([get(r, g) - get(r, base) for r in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                cells.append(f"{label} {d.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
            print(f"{c:5} {g:17} vs {base:9} " + "  ".join(cells))

if __name__ == "__main__":
    main()
