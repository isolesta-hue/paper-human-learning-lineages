"""Signals regulating the judgement style and the amount of practice.

Finding 30: judgement styles (persist, quit early, ...) were fixed rules with
fixed thresholds; the fixed "quit" style gave up before learning anything
when the start was slow. Finding 41: choosing WHAT to review by the surprise
(curiosity) signal recovers much of what is lost when memory cannot be
restored. Here the system paces itself:
  self-stop    -- during a learning session, practice on the new sentences
                  stops once the recent progress (gain over the last 3 checks
                  = 15 epochs) falls below RATIO of this session's own early
                  progress (gain over its first 3 checks). The yardstick is
                  the learner's own pace, not a fixed number, so a slow start
                  does not trigger it (no progress yet = nothing to compare
                  to). At least MIN_EPOCHS are practised.
  reallocate   -- the epochs saved by stopping are not thrown away: they are
                  spent reviewing older categories' heard sentences, chosen by
                  the curiosity signal (finding 41). The total practice time
                  per session stays 200 epochs; only its split between new
                  learning and review is decided by the system.
All signals come from real experience only (heard sentences).
Regimes: fixed (200 epochs on the session's sentences) / self-stop /
self-stop+reallocate; each without restore (like people: memory cannot be
turned back) and with the set-point restore of findings 39/41.
Platform as run_dopamine2.py (design A, experience flows back into memory,
imagination 0.3 + rehearsal, 25% random replay inside sessions, 6 sessions
+ 4 everyday cycles). Conditions: none / vec / hebb.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from run_unified_continual import CATEGORIES, sign_p
from run_seed_transfer import memorize, EPOCHS, CHECK_EVERY, REPLAY
from run_connection_seed import masked_items, partner_items, acc
from run_ratchet_seed import build_data, score_with
from run_three_designs import ValueOnly
from run_role1_anchor import make_internal2
from run_explore_imagine import imagine
from run_role3_imagine import PHASE_EPOCHS
from run_dopamine import restore_category, CYCLES, CYCLE_EPOCHS, EPS0, EXPECT_RATE
from run_dopamine2 import pick_replay, FLOOR

CONDITIONS = ["none", "vec", "hebb"]
REGIMES = ["fixed", "stop", "stop+review", "fixed|restore", "stop+review|restore"]
RATIO, WINDOW, MIN_EPOCHS = 0.1, 3, 15


def run(condition, regime, data, memory, seed):
    vecs, cats, split, partner = data
    restore = regime.endswith("|restore")
    style = regime.split("|")[0]
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = ValueOnly(start, seed)
    internal2 = make_internal2(condition, memory, seed)
    stored = {"z_work": start.clone(), "assoc": torch.zeros_like(model.assoc)}
    rng = np.random.default_rng(20_000 + seed)
    phase_rng = np.random.default_rng(40_000 + seed)
    order = [CATEGORIES[i] for i in rng.permutation(len(CATEGORIES))]
    schedule = [("session", c) for c in order] + [("cycle", None)] * CYCLES
    history, cat_step, seen_cats, seen_words = [], {}, [], []
    acc_now, expect, rpe = {}, {}, {}
    steps = []

    def practise(ctx, tgt, ids, epochs, monitor=None):
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        curve, used = [], 0
        for e in range(epochs):
            if monitor is not None and e % CHECK_EVERY == 0:
                with torch.no_grad():
                    curve.append(acc(model.complete(monitor[0]), monitor[1]))
                k = len(curve) - 1
                if e >= MIN_EPOCHS and k >= 2 * WINDOW:
                    early = curve[WINDOW] - curve[0]
                    recent = curve[k] - curve[k - WINDOW]
                    if early > 0 and recent < RATIO * early:
                        break
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(ctx), tgt)
            if ids is not None:
                loss = loss + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()
            used += 1
        return used

    for i, (kind, cat) in enumerate(schedule):
        if i > 0:
            sents = imagine(stored["z_work"], stored["assoc"], seen_words, phase_rng, EPS0)
            pctx, ptgt, _, _ = masked_items(sents)
            practise(pctx, ptgt, None, PHASE_EPOCHS)

        new_epochs = review_epochs = 0
        if kind == "session":
            old_cats = list(seen_cats)
            seen_cats.append(cat)
            cat_step[cat] = i
            new_words = list(split[cat]["train"])
            seen_words += new_words
            old_train = [w for c in old_cats for w in split[c]["train"]]
            replay_words = [int(w) for w in rng.choice(old_train, int(len(old_train) * REPLAY), replace=False)] if old_train else []
            n_replay = int(sum(len(split[c]["sentences"]) for c in old_cats) * REPLAY)
            replay_sents = pick_replay(rng, split, old_cats, {c: 1.0 for c in old_cats}, n_replay) if old_cats else []
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
            monitor = masked_items(split[cat]["sentences"])[:2] if style != "fixed" else None
            new_epochs = practise(ctx, tgt, torch.tensor(new_words + replay_words), EPOCHS, monitor)
            saved = EPOCHS - new_epochs
            if style == "stop+review" and saved > 0 and old_cats:
                weights = {c: abs(rpe.get(c, 0.0)) + FLOOR for c in old_cats}
                review = pick_replay(rng, split, old_cats, weights,
                                     min(sum(len(split[c]["sentences"]) for c in old_cats), 2 * n_replay + 2))
                rctx, rtgt, _, _ = masked_items(review)
                review_epochs = practise(rctx, rtgt, torch.tensor(old_train), saved)
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

        if restore and history:
            for c in seen_cats:
                if c == cat and kind == "session":
                    continue
                hx, hy = partner_items(split[c]["sentences"], partner)
                restore_category(model, stored, history, torch.tensor(split[c]["train"]), hx, hy, cat_step[c])
        history.append((i, stored["z_work"].clone(), stored["assoc"].clone()))

        for c in seen_cats:
            hx, hy = partner_items(split[c]["sentences"], partner)
            with torch.no_grad():
                a = acc(model.complete(hx), hy)
            if c in acc_now:
                lp = a - acc_now[c]
                rpe[c] = lp - expect.get(c, 0.0)
                expect[c] = expect.get(c, 0.0) + EXPECT_RATE * (lp - expect.get(c, 0.0))
            acc_now[c] = a

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
                          "new epochs": new_epochs if kind == "session" else np.nan,
                          "review epochs": review_epochs if kind == "session" else np.nan})
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

    print(f"=== {n_seeds} seeds; RATIO={RATIO} WINDOW={WINDOW} MIN_EPOCHS={MIN_EPOCHS}; columns: " + " ".join(labels) + " ===")
    for key in ["never-heard", "newest", "older", "new epochs", "review epochs"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            for g in REGIMES:
                row = [m([r[(c, g)][i][key] for r in results]) for i in range(n_steps)]
                print(f"{c:5} {g:20} " + " ".join(f"{v:>6}" for v in row))

    print("\n--- paired: never-heard (life mean / end / newest in sessions 2-6 / older in sessions 2-6) ---")
    for c in CONDITIONS:
        for g, base in [("stop", "fixed"), ("stop+review", "fixed"), ("stop+review", "stop"),
                        ("stop+review|restore", "fixed|restore")]:
            cells = []
            for label, get in [("life", lambda r, gg: np.mean([st["never-heard"] for st in r[(c, gg)]])),
                               ("end", lambda r, gg: r[(c, gg)][-1]["never-heard"]),
                               ("newest", lambda r, gg: np.mean([st["newest"] for st in r[(c, gg)][1:6]])),
                               ("older", lambda r, gg: np.mean([st["older"] for st in r[(c, gg)][1:6]]))]:
                d = np.array([get(r, g) - get(r, base) for r in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                cells.append(f"{label} {d.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
            print(f"{c:5} {g:20} vs {base:14} " + "  ".join(cells))


if __name__ == "__main__":
    main()
