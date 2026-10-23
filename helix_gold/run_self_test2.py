"""Self-test, second form: set aside only while judging, then give it back
(finding 45 follow-up, 2026-09-23).

Finding 45: the self-test is an honest yardstick (it levels off with the
outside test while the practice score keeps climbing, and it prevents the very
early stop of the first session), but setting a quarter of the experience
aside FOR GOOD cost more than it gained (-0.087). People do not lock their
mock exam away either: after the exam they study it too.

Here a new category's sentences are split once: while the category is being
learned (the first part of its session) a quarter is set aside and used only
as the self-test for the stop judgement (not practised, not seen by the
connection seed); once the stop rule fires, the set-aside quarter is given
back -- the rest of the session's 200 epochs practises ALL of the new
categories' sentences together with a review sample of older categories'
sentences, and the connection seed observes the returned sentences. Nothing
is lost for good. The self-test is only needed for the new categories of a
session and a category is new only once, so the "rotation" amounts to: each
quarter is set aside once, while its own category is judged.

Conditions (platform and stop rule exactly as run_self_test.py / finding 42)
  all/fixed           -- nothing set aside, 200 epochs on the session's sentences + 25% replay
  all/stop:practice   -- nothing set aside; stop by the practised sentences, then all new + review
  temp/stop:practice  -- a quarter set aside while learning; stop judged by the practised sentences;
                         then the quarter is given back (structure control)
  temp/stop:self      -- the same, stop judged by the set-aside quarter (the self-test)
Controls added after the first 100-seed run (is the gain from the stop JUDGEMENT or from the two-phase
schedule that mixes more older sentences in?):
  all/mixed           -- one phase: 200 epochs on the new sentences + the review sample from the start
  all/half            -- two phases with a fixed switch at 100 epochs, no judgement
`python run_self_test2.py <seeds> <workers> [comma-separated conditions]`
A "+keep" suffix on any condition keeps ONE Adam optimizer (and its state) for the whole life instead of making a
new one at every practice call (optimizer-restart check, finding 46 observation / finding 47).
"""
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
from run_self_test import positions, score, stop_point, HOLD, LR, REPLAY, CHECK_EVERY, RATIO, WINDOW, MIN_EPOCHS

CONDITIONS = ["all/fixed", "all/stop:practice", "temp/stop:practice", "temp/stop:self"]
CONTROLS = ["all/mixed", "all/half"]


def life(cond, d, seed):
    keep = cond.endswith("+keep")
    cond = cond[:-len("+keep")] if keep else cond
    temp = cond.startswith("temp")
    stop = "stop" in cond
    judge_on = cond.split(":")[-1] if stop else None
    torch.manual_seed(seed)
    rng = np.random.default_rng(50_000 + seed)
    cats = torch.tensor(d.cats)
    memory, _ = memorize(torch.tensor(d.vecs), seed)
    model = L.KnowDo2(d.V, d.n_cat, memory, seed)
    hebb = L.make_internal2("dhebb", d.V)
    sents, aside = {}, {}
    for c, cat in enumerate(d.categories):
        ss = [s for _, s, _ in cat["heard"]]
        idx = rng.permutation(len(ss))
        n_hold = int(round(len(ss) * HOLD)) if temp else 0
        aside[c] = [ss[i] for i in idx[:n_hold]]
        sents[c] = [ss[i] for i in idx[n_hold:]] + aside[c]           # all of the category, set-aside last
    hidden = {c: [(ctx, t) for _, ctx, t in D.unit_items(d.categories[c]["test"])] for c in range(d.n_cat)}
    order = [int(c) for c in rng.permutation(d.n_cat)]
    schedule = [("session", order[i:i + PER_SESSION]) for i in range(0, d.n_cat, PER_SESSION)] + [("cycle", [])] * CYCLES
    known, seen_cats, steps = set(), [], []
    life_opt = torch.optim.Adam(model.parameters(), lr=LR) if keep else None

    def practise(items, ids, epochs, monitor=None):
        ctx, tgt = L.items_tensor(items)
        ctx = L.bag(ctx, d.V)
        opt = life_opt if keep else torch.optim.Adam(model.parameters(), lr=LR)
        curve, used = [], 0
        for e in range(epochs):
            if monitor is not None and e % CHECK_EVERY == 0:
                curve.append(score(model, monitor))
                if stop_point(curve) == len(curve) - 1:
                    break
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(ctx), tgt)
            if ids is not None:
                loss = loss + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()
            used += 1
        return used

    def seed_connections(observed, new_words):
        hebb.observe(observed)
        conn = hebb.connections(sorted(known))
        nw = torch.tensor(sorted(new_words))
        with torch.no_grad():
            model.assoc[nw, :] = conn[nw, :]
            model.assoc[:, nw] = conn[:, nw]

    for i, (kind, new_cats) in enumerate(schedule):
        if i > 0:
            practise(positions(imagine(model, sorted(known), known, rng)), None, PHASE_EPOCHS)

        new_epochs = after_epochs = np.nan
        old_cats = list(seen_cats)
        seen_cats += new_cats
        if kind == "session":
            learn = [s for c in new_cats for s in sents[c][:len(sents[c]) - len(aside[c])]]
            held = [s for c in new_cats for s in aside[c]]
            new_words = {w for s in learn for w in s}
            known |= new_words
            old_sents = [s for c in old_cats for s in sents[c]]
            replay = [old_sents[j] for j in rng.choice(len(old_sents), int(len(old_sents) * REPLAY), replace=False)] if old_sents else []
            seed_connections(learn + replay, new_words)
            monitor = positions(learn if judge_on == "practice" else held) if stop else None
            review = [old_sents[j] for j in rng.choice(len(old_sents), min(len(old_sents), 2 * len(replay) + 2), replace=False)] if old_sents else []
            first = learn + (review if cond == "all/mixed" else replay)
            ids = torch.tensor(sorted({x for s in first for x in s}))
            new_epochs = practise(positions(first), ids, EPOCHS // 2 if cond == "all/half" else EPOCHS, monitor)
            after_epochs = 0 if cond == "all/mixed" else EPOCHS - new_epochs
            if held:                                      # give the set-aside quarter back
                back_words = {w for s in held for w in s} - known
                known |= back_words
                seed_connections(held, back_words | {w for s in held for w in s})
            if after_epochs > 0:
                rest = learn + held + review
                practise(positions(rest), torch.tensor(sorted({x for s in rest for x in s})), after_epochs)
        else:
            allsents = [s for c in seen_cats for s in sents[c]]
            hebb.observe(allsents)
            practise(positions(allsents), torch.tensor(sorted(known)), CYCLE_EPOCHS)

        per = {c: score(model, hidden[c]) for c in seen_cats}
        steps.append({"never-heard": score(model, [it for c in seen_cats for it in hidden[c]]),
                      "newest": float(np.mean([per[c] for c in new_cats])) if new_cats else np.nan,
                      "older": float(np.mean([per[c] for c in old_cats])) if (old_cats and kind == "session") else np.nan,
                      "new epochs": new_epochs, "after epochs": after_epochs})
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

    print(f"=== {n_seeds} seeds; hard level, design D + both seeds; set aside {HOLD} while judging, then given back; "
          f"RATIO={RATIO} WINDOW={WINDOW} MIN_EPOCHS={MIN_EPOCHS}; columns: " + " ".join(labels) + " ===")
    for key in ["never-heard", "newest", "older", "new epochs", "after epochs"]:
        print(f"\n--- {key} ---")
        for c in conds:
            print(f"{c:24} " + " ".join(f"{m([r[c][i][key] for r in results]):>6}" for i in range(n_steps)))

    print("\n--- paired: never-heard (life mean / end / session 1 / newest in sessions 2-8 / older in sessions 2-8) ---")
    for g, base in [("all/stop:practice", "all/fixed"), ("temp/stop:practice", "all/stop:practice"),
                    ("temp/stop:self", "temp/stop:practice"), ("temp/stop:self", "all/stop:practice"),
                    ("temp/stop:self", "all/fixed"), ("all/mixed", "all/fixed"), ("all/half", "all/fixed"),
                    ("all/stop:practice", "all/half"), ("all/stop:practice", "all/mixed"), ("all/half", "all/mixed"),
                    ("all/fixed+keep", "all/fixed"), ("all/mixed+keep", "all/mixed"), ("all/half+keep", "all/half"),
                    ("all/stop:practice+keep", "all/stop:practice"), ("all/mixed+keep", "all/fixed+keep"),
                    ("all/half+keep", "all/fixed+keep"), ("all/stop:practice+keep", "all/half+keep"),
                    ("all/stop:practice+keep", "all/mixed+keep")]:
        if g not in conds or base not in conds:
            continue
        cells = []
        for label, get in [("life", lambda r, gg: np.mean([st["never-heard"] for st in r[gg]])),
                           ("end", lambda r, gg: r[gg][-1]["never-heard"]),
                           ("s1", lambda r, gg: r[gg][0]["never-heard"]),
                           ("newest", lambda r, gg: np.mean([st["newest"] for st in r[gg][1:n_sess]])),
                           ("older", lambda r, gg: np.mean([st["older"] for st in r[gg][1:n_sess]]))]:
            dd = np.array([get(r, g) - get(r, base) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{label} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{g:24} vs {base:22} " + "  ".join(cells))


if __name__ == "__main__":
    main()
