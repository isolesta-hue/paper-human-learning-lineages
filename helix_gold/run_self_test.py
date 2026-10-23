"""Self-test: judging oneself on one's own UNPRACTISED experience (roadmap
step 3, 2026-09-23).

Finding 42: stopping when progress on the practised sentences levels off
stops too early -- the practised sentences are learned before the
understanding that carries over to never-heard sentences. People check
themselves with a mock exam instead of the exercises they already did.
Self-test here: for every category, a quarter of the learner's own heard
sentences (HOLD) is set aside and never used for anything but testing -- not
for practice, not for the connection seed (internal logic 2 observes only the
practised sentences), not for imagination. The learner does not know which
words form the true relations, so the self-test hides EVERY position of the
set-aside sentences in turn. The outside test (the hidden test sentences of
data_v2) is never seen by the learner.

Platform (as run_society.py, one learner): hard level of data_v2, 16
categories arriving 2 per session over 8 sessions + 2 everyday cycles; design
D (`lab2.KnowDo2`) with the vector seed and the directional connection seed;
learning rate 0.01, 25% replay of older practised sentences; an imagination
practice slot (50 epochs) before every step.

Conditions
  all/fixed           -- no set-aside: every heard sentence is practised, 200 epochs per session
  hold/fixed          -- a quarter set aside but not used (the cost of setting it aside)
  stop:practice       -- stop the session when progress on the session's practised sentences levels
                         off (finding 42's rule: recent gain over 3 checks < 0.1 x the session's early
                         gain, checks every 5 epochs, at least 15 epochs); saved epochs review older
                         categories' practised sentences (uniformly chosen)
  stop:self           -- the same rule on the session's self-test sentences
  compare:practice    -- after every step, the compare step of the original design (items 10-12):
                         for each part (vectors, connection table, skill) keep the working values or go
                         back to the stored ones, whichever scores better, judged on all practised sentences
  compare:self        -- the same, judged on all self-test sentences
  stop+compare:self   -- both, judged on the self-test
Yardstick check (in hold/fixed, where nothing is acted on): during every session the practice score, the
self-test score and the outside score of the new categories are recorded every 5 epochs; the stop rule is
applied afterwards to each yardstick to see where it would have stopped and what the outside score was there.
Representative values: HOLD 1/4, RATIO 0.1, WINDOW 3, MIN_EPOCHS 15 (finding 42), REPLAY 0.25, lr 0.01.
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

HOLD, LR, REPLAY = 0.25, 0.01, 0.25
RATIO, WINDOW, MIN_EPOCHS, CHECK_EVERY = 0.1, 3, 15, 5
CONDITIONS = ["all/fixed", "hold/fixed", "stop:practice", "stop:self", "compare:practice", "compare:self",
              "stop+compare:self"]
PARTS = ["z_work", "assoc", "skill"]


def positions(sents):
    return D.all_positions([("x", s, []) for s in sents])


def score(model, items):
    ctx, tgt = L.items_tensor(items)
    with torch.no_grad():
        return L.acc(model.complete(ctx), tgt)


def stop_point(curve):
    """index of the check at which finding 42's rule stops (None = never)."""
    for k in range(len(curve)):
        if k * CHECK_EVERY >= MIN_EPOCHS and k >= 2 * WINDOW:
            early, recent = curve[WINDOW] - curve[0], curve[k] - curve[k - WINDOW]
            if early > 0 and recent < RATIO * early:
                return k
    return None


def life(cond, d, seed):
    hold = cond != "all/fixed"
    stop = cond.startswith("stop")
    compare = "compare" in cond
    judge_on = cond.split(":")[-1] if ":" in cond else None
    torch.manual_seed(seed)
    rng = np.random.default_rng(50_000 + seed)
    cats = torch.tensor(d.cats)
    memory, _ = memorize(torch.tensor(d.vecs), seed)
    model = L.KnowDo2(d.V, d.n_cat, memory, seed)
    hebb = L.make_internal2("dhebb", d.V)
    prac, selfq = {}, {}
    for c, cat in enumerate(d.categories):
        sents = [s for _, s, _ in cat["heard"]]
        idx = rng.permutation(len(sents))
        n_hold = int(round(len(sents) * HOLD)) if hold else 0
        selfq[c] = [sents[i] for i in idx[:n_hold]]
        prac[c] = [sents[i] for i in idx[n_hold:]]
    hidden = {c: [(ctx, t) for _, ctx, t in D.unit_items(d.categories[c]["test"])] for c in range(d.n_cat)}
    order = [int(c) for c in rng.permutation(d.n_cat)]
    schedule = [("session", order[i:i + PER_SESSION]) for i in range(0, d.n_cat, PER_SESSION)] + [("cycle", [])] * CYCLES
    stored = {p: getattr(model, p).detach().clone() for p in PARTS}
    known, seen_cats, steps, yard = set(), [], [], []

    def practise(items, ids, epochs, monitor=None, record=None):
        ctx, tgt = L.items_tensor(items)
        ctx = L.bag(ctx, d.V)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        curve, used = [], 0
        for e in range(epochs + 1):
            if e % CHECK_EVERY == 0 and (monitor is not None or record is not None):
                if record is not None:
                    record.append({k: score(model, v) for k, v in record_sets.items()})
                if monitor is not None:
                    curve.append(score(model, monitor))
                    k = stop_point(curve)
                    if k is not None and k == len(curve) - 1:
                        break
            if e == epochs:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(ctx), tgt)
            if ids is not None:
                loss = loss + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()
            used += 1
        return used

    for i, (kind, new_cats) in enumerate(schedule):
        if i > 0:
            sents = imagine(model, sorted(known), known, rng)
            practise(positions(sents), None, PHASE_EPOCHS)

        new_epochs = review_epochs = np.nan
        old_cats = list(seen_cats)
        seen_cats += new_cats
        if kind == "session":
            new_sents = [s for c in new_cats for s in prac[c]]
            new_words = {w for s in new_sents for w in s}
            known |= new_words
            old_sents = [s for c in old_cats for s in prac[c]]
            replay = [old_sents[j] for j in rng.choice(len(old_sents), int(len(old_sents) * REPLAY), replace=False)] if old_sents else []
            hebb.observe(new_sents + replay)
            conn = hebb.connections(sorted(known))
            nw = torch.tensor(sorted(new_words))
            with torch.no_grad():
                model.assoc[nw, :] = conn[nw, :]
                model.assoc[:, nw] = conn[:, nw]
            monitor = None
            if stop:
                monitor = positions(new_sents if judge_on == "practice" else [s for c in new_cats for s in selfq[c]])
            record = None
            if cond == "hold/fixed":
                record_sets = {"practice": positions(new_sents),
                               "self": positions([s for c in new_cats for s in selfq[c]]),
                               "outside": [it for c in new_cats for it in hidden[c]]}
                record = []
            ids = torch.tensor(sorted(new_words | {x for s in replay for x in s}))
            new_epochs = practise(positions(new_sents + replay), ids, EPOCHS, monitor, record)
            if record is not None:
                yard.append(record)
            saved = EPOCHS - new_epochs
            review_epochs = 0
            if stop and saved > 0 and old_sents:
                review = [old_sents[j] for j in rng.choice(len(old_sents), min(len(old_sents), 2 * len(replay) + 2), replace=False)]
                review_epochs = practise(positions(review), torch.tensor(sorted({x for s in review for x in s})), saved)
        else:
            allsents = [s for c in seen_cats for s in prac[c]]
            hebb.observe(allsents)
            practise(positions(allsents), torch.tensor(sorted(known)), CYCLE_EPOCHS)

        if compare:
            judge = positions([s for c in seen_cats for s in (prac[c] if judge_on == "practice" else selfq[c])])
            for p in PARTS:
                param = getattr(model, p)
                working = param.detach().clone()
                s_work = score(model, judge)
                with torch.no_grad():
                    param.copy_(stored[p])
                s_kept = score(model, judge)
                with torch.no_grad():
                    if s_work >= s_kept:
                        param.copy_(working)
                        stored[p] = working
        per = {c: score(model, hidden[c]) for c in seen_cats}
        steps.append({"never-heard": score(model, [it for c in seen_cats for it in hidden[c]]),
                      "newest": float(np.mean([per[c] for c in new_cats])) if new_cats else np.nan,
                      "older": float(np.mean([per[c] for c in old_cats])) if (old_cats and kind == "session") else np.nan,
                      "new epochs": new_epochs, "review epochs": review_epochs})
    return steps, yard


def one_seed(seed):
    torch.set_num_threads(1)
    d = D.build("hard", seed)
    return {c: life(c, d, seed) for c in CONDITIONS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_steps = len(results[0][CONDITIONS[0]][0])
    n_sess = n_steps - CYCLES
    labels = [f"s{i+1}" if i < n_sess else f"c{i - n_sess + 1}" for i in range(n_steps)]

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    print(f"=== {n_seeds} seeds; hard level, design D + both seeds; HOLD={HOLD} RATIO={RATIO} WINDOW={WINDOW} "
          f"MIN_EPOCHS={MIN_EPOCHS}; columns: " + " ".join(labels) + " ===")
    for key in ["never-heard", "newest", "older", "new epochs", "review epochs"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            print(f"{c:18} " + " ".join(f"{m([r[c][0][i][key] for r in results]):>6}" for i in range(n_steps)))

    # yardstick check (hold/fixed, sessions 2..8): average curves and where the stop rule would fire
    print("\n--- yardstick check (hold/fixed, sessions 2-8, new categories): mean score every 5 epochs ---")
    sess = [rec for r in results for rec in r["hold/fixed"][1][1:]]
    for k in ["practice", "self", "outside"]:
        curve = np.mean([[pt[k] for pt in rec] for rec in sess], axis=0)
        print(f"{k:9} " + " ".join(f"{v:.2f}" for v in curve[::4]) + f"   (epochs 0,20,40,...; peak at epoch {int(np.argmax(curve)) * CHECK_EVERY})")
    for k in ["practice", "self"]:
        stops, out_at, out_end, out_best = [], [], [], []
        for rec in sess:
            kk = stop_point([pt[k] for pt in rec])
            kk = len(rec) - 1 if kk is None else kk
            stops.append(kk * CHECK_EVERY)
            out_at.append(rec[kk]["outside"])
            out_end.append(rec[-1]["outside"])
            out_best.append(max(pt["outside"] for pt in rec))
        print(f"stop by {k:8}: epoch {np.mean(stops):5.1f} (median {np.median(stops):.0f}); outside score there "
              f"{np.mean(out_at):.3f} vs at 200 epochs {np.mean(out_end):.3f} vs best along the way {np.mean(out_best):.3f}")

    print("\n--- paired: never-heard (life mean / end / newest in sessions 2-8 / older in sessions 2-8) ---")
    for g, base in [("hold/fixed", "all/fixed"), ("stop:practice", "hold/fixed"), ("stop:self", "hold/fixed"),
                    ("stop:self", "stop:practice"), ("compare:practice", "hold/fixed"), ("compare:self", "hold/fixed"),
                    ("compare:self", "compare:practice"), ("stop+compare:self", "hold/fixed"),
                    ("stop+compare:self", "all/fixed")]:
        cells = []
        for label, get in [("life", lambda r, gg: np.mean([st["never-heard"] for st in r[gg][0]])),
                           ("end", lambda r, gg: r[gg][0][-1]["never-heard"]),
                           ("newest", lambda r, gg: np.mean([st["newest"] for st in r[gg][0][1:n_sess]])),
                           ("older", lambda r, gg: np.mean([st["older"] for st in r[gg][0][1:n_sess]]))]:
            dd = np.array([get(r, g) - get(r, base) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{label} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{g:18} vs {base:14} " + "  ".join(cells))


if __name__ == "__main__":
    main()
