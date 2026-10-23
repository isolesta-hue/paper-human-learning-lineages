"""Dopamine / punishment / homeostasis: internal signals computed from the
time axis that regulate how the system learns.

Project idea (2026-09-23): in people, reaching a level of skill releases
dopamine; the brain also has punishment and homeostasis systems that work
apart from thinking. Here an internal logic watches the time-ordered record
of real performance and turns it into signals that move the knobs that were
fixed so far.

Signal (computed only from REAL experience -- accuracy on each category's
heard sentences after every step -- never from imagined sentences, so the
system cannot reward itself):
  progress   LP_c(t) = acc_c(t) - acc_c(t-1)                (per category)
  expected   E_c     = running mean of LP_c  (rate 0.5)
  surprise   RPE_c   = LP_c - E_c   ("better / worse than expected")
  dopamine   D(t)    = mean over categories of RPE_c
Knobs it moves:
  exploration -- the share of less-certain words in imagination (finding 36).
                 Dopamine: EPS <- EPS - K_EPS * D (progress worse than
                 expected -> explore more; better -> explore less).
  punishment  -- restoring a category from the past state that did best on
                 it (finding 39) only when that category just did clearly
                 worse than expected (RPE_c < -DELTA), instead of always.
                 (Finding 39: always restoring hurts the category being
                 learned right now.)
  homeostasis -- EPS is pulled back toward its baseline (rate 0.3) and kept
                 inside [0.05, 0.6], so it cannot run to an extreme.
Regimes (same life and platform for all):
  fixed            -- EPS 0.3, always restore per category (finding 39)
  fixed-norestore  -- EPS 0.3, never restore per category
  dopamine         -- EPS moved by D, always restore
  dopamine+punish  -- EPS moved by D, restore only on punishment
  full             -- dopamine + punishment + homeostasis
Platform: design A, experience flows back into memory (compare step on
everything heard), imagination with exploration + rehearsal before every
step (finding 36, rehearsal only), life = 6 sessions (one new category
each) + 4 everyday cycles (finding 37). Settings are single representative
values, not tuned.
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

CONDITIONS = ["none", "vec", "hebb"]
REGIMES = ["fixed", "fixed-norestore", "dopamine", "dopamine+punish", "full"]
CYCLES, CYCLE_EPOCHS = 4, 50
EPS0, K_EPS, DELTA, EXPECT_RATE = 0.3, 1.0, 0.02, 0.5
HOMEO_RATE, EPS_MIN, EPS_MAX = 0.3, 0.05, 0.6


def restore_category(model, stored, history, words, hx, hy, since):
    """Finding 39: restore a category's outgoing values from the past state that did best on its heard sentences."""
    with torch.no_grad():
        best_score, best_h = acc(model.complete(hx), hy), None
        for h in history:
            if h[0] < since:
                continue
            z_save, a_save = model.z_work.detach().clone(), model.assoc.detach().clone()
            model.z_work[words] = h[1][words]
            model.assoc[words, :] = h[2][words, :]
            score = acc(model.complete(hx), hy)
            model.z_work.copy_(z_save)
            model.assoc.copy_(a_save)
            if score > best_score:
                best_score, best_h = score, h
        if best_h is None:
            return False
        for target in (model.z_work, stored["z_work"]):
            target[words] = best_h[1][words]
        for target in (model.assoc, stored["assoc"]):
            target[words, :] = best_h[2][words, :]
    return True


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
    acc_hist, expect = {}, {}
    eps = EPS0
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
            sents = imagine(stored["z_work"], stored["assoc"], seen_words, phase_rng, eps)
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
            old_sents = [s for c in old_cats for s in split[c]["sentences"]]
            replay_sents = [old_sents[j] for j in rng.choice(len(old_sents), int(len(old_sents) * REPLAY), replace=False)] if old_sents else []
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

        # compare step (experience flows back into memory)
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

        # the signal: progress on real experience, compared with what was expected
        rpe = {}
        for c in seen_cats:
            hx, hy = partner_items(split[c]["sentences"], partner)
            with torch.no_grad():
                a = acc(model.complete(hx), hy)
            if c in acc_hist:
                lp = a - acc_hist[c]
                rpe[c] = lp - expect.get(c, 0.0)
                expect[c] = expect.get(c, 0.0) + EXPECT_RATE * (lp - expect.get(c, 0.0))
            acc_hist[c] = a
        dopamine = float(np.mean(list(rpe.values()))) if rpe else 0.0

        # restore (always, never, or only on punishment)
        restored = 0
        if regime != "fixed-norestore" and history:
            for c in seen_cats:
                if c not in rpe:
                    continue
                if regime in ("dopamine+punish", "full") and rpe[c] >= -DELTA:
                    continue
                hx, hy = partner_items(split[c]["sentences"], partner)
                if restore_category(model, stored, history, torch.tensor(split[c]["train"]), hx, hy, cat_step[c]):
                    restored += 1
        history.append((i, stored["z_work"].clone(), stored["assoc"].clone()))

        # exploration moved by dopamine; homeostasis pulls it back
        if regime in ("dopamine", "dopamine+punish", "full"):
            eps = float(np.clip(eps - K_EPS * dopamine, 0.0, 1.0))
        if regime == "full":
            eps = float(np.clip(eps + HOMEO_RATE * (EPS0 - eps), EPS_MIN, EPS_MAX))

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
                          "eps": eps, "dopamine": dopamine, "restored": restored})
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

    print(f"=== {n_seeds} seeds; EPS0={EPS0} K_EPS={K_EPS} DELTA={DELTA} homeostasis rate {HOMEO_RATE} "
          f"range [{EPS_MIN},{EPS_MAX}]; columns: " + " ".join(labels) + " ===")
    for key in ["never-heard", "newest", "older", "eps", "dopamine", "restored"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            for g in REGIMES:
                row = [m([r[(c, g)][i][key] for r in results]) for i in range(n_steps)]
                print(f"{c:5} {g:16} " + " ".join(f"{v:>6}" for v in row))

    print("\n--- paired vs fixed: mean over the life / at the end, never-heard (all categories) ---")
    for c in CONDITIONS:
        for g in REGIMES[1:]:
            cells = []
            for label, get in [("life mean", lambda r, gg: np.mean([st["never-heard"] for st in r[(c, gg)]])),
                               ("end", lambda r, gg: r[(c, gg)][-1]["never-heard"]),
                               ("newest (sessions 2-6)", lambda r, gg: np.mean([st["newest"] for st in r[(c, gg)][1:6]])),
                               ("older (sessions 2-6)", lambda r, gg: np.mean([st["older"] for st in r[(c, gg)][1:6]]))]:
                d = np.array([get(r, g) - get(r, "fixed") for r in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                cells.append(f"{label} {d.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
            print(f"{c:5} {g:16} " + "  ".join(cells))


if __name__ == "__main__":
    main()
