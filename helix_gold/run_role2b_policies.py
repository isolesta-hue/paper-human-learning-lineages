"""Role 2-b: WHEN and HOW the internal data is used to judge -- judgement styles.

Project decision (2026-09-22): people have no fixed rule for comparing an
attempt with what they already had in mind. Some persist to the end, some
give up quickly, some decide halfway whether the goal is still reachable,
and some estimate from their internal data alone whether it is worth trying
at all. So the judgement style is treated as an experimental variable.

External logic: design A (value-only), as run_role2_target.py. After any
stop, the compare -> update/restore step of run_role2_target.py runs, judged
on every sentence heard so far ("exp" mode).

Styles (settings are single representative values, not tuned):
  persist -- practise the full session, then compare.
  quit    -- stop as soon as practice stops paying off: accuracy on this
             session's heard sentences gained < 0.02 over the last 3 checks
             (15 epochs).
  midway  -- every 25 epochs, extrapolate the recent slope on this session's
             heard partner items to the end of the session; stop if the
             projection stays below the internal prediction's level
             (the goal). A goal of 0 never stops anyone.
  satisfied -- stop as soon as the goal is reached: accuracy on this
             session's heard partner items >= the internal prediction's level
             ("I did what I set out to do"). A goal of 0 is no goal: practise
             in full.
  pre     -- before any practice, from the internal data only: if the
             internal prediction already gets >= 0.9 of this session's heard
             partner items ("I already know this"), skip practice. Otherwise
             practise in full. (A first version used the internal
             prediction's softmax confidence with a 0.5 cut; over 72 words
             that confidence is only 0.03-0.12, so nobody ever skipped.)
Conditions: none / vec / hebb / vec+hebb / vec+bert.
Measured: effort (epochs actually practised), compare decisions, accuracy
on everything seen so far right after the decision (heard partner items and
never-heard sentences), next session's start, and the internal logic's
influence (each condition vs "none" under the same style).
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from run_unified_continual import STAGES, sign_p
from run_seed_transfer import memorize, EPOCHS, CHECK_EVERY, REPLAY
from run_connection_seed import Hebbian, MiniBert, masked_items, partner_items, acc
from run_ratchet_seed import build_data, score_with
from run_three_designs import ValueOnly
from run_role2_target import predict

CONDITIONS = ["none", "vec", "hebb", "vec+hebb", "vec+bert"]
STYLES = ["persist", "quit", "midway", "satisfied", "pre"]
QUIT_GAIN, QUIT_WINDOW = 0.02, 3
MIDWAY_EVERY = 25
PRE_ALREADY_KNOWN = 0.9


def run(condition, style, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = ValueOnly(start, seed)
    internal2 = Hebbian() if "hebb" in condition else MiniBert(memory, seed) if "bert" in condition else None
    stored = {"z_work": start.clone(), "assoc": torch.zeros_like(model.assoc)}
    rng = np.random.default_rng(20_000 + seed)
    sessions, seen_cats, seen_words = [], [], []

    for new_cats in STAGES:
        old_cats = list(seen_cats)
        seen_cats += new_cats
        new_words = [w for c in new_cats for w in split[c]["train"]]
        seen_words += new_words

        old_train = [w for c in old_cats for w in split[c]["train"]]
        replay_words = [int(w) for w in rng.choice(old_train, int(len(old_train) * REPLAY), replace=False)] if old_train else []
        new_sents = [s for c in new_cats for s in split[c]["sentences"]]
        old_sents = [s for c in old_cats for s in split[c]["sentences"]]
        replay_sents = [old_sents[i] for i in rng.choice(len(old_sents), int(len(old_sents) * REPLAY), replace=False)] if old_sents else []
        heard = new_sents + replay_sents

        if internal2 is not None:
            internal2.observe(heard)
            conn = internal2.connections(seen_words)
            nw = torch.tensor(new_words)
            stored["assoc"][nw, :] = conn[nw, :]
            stored["assoc"][:, nw] = conn[:, nw]
            with torch.no_grad():
                model.assoc[nw, :] = conn[nw, :]
                model.assoc[:, nw] = conn[:, nw]

        new_all = masked_items(new_sents)[:2]
        new_partner = partner_items(new_sents, partner)
        new_test = partner_items([s for c in new_cats for s in split[c]["test_sents"]], partner)
        with torch.no_grad():
            goal = acc(predict(stored["z_work"], stored["assoc"], new_partner[0]), new_partner[1])
            confidence = F.softmax(predict(stored["z_work"], stored["assoc"], new_all[0]), -1).max(-1).values.mean().item()
            test_at0 = acc(model.complete(new_test[0]), new_test[1])

        train_ids = torch.tensor(new_words + replay_words)
        ctx, tgt, _, _ = masked_items(heard)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        heard_curve, partner_curve = [], []
        budget = 0 if (style == "pre" and goal >= PRE_ALREADY_KNOWN) else EPOCHS
        effort = 0
        for epoch in range(budget + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    heard_curve.append(acc(model.complete(new_all[0]), new_all[1]))
                    partner_curve.append(acc(model.complete(new_partner[0]), new_partner[1]))
                k = len(heard_curve) - 1
                if style == "satisfied" and goal > 0 and partner_curve[k] >= goal - 1e-9:
                    break
                if style == "quit" and k >= QUIT_WINDOW and heard_curve[k] - heard_curve[k - QUIT_WINDOW] < QUIT_GAIN:
                    break
                if style == "midway" and epoch > 0 and epoch % MIDWAY_EVERY == 0 and goal > 0:
                    slope = (partner_curve[k] - partner_curve[k - MIDWAY_EVERY // CHECK_EVERY]) / MIDWAY_EVERY
                    if partner_curve[k] + slope * (EPOCHS - epoch) < goal - 1e-9:
                        break
            if epoch == budget:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()
            effort += 1

        judge = masked_items([s for c in seen_cats for s in split[c]["sentences"]])[:2]
        decisions = {}
        for name in ["z_work", "assoc"]:
            param = getattr(model, name)
            working = score_with(model, name, param.detach().clone(), judge)
            kept = score_with(model, name, stored[name], judge)
            if working > kept:
                stored[name] = param.detach().clone()
                decisions[name] = "update"
            elif working < kept:
                with torch.no_grad():
                    param.copy_(stored[name])
                decisions[name] = "restore"
            else:
                decisions[name] = "tie"

        seen_heard = partner_items([s for c in seen_cats for s in split[c]["sentences"]], partner)
        seen_test = partner_items([s for c in seen_cats for s in split[c]["test_sents"]], partner)
        with torch.no_grad():
            sessions.append({
                "effort": effort, "goal": goal, "confidence": confidence, "skipped": float(budget == 0),
                "decisions": decisions, "new_test_at0": test_at0,
                "new_heard_partner_end": partner_curve[-1],
                "new_test_end": acc(model.complete(new_test[0]), new_test[1]),
                "all_heard_partner": acc(model.complete(seen_heard[0]), seen_heard[1]),
                "all_never_heard": acc(model.complete(seen_test[0]), seen_test[1]),
            })
    return sessions


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(c, st): run(c, st, data, memory, seed) for c in CONDITIONS for st in STYLES}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))

    def m(vals):
        return f"{np.mean(np.array(vals, dtype=float)):.3f}"

    print(f"=== {n_seeds} seeds, design A, judged on everything heard so far ===")
    print("\n--- per session: effort (epochs practised of 200), skipped share, restores of connections/vectors ---")
    for st in STYLES:
        for c in CONDITIONS:
            cells = []
            for s in range(len(STAGES)):
                rs = [r[(c, st)][s] for r in results]
                cells.append(f"s{s+1} effort={m([x['effort'] for x in rs])} skip={m([x['skipped'] for x in rs])} "
                             f"goal={m([x['goal'] for x in rs])} conf={m([x['confidence'] for x in rs])} "
                             f"restore conn/vec={sum(x['decisions']['assoc'] == 'restore' for x in rs)}/"
                             f"{sum(x['decisions']['z_work'] == 'restore' for x in rs)}")
            print(f"{st:8} {c:9} " + " | ".join(cells))

    print("\n--- results per session (means) ---")
    keys = [("new_test_at0", "new never-heard at start"), ("new_heard_partner_end", "new heard partner at stop"),
            ("new_test_end", "new never-heard after decision"), ("all_heard_partner", "ALL heard partner after decision"),
            ("all_never_heard", "ALL never-heard after decision")]
    for key, label in keys:
        for s in range(len(STAGES)):
            print(f"s{s+1} {label:34} " + "  ".join(
                f"{st}:" + ",".join(m([r[(c, st)][s][key] for r in results]) for c in CONDITIONS) for st in STYLES))
    print("(order inside each style: " + ", ".join(CONDITIONS) + ")")

    print("\n--- internal logic's influence under each style: condition vs none, ALL never-heard after decision ---")
    for st in STYLES:
        for c in CONDITIONS[1:]:
            cells = []
            for s in range(len(STAGES)):
                d = np.array([r[(c, st)][s]["all_never_heard"] - r[("none", st)][s]["all_never_heard"] for r in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                cells.append(f"s{s+1} {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.2g})")
            print(f"{st:8} {c:9} " + "  ".join(cells))

    print("\n--- each style vs persist, same condition: ALL never-heard after decision, and effort ---")
    for st in STYLES[1:]:
        for c in CONDITIONS:
            cells = []
            for s in range(len(STAGES)):
                d = np.array([r[(c, st)][s]["all_never_heard"] - r[(c, "persist")][s]["all_never_heard"] for r in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                e = np.mean([r[(c, st)][s]["effort"] for r in results])
                cells.append(f"s{s+1} {d.mean():+.3f} ({n_pos}/{n_neg}) effort={e:.0f}")
            print(f"{st:8} {c:9} " + "  ".join(cells))


if __name__ == "__main__":
    main()
