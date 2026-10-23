"""Role 2 of internal data: the internal prediction as the level the external
logic aims for -- plus compare/restore judged on different grounds.

Project decision (2026-09-22, finding 28): internal data is not a competitor
the external logic overwrites. In humans it serves as
  1. the basis for understanding new input,
  2. the reference for "it went as expected" / "I reached the goal": the
     level the external learning has to reach, and
  3. material for internal thinking/simulation without outside input.
This script measures role 2. Restoring stays -- people also try something,
decide their old idea was better, and throw the new attempt away -- but WHAT
the decision is judged on is varied.

External logic: design A of finding 27 (value-only: the sentence answer is
cosine similarity of word vectors + the connection table).

Internal prediction = the same computation on the STORED tables only
(Module A memory + internal logic 2's connections, or whatever earlier
updates put there). It is fixed during a session and is the "expected"
answer.

Measured every session:
  catch-up   -- first check at which the external logic reaches the internal
                prediction's accuracy, on new heard sentences / new
                never-heard sentences / old categories' heard sentences;
                and whether it ends above it (surpass)
  surprise   -- new heard PARTNER items (the answerable positions; a filler
                position has no single right answer) split into "internal predicted it"
                (expected) and "internal got it wrong" (surprising):
                learning curve of each; for expected items, how far accuracy
                dips below the start and how long it takes to come back
                ("knew it in theory, can't do it at first")
Restore modes (end of each session, table by table, as findings 26-27):
  off  -- never compare
  val  -- judged on never-heard validation sentences (finding 27)
  exp  -- judged on every sentence actually heard so far (all sessions),
          i.e. "does the new way work on what I have experienced?"
Conditions: none / vec / hebb / vec+hebb / vec+bert (as finding 27).
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

CONDITIONS = ["none", "vec", "hebb", "vec+hebb", "vec+bert"]
MODES = ["off", "val", "exp"]


def predict(z, assoc, ctx):
    similarity = F.normalize(z[ctx].mean(dim=1), dim=-1) @ F.normalize(z, dim=-1).T
    return similarity + assoc[ctx].mean(dim=1)


def first_reach(curve, target):
    for i, v in enumerate(curve):
        if v >= target - 1e-9:
            return i
    return float("nan")


def run(condition, mode, data, memory, seed):
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

        sets = {
            "new heard": masked_items(new_sents)[:2],
            "new heard partner": partner_items(new_sents, partner),
            "new never-heard": partner_items([s for c in new_cats for s in split[c]["test_sents"]], partner),
        }
        if old_cats:
            sets["old heard partner"] = partner_items(old_sents, partner)
        with torch.no_grad():
            internal_level = {k: acc(predict(stored["z_work"], stored["assoc"], x), y) for k, (x, y) in sets.items()}
            hx, hy = sets["new heard partner"]
            expected = predict(stored["z_work"], stored["assoc"], hx).argmax(-1) == hy
        groups = {"expected": expected, "surprising": ~expected}

        train_ids = torch.tensor(new_words + replay_words)
        ctx, tgt, _, _ = masked_items(heard)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        curves = {k: [] for k in list(sets) + list(groups)}
        for epoch in range(EPOCHS + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    for k, (x, y) in sets.items():
                        curves[k].append(acc(model.complete(x), y))
                    right = model.complete(hx).argmax(-1) == hy
                    for g, m in groups.items():
                        curves[g].append(right[m].float().mean().item() if m.any() else float("nan"))
            if epoch == EPOCHS:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()

        decisions = {}
        if mode != "off":
            if mode == "val":
                judge = partner_items([s for c in seen_cats for s in split[c]["val_sents"]], partner)
            else:
                judge = masked_items([s for c in seen_cats for s in split[c]["sentences"]])[:2]
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
            after = {"all heard partner": acc(model.complete(seen_heard[0]), seen_heard[1]),
                     "all never-heard": acc(model.complete(seen_test[0]), seen_test[1])}

        exp_curve = np.array(curves["expected"])
        res = {"internal": internal_level, "decisions": decisions, "after": after,
               "n_surprising": int((~expected).sum()), "n_items": len(expected)}
        for k in sets:
            c = curves[k]
            res[k] = {"catch_up": first_reach(c, internal_level[k]), "end": c[-1],
                      "surpass": float(c[-1] > internal_level[k] + 1e-9), "at0": c[0], "at10": c[2]}
        sur = curves["surprising"]
        res["surprising"] = {"at0": sur[0], "at10": sur[2], "end": sur[-1],
                             "to_90": first_reach(sur, 0.9 * sur[-1]) if not np.isnan(sur[-1]) else float("nan")}
        if not np.isnan(exp_curve[0]):
            dip = exp_curve[0] - exp_curve.min()
            back = next((i for i in range(int(exp_curve.argmin()), len(exp_curve)) if exp_curve[i] >= exp_curve[0] - 1e-9),
                        float("nan"))
            res["expected"] = {"start": exp_curve[0], "min": exp_curve.min(), "dip": dip,
                               "dip_at": int(exp_curve.argmin()), "back_at": back, "end": exp_curve[-1]}
        else:
            res["expected"] = None
        sessions.append(res)
    return sessions


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(c, m): run(c, m, data, memory, seed) for c in CONDITIONS for m in MODES}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))

    def m(vals):
        vals = np.array(vals, dtype=float)
        ok = vals[~np.isnan(vals)]
        return f"{ok.mean():.3f}" if len(ok) else "  -  "

    def reached(vals):
        vals = np.array(vals, dtype=float)
        return f"{(~np.isnan(vals)).mean():.2f}"

    print(f"=== {n_seeds} seeds, design A, {EPOCHS} epochs/session (check every {CHECK_EVERY}) ===")

    print("\n--- internal prediction (the target) vs external logic, restore off ---")
    print("target = accuracy of the stored tables alone; reach = share of seeds that reach it;")
    print("catch = mean check index of reaching it (reached seeds only); end = external at session end")
    for s in range(len(STAGES)):
        for k in ["new heard", "new heard partner", "new never-heard", "old heard partner"]:
            if s == 0 and k.startswith("old"):
                continue
            cells = []
            for c in CONDITIONS:
                rs = [r[(c, "off")][s] for r in results]
                cells.append(f"{c}: target={m([x['internal'][k] for x in rs])} reach={reached([x[k]['catch_up'] for x in rs])} "
                             f"catch={m([x[k]['catch_up'] for x in rs])} end={m([x[k]['end'] for x in rs])} "
                             f"surpass={m([x[k]['surpass'] for x in rs])}")
            print(f"s{s+1} {k:18} | " + " | ".join(cells))

    print("\n--- surprise: new heard partner items the internal prediction got wrong / right (restore off) ---")
    for s in range(len(STAGES)):
        for c in CONDITIONS:
            rs = [r[(c, "off")][s] for r in results]
            share = np.mean([x["n_surprising"] / x["n_items"] for x in rs])
            sur = f"surprising {share:.2f} of items: at0={m([x['surprising']['at0'] for x in rs])} " \
                  f"at10={m([x['surprising']['at10'] for x in rs])} end={m([x['surprising']['end'] for x in rs])} " \
                  f"to90={m([x['surprising']['to_90'] for x in rs])}"
            ex = [x["expected"] for x in rs if x["expected"] is not None]
            exp = f"expected: start={m([e['start'] for e in ex])} min={m([e['min'] for e in ex])} " \
                  f"dip>0.05 in {np.mean([e['dip'] > 0.05 for e in ex]):.2f} of seeds, dip_at={m([e['dip_at'] for e in ex])} " \
                  f"back_at={m([e['back_at'] for e in ex])} end={m([e['end'] for e in ex])}" if ex else "expected: -"
            print(f"s{s+1} {c:9} {sur} || {exp}")

    print("\n--- restore modes: decisions (update/restore/tie) and effect ---")
    for c in CONDITIONS:
        for mode in ["val", "exp"]:
            for name in ["z_work", "assoc"]:
                cells = []
                for s in range(len(STAGES)):
                    d = [r[(c, mode)][s]["decisions"][name] for r in results]
                    cells.append(f"s{s+1} {d.count('update')}/{d.count('restore')}/{d.count('tie')}")
                print(f"{c:9} {mode} {'vectors' if name == 'z_work' else 'connections':11} " + "  ".join(cells))
    print("\nright after the decision, every category seen so far (heard partner / never-heard), by restore mode:")
    for c in CONDITIONS:
        for s in range(len(STAGES)):
            cells = []
            for key in ["all heard partner", "all never-heard"]:
                cells.append(key + ": " + " ".join(
                    f"{mode}={m([r[(c, mode)][s]['after'][key] for r in results])}" for mode in MODES))
                for mode in ["val", "exp"]:
                    d = np.array([r[(c, mode)][s]["after"][key] - r[(c, "off")][s]["after"][key] for r in results])
                    n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                    cells[-1] += f" [{mode}-off {n_pos}/{n_neg}]"
            print(f"{c:9} s{s+1}  " + "  |  ".join(cells))
    print("\nnext session, new never-heard sentences at0/at10, by restore mode:")
    for c in CONDITIONS:
        for s in range(1, len(STAGES)):
            print(f"{c:9} s{s+1}  " + "  ".join(
                f"{mode}: at0={m([r[(c, mode)][s]['new never-heard']['at0'] for r in results])} "
                f"at10={m([r[(c, mode)][s]['new never-heard']['at10'] for r in results])}" for mode in MODES))

if __name__ == "__main__":
    main()
