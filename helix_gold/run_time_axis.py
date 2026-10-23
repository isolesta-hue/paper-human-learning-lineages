"""Time-axis memory: keep every past state of the internal memory, in order,
and use past states.

Project idea (2026-09-22): people remember how their knowledge and their
judgement changed while growing up, and sometimes use an old state of
knowledge. Here the internal memory (stored vector table + connection
table) is kept at every step with a time stamp that learning never touches,
and every word remembers when it was first learned -- each word gets a
trajectory t0 (internal seed) -> t1 -> t2 -> ...

A longer life than before (the user allowed more continual steps): 6
learning sessions with ONE new category each (25% replay of earlier
sentences, as before), then 4 everyday cycles that hear everything again.
Experience flows back into the stored memory after every step (compare ->
update / restore judged on everything heard so far, as finding 33 "loop on"),
so the snapshots really hold what was learned at each time.

Uses of the time axis (no labels needed to choose which state to use):
  era recall -- answer an item from the snapshot taken right after the step
                in which its words were learned ("I learned that as a
                child"), instead of the current memory. The era of an item
                is the latest first-learned step among its context words.
  best-era recall -- answer an item from the past snapshot that did best on
                that category's HEARD sentences (experience, not the test
                sentences): "the time I was best at this".
  multi-point restore -- at the compare step, the working tables are
                compared with EVERY past snapshot, not only the last stored
                one, and the best one (on everything heard so far) is kept.
Measured after every step, for each category by its age (steps since it
was learned): the current memory, era recall, and the external logic, on
the heard partner items and on never-heard sentences of that category.
  per-category restore (cat-best) -- the external logic itself uses the
                time axis: after the compare step, for each category, if the
                working values do worse on that category's HEARD sentences
                than the past snapshot that did best on them, only that
                category's words (their rows of the vector table and their
                OUTGOING rows of the connection table) are restored from that
                snapshot, in the working copy and in the stored memory.
                (A first version also restored the columns -- the connections
                other words have TO this category. Those include what later
                categories learned against these words, so restoring them
                hurt the category being learned right now: 0.83 at age 0.)
                (multi failed because it judged on everything at once, while
                the disturbance happens category by category.)
External logic: design A. Conditions: none / vec / hebb / dhebb.
Restore: last (single stored copy, as before) / multi (best past snapshot) /
cat-best (per category, from the snapshot that did best on it).
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
from run_role2_target import predict
from run_role1_anchor import make_internal2

CONDITIONS = ["none", "vec", "hebb", "dhebb"]
RESTORES = ["last", "multi", "cat-best"]
CYCLES, CYCLE_EPOCHS = 4, 50


def run(condition, restore, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = ValueOnly(start, seed)
    internal2 = make_internal2(condition, memory, seed)
    stored = {"z_work": start.clone(), "assoc": torch.zeros_like(model.assoc)}
    rng = np.random.default_rng(20_000 + seed)
    order = [CATEGORIES[i] for i in rng.permutation(len(CATEGORIES))]
    schedule = [("session", c) for c in order] + [("cycle", None)] * CYCLES
    history = []            # (step, z, assoc) after each step
    learned_at = {}         # word -> step it was first learned
    cat_step = {}           # category -> step it was learned
    seen_cats, seen_words, steps = [], [], []

    def practise(ctx, tgt, ids, epochs):
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(ctx), tgt) + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()

    for i, (kind, cat) in enumerate(schedule):
        if kind == "session":
            old_cats = list(seen_cats)
            seen_cats.append(cat)
            cat_step[cat] = i
            new_words = list(split[cat]["train"])
            for w in new_words:
                learned_at.setdefault(w, i)
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

        # compare step: working tables vs the stored copy (last) or every past snapshot (multi)
        judge = masked_items([s for c in seen_cats for s in split[c]["sentences"]])[:2]
        for ti, name in enumerate(["z_work", "assoc"]):
            param = getattr(model, name)
            candidates = [stored[name]] if restore == "last" else [stored[name]] + [h[1 + ti] for h in history]
            scores = [score_with(model, name, c, judge) for c in candidates]
            working = score_with(model, name, param.detach().clone(), judge)
            best = int(np.argmax(scores))
            if working > scores[best]:
                stored[name] = param.detach().clone()
            elif working < scores[best]:
                stored[name] = candidates[best].clone()
                with torch.no_grad():
                    param.copy_(stored[name])
        if restore == "cat-best" and history:
            for c in seen_cats:
                hx, hy = partner_items(split[c]["sentences"], partner)
                words = torch.tensor(split[c]["train"])
                with torch.no_grad():
                    best_score, best_h = acc(model.complete(hx), hy), None
                    for h in history:
                        if h[0] < cat_step[c]:
                            continue
                        z_save, a_save = model.z_work.detach().clone(), model.assoc.detach().clone()
                        model.z_work[words] = h[1][words]
                        model.assoc[words, :] = h[2][words, :]
                        score = acc(model.complete(hx), hy)
                        model.z_work.copy_(z_save)
                        model.assoc.copy_(a_save)
                        if score > best_score:
                            best_score, best_h = score, h
                    if best_h is not None:
                        for target in (model.z_work, stored["z_work"]):
                            target[words] = best_h[1][words]
                        for target in (model.assoc, stored["assoc"]):
                            target[words, :] = best_h[2][words, :]
        history.append((i, stored["z_work"].clone(), stored["assoc"].clone()))

        # measurements per category, by age
        per_cat = {}
        for c in seen_cats:
            era = cat_step[c]
            ez, ea = history[era][1], history[era][2]
            hx, hy = partner_items(split[c]["sentences"], partner)
            with torch.no_grad():
                past = [acc(predict(h[1], h[2], hx), hy) for h in history if h[0] >= era]
            # among equally good past states, the most recent one (the more mature knowledge)
            bh = history[era + len(past) - 1 - int(np.argmax(past[::-1]))]
            row = {}
            for label, sents in [("heard", split[c]["sentences"]), ("never-heard", split[c]["test_sents"])]:
                x, y = partner_items(sents, partner)
                with torch.no_grad():
                    row[("current", label)] = acc(predict(stored["z_work"], stored["assoc"], x), y)
                    row[("era", label)] = acc(predict(ez, ea, x), y)
                    row[("best-era", label)] = acc(predict(bh[1], bh[2], x), y)
                    row[("external", label)] = acc(model.complete(x), y)
            per_cat[c] = (i - era, row)
        steps.append(per_cat)
    return steps


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(c, r): run(c, r, data, memory, seed) for c in CONDITIONS for r in RESTORES}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_steps = len(results[0][(CONDITIONS[0], RESTORES[0])])
    max_age = n_steps - 1

    def by_age(c, r, who, label, age):
        vals = []
        for res in results:
            for st in res[(c, r)]:
                for cat, (a, row) in st.items():
                    if a == age:
                        vals.append(row[(who, label)])
        return np.mean(vals) if vals else np.nan

    print(f"=== {n_seeds} seeds; 6 sessions (one new category each) + {CYCLES} everyday cycles ===")
    print("accuracy of a category's partner items by its AGE (steps since it was learned), averaged over categories/steps")
    for label in ["heard", "never-heard"]:
        print(f"\n--- {label} ---")
        print("age:                          " + " ".join(f"{a:>5}" for a in range(max_age + 1)))
        for c in CONDITIONS:
            for r in RESTORES:
                for who in ["current", "era", "best-era", "external"]:
                    row = [by_age(c, r, who, label, a) for a in range(max_age + 1)]
                    print(f"{c:6} {r:5} {who:9}      " + " ".join(f"{v:5.3f}" for v in row))

    print("\n--- middle of life (right after the 6th session, older categories only): recall vs current ---")
    for c in CONDITIONS:
        for label in ["heard", "never-heard"]:
            cells = []
            for who in ["era", "best-era"]:
                d = np.array([np.mean([row[(who, label)] - row[("current", label)]
                                       for _, (a, row) in [(k, v) for k, v in res[(c, "last")][5].items()] if a > 0])
                              for res in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                cells.append(f"{who}-current {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.2g})")
            print(f"{c:6} {label:11} " + "  ".join(cells))

    print("\n--- middle of life (after the 6th session, older categories): external logic, cat-best vs last ---")
    for c in CONDITIONS:
        for label in ["heard", "never-heard"]:
            d = np.array([np.mean([row[("external", label)] for _, (a, row) in res[(c, "cat-best")][5].items() if a > 0]) -
                          np.mean([row[("external", label)] for _, (a, row) in res[(c, "last")][5].items() if a > 0])
                          for res in results])
            n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
            print(f"{c:6} {label:11} {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.2g})")

    print("\n--- end of life, all categories: era recall vs current memory, and multi vs last restore ---")
    for c in CONDITIONS:
        for label in ["heard", "never-heard"]:
            cells = []
            for r in RESTORES:
                for who in ["era", "best-era"]:
                    d = np.array([np.mean([row[(who, label)] - row[("current", label)] for _, row in res[(c, r)][-1].values()])
                                  for res in results])
                    n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                    cells.append(f"{r}: {who}-current {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.2g})")
            for r in ["multi", "cat-best"]:
                d = np.array([np.mean([row[("external", label)] for _, row in res[(c, r)][-1].values()]) -
                              np.mean([row[("external", label)] for _, row in res[(c, "last")][-1].values()]) for res in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                cells.append(f"external {r}-last {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.2g})")
            print(f"{c:6} {label:11} " + "  ".join(cells))


if __name__ == "__main__":
    main()
