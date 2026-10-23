"""Role 1 of internal data: the basis for understanding NEW input.

"Knowing mom, then learning dad": a new word that always comes with a word
the learner already knows. Does what is already there (in the internal
logics, and in the external logic's values from earlier sessions) make the
new word attach faster than when both words of a pair are new?

Data: as findings 26-30 in session 1. In sessions 2 and 3, each new
category keeps one ordinary pair (both words new -- "within" pair) and
replaces its second pair by a CROSS pair: one new word + one already-known
word, a partner-less word of a category learned in an earlier session. The
two are heard together in this category's sentences from then on. The spare
new word of the category is only used for word classification. Of the 6
possible sentences per pair: 4 heard, 1 validation, 1 test (as before).

Items, sessions 2 and 3 (hide one word, the other three are given):
  within          -- both partners new
  cross old->new  -- the known word is given, the new one is hidden
  cross new->old  -- the new word is given, the known one is hidden
each on heard sentences (curve while learning) and on the never-heard test
sentences.
External logic: design A, restore off (persist style). Conditions:
none / vec / hebb / vec+hebb / vec+bert.
"""
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from run_unified_continual import CATEGORIES, WORD_NAMES, STAGES, RAW_DIM, N_CLASSES, sign_p
from run_seed_transfer import memorize, EPOCHS, CHECK_EVERY, REPLAY
from run_connection_seed import Hebbian, DirectionalHebbian, MiniBert, masked_items, acc, curve, SENT_LEN, NOISE
from run_three_designs import ValueOnly

N_TRAIN, N_VAL, N_TEST = 8, 2, 2
N_HEARD, N_VALSENT = 4, 1
CONDITIONS = ["none", "vec", "hebb", "vec+hebb", "vec+bert"]
KINDS = ["within", "cross old->new", "cross new->old"]


def build_data(seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, (N_CLASSES, RAW_DIM))
    vecs, cats, ids_by_cat = [], [], {}
    wid = 0
    for c, cat in enumerate(CATEGORIES):
        ids = []
        for _ in WORD_NAMES[cat]:
            vecs.append(np.clip(centers[c] + rng.normal(0, NOISE, RAW_DIM), 0, 1))
            cats.append(c)
            ids.append(wid)
            wid += 1
        ids_by_cat[cat] = [int(i) for i in rng.permutation(ids)]

    split, partner, kind = {}, {}, {}
    known_fillers = []
    for stage_i, stage in enumerate(STAGES):
        stage_fillers = []
        for cat in stage:
            ids = ids_by_cat[cat]
            train = ids[:N_TRAIN]
            if stage_i == 0:
                pairs = [(train[0], train[1]), (train[2], train[3])]
                fillers = train[4:8]
            else:
                old = known_fillers.pop(int(rng.integers(len(known_fillers))))
                pairs = [(train[0], train[1]), (train[2], old)]  # second pair: new word + known word
                fillers = train[3:7]                              # train[7] only used for classification
            for a, b in pairs:
                partner[a], partner[b] = b, a
            kind[pairs[0]] = "within"
            if stage_i > 0:
                kind[pairs[1]] = "cross"
            combos = [(fillers[i], fillers[j]) for i in range(len(fillers)) for j in range(i + 1, len(fillers))]
            heard, val, test = [], [], []
            for p in pairs:
                for k, ci in enumerate(rng.permutation(len(combos))):
                    s = list(p) + list(combos[ci])
                    s = [s[i] for i in rng.permutation(SENT_LEN)]
                    (heard if k < N_HEARD else val if k < N_HEARD + N_VALSENT else test).append(s)
            split[cat] = {"train": train, "val": ids[N_TRAIN:N_TRAIN + N_VAL], "test": ids[N_TRAIN + N_VAL:],
                          "sentences": heard, "val_sents": val, "test_sents": test, "pairs": pairs,
                          "fillers": fillers}
            stage_fillers += fillers
        known_fillers += stage_fillers
    return torch.tensor(np.array(vecs), dtype=torch.float32), torch.tensor(cats), split, partner


def kind_items(sentences, pairs):
    """Partner items of the given pairs, split into within / cross old->new / cross new->old.
    For a cross pair (new, old): hiding `new` is old->new, hiding `old` is new->old."""
    out = {k: ([], []) for k in KINDS}
    for pi, (a, b) in enumerate(pairs):
        for s in sentences:
            if a in s and b in s:
                for hidden in (a, b):
                    ctx = [w for w in s if w != hidden]
                    if pi == 0:
                        key = "within"
                    else:
                        key = "cross old->new" if hidden == a else "cross new->old"
                    out[key][0].append(ctx)
                    out[key][1].append(hidden)
    return {k: (torch.tensor(x), torch.tensor(y)) for k, (x, y) in out.items() if x}


def make_internal2(condition, memory, seed):
    if "dhebb" in condition:
        return DirectionalHebbian()
    if "hebb" in condition:
        return Hebbian()
    if "bert" in condition:
        return MiniBert(memory, seed)
    return None


def run(condition, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = ValueOnly(start, seed)
    internal2 = make_internal2(condition, memory, seed)
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
            with torch.no_grad():
                model.assoc[nw, :] = conn[nw, :]
                model.assoc[:, nw] = conn[:, nw]

        pairs = [p for c in new_cats for p in split[c]["pairs"]]
        heard_items, test_items = {}, {}
        for c in new_cats:
            for k, v in kind_items(split[c]["sentences"], split[c]["pairs"]).items():
                heard_items.setdefault(k, []).append(v)
            for k, v in kind_items(split[c]["test_sents"], split[c]["pairs"]).items():
                test_items.setdefault(k, []).append(v)
        heard_items = {k: (torch.cat([x for x, _ in v]), torch.cat([y for _, y in v])) for k, v in heard_items.items()}
        test_items = {k: (torch.cat([x for x, _ in v]), torch.cat([y for _, y in v])) for k, v in test_items.items()}

        train_ids = torch.tensor(new_words + replay_words)
        ctx, tgt, _, _ = masked_items(heard)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        curves = {("heard", k): [] for k in heard_items}
        curves.update({("test", k): [] for k in test_items})
        for epoch in range(EPOCHS + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    for k, (x, y) in heard_items.items():
                        curves[("heard", k)].append(acc(model.complete(x), y))
                    for k, (x, y) in test_items.items():
                        curves[("test", k)].append(acc(model.complete(x), y))
            if epoch == EPOCHS:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()
        sessions.append({key: curve(c) | {"end": c[-1]} for key, c in curves.items()})
    return sessions


def one_seed(seed, conditions=CONDITIONS):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {c: run(c, data, memory, seed) for c in conditions}


def main(conditions=CONDITIONS):
    CONDITIONS = conditions
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(partial(one_seed, conditions=conditions), range(n_seeds))

    def m(vals):
        return f"{np.mean(np.array(vals, dtype=float)):.3f}"

    def paired(get_a, get_b):
        d = np.array([get_a(r) - get_b(r) for r in results])
        n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        return f"{d.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds, design A, restore off ===")
    metrics = ["at0", "at10", "at25", "checks_to_90", "end"]
    for s in range(1, len(STAGES)):
        for where in ["heard", "test"]:
            print(f"\n--- session {s+1}, {where} sentences ---")
            for k in KINDS:
                for mt in metrics:
                    print(f"{k:15} {mt:12} " + "  ".join(
                        f"{c}={m([r[c][s][(where, k)][mt] for r in results])}" for c in CONDITIONS))
            print("condition vs none (at0 / at10 / checks_to_90, * p<0.01; for checks lower is better):")
            for k in KINDS:
                for c in CONDITIONS[1:]:
                    cells = [paired(lambda r, mt=mt: r[c][s][(where, k)][mt], lambda r, mt=mt: r["none"][s][(where, k)][mt])
                             for mt in ["at0", "at10", "checks_to_90"]]
                    print(f"  {k:15} {c:9} " + "  ".join(cells))
            print("cross vs within, same condition (at0 / at10 / checks_to_90):")
            for k in KINDS[1:]:
                for c in CONDITIONS:
                    cells = [paired(lambda r, mt=mt: r[c][s][(where, k)][mt], lambda r, mt=mt: r[c][s][(where, "within")][mt])
                             for mt in ["at0", "at10", "checks_to_90"]]
                    print(f"  {k:15} {c:9} " + "  ".join(cells))


if __name__ == "__main__":
    main()
