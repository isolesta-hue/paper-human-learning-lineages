"""Original design items 10-12: compare, then update or restore.

From the project's founding design (README, "처음 설계 원문"):
  10. when learning ends, compare the stored values with the values the
      external logic modified.
  11. if the modified values give higher output accuracy, update the stored
      values with them.
  12. if lower, keep the stored values as the backup and restore the
      modified values to them.

Finding 25 showed why this matters: a Hebbian connection seed alone recalls
partners in never-heard sentences 100% of the time, but after the external
logic trains on it the recall falls to 0.71 -- below starting from nothing.

Two stored tables, each compared on its own at the end of every session:
  vectors     -- starts as Module A's memory (internal logic 1)
  connections -- starts as internal logic 2's connection seed (zeros if
                 there is none)
Comparison: the external logic as it is, but with ONE table swapped for the
stored one, scored on VALIDATION sentences (never heard, never used in the
final score). Strictly higher -> stored := working (update). Strictly lower
-> working := stored (restore). Equal -> leave both.
New words each session get their stored rows from the internal logics; old
words keep whatever the stored tables hold, including earlier updates --
this is the loop where the external result flows back into the next start.

Data: exactly as finding 25 (2 partner pairs + 4 words without a partner per
category, C(4,2)=6 possible sentences per pair, 4 heard), except the 2
never-heard sentences per pair are split: 1 validation, 1 test. (A version
with 5 unpartnered words, 10 sentences per pair, was tried first; there the
Hebbian seed alone only reached ~0.68 instead of 1.00, which removes the
premise of this test, so the finding-25 data was kept.)
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from run_unified_continual import CATEGORIES, WORD_NAMES, STAGES, RAW_DIM, N_CLASSES, sign_p
from run_seed_transfer import memorize, placement, EPOCHS, CHECK_EVERY, REPLAY
from run_connection_seed import (Hebbian, MiniBert, External, masked_items, partner_items, acc, curve,
                                 SENT_LEN, NOISE)

torch.set_num_threads(1)

N_TRAIN, N_VAL, N_TEST = 8, 2, 2
N_PAIRS = 2
N_HEARD, N_VALSENT = 4, 1  # per pair; the remaining 1 of 6 is test
CONDITIONS = ["random", "vec", "vec+R", "vec+hebb", "vec+hebb+R", "vec+bert", "vec+bert+R"]


def build_data(seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, (N_CLASSES, RAW_DIM))
    vecs, cats, split, partner = [], [], {}, {}
    wid = 0
    for c, cat in enumerate(CATEGORIES):
        ids = []
        for _ in WORD_NAMES[cat]:
            vecs.append(np.clip(centers[c] + rng.normal(0, NOISE, RAW_DIM), 0, 1))
            cats.append(c)
            ids.append(wid)
            wid += 1
        ids = [int(i) for i in rng.permutation(ids)]
        train = ids[:N_TRAIN]
        pairs = [(train[2 * i], train[2 * i + 1]) for i in range(N_PAIRS)]
        fillers = train[2 * N_PAIRS:]
        for a, b in pairs:
            partner[a], partner[b] = b, a
        combos = [(fillers[i], fillers[j]) for i in range(len(fillers)) for j in range(i + 1, len(fillers))]
        heard, val, test = [], [], []
        for p in pairs:
            for k, ci in enumerate(rng.permutation(len(combos))):
                s = list(p) + list(combos[ci])
                s = [s[i] for i in rng.permutation(SENT_LEN)]
                (heard if k < N_HEARD else val if k < N_HEARD + N_VALSENT else test).append(s)
        split[cat] = {"train": train, "val": ids[N_TRAIN:N_TRAIN + N_VAL], "test": ids[N_TRAIN + N_VAL:],
                      "sentences": heard, "val_sents": val, "test_sents": test}
    return torch.tensor(np.array(vecs), dtype=torch.float32), torch.tensor(cats), split, partner


def score_with(model, name, table, items):
    """Validation recall with one of the external logic's tables swapped out."""
    param = getattr(model, name)
    with torch.no_grad():
        saved = param.detach().clone()
        param.copy_(table)
        score = acc(model.complete(items[0]), items[1])
        param.copy_(saved)
    return score


def run(condition, data, memory, seed):
    vecs, cats, split, partner = data
    ratchet = condition.endswith("+R")
    base = condition[:-2] if ratchet else condition
    start = memory if base != "random" else torch.randn_like(memory) * memory.std()
    model = External(start, seed)
    internal2 = {"vec+hebb": Hebbian, "vec+bert": lambda: MiniBert(memory, seed)}.get(base)
    internal2 = internal2() if internal2 else None
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

        train_ids = torch.tensor(new_words + replay_words)
        ctx, tgt, _, _ = masked_items(heard)
        new_ctx, new_tgt, _, _ = masked_items(new_sents)
        new_test = partner_items([s for c in new_cats for s in split[c]["test_sents"]], partner)

        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        sent_curve, test_curve = [], []
        for epoch in range(EPOCHS + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    sent_curve.append(acc(model.complete(new_ctx), new_tgt))
                    test_curve.append(acc(model.complete(new_test[0]), new_test[1]))
            if epoch == EPOCHS:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()

        all_test = partner_items([s for c in seen_cats for s in split[c]["test_sents"]], partner)
        with torch.no_grad():
            before_ratchet = acc(model.complete(all_test[0]), all_test[1])
            # where is the knowledge? test recall of each part on its own (reporting only)
            breakdown = {
                "stored conn alone": acc(stored["assoc"][all_test[0]].mean(1), all_test[1]),
                "working conn alone": acc(model.assoc[all_test[0]].mean(1), all_test[1]),
                "head alone": score_with(model, "assoc", torch.zeros_like(model.assoc), all_test),
                "full, stored conn in": score_with(model, "assoc", stored["assoc"], all_test),
            }

        decisions = {}
        if ratchet:
            val_items = partner_items([s for c in seen_cats for s in split[c]["val_sents"]], partner)
            for name in ["z_work", "assoc"]:
                param = getattr(model, name)
                working = score_with(model, name, param.detach().clone(), val_items)
                kept = score_with(model, name, stored[name], val_items)
                if working > kept:
                    stored[name] = param.detach().clone()
                    decisions[name] = "update"
                elif working < kept:
                    with torch.no_grad():
                        param.copy_(stored[name])
                    decisions[name] = "restore"
                else:
                    decisions[name] = "tie"

        trained = torch.tensor(seen_words)
        unseen = torch.tensor([w for c in seen_cats for w in split[c]["val"] + split[c]["test"]])
        with torch.no_grad():
            sessions.append({
                "test_partner": acc(model.complete(all_test[0]), all_test[1]),
                "test_partner_before": before_ratchet,
                "cls_unseen": acc(model.classify(unseen), cats[unseen]),
                "placement": placement(model, trained, cats[trained]),
                "sent_curve": curve(sent_curve),
                "test_curve": curve(test_curve),
                "decisions": decisions,
                "breakdown": breakdown,
            })
    return sessions


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    results = []
    for seed in range(n_seeds):
        data = build_data(seed)
        memory, _ = memorize(data[0], seed)
        results.append({c: run(c, data, memory, seed) for c in CONDITIONS})

    def ms(vals):
        vals = np.array(vals, dtype=float)
        return f"{vals.mean():.3f}+/-{vals.std():.3f}"

    print(f"=== {n_seeds} seeds, {EPOCHS} epochs/session, replay {REPLAY:.0%} ===")
    print("\n--- test partner recall (never-heard, never used for any decision), end of each session ---")
    for c in CONDITIONS:
        cells = []
        for s in range(len(STAGES)):
            cells.append(f"s{s+1} before={ms([r[c][s]['test_partner_before'] for r in results])} "
                         f"after={ms([r[c][s]['test_partner'] for r in results])}")
        print(f"{c:11} " + "  |  ".join(cells))

    print("\n--- where the knowledge sits: test recall of each part alone, before any ratchet decision ---")
    for c in ["vec", "vec+hebb", "vec+bert"]:
        for part in ["stored conn alone", "working conn alone", "head alone", "full, stored conn in"]:
            print(f"{c:9} {part:22} " + "  ".join(
                f"s{s+1}={ms([r[c][s]['breakdown'][part] for r in results])}" for s in range(len(STAGES))))

    print("\n--- after the final session ---")
    for key, label in [("cls_unseen", "category acc, never-trained words"),
                       ("placement", "placement between/within ratio")]:
        print(f"{label:34} " + "  ".join(f"{c}={ms([r[c][-1][key] for r in results])}" for c in CONDITIONS))

    print("\n--- ratchet decisions (update / restore / tie), per session ---")
    for c in [c for c in CONDITIONS if c.endswith("+R")]:
        for name in ["z_work", "assoc"]:
            cells = []
            for s in range(len(STAGES)):
                d = [r[c][s]["decisions"][name] for r in results]
                cells.append(f"s{s+1} {d.count('update')}/{d.count('restore')}/{d.count('tie')}")
            print(f"{c:11} {'vectors' if name == 'z_work' else 'connections':11} " + "  ".join(cells))

    print("\n--- speed while learning each session's new sentences ---")
    for task, label in [("test_curve", "partner recall, test sentences"), ("sent_curve", "heard sentences")]:
        for s in range(len(STAGES)):
            for m in ["at0", "at10", "at25", "checks_to_90"]:
                print(f"session{s+1} {label:31} {m:12} " +
                      "  ".join(f"{c}={ms([r[c][s][task][m] for r in results])}" for c in CONDITIONS))

    print("\n--- paired comparisons ---")
    pairs = [("vec+hebb+R", "vec+hebb"), ("vec+bert+R", "vec+bert"), ("vec+R", "vec"),
             ("vec+hebb+R", "random"), ("vec+hebb", "random")]
    for a, b in pairs:
        for s in range(len(STAGES)):
            for key in ["test_partner"] + ([("test_curve", "at10")] if s > 0 else []):
                def get(r, c):
                    return r[c][s][key[0]][key[1]] if isinstance(key, tuple) else r[c][s][key]
                d = np.array([get(r, a) - get(r, b) for r in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                kname = ".".join(key) if isinstance(key, tuple) else key
                print(f"{a:11} vs {b:9} session{s+1} {kname:18} diff={d.mean():+.3f}  "
                      f"better/worse={n_pos}/{n_neg}  p={sign_p(n_pos, n_neg):.3g}")


if __name__ == "__main__":
    main()
