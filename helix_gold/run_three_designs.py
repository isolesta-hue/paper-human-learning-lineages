"""Three ways for the external logic to use the internal seeds -- does the
internal logic INFLUENCE learning in any of them?

Question (project decision, 2026-09-22): not final accuracy, and not how much
the external logic changes the seeds (in humans, outside learning is
expected to dominate). Only: does the internal logic help anywhere, and
where? Inside each design, conditions differ ONLY in what the internal
logics hand over, and are compared against "none" on the same data.

Designs (from finding 26):
  A (가) value-only   -- no sentence head. The sentence answer is made only
                         from the two value tables: connection table + the
                         word vectors themselves (similarity of the context
                         to every word). All the external logic learns for
                         sentences is a change of values, so the original
                         design items 10-12 (compare -> update / restore)
                         apply table by table, as in finding 26.
  B (나) whole snapshot -- the external logic of findings 25-26 (with a
                         sentence head). Items 10-12 applied to the WHOLE
                         external logic: a snapshot is stored at the start
                         of each session (after new words are seeded); at the
                         end, the trained one is kept if it is at least as
                         good on validation sentences, otherwise restored.
  C (다) base + correction -- the external logic of findings 25-26, but the
                         connection table it uses is internal logic 2's
                         CURRENT connections (refreshed every session, never
                         written by the external logic) plus a correction
                         table that the external logic learns from zero.
                         No compare/restore.

Conditions inside every design:
  none      -- random word vectors, no connection seed
  vec       -- Module A memory, no connection seed
  hebb      -- random word vectors, Hebbian connection seed
  vec+hebb  -- both seeds
  vec+bert  -- Module A memory + BERT connection seed

Data: as finding 26 (partner pairs; never-heard sentences split into
validation and test).
"""
import copy
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_unified_continual import STAGES, N_CLASSES, V, sign_p
from run_seed_transfer import memorize, placement, LATENT, HIDDEN, EPOCHS, CHECK_EVERY, REPLAY
from run_connection_seed import Hebbian, MiniBert, External, masked_items, partner_items, acc, curve
from run_ratchet_seed import build_data, score_with

DESIGNS = ["A", "B", "C"]
CONDITIONS = ["none", "vec", "hebb", "vec+hebb", "vec+bert"]


class ValueOnly(nn.Module):
    """Design A: the sentence answer comes only from the two value tables."""

    def __init__(self, start_vectors, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.z_work = nn.Parameter(start_vectors.clone())
        self.assoc = nn.Parameter(torch.zeros(V, V))
        self.cls_head = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, N_CLASSES))

    def embed(self, ids):
        return self.z_work[ids]

    def classify(self, ids):
        return self.cls_head(self.z_work[ids])

    def complete(self, ctx):
        # cosine, so the size of the vectors cannot drown the connection table
        # (a first version used dot/sqrt(LATENT): memory vectors have norm ~18,
        # the similarity term reached 20-100 while connections are ~1, and every
        # condition just answered with a word already in the sentence)
        similarity = F.normalize(self.z_work[ctx].mean(dim=1), dim=-1) @ F.normalize(self.z_work, dim=-1).T
        return similarity + self.assoc[ctx].mean(dim=1)


class BasePlusCorrection(External):
    """Design C: internal connections as a fixed base, external learns a correction."""

    def __init__(self, start_vectors, seed):
        super().__init__(start_vectors, seed)
        self.register_buffer("base", torch.zeros(V, V))

    def complete(self, ctx):
        return self.sent_head(self.z_work[ctx].mean(dim=1)) + (self.base + self.assoc)[ctx].mean(dim=1)


def run(design, condition, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = {"A": ValueOnly, "B": External, "C": BasePlusCorrection}[design](start, seed)
    internal2 = None
    if "hebb" in condition:
        internal2 = Hebbian()
    elif "bert" in condition:
        internal2 = MiniBert(memory, seed)
    stored = {"z_work": start.clone(), "assoc": torch.zeros(V, V)}
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
            if design == "C":
                model.base.copy_(conn)
            else:
                nw = torch.tensor(new_words)
                stored["assoc"][nw, :] = conn[nw, :]
                stored["assoc"][:, nw] = conn[:, nw]
                with torch.no_grad():
                    model.assoc[nw, :] = conn[nw, :]
                    model.assoc[:, nw] = conn[:, nw]
        snapshot = copy.deepcopy(model.state_dict()) if design == "B" else None

        train_ids = torch.tensor(new_words + replay_words)
        new_ids = torch.tensor(new_words)
        ctx, tgt, _, _ = masked_items(heard)
        new_ctx, new_tgt, _, _ = masked_items(new_sents)
        new_test = partner_items([s for c in new_cats for s in split[c]["test_sents"]], partner)

        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        sent_curve, test_curve, cls_curve = [], [], []
        for epoch in range(EPOCHS + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    sent_curve.append(acc(model.complete(new_ctx), new_tgt))
                    test_curve.append(acc(model.complete(new_test[0]), new_test[1]))
                    cls_curve.append(acc(model.classify(new_ids), cats[new_ids]))
            if epoch == EPOCHS:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()

        val_items = partner_items([s for c in seen_cats for s in split[c]["val_sents"]], partner)
        decisions = {}
        if design == "A":
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
        elif design == "B":
            with torch.no_grad():
                working = acc(model.complete(val_items[0]), val_items[1])
                trained_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(snapshot)
                kept = acc(model.complete(val_items[0]), val_items[1])
                if working >= kept:
                    model.load_state_dict(trained_state)
                    decisions["whole"] = "update" if working > kept else "tie"
                else:
                    decisions["whole"] = "restore"

        trained = torch.tensor(seen_words)
        unseen = torch.tensor([w for c in seen_cats for w in split[c]["val"] + split[c]["test"]])
        all_test = partner_items([s for c in seen_cats for s in split[c]["test_sents"]], partner)
        with torch.no_grad():
            sessions.append({
                "test_partner": acc(model.complete(all_test[0]), all_test[1]),
                "cls_unseen": acc(model.classify(unseen), cats[unseen]),
                "placement": placement(model, trained, cats[trained]),
                "sent_curve": curve(sent_curve),
                "test_curve": curve(test_curve),
                "cls_curve": curve(cls_curve),
                "decisions": decisions,
            })
    return sessions


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(d, c): run(d, c, data, memory, seed) for d in DESIGNS for c in CONDITIONS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))

    def ms(vals):
        vals = np.array(vals, dtype=float)
        return f"{vals.mean():.3f}"

    def paired(d, cond, get):
        diff = np.array([get(r[(d, cond)]) - get(r[(d, "none")]) for r in results])
        n_pos, n_neg = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
        return diff.mean(), n_pos, n_neg, sign_p(n_pos, n_neg)

    print(f"=== {n_seeds} seeds, {EPOCHS} epochs/session, replay {REPLAY:.0%} ===")
    print("means; in [] = paired vs 'none' in the same design: better/worse, * p<0.01")

    metrics = []
    for s in range(len(STAGES)):
        for task, label in [("test_curve", "never-heard partner"), ("sent_curve", "heard sentences"),
                            ("cls_curve", "new-word category")]:
            for m in ["at0", "at10", "at25", "early_slope", "checks_to_90"]:
                metrics.append((f"s{s+1} {label} {m}", lambda x, s=s, t=task, m=m: x[s][t][m], m == "checks_to_90"))
    for key, label in [("test_partner", "final never-heard partner"), ("cls_unseen", "final never-trained word category"),
                       ("placement", "final placement ratio")]:
        metrics.append((label, lambda x, k=key: x[-1][k], False))

    for d in DESIGNS:
        print(f"\n##### design {d} #####")
        for label, get, lower_better in metrics:
            cells = [f"none={ms([get(r[(d, 'none')]) for r in results])}"]
            for c in CONDITIONS[1:]:
                mean = ms([get(r[(d, c)]) for r in results])
                _, n_pos, n_neg, p = paired(d, c, get)
                if lower_better:
                    n_pos, n_neg = n_neg, n_pos
                star = "*" if p < 0.01 else " "
                cells.append(f"{c}={mean}[{n_pos}/{n_neg}{star}]")
            print(f"{label:42} " + "  ".join(cells))
        if d in ("A", "B"):
            print("compare -> update/restore/tie decisions per session:")
            names = ["z_work", "assoc"] if d == "A" else ["whole"]
            for c in CONDITIONS:
                for name in names:
                    cells = []
                    for s in range(len(STAGES)):
                        dec = [r[(d, c)][s]["decisions"][name] for r in results]
                        cells.append(f"s{s+1} {dec.count('update')}/{dec.count('restore')}/{dec.count('tie')}")
                    print(f"  {c:9} {name:7} " + "  ".join(cells))


if __name__ == "__main__":
    main()
