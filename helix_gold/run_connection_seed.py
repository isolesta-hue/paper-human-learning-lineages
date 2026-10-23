"""Internal logic 2: a CONNECTION seed, built with no external feedback.

Roles (project decision, 2026-09-22):
  internal 1 (Module A) -- memorizes each word as a vector (vector seed).
  internal 2            -- in its own separate space, and with no external
                           feedback, works out which remembered words go
                           together (connection seed). Reads Module A's
                           memory, never modifies it.
  external (Module C)   -- starts from both seeds and corrects them by
                           feedback.

Two versions of internal 2, both feedback-free:
  hebb -- count which words appear in the same sentence, turn counts into
          positive pointwise mutual information. No learning at all.
  bert -- a one-layer self-attention model over Module A's vectors, trained
          by hiding one word of a heard sentence and guessing it from the
          rest ("input = answer", like BERT). Every word computes its own
          attention query, so it avoids the fixed-query flaw of the old
          Module B (finding 15). Its guesses given a single word become the
          connection values.

Data: the same 72-word vocabulary, but sentences now carry real structure.
Within each category, 4 of the 8 training words form 2 fixed partner pairs
("mom-dad"); the other 4 have no partner. Every sentence holds one pair plus
two of the unpartnered words. (A first version made all 8 words partners;
then "dad, x, y -> ?" had three equally valid answers -- mom, x's partner,
y's partner -- so the test could not be passed by anyone.) Each pair has
C(4,2)=6 possible sentences: 4 are heard, 2 are HELD OUT and never heard.
Connection is scored on the held-out ones: hide one partner, can the other
three words bring it back?

Conditions (identical except for the two seeds):
  random   -- random word vectors, no connection seed
  vec      -- Module A's memory, no connection seed (finding 24's "seed")
  vec+hebb -- Module A's memory + Hebbian connection seed
  vec+bert -- Module A's memory + BERT connection seed
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_unified_continual import CATEGORIES, WORD_NAMES, STAGES, RAW_DIM, N_CLASSES, V, sign_p
from run_seed_transfer import memorize, placement, LATENT, HIDDEN, LR, EPOCHS, CHECK_EVERY, REPLAY

torch.set_num_threads(1)

N_TRAIN, N_VAL, N_TEST = 8, 2, 2
NOISE = 0.2
SENT_LEN = 4
N_PAIRS = 2
TRAIN_SENT_PER_PAIR = 4  # of the 6 possible; the other 2 are held out
BERT_EPOCHS = 300
CONDITIONS = ["random", "vec", "vec+hebb", "vec+bert"]


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
        sents, held = [], []
        for p in pairs:
            order = rng.permutation(len(combos))
            for k, ci in enumerate(order):
                s = list(p) + list(combos[ci])
                s = [s[i] for i in rng.permutation(SENT_LEN)]
                (sents if k < TRAIN_SENT_PER_PAIR else held).append(s)
        split[cat] = {"train": train, "val": ids[N_TRAIN:N_TRAIN + N_VAL], "test": ids[N_TRAIN + N_VAL:],
                      "sentences": sents, "heldout": held}
    return torch.tensor(np.array(vecs), dtype=torch.float32), torch.tensor(cats), split, partner


def masked_items(sentences):
    """Every sentence once per position: (the other three words, hidden word)."""
    ctx, tgt, full, pos = [], [], [], []
    for s in sentences:
        for p in range(SENT_LEN):
            ctx.append([w for i, w in enumerate(s) if i != p])
            tgt.append(s[p])
            full.append(s)
            pos.append(p)
    return torch.tensor(ctx), torch.tensor(tgt), torch.tensor(full), torch.tensor(pos)


def partner_items(sentences, partner):
    """Held-out scoring: only the positions holding a partner-pair word."""
    ctx, tgt = [], []
    for s in sentences:
        for p, w in enumerate(s):
            if partner.get(w) in s:
                ctx.append([x for i, x in enumerate(s) if i != p])
                tgt.append(w)
    return torch.tensor(ctx), torch.tensor(tgt)


class Hebbian:
    def __init__(self):
        self.counts = torch.zeros(V, V)

    def observe(self, sentences):
        for s in sentences:
            for a in s:
                for b in s:
                    if a != b:
                        self.counts[a, b] += 1

    def connections(self, seen):
        c = self.counts
        total = c.sum()
        row = c.sum(1, keepdim=True)
        col = c.sum(0, keepdim=True)
        with torch.no_grad():
            pmi = torch.log(c * total / (row * col))
        pmi = torch.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
        mask = torch.zeros(V, dtype=torch.bool)
        mask[seen] = True
        return pmi * (mask[:, None] & mask[None, :])


class DirectionalHebbian(Hebbian):
    """Directional version (finding 31): based on P(b in the sentence | a in the
    sentence). Not normalised by how common b is, and not symmetric: "dad"
    (always heard with "mom") -> "mom" is P = 1.0, while "mom" (heard in many
    sentences) -> "dad" is only the share of mom's sentences that hold dad.
    Put on the same scale as the BERT seed: log P + log(number of known words),
    i.e. how much more likely than picking a known word at random, >= 0.
    (A first version used P itself, 0..1: next to the cosine term of design A,
    which favours the words already in the sentence, it was too small and even
    ordinary new pairs started at 0.22 instead of PPMI's 0.97.)"""

    def __init__(self):
        super().__init__()
        self.sentences_with = torch.zeros(V)

    def observe(self, sentences):
        super().observe(sentences)
        for s in sentences:
            for a in set(s):
                self.sentences_with[a] += 1

    def connections(self, seen):
        cond = self.counts / self.sentences_with.clamp(min=1)[:, None]
        conn = torch.where(cond > 0, torch.log(cond.clamp(min=1e-12)) + math.log(len(seen)), torch.zeros_like(cond))
        mask = torch.zeros(V, dtype=torch.bool)
        mask[seen] = True
        return conn.clamp(min=0) * (mask[:, None] & mask[None, :])


class MiniBert(nn.Module):
    def __init__(self, memory, seed):
        super().__init__()
        torch.manual_seed(30_000 + seed)
        self.memory = memory
        self.mask_vec = nn.Parameter(torch.zeros(LATENT))
        self.layer = nn.TransformerEncoderLayer(LATENT, nhead=2, dim_feedforward=HIDDEN, dropout=0.0, batch_first=True)
        self.out = nn.Linear(LATENT, V)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.005)

    def guess(self, full, pos):
        x = self.memory[full].clone()
        x[torch.arange(len(pos)), pos] = self.mask_vec
        h = self.layer(x)
        return self.out(h[torch.arange(len(pos)), pos])

    def observe(self, sentences):
        _, tgt, full, pos = masked_items(sentences)
        for _ in range(BERT_EPOCHS):
            self.opt.zero_grad()
            F.cross_entropy(self.guess(full, pos), tgt).backward()
            self.opt.step()

    def connections(self, seen):
        seen_t = torch.tensor(seen)
        full = torch.stack([seen_t, seen_t], dim=1)  # [word, hidden slot]
        with torch.no_grad():
            logits = self.guess(full, torch.ones(len(seen), dtype=torch.long))[:, seen_t]
            logp = F.log_softmax(logits, dim=1)
        conn = torch.zeros(V, V)
        conn[seen_t[:, None], seen_t[None, :]] = (logp + math.log(len(seen))).clamp(min=0)
        conn[seen_t, seen_t] = 0.0
        return conn


class External(nn.Module):
    def __init__(self, start_vectors, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.z_work = nn.Parameter(start_vectors.clone())
        self.assoc = nn.Parameter(torch.zeros(V, V))
        self.cls_head = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, N_CLASSES))
        self.sent_head = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, V))

    def embed(self, ids):
        return self.z_work[ids]

    def classify(self, ids):
        return self.cls_head(self.z_work[ids])

    def complete(self, ctx):
        return self.sent_head(self.z_work[ctx].mean(dim=1)) + self.assoc[ctx].mean(dim=1)


def acc(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()


def curve(values):
    arr = np.maximum.accumulate(np.array(values))
    target = arr[0] + 0.9 * (arr[-1] - arr[0])
    return {"at0": values[0], "at10": values[10 // CHECK_EVERY], "at25": values[25 // CHECK_EVERY],
            "early_slope": float(np.mean(np.diff(arr)[:3])),
            "checks_to_90": next(i for i, v in enumerate(arr) if v >= target - 1e-9)}


def run(condition, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition != "random" else torch.randn_like(memory) * memory.std()
    model = External(start, seed)
    internal2 = {"vec+hebb": Hebbian, "vec+bert": lambda: MiniBert(memory, seed)}.get(condition)
    internal2 = internal2() if internal2 else None
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

        train_ids = torch.tensor(new_words + replay_words)
        ctx, tgt, _, _ = masked_items(heard)
        new_ctx, new_tgt, _, _ = masked_items(new_sents)
        new_held = partner_items([s for c in new_cats for s in split[c]["heldout"]], partner)

        # the connection seed on its own, before the external logic touches it
        with torch.no_grad():
            seed_only = acc(model.assoc[new_held[0]].mean(dim=1), new_held[1]) if internal2 is not None else float("nan")

        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sent_curve, held_curve = [], []
        for epoch in range(EPOCHS + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    sent_curve.append(acc(model.complete(new_ctx), new_tgt))
                    held_curve.append(acc(model.complete(new_held[0]), new_held[1]))
            if epoch == EPOCHS:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()

        trained = torch.tensor(seen_words)
        unseen = torch.tensor([w for c in seen_cats for w in split[c]["val"] + split[c]["test"]])
        all_held = partner_items([s for c in seen_cats for s in split[c]["heldout"]], partner)
        with torch.no_grad():
            sessions.append({
                "heldout_partner": acc(model.complete(all_held[0]), all_held[1]),
                "cls_unseen": acc(model.classify(unseen), cats[unseen]),
                "placement": placement(model, trained, cats[trained]),
                "sent_curve": curve(sent_curve),
                "held_curve": curve(held_curve),
                "seed_only": seed_only,
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

    def row(fn):
        return "  ".join(f"{c}={ms([fn(r[c]) for r in results])}" for c in CONDITIONS)

    print(f"=== {n_seeds} seeds, {EPOCHS} epochs/session, replay {REPLAY:.0%} ===")
    print("\n--- after the final session ---")
    for key, label in [("heldout_partner", "partner recall, never-heard sentences"),
                       ("cls_unseen", "category acc, never-trained words"),
                       ("placement", "placement between/within ratio")]:
        print(f"{label:40} {row(lambda s: s[-1][key])}")

    print("\n--- connection seed ALONE on never-heard sentences, at the start of each session ---")
    for s in range(len(STAGES)):
        print(f"session{s+1} {row(lambda x: x[s]['seed_only'])}")

    print("\n--- each session, while learning its new sentences ---")
    for s in range(len(STAGES)):
        for task, label in [("held_curve", "partner recall on never-heard sentences"),
                            ("sent_curve", "heard sentences, all positions")]:
            for m in ["at0", "at10", "at25", "early_slope", "checks_to_90"]:
                print(f"session{s+1} {label:40} {m:12} {row(lambda x: x[s][task][m])}")

    print("\n--- paired vs 'vec' (Module A memory, no connection seed) ---")
    for cond in ["vec+hebb", "vec+bert", "random"]:
        for s in range(len(STAGES)):
            for task, m in [("held_curve", "at0"), ("held_curve", "at10"), ("sent_curve", "at10"), ("sent_curve", "at25")]:
                d = np.array([r[cond][s][task][m] - r["vec"][s][task][m] for r in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                print(f"{cond:9} session{s+1} {task}.{m:4} diff={d.mean():+.3f}  better/worse={n_pos}/{n_neg}  "
                      f"p={sign_p(n_pos, n_neg):.3g}")
        d = np.array([r[cond][-1]["heldout_partner"] - r["vec"][-1]["heldout_partner"] for r in results])
        n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        print(f"{cond:9} final heldout_partner     diff={d.mean():+.3f}  better/worse={n_pos}/{n_neg}  "
              f"p={sign_p(n_pos, n_neg):.3g}")


if __name__ == "__main__":
    main()
