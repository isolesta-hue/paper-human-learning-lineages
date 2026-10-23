"""Does starting from the internal memory's seed help the external logic?

Roles (project decision, 2026-09-22):
  internal logic 1 (Module A) -- memorizes each word as a vector against a
    frozen random decoder ("input = answer"), starting from the origin
    (the self anchor). No labels, no feedback. Supplies the VECTOR SEED.
  external logic (Module C) -- receives a WORKING COPY of those vectors and
    adjusts both the copy and its own heads by feedback (classification +
    sentence tasks). A's original memory is never touched.

The earlier code never actually passed A's memory to C (finding 23), so
this is the first real test of the seed hypothesis. Conditions, identical
except for where C's word vectors start:
  seed    -- working copy of Module A's memory
  random  -- random vectors with the same overall scale
  vanilla -- reference: an ordinary MLP reading the raw word features

Curriculum, baby-like: 3 sessions, 2 new categories each; while learning
new words, a fixed 25% of earlier words/sentences keep appearing.

Measured after each session, for everything seen so far:
  speed      -- accuracy on the words/sentences being taught, early in the
                session (at 10 and 25 epochs), early slope, checks to 90%
  placement  -- between-category / within-category distance ratio of the
                trained words' vectors (higher = better organized), and
                3-nearest-neighbour category purity of never-trained words
  connection -- masked-word sentence accuracy: given the other words of a
                sentence, does the missing one come out
  generalization -- category accuracy on words C never trained on
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_unified_continual import (
    CATEGORIES, WORD_NAMES, STAGES, SENT_LEN, RAW_DIM, N_CLASSES, V,
    build_data, sign_p,
)

torch.set_num_threads(1)

LATENT = 32
HIDDEN = 64
LR = 0.01
EPOCHS = 200
CHECK_EVERY = 5
REPLAY = 0.25
MEMORIZE_STEPS = 400
CONDITIONS = ["seed", "random", "vanilla"]


def memorize(vecs, seed):
    """Module A: for every word, find z so that the frozen random decoder
    reproduces the word's own features. Starts at the origin (self)."""
    g = torch.Generator().manual_seed(10_000 + seed)
    W = torch.randn(LATENT, RAW_DIM, generator=g) / math.sqrt(LATENT)
    z = torch.zeros(len(vecs), LATENT, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(MEMORIZE_STEPS):
        opt.zero_grad()
        loss = ((torch.sigmoid(z @ W) - vecs) ** 2).mean(dim=1).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = ((torch.sigmoid(z @ W) - vecs) ** 2).mean(dim=1)
    return z.detach(), mse


def memorize_rehearse(vecs, seed, check=None, rounds=10):
    """Module A with rehearsal (2026-10-17, readiness step 2 for the shared exam; memory-first rule 2 made a loop): memorise
    exactly as memorize() -- the same frozen decoder, the same start, the same optimiser -- then recall every element of
    `check` (default: all) and, while some is not recalled (the nearest element to its recalled features is another one)
    and the round recalled more than the round before, practise one more round of MEMORIZE_STEPS; at most `rounds` rounds.
    A child rehearses what it cannot yet recall; the stop is the learner's own recall check, not a score or a pass line.
    Why: on the exam's element table (about 4,000 elements, 8 x Dami's worlds) the first 400 steps recall 0.985 -- not a
    width limit (the same 32-dim memory recalls 0.9998 after 2,000 steps; diag_conv_elem2.out). If every element is
    recalled after the first round, the result is memorize()'s to the bit. Returns (z, mse, rounds used, recalled share)."""
    g = torch.Generator().manual_seed(10_000 + seed)
    W = torch.randn(LATENT, RAW_DIM, generator=g) / math.sqrt(LATENT)
    z = torch.zeros(len(vecs), LATENT, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    idx = torch.arange(len(vecs)) if check is None else torch.as_tensor(check)
    sq = (vecs ** 2).sum(1)
    best, used, share = -1, 0, 0.0
    for r in range(rounds):
        for _ in range(MEMORIZE_STEPS):
            opt.zero_grad()
            loss = ((torch.sigmoid(z @ W) - vecs) ** 2).mean(dim=1).sum()
            loss.backward()
            opt.step()
        used = r + 1
        with torch.no_grad():
            rec = torch.sigmoid(z[idx] @ W)
            d = (rec ** 2).sum(1, keepdim=True) + sq[None, :] - 2 * rec @ vecs.T
            ok = int((d.argmin(1) == idx).sum())
        share = ok / len(idx)
        if ok == len(idx) or ok <= best:
            break
        best = ok
    with torch.no_grad():
        mse = ((torch.sigmoid(z @ W) - vecs) ** 2).mean(dim=1)
    return z.detach(), mse, used, share


def memorize_linear(vecs, seed, dims, scale):
    """Module A for a channel whose features are coordinates (the "where" channel, finding 90): its own frozen decoder, linear
    and square -- the channel's features live in the first `dims` features (the rest are the centre 0.5), and the decoder is
    a fixed random rotation Q (dims = LATENT): features = 0.5 + (z / scale) Q, so memory holds a coordinate exactly and a
    straight line in the input stays straight. scale puts the channel's memories on the same size as the other memories (the
    caller passes their mean norm over this channel's; no free constant). Returns (z, decoder) with features = 0.5 + z decoder.
    History: the shared sigmoid decoder bends coordinates (template ceiling 58-90%); a linear non-square decoder shrinks recall
    toward the centre (recall -> itself 40-100%); a random square one is badly conditioned (memory norms up to 2600)."""
    g = torch.Generator().manual_seed(30_000 + seed)
    Q = torch.linalg.qr(torch.randn(dims, dims, generator=g))[0]
    z = (vecs[:, :dims] - 0.5) @ Q.T
    s = scale(z) if callable(scale) else scale
    return z * s, Q / s


class External(nn.Module):
    def __init__(self, condition, vecs, memory, seed):
        super().__init__()
        self.condition = condition
        self.vecs = vecs
        torch.manual_seed(seed)
        if condition == "vanilla":
            self.enc = nn.Sequential(nn.Linear(RAW_DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, LATENT), nn.Tanh())
        else:
            start = memory.clone() if condition == "seed" else torch.randn_like(memory) * memory.std()
            self.z_work = nn.Parameter(start)
        self.cls_head = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, N_CLASSES))
        self.sent_head = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, V))

    def embed(self, ids):
        if self.condition == "vanilla":
            return self.enc(self.vecs[ids])
        return self.z_work[ids]

    def classify(self, ids):
        return self.cls_head(self.embed(ids))

    def complete(self, ctx_ids):
        return self.sent_head(self.embed(ctx_ids).mean(dim=1))


def sentence_items(split, cats):
    ctx, tgt = [], []
    for cat in cats:
        for sent in split[cat]["sentences"]:
            for p in range(SENT_LEN):
                ctx.append([int(w) for i, w in enumerate(sent) if i != p])
                tgt.append(int(sent[p]))
    return torch.tensor(ctx), torch.tensor(tgt)


def acc(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()


def placement(model, ids, cats):
    with torch.no_grad():
        z = model.embed(ids)
    d = torch.cdist(z, z)
    same = cats[:, None] == cats[None, :]
    off = ~torch.eye(len(ids), dtype=torch.bool)
    within = d[same & off].mean()
    between = d[~same].mean()
    return (between / within).item()


def knn_purity(model, query_ids, query_cats, ref_ids, ref_cats, k=3):
    with torch.no_grad():
        d = torch.cdist(model.embed(query_ids), model.embed(ref_ids))
    nn_idx = d.topk(k, largest=False).indices
    return (ref_cats[nn_idx] == query_cats[:, None]).float().mean().item()


def curve_summary(values):
    arr = np.maximum.accumulate(np.array(values))
    target = arr[0] + 0.9 * (arr[-1] - arr[0])
    idx = next(i for i, v in enumerate(arr) if v >= target - 1e-9)
    per_epoch = CHECK_EVERY
    return {
        "at10": values[10 // per_epoch], "at25": values[25 // per_epoch],
        "early_slope": float(np.mean(np.diff(arr)[:3])),
        "checks_to_90": idx,
    }


def run(condition, data, memory, seed):
    vecs, cats, names, split = data
    model = External(condition, vecs, memory, seed)
    rng = np.random.default_rng(20_000 + seed)
    out = {"sessions": []}

    seen = []
    for s, new_cats in enumerate(STAGES):
        old_cats = list(seen)
        seen += new_cats

        new_words = [i for c in new_cats for i in split[c]["train"]]
        old_words = [i for c in old_cats for i in split[c]["train"]]
        replay_words = list(rng.choice(old_words, int(len(old_words) * REPLAY), replace=False)) if old_words else []
        train_ids = torch.tensor(new_words + replay_words)

        new_ctx, new_tgt = sentence_items(split, new_cats)
        if old_cats:
            old_ctx, old_tgt = sentence_items(split, old_cats)
            pick = torch.tensor(rng.choice(len(old_tgt), int(len(old_tgt) * REPLAY), replace=False))
            ctx = torch.cat([new_ctx, old_ctx[pick]])
            tgt = torch.cat([new_tgt, old_tgt[pick]])
        else:
            ctx, tgt = new_ctx, new_tgt

        new_ids = torch.tensor(new_words)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        cls_curve, sent_curve = [], []
        for epoch in range(EPOCHS + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    cls_curve.append(acc(model.classify(new_ids), cats[new_ids]))
                    sent_curve.append(acc(model.complete(new_ctx), new_tgt))
            if epoch == EPOCHS:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()

        trained = torch.tensor([i for c in seen for i in split[c]["train"]])
        unseen = torch.tensor([i for c in seen for i in split[c]["test"] + split[c]["val"]])
        all_ctx, all_tgt = sentence_items(split, seen)
        with torch.no_grad():
            res = {
                "cls_trained": acc(model.classify(trained), cats[trained]),
                "cls_unseen": acc(model.classify(unseen), cats[unseen]),
                "sentence": acc(model.complete(all_ctx), all_tgt),
                "placement": placement(model, trained, cats[trained]),
                "knn_unseen": knn_purity(model, unseen, cats[unseen], trained, cats[trained]),
                "cls_curve": curve_summary(cls_curve),
                "sent_curve": curve_summary(sent_curve),
            }
        out["sessions"].append(res)
    return out


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    results, mem_mse = [], []
    for seed in range(n_seeds):
        data = build_data(seed)
        memory, mse = memorize(data[0], seed)
        mem_mse.append(mse.mean().item())
        results.append({c: run(c, data, memory, seed) for c in CONDITIONS})

    def ms(vals):
        vals = np.array(vals, dtype=float)
        return f"{vals.mean():.3f}+/-{vals.std():.3f}"

    print(f"=== {n_seeds} seeds, {EPOCHS} epochs/session, replay {REPLAY:.0%} ===")
    print(f"Module A memorization error (MSE per word): {ms(mem_mse)}")

    print("\n--- after the final session, everything seen so far ---")
    for key, label in [("cls_trained", "category acc, trained words"),
                       ("cls_unseen", "category acc, words C never trained on"),
                       ("sentence", "sentence completion (connection)"),
                       ("placement", "placement: between/within distance ratio"),
                       ("knn_unseen", "placement: 3-NN purity of untrained words")]:
        row = "  ".join(f"{c}={ms([r[c]['sessions'][-1][key] for r in results])}" for c in CONDITIONS)
        print(f"{label:45} {row}")

    print("\n--- speed while learning each session's NEW words (trained words) ---")
    for task in ["cls_curve", "sent_curve"]:
        for s in range(len(STAGES)):
            for m in ["at10", "at25", "early_slope", "checks_to_90"]:
                row = "  ".join(f"{c}={ms([r[c]['sessions'][s][task][m] for r in results])}" for c in CONDITIONS)
                print(f"{task:10} session{s+1} {m:12} {row}")

    print("\n--- paired seed vs random (same data, same init except the word vectors) ---")
    checks = [("final", "cls_unseen"), ("final", "sentence"), ("final", "placement"), ("final", "knn_unseen")]
    for s in range(len(STAGES)):
        checks += [(s, ("cls_curve", "at10")), (s, ("sent_curve", "at10")), (s, ("sent_curve", "at25"))]
    for where, key in checks:
        def get(r, c):
            sess = r[c]["sessions"][-1 if where == "final" else where]
            return sess[key[0]][key[1]] if isinstance(key, tuple) else sess[key]
        diffs = np.array([get(r, "seed") - get(r, "random") for r in results])
        n_pos, n_neg = int((diffs > 1e-9).sum()), int((diffs < -1e-9).sum())
        name = f"{where if where == 'final' else 'session' + str(where + 1)} {key if isinstance(key, str) else '.'.join(key)}"
        print(f"{name:32} seed-random mean={diffs.mean():+.3f}  seed better/worse={n_pos}/{n_neg}  "
              f"p={sign_p(n_pos, n_neg):.3g}")


if __name__ == "__main__":
    main()
