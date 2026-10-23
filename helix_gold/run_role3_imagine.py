"""Role 3 of internal data: material for internal computation without outside
input -- imagining sentences and rehearsing them (original design item 14).

Between sessions, with no outside input, the internal logic builds sentences
from its own memory: start from a random word it knows, then keep adding
the word it expects most strongly given what is already there (the internal
prediction of run_role2_target.py, on the stored tables), until the sentence
has 4 words. The external logic then rehearses these imagined sentences
for a while, taking the internal's own words as the answers -- no feedback
from outside.

Phase between sessions (after session 1 and after session 2), same length
in all modes:
  rest     -- nothing happens
  imagine  -- rehearse the imagined sentences (internal answers only)
  replay   -- reference: rehearse the REAL heard sentences of the earlier
              sessions, with the real answers (outside feedback)
Measured: quality of the imagined sentences (share holding a true partner
pair), accuracy on everything seen so far right before and right after the
phase, and the next session's curves / its end on old and new material.
External logic: design A, restore off. Conditions: none / vec / hebb /
vec+hebb / vec+bert.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from run_unified_continual import STAGES, sign_p
from run_seed_transfer import memorize, EPOCHS, CHECK_EVERY, REPLAY
from run_connection_seed import Hebbian, MiniBert, masked_items, partner_items, acc, curve, SENT_LEN
from run_ratchet_seed import build_data
from run_three_designs import ValueOnly
from run_role2_target import predict

CONDITIONS = ["none", "vec", "hebb", "vec+hebb", "vec+bert"]
MODES = ["rest", "imagine", "replay"]
PHASE_EPOCHS = 50


def imagine(z, assoc, seen_words, rng):
    """One imagined sentence per known word as the starting point."""
    seen = torch.tensor(seen_words)
    sentences = []
    for startw in rng.permutation(seen_words):
        sent = [int(startw)]
        while len(sent) < SENT_LEN:
            with torch.no_grad():
                scores = predict(z, assoc, torch.tensor([sent]))[0, seen]
            scores[torch.isin(seen, torch.tensor(sent))] = -float("inf")
            sent.append(int(seen[scores.argmax()]))
        sentences.append(sent)
    return sentences


def run(condition, mode, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = ValueOnly(start, seed)
    internal2 = Hebbian() if "hebb" in condition else MiniBert(memory, seed) if "bert" in condition else None
    stored = {"z_work": start.clone(), "assoc": torch.zeros_like(model.assoc)}
    rng = np.random.default_rng(20_000 + seed)
    phase_rng = np.random.default_rng(40_000 + seed)
    sessions, seen_cats, seen_words = [], [], []

    def everything():
        heardp = partner_items([s for c in seen_cats for s in split[c]["sentences"]], partner)
        test = partner_items([s for c in seen_cats for s in split[c]["test_sents"]], partner)
        with torch.no_grad():
            return {"heard partner": acc(model.complete(heardp[0]), heardp[1]),
                    "never-heard": acc(model.complete(test[0]), test[1])}

    for si, new_cats in enumerate(STAGES):
        phase = {}
        if si > 0 and mode != "rest":
            phase["before"] = everything()
            if mode == "imagine":
                sents = imagine(stored["z_work"], stored["assoc"], seen_words, phase_rng)
                phase["quality"] = float(np.mean([any(partner.get(w) in s for w in s) for s in sents]))
            else:
                sents = [s for c in seen_cats for s in split[c]["sentences"]]
            pctx, ptgt, _, _ = masked_items(sents)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(PHASE_EPOCHS):
                opt.zero_grad()
                F.cross_entropy(model.complete(pctx), ptgt).backward()
                opt.step()
            phase["after"] = everything()

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

        new_heard = partner_items(new_sents, partner)
        new_test = partner_items([s for c in new_cats for s in split[c]["test_sents"]], partner)
        old_heard = partner_items(old_sents, partner) if old_sents else None
        train_ids = torch.tensor(new_words + replay_words)
        ctx, tgt, _, _ = masked_items(heard)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        c_new, c_test = [], []
        for epoch in range(EPOCHS + 1):
            if epoch % CHECK_EVERY == 0:
                with torch.no_grad():
                    c_new.append(acc(model.complete(new_heard[0]), new_heard[1]))
                    c_test.append(acc(model.complete(new_test[0]), new_test[1]))
            if epoch == EPOCHS:
                break
            opt.zero_grad()
            loss = F.cross_entropy(model.classify(train_ids), cats[train_ids]) + \
                F.cross_entropy(model.complete(ctx), tgt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            old_end = acc(model.complete(old_heard[0]), old_heard[1]) if old_heard else float("nan")
        sessions.append({"phase": phase, "new heard": curve(c_new), "new never-heard": curve(c_test),
                         "old heard end": old_end, "end": everything()})
    return sessions


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(c, md): run(c, md, data, memory, seed) for c in CONDITIONS for md in MODES}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))

    def m(vals):
        return f"{np.mean(np.array(vals, dtype=float)):.3f}"

    def paired(get, a, b):
        d = np.array([get(r, a) - get(r, b) for r in results])
        n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        return f"{d.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})"

    print(f"=== {n_seeds} seeds, design A, phase {PHASE_EPOCHS} epochs between sessions ===")
    print("\n--- the phase itself (before -> after), everything seen so far ---")
    for s in range(1, len(STAGES)):
        for c in CONDITIONS:
            cells = []
            for md in ["imagine", "replay"]:
                rs = [r[(c, md)][s]["phase"] for r in results]
                q = f" quality={m([x['quality'] for x in rs])}" if md == "imagine" else ""
                cells.append(f"{md}:{q} heard partner {m([x['before']['heard partner'] for x in rs])}->"
                             f"{m([x['after']['heard partner'] for x in rs])}, never-heard "
                             f"{m([x['before']['never-heard'] for x in rs])}->{m([x['after']['never-heard'] for x in rs])}")
            print(f"before s{s+1} {c:9} " + " | ".join(cells))

    print("\n--- next session: learning the new words, and the old ones at its end ---")
    for s in range(1, len(STAGES)):
        for c in CONDITIONS:
            cells = []
            for md in MODES:
                x = [r[(c, md)][s] for r in results]
                cells.append(f"{md}: new at0={m([y['new heard']['at0'] for y in x])} at10={m([y['new heard']['at10'] for y in x])} "
                             f"chk90={m([y['new heard']['checks_to_90'] for y in x])} old end={m([y['old heard end'] for y in x])} "
                             f"all never-heard end={m([y['end']['never-heard'] for y in x])}")
            print(f"s{s+1} {c:9} " + " | ".join(cells))

    print("\n--- paired vs rest (same condition): old heard end / all never-heard end / new heard at10 ---")
    for s in range(1, len(STAGES)):
        for c in CONDITIONS:
            for md in ["imagine", "replay"]:
                cells = [paired(lambda r, key, get=get: get(r[key][s]), (c, md), (c, "rest"))
                         for get in (lambda y: y["old heard end"], lambda y: y["end"]["never-heard"],
                                     lambda y: y["new heard"]["at10"])]
                print(f"s{s+1} {c:9} {md:8} " + "  ".join(cells))
    print("\n--- imagine: each condition vs none (all never-heard end) ---")
    for s in range(1, len(STAGES)):
        for c in CONDITIONS[1:]:
            print(f"s{s+1} {c:9} " + paired(lambda r, key: r[key][s]["end"]["never-heard"], (c, "imagine"), ("none", "imagine")))


if __name__ == "__main__":
    main()
