"""Role 3-b: does imagination get better as experience accumulates?

Project question (2026-09-22): people who know little speak badly and do not
know what to write; with more knowledge and experience they write and speak
fluently. In finding 32 the imagined sentences of the weak internal logics
were mostly nonsense -- but there the stored tables (the internal memory)
never took in anything the external logic learned: with no connection seed
they stayed random vectors and zeros however much was experienced.

Here experience can flow back into the internal memory: after every
session / cycle, the compare step of findings 29-30 ("exp": judged on every
sentence heard so far) updates the stored tables with the external logic's
values when they do better, or restores them when they do worse.
  loop off -- the stored tables only ever get the internal logics' seeds
  loop on  -- the stored tables also take in what the external logic learned
Life: the 3 learning sessions, then 4 "everyday" cycles in which everything
heard so far is heard again (with feedback, 50 epochs). Before each session
/ cycle (from the second on) the internal logic imagines sentences and the
external logic rehearses them (as finding 32). The internal logics seed new
words only (as before), so in "loop on" the growth comes from experience
flowing back, not from re-reading the internal logics.
Dose curve (loop on only): inside sessions 1 and 2, after 0/5/10/20/40/80/200
epochs of practice, what WOULD the internal memory imagine if the compare step
ran right then? (Measured on a copy; training is not touched.)
Measured at every step: imagination quality -- share of imagined sentences
holding a true partner pair, share whose 4 words all come from one category
("hangs together") -- and accuracy on never-heard sentences.
External logic: design A. Conditions: none / vec / hebb / vec+bert.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from run_unified_continual import STAGES, sign_p
from run_seed_transfer import memorize, EPOCHS, REPLAY
from run_connection_seed import Hebbian, MiniBert, masked_items, partner_items, acc
from run_ratchet_seed import build_data, score_with
from run_three_designs import ValueOnly
from run_role2_target import predict
from run_role3_imagine import imagine, PHASE_EPOCHS

CONDITIONS = ["none", "vec", "hebb", "vec+bert"]
LOOPS = ["off", "on"]
CYCLES = 4
CYCLE_EPOCHS = 50
DOSES = [0, 5, 10, 20, 40, 80, 200]


def quality(sents, partner, cats):
    return (float(np.mean([any(partner.get(w) in s for w in s) for s in sents])),
            float(np.mean([len({int(cats[w]) for w in s}) == 1 for s in sents])))


def compared(model, stored, judge):
    """The stored tables the compare step would produce right now (a copy)."""
    out = {}
    for name in ["z_work", "assoc"]:
        param = getattr(model, name)
        working = score_with(model, name, param.detach().clone(), judge)
        kept = score_with(model, name, stored[name], judge)
        out[name] = param.detach().clone() if working > kept else stored[name].clone()
    return out


def run(condition, loop, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = ValueOnly(start, seed)
    internal2 = Hebbian() if "hebb" in condition else MiniBert(memory, seed) if "bert" in condition else None
    stored = {"z_work": start.clone(), "assoc": torch.zeros_like(model.assoc)}
    rng = np.random.default_rng(20_000 + seed)
    phase_rng = np.random.default_rng(40_000 + seed)
    seen_cats, seen_words, steps = [], [], []
    schedule = [("session", st) for st in STAGES] + [("cycle", None)] * CYCLES

    def practise(ctx, tgt, ids, epochs, dose=None):
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for e in range(epochs):
            if dose is not None and e in DOSES:
                dose(e)
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(ctx), tgt)
            if ids is not None:
                loss = loss + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()
        if dose is not None and epochs in DOSES:
            dose(epochs)

    for i, (kind, new_cats) in enumerate(schedule):
        step = {"kind": kind}
        if i > 0:
            sents = imagine(stored["z_work"], stored["assoc"], seen_words, phase_rng)
            step["pair rate"], step["one category"] = quality(sents, partner, cats)
            pctx, ptgt, _, _ = masked_items(sents)
            practise(pctx, ptgt, None, PHASE_EPOCHS)

        if kind == "session":
            old_cats = list(seen_cats)
            seen_cats += new_cats
            new_words = [w for c in new_cats for w in split[c]["train"]]
            seen_words += new_words
            old_train = [w for c in old_cats for w in split[c]["train"]]
            replay_words = [int(w) for w in rng.choice(old_train, int(len(old_train) * REPLAY), replace=False)] if old_train else []
            new_sents = [s for c in new_cats for s in split[c]["sentences"]]
            old_sents = [s for c in old_cats for s in split[c]["sentences"]]
            replay_sents = [old_sents[j] for j in rng.choice(len(old_sents), int(len(old_sents) * REPLAY), replace=False)] if old_sents else []
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
            ctx, tgt, _, _ = masked_items(heard)
            dose = None
            if loop == "on" and i < 2:
                judge = masked_items([s for c in seen_cats for s in split[c]["sentences"]])[:2]
                step["dose"] = {}
                dose_rng = np.random.default_rng(50_000 + seed)

                def dose(e):
                    tables = compared(model, stored, judge)
                    step["dose"][e] = quality(imagine(tables["z_work"], tables["assoc"], seen_words, dose_rng),
                                              partner, cats)
            practise(ctx, tgt, torch.tensor(new_words + replay_words), EPOCHS, dose)
        else:
            heard = [s for c in seen_cats for s in split[c]["sentences"]]
            if internal2 is not None:
                internal2.observe(heard)
            ctx, tgt, _, _ = masked_items(heard)
            practise(ctx, tgt, torch.tensor(seen_words), CYCLE_EPOCHS)

        if loop == "on":
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

        test = partner_items([s for c in seen_cats for s in split[c]["test_sents"]], partner)
        heardp = partner_items([s for c in seen_cats for s in split[c]["sentences"]], partner)
        with torch.no_grad():
            step["never-heard"] = acc(model.complete(test[0]), test[1])
            step["internal never-heard"] = acc(predict(stored["z_work"], stored["assoc"], test[0]), test[1])
            step["internal heard"] = acc(predict(stored["z_work"], stored["assoc"], heardp[0]), heardp[1])
        steps.append(step)
    return steps


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(c, lp): run(c, lp, data, memory, seed) for c in CONDITIONS for lp in LOOPS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_steps = len(results[0][(CONDITIONS[0], LOOPS[0])])
    labels = [f"s{i+1}" if i < len(STAGES) else f"c{i - len(STAGES) + 1}" for i in range(n_steps)]

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}"

    print(f"=== {n_seeds} seeds, design A, 3 sessions + {CYCLES} everyday cycles ===")
    print("columns: " + " ".join(labels) + "  (imagination measured BEFORE that step, from s2 on)")
    for key in ["pair rate", "one category", "internal heard", "internal never-heard", "never-heard"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            for lp in LOOPS:
                row = [m([r[(c, lp)][i].get(key, np.nan) for r in results]) for i in range(n_steps)]
                print(f"{c:9} loop {lp:3} " + " ".join(row))
    print("\n--- dose curve (loop on): imagination quality if the compare step ran after N epochs ---")
    print("epochs: " + " ".join(f"{d:>5}" for d in DOSES))
    for si in range(2):
        for c in CONDITIONS:
            for qi, qname in enumerate(["pair rate", "one category"]):
                row = [m([r[(c, "on")][si]["dose"][d][qi] for r in results]) for d in DOSES]
                print(f"s{si+1} {c:9} {qname:12} " + " ".join(f"{v:>5}" for v in row))
    print("\n--- loop on vs off, pair rate before the last cycle / never-heard at the end ---")
    for c in CONDITIONS:
        cells = []
        for key, i in [("pair rate", n_steps - 1), ("never-heard", n_steps - 1)]:
            d = np.array([r[(c, "on")][i][key] - r[(c, "off")][i][key] for r in results])
            n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
            cells.append(f"{key} {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.2g})")
        print(f"{c:9} " + "  ".join(cells))
    print("\n--- does imagination improve with experience? pair rate at s2 vs before the last cycle ---")
    for c in CONDITIONS:
        for lp in LOOPS:
            d = np.array([r[(c, lp)][n_steps - 1]["pair rate"] - r[(c, lp)][1]["pair rate"] for r in results])
            n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
            print(f"{c:9} loop {lp:3} {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.2g})")


if __name__ == "__main__":
    main()
