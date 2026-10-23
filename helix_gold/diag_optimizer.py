"""Optimizer-restart cost (finding 46 observation): session-1 situation, 200 epochs in one go vs 100 + 100 with a
fresh Adam vs 100 + 100 continuing the same Adam; also how much the first steps after a restart move the values."""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import data_v2 as D
import lab2 as L
from run_seed_transfer import memorize
from run_self_test import positions, score


def one(seed):
    torch.set_num_threads(1)
    d = D.build("hard", seed)
    rng = np.random.default_rng(50_000 + seed)
    cats = torch.tensor(d.cats)
    new = [int(c) for c in rng.permutation(d.n_cat)[:2]]
    sents = [s for c in new for _, s, _ in d.categories[c]["heard"]]
    words = sorted({w for s in sents for w in s})
    test = [(ctx, t) for c in new for _, ctx, t in D.unit_items(d.categories[c]["test"])]
    memory, _ = memorize(torch.tensor(d.vecs), seed)
    ctx, tgt = L.items_tensor(positions(sents))
    bag = L.bag(ctx, d.V)
    ids = torch.tensor(words)
    out = {}
    for mode in ["one", "fresh", "continue"]:
        model = L.KnowDo2(d.V, d.n_cat, memory, seed)
        hebb = L.make_internal2("dhebb", d.V)
        hebb.observe(sents)
        conn = hebb.connections(words)
        with torch.no_grad():
            model.assoc[torch.tensor(words), :] = conn[torch.tensor(words), :]
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        jump = {}
        for e in range(200):
            if e == 100 and mode == "fresh":
                opt = torch.optim.Adam(model.parameters(), lr=0.01)
            if e in (100, 105):
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(bag), tgt) + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()
            if e in (100, 105):
                jump[e] = {n: float((p.detach() - before[n]).abs().mean()) for n, p in model.named_parameters() if n in ("assoc", "skill", "z_work")}
            if e == 99:
                out[mode + " @100"] = score(model, test)
        out[mode + " @200"] = score(model, test)
        out[mode + " jump"] = jump
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    with Pool(4) as pool:
        res = pool.map(one, range(n))
    for mode in ["one", "fresh", "continue"]:
        a100 = np.mean([r[mode + " @100"] for r in res])
        a200 = np.mean([r[mode + " @200"] for r in res])
        j = {e: {k: np.mean([r[mode + " jump"][e][k] for r in res]) for k in ("assoc", "skill", "z_work")} for e in (100, 105)}
        print(f"{mode:9} outside @100 {a100:.3f}  @200 {a200:.3f}   mean |change| of one step at epoch 100: "
              + " ".join(f"{k} {v:.5f}" for k, v in j[100].items()) + "   at epoch 105: " + " ".join(f"{k} {v:.5f}" for k, v in j[105].items()))
