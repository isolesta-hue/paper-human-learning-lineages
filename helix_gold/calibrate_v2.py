"""Difficulty check for data_v2: all categories learned at once (no continual
learning), 200 epochs; accuracy on never-heard test items per relation kind.
Target: the plain MLP should land around 0.5-0.7 on the hard level."""
import sys
from multiprocessing import Pool

import numpy as np
import torch

import data_v2 as D
import lab2 as L
from run_seed_transfer import memorize
from run_unified_continual import sign_p

MODELS = ["vanilla", "A-none", "A-vec+dhebb", "D-none", "D-vec", "D-dhebb", "D-vec+dhebb"]


def one(args):
    level, seed = args
    torch.set_num_threads(1)
    d = D.build(level, seed)
    cats = torch.tensor(d.cats)
    heard = [s for c in d.categories for s in c["heard"]]
    items = D.all_positions(heard)
    words = sorted({w for c in d.categories for w in c["train"]})
    seen = sorted({w for _, s, _ in heard for w in s} | set(words))
    test = [it for c in d.categories for it in D.unit_items(c["test"])]
    out = {}
    for m in MODELS:
        if m == "vanilla":
            model = L.Vanilla(d.V, d.n_cat, seed)
        else:
            vecs = torch.tensor(d.vecs)
            memory, _ = memorize(vecs, seed)
            start = memory if "vec" in m else torch.randn_like(memory) * memory.std()
            model = (L.KnowDo2 if m.startswith("D") else L.ValueOnly2)(d.V, d.n_cat, start, seed)
            if "dhebb" in m:
                ih = L.make_internal2("dhebb", d.V)
                ih.observe([s for _, s, _ in heard])
                with torch.no_grad():
                    model.assoc.copy_(ih.connections(seen))
        L.train(model, items, words, cats, 200)
        out[m] = L.evaluate_units(model, test)
    return level, out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    levels = sys.argv[2].split(",") if len(sys.argv) > 2 else list(D.LEVELS)
    jobs = [(lv, s) for lv in levels for s in range(n)]
    with Pool(4) as pool:
        res = pool.map(one, jobs)
    for lv in levels:
        rs = [o for l, o in res if l == lv]
        kinds = sorted(rs[0][MODELS[0]].keys())
        print(f"--- {lv} ---")
        for m in MODELS:
            print(f"{m:12} " + "  ".join(f"{k}={np.nanmean([r[m].get(k, np.nan) for r in rs]):.2f}" for k in kinds))
        print("paired, all kinds (mean difference, better:worse seeds, * = sign test p<0.01)")
        for a, b in [("D-vec+dhebb", "vanilla"), ("D-vec", "D-none"), ("D-dhebb", "D-none"),
                     ("D-vec+dhebb", "D-dhebb"), ("D-vec+dhebb", "D-vec")]:
            dd = np.array([r[a]["all"] - r[b]["all"] for r in rs])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            print(f"  {a:12} vs {b:12} {dd.mean():+.3f} ({n_pos}:{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
