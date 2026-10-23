"""Diagnostic for finding 27, design A (vec+hebb): at each session end, working vs stored
connection table, scored on HEARD sentences (what feedback covered) and on validation
(never-heard) sentences (what the restore decision used)."""
import sys
import numpy as np
import torch
import run_three_designs as T
import run_ratchet_seed as R
from run_seed_transfer import memorize
from run_connection_seed import masked_items, partner_items, acc

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = {k: [[] for _ in range(3)] for k in ["heard: working", "heard: stored", "heard partner: working",
                                             "heard partner: stored", "val: working", "val: stored",
                                             "stored conn alone on val"]}
orig = T.score_with
for seed in range(n):
    data = R.build_data(seed)
    memory, _ = memorize(data[0], seed)
    split, partner = data[2], data[3]
    state = {"s": -1, "calls": 0, "seen": []}

    def hook(model, name, table, items):
        if name == "assoc":
            state["calls"] += 1
            if state["calls"] % 2 == 0:  # second call = stored table
                s = state["calls"] // 2 - 1
                cats_seen = [c for st in T.STAGES[:s + 1] for c in st]
                heard = [x for c in cats_seen for x in split[c]["sentences"]]
                hc, ht, _, _ = masked_items(heard)
                hp = partner_items(heard, partner)
                with torch.no_grad():
                    rows["heard: working"][s].append(acc(model.complete(hc), ht))
                    rows["heard partner: working"][s].append(acc(model.complete(hp[0]), hp[1]))
                    rows["val: working"][s].append(acc(model.complete(items[0]), items[1]))
                rows["heard: stored"][s].append(orig(model, "assoc", table, (hc, ht)))
                rows["heard partner: stored"][s].append(orig(model, "assoc", table, hp))
                rows["val: stored"][s].append(orig(model, "assoc", table, items))
                rows["stored conn alone on val"][s].append(acc(table[items[0]].mean(1), items[1]))
        return orig(model, name, table, items)

    T.score_with = hook
    T.run("A", "vec+hebb", data, memory, seed)
    T.score_with = orig
for k, per in rows.items():
    print(f"{k:28} " + "  ".join(f"s{s+1}={np.mean(v):.3f}" for s, v in enumerate(per)))
