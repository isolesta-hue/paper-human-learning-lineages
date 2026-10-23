"""diagnosis of path_taught's first full run (2026-10-21): the TAUGHT body answered every item wrong (control items too, pair
items 0.000 at chance 0.5) -- a broken-body signature. Rebuild life_000's taught body exactly as path_taught.run_life and
look: NaN / inf in the model, the scores of a few items, the loss during the session. Read only (no rule changes)."""
import copy
import os
import sys

import numpy as np
import torch

import path_taught as P
import conv_v1 as CV
from exam.grade import load

ed = "/home/user/shared-curriculum/exam_dev/dev1000"
life = "life_000"
life_dir = os.path.join(ed, "lives", life)
w, qs = CV.convert_life(life_dir)
keys = {k["id"]: k for k in load(os.path.join(ed, "keys", f"{life}.jsonl"))}
from exam.checks import group, pass_line
qmap = {q.id: q for q in qs}
by_g = {}
for k in sorted(keys.values(), key=lambda k: k["id"]):     # the same plan as path_taught.run_life
    if pass_line(group(k)) is not None:
        by_g.setdefault(group(k), []).append(k)
extra, taught_ids, control_ids = [], [], []
for g, ks in by_g.items():
    taught_ids += [k["id"] for k in ks[0::2][:P.K]]
    control_ids += [k["id"] for k in ks[1::2][:P.K]]
    for k in ks[0::2][:P.K]:
        ms = P.taught_moments(qmap[k["id"]], k, w)
        if ms:
            extra += [list(m) for m in ms for _ in range(P.R)]
w_t = copy.copy(w)
w_t.cats = w.cats.copy()
for t in {t for m in extra for t in m}:
    if w_t.cats[t] < 0:
        w_t.cats[t] = w.sessions[0]
print("extra moments", len(extra), "| session moments", len(w.categories[w.sessions[0]]["heard"]),
      "| item-only elements given a category", int(((w.cats < 0) & (w_t.cats >= 0)).sum()), flush=True)
body = P.body_after(w_t, 0, extra)
m = body.me.model
bad = [(n, int(torch.isnan(p).sum()), int(torch.isinf(p).sum())) for n, p in m.named_parameters() if not torch.isfinite(p).all()]
print("non-finite parameters:", bad[:10])
for name in ("assoc", "z_work", "vecs"):
    t = getattr(m, name, None)
    if isinstance(t, torch.Tensor):
        print(name, "finite", bool(torch.isfinite(t).all()), "max |x|", float(t.abs().max()) if torch.isfinite(t).all() else "-")
for qid in taught_ids[:200:20] + control_ids[:200:20]:
    q = qmap[qid]
    row = CV.answer(body, q, 0)
    sc = list(row["scores"].values())
    print(q.id, q.format, "n", len(sc), "nan", sum(1 for v in sc if v != v), "distinct", len(set(sc)), "max", max(sc),
          "min", min(sc))
lp = body._logp(qmap[taught_ids[0]].sequence(w.boundary))
print("logp of the first taught item's query: finite", bool(torch.isfinite(lp).all()), "max", float(lp.max()),
      "top elements", [w.names[int(i)] for i in lp.topk(5).indices])
