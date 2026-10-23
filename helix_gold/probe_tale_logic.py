"""D22 body-logic probe (no training): the world model alone, fed the heard moments directly. Stage A (the preview window):
everything heard in the phase-0 rooms (the preview tales of the phase-1 rooms included) -> the phase-1 rooms' real gravity
questions (marked / unmarked previews). Stage B (every room heard) -> tale questions (marked) and the real gravity questions
of clean / drop / turned rooms. Answer = the world model's chosen cell (or silent)."""
import sys; sys.path.insert(0, ".")
import numpy as np, torch, integrated as I, worldmodel as WM

def life(w, seed, mode, cats):
    torch.manual_seed(seed)
    me = I.Learner3(w, seed, content=True, where="linear")
    wm = WM.WorldBooks(me)
    wm.bind, wm.track, wm.grav, wm.last = True, True, True, True
    wm.tale_mode, wm.tale_mark = mode, w.tale_mark
    me.keep_order, me.order_log = True, []
    sents = [s for c in cats for _, s, _ in w.categories[c]["heard"]]
    me.chunks.observe(sents, "world")
    me.order_log.extend(list(s) for s in sents)
    me.known |= {x for s in sents for x in s}
    return wm

def score(wm, items):
    if not items:
        return "  -  "
    ok = sil = 0
    for s, a in items:
        ctx = torch.tensor([s])
        row = wm.term(ctx)[0]
        if float(row.abs().sum()) == 0:
            sil += 1
        elif int(row.argmax()) == a:
            ok += 1
    return f"{ok / len(items):.2f}/{sil / len(items):.2f}"

for seed in (0, 1, 2):
    w = I.build_world("ascii-tale", seed)
    ph0 = [0] + [c for c in range(1, w.n_cat) if w.categories[c]["phase"] == 0]
    allc = list(range(w.n_cat))
    G = lambda cs: [(s, a[0]) for c in cs for k, s, a in w.categories[c].get("grav", []) if k in ("grav floor", "grav stack")]
    pre = {v: [c for c in range(1, w.n_cat) if w.categories[c].get("pre_drop") is v] for v in (False, True)}
    groups = {"clean": [c for c in allc if w.categories[c].get("tale_drop") is False],
              "drop": [c for c in allc if w.categories[c].get("tale_drop") is True],
              "turned": [c for c in allc if w.categories[c].get("flipped") is True]}
    tin = [(s, a[0]) for c in allc for k, s, a in w.categories[c].get("tale", []) if k == "tale in"]
    print(f"seed {seed}   (right/silent)  preview marked | preview unmarked || tale in | real clean | real drop | real turned | flagged")
    for mode in (None, "ignore", "one", "anchor"):
        a = life(w, seed, mode, ph0)
        a.refresh(); fa = a.tale_flagged
        b = life(w, seed, mode, allc)
        b.refresh()
        print(f"  {str(mode):7} {score(a, G(pre[False])):>14} | {score(a, G(pre[True])):>16} || {score(b, tin):>7} | "
              f"{score(b, G(groups['clean'])):>10} | {score(b, G(groups['drop'])):>9} | {score(b, G(groups['turned'])):>11} | {fa}/{b.tale_flagged}")
