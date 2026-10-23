"""B stage, first diagnosis (2026-09-30): on the glyph-picture world the content layer's score FALLS over the life (0.69 -> 0.45).
Hypothesis: positions are shared by every room, so (verb, from) -> to needs the verb and the position TOGETHER; the learner
may not use the verb (the type marker) to pick the template. Diagnosis order (memory-first rule 4): memory -> retrieval ->
logic -> structure.
  memory     -- element check (data_ascii.element_check; passed 100% at 5 x 5)
  structure  -- can the answer be expressed in the learner's memory space at all? per verb, how constant is z(to) - z(from)
                over positions (mean cosine to the verb's mean offset), and does "z(from) + the verb's mean offset" land nearest to
                z(to) (the template ceiling in memory space)
  logic      -- what the learner answers on motion items: right / the start position again / the position ANOTHER verb would
                give (verb confusion) / another position / not a position at all; naming items by kind
Usage: python diag_ascii.py <seeds> [kind] [cue] [linear] [noroles] [curric] (content layer with the start settings, as integrated.life_fixed; "kind" wakes
finding 90's kind-first answer)."""
import sys

import numpy as np
import torch

import data_ascii as A
import data_v3 as W
import integrated as I
from run_schema import PER_SESSION


def motion_view(w, me, items):
    """classify answers on motion items (ctx has one verb and one other position; hidden = the 'to' position)."""
    if not items:                                   # (placement rooms have no motion)
        return {}
    ctx, tgt = I.L.items_tensor([(cx, t) for _, cx, t in items])
    with torch.no_grad():
        pred = me.model.complete(ctx).argmax(-1).tolist()
    out = {"right": 0, "start": 0, "other verb": 0, "other pos": 0, "not pos": 0}
    for (kind, cx, t), p in zip(items, pred):
        verb = [x for x in cx if x in w.verb_of]
        frm = [x for x in cx if x in w.pos_of and w.role_of[x] in ("from", "at")]
        if p == t:
            out["right"] += 1
        elif p not in w.pos_of:
            out["not pos"] += 1
        elif frm and p == frm[0]:
            out["start"] += 1
        else:
            fx, fy = w.pos_of[frm[0]] if frm else (99, 99)
            px, py = w.pos_of[p]
            others = [(fx + d[0] * s, fy + d[1] * s) for k, (_, d) in enumerate(A.VERBS) for s in (1, -1)
                      if not verb or k != w.verb_of[verb[0]] or s == -1]
            out["other verb" if (px, py) in others else "other pos"] += 1
    n = max(len(items), 1)
    return {k: round(v / n, 3) for k, v in out.items()}


def template_space(w, me):
    """per verb: consistency of the memory-space offset z(to) - z(from) over all cells, and the template ceiling."""
    z = me.memory.numpy()
    P = list(w.pos_to.values())
    zp = z[P]
    res = {}
    for k, (name, d) in enumerate(A.VERBS):
        pairs = [(w.pos_from[p], w.pos_to[(p[0] + d[0], p[1] + d[1])]) for p in w.pos_from
                 if (p[0] + d[0], p[1] + d[1]) in w.pos_to]
        off = np.array([z[b] - z[a] for a, b in pairs])
        m = off.mean(0)
        cos = (off @ m) / (np.linalg.norm(off, axis=1) * np.linalg.norm(m) + 1e-9)
        hit = [P[int(np.argmin(((zp - (z[a] + m)) ** 2).sum(1)))] == b for a, b in pairs]
        res[name] = (round(float(cos.mean()), 3), round(float(np.mean(hit)), 3))
    return res


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    settings = dict(I.START, kind_first="kind" in sys.argv[2:],   # "kind": finding 90's kind-first part woken
                    cue_templates="place" if "place" in sys.argv[2:] else "cue" in sys.argv[2:],  # "cue" / "place" (cue + place)
                    # finding 92 (template repairs): "fix" = kind-consistent templates + specific before general + cued
                    # templates vote on their own pairs + the purest cue first
                    cue_kinds="fix" in sys.argv[2:], cue_specific="fix" in sys.argv[2:],
                    cue_members_vote="fix" in sys.argv[2:], cue_pure="fix" in sys.argv[2:])
    for seed in range(n_seeds):
        w = A.build(seed, roles="noroles" not in sys.argv[2:], curriculum="curric" in sys.argv[2:])
        torch.manual_seed(seed)
        rng = np.random.default_rng(70_000 + seed)
        lin = "linear" in sys.argv[2:]              # finding 90 (structure): the "where" channel's own linear memory
        me = I.Learner3(w, seed, where="linear" if lin else None)
        print(f"seed {seed}: element check {A.element_check(w, me.memory, seed, linear_pos=me.where_decoder if lin else None)}")
        print(f"  structure (per verb: offset consistency, template ceiling in memory space): {template_space(w, me)}")
        sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
        tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
        order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
        sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
        seen = []
        for i, new_cats in enumerate(sessions):
            old = list(seen)
            seen += new_cats
            me.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, settings, rng)
            if i in (0, 3, 7, 11, len(sessions) - 1):
                items = [(k, cx, t) for c in seen for k, cx, t in tests[c]]
                by = {}
                for k, cx, t in items:
                    by.setdefault("R" if k.startswith("R") else k, []).append((cx, t))
                accs = {k: round(I.acc_items(me.model, v), 3) for k, v in sorted(by.items())}
                mot = [(k, cx, t) for k, cx, t in items if k.startswith("R") or k in ("once", "unseen")]
                newest = [(k, cx, t) for c in new_cats for k, cx, t in tests[c] if k.startswith("R") or k in ("once", "unseen")]
                print(f"  session {i + 1}: {accs}")
                print(f"     motion answers (all seen): {motion_view(w, me, mot)}  newest rooms: {motion_view(w, me, newest)}")


if __name__ == "__main__":
    main()
