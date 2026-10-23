"""finding 90, second diagnosis: with the kind-first part on, ~18% of motion answers are the cell ANOTHER verb would give. Which
confusion? By room phase (1 basic / 2 new verb / 3 upside-down rooms): right / REVERSE (the same verb's unflipped or flipped
direction -- the room's rule mixed up) / other verb / other cell. Plus the bare formula test: (verb, from) alone, textbook rule."""
import sys

import numpy as np
import torch

import data_ascii as A
import data_v3 as W
import integrated as I
from run_schema import PER_SESSION


def answers(w, me, items):
    ctx, tgt = I.L.items_tensor([(cx, t) for cx, t in items])
    with torch.no_grad():
        return me.model.complete(ctx).argmax(-1).tolist()


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tot = {}
    for seed in range(seeds):
        w = A.build(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(70_000 + seed)
        me = I.Learner3(w, seed)
        settings = dict(I.START, kind_first=True)
        sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
        tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
        order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
        sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
        seen = []
        for new_cats in sessions:
            old = list(seen)
            seen += new_cats
            me.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, settings, rng)
        for c in seen:
            if c == 0:
                continue
            ph, fl = w.categories[c]["phase"], w.categories[c]["flipped"]
            its = [(cx, t) for k, cx, t in tests[c] if k.startswith("R") or k in ("once", "unseen")]
            for (cx, t), p in zip(its, answers(w, me, its)):
                v = [x for x in cx if x in w.verb_of][0]
                f = [x for x in cx if x in w.pos_of and w.role_of[x] in ("from", "at")][0]
                fx, fy = w.pos_of[f]
                d = A.VERBS[w.verb_of[v]][1]
                rev = (fx - d[0], fy - d[1]) if not fl else (fx + d[0], fy + d[1])
                if p == t:
                    r = "right"
                elif p in w.pos_of and w.pos_of[p] == rev:
                    r = "reverse"
                elif p in w.pos_of and any(w.pos_of[p] == (fx + e[0] * s, fy + e[1] * s) for _, e in A.VERBS for s in (1, -1)):
                    r = "other verb"
                else:
                    r = "other cell"
                tot.setdefault(f"phase {ph + 1}", {}).setdefault(r, 0)
                tot[f"phase {ph + 1}"][r] += 1
        bare = [([w.verbs[k], w.pos_from[p]], w.pos_to[(p[0] + d[0], p[1] + d[1])]) for k, (_, d) in enumerate(A.VERBS)
                for p in w.pos_from if (p[0] + d[0], p[1] + d[1]) in w.pos_to]
        noverb = [([w.pos_from[p]], w.pos_to[(p[0] + d[0], p[1] + d[1])]) for k, (_, d) in enumerate(A.VERBS)
                  for p in w.pos_from if (p[0] + d[0], p[1] + d[1]) in w.pos_to]
        tot.setdefault("bare formula (verb, from)", []).append(I.acc_items(me.model, bare))
        tot.setdefault("no verb (from only)", []).append(I.acc_items(me.model, noverb))
    for k, v in tot.items():
        if isinstance(v, dict):
            n = sum(v.values())
            print(k, {r: round(x / n, 3) for r, x in sorted(v.items())}, f"(n={n})")
        else:
            print(k, round(float(np.mean(v)), 3))


if __name__ == "__main__":
    main()
