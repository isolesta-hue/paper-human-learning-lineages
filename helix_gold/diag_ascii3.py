"""finding 91 follow-up: where does the late-life fall come from? One life (kind-first + cue templates + linear where memory);
after sessions 4, 8, 12, 16: motion items of the NEWEST rooms and of ALL rooms, answered by (a) the whole learner, (b) the
templates alone (pattern term incl. cue templates), (c) the habit alone (no templates, no whole-sentence memory), and the
bare formula (verb, from) -> to by the templates alone."""
import sys

import numpy as np
import torch

import data_ascii as A
import data_v3 as W
import integrated as I
from run_schema import PER_SESSION


def acc_parts(me, items):
    if not items:
        return {}
    ctx, tgt = I.L.items_tensor(items)
    m = me.model
    with torch.no_grad():
        whole = m.complete(ctx)
        pt = m.kind_first(ctx, m.pattern_term(ctx) - 1e-6 * torch.arange(m.C.shape[1]).float())
        on = m.chunk_on
        m.chunk_on = False
        up = m.use_pattern
        m.use_pattern = False
        hab = m.complete(ctx)
        m.use_pattern, m.chunk_on = up, on
    f = lambda z: round(float((z.argmax(1) == tgt).float().mean()), 3)
    return {"whole": f(whole), "templates": f(pt), "habit": f(hab)}


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    w = A.build(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    me = I.Learner3(w, seed, where="linear")
    settings = dict(I.START, kind_first=True, cue_templates="place" if "place" in sys.argv[2:] else True, cue_specific="specific" in sys.argv[2:],
                    cue_kinds="kinds" in sys.argv[2:], cue_members_vote="vote" in sys.argv[2:],
                    cue_pure="pure" in sys.argv[2:])
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    seen = []
    for i, new_cats in enumerate(sessions):
        old = list(seen)
        seen += new_cats
        me.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, settings, rng)
        if i + 1 in (4, 8, 12, 16):
            mot = lambda cs, kinds: [(cx, t) for c in cs for k, cx, t in tests[c] if k in kinds or (k.startswith("R") and "R" in kinds)]
            bare = [([w.verbs[k], w.pos_from[p]], w.pos_to[(p[0] + d[0], p[1] + d[1])]) for k, (_, d) in enumerate(A.VERBS)
                    for p in w.pos_from if (p[0] + d[0], p[1] + d[1]) in w.pos_to]
            print(f"session {i + 1}: once newest {acc_parts(me, mot(new_cats, ('once',)))} | once all {acc_parts(me, mot(seen, ('once',)))}"
                  f" | R all {acc_parts(me, mot(seen, ('R',)))} | bare formula {acc_parts(me, bare)}", flush=True)


if __name__ == "__main__":
    main()
