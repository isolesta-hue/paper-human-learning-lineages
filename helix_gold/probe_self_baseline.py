"""Review check (2026-10-12): can finding 140's "self which" / "self mirror" items be solved WITHOUT contingency?
Baseline: among the things that move in the test item, pick the one that co-occurred most with any action word (ACT0-3) in the
heard sentences (in the heard data only "me" moves with an action). Read-only: builds the world, no learner.
What this assumes: the candidates are the movers named in the item (three per action); counts over all rooms' heard sentences."""
import collections, sys
import run_long as RL

for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    w = RL.build_world("ascii-self", seed)
    acts = {int(t) for t in w.actions}
    co = collections.Counter()
    for c in range(w.n_cat):
        for item in w.categories[c]["heard"]:
            s = [int(t) for t in item[1]]
            if any(t in acts for t in s):
                co.update(s)
    out = collections.defaultdict(lambda: [0, 0])
    for c in range(w.n_cat):
        for kind, sent, ans in w.categories[c].get("self", []):
            sent, cands, i = [int(t) for t in sent], [], 0
            while i < len(sent):
                if sent[i] in acts:
                    j = i + 1
                    while j + 2 < len(sent) and sent[j] not in acts and sent[j] != int(w.self_q):
                        cands.append(sent[j]); j += 3
                    i = j
                else:
                    i += 1
            pick = max(sorted(set(cands)), key=lambda t: co[t])
            out[kind][0] += int(pick == int(ans[0])); out[kind][1] += 1
    print("seed", seed, {k: f"{a}/{n} = {a / n:.3f}" for k, (a, n) in out.items()})
