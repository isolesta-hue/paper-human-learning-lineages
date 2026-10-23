"""Review rule check (2026-10-12) for finding 139: trivial baselines on "goal which" / "goal next". Read-only, no learner.
(a) "goal which": pick one of the two named things at random (expected 0.5); pick the one named first after the agent.
(b) "goal next": repeat the last given step word; the most frequent word among the given steps; uniform over the 4 motion
words (0.25). What this assumes: the item layout of data_ascii goal=True (agent, cell, two named things with cells, steps)."""
import collections, sys
import numpy as np
import run_long as RL

for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5):
    w = RL.build_world("ascii-goal", seed)
    verbs = {int(t) for t in w.verbs}
    res = collections.defaultdict(lambda: [0, 0])
    for c in range(w.n_cat):
        for kind, sent, ans in w.categories[c].get("goal", []):
            sent = [int(t) for t in sent]
            if kind == "goal which":
                first_named = sent[2]
                res["which: named first"][0] += int(first_named == int(ans[0])); res["which: named first"][1] += 1
            elif kind == "goal next":
                steps = [t for t in sent[6:] if t in verbs]
                a = int(ans[0])
                res["next: repeat last"][0] += int(steps[-1] == a); res["next: repeat last"][1] += 1
                maj = collections.Counter(steps).most_common(1)[0][0]
                res["next: majority of steps"][0] += int(maj == a); res["next: majority of steps"][1] += 1
    print("seed", seed, {k: f"{x}/{n} = {x / n:.3f}" for k, (x, n) in res.items()})
