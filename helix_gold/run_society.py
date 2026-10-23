"""A small society: envy and imitation between learners (project idea,
2026-09-23: people watch someone who does better, envy them, take them as a
goal and as a yardstick for judging themselves, and imitate them).

Society: N individuals living through the same curriculum (the hard level of
data_v2: 16 categories arriving 2 per session over 8 sessions, then 2 everyday
cycles). They differ the way people do:
  memory     -- each has its own Module A memory (own seed)
  experience -- each hears its own random 2/3 of every category's sentences
  strategy   -- each has its own learning settings: learning rate and share of
                older sentences replayed in a session (drawn from a small grid
                of representative values, all different at the start)
Learner: design D (finding 34, `lab2.KnowDo2`), vector seed + directional
connection seed (internal logic 2) from the individual's own sentences -- on
the hard data this matches the plain MLP (see the base-learner check).
Every step: a practice slot (imagination, 50 epochs, finding 36), then the
session / cycle, then a public quiz.

Envy is computed only from real performance: everybody's accuracy on a
public quiz (the val sentences of the categories met so far) is visible to
everybody, like a class ranking. Nobody sees the hidden test sentences.
  gap to peer j   = j's quiz accuracy - mine
  target          -- best: the peer with the largest gap / near: the peer just
                     above me (smallest gap) -- "zone of proximal development"
  single / accum. -- envy on this step's gap, or on its running mean (rate 0.5,
                     like the expectation of finding 40) so one lucky step
                     does not trigger it
  a peer counts only when the gap exceeds DELTA.
What envy does (the "imitation types"):
  goal       -- the next session's replay of older categories is allocated in
                proportion to how far behind the target I am on each (gap_c +
                FLOOR) instead of uniformly (review toward the goal)
  behaviour  -- in the practice slot, sentences starting from the words of the
                categories where I am behind are produced by the TARGET's
                imagination instead of mine: I practise on what I watch them do
                (their mistakes included)
  strategy   -- I copy the target's learning settings
  diversity limit -- a strategy may be held by at most N/2 individuals; a copy
                that would exceed it is not made (keeps variety in the society)
Conditions: alone / goal / behaviour / strategy / all (best target, single
gap, no limit) / all+near / all+accum / all+limit / all+safe (near + accum +
limit, the "education system" setting).
Measured on the hidden test sentences of the categories met so far: society
mean, best and worst individual, spread (best - worst), number of distinct
strategies.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import data_v2 as D
import lab2 as L
from run_seed_transfer import memorize
from run_unified_continual import sign_p

N_IND = 4
LEVEL, PER_SESSION, CYCLES = "hard", 2, 2
EPOCHS, CYCLE_EPOCHS, PHASE_EPOCHS = 200, 50, 50
HEAR_SHARE = 2 / 3
LRS, REPLAYS = [0.003, 0.01, 0.03], [0.1, 0.25, 0.5]
DELTA, FLOOR, ACCUM_RATE = 0.02, 0.01, 0.5
IMAG_LEN, EPS, TOP_K = 4, 0.3, 5
CONDITIONS = ["alone", "goal", "behaviour", "strategy", "all", "all+near", "all+accum", "all+limit", "all+safe"]


def flags(cond):
    parts = set(cond.split("+"))
    every = "all" in parts
    safe = "safe" in parts
    return {"goal": every or "goal" in parts, "behaviour": every or "behaviour" in parts,
            "strategy": every or "strategy" in parts,
            "near": safe or "near" in parts, "accum": safe or "accum" in parts, "limit": safe or "limit" in parts}


def imagine(model, starts, known, rng):
    """Sentences of IMAG_LEN words from each start word, built from the model's own completions
    (greedy, or one of the top K with probability EPS), using only words it knows."""
    known = torch.tensor(sorted(known))
    sents = torch.tensor(starts).unsqueeze(1)
    with torch.no_grad():
        for _ in range(IMAG_LEN - 1):
            scores = model.complete(sents)[:, known]
            scores[torch.isin(known.unsqueeze(0).expand(len(sents), -1), sents.unsqueeze(-1)).any(1)] = -float("inf")
            order = scores.argsort(dim=1, descending=True)
            explore = torch.tensor(rng.random(len(sents)) < EPS)
            pick = torch.where(explore, order[torch.arange(len(sents)), torch.tensor(rng.integers(0, TOP_K, len(sents)))],
                               order[:, 0])
            sents = torch.cat([sents, known[pick].unsqueeze(1)], 1)
    return sents.tolist()


class Person:
    def __init__(self, d, seed, k, strategy):
        self.k = k
        self.rng = np.random.default_rng(1000 * seed + k)
        memory, _ = memorize(torch.tensor(d.vecs), 1000 * seed + k)
        self.model = L.KnowDo2(d.V, d.n_cat, memory, 1000 * seed + k)
        self.hebb = L.make_internal2("dhebb", d.V)
        self.heard = {}                      # category -> my sentences
        for c, cat in enumerate(d.categories):
            idx = self.rng.permutation(len(cat["heard"]))[:int(round(len(cat["heard"]) * HEAR_SHARE))]
            self.heard[c] = [cat["heard"][i][1] for i in sorted(idx)]
        self.lr, self.replay = strategy
        self.known = set()
        self.V = d.V
        self.accum = {}                      # peer -> running mean of the gap to that peer
        self.target, self.cat_gap = None, {}

    def practise(self, items, ids, cats, epochs):
        ctx, tgt = L.items_tensor(items)
        ctx = L.bag(ctx, self.V)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(self.model.complete(ctx), tgt)
            if ids is not None:
                loss = loss + F.cross_entropy(self.model.classify(ids), cats[ids])
            loss.backward()
            opt.step()

    def learn_connections(self, sentences, new_words):
        self.hebb.observe(sentences)
        conn = self.hebb.connections(sorted(self.known))
        nw = torch.tensor(sorted(new_words))
        with torch.no_grad():
            self.model.assoc[nw, :] = conn[nw, :]
            self.model.assoc[:, nw] = conn[:, nw]


def quiz(model, items_by_cat, cats_seen):
    out = {}
    with torch.no_grad():
        for c in cats_seen:
            ctx, tgt = L.items_tensor(items_by_cat[c])
            out[c] = L.acc(model.complete(ctx), tgt)
    return out


def life(cond, d, seed):
    f = flags(cond)
    cats = torch.tensor(d.cats)
    rng = np.random.default_rng(30_000 + seed)
    grid = [(lr, rp) for lr in LRS for rp in REPLAYS]
    start_strats = [grid[i] for i in rng.permutation(len(grid))[:N_IND]]
    people = [Person(d, seed, k, start_strats[k]) for k in range(N_IND)]
    order = [int(c) for c in rng.permutation(d.n_cat)]
    schedule = [("session", order[i:i + PER_SESSION]) for i in range(0, d.n_cat, PER_SESSION)] + [("cycle", [])] * CYCLES
    public = {c: [(ctx, t) for _, ctx, t in D.unit_items(d.categories[c]["val"])] for c in range(d.n_cat)}
    hidden = {c: [(ctx, t) for _, ctx, t in D.unit_items(d.categories[c]["test"])] for c in range(d.n_cat)}
    seen_cats, steps = [], []
    envy_events = copies = 0

    for i, (kind, new_cats) in enumerate(schedule):
        # practice slot: imagination; with behaviour imitation, the target demonstrates the categories I am behind on
        if i > 0:
            demos = {}
            for p in people:
                behind = [c for c, g in p.cat_gap.items() if g > DELTA] if (f["behaviour"] and p.target is not None) else []
                t = people[p.target] if behind else None
                watched = {w for c in behind for w in d.categories[c]["train"] if w in t.known} if t else set()
                own = sorted(p.known - watched)
                sents = imagine(p.model, own, p.known, p.rng) if own else []
                if watched:
                    sents += imagine(t.model, sorted(watched), t.known, t.rng)
                demos[p.k] = sents
            for p in people:                   # everybody watches first, then practises
                p.practise(D.all_positions([("x", s, []) for s in demos[p.k]]), None, cats, PHASE_EPOCHS)

        old_cats = list(seen_cats)
        seen_cats += new_cats
        for p in people:
            if kind == "session":
                new_sents = [s for c in new_cats for s in p.heard[c]]
                new_words = {w for s in new_sents for w in s}
                p.known |= new_words
                replay = []
                if old_cats:
                    n = int(sum(len(p.heard[c]) for c in old_cats) * p.replay)
                    w = np.array([(max(p.cat_gap.get(c, 0.0), 0.0) + FLOOR) if (f["goal"] and p.target is not None) else 1.0
                                  for c in old_cats])
                    pools = {c: list(p.rng.permutation(len(p.heard[c]))) for c in old_cats}
                    for _ in range(n):
                        live = [j for j, c in enumerate(old_cats) if pools[c]]
                        c = old_cats[live[int(p.rng.choice(len(live), p=w[live] / w[live].sum()))]]
                        replay.append(p.heard[c][pools[c].pop()])
                p.learn_connections(new_sents + replay, new_words)
                ids = torch.tensor(sorted(new_words | {x for s in replay for x in s}))
                p.practise(D.all_positions([("x", s, []) for s in new_sents + replay]), ids, cats, EPOCHS)
            else:
                allsents = [s for c in seen_cats for s in p.heard[c]]
                p.hebb.observe(allsents)
                p.practise(D.all_positions([("x", s, []) for s in allsents]), torch.tensor(sorted(p.known)), cats, CYCLE_EPOCHS)

        # public quiz, visible to everybody
        scores = [quiz(p.model, public, seen_cats) for p in people]
        totals = [float(np.mean(list(s.values()))) for s in scores]
        for p in people:
            gaps = {}
            for q in people:
                if q.k == p.k:
                    continue
                g = totals[q.k] - totals[p.k]
                p.accum[q.k] = p.accum.get(q.k, 0.0) + ACCUM_RATE * (g - p.accum.get(q.k, 0.0))
                gaps[q.k] = p.accum[q.k] if f["accum"] else g
            ahead = {q: g for q, g in gaps.items() if g > DELTA}
            if not ahead:
                p.target, p.cat_gap = None, {}
                continue
            envy_events += 1
            p.target = (min if f["near"] else max)(ahead, key=ahead.get)
            p.cat_gap = {c: scores[p.target][c] - scores[p.k][c] for c in seen_cats}
        if f["strategy"]:
            # everybody decides on the settings as they were at the quiz; copies are made one person at a time
            # (random order) so the diversity limit sees the copies already made
            before = {p.k: (p.lr, p.replay) for p in people}
            for k in rng.permutation(N_IND):
                p = people[int(k)]
                if p.target is None:
                    continue
                s = before[p.target]
                if s == (p.lr, p.replay):
                    continue
                holders = sum((q.lr, q.replay) == s for q in people)
                if f["limit"] and holders + 1 > N_IND // 2:
                    continue
                p.lr, p.replay = s
                copies += 1

        test = []
        for p in people:
            items = [it for c in seen_cats for it in hidden[c]]
            ctx, tgt = L.items_tensor(items)
            with torch.no_grad():
                test.append(L.acc(p.model.complete(ctx), tgt))
        steps.append({"mean": float(np.mean(test)), "best": max(test), "worst": min(test),
                      "spread": max(test) - min(test), "quiz": float(np.mean(totals)),
                      "strategies": len({(p.lr, p.replay) for p in people}),
                      "envy": envy_events, "copies": copies,
                      "each": [((p.lr, p.replay), a) for p, a in zip(people, test)]})
    return steps


def one_seed(seed):
    torch.set_num_threads(1)
    d = D.build(LEVEL, seed)
    return {c: life(c, d, seed) for c in CONDITIONS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_steps = len(results[0][CONDITIONS[0]])
    n_sess = n_steps - CYCLES
    labels = [f"s{i+1}" if i < n_sess else f"c{i - n_sess + 1}" for i in range(n_steps)]
    print(f"=== {n_seeds} seeds; {N_IND} individuals, level {LEVEL}, {PER_SESSION} categories per session; "
          f"DELTA={DELTA} FLOOR={FLOOR} ACCUM_RATE={ACCUM_RATE} hear {HEAR_SHARE:.2f}; columns: " + " ".join(labels) + " ===")
    for key in ["mean", "best", "worst", "spread", "strategies", "copies"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            row = [np.mean([r[c][i][key] for r in results]) for i in range(n_steps)]
            print(f"{c:10} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired (life mean / end), society mean, best, worst, spread ---")
    for g, base in [(c, "alone") for c in CONDITIONS[1:]] + [(c, "all") for c in CONDITIONS[5:]]:
        cells = []
        for key in ["mean", "best", "worst", "spread"]:
            for label, get in [("life", lambda r, gg: np.mean([st[key] for st in r[gg]])), ("end", lambda r, gg: r[gg][-1][key])]:
                dd = np.array([get(r, g) - get(r, base) for r in results])
                n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
                cells.append(f"{key[:5]}-{label} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{g:10} vs {base:6} " + "  ".join(cells))


if __name__ == "__main__":
    main()
