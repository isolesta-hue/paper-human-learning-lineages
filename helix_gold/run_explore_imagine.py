"""Exploring imagination: hypotheses, checked later against real experience.

Finding 32: imagination that always takes the most expected word only
re-confirms what the internal memory already believes. People also try
"what if it is like this?" and later check it against what really happens.

Between steps (sessions, then everyday cycles) the internal logic imagines
one sentence per known word, as in finding 32, but each added word is
  greedy   -- always the most expected word, or
  explore  -- with probability EPS a random one of the TOP_K most expected
              (a less certain guess), otherwise the most expected.
Every word pair in an imagined sentence is a hypothesis "these go together",
stamped with the step it was made in. At the next step, after the real
sentences of that step are heard, each hypothesis whose two words were both
heard is checked, and graded: the evidence for a -> b is the share of the
heard sentences holding a that also hold b (always together = 1, sometimes =
in between, never = 0). The stored connection a -> b moves a fraction ETA of
the way toward that evidence, on the scale of the other connection seeds
(log(share) + log(number of known words), >= 0); same for b -> a. A pair
that was heard but never together is pulled toward 0 (refuted); one that
always comes together is pulled up (confirmed). No outside feedback is used:
only the heard sentences themselves ("input = answer"). Only pairs the
internal logic thought of get checked -- so what it can learn depends on
what it imagines, which is where exploration matters.
(A first version counted a pair as confirmed if the two words shared even
one sentence and added +1 / -1. Partner-less words that share a sentence now
and then got as much credit as real partners, and the already good Hebbian
memories got worse, 0.69 -> 0.45.)
The external logic's values do NOT flow back (compare loop off), so any
growth of the internal memory comes from checking its own hypotheses.
The external logic rehearses the imagined sentences as in finding 32.

Modes: imagine (finding 32: rehearse, never check) / check-greedy /
check-explore, each checking on the evidence of ONE step (finding 36), and
check-greedy-hist / check-explore-hist, which check on the whole time-ordered
record of experience so far (every sentence heard at every step, kept on
the time axis) -- so one step with little evidence (e.g. old words that only
showed up in the 25% replay) cannot swing a well-founded connection. Life: 3 sessions + 4 everyday cycles (as finding 33).
Conditions: none / vec / hebb / dhebb / vec+bert. External logic: design A.
Measured per step: share of hypotheses confirmed; imagination quality
(true partner pair / one category); the internal memory alone on
never-heard sentences; the external logic on never-heard sentences.
"""
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from run_unified_continual import STAGES, sign_p
from run_seed_transfer import memorize, EPOCHS, REPLAY
from run_connection_seed import masked_items, partner_items, acc, SENT_LEN
from run_ratchet_seed import build_data
from run_three_designs import ValueOnly
from run_role2_target import predict
from run_role3_imagine import PHASE_EPOCHS
from run_role3b_experience import quality, CYCLES, CYCLE_EPOCHS
from run_role1_anchor import make_internal2

CONDITIONS = ["none", "vec", "hebb", "dhebb", "vec+bert"]
MODES = ["imagine", "check-greedy", "check-explore", "check-greedy-hist", "check-explore-hist"]
EPS, TOP_K, ETA = 0.3, 5, 0.5


def imagine(z, assoc, seen_words, rng, eps):
    seen = torch.tensor(seen_words)
    sentences = []
    for startw in rng.permutation(seen_words):
        sent = [int(startw)]
        while len(sent) < SENT_LEN:
            with torch.no_grad():
                scores = predict(z, assoc, torch.tensor([sent]))[0, seen]
            scores[torch.isin(seen, torch.tensor(sent))] = -float("inf")
            order = scores.argsort(descending=True)
            pick = order[int(rng.integers(min(TOP_K, len(order) - len(sent))))] if rng.random() < eps else order[0]
            sent.append(int(seen[pick]))
        sentences.append(sent)
    return sentences


def run(condition, mode, data, memory, seed):
    vecs, cats, split, partner = data
    start = memory if condition.startswith("vec") else torch.randn_like(memory) * memory.std()
    model = ValueOnly(start, seed)
    internal2 = make_internal2(condition, memory, seed)
    stored = {"z_work": start.clone(), "assoc": torch.zeros_like(model.assoc)}
    rng = np.random.default_rng(20_000 + seed)
    phase_rng = np.random.default_rng(40_000 + seed)
    seen_cats, seen_words, steps, pending = [], [], [], []
    schedule = [("session", st) for st in STAGES] + [("cycle", None)] * CYCLES
    eps = EPS if mode.startswith("check-explore") else 0.0
    record = []             # time-ordered record of heard sentences: (step, sentence)

    def practise(ctx, tgt, ids, epochs):
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model.complete(ctx), tgt)
            if ids is not None:
                loss = loss + F.cross_entropy(model.classify(ids), cats[ids])
            loss.backward()
            opt.step()

    for i, (kind, new_cats) in enumerate(schedule):
        step = {}
        if i > 0:
            sents = imagine(stored["z_work"], stored["assoc"], seen_words, phase_rng, eps)
            step["pair rate"], step["one category"] = quality(sents, partner, cats)
            if mode != "imagine":
                pending = [(a, b) for s in sents for x, a in enumerate(s) for b in s[x + 1:]]
            pctx, ptgt, _, _ = masked_items(sents)
            practise(pctx, ptgt, None, PHASE_EPOCHS)

        if kind == "session":
            old_cats = list(seen_cats)
            seen_cats += new_cats
            new_words = [w for c in new_cats for w in split[c]["train"]]
            seen_words += new_words
            old_train = [w for c in old_cats for w in split[c]["train"]]
            replay_words = [int(w) for w in rng.choice(old_train, int(len(old_train) * REPLAY), replace=False)] if old_train else []
            new_sents = [s for c in new_cats for s in split[c]["sentences"]]
            old_sents = [s for c in old_cats for s in split[c]["sentences"]]
            replay_sents = [old_sents[j] for j in rng.choice(len(old_sents), int(len(old_sents) * REPLAY), replace=False)] if old_sents else []
            heard = new_sents + replay_sents
            ids = torch.tensor(new_words + replay_words)
            epochs = EPOCHS
            if internal2 is not None:
                internal2.observe(heard)
                conn = internal2.connections(seen_words)
                nw = torch.tensor(new_words)
                stored["assoc"][nw, :] = conn[nw, :]
                stored["assoc"][:, nw] = conn[:, nw]
                with torch.no_grad():
                    model.assoc[nw, :] = conn[nw, :]
                    model.assoc[:, nw] = conn[:, nw]
        else:
            heard = [s for c in seen_cats for s in split[c]["sentences"]]
            if internal2 is not None:
                internal2.observe(heard)
            ids = torch.tensor(seen_words)
            epochs = CYCLE_EPOCHS

        record += [(i, snt) for snt in heard]
        if pending:
            n_with, n_both = {}, {}
            evidence = [snt for _, snt in record] if mode.endswith("-hist") else heard
            for snt in evidence:
                for a in set(snt):
                    n_with[a] = n_with.get(a, 0) + 1
                    for b in set(snt):
                        if a != b:
                            n_both[(a, b)] = n_both.get((a, b), 0) + 1
            log_n = np.log(len(seen_words))
            checked = confirmed = 0
            for a, b in set(pending):
                if a in n_with and b in n_with:
                    checked += 1
                    confirmed += (a, b) in n_both
                    for x, y in ((a, b), (b, a)):
                        share = n_both.get((x, y), 0) / n_with[x]
                        target = max(0.0, np.log(share) + log_n) if share > 0 else 0.0
                        old = float(stored["assoc"][x, y])
                        stored["assoc"][x, y] = old + ETA * (target - old)
            step["confirmed"] = confirmed / checked if checked else float("nan")
            pending = []

        ctx, tgt, _, _ = masked_items(heard)
        practise(ctx, tgt, ids, epochs)

        test = partner_items([s for c in seen_cats for s in split[c]["test_sents"]], partner)
        with torch.no_grad():
            step["never-heard"] = acc(model.complete(test[0]), test[1])
            step["internal never-heard"] = acc(predict(stored["z_work"], stored["assoc"], test[0]), test[1])
        # how well the stored connections separate a word's true partner from the
        # partner-less words of its category (diagnostic for the checking step)
        with torch.no_grad():
            pp, pf = [], []
            for c in seen_cats:
                fill = [w for w in split[c]["train"] if w not in partner]
                for w in split[c]["train"]:
                    if w in partner:
                        pp.append(float(stored["assoc"][w, partner[w]]))
                        pf += [float(stored["assoc"][w, f]) for f in fill]
            step["to partner"], step["to filler"] = float(np.mean(pp)), float(np.mean(pf))
        steps.append(step)
    return steps


def one_seed(seed):
    torch.set_num_threads(1)
    data = build_data(seed)
    memory, _ = memorize(data[0], seed)
    return {(c, md): run(c, md, data, memory, seed) for c in CONDITIONS for md in MODES}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_steps = len(results[0][(CONDITIONS[0], MODES[0])])
    labels = [f"s{i+1}" if i < len(STAGES) else f"c{i - len(STAGES) + 1}" for i in range(n_steps)]

    def m(vals):
        vals = np.array(vals, dtype=float)
        return f"{np.nanmean(vals):.3f}" if (~np.isnan(vals)).any() else "  -  "

    print(f"=== {n_seeds} seeds, EPS={EPS}, TOP_K={TOP_K}, ETA={ETA}; columns: " + " ".join(labels) + " ===")
    for key in ["confirmed", "pair rate", "one category", "internal never-heard", "never-heard", "to partner", "to filler"]:
        print(f"\n--- {key} ---")
        for c in CONDITIONS:
            for md in MODES:
                row = [m([r[(c, md)][i].get(key, np.nan) for r in results]) for i in range(n_steps)]
                print(f"{c:9} {md:13} " + " ".join(f"{v:>6}" for v in row))

    print("\n--- paired at the end of life: internal never-heard / external never-heard ---")
    for c in CONDITIONS:
        for a, b in [("check-greedy", "imagine"), ("check-explore", "imagine"), ("check-explore", "check-greedy"),
                     ("check-greedy-hist", "check-greedy"), ("check-explore-hist", "check-explore"),
                     ("check-explore-hist", "imagine")]:
            cells = []
            for key in ["internal never-heard", "never-heard"]:
                d = np.array([r[(c, a)][-1][key] - r[(c, b)][-1][key] for r in results])
                n_pos, n_neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                cells.append(f"{key} {d.mean():+.3f} ({n_pos}/{n_neg}, p={sign_p(n_pos, n_neg):.2g})")
            print(f"{c:9} {a} vs {b:13} " + "  ".join(cells))


if __name__ == "__main__":
    main()
