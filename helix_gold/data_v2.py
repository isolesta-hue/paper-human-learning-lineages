"""Harder synthetic word data with a difficulty ladder (project decision,
2026-09-23: the old task -- 72 words, one partner per word, 4-word sentences --
saturates near 1.0 and squeezes the differences between methods).

Everything is synthetic on purpose: the true structure is known exactly, so
every error can be attributed to a relation type.

Words
  super-categories -> categories -> words. A word's raw features = its
  super-category centre + its category offset + word noise, so categories of
  the same super-category look alike (hierarchy).
Relation units inside a category (the thing to learn):
  pair    -- two words that always come together ("mom - dad")
  triple  -- three words that always come together ("breakfast - lunch - dinner")
  chain   -- a - b and b - c come together, a - c NEVER do (tested on a -> c:
             can the learner link what it never heard together?)
  cross   -- a word of this category that always comes with a word of ANOTHER
             category ("doctor - hospital"); feature similarity does not help
  poly    -- one word that belongs to two categories and has a different
             partner in each ("bank": river / money); the right partner
             depends on the rest of the sentence
Sentences: one relation unit + FILL..FILL_MAX partner-less words of the same
category, shuffled; length therefore varies. A share NOISE of the heard
sentences are random words of the category (wrong experience).
Held out: for each unit, some filler combinations are never heard; test items
hide one unit member in those never-heard sentences.

Levels (the ladder):
  easy   -- 8 categories x 16 words, pairs + triples, 1-2 fillers, no noise
  medium -- 12 categories x 20 words, + cross, 1-3 fillers, 10% noise
  hard   -- 16 categories x 24 words, + chain + poly, 1-4 fillers, 20% noise
"""
import numpy as np

RAW_DIM = 40
LEVELS = {
    "easy":   dict(n_super=2, n_cat=8,  n_words=16, units=["pair", "pair", "triple"],
                   fill=(1, 2), noise=0.0),
    "medium": dict(n_super=3, n_cat=12, n_words=20, units=["pair", "pair", "triple", "cross"],
                   fill=(1, 3), noise=0.1),
    "hard":   dict(n_super=4, n_cat=16, n_words=24, units=["pair", "triple", "cross", "chain", "poly"],
                   fill=(1, 4), noise=0.2),
}
N_VAL_WORDS, N_TEST_WORDS = 2, 2          # per category, for word classification
HEARD, VAL, TEST = 6, 2, 3                # sentences per relation (unit part) per category


class Data:
    """vecs (V x RAW_DIM), cats (V,), categories: list of dicts, and helpers."""


def build(level, seed):
    cfg = LEVELS[level]
    rng = np.random.default_rng(seed)
    per_super = cfg["n_cat"] // cfg["n_super"]
    super_centres = rng.uniform(0.25, 0.75, (cfg["n_super"], RAW_DIM))
    vecs, cats, supers = [], [], []
    cat_words = []
    for c in range(cfg["n_cat"]):
        s = c // per_super
        centre = np.clip(super_centres[s] + rng.normal(0, 0.12, RAW_DIM), 0, 1)
        ids = []
        for _ in range(cfg["n_words"]):
            vecs.append(np.clip(centre + rng.normal(0, 0.15, RAW_DIM), 0, 1))
            cats.append(c)
            supers.append(s)
            ids.append(len(vecs) - 1)
        cat_words.append([int(i) for i in rng.permutation(ids)])

    d = Data()
    d.level, d.cfg = level, cfg
    d.vecs = np.array(vecs, dtype=np.float32)
    d.cats = np.array(cats)
    d.supers = np.array(supers)
    d.V, d.n_cat = len(vecs), cfg["n_cat"]
    d.categories = []

    # word pools per category
    pools = []
    for c in range(cfg["n_cat"]):
        words = cat_words[c]
        pools.append({"val": words[:N_VAL_WORDS], "test": words[N_VAL_WORDS:N_VAL_WORDS + N_TEST_WORDS],
                      "free": words[N_VAL_WORDS + N_TEST_WORDS:]})

    # relation units; cross partners come from the next category, poly words are shared with the next category
    units_by_cat = [[] for _ in range(cfg["n_cat"])]
    for c in range(cfg["n_cat"]):
        other = (c + 1) % cfg["n_cat"]
        for kind in cfg["units"]:
            take = lambda k, cc=c: [pools[cc]["free"].pop() for _ in range(k)]
            if kind == "pair":
                units_by_cat[c].append(("pair", take(2)))
            elif kind == "triple":
                units_by_cat[c].append(("triple", take(3)))
            elif kind == "chain":
                a, b, cw = take(3)
                units_by_cat[c].append(("chain-ab", [a, b]))
                units_by_cat[c].append(("chain-bc", [b, cw]))
                units_by_cat[c].append(("chain-test", [a, cw]))       # never heard together
            elif kind == "cross":
                mine = take(1)[0]
                theirs = pools[other]["free"].pop()
                units_by_cat[c].append(("cross", [mine, theirs]))
            elif kind == "poly":
                shared = take(1)[0]
                partner_here = take(1)[0]
                partner_there = pools[other]["free"].pop()
                units_by_cat[c].append(("poly", [shared, partner_here]))
                units_by_cat[other].append(("poly-other", [shared, partner_there]))

    for c in range(cfg["n_cat"]):
        fillers = list(pools[c]["free"])
        train_words = sorted({w for _, ws in units_by_cat[c] for w in ws if d.cats[w] == c} | set(fillers))
        heard, val, test = [], [], []
        for kind, ws in units_by_cat[c]:
            if kind == "chain-test":
                a, cw = ws
                for _ in range(TEST):
                    k = int(rng.integers(cfg["fill"][0], cfg["fill"][1] + 1))
                    s = [a, cw] + [int(x) for x in rng.choice(fillers, k, replace=False)]
                    test.append((kind, [s[i] for i in rng.permutation(len(s))], [a, cw]))
                continue
            seen = set()
            for bucket, n in ((heard, HEARD), (val, VAL), (test, TEST)):
                made = 0
                while made < n:
                    k = int(rng.integers(cfg["fill"][0], cfg["fill"][1] + 1))
                    fill = tuple(sorted(int(x) for x in rng.choice(fillers, k, replace=False)))
                    if fill in seen:
                        if len(seen) > 200:
                            break
                        continue
                    seen.add(fill)
                    s = list(ws) + list(fill)
                    bucket.append((kind, [s[i] for i in rng.permutation(len(s))], list(ws)))
                    made += 1
        n_noise = int(round(len(heard) * cfg["noise"] / max(1e-9, 1 - cfg["noise"])))
        for _ in range(n_noise):
            k = int(rng.integers(3, 3 + cfg["fill"][1]))
            s = [int(x) for x in rng.choice(train_words, min(k, len(train_words)), replace=False)]
            heard.append(("noise", s, []))
        d.categories.append({"train": train_words, "val_words": pools[c]["val"], "test_words": pools[c]["test"],
                             "units": units_by_cat[c], "fillers": fillers,
                             "heard": heard, "val": val, "test": test})
    return d


def unit_items(sentences):
    """Test items: hide each unit member that has its unit partner(s) in the sentence.
    Returns list of (kind, context_words, hidden_word)."""
    items = []
    for kind, s, ws in sentences:
        if kind == "noise":
            continue
        if kind == "chain-test":
            a, cw = ws
            items.append((kind, [w for w in s if w != cw], cw))
            items.append((kind, [w for w in s if w != a], a))
            continue
        for w in ws:
            items.append((kind, [x for x in s if x != w], w))
    return items


def all_positions(sentences):
    """Training items: every position of every heard sentence (noise included)."""
    return [([x for j, x in enumerate(s) if j != i], w) for _, s, _ in sentences for i, w in enumerate(s)]


def describe(d):
    n_heard = sum(len(c["heard"]) for c in d.categories)
    lens = [len(s) for c in d.categories for _, s, _ in c["heard"]]
    kinds = sorted({k for c in d.categories for k, _, _ in c["test"]})
    return (f"{d.level}: {d.V} words, {d.n_cat} categories, {n_heard} heard sentences "
            f"(length {min(lens)}-{max(lens)}), test kinds {kinds}")


if __name__ == "__main__":
    for lv in LEVELS:
        print(describe(build(lv, 0)))
