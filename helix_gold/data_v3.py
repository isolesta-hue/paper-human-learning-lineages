"""A world whose relations repeat (roadmap step 5, 2026-09-23): the same few kinds of relation recur in every
category with different words, and show in the words' features, so that what was learned about older
categories can be carried over to new ones.

Words: like data_v2's hard level (4 super-categories x 4 categories x 28 words, features = super centre +
category offset + word noise, RAW_DIM 40).
Relation kinds ("templates"): R0..R3, each a fixed random offset DELTA_k in feature space. A template pair
(a, b) of kind k: b's features = a's features + DELTA_k + small noise (clipped to [0, 1]). R0-R2 occur in
every category; R3 is NEW -- it occurs only in the second half of the categories (the life presents the
first half first), so a learner has to make a new template for it.
Per category:
  template pairs   -- two pairs of each kind present (R0, R1, R2, and R3 in the second half): one heard in HEARD
                      sentences, the other ("once") in ONE sentence only -- from one sentence, experience alone cannot
                      tell the partner from the fillers of that sentence; a learner with the template can (one-shot
                      learning, like rats that learned a schema, Tse et al. 2007)
  unique pair      -- one pair with no template (b is an ordinary word): has to be learned as it is
  unseen pair      -- one more pair of kind R0, whose two words are heard ONLY apart (each in filler
                      sentences), never together; tested by hiding one given the other. Only a learner that
                      carries the template over can get it (chance otherwise).
  fillers          -- the remaining words (12 in the first half, 8 in the second)
Sentences: a heard pair + 1-4 fillers; each heard pair gets HEARD sentences; 10% noise sentences. Test
items: a unit member hidden in never-heard filler combinations (as data_v2), plus the unseen pair.
The random world (schema=False) is built identically except that no pair follows a template (b is an
ordinary word) -- same counts, same sentences, same tests. The shifting world (shift=True) is the schema world
whose rules change in the second half: every relation kind gets a NEW direction there (templates learned in the
first half become wrong).
Context worlds (finding 58, choosing templates by context):
  local=True    -- relations belong to places: every super-category has its OWN direction for R0-R2 (like each sport
                   having its own rules); R3 (the new kind) is shared by all. 13 directions instead of 4.
  back=True     -- rules change and come back: the first half uses rule set A, then half of the second-half
                   categories use rule set B (new directions for every kind), then the other half returns to A
                   (w.phases gives the presentation order: first half, B, back to A).
  conj=True     -- (finding 62) every category also has an XOR block of 6 more words: cues x, y, heads a, d, answers
                   b, c with (x, a) -> b, (y, a) -> c, (x, d) -> c, (y, d) -> b, each heard in HEARD sentences. Each
                   word alone is tied to both answers equally; only the two TOGETHER tell the answer (negative
                   patterning / configural learning, Pearce 1994). Tested by hiding the answer ("conj" items).
  long=True     -- (finding 64) a 3-word phrase block of 7 more words: (u, v, w) -> t, and every 2-word part of it also
                   belongs to another phrase: (u, v) -> s1, (v, w) -> s2, (u, w) -> s3, each heard in HEARD sentences.
                   Any two of u, v, w point to t and to another answer equally; only the three together tell t.
                   Tested by hiding t ("long" items). (Finding 64: the pairwise habit solves this by ADDING its
                   pairwise ties -- t is tied to three of the words, s1 to two -- so it is not a true 3-word pattern.)
  compose=True  -- (finding 71) every category also has a composed pair: c = a + DELTA_k1 + DELTA_k2 (two relation kinds
                   of the category in a row), heard only apart, tested by hiding one given the other ("compose" items)
                   -- only a learner that can chain templates (A -> B -> C) can get it.
  n_cat, shift2 -- (the long test, 2026-09-26) a longer life: n_cat=32 categories (8 per super-category, 16 sessions); with
                   shift2 the life has two changes -- the first half (rule set A), then the new kind R3 arrives (c % 4 == 1,
                   rule set A), then EVERY relation kind changes direction (c % 4 == 3, rule set B), presented in that order.
  share=True    -- (finding 111, world ladder rung 1, 2026-10-03) words shared across topics, as in real text: N_SHARED
                   common words (features near the global centre, belonging to no category -- category id -100, which the
                   classification loss ignores) with Zipf frequencies (p ~ 1/rank). From the SECOND half of the life on
                   (the categories presented in sessions 9-16) every filler of a heard or test sentence becomes a common word
                   with probability P_SHARE. The base world is built first with the same draws (a separate generator for
                   the common words), so with share=False nothing changes. New sentences now move connections that old
                   sentences also use (the interference the categories' own words never had, finding 107).
  blind=True    -- (world ladder rung 2, 2026-10-04) words without senses, as in real language where only concrete words
                   are grounded ("dog") and the rest ("yesterday", "and", "kind") are learned from context: in every category
                   of the SECOND half, a share P_BLIND of its words (any role -- pair members included) lose their features
                   (replaced by the global centre + word noise, like the common words: they tell nothing about category or
                   relation). A separate generator after the base world, so blind=False changes nothing. Meant on top of
                   rung 1 (share=True): the ladder is cumulative.
  parity=True   -- (finding 65) a true 3-word pattern: slots {x1, x2}, {y1, y2}, {z1, z2}; one word from each slot, and the
                   answer is b if the three choices have even parity, c otherwise (8 combinations x PARITY_HEARD sentences).
                   Every single word AND every pair of words is tied to b and c equally; only all three tell. Tested
                   by hiding the answer (counted as "long" items).
"""
import numpy as np

RAW_DIM = 40
N_SUPER, N_CAT, N_WORDS = 4, 16, 28
N_KINDS, NEW_KIND = 4, 3
DELTA_NORM, PAIR_NOISE = 1.0, 0.03        # |DELTA| comparable to a word's own noise (0.15 x sqrt(40) = 0.95)
HEARD, TEST, NOISE, FILL = 6, 3, 0.1, (1, 4)
N_SHARED, P_SHARE = 48, 0.5                     # world ladder rung 1 (finding 111); representative values, declared
P_BLIND = 0.5                                   # world ladder rung 2: share of a second-half category's words without senses
PARITY_HEARD = 3                                # parity block: 8 combinations x 3 sentences (finding 65)


class World:
    """vecs (V x RAW_DIM), cats, n_cat, V, categories (list of dicts), first_half / second_half, deltas."""


def directions(rng, n):
    d = rng.normal(0, 1, (n, RAW_DIM))
    return d * DELTA_NORM / np.linalg.norm(d, axis=1, keepdims=True)


def build(seed, schema=True, shift=False, local=False, back=False, conj=False, long=False, parity=False, compose=False,
          n_cat=N_CAT, shift2=False, share=False, blind=False):
    rng = np.random.default_rng(seed)
    w = World()
    w.schema = schema
    super_centres = rng.uniform(0.3, 0.7, (N_SUPER, RAW_DIM))
    w.deltas = directions(rng, N_KINDS)
    if shift or back or shift2:                     # the world's rules change: second-half categories get new directions
        w.deltas2 = directions(rng, N_KINDS)
    if local:                                       # every super-category has its own R0-R2 (R3 stays shared)
        w.local_deltas = directions(rng, N_SUPER * NEW_KIND).reshape(N_SUPER, NEW_KIND, RAW_DIM)

    def rule(c, k):
        """(direction id, direction) of relation kind k in category c."""
        second = c % 2 == 1
        if local and k < NEW_KIND:
            s = c // (n_cat // N_SUPER)
            return N_KINDS + s * NEW_KIND + k, w.local_deltas[s, k]
        if (shift and second) or (back and c % 4 == 1) or (shift2 and c % 4 == 3):
            return N_KINDS + k, w.deltas2[k]
        return k, w.deltas[k]

    w.pair_dirs = []
    vecs, cats = [], []
    w.categories = []
    per_super = n_cat // N_SUPER
    for c in range(n_cat):
        centre = np.clip(super_centres[c // per_super] + rng.normal(0, 0.12, RAW_DIM), 0, 1)
        second = c % 2 == 1                         # every other category belongs to the second half
        kinds = list(range(NEW_KIND)) + ([NEW_KIND] if second else [])
        ids = []

        def word(base=None, kind=None):
            if base is not None and schema:
                d_id, delta = rule(c, kind)
                v = np.clip(base + delta + rng.normal(0, PAIR_NOISE, RAW_DIM), 0, 1)
                w.pair_dirs.append((ids[-1], len(vecs), d_id))
            else:
                v = np.clip(centre + rng.normal(0, 0.15, RAW_DIM), 0, 1)
            vecs.append(v)
            cats.append(c)
            ids.append(len(vecs) - 1)
            return len(vecs) - 1

        units = []
        for k in kinds:
            a = word()
            units.append((f"R{k}", [a, word(vecs[a], k)]))
            a = word()
            units.append(("once", [a, word(vecs[a], k)]))       # same kind, heard in ONE sentence only
        a = word()
        units.append(("unique", [a, word()]))
        a = word()
        unseen = ("unseen", [a, word(vecs[a], 0)])
        combos = []
        if conj:                                    # the answer depends on two context words TOGETHER (XOR)
            x, y, a, d, b, c2 = (word() for _ in range(6))
            combos = [[x, a, b], [y, a, c2], [x, d, c2], [y, d, b]]
        phrases = []
        if long:                                    # a 3-word phrase whose every 2-word part also belongs to another phrase
            u, v, x3, t, s1, s2, s3 = (word() for _ in range(7))
            phrases = [[u, v, x3, t], [u, v, s1], [v, x3, s2], [u, x3, s3]]
        comp = None
        if compose and schema:                      # a pair made of TWO relations in a row, never heard together
            k1, k2 = (int(x) for x in rng.choice(len(kinds), 2, replace=False))
            a = word()
            d1, v1 = rule(c, kinds[k1])
            d2, v2 = rule(c, kinds[k2])
            b = len(vecs)
            vecs.append(np.clip(vecs[a] + v1 + v2 + rng.normal(0, PAIR_NOISE, RAW_DIM), 0, 1))
            cats.append(c)
            ids.append(b)
            comp = ("compose", [a, b])
        par = []
        if parity:                                  # 3-word parity: one word from each of three slots decides the answer
            x1, x2, y1, y2, z1, z2, b, c2 = (word() for _ in range(8))
            for i1, xx in enumerate((x1, x2)):
                for i2, yy in enumerate((y1, y2)):
                    for i3, zz in enumerate((z1, z2)):
                        par.append([xx, yy, zz, b if (i1 + i2 + i3) % 2 == 0 else c2])
        while len(ids) < N_WORDS + (6 if conj else 0) + (7 if long else 0) + (8 if parity else 0) + (2 if comp else 0):
            word()
        fillers = [i for i in ids if not any(i in ws for _, ws in units + [unseen] + [("conj", cb) for cb in combos]
                                             + [("long", ph) for ph in phrases] + [("par", p) for p in par]
                                             + ([comp] if comp else []))]
        heard, test = [], []
        for kind, ws in units:
            seen = set()
            for bucket, n in ((heard, 1 if kind == "once" else HEARD), (test, TEST)):
                made = 0
                while made < n:
                    k = int(rng.integers(FILL[0], FILL[1] + 1))
                    fill = tuple(sorted(int(x) for x in rng.choice(fillers, k, replace=False)))
                    if fill in seen:
                        continue
                    seen.add(fill)
                    s = list(ws) + list(fill)
                    bucket.append((kind, [s[i] for i in rng.permutation(len(s))], list(ws)))
                    made += 1
        for x in unseen[1]:                         # the unseen pair's words: heard only apart, as fillers
            for _ in range(2):
                k = int(rng.integers(FILL[0], FILL[1] + 1))
                s = [x] + [int(f) for f in rng.choice(fillers, k, replace=False)]
                heard.append(("apart", [s[i] for i in rng.permutation(len(s))], []))
        for _ in range(TEST):
            k = int(rng.integers(FILL[0], FILL[1] + 1))
            s = list(unseen[1]) + [int(f) for f in rng.choice(fillers, k, replace=False)]
            test.append(("unseen", [s[i] for i in rng.permutation(len(s))], list(unseen[1])))
        for cb in combos:                           # cue + head -> answer; tested by hiding the answer
            seen = set()
            for bucket, n in ((heard, HEARD), (test, TEST)):
                made = 0
                while made < n:
                    k = int(rng.integers(FILL[0], FILL[1] + 1))
                    fill = tuple(sorted(int(f) for f in rng.choice(fillers, k, replace=False)))
                    if fill in seen:
                        continue
                    seen.add(fill)
                    s = list(cb) + list(fill)
                    bucket.append(("conj", [s[i] for i in rng.permutation(len(s))], [cb[2]]))
                    made += 1
        for j, ph in enumerate(phrases):            # all heard; only the 3-word phrase's answer is tested
            seen = set()
            for bucket, n in ((heard, HEARD), (test, TEST if j == 0 else 0)):
                made = 0
                while made < n:
                    k = int(rng.integers(FILL[0], FILL[1] + 1))
                    fill = tuple(sorted(int(f) for f in rng.choice(fillers, k, replace=False)))
                    if fill in seen:
                        continue
                    seen.add(fill)
                    s = list(ph) + list(fill)
                    bucket.append(("long", [s[i] for i in rng.permutation(len(s))], [ph[-1]]))
                    made += 1
        if comp:                                    # heard only apart (as fillers), tested together, like the unseen pair
            for x in comp[1]:
                for _ in range(2):
                    k = int(rng.integers(FILL[0], FILL[1] + 1))
                    s = [x] + [int(f) for f in rng.choice(fillers, k, replace=False)]
                    heard.append(("apart", [s[i] for i in rng.permutation(len(s))], []))
            for _ in range(TEST):
                k = int(rng.integers(FILL[0], FILL[1] + 1))
                s = list(comp[1]) + [int(f) for f in rng.choice(fillers, k, replace=False)]
                test.append(("compose", [s[i] for i in rng.permutation(len(s))], list(comp[1])))
        for p in par:                               # each of the 8 combinations heard PARITY_HEARD times, answer tested
            seen = set()
            for bucket, n in ((heard, PARITY_HEARD), (test, 1)):
                made = 0
                while made < n:
                    k = int(rng.integers(FILL[0], FILL[1] + 1))
                    fill = tuple(sorted(int(f) for f in rng.choice(fillers, k, replace=False)))
                    if fill in seen:
                        continue
                    seen.add(fill)
                    s = list(p) + list(fill)
                    bucket.append(("long", [s[i] for i in rng.permutation(len(s))], [p[-1]]))
                    made += 1
        for _ in range(int(round(len(heard) * NOISE))):
            heard.append(("noise", [int(x) for x in rng.choice(ids, 4, replace=False)], []))
        w.categories.append({"units": units, "unseen": unseen, "fillers": fillers, "heard": heard, "test": test,
                             "second": second, "words": ids, "conj": combos, "long": phrases})
    if share:                                       # rung 1: common words, Zipf, in the second half's fillers
        srng = np.random.default_rng(seed + 1_000_003)
        centre = super_centres.mean(0)
        common = list(range(len(vecs), len(vecs) + N_SHARED))
        for _ in common:
            vecs.append(np.clip(centre + srng.normal(0, 0.15, RAW_DIM), 0, 1))
            cats.append(-100)
        zipf = 1.0 / np.arange(1, N_SHARED + 1)
        zipf /= zipf.sum()
        for cat in w.categories:
            if not cat["second"]:
                continue
            fill = set(cat["fillers"])
            for bucket in (cat["heard"], cat["test"]):
                for j, (kind, sent, ws) in enumerate(bucket):
                    if kind == "noise":
                        continue
                    slots = [i for i, x in enumerate(sent) if x in fill and srng.random() < P_SHARE]
                    if not slots:
                        continue
                    picks = srng.choice(common, len(slots), replace=False, p=zipf)
                    sent = list(sent)
                    for i, x in zip(slots, picks):
                        sent[i] = int(x)
                    bucket[j] = (kind, sent, ws)
        w.common = common
    if blind:                                       # rung 2: half the words of the second half lose their senses
        brng = np.random.default_rng(seed + 2_000_003)
        centre = super_centres.mean(0)
        w.blind = []
        for cat in w.categories:
            if not cat["second"]:
                continue
            ids_c = list(cat["words"])
            for x in brng.choice(ids_c, int(round(len(ids_c) * P_BLIND)), replace=False):
                vecs[int(x)] = np.clip(centre + brng.normal(0, 0.15, RAW_DIM), 0, 1)
                w.blind.append(int(x))
    w.vecs = np.array(vecs, dtype=np.float32)
    w.cats = np.array(cats)
    w.V, w.n_cat = len(vecs), n_cat
    w.first_half = [c for c in range(n_cat) if not w.categories[c]["second"]]
    w.second_half = [c for c in range(n_cat) if w.categories[c]["second"]]
    w.phases = [w.first_half, w.second_half]
    if back or shift2:                              # shift2: the new kind R3 arrives (c % 4 == 1), then every rule changes
                                                    # (c % 4 == 3) -- two changes in one life (the long test, 2026-09-26)
        w.phases = [w.first_half, [c for c in w.second_half if c % 4 == 1], [c for c in w.second_half if c % 4 == 3]]
    return w


def true_pairs(w):
    """developer view: direction id -> list of (a, b) template pairs (heard, once and unseen); random world: by kind."""
    if w.schema:
        out = {}
        for a, b, d in w.pair_dirs:
            out.setdefault(d, []).append((a, b))
        return out
    out = {k: [] for k in range(N_KINDS)}
    for cat in w.categories:
        kinds_here = [int(kind[1:]) for kind, _ in cat["units"] if kind.startswith("R")]
        for (kind, ws), k in zip([u for u in cat["units"] if u[0] in ("once",) or u[0].startswith("R")],
                                 [k for k in kinds_here for _ in range(2)]):
            out[k].append(tuple(ws))
        out[0].append(tuple(cat["unseen"][1]))
    return out


def unit_items(sentences):
    """(kind, context, hidden) for every unit member of every test sentence."""
    items = []
    for kind, s, ws in sentences:
        if kind in ("noise", "apart"):
            continue
        for x in ws:
            items.append((kind, [y for y in s if y != x], x))
    return items


def describe(w):
    n_heard = sum(len(c["heard"]) for c in w.categories)
    d = np.linalg.norm(w.deltas, axis=1)
    return (f"{'schema' if w.schema else 'random'} world: {w.V} words, {w.n_cat} categories, {n_heard} heard sentences, "
            f"|DELTA| {d.round(2).tolist()}, word spread {np.linalg.norm(w.vecs - w.vecs.mean(0), axis=1).mean():.2f}")


if __name__ == "__main__":
    for sch in (True, False):
        print(describe(build(0, sch)))
