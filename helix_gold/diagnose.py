"""Read-only self-diagnosis for the developer (the fixed OUTER layer, roadmap
step 2, 2026-09-23).

Rules (README "개발 목표와 로드맵", safety structure):
  * no learning code imports this file; it is run by the developer from
    outside the loop
  * it only reads: every tensor is detached and copied, nothing is written back
  * it reports numbers and never judges -- drifting away from the original
    memory can be normal learning ("preservation is not the goal"); the
    developer decides what the numbers mean
  * it may use the known structure of the synthetic data (which words truly
    belong together), because it is the developer's instrument, never a
    training signal

What it looks at, against the things that never change (the original Module A
memory, the frozen decoder, a fixed reference test):
  drift        -- per word, how far the working vector has moved from its
                  original memory (relative distance, cosine to the original)
  shape        -- whether the relations between words still look like the
                  original: correlation of the pairwise-distance matrices
                  (scatter / rearrangement), mean pairwise cosine and
                  participation ratio of the vectors (collapse: everything
                  squeezed into a few directions)
  meaning      -- read the working vectors through the frozen decoder: does
                  each vector still decode closest to its own word (identity
                  kept) and how large is the reconstruction error
  categories   -- do the vectors still separate the categories
                  (leave-one-out nearest category centre)
  connections  -- the connection table: how much of its positive weight sits
                  on truly related pairs (signal share), how many true pairs
                  are positive (coverage), negative weight
  skill        -- design D's per-word production skill (spread)
  reference    -- accuracy on a fixed reference test (never-heard sentences
                  of every category), overall and per relation kind
  reliance     -- where the answers actually come from: the drop in reference
                  accuracy when, in a COPY, the word vectors are shuffled
                  among words / put back to the original memory, the
                  connection table is emptied, or the skill is flattened to
                  its mean. (Note: "shape corr" is scale-free, so a uniform
                  collapse keeps it; "pairwise cos" is the collapse signal.)

Usage from a developer harness:
    import diagnose
    base = diagnose.Reference(data, seed)          # original memory + frozen decoder
    report = diagnose.diagnose(model, base, words) # dict of numbers
    print(diagnose.format_report({"trained": report, ...}))
`python diagnose.py [seeds]` runs the tool check: a trained learner and
deliberately damaged copies of it, to see whether each kind of damage shows up
in a different place.
"""
import math
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import data_v2 as D
from run_seed_transfer import memorize, LATENT, RAW_DIM

RELATION_KINDS = {"pair", "triple", "cross", "chain-ab", "chain-bc", "poly", "poly-other"}


class Reference:
    """The things that never change: original Module A memory, the frozen decoder, the reference test, the true relations."""

    def __init__(self, data, seed):
        self.data = data
        self.vecs = torch.tensor(data.vecs)
        self.memory, _ = memorize(self.vecs, seed)
        g = torch.Generator().manual_seed(10_000 + seed)       # the same frozen decoder memorize() used
        self.decoder = torch.randn(LATENT, RAW_DIM, generator=g) / math.sqrt(LATENT)
        self.test = [it for c in data.categories for it in D.unit_items(c["test"])]
        self.true_pairs = set()
        for c in data.categories:
            for kind, ws in c["units"]:
                if kind in RELATION_KINDS:
                    self.true_pairs |= {(a, b) for a in ws for b in ws if a != b}


def _upper(m):
    i, j = torch.triu_indices(len(m), len(m), 1)
    return m[i, j]


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))


def _participation_ratio(z):
    ev = torch.linalg.eigvalsh(torch.cov(z.T)).clamp(min=0)
    return float(ev.sum() ** 2 / (ev ** 2).sum().clamp(min=1e-12))


def _mean_pairwise_cos(z):
    zn = F.normalize(z, dim=1)
    return float(_upper(zn @ zn.T).mean())


def _category_separation(z, cats):
    """leave-one-out nearest category centre accuracy."""
    hits = 0
    labels = sorted(set(cats.tolist()))
    sums = {c: z[cats == c].sum(0) for c in labels}
    counts = {c: int((cats == c).sum()) for c in labels}
    for i in range(len(z)):
        best, best_d = None, float("inf")
        for c in labels:
            n = counts[c] - (1 if cats[i] == c else 0)
            if n == 0:
                continue
            centre = (sums[c] - (z[i] if cats[i] == c else 0)) / n
            d = float((z[i] - centre).norm())
            if d < best_d:
                best, best_d = c, d
        hits += best == cats[i]
    return hits / len(z)


def diagnose(model, ref, words=None):
    """model: an object with z_work / assoc (and optionally skill) and complete(), or a dict of those tensors.
    words: the words to look at (default: every word that occurs in heard sentences)."""
    get = (lambda k: model.get(k)) if isinstance(model, dict) else (lambda k: getattr(model, k, None))
    z = get("z_work").detach().clone()
    assoc = get("assoc").detach().clone()
    skill = get("skill")
    if words is None:
        words = sorted({w for c in ref.data.categories for _, s, _ in c["heard"] for w in s})
    w = torch.tensor(words)
    zw, mw = z[w], ref.memory[w]
    out = {}

    # drift
    rel = (zw - mw).norm(dim=1) / mw.norm(dim=1).clamp(min=1e-12)
    out["drift mean"] = float(rel.mean())
    out["drift max"] = float(rel.max())
    out["cos to original"] = float(F.cosine_similarity(zw, mw, dim=1).mean())

    # shape
    out["shape corr"] = _corr(_upper(torch.cdist(zw, zw)), _upper(torch.cdist(mw, mw)))
    out["pairwise cos"] = _mean_pairwise_cos(zw)
    out["pairwise cos (orig)"] = _mean_pairwise_cos(mw)
    out["part. ratio"] = _participation_ratio(zw)
    out["part. ratio (orig)"] = _participation_ratio(mw)

    # meaning through the frozen decoder
    dec = torch.sigmoid(zw @ ref.decoder)
    out["decode error"] = float(((dec - ref.vecs[w]) ** 2).mean())
    nearest = torch.cdist(dec, ref.vecs).argmin(dim=1)
    out["decode identity"] = float((nearest == w).float().mean())

    # categories
    cats = torch.tensor(ref.data.cats)[w]
    out["category sep"] = _category_separation(zw, cats)
    out["category sep (orig)"] = _category_separation(mw, cats)

    # connections
    a = assoc[w][:, w]
    pos = a.clamp(min=0)
    idx = {int(x): k for k, x in enumerate(words)}
    true = [(idx[p], idx[q]) for p, q in ref.true_pairs if p in idx and q in idx]
    ti = torch.tensor(true) if true else torch.zeros((0, 2), dtype=torch.long)
    true_mass = float(pos[ti[:, 0], ti[:, 1]].sum()) if len(ti) else 0.0
    out["conn signal share"] = true_mass / max(float(pos.sum()), 1e-12)
    out["conn coverage"] = float((a[ti[:, 0], ti[:, 1]] > 0).float().mean()) if len(ti) else float("nan")
    out["conn density"] = float((a != 0).float().mean())
    out["conn negative"] = float((-a).clamp(min=0).sum() / max(float(a.abs().sum()), 1e-12))

    # skill
    if skill is not None:
        s = F.softplus(skill.detach().clone())[w]
        out["skill mean"] = float(s.mean())
        out["skill spread"] = float(s.std() / s.mean().clamp(min=1e-12))

    # fixed reference test
    if hasattr(model, "complete"):
        import lab2 as L
        with torch.no_grad():
            res = L.evaluate_units(model, ref.test)
        out["reference all"] = res["all"]
        for k in sorted(res):
            if k != "all":
                out[f"reference {k}"] = res[k]
        out.update(_reliance(model, ref, res["all"]))
    return out


def _reliance(model, ref, acc_now):
    import copy
    import lab2 as L
    out = {}
    swaps = {"z shuffled": lambda m: m.z_work.copy_(m.z_work[torch.randperm(len(m.z_work), generator=torch.Generator().manual_seed(0))]),
             "z original": lambda m: m.z_work.copy_(ref.memory),
             "conn emptied": lambda m: m.assoc.zero_()}
    if getattr(model, "skill", None) is not None:
        swaps["skill flat"] = lambda m: m.skill.fill_(float(m.skill.mean()))
    for name, swap in swaps.items():
        m = copy.deepcopy(model)
        with torch.no_grad():
            swap(m)
            out[f"reliance {name}"] = acc_now - L.evaluate_units(m, ref.test)["all"]
    return out


def format_report(reports):
    """reports: {label: diagnose() output} -> aligned text table, one column per label."""
    labels = list(reports)
    keys = [k for k in reports[labels[0]]]
    width = max(10, max(len(l) for l in labels))
    lines = [f"{'':22}" + "".join(f"{l:>{width + 2}}" for l in labels)]
    for k in keys:
        lines.append(f"{k:22}" + "".join(f"{reports[l].get(k, float('nan')):>{width + 2}.3f}" for l in labels))
    return "\n".join(lines)


# ---------------------------------------------------------------- tool check

def _tool_check(seed):
    """Train a design-D learner on the hard data, then damage copies of it in different ways."""
    import copy
    import lab2 as L
    torch.set_num_threads(1)
    d = D.build("hard", seed)
    ref = Reference(d, seed)
    cats = torch.tensor(d.cats)
    heard = [s for c in d.categories for s in c["heard"]]
    words = sorted({w for c in d.categories for w in c["train"]})
    seen = sorted({w for _, s, _ in heard for w in s} | set(words))
    model = L.KnowDo2(d.V, d.n_cat, ref.memory, seed)
    ih = L.make_internal2("dhebb", d.V)
    ih.observe([s for _, s, _ in heard])
    with torch.no_grad():
        model.assoc.copy_(ih.connections(seen))
    reports = {"start": diagnose(model, ref, seen)}
    L.train(model, D.all_positions(heard), words, cats, 200)
    reports["trained"] = diagnose(model, ref, seen)
    rng = torch.Generator().manual_seed(seed)

    noisy = copy.deepcopy(model)                      # scatter: large random moves of every vector
    with torch.no_grad():
        noisy.z_work.add_(torch.randn(noisy.z_work.shape, generator=rng) * noisy.z_work.std())
    reports["noise z"] = diagnose(noisy, ref, seen)

    squeezed = copy.deepcopy(model)                   # collapse: every vector pulled toward the common mean
    with torch.no_grad():
        mean = squeezed.z_work.mean(0, keepdim=True)
        squeezed.z_work.copy_(mean + 0.1 * (squeezed.z_work - mean))
    reports["collapse z"] = diagnose(squeezed, ref, seen)

    shuffled = copy.deepcopy(model)                   # connections scrambled (same values, wrong places)
    with torch.no_grad():
        perm = torch.randperm(d.V, generator=rng)
        shuffled.assoc.copy_(shuffled.assoc[perm][:, perm])
    reports["shuffle conn"] = diagnose(shuffled, ref, seen)

    misled = copy.deepcopy(model)                     # long training on nonsense: random words of all categories
    nonsense = [[int(x) for x in torch.randperm(len(seen), generator=rng)[:5]] for _ in range(len(heard))]
    nonsense = [[seen[i] for i in s] for s in nonsense]
    L.train(misled, D.all_positions([("x", s, []) for s in nonsense]), words, cats, 200)
    reports["nonsense train"] = diagnose(misled, ref, seen)
    return reports


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(min(4, n)) as pool:
        runs = pool.map(_tool_check, range(n))
    mean = {lab: {k: float(np.mean([r[lab][k] for r in runs])) for k in runs[0][lab]} for lab in runs[0]}
    print(f"=== tool check, hard level, design D + both seeds, {n} seeds (mean) ===")
    print(format_report(mean))
