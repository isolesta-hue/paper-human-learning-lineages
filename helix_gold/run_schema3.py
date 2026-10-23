"""Content layer, third experiment: trust templates by their predictions, and let habit and understanding be weighed
by reliability (finding 53 follow-up, 2026-09-24).

Finding 53: useless templates are confirmed often, so counting confirmations cannot filter them (superstition, like
Skinner's pigeons); and practice overwrites the templates' first-sight understanding because the trained parts
(the "habit": connection table, skill) grow while the pattern channel stays the same size. Lessons taken from the
research on changing habits (discussion with the user): habits are not erased but compete with new learning
(Bouton); change needs a prediction error (reconsolidation; confident errors are corrected best, Butterfield &
Metcalfe 2001); habits are weakest where the situation is new (Verplanken & Wood) -- so trust the template for new
words and the habit for well-practised ones.
  prediction trust -- at the start of a session, before any feedback, every active template predicts the partner of
                      each new word (the new word it matches best, match > 0.5). After the session the learner's own
                      experience scores it: right if the two were heard together (P(b|a) >= STRONG), wrong otherwise.
                      trust = (right + 1) / (predictions + 2) (0.5 = not sure yet); a template with >= 3 predictions
                      and trust < 0.5 rests (kept, not used). Replaces the confirmation count n/(n+1).
                      A template that has been usable for NO_PREDICTION_REST sessions without ever predicting anything
                      for new words also rests: it only explains the pairs it was made from (an after-the-fact
                      explanation, not a prediction -- first check: the useless templates never predict on new words,
                      so the prediction score alone left them at "not sure" and they kept disturbing old pairs).
  arbitration      -- the habit's scores (design D's answer) are standardised per question (mean 0, spread 1), so that
                      however much practice inflates them they keep the same size, and their ranking is unchanged;
                      the template evidence is added as log(#known) x best over context words a of
                      (template weight x match) x (1 - e_a), e_a = n_a / (n_a + 1) for a word heard n_a times -- the
                      habit's experience with that word: a template counts most where the habit has least experience.
                      (First attempt: mixing the two as probabilities, p = (1-g) softmax(habit) + g q. On new contexts
                      the habit's probability sat, overconfidently, on some old word and the template never won --
                      first-sight understanding fell from ~0.2 to ~0.03 on a check seed. Dropped.)
  control          -- D-norm: the standardised habit alone, to separate the effect of standardising from the templates.
Learners (same seeds, schema and random world): D / D-norm / pattern+explain (finding 53's best) / trust+explain /
norm+explain / trust+norm+explain.
Representative values: rest below trust 0.5 after 3 predictions or after 3 usable sessions without a prediction;
Laplace prior (1, 2).
"""
import math
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import data_v3 as W
import lab2 as L
from run_seed_transfer import memorize
from run_unified_continual import sign_p
from run_schema import (PatternMemory, PatternKnowDo, ConfDirectionalHebbian, acc_items, D_positions,
                        EPOCHS, REVIEW, LR, PER_SESSION, CHECKS, STRONG, WORLDS, MIN_SEEN, VIGILANCE)
from run_schema2 import explain_weights

LEARNERS = ["D", "D-norm", "pattern+explain", "trust+explain", "norm+explain", "trust+norm+explain"]
REST_AFTER, REST_BELOW, NO_PREDICTION_REST = 3, 0.5, 3


class TrustPatternMemory(PatternMemory):
    """templates: [direction, confirmations, predictions, right]."""

    def __init__(self, use_trust, mdl=False, min_count=2):
        super().__init__()
        self.use_trust = use_trust
        self.session = 0
        self.use_mdl = mdl              # finding 59: a template is used only while it has paid for itself
        self.min_count = min_count      # confirmations to be usable (2; the count control of finding 59 uses more)
        self.bits = {}                  # template -> prequential savings minus its own description length (bits)
        self.desc = {}                  # template -> its description length (bits)
        self.organize = None            # finding 60: None / "merge" / "family"
        self.null = "uniform"           # the no-template code: "uniform" (finding 59) / "near" (finding 60)
        self.groups = []                # finding 61, versions: {"i", "j", "m", "D"} -- separate (i, j) vs merged (m)
        self.in_group = {}              # template -> index of its open version group

    def rank_bits(self, a, b, t, known, memory):
        """bits to name b given a with template t (prequential: t as it was BEFORE this pair): 1 bit for the sign +
        Elias-gamma code of b's rank among the known words by distance to m_a +/- t (rank 1 = the nearest)."""
        kw = torch.tensor(sorted(set(known) - {a}))
        best = None
        for sign in (1, -1):
            d = (memory[kw] - (memory[a] + sign * t)).norm(dim=1)
            r = int((d < float((memory[b] - (memory[a] + sign * t)).norm())).sum()) + 1
            best = r if best is None else min(best, r)
        return 1 + 2 * math.floor(math.log2(best)) + 1

    def absorb(self, hebb, known, memory):
        """as PatternMemory.absorb, and remembers which pairs built which template (self.members), so that a template
        does not vote on the pairs it was made from (finding 56: that would count the same evidence twice)."""
        if not hasattr(self, "members"):
            self.members = {}
        cond = hebb.counts / hebb.with_.clamp(min=1)[:, None]
        for a in known:
            if hebb.with_[a] < MIN_SEEN:
                continue
            for b in torch.nonzero(cond[a] >= STRONG).flatten().tolist():
                key = (min(a, b), max(a, b))
                if b not in known or key in self.done:
                    continue
                self.done.add(key)
                ch = getattr(self, "channel_of", None)
                if ch is not None and ch[a] != ch[b]:
                    continue                        # finding 90: a direction means something only inside one sense's features
                d = memory[key[1]] - memory[key[0]]
                best, best_cos = None, 0.0
                for k, t in enumerate(self.templates):
                    if t[1] == 0:                   # merged into another template (finding 60)
                        continue
                    c = float(F.cosine_similarity(d, t[0], dim=0))
                    ratio = float(d.norm() / t[0].norm().clamp(min=1e-9))
                    if abs(c) > abs(best_cos) and 0.5 <= ratio <= 2.0:
                        best, best_cos = k, c
                if best is not None and abs(best_cos) >= VIGILANCE:
                    t = self.templates[best]
                    # MDL (two-part code): without a template, b is named by its rank among the known words by nearness
                    # to a (finding 60 correction -- finding 59 used log2(#known), naming b uniformly, but a pair's partner
                    # is near a anyway, so that null made every template look cheap); with template `best`: which
                    # template (adaptive frequency code: confirmed n_k of N times costs log2((N + 1) / n_k), the 1 left
                    # for "a new one", PPM method A) + sign + rank by nearness to m_a +/- t
                    N = sum(u[1] for u in self.templates)
                    null = (self.rank_bits(key[0], key[1], torch.zeros_like(t[0]), known, memory) - 1
                            if self.null == "near" else math.log2(len(known)))
                    saved = (null - math.log2((N + 1) / t[1])
                             - self.rank_bits(key[0], key[1], t[0], known, memory))
                    g = self.in_group.get(best)
                    if g is not None:
                        self.version_score(g, key, known, memory)
                    self.bits[best] += saved
                    t[1] += 1
                    t[0] = t[0] + ((d if best_cos > 0 else -d) - t[0]) / t[1]
                    self.members[best].add(key)
                    if g is not None:
                        self.version_mirror(g, best, key, d)
                else:
                    self.templates.append([d.clone(), 1])
                    self.members[len(self.templates) - 1] = {key}
                    # the template's own description: the founding pair it was made from (two words among the known)
                    self.bits[len(self.templates) - 1] = -2 * math.log2(len(known))
                    self.desc[len(self.templates) - 1] = 2 * math.log2(len(known))
        for t in self.templates:
            while len(t) < 4:
                t.append(0)
            if len(t) == 4 and t[1] >= 2:
                t.append(self.session + 1)              # usable from the next session on

    def family(self, k):
        """templates close enough to k to have been assimilated into it (the system's own VIGILANCE and length rule)."""
        t = self.templates[k][0]
        out = []
        for j, u in enumerate(self.templates):
            if u[1] == 0:
                continue
            ratio = float(u[0].norm() / t.norm().clamp(min=1e-9))
            if abs(float(F.cosine_similarity(t, u[0], dim=0))) >= VIGILANCE and 0.5 <= ratio <= 2.0:
                out.append(j)
        return out

    def account(self, k):
        """bits the template has paid back. family (finding 60, pooling): the family's savings together, paying ONE
        description -- as if the pieces were one template, without changing any of them."""
        if self.organize != "family":
            return self.bits.get(k, 0.0)
        fam = self.family(k)
        return sum(self.bits[j] + self.desc[j] for j in fam) - max(self.desc[j] for j in fam)

    def pair_bits(self, pairs, t, known, memory):
        return sum(self.rank_bits(a, b, t, known, memory) for a, b in pairs)

    def merge(self, known, memory):
        """finding 60, merging (consolidation): two templates become one if that describes their pairs in fewer bits --
        one description instead of two, against the extra bits of coding both groups of pairs with the shared direction:
        merge if bits(all pairs | merged) - bits(pairs_i | t_i) - bits(pairs_j | t_j) < description. Candidates: every
        live template's nearest (by |cos|) live template; the closest pairs are tried first; repeated until nothing merges.
        The merged direction is the count-weighted mean (sign-aligned); its account = both accounts + the refunded
        description."""
        merged_any = True
        while merged_any:
            merged_any = False
            live = [k for k, t in enumerate(self.templates) if t[1] > 0]
            if len(live) < 2:
                return
            dirs = torch.stack([self.templates[k][0] for k in live])
            cos = F.cosine_similarity(dirs.unsqueeze(1), dirs.unsqueeze(0), dim=2)
            cos.fill_diagonal_(0)
            best = cos.abs().argmax(dim=1)
            cands = sorted({(min(i, int(j)), max(i, int(j))) for i, j in enumerate(best.tolist())},
                           key=lambda p: -abs(float(cos[p[0], p[1]])))
            gone = set()
            for i, j in cands:
                if i in gone or j in gone:
                    continue
                ki, kj = live[i], live[j]
                ti, tj = self.templates[ki], self.templates[kj]
                sign = 1.0 if float(cos[i, j]) > 0 else -1.0
                tm = (ti[0] * ti[1] + sign * tj[0] * tj[1]) / (ti[1] + tj[1])
                pi, pj = self.members[ki], self.members[kj]
                extra = (self.pair_bits(pi | pj, tm, known, memory) - self.pair_bits(pi, ti[0], known, memory)
                         - self.pair_bits(pj, tj[0], known, memory))
                d = min(self.desc[ki], self.desc[kj])
                if extra < d:
                    ti[0] = tm
                    ti[1] += tj[1]
                    tj[1] = 0
                    self.members[ki] = pi | pj
                    self.members[kj] = set()
                    self.bits[ki] += self.bits[kj] + d
                    self.desc[ki] = max(self.desc[ki], self.desc[kj])
                    gone |= {i, j}
                    merged_any = True

    def add_pair(self, k, key, d):
        t = self.templates[k]
        c = float(F.cosine_similarity(d, t[0], dim=0))
        t[1] += 1
        t[0] = t[0] + ((d if c > 0 else -d) - t[0]) / t[1]
        self.members[k].add(key)

    def propose_versions(self, known, memory):
        """finding 61 (the user's proposal: conclude fast but keep it open): every merge the consolidation rule of finding
        60 would make is NOT made; instead the merged template is added as a second version next to the two separate
        ones, and both versions are judged on the pairs that come AFTER (prequential)."""
        live = [k for k, t in enumerate(self.templates) if t[1] > 0 and k not in self.in_group]
        if len(live) < 2:
            return
        dirs = torch.stack([self.templates[k][0] for k in live])
        cos = F.cosine_similarity(dirs.unsqueeze(1), dirs.unsqueeze(0), dim=2)
        cos.fill_diagonal_(0)
        best = cos.abs().argmax(dim=1)
        cands = sorted({(min(i, int(j)), max(i, int(j))) for i, j in enumerate(best.tolist())},
                       key=lambda p: -abs(float(cos[p[0], p[1]])))
        taken = set()
        for i, j in cands:
            if i in taken or j in taken:
                continue
            ki, kj = live[i], live[j]
            ti, tj = self.templates[ki], self.templates[kj]
            sign = 1.0 if float(cos[i, j]) > 0 else -1.0
            tm = (ti[0] * ti[1] + sign * tj[0] * tj[1]) / (ti[1] + tj[1])
            pi, pj = self.members[ki], self.members[kj]
            extra = (self.pair_bits(pi | pj, tm, known, memory) - self.pair_bits(pi, ti[0], known, memory)
                     - self.pair_bits(pj, tj[0], known, memory))
            d = min(self.desc[ki], self.desc[kj])
            if extra < d:
                km = len(self.templates)
                self.templates.append([tm, ti[1] + tj[1], 0, 0])
                self.members[km] = pi | pj
                self.bits[km] = self.bits[ki] + self.bits[kj] + d
                self.desc[km] = max(self.desc[ki], self.desc[kj])
                self.groups.append({"i": ki, "j": kj, "m": km, "D": 0.0})
                for k in (ki, kj, km):
                    self.in_group[k] = len(self.groups) - 1
                taken |= {i, j}

    def version_score(self, g, key, known, memory):
        """before a new pair updates anything: bits to name its partner with the separate versions (the better of the
        two + 1 bit for which) and with the merged one; D += separate - merged (D > 0: the merged version predicts
        better). Both versions start equal (D = 0)."""
        G = self.groups[g]
        sep = min(self.rank_bits(key[0], key[1], self.templates[G[k]][0], known, memory) for k in ("i", "j")) + 1
        G["D"] += sep - self.rank_bits(key[0], key[1], self.templates[G["m"]][0], known, memory)

    def version_mirror(self, g, best, key, d):
        """the pair joined `best`; the other version takes it in too (the merged one, or the closer separate one), so
        both versions keep learning. A version behind by more than a description is dropped."""
        G = self.groups[g]
        if best == G["m"]:
            ci, cj = (abs(float(F.cosine_similarity(d, self.templates[G[k]][0], dim=0))) for k in ("i", "j"))
            self.add_pair(G["i"] if ci >= cj else G["j"], key, d)
        else:
            self.add_pair(G["m"], key, d)
        if abs(G["D"]) > self.desc[G["m"]]:
            losers = (G["i"], G["j"]) if G["D"] > 0 else (G["m"],)
            for k in losers:
                self.templates[k][1] = 0
            for k in (G["i"], G["j"], G["m"]):
                self.in_group.pop(k, None)

    def in_use_version(self, k):
        """a template in an open group is used only on the leading side (ties: the separate, older versions)."""
        g = self.in_group.get(k)
        if g is None:
            return True
        G = self.groups[g]
        return (k == G["m"]) == (G["D"] > 0)

    def weighted(self):
        """(index, direction, weight) of the templates in use."""
        out = []
        for k, t in enumerate(self.templates):
            if t[1] < self.min_count:
                continue
            if not self.in_use_version(k):
                continue
            if self.use_mdl and self.account(k) <= 0:
                continue
            if not self.use_trust:
                out.append((k, t[0], t[1] / (t[1] + 1)))
                continue
            trust = (t[3] + 1) / (t[2] + 2)
            if t[2] >= REST_AFTER and trust < REST_BELOW:
                continue
            if t[2] == 0 and len(t) > 4 and self.session - t[4] >= NO_PREDICTION_REST:
                continue
            out.append((k, t[0], trust))
        return out

    def predict(self, new_words, memory):
        """before feedback: (template index, a, predicted b) for every new word a and template in use."""
        preds = []
        words = torch.tensor(sorted(new_words))
        m = memory[words]
        for k, t, _ in self.weighted():
            best = torch.full((len(words),), -1.0)
            arg = torch.zeros(len(words), dtype=torch.long)
            for sign in (1, -1):
                match = 1 - torch.cdist(m + sign * t, m) / t.norm()
                match.fill_diagonal_(-1)
                v, j = match.max(dim=1)
                better = v > best
                best = torch.where(better, v, best)
                arg = torch.where(better, j, arg)
            for i in range(len(words)):
                if best[i] > 0.5:
                    preds.append((k, int(words[i]), int(words[arg[i]])))
        return preds

    def score(self, preds, hebb):
        cond = hebb.counts / hebb.with_.clamp(min=1)[:, None]
        for k, a, b in preds:
            if hebb.with_[a] < 1:
                continue
            self.templates[k][2] += 1
            self.templates[k][3] += int(cond[a, b] >= STRONG)


class ContradictionTrust(TrustPatternMemory):
    """prediction a -> b: right if heard together, wrong only if a is strongly tied to some OTHER word, else no
    verdict (finding 54: counting "never heard together" as wrong cut the trust of true templates)."""

    def score(self, preds, hebb):
        cond = hebb.counts / hebb.with_.clamp(min=1)[:, None]
        for k, a, b in preds:
            if hebb.with_[a] < 1:
                continue
            if cond[a, b] >= STRONG:
                self.templates[k][2] += 1
                self.templates[k][3] += 1
            else:
                others = cond[a] >= STRONG
                others[b] = False
                others[a] = False
                if bool(others.any()):
                    self.templates[k][2] += 1


def _rows(M, idx):
    """M[idx] for a dense or sparse (COO) V x V matrix; idx an integer tensor of any shape."""
    if not M.is_sparse:
        return M[idx]
    return torch.index_select(M, 0, idx.reshape(-1)).to_dense().reshape(*idx.shape, M.shape[1])


def _max_rows_by_length(get, ctx):
    """max over the positions of get(ctx) (padded positions give rows of zeros), computed per width class so that a few
    long sentences do not make every row as wide as the longest (2026-10-14: W5's stories of ten things made the practice
    batches ~100 wide where most sentences have ~10 words). Exactly the padded result: a row shorter than the batch width
    also takes max(., 0), which is what the zero rows of its padding gave."""
    lens = (ctx >= 0).sum(1)
    W = ctx.shape[1]
    # width classes: the next power of two at or above each row's length (capped at the batch width) -- few classes, so
    # few calls; positions between a row's length and its class width are padding (zero rows), as in the full batch
    width = torch.minimum(torch.pow(2, torch.ceil(torch.log2(lens.clamp(min=1).float()))).long(), torch.tensor(W))
    out = None
    for Wc in torch.unique(width).tolist():
        idx = torch.nonzero(width == Wc).flatten()
        m = get(ctx[idx, :Wc]).max(dim=1).values
        short = lens[idx] < W
        if short.any():
            m = torch.where(short.unsqueeze(1), m.clamp(min=0), m)
        if out is None:
            out = torch.zeros(len(ctx), m.shape[1], dtype=m.dtype)
        out[idx] = m
    return out


def template_matrices(pattern, memory, known, V, independent=False, place=None):
    """[(template index, V x V weight x match (0..1))] for the templates in use, for known word pairs; independent: a
    template gives no evidence on the pairs it was built from; place: {template: (V,) applicability of the template
    where the context word a lives} (finding 58) multiplying row a."""
    return list(iter_template_matrices(pattern, memory, known, V, independent, place))


def iter_template_matrices(pattern, memory, known, V, independent=False, place=None):
    """template_matrices one template at a time (2026-10-21, memory only: conf_matrix keeps a running maximum instead of
    every V x V matrix at once -- with long heard units the templates in use grew to ~100 x 63 MB per session; the values
    and their order are the same, checked equal to the last bit)."""
    words = torch.tensor(sorted(known))
    pos = {int(x): i for i, x in enumerate(words.tolist())}
    m = memory[words]
    for k, t, wgt in pattern.weighted():
        Ck = torch.zeros(V, V)
        for sign in (1, -1):
            match = 1 - torch.cdist(m + sign * t, m) / t.norm()
            match.fill_diagonal_(0)
            if independent:
                for a, b in pattern.members.get(k, ()):
                    if a in pos and b in pos:
                        match[pos[a], pos[b]] = 0
                        match[pos[b], pos[a]] = 0
            val = torch.where(match > 0.5, wgt * match, torch.zeros_like(match))
            if place is not None:
                val = val * place[k][words].unsqueeze(1)
            Ck[words.unsqueeze(1), words.unsqueeze(0)] = torch.maximum(Ck[words][:, words], val)
        yield k, Ck


def conf_matrix(pattern, memory, known, V, independent=False, place=None):
    """weight x match (0..1) of the best template in use, for known word pairs."""
    C = torch.zeros(V, V)
    for _, Ck in iter_template_matrices(pattern, memory, known, V, independent, place):
        C = torch.maximum(C, Ck)
    return C


def place_applicability(pattern, memory, cats, known, V):
    """finding 58, the place axis of the templates' coordinate system: a template's places are the categories whose pairs
    confirmed it (a category's place = the centre of its known words' memory vectors). For a context word in category c,
    template k applies by exp(-(d_k^2 - d_min^2) / 2h^2): d_k = distance from c to k's nearest place, d_min = the same
    for the nearest template (a template confirmed near here, or the nearest there is, counts fully; one confirmed only
    in far places hardly counts), h = median distance from a known category to its nearest other known category."""
    kw = torch.tensor(sorted(known))
    kc = sorted({int(c) for c in cats[kw].tolist()})
    cen = torch.stack([memory[kw[cats[kw] == c]].mean(0) for c in kc])
    idx = {c: i for i, c in enumerate(kc)}
    D = torch.cdist(cen, cen)
    h = float((D + torch.eye(len(kc)) * 1e9).min(dim=1).values.median()) if len(kc) > 1 else 1.0
    used = pattern.weighted()
    d = torch.stack([D[:, sorted({idx[int(cats[a])] for a, _ in pattern.members[k]})].min(dim=1).values for k, _, _ in used])
    app = torch.exp(-(d ** 2 - d.min(dim=0).values ** 2) / (2 * h ** 2))       # (templates, known categories)
    out = {}
    for j, (k, _, _) in enumerate(used):
        v = torch.zeros(V)
        v[kw] = app[j][torch.tensor([idx[int(c)] for c in cats[kw].tolist()])]
        out[k] = v
    return out


def trial(mats, model, nctx, ntgt, n_known, mask_context):
    """finding 58, trying each template on the current situation: on the new sentences just heard (before practice),
    how many answers does this template alone turn wrong -> right (helped) and right -> wrong (harmed)?
    {template: (helped + 1) / (helped + harmed + 2)}."""
    rel = {}
    with torch.no_grad():
        z = model.habit(nctx)
        without = z.argmax(dim=1) == ntgt
        pad = (nctx >= 0).float().unsqueeze(-1)
        cm = context_mask(nctx, z.shape[1]) == 0 if mask_context else None
        for k, Ck in mats:
            boost = (torch.log1p(Ck * (n_known - 1))[nctx.clamp(min=0)] * pad).max(dim=1).values
            if cm is not None:
                boost = boost * cm
            with_t = (z + boost).argmax(dim=1) == ntgt
            h, b = float((with_t & ~without).sum()), float((without & ~with_t).sum())
            rel[k] = (h + 1) / (h + b + 2)
    return rel


def _tail(raw, skill, pterm):
    """the training path's dense tail of NormPatternKnowDo.complete: design D's skill x value, the standardised habit, +
    template evidence -- exactly habit() + pterm, in one function so that torch.compile can fuse its elementwise steps
    (optimisation pass, 2026-09-28: these memory-bound steps on (items x V) matrices were most of the practice time)."""
    v = F.softplus(skill) * raw
    c = v - v.mean(dim=1, keepdim=True)
    return c / (c.pow(2).sum(dim=1, keepdim=True) / (v.shape[1] - 1)).sqrt().clamp(min=1e-6) + pterm


def _tail_known(raw, skill, pterm, m):
    """D26 (1) S1 ("+knownonly"): the same tail, standardised over the words heard so far only (m: 1 = heard). Found by
    the inertness audit (probe_inert.py, 2026-10-10): the standardisation over ALL V words made every word of the world --
    heard or not -- part of every answer's scale, so the never-heard words' values still moved under the answer mask."""
    v = F.softplus(skill) * raw
    n = m.sum()
    c = (v - (v * m).sum(dim=1, keepdim=True) / n) * m
    return c / (c.pow(2).sum(dim=1, keepdim=True) / (n - 1)).sqrt().clamp(min=1e-6) + pterm


FUSED = {"tail": _tail, "tail_known": _tail_known}


def fused(name):
    """the compiled version of a helper (compiled once per process, shapes dynamic); the plain one if compiling fails or
    VOYAGER_COMPILE=0."""
    import os
    f = FUSED[name]
    if os.environ.get("VOYAGER_COMPILE", "1") == "0":
        return f
    key = name + "_compiled"
    if key not in FUSED:
        try:
            FUSED[key] = torch.compile(f, dynamic=True)
        except Exception:
            FUSED[key] = f
    return FUSED[key]


class NormPatternKnowDo(L.KnowDo2):
    """answer = standardised habit (design D) + log(#known) x template evidence gated by the habit's experience."""

    def __init__(self, V, n_cat, start_vectors, seed, use_pattern=True):
        super().__init__(V, n_cat, start_vectors, seed)
        self.use_pattern = use_pattern
        self.register_buffer("C", torch.zeros(V, V))
        self.register_buffer("e", torch.zeros(V))
        self.scale = 1.0

    mask_context = False          # finding 57: no template evidence for words already in the sentence

    cue_C = None                  # finding 90: {cue word: V x V} templates used only when the cue is in the question

    def pattern_term(self, ctx):
        S = self.scale * _max_rows_by_length(
            lambda c: self.C[c.clamp(min=0)] * (1 - self.e[c.clamp(min=0)]).unsqueeze(-1) * (c >= 0).float().unsqueeze(-1), ctx)
        if self.cue_C and not ctx.is_sparse:
            # specific before general (finding 91 follow-up, cue_specific): a question that has a cue with templates is answered
            # by the cue's templates alone -- the plain templates (every direction ever seen) are set aside for it
            if getattr(self, "cue_specific", False):
                cued_rows = torch.zeros(len(ctx), dtype=torch.bool)
                for c in self.cue_C:
                    cued_rows |= (ctx == c).any(dim=1)
                S = torch.where(cued_rows.unsqueeze(1), torch.zeros_like(S), S)
            self.cue_purity = None
            if getattr(self, "cue_pure", False) and self.kind_of is not None:
                # the purest cue first (finding 91 follow-up): among the cues in a question, for the kinds the question
                # has, a cue whose confirmations sit in ONE template (a verb: one direction) outranks a cue whose
                # confirmations spread over several (an object's picture: every verb heard with it). purity = largest n /
                # total n of the cue's templates that start from a kind present in the question. Ties keep all.
                kinds_q = [set(self.kind_of[r[r >= 0]].tolist()) for r in ctx]
                pur = {}
                for c, Cc in self.cue_C.items():
                    if not isinstance(Cc, list):
                        continue
                    has_c = (ctx == c).any(dim=1)
                    p = torch.zeros(len(ctx))
                    for i in torch.nonzero(has_c).flatten().tolist():
                        ns = [n for _, _, n, src in Cc if src & kinds_q[i]]
                        p[i] = max(ns) / sum(ns) if ns else 0.0
                    pur[c] = p
                if pur:
                    top = torch.stack(list(pur.values())).max(dim=0).values
                    self.cue_purity = {c: (p >= top) & (p > 0) for c, p in pur.items()}
            for c, Cc in self.cue_C.items():
                has = (ctx == c).any(dim=1)
                if not has.any():
                    continue
                sub = ctx[has]
                if not isinstance(Cc, list):
                    r = _max_rows_by_length(lambda c: _rows(Cc, c.clamp(min=0)) * (c >= 0).float().unsqueeze(-1), sub)
                    S[has] = torch.maximum(S[has], self.scale * r)
                    continue
                # templates with a cue and a place: the ones confirmed where the question is asked, else all of them
                pl = torch.where(sub >= 0, self.place_of[sub.clamp(min=0)], torch.full_like(sub, -1))
                here = [torch.tensor([bool(P) and bool(torch.isin(pl_i, torch.tensor(sorted(P))).any()) for pl_i in pl])
                        for _, P, _, _ in Cc]
                local = torch.stack(here).any(dim=0) if here else torch.zeros(len(sub), dtype=torch.bool)
                best = torch.zeros(len(sub), S.shape[1])
                if getattr(self, "cue_purity", None) is not None:
                    keep = self.cue_purity[c][has]
                    here = [h & keep for h in here]
                    local = local & keep
                    no = ~keep
                else:
                    no = torch.zeros(len(sub), dtype=torch.bool)
                for (Ct, _, n_t, _), h in zip(Cc, here):
                    use = (h | ~local) & ~no
                    if use.any():
                        r = _max_rows_by_length(lambda c: _rows(Ct, c.clamp(min=0)) * (c >= 0).float().unsqueeze(-1),
                                                sub[use])
                        best[use] = torch.maximum(best[use], r)
                S[has] = torch.maximum(S[has], self.scale * best)
        if self.mask_context:
            S = S * (context_mask(ctx, S.shape[1]) == 0)
        return S

    temps = None                  # calibration (finding 56): list of (upper edge of least-familiar experience, temperature)

    def habit(self, ctx):
        v = super().complete(ctx)
        km = getattr(self, "known_mask", None)
        if km is not None:                          # D26 (1) S1: standardised over the heard words only (see _tail_known)
            m = km.float()
            n = m.sum()
            c = (v - (v * m).sum(dim=1, keepdim=True) / n) * m
            return c / (c.pow(2).sum(dim=1, keepdim=True) / (n - 1)).sqrt().clamp(min=1e-6)
        # the same standardisation as (v - mean) / std (unbiased), written out: PyTorch's CPU std is ~5x slower than this
        # (optimisation pass, 2026-09-28; equal up to float rounding, ~1e-6)
        c = v - v.mean(dim=1, keepdim=True)
        return c / (c.pow(2).sum(dim=1, keepdim=True) / (v.shape[1] - 1)).sqrt().clamp(min=1e-6)

    def assoc_share(self, ctx):
        """the connection table's share of the standardised habit (finding 112's exact split, before the temperature)."""
        if getattr(self, "attn", None) is not None:
            b = L.attention_bag(self, ctx)
            am = torch.sparse.mm(b, self.assoc)
            zb = torch.sparse.mm(b, self.z_work)
        else:
            am, zb = L.masked_mean(self.assoc, ctx), L.masked_mean(self.z_work, ctx)
        k = F.softplus(self.skill)
        v = k * (F.normalize(zb, dim=-1) @ F.normalize(self.z_work, dim=-1).T) + k * am
        km = getattr(self, "known_mask", None)
        if km is not None:                          # D26 (1) S1: over the heard words only, as habit()
            m = km.float()
            n = m.sum()
            sd = ((((v - (v * m).sum(1, keepdim=True) / n) * m).pow(2).sum(1, keepdim=True)) / (n - 1)).sqrt().clamp(min=1e-6)
            c = k * am
            return (c - (c * m).sum(1, keepdim=True) / n) * m / sd
        sd = ((v - v.mean(1, keepdim=True)).pow(2).sum(1, keepdim=True) / (v.shape[1] - 1)).sqrt().clamp(min=1e-6)
        c = k * am
        return (c - c.mean(1, keepdim=True)) / sd

    def temperature(self, ctx):
        """per question: the temperature of the bin of its least familiar context word (1 if not calibrated)."""
        if self.temps is None:
            return torch.ones(len(ctx))
        e_min = torch.where(ctx >= 0, self.e[ctx.clamp(min=0)], torch.ones_like(ctx, dtype=torch.float)).min(dim=1).values
        T = torch.ones(len(ctx))
        lower = -1.0
        for upper, t in self.temps:
            T = torch.where((e_min > lower) & (e_min <= upper), torch.full_like(T, t), T)
            lower = upper
        return T

    chunks = None                 # finding 62: whole-sentence memory (ChunkMemory), used when answering
    chunk_on = False
    route = False                 # finding 81: whole-sentence memory routed per question: "rare" (chunk_gate) / "trust"
    agree2 = False                # tier 2: the second vote
    side_w = 1.0                  # its weight when answering (finding 86 (a): a method-layer knob)
    side_on = False               # the self-made side part (model.side, V x V) is created when self-made practice uses it
    cooc = None                   # the Hebbian co-occurrence counts (V x V), set by the learner each session
    freq = None                   # sentences each word was heard in (V), set with cooc

    def chunk_gate(self, ctx, z0):
        """finding 81 (the reaction-norm map: whole-sentence memory helps on questions whose answer was heard with the
        question's words, and gets in the way of the templates on questions whose two words were only ever heard apart):
        memory can only testify about what it has seen. Per question: the answer the other routes pick (habit + templates,
        z0) -- has it EVER occurred in a sentence with any word of the question? If not, whole-sentence memory has nothing
        to say about it and its evidence is left out for this question. No constant."""
        # first version: "with ANY word of the question" -- never fired (the category's filler words co-occur with nearly
        # everything). The question's most telling word is its RAREST (fewest sentences; the idf logic of finding 63):
        top = z0.masked_fill(context_mask(ctx, z0.shape[1]) > 0, float("-inf")).argmax(1)
        f = torch.where(ctx >= 0, self.freq[ctx.clamp(min=0)].float(), torch.full(ctx.shape, float("inf")))
        rare = ctx.gather(1, f.argmin(dim=1, keepdim=True)).squeeze(1)
        return ((self.cooc[rare, top] + self.cooc[top, rare]) > 0).float().unsqueeze(1)

    def complete(self, ctx, pterm=None):
        if ctx.is_sparse and self.use_pattern and pterm is not None and self.temps is None:
            # practice: the same sum as below, through the fused tail
            km = getattr(self, "known_mask", None)
            if km is not None:                      # D26 (1) S1: standardised over the heard words only
                return self.only_known(fused("tail_known")(L.ValueOnly2.complete(self, ctx), self.skill, pterm, km.float()))
            return self.only_known(fused("tail")(L.ValueOnly2.complete(self, ctx), self.skill, pterm))
        v = self.habit(ctx)
        aw = getattr(self, "assoc_w", 1.0)
        if not ctx.is_sparse and aw != 1.0:
            # finding 115: the connection table's weight when answering (a method-layer setting; 1 = as before) -- its
            # share of the standardised habit, exactly as run_long.answer_parts splits it
            v = v - (1.0 - aw) * self.assoc_share(ctx)
        if not ctx.is_sparse:                       # calibration acts when answering, not on the habit's training
            v = v / self.temperature(ctx).unsqueeze(1)
            if self.chunk_on and self.chunks is not None:
                ev = self.chunks.evidence(ctx)
                if self.route:
                    z0 = v + (self.pattern_term(ctx) if self.use_pattern and pterm is None else
                              (pterm if self.use_pattern else 0))
                    ev = ev * (self.chunk_gate(ctx, z0) if self.route == "rare" else self.chunks.route_gate(ctx, z0, ev))
                if self.agree2:
                    # tier 2 (the second vote, 2026-09-29, the user: counting the same evidence twice is a check, and a
                    # slight push is fine): where the memory agrees with the other routes' answer, its evidence counts twice
                    z0 = v + (self.pattern_term(ctx) if self.use_pattern and pterm is None else
                              (pterm if self.use_pattern else 0))
                    cm = context_mask(ctx, ev.shape[1]) > 0
                    agree = z0.masked_fill(cm, float("-inf")).argmax(1) == ev.masked_fill(cm, float("-inf")).argmax(1)
                    ev = ev * (1 + agree.float()).unsqueeze(1)
                v = v + getattr(self, "chunk_w", 1.0) * ev
        if not ctx.is_sparse and self.side_on:
            v = v + self.side_w * L.masked_mean(self.side, ctx)   # the self-made side part (answering: main + w x side)
        wm_t = None
        # the leaning test (2026-10-10, "+wmtest"; findings 137-140's smokes): with wm.test_only set, the world model does
        # not answer during practice (passes that build a gradient) -- it answers only when the learner is asked (tests and
        # its own no-gradient decisions); off = unchanged
        # (the practice passes take sparse bags, so the world model never answered in them anyway: "+wmtest" is a no-op by
        # construction -- finding 141's smoke. "+wmmeasure": wm.silent through a whole session -- the learner's own
        # no-gradient choices (what to practise again, settings, options) do not see the part; it answers only when measured)
        if not ctx.is_sparse and getattr(self, "wm", None) is not None and not getattr(self.wm, "silent", False) and \
                not (getattr(self.wm, "test_only", False) and torch.is_grad_enabled()):
            wm_t = self.wm.term(ctx)
            v = v + wm_t                            # world-model stage A (worldmodel.py): rolled-out state, its own part
        out = self.kind_first(ctx, v if not self.use_pattern else v + (self.pattern_term(ctx) if pterm is None else pterm))
        if wm_t is not None and getattr(self.wm, "last", False):
            # finding 125 ("+wmlast"): the decided world model speaks LAST -- where its rollout lands decidedly on a cell
            # the kind gate allows, that cell is put on top of every other part's answer by the rollout's own evidence
            # (cell = the row's best + its evidence; no new constant). Finding 124: a product of experts won the stories
            # only because the world model, silent on the calibration items, got an infinite weight when it spoke.
            fired = (wm_t > 0) & (out > -1e8)
            lift = torch.where(fired, out.max(1, keepdim=True).values - out + wm_t, torch.zeros_like(out))
            out = out + lift
            self.wm.last_out = wm_t + lift
        sc = getattr(self, "scene", None)
        if sc is not None and not ctx.is_sparse:
            # D75 step 1 ("+scene", scene_part.py): the scene part's rolled-out cell speaks last where it fires (as finding
            # 125's rule for the world model); silent rows unchanged
            sc_t = sc.term(ctx)
            fired = (sc_t > 0) & (out > -1e8)
            out = out + torch.where(fired, out.max(1, keepdim=True).values - out + sc_t, torch.zeros_like(out))
        return self.only_known(out)

    def only_known(self, out):
        """D26 (1) S1 ("+knownonly", 2026-10-10): only words heard so far are answer candidates -- in practice too (the
        loss's denominator), so a word never heard is neither answered nor pushed away as a wrong answer before it is
        met (the step that makes appending words later the same as having their places from birth). Off = unchanged."""
        m = getattr(self, "known_mask", None)
        return out if m is None else out.masked_fill(~m, -1e9)

    def classify(self, ids):
        out = super().classify(ids)
        m = getattr(self, "known_cat_mask", None)     # D26 (1) S1: only rooms already met are classes
        return out if m is None else out.masked_fill(~m, -1e9)

    def kind_first(self, ctx, v):
        """finding 90's part (answer the KIND first, 2026-09-30): the elements carry channel tags (position / motion word /
        name / picture ...). From first-hand moments the learner has recorded, for every kind-composition of a question
        ("one position, one motion word, two pictures"), which kinds have been the missing one. When answering, only
        elements of those kinds can be the answer (a composition never met: no restriction). Learned from the record, no
        constant. Off unless woken (kind_gate)."""
        if ctx.is_sparse or not getattr(self, "kind_gate", False) or getattr(self, "kind_of", None) is None:
            return v
        valid = ctx >= 0
        counts = torch.zeros(len(ctx), self.n_kinds, dtype=torch.long)
        counts.scatter_add_(1, self.kind_of[ctx.clamp(min=0)], valid.long())
        allowed = torch.ones(len(ctx), self.n_kinds, dtype=torch.bool)
        for i, sig in enumerate(map(tuple, counts.tolist())):
            seen = self.kind_table.get(sig)
            if seen:
                allowed[i] = False
                allowed[i, list(seen)] = True
        return v.masked_fill(~allowed[:, self.kind_of], -1e9)


class CuedPatterns:
    """finding 90, templates WITH A CUE (2026-09-30): in the glyph world the pair (from, to) is strongly tied only when the
    motion word is given -- P(to | from) is spread over the verbs, P(to | from, verb) = 1. The relation is three-way (cue +
    pair). So the template search of internal logic 2 is widened: within the first-hand sentences that contain a word c,
    the pairs (a, b) strongly tied there (n(c, a, b) / n(c, a) >= STRONG, n(c, a) >= MIN_SEEN -- the same constants as the
    plain search) give directions that are grouped into c's own templates (the same assimilate / accommodate rule,
    VIGILANCE). Every word is a candidate cue; which words work as cues is learned from the record. When answering, c's
    templates are used only if c is in the question. A template gives no evidence on the pairs it was built from."""

    def __init__(self, channel_of=None, place_of=None, kind_of=None):
        self.channel_of = channel_of    # sense channel of every element (None: one sense, all pairs allowed)
        # kinds (finding 91 follow-up): a template learned on (from -> to) pairs is applied only from elements of the kinds its
        # pairs started at, to elements of the kinds they ended at -- its reverse twin (to -> from) must not carry a "from"
        # position somewhere (it put the wrong cell first in the first check). Learned from the members, no constant.
        self.kind_of = kind_of
        self.t_kinds = {}
        # finding 91's follow-up, templates with a cue AND a place (2026-09-30): in the upside-down rooms "falls" moves up. The
        # rooms look like any other room -- only what was heard THERE tells the rule. Every template remembers the places
        # (rooms) its pairs were heard in; when answering in a place, the cue's templates confirmed in that place are used,
        # and only if none was confirmed there, all of the cue's templates (as without places). place_of: element -> place
        # (-1 for elements every place shares). No constant.
        self.place_of = place_of
        self.places_cab, self.t_places = {}, {}
        self.n_c, self.n_ca, self.n_cab = {}, {}, {}
        self.templates = {}             # cue -> [[direction, count]]
        self.members = {}               # (cue, template index) -> pairs
        self.done = set()

    def __deepcopy__(self, memo):
        """speed pass (2026-10-12, probe_clone.out: this deep copy was 97% of Learner3.clone, ~15% of a life). The count
        tables hold only immutable keys (ints, tuples of ints) and int values, the place / kind / member tables only
        immutable elements -- so a one-level copy is the same as a deep one; sets are rebuilt from their list as deepcopy
        does (same iteration order). Templates (tensors) and the world's tag arrays are still deep-copied. Exactness:
        check_speed.out (old vs new lives, seed 0)."""
        import copy
        c = object.__new__(type(self))
        memo[id(self)] = c
        for k, v in self.__dict__.items():
            if k in ("n_c", "n_ca", "n_cab"):
                c.__dict__[k] = dict(v)
            elif k in ("places_cab", "t_places", "t_kinds"):
                c.__dict__[k] = {kk: set(list(s)) for kk, s in v.items()}
            elif k == "members":
                c.__dict__[k] = {kk: list(s) for kk, s in v.items()}
            elif k == "done":
                c.__dict__[k] = set(list(v))
            else:
                c.__dict__[k] = copy.deepcopy(v, memo)
        return c

    def observe(self, sentences):
        for s_ in sentences:
            ws = sorted(set(s_))
            here = ({int(self.place_of[x]) for x in ws if self.place_of[x] >= 0} if self.place_of is not None else set())
            for c in ws:
                rest = [x for x in ws if x != c]
                self.n_c[c] = self.n_c.get(c, 0) + 1
                for a in rest:
                    self.n_ca[(c, a)] = self.n_ca.get((c, a), 0) + 1
                    for b in rest:
                        if b != a:
                            self.n_cab[(c, a, b)] = self.n_cab.get((c, a, b), 0) + 1
                            if here:
                                self.places_cab.setdefault((c, a, b), set()).update(here)

    def absorb(self, known, memory):
        for (c, a, b), n in self.n_cab.items():
            if (c, a, b) in self.done or a not in known or b not in known or c not in known:
                continue
            m = self.n_ca[(c, a)]
            if m < MIN_SEEN or n / m < STRONG:
                continue
            self.done.add((c, a, b))
            if self.channel_of is not None and self.channel_of[a] != self.channel_of[b]:
                continue                            # a direction means something only inside one sense's features
            d = memory[b] - memory[a]
            ts = self.templates.setdefault(c, [])
            best, best_cos = None, 0.0
            for i, t in enumerate(ts):
                cs = float(F.cosine_similarity(d, t[0], dim=0))
                ratio = float(d.norm() / t[0].norm().clamp(min=1e-9))
                if cs > best_cos and 0.5 <= ratio <= 2.0:      # directed: the cue fixes the direction's sign
                    best, best_cos = i, cs
            if best is not None and best_cos >= VIGILANCE:
                t = ts[best]
                t[1] += 1
                t[0] = t[0] + (d - t[0]) / t[1]
                self.members.setdefault((c, best), []).append((a, b))
                if self.kind_of is not None:
                    self.t_kinds.setdefault((c, best), set()).add((int(self.kind_of[a]), int(self.kind_of[b])))
                self.t_places.setdefault((c, best), set()).update(self.places_cab.get((c, a, b), ()))
            else:
                ts.append([d.clone(), 1])
                self.members.setdefault((c, len(ts) - 1), []).append((a, b))
                if self.kind_of is not None:
                    self.t_kinds.setdefault((c, len(ts) - 1), set()).add((int(self.kind_of[a]), int(self.kind_of[b])))
                self.t_places.setdefault((c, len(ts) - 1), set()).update(self.places_cab.get((c, a, b), ()))

    def matrices(self, memory, known, V, n_known, by_place=False, members_vote=False, detail=False):
        """{cue: V x V evidence (log1p units as model.C)} for cues with an established template (count >= 2); by_place:
        {cue: [(V x V evidence of one template, the places it was confirmed in)]}."""
        words = torch.tensor(sorted(known))
        m = memory[words]
        pos = {int(x): i for i, x in enumerate(words.tolist())}
        out = {}
        for c, ts in self.templates.items():
            Cc = None
            for i, (t, n) in enumerate(ts):
                if n < 2:
                    # (tried 2026-09-30: keeping once-confirmed templates as LOCAL knowledge of their place -- worse: junk
                    # local templates (object pictures as cues) pushed the good general ones aside in ordinary rooms)
                    continue
                match = 1 - torch.cdist(m + t, m) / t.norm()
                match.fill_diagonal_(0)
                if not members_vote:
                    # (finding 91 follow-up, members_vote: a cued template's evidence is three-way -- the pairwise habit
                    # never holds it -- so voting on its own pairs does not count the same evidence twice)
                    for a, b in self.members.get((c, i), ()):
                        if a in pos and b in pos:
                            match[pos[a], pos[b]] = 0
                val = torch.where(match > 0.5, n / (n + 1) * match, torch.zeros_like(match))
                if self.kind_of is not None and (c, i) in self.t_kinds:
                    kw = torch.tensor(self.kind_of)[words]
                    src = torch.tensor(sorted({ka for ka, _ in self.t_kinds[(c, i)]}))
                    dst = torch.tensor(sorted({kb for _, kb in self.t_kinds[(c, i)]}))
                    val = val * torch.isin(kw, src).float().unsqueeze(1) * torch.isin(kw, dst).float().unsqueeze(0)
                if by_place or detail:
                    full = torch.zeros(V, V)
                    full[words.unsqueeze(1), words.unsqueeze(0)] = val
                    src = {ka for ka, _ in self.t_kinds.get((c, i), ())}
                    # stored sparse (2026-10-08, memory pass: most entries are 0 and log1p(0) = 0 -- the same values;
                    # long multi-moment stories made thousands of dense V x V templates, copied into every branch)
                    out.setdefault(c, []).append((torch.log1p(full * (n_known - 1)).to_sparse(),
                                                  set(self.t_places.get((c, i), ())) if by_place else set(), n, src))
                    continue
                Cc = val if Cc is None else torch.maximum(Cc, val)
            if Cc is not None:
                full = torch.zeros(V, V)
                full[words.unsqueeze(1), words.unsqueeze(0)] = Cc
                out[c] = torch.log1p(full * (n_known - 1)).to_sparse()
        return out


class ChunkMemory:
    """finding 62 (the user's proposal: besides solving with templates, look up whether the phrase itself is remembered
    whole, and compare): every heard sentence is kept whole, as an episode (the hippocampus keeps episodes whole and
    conjunctive; complementary learning systems, McClelland et al. 1995). To answer, find the remembered sentences that
    share the MOST words with the question; if they share at least two (a phrase, not a single word -- one shared word is
    what the pairwise habit already holds, and counting it again would count the same evidence twice, finding 56),
    each word gets the share of those sentences it appears in, r, as evidence log(1 + r (V - 1)) (the template units,
    finding 55). Words already in the question get none (finding 57)."""

    def __init__(self, V, mode="plain"):
        self.V = V
        self.index = {}
        self.rows = []
        self.n_known = 2
        self.mode = mode            # finding 63: "plain" / "idf" / "chance" / "arbiter"; 64: "ladder" / "consensus"
        self.info_weight = False    # finding 65: consensus pieces weighted by how decided they are
        self.allowed = {}           # arbiter: match shape -> use the memory there?
        self.KH, self.KB = {}, {}   # arbiter: per match shape, times the memory helped / harmed
        self.route_ok, self.RH, self.RB = {}, {}, {}   # finding 81 v3: per override class, allowed / helped / harmed
        self.source = []            # the source of each remembered sentence (memory-first rule, 2026-09-29)

    def consensus(self, ctx, cm, E, N, df, exclude):
        """finding 64, the user's overlapping pieces: every two words of the question are a piece; each piece that some
        remembered sentence contains gives its candidates (the share of its sentences each word appears in), and a
        candidate's score is the weighted agreement of the pieces, weight = the rarity of the piece's words (idf, finding
        63) -- pieces that share a word ("the apple from the tree" + "from the tree fell") and agree on an answer make
        it strong; chance overlaps point at scattered words and cancel out (a product-of-experts-like intersection)."""
        idf = torch.log(N / df.clamp(min=1))
        inv = {}
        for j, row in enumerate(self.rows[:N]):
            for x in torch.nonzero(row).flatten().tolist():
                inv.setdefault(x, set()).add(j)
        words_of = [set(torch.nonzero(row).flatten().tolist()) for row in self.rows[:N]]
        out = torch.zeros(len(ctx), self.V)
        for i in range(len(ctx)):
            Q = sorted({int(x) for x in ctx[i].tolist() if x >= 0})
            skip = int(exclude[i]) if exclude is not None else -1
            score, total = {}, 0.0
            for a_ in range(len(Q)):
                for b_ in range(a_ + 1, len(Q)):
                    eps = (inv.get(Q[a_], set()) & inv.get(Q[b_], set())) - {skip}
                    if not eps:
                        continue
                    wgt = float(idf[Q[a_]] + idf[Q[b_]])
                    counts = {}
                    for j in eps:
                        for x in words_of[j]:
                            counts[x] = counts.get(x, 0) + 1
                    if self.info_weight:
                        # finding 65: a piece counts by how DECIDED it is -- 1 - entropy / max entropy of the words it points
                        # at (outside the question): a real phrase points at one answer, a chance overlap scatters
                        cand = [c for x, c in counts.items() if x not in Q]
                        if len(cand) > 1:
                            pr = torch.tensor(cand, dtype=torch.float) / sum(cand)
                            wgt *= float(1 + (pr * pr.log()).sum() / math.log(len(cand)))
                    total += wgt
                    for x, c in counts.items():
                        score[x] = score.get(x, 0.0) + wgt * c / len(eps)
            if total > 0:
                for x, v in score.items():
                    out[i, x] = v / total
        out = out * (cm == 0)
        return torch.log1p(out * (self.n_known - 1))

    def route_classes(self, ctx, z0, ev, exclude=None):
        """finding 81, v3 (the rare-word gate fixed the questions whose two words were only heard apart but broke once-heard
        pairs: from the question alone the two look the same). Per question: a = the answer without the memory (habit +
        templates), b = the answer with it. When the memory overrides (a != b), the case is classed by what the memory can
        testify about: has a / has b ever been in a remembered sentence together with the question's rarest word (idf,
        finding 63)? -- 4 classes; trust per class is learned (route_trust). Co-occurrence is counted from the remembered
        sentences themselves, leaving out the excluded one (so the self-check sees what the test will see)."""
        cm = context_mask(ctx, self.V)
        a = z0.masked_fill(cm > 0, float("-inf")).argmax(1)
        b = (z0 + ev).masked_fill(cm > 0, float("-inf")).argmax(1)
        E = torch.stack(self.rows)
        df = E.sum(dim=0)
        f = torch.where(ctx >= 0, df[ctx.clamp(min=0)], torch.full(ctx.shape, float("inf")))
        rare = ctx.gather(1, f.argmin(dim=1, keepdim=True)).squeeze(1)

        def together(c):
            n = (E[:, rare] * E[:, c]).sum(dim=0)
            if exclude is not None:
                ok = exclude >= 0
                n[ok] = n[ok] - E[exclude[ok], rare[ok]] * E[exclude[ok], c[ok]]
            return n > 0
        return a, b, together(a), together(b)

    def route_gate(self, ctx, z0, ev, exclude=None):
        a, b, sa, sb = self.route_classes(ctx, z0, ev, exclude)
        keep = [bool(a[i] == b[i]) or self.route_ok.get((bool(sa[i]), bool(sb[i])), True) for i in range(len(ctx))]
        return torch.tensor(keep, dtype=torch.float).unsqueeze(1)

    @staticmethod
    def shape(n_shared, n_eps):
        """the shape of a match (finding 63, arbiter): how many words matched (2 / 3 / 4+) x how many remembered sentences
        matched equally well (1 / 2-3 / 4+)."""
        return (min(int(n_shared), 4), 1 if n_eps <= 1 else (2 if n_eps <= 3 else 4))

    def observe(self, sentences, source="world"):
        """source (the memory-first rule, 2026-09-29): where a remembered sentence came from -- "world" (heard first-hand),
        "replay" (re-heard from own memory), later "self-made" -- kept per row as the memory's source axis."""
        for s in sentences:
            key = tuple(sorted(set(s)))
            if key not in self.index:
                self.index[key] = len(self.rows)
                row = torch.zeros(self.V)
                row[list(key)] = 1.0
                self.rows.append(row)
                self.source.append(source)

    def evidence(self, ctx, exclude=None, shapes=False):
        if not self.rows:
            z = torch.zeros(len(ctx), self.V)
            return (z, [None] * len(ctx)) if shapes else z
        E = torch.stack(self.rows)
        N = len(E)
        df = E.sum(dim=0)
        cm = context_mask(ctx, self.V)
        Q = (cm > 0).float()
        ov = Q @ E.T
        if exclude is not None:                     # leave-one-out: not the sentence the question was made from
            ok = exclude >= 0                       # (-1: the sentence is not in memory)
            ov[torch.arange(len(ctx))[ok], exclude[ok]] = -1
        if self.mode == "consensus":
            ev = self.consensus(ctx, cm, E, N, df, exclude)
            return (ev, [None] * len(ctx)) if shapes else ev
        if self.mode in ("idf", "ladder", "idf-arbiter"):   # idf-arbiter: finding 74 note, the arbiter on idf matching
            # finding 63 (1): a shared rare word is strong evidence, a shared common word weak (inverse document
            # frequency, Sparck Jones 1972): the best-matching sentences are those whose shared words are rarest
            idf = torch.log(N / df.clamp(min=1))
            score = torch.where(ov >= 2, (Q * idf) @ E.T, torch.full_like(ov, -1.0))
            M = score.max(dim=1, keepdim=True).values
            sel = ((score == M) & (M > 0)).float()
        else:
            M = ov.max(dim=1, keepdim=True).values
            sel = ((ov == M) & (M >= 2)).float()
        if self.mode == "chance":
            # finding 63 (2): is this match more than chance? For the words a question shares with a remembered sentence,
            # observed O = sentences containing all of them, expected E = N x product of (df / N) if words fell into
            # sentences independently; the match counts by 1 - E / O (a set phrase: O >> E; filler words that happen to
            # meet: O ~ E -> ~0). No free constant.
            w = torch.zeros_like(sel)
            for i, j in torch.nonzero(sel).tolist():
                shared = torch.nonzero(Q[i] * E[j]).flatten()
                O = float(E[:, shared].prod(dim=1).sum())
                Ex = N * float((df[shared] / N).prod())
                w[i, j] = max(0.0, 1 - Ex / max(O, 1e-9))
            r = (w @ E) / sel.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            r = (sel @ E) / sel.sum(dim=1, keepdim=True).clamp(min=1)
        r = r * (cm == 0)
        ev = torch.log1p(r * (self.n_known - 1))
        if self.mode == "ladder":
            # finding 64, the user's ladder: the shape of a match is only how many words it shares (2 / 3 / 4+); a rung
            # where remembered matches harmed more than they helped is switched off, so only longer matches count there
            nsh = (ov * sel).max(dim=1).values
            shp = [(min(int(nsh[i]), 4),) if sel[i].sum() > 0 else None for i in range(len(ctx))]
        elif (shapes or self.mode in ("arbiter", "idf-arbiter")) and self.mode != "idf":
            shp = [self.shape(M[i, 0], sel[i].sum()) if sel[i].sum() > 0 else None for i in range(len(ctx))]
        else:
            shp = [None] * len(ctx)
        if self.mode in ("arbiter", "ladder", "idf-arbiter") and not shapes:
            # finding 63 (3): the memory is used only for the match shapes where it has helped at least as often as it
            # harmed (per shape, the stakes rule of finding 61 -- the user's "compare the two routes")
            keep = torch.tensor([bool(sh is not None and self.allowed.get(sh, True)) for sh in shp], dtype=torch.float)
            ev = ev * keep.unsqueeze(1)
        return (ev, shp) if shapes else ev


VERIFY_RANKS = [1, 2, 3, 5, 10]                    # finding 66: candidate thresholds a learned verify procedure can pick


def formatted(items, cats, known_by_cat, rng):
    """finding 66, three question formats made from the same (context, answer) items: blank (produce the word), options
    (the answer + 3 other known words of its category, pick one), candidate (a word that is the answer half of the time,
    else another known word of its category: yes or no?)."""
    out = []
    for cx, t in items:
        pool = [x for x in known_by_cat[int(cats[t])] if x != t and x not in cx]
        if len(pool) < 3:
            continue
        opts = [t] + [int(x) for x in rng.choice(pool, 3, replace=False)]
        cand = t if rng.random() < 0.5 else int(rng.choice(pool))
        out.append((cx, t, opts, cand))
    return out


def procedures(model, fitems):
    """every procedure on every item, masked scores (a word in the question is never the answer, finding 57):
    {shape: {procedure: tensor of right/wrong}}. Procedures: F = the best word of all, C = the best of the options,
    V_r = "yes" iff the candidate ranks in the top r. A procedure that gives the wrong kind of answer is wrong."""
    if not fitems:
        return {}
    ctx, tgt = L.items_tensor([(cx, t) for cx, t, _, _ in fitems])
    with torch.no_grad():
        z = model.complete(ctx)
    z = z.masked_fill(context_mask(ctx, z.shape[1]) > 0, float("-inf"))
    opts = torch.tensor([o for _, _, o, _ in fitems])
    cand = torch.tensor([c for _, _, _, c in fitems])
    F_ok = z.argmax(1) == tgt
    C_ok = opts.gather(1, z.gather(1, opts).argmax(1, keepdim=True)).squeeze(1) == tgt
    rank = (z > z.gather(1, cand.unsqueeze(1))).sum(1) + 1
    truth = cand == tgt
    no = torch.zeros_like(F_ok)
    res = {"blank": {"F": F_ok, "C": no}, "options": {"F": F_ok, "C": C_ok}, "candidate": {"F": no, "C": no}}
    for r in VERIFY_RANKS:
        res["candidate"][f"V{r}"] = (rank <= r) == truth
        res["blank"][f"V{r}"] = no
        res["options"][f"V{r}"] = no
    return res


def help_harm(C, model, sentences, n_known, mask_context):
    """on these sentences (not yet practised): answers the template evidence turns wrong -> right / right -> wrong."""
    nctx, ntgt = L.items_tensor(D_positions(sentences))
    with torch.no_grad():
        z = model.habit(nctx)
        rows = torch.log1p(C * (n_known - 1))[nctx.clamp(min=0)] * (nctx >= 0).float().unsqueeze(-1)
        boost = rows.max(dim=1).values
        if mask_context:
            boost = boost * (context_mask(nctx, boost.shape[1]) == 0)
        with_t = (z + boost).argmax(dim=1) == ntgt
    without = z.argmax(dim=1) == ntgt
    return float((with_t & ~without).sum()), float((without & ~with_t).sum())


def context_mask(ctx, V):
    """(n, V): how often each word already occurs in the question's context (padding ignored)."""
    return torch.zeros(len(ctx), V).scatter_add_(1, ctx.clamp(min=0), (ctx >= 0).float())


def fit_temperature(z, tgt):
    """temperature scaling (Guo et al. 2017): the T minimising the negative log-likelihood of softmax(z / T)."""
    grid = torch.logspace(-0.5, 1.5, 41)
    nll = torch.stack([F.cross_entropy(z / t, tgt) for t in grid])
    return float(grid[int(nll.argmin())])


CALIB_EDGES = [0.6, 0.8, 1.0]


def life(world_name, learner, seed, evidence=False, contradiction=False, calibrate=None, independent=False,
         reality=None, mask_context=False, select=None, mdl=False, min_count=2, organize=None, null="uniform",
         stakes=False, chunk=None, formats=None):
    """evidence: template evidence in log-probability units, log(1 + r m (V - 1)) (finding 55);
    contradiction: prediction scoring refutes only on contradiction (finding 55);
    calibrate: None / "bins" / "global" -- temperature of the habit fitted on the learner's own record of how its
    habit did on just-heard, not yet practised sentences (finding 56)."""
    calib_z, calib_t, calib_e = [], [], []
    H = B = 0.0                                     # reality filter (finding 57): times the templates helped / harmed
    SH = SB = 0.0                                   # stakes (finding 61): the same count, with every template in use
    KH = KB = 0.0                                   # chunk route trust (finding 62): times whole-sentence memory helped / harmed

    proof = np.nan
    rel = 0.5
    w = W.build(seed, schema=world_name not in ("random", "rconj"), shift=world_name == "shift", local=world_name == "local",
                back=world_name == "back", conj=world_name in ("conj", "rconj"), long=world_name == "long",
                parity=world_name == "parity")
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    FS, policy = {}, {}                             # finding 66: exam record per (shape, procedure) and the chosen procedure
    fmt_rng_seed = 90_000 + seed
    chunks = (ChunkMemory(w.V, mode="consensus" if chunk == "consensus2" else
                          (chunk if chunk in ("idf", "chance", "arbiter", "ladder", "consensus") else "plain"))
              if chunk else None)
    if chunk == "consensus2":
        chunks.info_weight = True
    vecs = torch.tensor(w.vecs)
    cats = torch.tensor(w.cats)
    memory, _ = memorize(vecs, seed)
    use_pattern = learner not in ("D", "D-norm")
    norm = "norm" in learner
    if learner == "D":
        model = L.KnowDo2(w.V, w.n_cat, memory, seed)
    elif norm:
        model = NormPatternKnowDo(w.V, w.n_cat, memory, seed, use_pattern=use_pattern)
        model.mask_context = mask_context
    else:
        model = PatternKnowDo(w.V, w.n_cat, memory, seed)
    hebb = ConfDirectionalHebbian(w.V)
    pattern = ((ContradictionTrust if contradiction else TrustPatternMemory)("trust" in learner, mdl=mdl, min_count=min_count)
               if use_pattern else None)
    if use_pattern:
        pattern.organize = organize
        pattern.null = null
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    true = W.true_pairs(w)
    T = {k: torch.stack([memory[b] - memory[a] for a, b in pairs]).mean(0) for k, pairs in true.items()}
    known, seen, steps = set(), [], []
    for i, new_cats in enumerate(sessions):
        old = list(seen)
        seen += new_cats
        new_sents = [s for c in new_cats for s in sents[c]]
        new_words = {x for s in new_sents for x in s}
        known |= new_words
        old_sents = [s for c in old for s in sents[c]]
        review = [old_sents[j] for j in rng.choice(len(old_sents), int(round(len(old_sents) * REVIEW)), replace=False)] if old_sents else []
        heard = new_sents + review
        diag = {"place right": [], "place wrong": [], "trial right": [], "trial wrong": []}
        if use_pattern:
            pattern.session = i
        preds = pattern.predict(new_words, memory) if use_pattern else []
        hebb.observe(heard)
        if chunks is not None:
            chunks.observe(heard)
            chunks.n_known = len(known)
            model.chunks = chunks
        conn = hebb.connections(sorted(known))
        nw = torch.tensor(sorted(new_words))
        with torch.no_grad():
            model.assoc[nw, :] = conn[nw, :]
            model.assoc[:, nw] = conn[:, nw]
        C = None
        if use_pattern:
            place = None
            if select and "place" in select and pattern.weighted():
                place = place_applicability(pattern, memory, cats, known, w.V)
            if select and "trial" in select:
                mats = template_matrices(pattern, memory, known, w.V, independent=independent, place=place)
                nctx, ntgt = L.items_tensor(D_positions(new_sents))
                rel_k = trial(mats, model, nctx, ntgt, len(known), mask_context)
                C = torch.zeros(w.V, w.V)
                for k, Ck in mats:
                    C = torch.maximum(C, Ck * rel_k[k])
            elif stakes:
                # finding 61, stakes: how costly is it to act on unproven templates HERE? With every template in use, on
                # the new sentences just heard: answers turned wrong -> right (helped) and right -> wrong (harmed).
                # While helping has outweighed harming, act on hunches (all templates); otherwise demand proof (bits).
                pattern.use_mdl = False
                C = conf_matrix(pattern, memory, known, w.V, independent=independent)
                h, b = help_harm(C, model, new_sents, len(known), mask_context)
                SH, SB = SH + h, SB + b
                pattern.use_mdl = SB > SH
                if pattern.use_mdl:
                    C = conf_matrix(pattern, memory, known, w.V, independent=independent)
                proof = pattern.use_mdl
            else:
                C = conf_matrix(pattern, memory, known, w.V, independent=independent, place=place)
            if select and w.schema and pattern.weighted():
                # developer view: is a template's weight here larger where its relation really holds?
                here = {c: {d for a, _, d in w.pair_dirs if int(cats[a]) == c} for c in new_cats}
                for k, t, _ in pattern.weighted():
                    d_true = max(T, key=lambda d: abs(float(F.cosine_similarity(t, T[d], dim=0))))
                    if abs(float(F.cosine_similarity(t, T[d_true], dim=0))) < 0.8:
                        continue
                    for c in new_cats:
                        side = "right" if d_true in here[c] else "wrong"
                        if place is not None:
                            a0 = w.categories[c]["words"][0]
                            diag["place " + side].append(float(place[k][a0]))
                        if "trial" in select:
                            diag["trial " + side].append(rel_k[k])
            with torch.no_grad():
                if norm:
                    if reality:
                        # reality filter: before practice, on the new sentences just heard, did the template evidence turn
                        # wrong answers right (helped) or right answers wrong (harmed)?
                        nctx, ntgt = L.items_tensor(D_positions(new_sents))
                        full = torch.log1p(C * (len(known) - 1))
                        with torch.no_grad():
                            z = model.habit(nctx)
                            rows = full[nctx.clamp(min=0)] * (nctx >= 0).float().unsqueeze(-1)
                            boost = rows.max(dim=1).values
                            if mask_context:
                                boost = boost * (context_mask(nctx, boost.shape[1]) == 0)
                            with_t = (z + boost).argmax(dim=1) == ntgt
                        without = z.argmax(dim=1) == ntgt
                        h, b = float((with_t & ~without).sum()), float((without & ~with_t).sum())
                        if reality == "adaptive":           # Pearce-Hall: forget in proportion to the surprise
                            rho = abs((h + 1) / (h + b + 2) - rel)
                            H, B = (1 - rho) * H + h, (1 - rho) * B + b
                        else:
                            H, B = H + h, B + b
                        rel = (H + 1) / (H + B + 2)
                        C = C * rel
                    model.C.copy_(torch.log1p(C * (len(known) - 1)) if evidence else C)
                    model.e.copy_(torch.zeros(w.V) if independent else hebb.with_ / (hebb.with_ + 1))
                    model.scale = 1.0 if evidence else math.log(len(known))
                else:
                    model.pattern.copy_(C * math.log(len(known)))
        if calibrate and norm:
            # self-test before practice: how did the habit do on the new sentences it has just heard and not practised?
            nctx, ntgt = L.items_tensor(D_positions(new_sents))
            with torch.no_grad():
                calib_z.append(model.habit(nctx))
                calib_t.append(ntgt)
                calib_e.append(torch.where(nctx >= 0, model.e[nctx.clamp(min=0)],
                                           torch.ones_like(nctx, dtype=torch.float)).min(dim=1).values)
            Z, Tg, E = torch.cat(calib_z), torch.cat(calib_t), torch.cat(calib_e)
            if calibrate == "global":
                model.temps = [(1.0, fit_temperature(Z, Tg))]
            else:
                temps, lower = [], -1.0
                for upper in CALIB_EDGES:
                    sel = (E > lower) & (E <= upper)
                    temps.append((upper, fit_temperature(Z[sel], Tg[sel]) if int(sel.sum()) >= 20 else 1.0))
                    lower = upper
                model.temps = temps
        if chunks is not None:
            if chunk == "trust":
                # the chunk route's own reality check: on the new sentences just heard (before practice), leaving out the
                # very sentence each question was made from -- did whole-sentence memory turn answers wrong -> right
                # (helped) or right -> wrong (harmed)? Used while it has helped at least as often as it harmed.
                items, ep = [], []
                for s_ in new_sents:
                    for cx, t in D_positions([s_]):
                        items.append((cx, t))
                        ep.append(chunks.index[tuple(sorted(set(s_)))])
                nctx, ntgt = L.items_tensor(items)
                with torch.no_grad():
                    model.chunk_on = False
                    base = model.complete(nctx)
                    withc = base + chunks.evidence(nctx, exclude=torch.tensor(ep))
                ok0, ok1 = base.argmax(1) == ntgt, withc.argmax(1) == ntgt
                KH, KB = KH + float((ok1 & ~ok0).sum()), KB + float((ok0 & ~ok1).sum())
                model.chunk_on = KH >= KB
            elif chunk in ("arbiter", "ladder"):
                items, ep = [], []
                for s_ in new_sents:
                    for cx, t in D_positions([s_]):
                        items.append((cx, t))
                        ep.append(chunks.index[tuple(sorted(set(s_)))])
                nctx, ntgt = L.items_tensor(items)
                with torch.no_grad():
                    model.chunk_on = False
                    base = model.complete(nctx)
                    ev, shp = chunks.evidence(nctx, exclude=torch.tensor(ep), shapes=True)
                ok0, ok1 = base.argmax(1) == ntgt, (base + ev).argmax(1) == ntgt
                for i, sh in enumerate(shp):
                    if sh is None:
                        continue
                    chunks.KH[sh] = chunks.KH.get(sh, 0) + float(ok1[i] & ~ok0[i])
                    chunks.KB[sh] = chunks.KB.get(sh, 0) + float(ok0[i] & ~ok1[i])
                chunks.allowed = {sh: chunks.KH[sh] >= chunks.KB[sh] for sh in chunks.KH}
                model.chunk_on = True
            else:
                model.chunk_on = True
        known_by_cat = {}
        for x in sorted(known):
            known_by_cat.setdefault(int(cats[x]), []).append(x)
        if formats and formats.startswith("learned"):
            # finding 66: the teacher's practice questions in all three formats, made from the new sentences just heard
            # (before practice); every procedure is tried on each (trying all is cheap, finding 58) and each question
            # SHAPE gets the procedure that has been right most often so far -- the system learns which procedure a
            # shape of question calls for, it is not told the format's name
            exam = procedures(model, formatted(D_positions(new_sents), cats, known_by_cat, np.random.default_rng(80_000 + seed + i)))
            for shape, procs in exam.items():
                for p, ok in procs.items():
                    FS[(shape, p)] = FS.get((shape, p), 0) + float(ok.sum())
            for shape in ("blank", "options", "candidate"):
                policy[shape] = max((p for (sh, p) in FS if sh == shape), key=lambda p: FS[(shape, p)])
        ctx, tgt = L.items_tensor(D_positions(heard))
        bag = L.bag(ctx, w.V)
        choose_opts = None
        if formats == "learned+practice":
            # finding 66: using the format itself -- the system makes its own options questions from what it heard (the
            # answer + 3 known words of its category) and practises picking among them (a contrastive loss)
            pr = np.random.default_rng(85_000 + seed + i)
            rows = []
            for cx, t in D_positions(heard):
                pool = [x for x in known_by_cat[int(cats[t])] if x != t and x not in cx]
                rows.append([t] + ([int(x) for x in pr.choice(pool, 3, replace=False)] if len(pool) >= 3 else [t, t, t]))
            choose_opts = torch.tensor(rows)
        extra = {}
        if use_pattern:
            extra = {"pterm": model.pattern_term(ctx)}
        wts = explain_weights(heard, model, C) if use_pattern else torch.ones(len(tgt))
        ids = torch.tensor(sorted({x for s in heard for x in s}))
        newest_once = [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "once"]
        newest_unseen = [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "unseen"]
        curve_once, curve_unseen = {}, {}
        newest_conj = [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind in ("conj", "long")]
        curve_conj = {}
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for e in range(EPOCHS + 1):
            if e in CHECKS:
                curve_once[e] = acc_items(model, newest_once)
                curve_conj[e] = acc_items(model, newest_conj)
                curve_unseen[e] = acc_items(model, newest_unseen)
            if e == EPOCHS:
                break
            opt.zero_grad()
            out = model.complete(bag, **extra)
            ce = F.cross_entropy(out, tgt, reduction="none")
            loss = (ce * wts).sum() / wts.sum() + F.cross_entropy(model.classify(ids), cats[ids])
            if choose_opts is not None:
                loss = loss + F.cross_entropy(out.gather(1, choose_opts), torch.zeros(len(tgt), dtype=torch.long))
            loss.backward()
            opt.step()
        if use_pattern:
            pattern.score(preds, hebb)
            pattern.absorb(hebb, known, memory)
            if organize == "merge":
                pattern.merge(known, memory)
            elif organize == "versions":
                pattern.propose_versions(known, memory)
        used = pattern.weighted() if use_pattern else []
        junk = sum(1 for _, t, _ in used if max(abs(float(F.cosine_similarity(t, T[k], dim=0))) for k in T) < 0.8)
        fmt = {}
        if formats:
            items = [(cx, t) for c in seen for kind, cx, t in tests[c]]
            res = procedures(model, formatted(items, cats, known_by_cat, np.random.default_rng(fmt_rng_seed)))
            use = ({"blank": "F", "options": "F", "candidate": "V1"} if formats == "fixed" else
                   {"blank": "F", "options": "C", "candidate": "V1"} if formats == "dev" else policy)
            fmt = {f"fmt {sh}": float(res[sh][use[sh]].float().mean()) for sh in ("blank", "options", "candidate")}
            fmt["verify r"] = float(use["candidate"][1:])
        steps.append({"curve once": curve_once, "curve unseen": curve_unseen, "curve conj": curve_conj, **fmt,
                      "conj all": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "conj"]),
                      "long all": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "long"]),
                      "chunk on": float(model.chunk_on) if chunks is not None else np.nan,
                      "never-heard": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c]]),
                      "pairs all": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind.startswith("R")]),
                      "once all": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "once"]),
                      "unseen all": acc_items(model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "unseen"]),
                      "used": len(used) if use_pattern else np.nan, "junk used": junk if use_pattern else np.nan,
                      "temps": [t for _, t in model.temps] if getattr(model, "temps", None) else [],
                      "rel": rel if reality else np.nan,
                      "proof": float(proof) if stakes else np.nan,
                      "groups": sum(1 for G in pattern.groups if G["i"] in pattern.in_group) if use_pattern else np.nan,
                      "merged lead": (np.mean([G["D"] > 0 for G in pattern.groups if G["i"] in pattern.in_group])
                                      if use_pattern and any(G["i"] in pattern.in_group for G in pattern.groups) else np.nan),
                      **{key: float(np.mean(v)) if v else np.nan for key, v in diag.items()}})
    return steps


def one_seed(seed):
    torch.set_num_threads(1)
    return {(wn, ln): life(wn, ln, seed) for wn in WORLDS for ln in LEARNERS}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with Pool(workers) as pool:
        results = pool.map(one_seed, range(n_seeds))
    n_sess = len(results[0][(WORLDS[0], LEARNERS[0])])
    print(f"=== {n_seeds} seeds; data_v3; rest below trust {REST_BELOW} after {REST_AFTER} predictions ===")
    for key, label in [("curve once", "newest ONE-SHOT pairs, epoch 0 / 200"), ("curve unseen", "newest UNSEEN pairs, epoch 0 / 200")]:
        print(f"\n--- {label} (sessions 1..{n_sess}) ---")
        for wn in WORLDS:
            for ln in LEARNERS:
                print(f"{wn:6} {ln:18} " + " | ".join(
                    f"{np.nanmean([r[(wn, ln)][i][key][0] for r in results]):.3f} {np.nanmean([r[(wn, ln)][i][key][200] for r in results]):.3f}"
                    for i in range(n_sess)))
    for key in ["never-heard", "pairs all", "once all", "unseen all", "used", "junk used"]:
        print(f"\n--- {key} (sessions 1..{n_sess}) ---")
        for wn in WORLDS:
            for ln in LEARNERS:
                row = [np.nanmean([r[(wn, ln)][i][key] for r in results]) for i in range(n_sess)]
                print(f"{wn:6} {ln:18} " + " ".join(f"{v:6.3f}" for v in row))

    print("\n--- paired ---")
    def cmp(ga, gb, label):
        cells = []
        for lab, get in [("life", lambda r, g: np.mean([st["never-heard"] for st in r[g]])),
                         ("end", lambda r, g: r[g][-1]["never-heard"]),
                         ("pairs end", lambda r, g: r[g][-1]["pairs all"]),
                         ("once end", lambda r, g: r[g][-1]["once all"]),
                         ("unseen end", lambda r, g: r[g][-1]["unseen all"]),
                         ("once kept", lambda r, g: np.mean([st["curve once"][200] - st["curve once"][0] for st in r[g][1:]]))]:
            dd = np.array([get(r, ga) - get(r, gb) for r in results])
            n_pos, n_neg = int((dd > 1e-9).sum()), int((dd < -1e-9).sum())
            cells.append(f"{lab} {dd.mean():+.3f} ({n_pos}/{n_neg}{'*' if sign_p(n_pos, n_neg) < 0.01 else ''})")
        print(f"{label:42} " + "  ".join(cells))
    for wn in WORLDS:
        cmp((wn, "D-norm"), (wn, "D"), f"{wn}: D-norm vs D")
        cmp((wn, "pattern+explain"), (wn, "D"), f"{wn}: pattern+explain vs D")
        cmp((wn, "trust+explain"), (wn, "pattern+explain"), f"{wn}: trust+explain vs pattern+explain")
        cmp((wn, "norm+explain"), (wn, "D-norm"), f"{wn}: norm+explain vs D-norm")
        cmp((wn, "trust+norm+explain"), (wn, "norm+explain"), f"{wn}: trust+norm+explain vs norm+explain")
        cmp((wn, "trust+norm+explain"), (wn, "D"), f"{wn}: trust+norm+explain vs D")


if __name__ == "__main__":
    main()
