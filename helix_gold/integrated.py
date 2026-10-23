"""One body for all the parts (roadmap step 6, integration; 2026-09-25, design agreed with the user).

Until now the parts lived in two lines of code: the method layer (branching, the cross-validated judge, successive halving,
stable/plastic split -- run_branching*.py on data_v2) and the content layer (the standardised habit, templates with the
stakes rule, whole-sentence memory with rarity weighting, question formats -- run_schema3.py on data_v3). Here the content
layer becomes one learner, Learner3, whose session() is run_schema3.life's session body moved as it is, with the method
layer's knobs as settings (review share, learning rate, order, review picking) -- so the method layer can branch it.

Content configuration (the best of findings 57-66): standardised habit + independent template evidence + context mask
(57) + stakes rule (61) + whole-sentence memory with idf (63) + learned question-format procedures (66).
Step A (regression test, `python integrated.py check`): with the start settings Learner3 must give the SAME numbers as
run_schema3.life with the same configuration.
"""
import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F

import data_v3 as W
import lab2 as L
from run_seed_transfer import memorize, memorize_linear
from run_schema import ConfDirectionalHebbian, acc_items, D_positions, EPOCHS, PER_SESSION, CHECKS, LR, REVIEW, STRONG
from run_schema2 import explain_weights
import run_schema3 as S3

START = {"review": REVIEW, "lr": LR, "order": "mixed", "pick": "uniform"}
FLOOR = 0.01


def build_world(world_name, seed):
    if world_name == "life32":                      # the long test: 32 categories, 16 sessions, two changes
        return W.build(seed, n_cat=32, shift2=True)
    if world_name == "life32share":                 # world ladder rung 1 (finding 111): common words in the second half
        return W.build(seed, n_cat=32, shift2=True, share=True)
    if world_name == "life32blind":                 # world ladder rung 2: + half the second half's words without senses
        return W.build(seed, n_cat=32, shift2=True, share=True, blind=True)
    if world_name == "ascii":                       # B stage: the glyph-picture world model (data_ascii.py)
        import data_ascii
        return data_ascii.build(seed)
    if world_name in ("ascii-scene", "ascii-sceneshuf"):   # D75 step 1: the walk told as sentence + the scene after
        import data_ascii
        return data_ascii.build(seed, scene=world_name == "ascii-scene", scene_shuffle=world_name == "ascii-sceneshuf")
    if world_name in ("ascii-d72cat", "ascii-d72verb", "ascii-d72chain", "ascii-d72neg", "ascii-d72ice",
                      "ascii-d72next"):   # D72: + a subject
        import data_ascii
        return data_ascii.build(seed, d72=world_name[len("ascii-d72"):])
    if world_name == "ascii-kor":                   # finding 123: Korean particle stories (sentence -> scene)
        import data_ascii
        return data_ascii.build(seed, korean=True)
    if world_name == "ascii-multi":                 # stage-1 path step 2: multi-moment Korean stories (state over time)
        import data_ascii
        return data_ascii.build(seed, multi=True)
    if world_name == "ascii-objs":                  # stage-1 path step 4: objects with properties (heavy / light)
        import data_ascii
        return data_ascii.build(seed, props=True)
    if world_name == "ascii-tale":                  # D22: frames vs reality (the gravity world plus told tales)
        import data_ascii
        return data_ascii.build(seed, tale=True)
    if world_name == "ascii-grav":                  # stage-1 path step 4+: gravity / support
        import data_ascii
        return data_ascii.build(seed, grav=True)
    if world_name == "ascii-frame":                 # stage-1 path step 4+: reference frames (rooms turned a quarter / half)
        import data_ascii
        return data_ascii.build(seed, frame=True)
    if world_name in ("ascii-sp", "ascii-spshuf"):  # stage-1 track A W3: the four spatial words (+ the shuffle control)
        import data_ascii
        return data_ascii.build(seed, sp=world_name == "ascii-sp", sp_shuffle=world_name == "ascii-spshuf")
    if world_name in ("ascii-when", "ascii-whenshuf"):   # stage-1 track A W4: "when" questions (+ the shuffle control)
        import data_ascii
        return data_ascii.build(seed, when=world_name == "ascii-when", when_shuffle=world_name == "ascii-whenshuf")
    if world_name in ("ascii-c8", "ascii-c8shuf"):       # stage-1 track A W7b: criterion 8 keys ("where before", long stories)
        import data_ascii
        return data_ascii.build(seed, c8=world_name == "ascii-c8", c8_shuffle=world_name == "ascii-c8shuf")
    if world_name in ("ascii-bi", "ascii-bishuf"):       # stage-1 track A W6: two languages, Korean- and English-like
        import data_ascii
        return data_ascii.build(seed, bi=world_name == "ascii-bi", bi_shuffle=world_name == "ascii-bishuf")
    if world_name in ("ascii-ten", "ascii-tenshuf"):     # stage-1 track A W5: numbers 1-10, put in / take out (+ shuffle)
        import data_ascii
        return data_ascii.build(seed, ten=world_name == "ascii-ten", ten_shuffle=world_name == "ascii-tenshuf")
    if world_name == "ascii-contain":               # stage-1 path step 4+: containment (inside / beside)
        import data_ascii
        return data_ascii.build(seed, contain=True)
    if world_name == "ascii-count":                 # stage-1 path step 4+: small numbers (1-3)
        import data_ascii
        return data_ascii.build(seed, count=True)
    if world_name == "ascii-self":                  # D2 the self by contingency (ledger D2)
        import data_ascii
        return data_ascii.build(seed, self_=True)
    if world_name == "ascii-goal":                  # D2 goal-directed motion (ledger D2)
        import data_ascii
        return data_ascii.build(seed, goal=True)
    if world_name == "ascii-viewpt":                # D2 viewpoint tests: the three-view world + back view + turns
        import data_ascii
        return data_ascii.build(seed, views=2)
    if world_name == "ascii-views":                 # stage-1 path step 4+: the three-view world (ledger D17 v1)
        import data_ascii
        return data_ascii.build(seed, views=True)
    if world_name in ("ascii-numrow", "ascii-numown"):    # stage-1 path step 4+: the number-line room (ledger D13 / D15)
        import data_ascii
        return data_ascii.build(seed, numline=world_name[len("ascii-num"):])
    if world_name == "ascii-succ":                  # stage-1 path step 4+: the successor rule (ledger D1a)
        import data_ascii
        return data_ascii.build(seed, succ=True)
    if world_name == "ascii-seq2":                  # D80 v2: the "@seq" world, a walk's last moves told only in words
        import data_ascii
        return data_ascii.build(seed, seq_told=data_ascii.SEQ_TOLD)
    if world_name == "ascii-seq":                   # D80 v1: the "@scene" world plus question moments and tests
        import data_ascii
        return data_ascii.build(seed, seq=True)
    if world_name == "ascii-nwf":                   # Q14 frame test: the "@nw" world plus a sentence frame (finding 167)
        import data_ascii
        return data_ascii.build(seed, nwf=True)
    if world_name == "ascii-nw":                    # stage-1 track A W2: new words heard 1 / 2 / 4 / 8 times (criterion 2)
        import data_ascii
        return data_ascii.build(seed, nw=True)
    if world_name == "ascii-hold":                  # world-model stage B: one motion word unheard in each upside-down room
        import data_ascii
        return data_ascii.build(seed, hold=True)
    return W.build(seed, schema=world_name not in ("random", "rconj"), conj=world_name in ("conj", "rconj"),
                   parity=world_name == "parity", compose=world_name == "compose",
                   shift=world_name == "shift", local=world_name == "local", back=world_name == "back")


class Learner3:
    def __init__(self, w, seed, content=True, chunk=True, compose=False, setter=None, where=None, attend=False,
                 rehearse=False):
        self.w, self.seed = w, seed
        self.compose = compose                      # finding 71: templates chained (A -> B -> C)
        self.setter = setter                        # finding 71: None / "hard" (setter and solver) / "random" (control)
        self.content = content                      # False: the standardised habit alone (templates and memory off)
        self.use_chunk = chunk                      # False: content layer without whole-sentence memory (finding 70)
        self.cats = torch.tensor(w.cats)
        if rehearse:                                # readiness step 2 (2026-10-17): rehearse until every element is recalled
            from run_seed_transfer import memorize_rehearse   # (the where channel's own linear memory is exact: not checked here)
            chk = [t_ for t_ in range(w.V) if not (where == "linear" and t_ in getattr(w, "pos_of", {}))]
            self.memory, _, self.rehearse_rounds, self.rehearse_recall = memorize_rehearse(torch.tensor(w.vecs), seed, chk)
        else:
            self.memory, _ = memorize(torch.tensor(w.vecs), seed)
        if where == "linear" and hasattr(w, "kind_of"):   # finding 90 (structure): the "where" channel gets its own linear memory
            idx = torch.tensor(sorted(w.pos_of))
            others = torch.ones(w.V, dtype=torch.bool)
            others[idx] = False
            size = float(self.memory[others].norm(dim=1).mean())
            z, self.where_decoder = memorize_linear(torch.tensor(w.vecs)[idx], seed, w.where_dims,
                                                    lambda z_: size / float(z_.norm(dim=1).mean()))
            self.memory[idx] = z
        self.model = S3.NormPatternKnowDo(w.V, w.n_cat, self.memory, seed, use_pattern=True)
        self.model.mask_context = True
        if attend:                                  # finding 101: attention over the context words, starting as the plain mean
            self.model.attn = torch.nn.Parameter(torch.zeros(self.memory.shape[1], self.memory.shape[1]))
        if hasattr(w, "kind_of"):                   # finding 90: channel tags -> the kind-first answer (dormant unless woken)
            self.model.kind_of = torch.tensor(w.kind_of)
            self.model.n_kinds = len(w.kinds)
            self.model.kind_table = {}
        self.hebb = ConfDirectionalHebbian(w.V)
        # the source axis of the connection record (memory-first rule, 2026-09-29): the Hebbian record counts everything
        # heard; the replayed part (review sentences re-heard from the learner's own memory) is kept beside it, so that the
        # first-hand part is total - replay. Views by source: source_view().
        self.replay = L.Hebbian2(w.V)
        self.rec = self.hebb                            # the record in use this session (set in prepare)
        self.pattern = S3.TrustPatternMemory(False, mdl=False, min_count=2)
        self.pattern.null = "near"
        if hasattr(w, "channel_of") and where == "linear":  # finding 90: templates only inside one sense (with its own memory)
            self.pattern.channel_of = w.channel_of
        self.chunks = S3.ChunkMemory(w.V, mode="idf")
        self.known = set()
        self.memory_hash = self.fingerprint()       # outer layer integrity (checksum): the original memory must never change
        self.intact = True
        self.SH = self.SB = 0.0
        self.FS, self.policy = {}, {}
        self.fmt_practice = self.fmt_hard = False
        self.i = 0
        # the plasticity record (2026-10-02, Pioneer's idea): per word, how many sessions it has been heard in first-hand
        # (age), and how many sessions in a row its habit row moved the same way (streak); kept always, used when "plastic"
        self.plas = {"age": torch.zeros(w.V), "streak": torch.zeros(w.V), "prev": None}
        # finding 105's follow-up: the same record per CONNECTION (word pair of the habit's link table): sessions the pair was
        # heard together first-hand, and the run of sessions its link moved the same way (sign of the change)
        self.plas2 = {"age": torch.zeros(w.V, w.V), "streak": torch.zeros(w.V, w.V), "prev": None}

    def source_view(self, source="all"):
        """the connection record along its source axis: "all" (as heard, first-hand + replayed), "world" (first-hand only),
        "replay" (replayed only) -- an object with .counts / .with_ and the same connections() as the record."""
        if source == "all":
            return self.hebb
        v = copy.copy(self.hebb)
        if source == "world":
            v.counts, v.with_ = self.hebb.counts - self.replay.counts, self.hebb.with_ - self.replay.with_
        else:
            v.counts, v.with_ = self.replay.counts, self.replay.with_
        return v

    def fingerprint(self):
        import hashlib
        return hashlib.sha256(self.memory.numpy().tobytes()).hexdigest()

    def clone(self):
        c = copy.copy(self)
        self.model.chunks = None
        c.model = copy.deepcopy(self.model)
        self.model.chunks = self.chunks
        c.hebb = copy.deepcopy(self.hebb)
        c.replay = copy.deepcopy(self.replay)
        c.rec = c.hebb                                  # (a branch's prepare sets its own view)
        c.pattern = copy.deepcopy(self.pattern)
        c.chunks = copy.deepcopy(self.chunks)
        c.model.chunks = c.chunks
        c.known = set(self.known)
        c.FS, c.policy = dict(self.FS), dict(self.policy)
        if hasattr(self, "cued"):
            c.cued = copy.deepcopy(self.cued)
        c.plas = copy.deepcopy(self.plas)
        c.plas2 = copy.deepcopy(self.plas2)
        if getattr(self, "keep_order", False):
            c.order_log = list(self.order_log)
        if getattr(c.model, "wm", None) is not None:    # world-model stage A: the branch's books read the branch's memory
            c.model.wm.learner = c
        c.st = None
        if getattr(self, "d25", None) is not None:     # D25: a branch practises with its own copy of the store
            c.d25 = self.d25.copy()
        return c

    def pick_review(self, old_by_cat, s, rng, weights):
        cats = [c for c in old_by_cat if old_by_cat[c]]
        old_sents = [x for c in cats for x in old_by_cat[c]]
        n_rev = int(round(len(old_sents) * s["review"]))
        if not old_sents:
            return []
        if s.get("forget_pick"):
            # finding 108 (the user's choice (b), 2026-10-03: a late-life knob): review the categories being FORGOTTEN --
            # weight = how much the category's own heard sentences dropped since the last session (the learner's record,
            # self.drop, kept by the life loop) + the same floor as the curious pick; no new constant
            drop = getattr(self, "drop", {})
            w = np.array([max(0.0, drop.get(c, 0.0)) + FLOOR for c in cats])
            pools = {c: list(rng.permutation(len(old_by_cat[c]))) for c in cats}
            review = []
            for _ in range(n_rev):
                live = [j for j, c in enumerate(cats) if pools[c]]
                c = cats[live[int(rng.choice(len(live), p=w[live] / w[live].sum()))]]
                review.append(old_by_cat[c][pools[c].pop()])
            return review
        if s["pick"] == "uniform":                  # exactly run_schema3's draw
            return [old_sents[j] for j in rng.choice(len(old_sents), n_rev, replace=False)]
        w = np.array([abs(weights.get(c, 0.0)) + FLOOR for c in cats])   # curious (finding 41)
        pools = {c: list(rng.permutation(len(old_by_cat[c]))) for c in cats}
        review = []
        for _ in range(n_rev):
            live = [j for j, c in enumerate(cats) if pools[c]]
            c = cats[live[int(rng.choice(len(live), p=w[live] / w[live].sum()))]]
            review.append(old_by_cat[c][pools[c].pop()])
        return review

    def session(self, new_sents, old_by_cat, s, rng, weights=None, checks=None):
        """one session with settings s (run_schema3.life's session body with its content configuration).
        checks: {name: items} measured at CHECKS epochs (curves), or None."""
        self.prepare(new_sents, old_by_cat, s, rng, weights)
        curves = self.train_to(EPOCHS, checks)
        self.finish()
        return curves

    def prepare(self, new_sents, old_by_cat, s, rng, weights=None):
        """everything before practice: review, connection seeds, templates + stakes, whole-sentence memory, format exam,
        and the practice phases (so that practice can run in stages -- successive halving continues a branch)."""
        model, pattern, chunks, memory, cats, w = self.model, self.pattern, self.chunks, self.memory, self.cats, self.w
        self.wake(s)
        model.kind_gate = bool(s.get("kind_first", False))
        if getattr(model, "kind_of", None) is not None:  # the kind-composition record, from first-hand moments only
            ko = model.kind_of.tolist()
            for sent in new_sents:
                for j, x in enumerate(sent):
                    sig = [0] * model.n_kinds
                    for y in sent[:j] + sent[j + 1:]:
                        sig[ko[y]] += 1
                    model.kind_table.setdefault(tuple(sig), set()).add(ko[x])
        new_words = {x for x in new_sents for x in x}
        self.plas["start"] = model.z_work.detach().clone()
        self.plas["heard"] = sorted(new_words)
        self.plas2["start"] = model.assoc.detach().clone()
        self.plas2["sents"] = [sorted(set(x)) for x in new_sents]
        self.known |= new_words
        if getattr(model, "known_only", False):    # D26 (1) S1: the answer candidates = the words heard so far
            model.known_mask = torch.zeros(w.V, dtype=torch.bool)
            model.known_mask[sorted(self.known)] = True
            model.known_cat_mask = torch.zeros(w.n_cat, dtype=torch.bool)
            model.known_cat_mask[sorted({int(cats[x]) for x in self.known})] = True
        known = self.known
        review = self.pick_review(old_by_cat, s, rng, weights or {})
        heard = new_sents + review
        pattern.session = self.i
        preds = pattern.predict(new_words, memory) if self.content else []
        self.hebb.observe(heard)
        self.replay.observe(review)
        model.cooc, model.freq = self.hebb.counts, self.hebb.with_
        # the record this session's use reads: all of it, or a slice along its source axis (s["hebb_source"])
        self.rec = self.source_view(s.get("hebb_source", "all"))
        conn = self.rec.connections(sorted(known))
        nw = torch.tensor(sorted(new_words))
        with torch.no_grad():
            model.assoc[nw, :] = conn[nw, :]
            model.assoc[:, nw] = conn[:, nw]
        C = torch.zeros(w.V, w.V)
        if getattr(self, "keep_order", False):     # finding 123: the ordered first-hand record (word order as heard)
            self.order_log.extend(list(x) for x in new_sents)
        if self.content and self.use_chunk:
            chunks.observe(new_sents, "world")
            chunks.observe(review, "replay")                # (review sentences are remembered already: no new rows)
            chunks.n_known = len(known)
            model.chunks = chunks
        wm_g = getattr(model, "wm", None)
        if wm_g is not None and getattr(wm_g, "gate", False):
            wm_g.grade_parts(new_sents)             # D69: the parts graded on this session's sentences (tracker's rule)
        if wm_g is not None and getattr(wm_g, "useful", False):
            wm_g.grade_use(new_sents)               # D83 (1): each part's usefulness per question form, judged alone
        if self.content:
            pattern.use_mdl = False                 # stakes (finding 61)
            C = S3.conf_matrix(pattern, memory, known, w.V, independent=True)
            h, b = S3.help_harm(C, model, new_sents, len(known), True)
            self.SH, self.SB = self.SH + h, self.SB + b
            if s.get("thompson"):
                # tier 2 (Thompson sampling, 2026-09-29): with few records the stakes rule's sign is noise -- draw the chance
                # that hunches help from Beta(helped + 1, harmed + 1) and act on hunches if the draw is above even
                tr = np.random.default_rng(97_000 + self.seed + self.i)
                pattern.use_mdl = bool(tr.beta(self.SH + 1, self.SB + 1) < 0.5)
            else:
                pattern.use_mdl = self.SB > self.SH
            if pattern.use_mdl:
                C = S3.conf_matrix(pattern, memory, known, w.V, independent=True)
            if self.compose:
                C = torch.maximum(C, composed_matrix(pattern, memory, known, w.V))
            model.chunk_on = self.use_chunk         # whole-sentence memory with idf (finding 63)
        with torch.no_grad():
            model.C.copy_(torch.log1p(C * (len(known) - 1)))
            model.e.copy_(torch.zeros(w.V))
            model.scale = s.get("pattern_w", 1.0)     # finding 95: the templates' weight (1 = as before)
        # finding 95: the whole-sentence memory's weight (1 = as before); finding 115: times the self-experiment's choice
        model.chunk_w = s.get("chunk_w", 1.0) * s.get("tune_chunk", 1.0)
        model.assoc_w = s.get("assoc_w", 1.0)          # finding 115: the connection table's answering weight (1 = as before)
        if self.content and s.get("cue_templates"):    # finding 90: templates with a cue (dormant unless woken)
            by_place = s.get("cue_templates") == "place"      # finding 91 follow-up: cue + place
            if not hasattr(self, "cued"):
                self.cued = S3.CuedPatterns(getattr(w, "channel_of", None),
                                            np.where(w.cats > 0, w.cats, -1) if hasattr(w, "channel_of") else None,
                                            getattr(w, "kind_of", None) if s.get("cue_kinds") else None)
                model.place_of = torch.tensor(np.where(w.cats > 0, w.cats, -1))
            self.cued.observe(new_sents)
            self.cued.absorb(known, memory)
            model.cue_pure = bool(s.get("cue_pure", False))
            model.cue_C = self.cued.matrices(memory, known, w.V, len(known), by_place=by_place,
                                             members_vote=bool(s.get("cue_members_vote", False)),
                                             detail=model.cue_pure)
            model.cue_specific = bool(s.get("cue_specific", False))
        else:
            model.cue_C = None
        if self.content and self.use_chunk and chunks.mode in ("arbiter", "ladder", "idf-arbiter"):
            self.shape_trust(new_sents)
        if self.content and self.use_chunk and model.route == "trust":
            self.route_trust(new_sents)
        known_by_cat = {}
        for x in sorted(known):
            known_by_cat.setdefault(int(cats[x]), []).append(x)
        self.known_by_cat = known_by_cat
        exam = S3.procedures(model, S3.formatted(D_positions(new_sents), cats, known_by_cat,
                                                  np.random.default_rng(80_000 + self.seed + self.i)))
        for shape, procs in exam.items():             # question formats (finding 66)
            for p, ok in procs.items():
                self.FS[(shape, p)] = self.FS.get((shape, p), 0) + float(ok.sum())
        for shape in ("blank", "options", "candidate"):
            self.policy[shape] = max((p for (sh, p) in self.FS if sh == shape), key=lambda p: self.FS[(shape, p)])
        if s.get("verify_cont"):
            # tier 2 (a continuous yes/no threshold, 2026-09-29): instead of choosing among ranks 1/2/3/5/10, keep every exam
            # answer's (rank, truth) and use the rank cut that would have been right most often so far (no list of cuts)
            rk, tr_ = self.cand_ranks(S3.formatted(D_positions(new_sents), cats, known_by_cat,
                                                   np.random.default_rng(80_000 + self.seed + self.i)))
            self.RK = torch.cat([getattr(self, "RK", torch.zeros(0, dtype=torch.long)), rk])
            self.TR = torch.cat([getattr(self, "TR", torch.zeros(0, dtype=torch.bool)), tr_])
            cuts = torch.unique(self.RK)
            right = torch.stack([((self.RK <= c) == self.TR).sum() for c in cuts])
            self.policy["candidate"] = f"Vc{int(cuts[int(right.argmax())])}"
        extra_items = self.setter_items(heard) if self.setter else []
        if getattr(self, "d25", None) is not None:     # D25 (2026-10-13): the parts' confident answers moved to the body
            extra_items = extra_items + self.d25.items(self, s, D_positions(heard) if self.d25.mode == "E" else None)
        phases = [heard] if s["order"] == "mixed" else [new_sents, heard]
        n_ep = [EPOCHS] if s["order"] == "mixed" else [EPOCHS // 2, EPOCHS - EPOCHS // 2]
        prepared, selfmask = [], {}
        newmasks = {}
        n_new_items = len(D_positions(new_sents))
        for p, (sents_p, n) in enumerate(zip(phases, n_ep)):
            items = D_positions(sents_p)
            wts = explain_weights(sents_p, model, C)
            n_heard_items = len(items)
            # which items come from this session's NEW sentences (heard = new_sents + review, in that order; D_positions
            # keeps the sentence order) -- finding 106's follow-up needs the new material's share of each update
            newm = torch.zeros(len(items), dtype=torch.bool)
            newm[:min(n_new_items, len(items)) if sents_p is not new_sents else len(items)] = True
            if extra_items and p == len(phases) - 1:
                items = items + extra_items
                wts = torch.cat([wts, torch.ones(len(extra_items))])
            if s.get("disagree") and p == len(phases) - 1:
                # tier 2 (practice where the routes disagree, 2026-09-29): items whose answer the templates change (habit
                # alone vs habit + templates pick different words) are practised once more -- where the learner's own
                # knowledge conflicts is where there is something to settle
                dctx, _ = L.items_tensor(items)
                with torch.no_grad():
                    h = model.habit(dctx)
                    dis = (h.argmax(1) != (h + model.pattern_term(dctx)).argmax(1)).nonzero().flatten().tolist()
                items = items + [items[j] for j in dis]
                wts = torch.cat([wts, wts[dis]])
                newm = torch.cat([newm, newm[dis]])
            ctx, tgt = L.items_tensor(items)
            opts = self.option_rows(items, p) if self.fmt_practice else None
            bag_p = L.bag(ctx, w.V)
            if s.get("self_side") == "capacity" and getattr(model, "side", None) is None:
                model.side = torch.nn.Parameter(torch.zeros(w.V, w.V))
            if s.get("self_side") is True and extra_items and p == len(phases) - 1:
                if getattr(model, "side", None) is None:
                    model.side = torch.nn.Parameter(torch.zeros(w.V, w.V))
                sm = torch.zeros(len(items), dtype=torch.bool)
                sm[n_heard_items:n_heard_items + len(extra_items)] = True
                selfmask[id(bag_p)] = sm
            if len(newm) < len(items):                  # self-made items appended above: not new material
                newm = torch.cat([newm, torch.zeros(len(items) - len(newm), dtype=torch.bool)])
            newmasks[id(bag_p)] = newm
            prepared.append((bag_p, tgt, {"pterm": model.pattern_term(ctx)}, wts,
                             torch.tensor(sorted({x for x in sents_p for x in x})), n, opts))
        rate = None
        if s.get("plastic") in ("pair", "pairnew"):
            # finding 105's follow-up: per CONNECTION, not per word (finding 105: per-word age froze the positions and verbs,
            # heard in every session, and so slowed the NEW links they had to make). Only the link table moves this way;
            # rate 1 / (1 + the pair's age), full for a link that moved the same way three sessions in a row
            r2 = 1.0 / (1.0 + self.plas2["age"])
            self.st_rate2 = torch.where(self.plas2["streak"] >= 2, torch.ones_like(r2), r2)
        elif s.get("plastic"):
            # Pioneer's idea (the user, 2026-10-02): old memory changes slowly, new memory fast, and a steady direction seeps
            # into the old. Per word: rate 1 / (1 + age) -- the running-mean step (a word heard in k past sessions is the mean
            # of k + 1 sessions' evidence; no constant) -- and full rate (1) for a word whose row moved the same way three
            # sessions in a row (streak >= 2: the user's three-in-a-row rule): steady evidence is believed
            rate = 1.0 / (1.0 + self.plas["age"])
            rate = torch.where(self.plas["streak"] >= 2, torch.ones_like(rate), rate)
        self.st = {"rate2": getattr(self, "st_rate2", None) if s.get("plastic") == "pair" else None,
                   "rate3": getattr(self, "st_rate2", None) if s.get("plastic") == "pairnew" else None,
                   "newmasks": newmasks, "rate": rate, "phases": prepared, "opt": torch.optim.Adam(model.parameters(), lr=s["lr"], fused=True), "done": 0, "preds": preds,
                   "selfmask": selfmask, "capacity": s.get("self_side") == "capacity"}
        model.side_on = bool(s.get("self_side")) and getattr(model, "side", None) is not None
        model.side_w = s.get("side_w", 1.0)
        if model.side_on and s.get("side_decay", 1.0) != 1.0:
            with torch.no_grad():
                model.side.mul_(s["side_decay"])       # finding 86 (a): the side part forgets at the chosen speed

    PARTS = ("chunk_mode", "compose", "setter", "versions", "fmt_practice")

    def wake(self, s):
        """dormant parts (the user's principle, 2026-09-26: a part that did not help is switched off, not removed, and can be
        woken when a problem comes): a settings dict may carry a part's switch, so the method layer can try waking it as a
        branch like any knob. Absent keys keep the learner as it is (the regression test is unchanged).
          chunk_mode    -- whole-sentence memory: "idf" (in use) / "idf-route" (81: idf + per-question routing) / "arbiter" (63) /
                           "ladder" (64) / "consensus" (64-65) /
                           "idf-arbiter" (finding 74's note: the arbiter's per-shape judgement on idf matching -- untested so far)
          compose       -- templates chained (71)
          setter        -- None / "hard" (71's setter, dormant) / "random" (71's self-made recombination practice)
          versions      -- templates kept in two versions instead of merged (61)
          fmt_practice  -- practising the options format (66)"""
        if "chunk_mode" in s:
            # finding 81: idf matching + per-question routing ("idf-route": rare-word gate; "idf-trust": learned per class)
            self.model.route = {"idf-route": "rare", "idf-trust": "trust"}.get(s["chunk_mode"], False)
            self.chunks.mode = "idf" if self.model.route else s["chunk_mode"]
        if "compose" in s:
            self.compose = s["compose"]
        if "setter" in s:
            self.setter = s["setter"]
        if "versions" in s:
            self.pattern.organize = "versions" if s["versions"] else None
        self.fmt_hard = s.get("fmt_hard", False)
        self.fmt_practice = s.get("fmt_practice", False) or self.fmt_hard
        self.model.agree2 = s.get("agree2", False)

    def shape_trust(self, new_sents):
        """arbiter / ladder (findings 63-64, run_schema3.life's chunk trust moved as it is): on the new sentences just heard
        (before practice, leaving out each question's own sentence), per match shape: did whole-sentence memory turn answers
        wrong -> right or right -> wrong? A shape is used while it has helped at least as often as it harmed."""
        model, chunks = self.model, self.chunks
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

    def route_trust(self, new_sents):
        """finding 81 v3: on the new sentences just heard (before practice, leaving out each question's own sentence -- from
        the memory AND from the co-occurrence it testifies with), per override class: did letting the memory override turn
        the answer wrong -> right or right -> wrong? A class is allowed while it has helped at least as often as it harmed
        (finding 61's stakes rule, as the arbiter's shapes)."""
        model, chunks = self.model, self.chunks
        items, ep = [], []
        for s_ in new_sents:
            for cx, t in D_positions([s_]):
                items.append((cx, t))
                ep.append(chunks.index[tuple(sorted(set(s_)))])
        nctx, ntgt = L.items_tensor(items)
        ex = torch.tensor(ep)
        with torch.no_grad():
            model.chunk_on = False
            z0 = model.complete(nctx)
            ev = chunks.evidence(nctx, exclude=ex)
            a, b, sa, sb = chunks.route_classes(nctx, z0, ev, exclude=ex)
        for i in range(len(items)):
            if a[i] == b[i]:
                continue
            key = (bool(sa[i]), bool(sb[i]))
            chunks.RH[key] = chunks.RH.get(key, 0) + float(b[i] == ntgt[i])
            chunks.RB[key] = chunks.RB.get(key, 0) + float(a[i] == ntgt[i])
        chunks.route_ok = {k: chunks.RH[k] >= chunks.RB.get(k, 0) for k in chunks.RH}
        model.chunk_on = True

    def option_rows(self, items, p):
        """finding 66's format practice, moved as it is: for each practice item the answer + 3 known words of its category.
        fmt_hard (tier 2, confusable wrong options, 2026-09-29): the 3 words of its category the learner itself now rates
        highest (the most tempting wrong answers, as people learn from close distractors) instead of 3 at random."""
        pr = np.random.default_rng(85_000 + self.seed + self.i + 1000 * p)
        z = None
        if self.fmt_hard:
            with torch.no_grad():
                z = self.model.complete(L.items_tensor(items)[0])
        rows = []
        for j, (cx, t) in enumerate(items):
            pool = [x for x in self.known_by_cat[int(self.cats[t])] if x != t and x not in cx]
            if len(pool) < 3:
                rows.append([t, t, t, t])
            elif z is not None:
                rows.append([t] + [pool[k] for k in torch.topk(z[j, pool], 3).indices.tolist()])
            else:
                rows.append([t] + [int(x) for x in pr.choice(pool, 3, replace=False)])
        return torch.tensor(rows)

    def cand_ranks(self, fitems):
        """(rank of the candidate among all words, is it the answer) for yes/no questions (verify_cont)."""
        if not fitems:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.bool)
        ctx, tgt = L.items_tensor([(cx, t) for cx, t, _, _ in fitems])
        with torch.no_grad():
            z = self.model.complete(ctx)
        z = z.masked_fill(S3.context_mask(ctx, z.shape[1]) > 0, float("-inf"))
        cand = torch.tensor([c for _, _, _, c in fitems])
        return (z > z.gather(1, cand.unsqueeze(1))).sum(1) + 1, cand == tgt

    def setter_items(self, heard):
        """finding 71, setter and solver (the user's idea, like a game-playing AI against itself): the setter makes questions
        from what was just heard -- a word with a strongly tied partner, the partner, and other known words of its category
        (a new combination, as Learner3.recombined_score) -- and keeps those the solver (the same learner, before practice)
        gets WRONG: the setter wins when the solver fails on a question whose answer is certain (it comes from a real heard
        sentence, so the setter cannot exploit a wrong model, finding 36's lesson). They are practised with the rest.
        "random": the same number of such questions, not chosen by failure (control)."""
        rng = np.random.default_rng(95_000 + self.seed + self.i)
        cond = self.rec.counts / self.rec.with_.clamp(min=1)[:, None]
        cand = []
        for s_ in heard:
            for t in s_:
                others = [x for x in s_ if x != t]
                if not others:
                    continue
                keep = max(others, key=lambda x: float(cond[t, x] + cond[x, t]))
                if max(float(cond[t, keep]), float(cond[keep, t])) < STRONG:
                    continue
                pool = [x for x in self.known_by_cat.get(int(self.cats[t]), []) if x not in s_]
                k = min(len(others) - 1, len(pool))
                cand.append(([keep] + ([int(x) for x in rng.choice(pool, k, replace=False)] if k > 0 else []), t))
        if not cand:
            return []
        ctx, tgt = L.items_tensor(cand)
        with torch.no_grad():
            wrong = (self.model.complete(ctx).argmax(1) != tgt).numpy()
        n = int(wrong.sum())
        if self.setter == "hard":
            return [c for c, bad in zip(cand, wrong) if bad]
        return [cand[j] for j in rng.choice(len(cand), n, replace=False)] if n else []

    def train_to(self, total, checks=None):
        """practise on until `total` epochs of this session are done (own optimizer kept); returns curves if checks."""
        model, cats, st = self.model, self.cats, self.st
        curves = {k: {} for k in (checks or {})}
        start = 0
        for bag, tgt, extra, wts, ids, n, opts in st["phases"]:
            for e in range(start, start + n):
                if st["done"] <= e < total:
                    if checks is not None and e in CHECKS:
                        for k, items in checks.items():
                            curves[k][e] = acc_items(model, items)
                    opt = st["opt"]
                    opt.zero_grad()
                    out = model.complete(bag, **extra)
                    sm = st.get("selfmask", {}).get(id(bag))
                    if st.get("capacity"):
                        # control (finding 85): the same extra part, trained by every item with no source separation --
                        # is the gain the separation, or only an extra V x V table?
                        out = out + L.masked_mean(model.side, bag)
                    elif sm is not None:
                        # self-made practice through its own side part (the memory-first rule's source axis, applied to
                        # learning): the self-made items move only the side part; the main habit sees them as fixed
                        out = torch.where(sm.unsqueeze(1), out.detach() + L.masked_mean(model.side, bag), out)
                    loss = S3.fused("wce")(out, tgt, wts) + F.cross_entropy(model.classify(ids), cats[ids])
                    if opts is not None:
                        loss = loss + F.cross_entropy(out.gather(1, opts), torch.zeros(len(tgt), dtype=torch.long))
                    loss.backward()
                    rate3 = st.get("rate3")
                    nm = st["newmasks"].get(id(bag)) if rate3 is not None else None
                    if nm is not None and nm.any():
                        # finding 106's follow-up: the new material's share of the link-table gradient, from its own
                        # forward pass (the compiled practice path cannot keep its graph for a second backward)
                        g_all = model.assoc.grad.detach().clone()
                        out_n = model.complete(bag, **extra)
                        if sm is not None:
                            out_n = torch.where(sm.unsqueeze(1), out_n.detach() + L.masked_mean(model.side, bag), out_n)
                        ce = F.cross_entropy(out_n, tgt, reduction="none") * wts
                        g_new = torch.autograd.grad(ce[nm].sum() / wts.sum(), model.assoc)[0]
                    rate = st.get("rate")
                    rate2 = st.get("rate2")
                    if rate is not None:
                        before = [p.detach().clone() for p in (model.z_work, model.assoc, model.skill)]
                    if rate2 is not None or (nm is not None and nm.any()):
                        a0 = model.assoc.detach().clone()
                    opt.step()
                    if nm is not None and nm.any():
                        # only the part of the step that came from NEW material is slowed (1 / (1 + the pair's age), full
                        # for a steady link); the part from review -- practising the same thing again -- moves at full step
                        with torch.no_grad():
                            g_rest = g_all - g_new
                            share = g_new.abs() / (g_new.abs() + g_rest.abs()).clamp(min=1e-12)
                            keep = 1.0 - share * (1.0 - rate3)
                            model.assoc.copy_(a0 + keep * (model.assoc - a0))
                    if rate2 is not None:
                        with torch.no_grad():
                            model.assoc.copy_(a0 + rate2 * (model.assoc - a0))
                    if rate is not None:
                        # old words move slowly, new words fast (per-row step = rate x the optimizer's step)
                        with torch.no_grad():
                            for p, b0 in zip((model.z_work, model.assoc, model.skill), before):
                                r = rate if p.dim() == 1 else rate.unsqueeze(1)
                                p.copy_(b0 + r * (p - b0))
            start += n
        st["done"] = max(st["done"], min(total, EPOCHS))
        if checks is not None and st["done"] == EPOCHS and EPOCHS in CHECKS:
            for k, items in checks.items():
                curves[k][EPOCHS] = acc_items(model, items)
        return curves

    def finish(self):
        # the plasticity record: this session's move of every habit row, compared with the last session's (same direction =
        # cosine > 0); the heard words grow older
        d = self.model.z_work.detach() - self.plas["start"]
        moved = d.norm(dim=1) > 0
        if self.plas["prev"] is not None:
            same = (F.cosine_similarity(d, self.plas["prev"], dim=1) > 0) & moved
            self.plas["streak"] = torch.where(same, self.plas["streak"] + 1, torch.zeros_like(self.plas["streak"]))
        self.plas["prev"] = torch.where(moved.unsqueeze(1), d, self.plas["prev"] if self.plas["prev"] is not None else d)
        self.plas["age"][self.plas["heard"]] += 1
        d2 = self.model.assoc.detach() - self.plas2["start"]
        sg = torch.sign(d2)
        if self.plas2["prev"] is not None:
            same = (sg == self.plas2["prev"]) & (sg != 0)
            self.plas2["streak"] = torch.where(same, self.plas2["streak"] + 1, torch.zeros_like(self.plas2["streak"]))
        self.plas2["prev"] = torch.where(sg != 0, sg, self.plas2["prev"] if self.plas2["prev"] is not None else sg)
        seen = torch.zeros_like(self.plas2["age"], dtype=torch.bool)
        for u in self.plas2["sents"]:
            idx = torch.tensor(u)
            seen[idx.unsqueeze(1), idx.unsqueeze(0)] = True
        self.plas2["age"] += seen.float()
        if self.content:
            self.pattern.score(self.st["preds"], self.rec)
            self.pattern.absorb(self.rec, self.known, self.memory)
            if self.pattern.organize == "versions":
                self.pattern.propose_versions(self.known, self.memory)
        self.st = None
        self.i += 1
        self.intact = self.intact and self.fingerprint() == self.memory_hash

    def judge_score(self, sentences):
        """score on these held-out sentences for the method layer's judge. Whole-sentence memory must not recall the very
        sentence a question was made from (integration step B, first check: the judge's held-out OLD sentences were
        remembered whole, so the judge graded verbatim recall -- finding 56's independence, as finding 62's leave-one-out)."""
        items, ep = [], []
        for s_ in sentences:
            for cx, t in D_positions([s_]):
                items.append((cx, t))
                ep.append(self.chunks.index.get(tuple(sorted(set(s_))), -1))
        if not items:
            return float("nan")
        ctx, tgt = L.items_tensor(items)
        with torch.no_grad():
            on = self.model.chunk_on
            self.model.chunk_on = False
            z = self.model.complete(ctx)
            self.model.chunk_on = on
            if self.content and on:
                ev = self.chunks.evidence(ctx, exclude=torch.tensor(ep))
                if self.model.route == "rare":
                    ev = ev * self.model.chunk_gate(ctx, z)
                elif self.model.route == "trust":
                    ev = ev * self.chunks.route_gate(ctx, z, ev, exclude=torch.tensor(ep))
                z = z + getattr(self.model, "chunk_w", 1.0) * ev
        return L.acc(z, tgt)

    def recombined_score(self, sentences, rng):
        """score on NEW combinations the learner makes itself from these (already practised) sentences (integration step
        B, second check: on old categories finding 50's judge still graded familiarity -- a slow learning rate kept the
        practised sentences and looked best, while the outside test, never-heard combinations, preferred a fast one;
        finding 48's bias, back in a world where remembering and generalising part ways). For each word t of a sentence:
        that has a strongly tied partner in it (P >= STRONG, the learner's own record -- the words it knows go together,
        like the outside test's hidden pair members), the partner stays, the rest of the
        context is replaced by as many other known words of t's category -- a question in the outside test's shape that
        was never heard. No free constant."""
        cond = self.rec.counts / self.rec.with_.clamp(min=1)[:, None]
        items = []
        for s_ in sentences:
            for t in s_:
                others = [x for x in s_ if x != t]
                if not others:
                    continue
                keep = max(others, key=lambda x: float(cond[t, x] + cond[x, t]))
                if max(float(cond[t, keep]), float(cond[keep, t])) < STRONG:
                    continue                        # only words the learner knows a partner of (its own strong tie)
                pool = [x for x in self.known_by_cat.get(int(self.cats[t]), []) if x not in s_]
                k = min(len(others) - 1, len(pool))
                ctx = [keep] + ([int(x) for x in rng.choice(pool, k, replace=False)] if k > 0 else [])
                items.append((ctx, t))
        if not items:
            return float("nan")
        ctx, tgt = L.items_tensor(items)
        with torch.no_grad():
            return L.acc(self.model.complete(ctx), tgt)

    def world_score(self, sentences, rng):
        """the imagination eye in the world's own question shape (2026-10-01, finding 98's (a)). recombined_score keeps ONE
        partner and fills the rest with t's category -- in the text world that is the outside test's shape, in the glyph world
        it is not (t = a "to" position: the verb and the start are thrown away and other positions put in; finding 95: that
        eye is blind there). Here, for each word t: every word the learner knows goes with t (its own record, P >= STRONG either
        way) stays -- the core of the moment -- and every other word x is replaced by another known word of x's OWN category
        and kind (the background changes, the core does not: "the same event, never seen in this company"). No constant."""
        items = self.world_items(sentences, rng)
        if not items:
            return float("nan")
        ctx, tgt = L.items_tensor(items)
        with torch.no_grad():
            return L.acc(self.model.complete(ctx), tgt)

    def world_items(self, sentences, rng):
        """world_score's questions (split out for finding 114's world-shaped self-experiment; the same draws)."""
        cond = self.rec.counts / self.rec.with_.clamp(min=1)[:, None]
        ko = getattr(self.model, "kind_of", None)
        ko = ko.tolist() if ko is not None else None
        bg = None
        if getattr(self, "bg_pool", False):
            # finding 118: the background is swapped only for words the learner's own record gives no strong partner
            # (either way) -- the outside test's background is partnerless filler words; pair members in the background
            # make the whole-sentence memory pull in wrong sentences, so the eye judged that memory too harshly late in life
            strong = torch.maximum(cond, cond.T).max(dim=1).values >= STRONG
            bg = set(torch.nonzero(~strong).flatten().tolist())
        items = []
        for s_ in sentences:
            for t in s_:
                others = [x for x in s_ if x != t]
                core = [x for x in others if max(float(cond[t, x]), float(cond[x, t])) >= STRONG]
                if not core:
                    continue                        # only words the learner knows a partner of (as recombined_score)
                ctx, used = list(core), set(s_)
                for x in others:
                    if x in core:
                        continue
                    pool = [y for y in self.known_by_cat.get(int(self.cats[x]), [])
                            if y not in used and (ko is None or ko[y] == ko[x]) and (bg is None or y in bg)]
                    if pool:
                        y = int(rng.choice(pool))
                        ctx.append(y)
                        used.add(y)
                items.append((ctx, t))
        return items

    def formats_score(self, items, rng_seed):
        res = S3.procedures(self.model, S3.formatted(items, self.cats, self.known_by_cat, np.random.default_rng(rng_seed)))
        out = {f"fmt {sh}": float(res[sh][self.policy[sh]].float().mean()) for sh in ("blank", "options")}
        if self.policy["candidate"].startswith("Vc"):
            rk, tr_ = self.cand_ranks(S3.formatted(items, self.cats, self.known_by_cat, np.random.default_rng(rng_seed)))
            out["fmt candidate"] = float(((rk <= int(self.policy["candidate"][2:])) == tr_).float().mean())
        else:
            out["fmt candidate"] = float(res["candidate"][self.policy["candidate"]].float().mean())
        return out


def _wce(out, tgt, wts):
    """the weighted practice loss (optimisation pass, 2026-09-28: fused with torch.compile)."""
    return (F.cross_entropy(out, tgt, reduction="none") * wts).sum() / wts.sum()


S3.FUSED["wce"] = _wce


def composed_matrix(pattern, memory, known, V, min_count=4):
    """finding 71, templates chained -- a simulation inside the templates: for every two templates in use with at least
    min_count confirmations (a representative value: the spurious ones mostly have 2-3), the direction t1 + t2 and t1 - t2
    is a composed template (A -> B -> C) of weight w1 x w2; evidence as for templates (match > 0.5, both signs)."""
    used = [(k, t, wgt) for k, t, wgt in pattern.weighted() if pattern.templates[k][1] >= min_count]
    C = torch.zeros(V, V)
    if len(used) < 2:
        return C
    words = torch.tensor(sorted(known))
    m = memory[words]
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            for sgn in (1.0, -1.0):
                t = used[i][1] + sgn * used[j][1]
                wgt = used[i][2] * used[j][2]
                for sign in (1, -1):
                    match = 1 - torch.cdist(m + sign * t, m) / t.norm()
                    match.fill_diagonal_(0)
                    val = torch.where(match > 0.5, wgt * match, torch.zeros_like(match))
                    C[words.unsqueeze(1), words.unsqueeze(0)] = torch.maximum(C[words][:, words], val)
    return C


def measure(me, tests, seen, new_cats, curves, seed):
    out = {"never-heard": acc_items(me.model, [(cx, t) for c in seen for kind, cx, t in tests[c]]),
           "newest": acc_items(me.model, [(cx, t) for c in new_cats for kind, cx, t in tests[c]]) if new_cats else np.nan,
           "conj all": acc_items(me.model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "conj"]),
           "once all": acc_items(me.model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "once"]),
           "pairs all": acc_items(me.model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind.startswith("R")]),
           "compose all": acc_items(me.model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "compose"]),
           "long all": acc_items(me.model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "long"]),
           "unseen all": acc_items(me.model, [(cx, t) for c in seen for kind, cx, t in tests[c] if kind == "unseen"]),
           "curve once": curves.get("once", {}), "curve conj": curves.get("conj", {})}
    out.update(me.formats_score([(cx, t) for c in seen for kind, cx, t in tests[c]], 90_000 + seed))
    return out


def life_fixed(world_name, seed):
    """the content layer alone with the start settings (step A: must equal run_schema3.life)."""
    w = build_world(world_name, seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 + seed)
    me = Learner3(w, seed)
    sents = {c: [s for _, s, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    tests = {c: W.unit_items(w.categories[c]["test"]) for c in range(w.n_cat)}
    order = [int(c) for phase in w.phases for c in rng.permutation(phase)]
    sessions = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    seen, steps = [], []
    for new_cats in sessions:
        old = list(seen)
        seen += new_cats
        checks = {"once": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind == "once"],
                  "conj": [(cx, t) for c in new_cats for kind, cx, t in tests[c] if kind in ("conj", "long")]}
        curves = me.session([s for c in new_cats for s in sents[c]], {c: sents[c] for c in old}, START, rng, checks=checks)
        steps.append(measure(me, tests, seen, new_cats, curves, seed))
    return steps


def check(seeds=(0, 1), worlds=("conj", "schema")):
    """step A: the moved learner against run_schema3.life with the same configuration."""
    for wn in worlds:
        for seed in seeds:
            a = life_fixed(wn, seed)
            b = S3.life(wn, "norm+explain", seed, evidence=True, independent=True, mask_context=True, stakes=True,
                        null="near", chunk="idf", formats="learned")
            for k in ("never-heard", "conj all", "once all", "fmt options", "fmt candidate"):
                va = [st[k] for st in a]
                vb = [st[k] for st in b]
                same = np.allclose(np.nan_to_num(va), np.nan_to_num(vb), atol=1e-9)
                print(f"{wn} seed {seed} {k:13} {'SAME' if same else 'DIFF'}  {np.round(va, 4).tolist()}"
                      + ("" if same else f"  vs {np.round(vb, 4).tolist()}"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
