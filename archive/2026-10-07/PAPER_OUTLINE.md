# Paper outline (draft v0.1) — learners modelled on human learning

Public names (decided 2026-10-06, see NAMES.md): **Helix Gold** (internal: Voyager, body GRA), **Helix Teal** (internal: Pioneer, transformer body),
**Helix Blue** (internal: New Horizons, BatonNet body). Source records keep the internal names; the paper uses the public ones only.
Every number cites its source record: "F n" = Helix Gold finding n, "P n" = Helix Teal finding n, "NH n" = Helix Blue finding n.

## Title candidates
1. *When does learning to learn pay off? A small closed-loop self-improving learner, three bodies, and a world that gets harder*
2. *Self-observation, self-experiment, self-correction: a pre-registered study of small self-modifying learners*
3. *Learning how to learn in a toy world: 117 pre-registered findings from a human–AI research collaboration*

## Abstract (draft)
We study a small "closed recursive improvement" loop: a learner whose method layer branches its own training, judges the
branches with self-made tests, and changes its learning settings, compared against the same learner with fixed settings on
the same data (the *completion criterion*: the self-modifying learner must beat the fixed one on an outside test the learner
never sees, and the gap must widen over sessions). The same principles, worlds and criterion are implemented in three
different bodies — a modular associative memory (GRA), a transformer, and a brain-inspired lattice with a hippocampus-like
memory (BatonNet). Every comparison is pre-registered, run on 30 and, if undecided, 100 seeds with paired sign tests, and
regression-checked; negative results are kept. Main results: (i) the criterion is met in all three bodies (GRA F 103; BatonNet
NH 27; transformer F 73/76 with the method layer), but in GRA the gap stops widening once the hand-improved body leaves the
method layer little to fix; (ii) **a world that keeps getting harder restores the widening** (F 111: late widening +0.033,
78:22, 100 seeds, vs +0.005 in the plain world); (iii) giving the learner **exact self-observation** — the answer score splits
exactly into its parts — lets it diagnose its own failures (F 112) and run **counterfactual self-experiments** without
relearning (F 113–114); choosing part weights this way beats the knob-only method layer in the harder world (F 115: end
+0.050, 25:5) while in the plain world it only speeds learning (life +0.018, 66:34, 100 seeds). (iv) Extending the method
layer's menu by itself does not help (F 105–109), and a counterfactual eye is only as good as the shape of the self-test
it runs on (F 113 vs 114). We also report the research process: a non-specialist directing an AI coding assistant over
~two weeks with a fixed set of research rules.

## 1. Introduction
- Motivation: the user's question — what do human learners have (self-monitoring, emotions as modes, self-made study
  methods) that LLMs/MLPs lack; can a *small* system improve how it learns, measurably?
- Contributions: the completion criterion as a test of learning-to-learn; three bodies under one protocol; the spiral
  result (F 111); exact self-observation → self-experiment (F 112–115); a long negative-results record.

## 2. Setup
- Worlds: text world `life32` (32 categories, 16 sessions, two rule changes; relation templates; once-heard and
  never-heard tests) and glyph-picture world (egocentric 5×5 positions, motion verbs as relations, upside-down rooms).
  World ladder: rung 1 common words with Zipf frequencies (F 111), rung 2 senseless words (F 116: late widening +0.037, 24:6, p=0.0014; the cost of senseless words is paid only by the fixed learner).
- Bodies: GRA (element memory, habit, Hebbian record, templates, whole-sentence memory, side part); transformer (Helix Teal);
  BatonNet (lattice cortex + DG/CA hippocampus, context rule sets; Helix Blue, NH).
- Method layer: branches, successive halving, judge on held-out own sentences, knobs; modes (protect/settle/explore, F 104);
  rediscover knobs (hand-found improvements given as knobs that start off, F 103).
- Protocol: pre-registered comparisons in each runner's docstring; 30 → 100 seeds, sign tests p < 0.005; regression check
  after every code change; dormant parts kept; no constant tuning (representative values declared).

## 3. Results
3.1 Completion criterion across bodies (F 103, NH 27, F 73/76, P 12/21/22).
3.2 "We took the method layer's share" (F 103): improvements we found by hand, given back as knobs, are rediscovered.
3.3 The plateau and failed fixes (F 105–109): plasticity variants, forgetting review, self-made options.
3.4 The spiral: harder worlds restore widening (F 111, F 116) — mostly by the fixed learner falling more.
3.5 Self-observation (F 112): exact part attribution; the glyph world's late collapse = whole-sentence-memory errors.
3.6 Self-experiment (F 113–115): the self-test's shape matters; world-shaped questions; wins in the harder world.
3.7 Negative results worth knowing (F 117: the method layer's eye must see the weights the learner answers with; F 109).
3.8 Memory compression (F 110): the connection table is a sparse exception memory (top-k per row keeps 95–98%; low-rank fails).

## 4. Discussion
- Learning-to-learn needs something to fix (spiral principle) — links to open-endedness (POET).
- Self-monitoring is the missing piece, not more knobs; the four missing mechanisms (README idea, 2026-10-05).
- Limits: tiny synthetic worlds; no standard benchmarks yet; method layer uses hand-designed knobs; per-world methods.

## 5. Research process (short)
- Human–AI collaboration: the user set goals/principles and made decisions; the AI assistant designed, coded, ran and
  reported; rules (pre-registration, regression checks, keep failures) made it auditable.

## Figures / tables to make
1. Completion-gap curves by session: plain vs rung 1 vs rung 2 (F 111, F 116).
2. Part attribution of wrong answers over the life, both worlds (F 112).
3. What-if curves: self vs world-shaped vs outside (F 113–114).
4. Table: all findings with verdict (adopted / rejected / dormant).

## Needed before release (see RELEASE_CHECKLIST.md)
- External baselines (small MLP / transformer of matched size on the same worlds; Helix Teal covers part).
- One script that regenerates the paper's numbers from the caches.
- English summary README in each public code copy (README_EN.md here is the summary for the write-up).
