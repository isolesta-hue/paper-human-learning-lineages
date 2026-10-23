# When does learning to learn pay off? Small self-modifying learners in a world that gets harder

*Technical report — draft v0.1 (private). Numbers cite source findings: F n = Helix Gold, P n = Helix Teal, NH n = Helix Blue.
Every number here must be re-checked against the regeneration script before release (see REPRODUCTION_PLAN.md).*

## Abstract
We study whether a small learner can improve how it learns within one life, and measure it with a strict test: the
self-modifying learner versus the same learner with fixed settings, on the same data, scored on an outside test it never
sees; the gap must widen over sessions. The same protocol is applied to three bodies: a modular associative memory (GRA,
Helix Gold), a small transformer (Helix Teal) and a brain-inspired lattice with a hippocampus-like memory (BatonNet, Helix Blue).
The criterion is met in all three. In Helix Gold the gap stops widening once a hand-improved body leaves little to fix, and a
world that keeps getting harder restores the widening (late widening +0.033, 78:22, 100 seeds). Giving the learner exact
observation of its own answer process lets it diagnose its errors and run counterfactual self-experiments without
relearning; a method layer that sets part weights by such self-experiments beats the knob-only method layer in the harder
world (end +0.050, 25:5). All comparisons are pre-registered, run on 30 then 100 seeds when undecided, and negative results
are reported.

## 1. Introduction
Human learners do more than adjust weights: they notice their own errors, try a different way of studying, and keep what
works. Large models acquire much of this implicitly; we ask whether a *small, inspectable* system can do a measurable part
of it explicitly. The design follows a fixed order — a dashboard of self-measurements, modes chosen from the dashboard
(the user's "emotions as modes"), and actions chosen within a mode — and grows the learner's self-knowledge step by step.

**Contributions.** (1) The completion criterion as an operational test of learning-to-learn within one life. (2) One protocol
applied to three different bodies. (3) The spiral result: learning-to-learn pays off when the world keeps giving something
to fix (F 111, F 116). (4) Exact self-observation used for counterfactual self-experiments, and the first self-made method
that beats the hand-designed knob menu (F 112–115). (5) A long, pre-registered negative-results record.

## 2. Worlds
- **Text world `life32`.** 32 categories of words with sensory features, 16 sessions, two rule changes (new categories and
  changed relations). Sentences combine words through relation templates. Outside tests: never-heard combinations, once-heard
  pairs, old categories (forgetting).
- **World ladder (stacked).** Rung 1 (`@share`, F 111): from session 9, half of the filler words are 48 common words shared by
  all topics with Zipf frequencies. Rung 2 (`@blind`, F 116): in addition, half of each new category's words lose their sensory
  features (learnable only from context, like abstract and function words). Rung 3 (word order changes meaning) needs a new
  body part; rung 4 is very easy real sentences, Korean and English separately.
- **Glyph-picture world.** Egocentric 5×5 positions, motion verbs as relations, upside-down rooms.

## 3. The learner (Helix Gold)
- **Body (GRA).** Element memory (word vectors grown from an origin), habit (similarity + association table), Hebbian
  co-occurrence record, relation templates (b ≈ a + Δ_k), whole-sentence (chunk) memory, a side part, and a kind gate.
- **Method layer.** Each session: branch the learner over settings (learning rate, review share, knobs), prune by successive
  halving, judge on held-out own sentences and on world-shaped self-made questions, keep the winner.
- **Modes (F 104).** Protect / settle / explore, chosen by a three-session rule from the dashboard; each mode fixes a bundle
  of allowed changes.
- **Self-observation (F 112).** The answer score is split exactly into part shares (similarity, association, template, chunk,
  side; gate as a mask; split error ≤ 1.5e-5). Each wrong answer gets a label from the learner's own knowledge (kind, topic,
  pulled by the record, near neighbour, other) and the part that made it win.
- **Self-experiment (F 113–115).** With exact shares, "what if part p weighed w ∈ {0, 0.5, 1, 1.5}?" is computed on self-made
  questions without relearning; the method layer uses the best weight for answering (`+tune`).

## 4. Protocol
Pre-registered comparisons are written in each runner's docstring before running. 30 seeds; if undecided (p ≥ 0.005, paired
sign test) then 100. Regression check (same numbers as before on cached seeds) after every code change. No constant tuning:
representative values are declared. Dormant parts are kept, never deleted. Reported quantities: end gap, widening
(sessions 13–16 minus 2–4), late widening (13–16 minus 6–8), and headroom-normalised gap.

## 5. Results
### 5.1 The criterion in three bodies
Helix Gold meets the criterion (F 103, 100 seeds; fixed = the old body's starting settings, self-modifying = same body +
method layer with improvement knobs): end +0.113 (97:3), widening +0.013 (65:35, p=0.0035). Helix Teal alone plateaus
(never-heard 0.22 → 0.13) but the method layer widens its gap at every change, +0.008 → +0.078 (27:3), first raising
review against forgetting, then the learning rate after the rule change — the same prescription order as in GRA (F 76).
Helix Blue meets the criterion with BatonNet as the body (NH 27).

### 5.2 Rediscovery (F 103)
Starting from the old body and given our hand-found improvements as knobs that start off, the method layer closes 89% of
the gap to the hand-improved body (knob-less method layer: 24%; 30 seeds; with vs without knobs end +0.084, 25:4). The plateau we saw was partly ours: we had taken the
method layer's share by improving the body by hand.

### 5.3 What did not help (F 105–109)
Per-word plasticity (old slow, new fast), forgetting-aware review, and letting the method layer create its own options
were all used but gave no gain (e.g. F 109, 100 seeds: end −0.005, 55:45). They are kept as dormant parts.

### 5.4 The spiral (F 111, F 116)
| World | end gap | late widening (13–16 − 6–8) |
|---|---|---|
| plain | +0.113 (97:3) | +0.005 (60:40) |
| rung 1, shared words | +0.143 (99:1) | **+0.033 (78:22, p<0.0001)** |
| rung 2, + senseless words (30 seeds) | +0.143 (30:0) | +0.037 (24:6, p=0.0014) |

The harder world widens the gap mostly because the fixed learner falls more; the cost of senseless words is paid only by
the fixed learner (−0.008, 4:26), the method layer absorbs it (+0.001).

### 5.5 Self-observation (F 112)
In the glyph world, the late-life collapse is almost entirely made by the chunk memory: its share of wrong answers rises
from 0.075 to 0.652 over the life, while the association table's falls from 0.59 to 0.29.

### 5.6 Self-experiment (F 113–115)
On held-out own sentences, the self-test agrees with the outside test only late in life (chunk weight rank correlation
−0.50 → +0.82) and is biased for the association table (F 113). Replacing it with **world-shaped** self-made questions
(core pair kept, background swapped for other known words) makes the chosen weights help outside too (chunk, sessions 9–16
+0.0087, 10:0) (F 114). Used by the method layer (F 115): rung-1 world, end +0.050 (25:5, p=0.0003) and life +0.038 over the
knob-only method layer; completion gap +0.183 vs +0.134. Plain world: faster early (life +0.018, 66:34, 100 seeds) but equal
at the end.

### 5.7 A rejected hypothesis (F 117)
Using the chosen weights only for answering, while the method layer's branches are judged at weight 1, is worse (plain end
−0.032, 8:22): the eye that judges must see the learner as it actually answers.

### 5.8 Memory compression (F 110)
Keeping only the top-k cells per row of the connection table (the rest as the row mean) keeps 98% / 97% / 95% of
never-heard accuracy at 1/8 / 1/32 / 1/128 of the cells (3 cells per row); low-rank products keep only 55–75%. Scaling is
V×k instead of V².

## 6. Discussion
- **Learning-to-learn needs a world with something left to fix.** This is a small, controlled instance of the open-endedness
  argument (POET): the method layer's value shows where the fixed learner fails.
- **Self-observation before more knobs.** Extending the menu did not help (F 105–109); seeing the answer process exactly
  and experimenting on it did (F 115).
- **The eye's shape is the bottleneck.** Self-tests that do not resemble the world mislead (F 113); a world-shaped eye still
  over-penalises chunk memory in the plain world (open question, F 118 in progress).
- **Limits.** Tiny synthetic worlds; hand-designed knobs; per-world methods; no standard benchmark yet; matched-size outside
  baselines partly missing (Helix Teal covers the transformer side).

## 7. Next steps (pre-registered order)
(1) Let the learner score its own what-if predictions to set how much to trust them (surprise / confusion modes);
(2) an alarm when two eyes disagree; (3) judging procedures themselves as method material; then a method notebook within and
across lives, and a world-model-first body for trying-out in the glyph world.

## Authorship and AI-use statement
**Author:** the repository owner (sole human author; responsible for all content and decisions).
AI systems are not listed as authors (they cannot take responsibility for the work, and most venues and arXiv do not allow
it); their roles are disclosed here. Sources: the commit trailers of the source repositories and the owner's uploaded
conversation logs, summarised in the Helix Gold README ("ChatGPT / Gemini" section).
- **Development and experiments (Helix Gold, Helix Teal, Helix Blue):** Claude Opus 5.5 (Anthropic), via Claude Code — designed
  and wrote the code, ran the experiments, kept the records and drafted this report, under the owner's direction and
  research rules (a few Helix Teal commits: Claude Sonnet 5.5).
- **Idea discussion:** ChatGPT (OpenAI, free tier) and Gemini (Google, free tier). The owner discussed about 70 questions
  with both — uneven self-improvement (Goodhart), alignment as education, compression as intelligence, the collapse of
  self-training versus keeping old memories for comparison, a fixed "self" as the reference point of every world model,
  glyph-picture world models. Each idea was later checked by Claude against our experimental results (adopted, refined, or
  out of scope); ChatGPT more often added sources and corrections, Gemini more often agreed.
- **Independent implementations used as references:** both assistants implemented the same early design. Adopted from
  Gemini: a frozen backbone with small adapters, and snapshot-and-rollback around each stage; from ChatGPT: optimising the
  vectors directly, drift diagnostics between groups, multi-seed statistics, and the original-memory / working-copy flow
  (which matched our design). Gemini's adapter code was kept as a comparison sub-track (F 18–21).
- **Process:** the owner set goals and principles and made every decision; the rules (pre-registration, two-stage seeds,
  regression checks, keeping failures, declaring representative values) make the record auditable.
