# Related work (draft) — and how this study differs

- **Meta-learning / learning to learn**: MAML (Finn et al. 2017), learned optimizers, in-context learning. They learn the
  learning rule across many tasks; here one learner changes its own settings within one life, judged against its fixed twin.
- **Hyperparameter optimisation during training**: Hyperband / successive halving (Li et al. 2017), Population Based
  Training (Jaderberg et al. 2017). Our method layer is close to these; the difference is the self-made judge (no outside
  validation set) and the completion criterion.
- **Self-referential / self-improving systems**: Gödel machine (Schmidhuber 2003), self-referential weight matrices
  (Irie et al. 2022), STaR (Zelikman et al. 2022), Reflexion (Shinn et al. 2023), AlphaEvolve / FunSearch.
- **Open-endedness**: POET (Wang et al. 2019) — environments that get harder with the learner; our F 111 is a small,
  controlled instance of "the world must keep giving something to fix".
- **Complementary learning systems**: McClelland et al. 1995; our whole-sentence memory vs templates, BatonNet hippocampus
  vs cortex (NH 5).
- **Knowledge graph embeddings**: TransE (Bordes et al. 2013) — our templates (b = a + Δ_k) are the same form.
- **Metacognition**: Nelson & Narens 1990; Flavell 1979; error monitoring and perseveration (Milner 1963).
- **Continual learning**: catastrophic interference, replay.
Differences in one line: a single small learner, three bodies under one protocol, a strict pre-registered comparison
against its own fixed twin, exact self-observation used for counterfactual self-experiments, and a full negative-results log.
- **Name clash noted**: "Voyager" (Wang et al. 2023, an LLM agent in Minecraft) is unrelated; this is why the public name is Helix Gold.
