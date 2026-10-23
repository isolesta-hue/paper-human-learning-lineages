"""Sequential two-stage learning test -- does identity-axis freezing protect
previously-learned ("stage 1") mappings from being overwritten while the
classifier learns NEW ("stage 2") words? Now compared against a genuinely
different protection mechanism borrowed from an outside independent
implementation (Gemini's adapter design), across a spectrum of how much
gets frozen:

  control_dim0      -- nothing frozen (full fine-tune every stage)
  candidate_dim3     -- our identity_dim=3 axis mask (encoder input axes only)
  full_freeze_dim16  -- our same mechanism, but ALL 16 encoder axes frozen
                        (classifier head still free)
  adapter            -- Gemini's design: encoder AND classifier head both
                        frozen completely after stage 1; stage 2 is learned
                        by a small separate bottleneck module added on top,
                        zero-initialized so it starts as a no-op. Stage-1
                        items are always evaluated with this adapter
                        bypassed, so stage-1 accuracy is architecturally
                        guaranteed unchanged -- not a trained-in property,
                        a structural one.

Why the spectrum, not just two conditions: an earlier version of this
script found identity_dim=3 gave NO measurable protection against
forgetting (300 seeds, sign-test p=0.21 -- not just underpowered, genuinely
absent). The suspected reason: identity_dim only ever masks encoder input
axes, never the classifier weights (W_cls/b_cls), and this scenario's
forgetting is plausibly driven mostly by the classifier head being
overwritten for stage 2, a path identity_dim never touches no matter how
large. full_freeze_dim16 isolates "does freezing the WHOLE encoder (still
leaving the classifier free) help at all", and adapter isolates "what if
the classifier is frozen too, Gemini-style, with a fresh small module
absorbing stage 2 instead."

Note on the two outside ChatGPT logs also reviewed for this test: they were
NOT ported here. Their first script has no old/new split at all (it
re-learns the SAME 32-word vocabulary across "cycles", comparing an
identity vs. a fixed-permutation internal target) -- a different question
from forgetting. Their second script does have an old/new split, but its
own reported result is "seeds with old-word drop: 0/20" for BOTH of its
conditions -- i.e. their own setup doesn't reproduce forgetting either, so
porting it here would not add a comparison point, only reproduce the same
"too easy, nothing to forget" trap this script's first version fell into.
"""
import math

import numpy as np

from imitation_core import (
    AutoencoderCore,
    InternalImitationLearner,
    ExternalLearner,
    RatchetMemory,
    softmax,
)

RAW_DIM = 40
LATENT_DIM = 16
IDENTITY_DIM = 3
BOTTLENECK_DIM = 8  # Gemini used 64/256 = 1/4 of hidden_dim; 8/16 matches that ratio
CATEGORIES = ["animal", "vehicle", "food", "color", "emotion", "weather"]
N_CLASSES = len(CATEGORIES)
WORDS_PER_CATEGORY_PER_STAGE = 45
WORD_NOISE_STD = 0.2
EPOCHS_PER_STAGE = 10
SEEDS = list(range(300))
OVERLAP_CATEGORIES = 3  # of N_CLASSES=6: how many categories keep the SAME
# center in stage 2 ("shared" structure -- e.g. classes 0..2), vs. getting
# an independently redrawn center ("novel" -- classes 3..5). The fully
# disjoint version (0 overlap) found Gemini's adapter mechanism protects
# old knowledge perfectly but nearly fails to learn stage 2 at all
# (0.235 acc, near chance); the held-out check on Gemini's own code
# suggested this happens specifically when stage 2 requires genuine
# generalization to an unrelated input distribution, not when there is
# some transferable shared structure. This tests that directly: does
# giving HALF the categories real shared structure change the tradeoff,
# and does it change differently for the "shared" vs "novel" halves?


def build_staged_vocab(seed):
    """Categories [0, OVERLAP_CATEGORIES) keep the exact same center between
    stage 1 and stage 2 (shared structure); categories
    [OVERLAP_CATEGORIES, N_CLASSES) get an independently redrawn center for
    stage 2 (novel structure, same as the fully-disjoint version of this
    script). Same label set throughout -- only some categories' meaning
    "moves" and some don't."""
    rng = np.random.default_rng(seed)
    centers1 = rng.uniform(0.2, 0.8, (N_CLASSES, RAW_DIM))
    centers2 = centers1.copy()
    for c in range(OVERLAP_CATEGORIES, N_CLASSES):
        centers2[c] = rng.uniform(0.2, 0.8, RAW_DIM)

    def make_pool(centers, n_per_cat):
        X, y = [], []
        for c in range(N_CLASSES):
            for _ in range(n_per_cat):
                X.append(np.clip(centers[c] + rng.normal(0, WORD_NOISE_STD, RAW_DIM), 0, 1))
                y.append(c)
        X, y = np.array(X), np.array(y)
        perm = rng.permutation(len(X))
        return X[perm], y[perm]

    X1, y1 = make_pool(centers1, WORDS_PER_CATEGORY_PER_STAGE)
    X2, y2 = make_pool(centers2, WORDS_PER_CATEGORY_PER_STAGE)
    return X1, y1, X2, y2


def split_train_val(X, y, val_fraction=0.2):
    n_val = max(6, int(len(X) * val_fraction))
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]


def train_epochs(external, X_train, y_train, epochs, rng):
    n = len(X_train)
    for _ in range(epochs):
        order = rng.permutation(n)
        for idx in order:
            external.train_step(X_train[idx], y_train[idx], update_encoder=True)


class BottleneckAdapter:
    """Gemini's adapter: latent -> bottleneck -> latent residual, added to
    the (frozen) encoder's output before the (frozen) classifier sees it.
    Last layer is zero-initialized so the adapter starts as an exact no-op
    -- z_adapted == z until training moves it, same trick Gemini's code
    uses (nn.init.zeros_ on the adapter's final layer)."""

    def __init__(self, latent_dim, bottleneck_dim, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(latent_dim), (latent_dim, bottleneck_dim))
        self.b1 = np.zeros(bottleneck_dim)
        self.W2 = np.zeros((bottleneck_dim, latent_dim))
        self.b2 = np.zeros(latent_dim)

    def apply(self, z):
        h = np.maximum(0, z @ self.W1 + self.b1)
        delta = h @ self.W2 + self.b2
        return z + delta, h

    def train_step(self, z, d_z_adapted, lr):
        h = np.maximum(0, z @ self.W1 + self.b1)
        d_W2 = np.outer(h, d_z_adapted)
        d_b2 = d_z_adapted
        d_h = d_z_adapted @ self.W2.T
        d_pre_h = d_h * (h > 0)
        d_W1 = np.outer(z, d_pre_h)
        d_b1 = d_pre_h
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2


def predict_bypass(external, X):
    """Predictions through the frozen encoder+classifier ONLY, no adapter --
    the path stage-1 items always use, matching Gemini's active_stage=None
    benchmark evaluation."""
    Z = np.tanh(X @ external.core.W_enc + external.core.b_enc)
    logits = Z @ external.W_cls + external.b_cls
    return np.argmax(logits, axis=1)


def predict_adapter(external, adapter, X):
    preds = []
    for x in X:
        z = external.core.encode(x)
        z_adapted, _ = adapter.apply(z)
        logits = z_adapted @ external.W_cls + external.b_cls
        preds.append(np.argmax(logits))
    return np.array(preds)


def group_accuracy(preds, y, mask=None):
    if mask is not None:
        preds, y = preds[mask], y[mask]
    if len(y) == 0:
        return float("nan")
    return float(np.mean(preds == y))


def train_adapter(external, adapter, X_train, y_train, epochs, rng):
    n = len(X_train)
    n_classes = external.W_cls.shape[1]
    for _ in range(epochs):
        order = rng.permutation(n)
        for idx in order:
            x, label = X_train[idx], y_train[idx]
            z = external.core.encode(x)
            z_adapted, h = adapter.apply(z)
            probs = softmax(z_adapted @ external.W_cls + external.b_cls)
            onehot = np.zeros(n_classes)
            onehot[label] = 1.0
            d_logits = probs - onehot
            d_z_adapted = d_logits @ external.W_cls.T
            adapter.train_step(z, d_z_adapted, lr=external.lr)


def run_pipeline(seed):
    X1, y1, X2, y2 = build_staged_vocab(seed)
    X1_train, y1_train, X1_val, y1_val = split_train_val(X1, y1)
    X2_train, y2_train, X2_val, y2_val = split_train_val(X2, y2)

    shared_mask_1 = y1_val < OVERLAP_CATEGORIES
    novel_mask_1 = ~shared_mask_1
    shared_mask_2 = y2_val < OVERLAP_CATEGORIES
    novel_mask_2 = ~shared_mask_2

    frozen_core = AutoencoderCore(RAW_DIM, LATENT_DIM, seed=seed)
    internal = InternalImitationLearner(frozen_core)
    for x in X1:
        internal.observe(x)
    for x in X2:
        internal.observe(x)  # Module A's own memory, never touches Module C
    ratchet = RatchetMemory(frozen_core)

    result = {"seed": seed}

    def record(label, preds1_after1, preds1_after2, preds2_after2):
        stage1_after_stage1 = group_accuracy(preds1_after1, y1_val)
        stage1_after_stage2 = group_accuracy(preds1_after2, y1_val)
        result[label] = {
            "stage1_after_stage1": stage1_after_stage1,
            "stage1_after_stage2": stage1_after_stage2,
            "forgetting": stage1_after_stage1 - stage1_after_stage2,
            "forgetting_shared": group_accuracy(preds1_after1, y1_val, shared_mask_1)
                                  - group_accuracy(preds1_after2, y1_val, shared_mask_1),
            "forgetting_novel": group_accuracy(preds1_after1, y1_val, novel_mask_1)
                                 - group_accuracy(preds1_after2, y1_val, novel_mask_1),
            "stage2_after_stage2": group_accuracy(preds2_after2, y2_val),
            "stage2_shared": group_accuracy(preds2_after2, y2_val, shared_mask_2),
            "stage2_novel": group_accuracy(preds2_after2, y2_val, novel_mask_2),
        }

    # --- identity_dim spectrum: 0 (control), 3 (candidate), 16 (full freeze) ---
    for label, identity_dim in [("control_dim0", 0), ("candidate_dim3", IDENTITY_DIM),
                                 ("full_freeze_dim16", LATENT_DIM)]:
        core = ratchet.hand_out_copy()
        external = ExternalLearner(core, N_CLASSES, seed=seed, identity_dim=identity_dim)

        train_epochs(external, X1_train, y1_train, EPOCHS_PER_STAGE, rng=np.random.default_rng(seed + 100))
        preds1_after1 = external.predict(X1_val)

        train_epochs(external, X2_train, y2_train, EPOCHS_PER_STAGE, rng=np.random.default_rng(seed + 200))
        preds1_after2 = external.predict(X1_val)
        preds2_after2 = external.predict(X2_val)

        record(label, preds1_after1, preds1_after2, preds2_after2)

    # --- Gemini-style adapter: encoder+classifier both frozen after stage 1 ---
    core = ratchet.hand_out_copy()
    external = ExternalLearner(core, N_CLASSES, seed=seed, identity_dim=0)
    train_epochs(external, X1_train, y1_train, EPOCHS_PER_STAGE, rng=np.random.default_rng(seed + 100))
    preds1_after1 = predict_bypass(external, X1_val)

    adapter = BottleneckAdapter(LATENT_DIM, BOTTLENECK_DIM, seed=seed + 400)
    train_adapter(external, adapter, X2_train, y2_train, EPOCHS_PER_STAGE, rng=np.random.default_rng(seed + 200))
    preds1_after2 = predict_bypass(external, X1_val)  # bypasses adapter -- should be IDENTICAL by construction
    preds2_after2 = predict_adapter(external, adapter, X2_val)

    record("adapter", preds1_after1, preds1_after2, preds2_after2)

    return result


def sign_test_pvalue(n_pos, n_neg):
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def paired_report(results, label, metric):
    diffs = np.array([r[label][metric] - r["control_dim0"][metric] for r in results])
    n_pos, n_neg = int(np.sum(diffs > 0)), int(np.sum(diffs < 0))
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs) / np.sqrt(len(diffs))) if len(diffs) else float("nan")
    p = sign_test_pvalue(n_pos, n_neg)
    print(f"{label:>18} {metric:>18}: mean(diff-vs-control)={mean_diff:+.4f} +/- SE {se_diff:.4f}  "
          f"n_pos={n_pos} n_neg={n_neg}  sign-test p={p:.4g}")


def main():
    results = [run_pipeline(seed) for seed in SEEDS]
    labels = ["control_dim0", "candidate_dim3", "full_freeze_dim16", "adapter"]

    def agg(label, metric):
        vals = [r[label][metric] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    print(f"=== {len(SEEDS)} seeds, {EPOCHS_PER_STAGE} epochs/stage, "
          f"{WORDS_PER_CATEGORY_PER_STAGE} words/category/stage, "
          f"{OVERLAP_CATEGORIES}/{N_CLASSES} categories share stage1/stage2 centers ===")
    print()
    print("=== stage-1 (old) validation accuracy, before vs after learning stage-2 (new) ===")
    for label in labels:
        before_m, before_s = agg(label, "stage1_after_stage1")
        after_m, after_s = agg(label, "stage1_after_stage2")
        forget_m, forget_s = agg(label, "forgetting")
        print(f"{label:>18}: before={before_m:.3f}+/-{before_s:.3f}  "
              f"after={after_m:.3f}+/-{after_s:.3f}  forgetting={forget_m:+.3f}+/-{forget_s:.3f}")

    print()
    print("=== forgetting, broken down by whether the category's center was SHARED or NOVEL in stage 2 ===")
    for label in labels:
        sm, ss = agg(label, "forgetting_shared")
        nm, ns = agg(label, "forgetting_novel")
        print(f"{label:>18}: shared-cat forgetting={sm:+.3f}+/-{ss:.3f}   novel-cat forgetting={nm:+.3f}+/-{ns:.3f}")

    print()
    print("=== stage-2 (new) validation accuracy after stage-2 training (did it still learn the new task?) ===")
    for label in labels:
        m, s = agg(label, "stage2_after_stage2")
        sm, ss = agg(label, "stage2_shared")
        nm, ns = agg(label, "stage2_novel")
        print(f"{label:>18}: overall={m:.3f}+/-{s:.3f}   shared-cat={sm:.3f}+/-{ss:.3f}   novel-cat={nm:.3f}+/-{ns:.3f}")

    print()
    print("=== paired comparison vs control_dim0, same seed both sides ===")
    for label in ["candidate_dim3", "full_freeze_dim16", "adapter"]:
        paired_report(results, label, "forgetting")
        paired_report(results, label, "stage2_after_stage2")


if __name__ == "__main__":
    main()
