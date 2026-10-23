"""Is the mode-collapse finding (README findings 12-13) specific to IMAGES, or
does it show up with a completely different input modality too?

Every experiment so far (run_experiment.py, identity_sweep.py,
self_anchor_control.py) used sklearn's handwritten-digit images as input --
despite the project's own "a baby learns words" framing, "word" in the code's
docstrings/comments was always just a generic placeholder for "one input
item," never actual text. That's worth testing directly: if blending several
DIGIT IMAGES and decoding produces a blurry, out-of-distribution image (the
explanation in finding 13), that's plausibly an images-specific effect. If
the SAME collapse-to-one-class happens with words -- which have no pixel-like
"blur" concept at all -- that would mean the cause is a property of the
algorithm itself (Module A/B/C are literally just numpy operations on
whatever-dimensional vectors; nothing in them is image-specific), not of the
data being images.

No real NLP library is used (matches the project's "numpy only" rule): each
of a handful of semantic categories gets a fixed random "meaning center" in
R^RAW_DIM, and:
  - a WORD's vector = that center + small noise (so words in the same
    category cluster tightly, the way same-digit images do)
  - a SENTENCE's vector = the mean of 3 words' vectors from the SAME
    category (a standard bag-of-words sentence representation) -- smoother
    and more centrally located than any single word, the way a sentence
    embedding is smoother than one word's embedding

Two vocabularies are tested, holding the algorithm (imitation_mlp.py)
completely fixed:
  A. words only
  B. words + sentences mixed together

If collapse appears in vocabulary A (words only, no sentences, no images at
all), the cause is confirmed to be in the algorithm, not the data modality.
"""
import numpy as np

from imitation_mlp import (
    SELF_KEY,
    AutoencoderCore,
    InternalImitationLearner,
    ExternalLearner,
    RatchetMemory,
    AttentionCombiner,
)
from run_experiment import train_ratcheted

RAW_DIM = 40          # synthetic "raw embedding" dimension (analogous to 64 pixels for digits)
LATENT_DIM = 16        # same as the digits experiments, for direct comparability
IDENTITY_DIM = 3
CATEGORIES = ["animal", "vehicle", "food", "color", "emotion", "weather"]
WORDS_PER_CATEGORY = 10
SENTENCES_PER_CATEGORY = 10
EPOCHS = 15
N_ROUNDS = 200
SEEDS = [0, 1, 2, 3, 4]


def build_word_vocab(seed):
    """Category centers are themselves random per seed (a fresh "meaning
    space" each run) but fixed once drawn -- analogous to a fresh digits
    dataset split. Each word = center + small gaussian noise, clipped to
    [0, 1] to match AutoencoderCore's sigmoid decoder range (same convention
    as digits.data / 16.0)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, (len(CATEGORIES), RAW_DIM))

    X, y = [], []
    for cat_idx in range(len(CATEGORIES)):
        for _ in range(WORDS_PER_CATEGORY):
            vec = centers[cat_idx] + rng.normal(0, 0.05, RAW_DIM)
            X.append(np.clip(vec, 0, 1))
            y.append(cat_idx)
    return np.array(X), np.array(y), centers, rng


def build_sentence_vocab(centers, rng):
    """A sentence = mean of 3 words drawn fresh from the SAME category
    (bag-of-words). Regenerating words on the fly (rather than reusing the
    word vocab's exact vectors) keeps sentences a genuinely distinct set of
    points, not just re-labeled duplicates of the word vocab."""
    X, y = [], []
    for cat_idx in range(len(CATEGORIES)):
        for _ in range(SENTENCES_PER_CATEGORY):
            words = centers[cat_idx] + rng.normal(0, 0.05, (3, RAW_DIM))
            sentence_vec = words.mean(axis=0)
            X.append(np.clip(sentence_vec, 0, 1))
            y.append(cat_idx)
    return np.array(X), np.array(y)


def run_vocab_test(X, y, seed, label):
    n = len(X)
    n_val = max(6, n // 5)
    perm = np.random.default_rng(seed + 900).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    n_classes = len(CATEGORIES)

    frozen_core = AutoencoderCore(RAW_DIM, LATENT_DIM, seed=seed)
    internal = InternalImitationLearner(frozen_core)
    for x in X:  # Module A observes the WHOLE vocabulary (train + val)
        internal.observe(x)

    ratchet = RatchetMemory(frozen_core)
    candidate_core = ratchet.hand_out_copy()
    candidate_external = ExternalLearner(candidate_core, n_classes, seed=seed, identity_dim=IDENTITY_DIM)
    rng = np.random.default_rng(seed + 300)
    for _epoch in range(EPOCHS):
        order = rng.permutation(len(X_train))
        for idx in order:
            candidate_external.train_step(X_train[idx], y_train[idx])

    combiner = AttentionCombiner(internal, candidate_external, seed=seed,
                                  pool_size=min(20, len(X) - 1))
    blend_labels = [combiner.combine_once(confidence_threshold=0.6)["label"] for _ in range(N_ROUNDS)]

    real_preds = candidate_external.predict(X)
    val_acc = candidate_external.accuracy(X_val, y_val)

    print(f"[seed {seed}] {label}: n={n}  val_acc={val_acc:.3f}")
    print(f"   blended-attention predicted-label counts (200 rounds): "
          f"unique={np.count_nonzero(np.bincount(blend_labels, minlength=n_classes))}  "
          f"{np.bincount(blend_labels, minlength=n_classes)}")
    print(f"   same classifier on REAL (unblended) vocab items:      "
          f"{np.bincount(real_preds, minlength=n_classes)}  (true dist: {np.bincount(y, minlength=n_classes)})")


def main():
    print("=== Vocabulary A: WORDS ONLY (no sentences, no images at all) ===")
    for seed in SEEDS:
        X_words, y_words, centers, rng = build_word_vocab(seed)
        run_vocab_test(X_words, y_words, seed, "words-only")

    print()
    print("=== Vocabulary B: WORDS + SENTENCES MIXED ===")
    for seed in SEEDS:
        X_words, y_words, centers, rng = build_word_vocab(seed)
        X_sent, y_sent = build_sentence_vocab(centers, rng)
        X_mixed = np.concatenate([X_words, X_sent], axis=0)
        y_mixed = np.concatenate([y_words, y_sent], axis=0)
        run_vocab_test(X_mixed, y_mixed, seed, "words+sentences")

    print()
    print("Reading guide: 'unique' should be close to the number of categories")
    print(f"({len(CATEGORIES)}) if blending doesn't collapse. unique=1 here, with no images")
    print("involved at all, would mean the collapse is in the algorithm itself, not in images.")


if __name__ == "__main__":
    main()
