"""Complexity ladder: instead of testing the full Module A/B/C pipeline as one
block, start from the simplest possible thing (word vectors -> linear
classifier, trained by hand-rolled backprop -- no encoder, no memory, no
blending, no attention) and add exactly ONE feature at a time. After each
addition, check whether the "blend several stored vectors, then classify"
step -- when it exists at that level -- collapses to one predicted class.

This isolates WHICH feature/complexity level actually causes the collapse
found in findings 12-14, rather than only comparing "everything on" vs
"everything off."

Per project decision (2026-09-22): the numpy-only rule is lifted, but nothing
here needs anything heavier -- kept in numpy so every gradient stays visible
in the code. Text/word data only; images are deliberately deferred until the
code is stable on this simpler modality.

Levels:
  0. raw word vector -> linear softmax classifier (no encoder, no memory, no blending)
  1. + an encoder (vectorization step) between raw input and the classifier,
     trained jointly end to end -- still no memory, no blending
  2. + an explicit per-word memory table (each word's own latent vector saved) --
     still no blending; just confirms having a memory doesn't itself break anything
  3. blend exactly 2 random memory vectors (plain unweighted mean), then classify
  4. blend 20 random memory vectors (plain unweighted mean), then classify
  5. blend the same 20, but with learned ATTENTION WEIGHTS instead of a plain mean
"""
import numpy as np

RAW_DIM = 40
LATENT_DIM = 16
CATEGORIES = ["animal", "vehicle", "food", "color", "emotion", "weather"]
N_CLASSES = len(CATEGORIES)
WORDS_PER_CATEGORY = 15
LR = 0.1
EPOCHS = 30
N_ROUNDS = 200
SEEDS = [0, 1, 2, 3, 4]


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def build_words(seed):
    """Same synthetic word vocabulary idea as word_vocab_test.py: each
    category gets a fixed random center in R^RAW_DIM, each word = center +
    small noise, clipped to [0, 1]."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, (N_CLASSES, RAW_DIM))
    X, y = [], []
    for c in range(N_CLASSES):
        for _ in range(WORDS_PER_CATEGORY):
            X.append(np.clip(centers[c] + rng.normal(0, 0.05, RAW_DIM), 0, 1))
            y.append(c)
    return np.array(X), np.array(y), rng


# --- Level 0: pure linear classifier on raw word vectors, no encoder at all ---
class LinearClassifier:
    def __init__(self, in_dim, n_classes, seed):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1 / np.sqrt(in_dim), (in_dim, n_classes))
        self.b = np.zeros(n_classes)

    def forward(self, x):
        return softmax(x @ self.W + self.b)

    def train_step(self, x, y_idx, lr=LR):
        probs = self.forward(x)
        onehot = np.zeros_like(probs)
        onehot[y_idx] = 1.0
        d = probs - onehot
        self.W -= lr * np.outer(x, d)
        self.b -= lr * d

    def predict(self, X):
        return np.array([np.argmax(self.forward(x)) for x in X])

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


def level0_test(seed):
    X, y, rng = build_words(seed)
    clf = LinearClassifier(RAW_DIM, N_CLASSES, seed)
    for _ in range(EPOCHS):
        for i in rng.permutation(len(X)):
            clf.train_step(X[i], y[i])
    acc = clf.accuracy(X, y)
    print(f"[seed {seed}] Level 0 (raw vector -> linear classifier, no blending): acc={acc:.3f}")


# --- Level 1: add an encoder (vectorization step), trained jointly, still no blending ---
class EncoderClassifier:
    """raw -> tanh(encode) -> linear classify, trained end to end by
    hand-rolled backprop through both layers."""

    def __init__(self, in_dim, latent_dim, n_classes, seed):
        rng = np.random.default_rng(seed)
        self.W_enc = rng.normal(0, 1 / np.sqrt(in_dim), (in_dim, latent_dim))
        self.b_enc = np.zeros(latent_dim)
        self.W_cls = rng.normal(0, 1 / np.sqrt(latent_dim), (latent_dim, n_classes))
        self.b_cls = np.zeros(n_classes)

    def encode(self, x):
        return np.tanh(x @ self.W_enc + self.b_enc)

    def forward(self, x):
        z = self.encode(x)
        return z, softmax(z @ self.W_cls + self.b_cls)

    def train_step(self, x, y_idx, lr=LR):
        z, probs = self.forward(x)
        onehot = np.zeros_like(probs)
        onehot[y_idx] = 1.0
        d_logits = probs - onehot
        self.W_cls -= lr * np.outer(z, d_logits)
        self.b_cls -= lr * d_logits
        dz = d_logits @ self.W_cls.T
        d_pre = dz * (1 - z ** 2)
        self.W_enc -= lr * np.outer(x, d_pre)
        self.b_enc -= lr * d_pre

    def predict(self, X):
        return np.array([np.argmax(self.forward(x)[1]) for x in X])

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


def train_encoder_classifier(seed):
    X, y, rng = build_words(seed)
    clf = EncoderClassifier(RAW_DIM, LATENT_DIM, N_CLASSES, seed)
    for _ in range(EPOCHS):
        for i in rng.permutation(len(X)):
            clf.train_step(X[i], y[i])
    return clf, X, y, rng


def level1_test(seed):
    clf, X, y, rng = train_encoder_classifier(seed)
    acc = clf.accuracy(X, y)
    print(f"[seed {seed}] Level 1 (+ encoder/vectorization step, no blending): acc={acc:.3f}")


# --- Level 2: add an explicit per-word memory table, still no blending ---
def level2_setup(seed):
    clf, X, y, rng = train_encoder_classifier(seed)
    memory = {i: clf.encode(X[i]) for i in range(len(X))}
    return clf, X, y, memory, rng


def level2_test(seed):
    clf, X, y, memory, rng = level2_setup(seed)
    acc = clf.accuracy(X, y)
    print(f"[seed {seed}] Level 2 (+ per-word memory table, no blending yet): "
          f"acc={acc:.3f}, memory size={len(memory)}")


# --- Levels 3-5: blend K stored memory vectors, then classify the blend ---
def blend_and_classify(clf, memory, keys, rng, k, weighted=False, W_k=None, q=None):
    idx = rng.choice(len(keys), size=k, replace=False)
    Z = np.stack([memory[keys[i]] for i in idx])
    if weighted:
        K = Z @ W_k
        w = softmax(K @ q)
    else:
        w = np.ones(k) / k
    combined = w @ Z
    probs = softmax(combined @ clf.W_cls + clf.b_cls)
    return int(np.argmax(probs))


def level3_test(seed, k=2):
    clf, X, y, memory, rng = level2_setup(seed)
    keys = list(memory.keys())
    labels = [blend_and_classify(clf, memory, keys, rng, k) for _ in range(N_ROUNDS)]
    counts = np.bincount(labels, minlength=N_CLASSES)
    print(f"[seed {seed}] Level 3 (blend {k} random memories, UNWEIGHTED mean, then classify): "
          f"unique={np.count_nonzero(counts)}  {counts}")


def level4_test(seed, k=20):
    clf, X, y, memory, rng = level2_setup(seed)
    keys = list(memory.keys())
    k = min(k, len(keys))
    labels = [blend_and_classify(clf, memory, keys, rng, k) for _ in range(N_ROUNDS)]
    counts = np.bincount(labels, minlength=N_CLASSES)
    print(f"[seed {seed}] Level 4 (blend {k} random memories, UNWEIGHTED mean, then classify): "
          f"unique={np.count_nonzero(counts)}  {counts}")


def level5_test(seed, k=20):
    clf, X, y, memory, rng = level2_setup(seed)
    keys = list(memory.keys())
    k = min(k, len(keys))
    W_k = rng.normal(0, 3 / np.sqrt(LATENT_DIM), (LATENT_DIM, LATENT_DIM))
    q = rng.normal(0, 3 / np.sqrt(LATENT_DIM), LATENT_DIM)
    labels = [blend_and_classify(clf, memory, keys, rng, k, weighted=True, W_k=W_k, q=q)
              for _ in range(N_ROUNDS)]
    counts = np.bincount(labels, minlength=N_CLASSES)
    print(f"[seed {seed}] Level 5 (blend {k} random memories, ATTENTION-WEIGHTED, then classify): "
          f"unique={np.count_nonzero(counts)}  {counts}")


def main():
    for seed in SEEDS:
        level0_test(seed)
        level1_test(seed)
        level2_test(seed)
        level3_test(seed)
        level4_test(seed)
        level5_test(seed)
        print()

    print("Reading guide: levels 0-2 have no blending step at all, so 'collapse' doesn't")
    print("apply to them -- they're sanity checks that the classifier itself works (acc")
    print(f"should be near 1.0). From level 3 onward, 'unique' should stay near {N_CLASSES}")
    print("(the number of categories) if blending several memories doesn't cause collapse.")
    print("Whichever level's unique count first drops to 1-2 marks where the problem starts.")


if __name__ == "__main__":
    main()
