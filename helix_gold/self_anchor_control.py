"""Control experiment: is Module B's "always collapses to one predicted
class" finding (README finding 12) actually caused by self being a fixed,
meaningful center point -- or does the same collapse happen for a reason
that has nothing to do with self at all?

Three groups, same trained classifier and Module A memory each time:
  A. self-anchored  -- the shipped AttentionCombiner (fixed q_self every round)
  B. random-anchor  -- RandomAnchorCombiner below: a fresh, genuinely
     different REAL memory vector as the query every round (no fixed center
     at all; the same score_scale/temp/pool_size mechanics otherwise)
  C. no-reinforcement -- group A's mechanism again, but with learn=False, so
     q_self/W_k/W_v never move from their fresh random initialization

If self were doing something specifically meaningful, group B (no self, no
fixed center, anchor spread ~uniformly across all 10 classes every run)
should look different from group A. If the collapse were a product of the
200-round reinforcement/self-training loop, group C (zero training) should
look different from group A too. Empirically, neither difference shows up:
all three groups collapse to a single predicted class, and self-anchored
and random-anchor collapse to the SAME class per seed. Meanwhile the same
classifier correctly and diversely classifies the 400 real (unblended)
training images. The real cause: decode(any softmax-blend of several
encoded digit vectors) lands in a region of image space this classifier
was never trained on and has an out-of-distribution blind spot for --
nothing to do with self, training, or which specific vectors got blended.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from imitation_mlp import (
    SELF_KEY,
    AutoencoderCore,
    InternalImitationLearner,
    ExternalLearner,
    RatchetMemory,
    AttentionCombiner,
    softmax,
)

LATENT_DIM = 16
IDENTITY_DIM = 3
EPOCHS = 15
IMITATION_N = 400
N_ROUNDS = 200
SEEDS = [0, 1, 2, 3, 4]


class RandomAnchorCombiner:
    """Group B: identical mechanics to AttentionCombiner (pool_size,
    score_scale, learnable temp) except the query is a fresh, genuinely
    different REAL memory vector each round instead of a permanently fixed
    self vector. Needs its own W_q since the anchor is a real, nonzero
    vector every time -- self_vector's zero-vector dead-gradient trap
    (see AttentionCombiner's docstring) doesn't apply here.
    """

    def __init__(self, internal, external, attn_dim=None, lr=0.05, seed=3,
                 pool_size=20, score_scale=3.0):
        self.internal = internal
        self.external = external
        latent_dim = internal.core.W_enc.shape[1]
        self.attn_dim = attn_dim or latent_dim
        rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(latent_dim)
        self.W_q = rng.normal(0, scale * score_scale, (latent_dim, self.attn_dim))
        self.W_k = rng.normal(0, scale * score_scale, (latent_dim, self.attn_dim))
        self.W_v = rng.normal(0, scale, (latent_dim, latent_dim))
        self.temp = 1.0
        self.lr = lr
        self.rng = rng
        self.pool_size = pool_size

    def combine_once(self, confidence_threshold=0.6, learn=True):
        all_keys = [k for k in self.internal.memory.keys() if k != SELF_KEY]
        if len(all_keys) < 3:
            return {"used": False}
        anchor_key = all_keys[self.rng.integers(len(all_keys))]
        anchor_vec = self.internal.memory[anchor_key]

        other_all = [k for k in all_keys if k != anchor_key]
        pool_n = min(self.pool_size, len(other_all))
        pool_idx = self.rng.choice(len(other_all), size=pool_n, replace=False)
        other_keys = [other_all[i] for i in pool_idx]
        Z = np.stack([self.internal.memory[k] for k in other_keys])

        q = anchor_vec @ self.W_q
        K = Z @ self.W_k
        V = Z @ self.W_v
        raw_scores = (K @ q) / np.sqrt(self.attn_dim)
        scores = raw_scores * self.temp
        weights = softmax(scores)
        combined = weights @ V
        x_imagined = self.internal.core.decode(combined)
        z_ext, probs = self.external.forward(x_imagined)
        label = int(np.argmax(probs))
        confidence = float(probs[label])
        top_partner = other_keys[int(np.argmax(weights))]

        if confidence >= confidence_threshold and learn:
            ext, core = self.external, self.internal.core
            onehot = np.zeros_like(probs)
            onehot[label] = 1.0
            dlogits = probs - onehot
            dz_ext = dlogits @ ext.W_cls.T
            dx = (dz_ext * (1 - z_ext ** 2)) @ ext.core.W_enc.T
            dx_pre = dx * x_imagined * (1 - x_imagined)
            dcombined = dx_pre @ core.W_dec.T
            dweights = V @ dcombined
            dV = np.outer(weights, dcombined)
            dscores = weights * (dweights - np.sum(weights * dweights))
            dtemp = float(np.dot(dscores, raw_scores))
            draw = dscores * self.temp
            dK = np.outer(draw, q) / np.sqrt(self.attn_dim)
            dq = (draw @ K) / np.sqrt(self.attn_dim)
            self.W_k -= self.lr * (Z.T @ dK)
            self.W_v -= self.lr * (Z.T @ dV)
            self.W_q -= self.lr * np.outer(anchor_vec, dq)
            self.temp = max(self.temp - self.lr * dtemp, 0.05)

        return {"confidence": confidence, "label": label, "anchor": anchor_key, "top_partner": top_partner}


def run_seed(X_train, y_train, X_val, y_val, seed):
    frozen_core = AutoencoderCore(X_train.shape[1], LATENT_DIM, seed=seed)
    internal = InternalImitationLearner(frozen_core)
    imitation_sample = X_train[:IMITATION_N]
    imitation_labels = y_train[:IMITATION_N]
    for x in imitation_sample:
        internal.observe(x)
    key_to_label = {internal._key(x): int(l) for x, l in zip(imitation_sample, imitation_labels)}

    ratchet = RatchetMemory(frozen_core)
    candidate_core = ratchet.hand_out_copy()
    candidate_external = ExternalLearner(candidate_core, 10, seed=seed, identity_dim=IDENTITY_DIM)
    rng = np.random.default_rng(seed + 300)
    n = len(X_train)
    for _epoch in range(EPOCHS):
        order = rng.permutation(n)
        for idx in order:
            candidate_external.train_step(X_train[idx], y_train[idx])

    # Group A: self-anchored, trained (shipped design)
    self_combiner = AttentionCombiner(internal, candidate_external, seed=seed, pool_size=20)
    self_labels = [self_combiner.combine_once(confidence_threshold=0.6)["label"] for _ in range(N_ROUNDS)]

    # Group B: random-anchor control, trained
    rand_combiner = RandomAnchorCombiner(internal, candidate_external, seed=seed, pool_size=20)
    rand_labels, rand_anchor_true = [], []
    for _ in range(N_ROUNDS):
        r = rand_combiner.combine_once(confidence_threshold=0.6)
        rand_labels.append(r["label"])
        rand_anchor_true.append(key_to_label.get(r["anchor"], -1))

    # Group C: self-anchored, NO reinforcement at all (fresh random weights)
    fresh_combiner = AttentionCombiner(internal, candidate_external, seed=seed, pool_size=20)
    no_learn_labels = [fresh_combiner.combine_once(confidence_threshold=0.6, learn=False)["label"]
                        for _ in range(N_ROUNDS)]

    real_preds = candidate_external.predict(imitation_sample)

    return {
        "seed": seed,
        "self_counts": np.bincount(self_labels, minlength=10),
        "rand_counts": np.bincount(rand_labels, minlength=10),
        "rand_anchor_true_counts": np.bincount([l for l in rand_anchor_true if l >= 0], minlength=10),
        "no_learn_counts": np.bincount(no_learn_labels, minlength=10),
        "real_image_pred_counts": np.bincount(real_preds, minlength=10),
    }


def main():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    for seed in SEEDS:
        r = run_seed(X_train, y_train, X_val, y_val, seed)
        print(f"[seed {r['seed']}]")
        print(f"  A self-anchored (trained):        unique={np.count_nonzero(r['self_counts'])}  {r['self_counts']}")
        print(f"  B random-anchor (trained, no self): unique={np.count_nonzero(r['rand_counts'])}  {r['rand_counts']}")
        print(f"     (true class of the random anchor itself, for reference: {r['rand_anchor_true_counts']})")
        print(f"  C self-anchored (NO reinforcement): unique={np.count_nonzero(r['no_learn_counts'])}  {r['no_learn_counts']}")
        print(f"  reference: same classifier on REAL (unblended) images:  {r['real_image_pred_counts']}")
        print()

    print("Reading guide: if 'self' as a fixed center were doing something meaningful, B should")
    print("look different from A. If the 200-round reinforcement loop were the cause, C should look")
    print("different from A. Neither difference should appear if the true cause is a classifier")
    print("out-of-distribution blind spot for blended/decoded inputs, unrelated to self or training.")


if __name__ == "__main__":
    main()
