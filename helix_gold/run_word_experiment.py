"""Re-run the project's original research plan -- does the identity/context
axis split actually protect identity while letting context adapt? does self
stay untouched? what does the inherited-vs-cold-start convergence curve
really look like? -- on a codebase with Module B (attention) removed
entirely, and on text/word data instead of digit images.

Why: findings 12-15 traced the mode-collapse bug precisely to Module B's
attention weighting (imitation_core.py has no such thing -- only Module A's
vectorization and Module C's ratcheted supervised learning). Carrying a
known-broken piece into further tests would just confound the results, so
Module B is set aside for now and revisited separately once its weighting
scheme is fixed on its own. Per project decision (2026-09-22): text/word
data only for now; images return once Module A/C is confirmed solid here.

This mirrors run_experiment.py's structure (train_ratcheted, curve_metrics,
cold/baseline/candidate comparison, identity/context drift, integrity
checks) but on a synthetic word vocabulary instead of sklearn digits, and
with no numpy-only constraint (still implemented in numpy -- nothing here
needs anything heavier).
"""
import numpy as np

from imitation_core import (
    SELF_KEY,
    AutoencoderCore,
    InternalImitationLearner,
    ExternalLearner,
    RatchetMemory,
    pairwise_distance_mean,
    group_distance_mean,
    per_sample_drift,
)

RAW_DIM = 40
LATENT_DIM = 16
IDENTITY_DIM = 3
CATEGORIES = ["animal", "vehicle", "food", "color", "emotion", "weather"]
N_CLASSES = len(CATEGORIES)
WORDS_PER_CATEGORY = 90  # 540 words total (6x the original 15/90) -- see
# note below on why the first run's curve metrics were noisy and why more
# words (not more epochs) is the right lever to fix it.
EPOCHS = 20
SEEDS = [0, 1, 2, 3, 4]

# Why scale up the VOCABULARY (not epochs, not checks_per_epoch): val_acc is
# measured on a held-out set of n_val samples, so it can only take values in
# increments of 1/n_val. At the old size (90 words -> n_val=18), one flipped
# prediction moves val_acc by 1/18 =~ 5.6%. checks_to_90pct_gain in
# particular reads off exactly WHICH checkpoint first crosses a threshold,
# so a single sample flipping earlier/later by chance can shift that index
# by several steps -- that's quantization noise in the metric itself, not
# a real difference in learning dynamics, and it doesn't average out across
# epochs, only across a bigger val set. At 540 words -> n_val=108, the step
# shrinks to 1/108 =~ 0.9%, a ~6x finer accuracy readout, which is what
# should make first_jump/mean_early_slope/checks_to_90pct_gain track the
# actual curve shape instead of sampling noise.


WORD_NOISE_STD = 0.2  # first run used 0.05, which was too easy: every condition
# (cold/baseline/candidate) hit exactly 1.000 accuracy, so the ratchet's
# accuracy comparison had nothing to discriminate on (a tie always rolls
# back). 0.2 puts the noise's overall scale (std * sqrt(RAW_DIM) =~ 1.26) in
# the same range as the ~1.16 minimum distance between category centers, so
# categories genuinely overlap a little instead of being trivially separable.


def build_word_vocab(seed):
    """Each category gets a fixed random "meaning center" in R^RAW_DIM; each
    word = center + noise, clipped to [0, 1] to match the sigmoid decoder's
    range (same convention as digits.data / 16.0 in the earlier image
    experiments)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, (N_CLASSES, RAW_DIM))
    X, y = [], []
    for c in range(N_CLASSES):
        for _ in range(WORDS_PER_CATEGORY):
            X.append(np.clip(centers[c] + rng.normal(0, WORD_NOISE_STD, RAW_DIM), 0, 1))
            y.append(c)
    X, y = np.array(X), np.array(y)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def train_ratcheted(external, X_train, y_train, X_val, y_val, epochs,
                     update_encoder=True, rng=None, drift_track=None, checks_per_epoch=8):
    """Same ratchet as run_experiment.py: keep the best-val-accuracy snapshot
    seen during training, and record fine-grained (sub-epoch) history so the
    curve's SHAPE -- not just its endpoint -- can be read off."""
    rng = rng or np.random.default_rng(123)
    n = len(X_train)
    check_every = max(1, n // checks_per_epoch)

    def current_drift():
        if drift_track is None:
            return {}
        imitation_sample, Z_original, identity_dim = drift_track
        Z_now = np.stack([external.core.encode(x) for x in imitation_sample])
        return per_sample_drift(Z_original, Z_now, identity_dim=identity_dim)

    best_acc = external.accuracy(X_val, y_val)
    best_snapshot = external.snapshot()
    history = [{"step": 0, "val_acc": best_acc, **current_drift()}]

    step = 0
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for idx in order:
            external.train_step(X_train[idx], y_train[idx], update_encoder=update_encoder)
            step += 1
            if step % check_every == 0:
                acc = external.accuracy(X_val, y_val)
                history.append({"step": step, "val_acc": acc, **current_drift()})
                if acc > best_acc:
                    best_acc = acc
                    best_snapshot = external.snapshot()

    external.restore(best_snapshot)
    return best_acc, history


def curve_metrics(history, key="val_acc", early_checks=3, gain_fraction=0.9):
    """Same shape-not-endpoint metrics as run_experiment.py."""
    vals = np.array([h[key] for h in history])
    steps = [h["step"] for h in history]
    running_max = np.maximum.accumulate(vals)
    deltas = np.diff(running_max)

    start, final = running_max[0], running_max[-1]
    total_gain = float(final - start)
    target = start + gain_fraction * total_gain
    idx = next((i for i, v in enumerate(running_max) if v >= target), len(running_max) - 1)

    early = deltas[:early_checks]
    late = deltas[early_checks:]
    return {
        "total_gain": total_gain,
        "first_jump": float(deltas[0]) if len(deltas) else float("nan"),
        "mean_early_slope": float(np.mean(early)) if len(early) else float("nan"),
        "mean_late_slope": float(np.mean(late)) if len(late) else float("nan"),
        "checks_to_90pct_gain": idx,
    }


def run_pipeline(seed):
    X, y = build_word_vocab(seed)
    n_val = max(12, len(X) // 5)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    # --- cold-start: fresh random core, no Module A involvement at all ---
    cold_core = AutoencoderCore(RAW_DIM, LATENT_DIM, seed=seed)
    cold_external = ExternalLearner(cold_core, N_CLASSES, seed=seed)
    cold_acc, cold_history = train_ratcheted(cold_external, X_train, y_train, X_val, y_val, EPOCHS,
                                              rng=np.random.default_rng(seed + 100))

    # --- Module A: frozen core, observes the WHOLE vocabulary ---
    frozen_core = AutoencoderCore(RAW_DIM, LATENT_DIM, seed=seed)
    internal = InternalImitationLearner(frozen_core)
    self_vector_before = internal.memory[SELF_KEY].copy()
    for x in X:
        internal.observe(x)

    ratchet = RatchetMemory(frozen_core)
    Z_original = np.stack([internal.memory[internal._key(x)] for x in X])

    baseline_core = ratchet.hand_out_copy()
    baseline_external = ExternalLearner(baseline_core, N_CLASSES, seed=seed)
    baseline_acc, baseline_history = train_ratcheted(baseline_external, X_train, y_train, X_val, y_val,
                                                       EPOCHS, update_encoder=False,
                                                       rng=np.random.default_rng(seed + 200))

    candidate_core = ratchet.hand_out_copy()
    candidate_external = ExternalLearner(candidate_core, N_CLASSES, seed=seed, identity_dim=IDENTITY_DIM)
    candidate_acc, candidate_history = train_ratcheted(
        candidate_external, X_train, y_train, X_val, y_val, EPOCHS,
        rng=np.random.default_rng(seed + 300),
        drift_track=(X, Z_original, IDENTITY_DIM))

    decision = ratchet.resolve(candidate_external.core, candidate_acc, baseline_acc)

    # --- distance-structure diagnostics: does fine-tuning distort relationships between categories? ---
    Z_working = np.stack([candidate_external.core.encode(x) for x in X])
    space_before, space_after = {}, {}
    for c in range(N_CLASSES):
        mask = y == c
        space_before[f"within_{c}"] = pairwise_distance_mean(Z_original[mask])
        space_after[f"within_{c}"] = pairwise_distance_mean(Z_working[mask])
    space_before["between_0_1"] = group_distance_mean(Z_original[y == 0], Z_original[y == 1])
    space_after["between_0_1"] = group_distance_mean(Z_working[y == 0], Z_working[y == 1])

    drift = per_sample_drift(Z_original, Z_working, identity_dim=IDENTITY_DIM)

    # --- integrity checks ---
    memory_untouched = all(
        np.array_equal(internal.memory[internal._key(x)], Z_original[i])
        for i, x in enumerate(X)
    )
    self_untouched = np.array_equal(internal.memory[SELF_KEY], self_vector_before)
    identity_weights_untouched = (
        np.array_equal(frozen_core.W_enc[:, :IDENTITY_DIM], candidate_external.core.W_enc[:, :IDENTITY_DIM])
        and np.array_equal(frozen_core.b_enc[:IDENTITY_DIM], candidate_external.core.b_enc[:IDENTITY_DIM])
    )

    return {
        "seed": seed,
        "cold_acc": cold_acc, "baseline_acc": baseline_acc, "candidate_acc": candidate_acc,
        "decision": decision,
        "space_before": space_before, "space_after": space_after,
        "identity_cosine": drift["identity_cosine"], "context_cosine": drift["context_cosine"],
        "memory_untouched": memory_untouched, "self_untouched": self_untouched,
        "identity_weights_untouched": identity_weights_untouched,
        "cold_curve": curve_metrics(cold_history), "candidate_curve": curve_metrics(candidate_history),
    }


def print_curve_comparison(results, label_a, key_a, label_b, key_b):
    def agg(curve_key, metric):
        vals = [r[curve_key][metric] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    metrics = ["first_jump", "mean_early_slope", "mean_late_slope", "checks_to_90pct_gain", "total_gain"]
    print(f"{'metric':>22} {label_a:>18} {label_b:>18}")
    for m in metrics:
        am, asd = agg(key_a, m)
        bm, bsd = agg(key_b, m)
        print(f"{m:>22} {am:>10.3f}+/-{asd:<5.3f} {bm:>10.3f}+/-{bsd:<5.3f}")


def main():
    results = [run_pipeline(seed) for seed in SEEDS]

    for r in results:
        print(f"[seed {r['seed']}] cold={r['cold_acc']:.3f} baseline={r['baseline_acc']:.3f} "
              f"candidate={r['candidate_acc']:.3f} decision={r['decision']:<11} "
              f"memory_untouched={r['memory_untouched']} self_untouched={r['self_untouched']}")

    def agg(key):
        vals = [r[key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    print()
    print("=== final accuracy, for reference only (converges regardless -- see curves below) ===")
    for key in ["cold_acc", "baseline_acc", "candidate_acc"]:
        m, s = agg(key)
        print(f"{key}: {m:.3f} +/- {s:.3f}")

    print()
    print("=== accuracy CURVE shape: cold-start vs. candidate (inherits Module A) ===")
    print_curve_comparison(results, "cold-start", "cold_curve", "candidate", "candidate_curve")

    print()
    print(f"=== identity axes [:{IDENTITY_DIM}] vs context axes [{IDENTITY_DIM}:] ===")
    idm, ids = agg("identity_cosine")
    ctm, cts = agg("context_cosine")
    print(f"identity-axis cosine (vs Module A's own refined memory): {idm:.3f} +/- {ids:.3f}")
    print(f"context-axis cosine  (vs Module A's own refined memory): {ctm:.3f} +/- {cts:.3f}")
    id_weights_rate = np.mean([r["identity_weights_untouched"] for r in results])
    print(f"identity-axis ENCODER WEIGHTS byte-identical to Module A's core: {id_weights_rate:.0%} of seeds")

    untouched_rate = np.mean([r["memory_untouched"] for r in results])
    self_rate = np.mean([r["self_untouched"] for r in results])
    print()
    print(f"Module A memory integrity check passed in {untouched_rate:.0%} of seeds")
    print(f"Self-anchor integrity check passed in {self_rate:.0%} of seeds")

    print()
    print("=== category distance structure, mean over seeds (within-category spread) ===")
    for c in range(N_CLASSES):
        before = np.mean([r["space_before"][f"within_{c}"] for r in results])
        after = np.mean([r["space_after"][f"within_{c}"] for r in results])
        print(f"{CATEGORIES[c]:10}: before={before:.3f} after={after:.3f} delta={after - before:+.3f}")
    b0 = np.mean([r["space_before"]["between_0_1"] for r in results])
    a0 = np.mean([r["space_after"]["between_0_1"] for r in results])
    print(f"{'animal<->vehicle':10}: before={b0:.3f} after={a0:.3f} delta={a0 - b0:+.3f}")


if __name__ == "__main__":
    main()
