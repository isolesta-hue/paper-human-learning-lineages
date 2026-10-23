"""Toy end-to-end run of internal imitation (Module A) -> ratchet + external
supervised learning (Module C) -> attention-based combination (Module B).

This version borrows three ideas spotted in independently-written ChatGPT/Gemini
attempts at the same design:
  - Gemini: per-epoch checkpoint + restore-to-best (a finer-grained ratchet than
    just comparing final accuracy once).
  - GPT: distance-drift diagnostics (are relationships between concepts distorted
    by external fine-tuning?) and multi-seed statistics instead of a single run.

It also tests two follow-up hypotheses about what "fixed" should mean:
  - IDENTITY_DIM: instead of freezing Module C's encoder wholesale, only the
    first IDENTITY_DIM latent axes are protected from its gradient. If that's
    a good split, the identity-axis cosine should stay near 1.0 across epochs
    while the context-axis cosine is free to (and expected to) drift.
  - The "self" anchor (imitation_mlp.SELF_KEY): Module B's attention is always
    anchored on a permanent, never-learned self vector rather than a randomly
    picked memory, and we check it never moves across the whole run.

IMPORTANT framing: final accuracy is not the thing to read off these runs.
With a supervised feedback loop running for enough epochs, final accuracy
converges regardless of what's inherited from Module A -- that's expected and
not informative on its own. What actually distinguishes "inherited a useful
starting point" from "started from scratch" is the SHAPE of the accuracy
curve: how steep the early climb is, how many epochs it takes to capture most
of the total improvement, and where the climb flattens out. curve_metrics()
below exists specifically to make that comparison instead of comparing final
numbers.

Dataset: sklearn's 8x8 digit images (small, fast, nothing to download).
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
    pairwise_distance_mean,
    group_distance_mean,
    per_sample_drift,
)

LATENT_DIM = 16
IDENTITY_DIM = 3  # first 3 of 16 axes: permanent identity, never fine-tuned. The
# identity_dim sweep (identity_sweep.py) showed convergence speed degrades
# roughly monotonically with identity_dim, with a steep knee around 8-10/16 --
# 3/16 keeps that capacity cost small while still protecting a real identity.
EPOCHS = 15
IMITATION_N = 400
N_DREAM_ROUNDS = 200
SEEDS = [0, 1, 2, 3, 4]


def train_ratcheted(external, X_train, y_train, X_val, y_val, epochs,
                     update_encoder=True, rng=None, drift_track=None, checks_per_epoch=8):
    """Per-epoch ratchet: keep the best-val-accuracy snapshot seen during
    training and restore it at the end, instead of trusting whatever the last
    epoch happens to land on. Always returns the FULL history at fine-grained
    step resolution (not just the final number, and not just one point per
    epoch), because the history's shape -- not its endpoint -- is what we
    actually care about here.

    checks_per_epoch controls resolution: with the default of 8, epoch 1 alone
    gets 8 checkpoints. This matters because in this setup almost all of the
    improvement happens WITHIN epoch 1 -- one checkpoint per epoch is too
    coarse to see the shape of that climb at all, let alone compare it between
    conditions.

    drift_track, if given, is (imitation_sample, Z_original, identity_dim):
    also records identity/context-axis cosine vs. Module A's memory at every
    checkpoint, for runs where that comparison makes sense (i.e. not for a
    cold-start encoder that never had a Module A vector to inherit).
    """
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
    for _epoch in range(1, epochs + 1):
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
    """Characterizes the SHAPE of an accuracy curve rather than its endpoint:
    how much of the total improvement happens in the very first checkpoint
    (first_jump), how steep the early vs. late climb is (mean_early_slope/
    mean_late_slope, in accuracy points per checkpoint), and how many
    checkpoints it takes to capture `gain_fraction` of the total improvement
    over the run (checks_to_90pct_gain).

    Uses the running max of val_acc (not the raw per-checkpoint value) since
    the ratchet itself only ever keeps the best snapshot seen -- a dip at a
    later checkpoint isn't a real regression in what gets kept, so it
    shouldn't count as one in the slope either.
    """
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
        "steps_to_90pct_gain": steps[idx],
    }


def run_pipeline(seed, X_train, X_val, y_train, y_val, n_classes):
    cold_core = AutoencoderCore(X_train.shape[1], LATENT_DIM, seed=seed)
    cold_external = ExternalLearner(cold_core, n_classes, seed=seed)
    cold_acc, cold_history = train_ratcheted(cold_external, X_train, y_train, X_val, y_val, EPOCHS,
                                              rng=np.random.default_rng(seed + 100))

    # --- Module A: frozen core, per-word vectors only ---
    frozen_core = AutoencoderCore(X_train.shape[1], LATENT_DIM, seed=seed)
    internal = InternalImitationLearner(frozen_core)
    self_vector_before = internal.memory[SELF_KEY].copy()
    imitation_sample = X_train[:IMITATION_N]
    imitation_labels = y_train[:IMITATION_N]
    for x in imitation_sample:
        internal.observe(x)

    ratchet = RatchetMemory(frozen_core)
    Z_original = np.stack([internal.memory[internal._key(x)] for x in imitation_sample])

    baseline_core = ratchet.hand_out_copy()
    baseline_external = ExternalLearner(baseline_core, n_classes, seed=seed)
    baseline_acc, baseline_history = train_ratcheted(baseline_external, X_train, y_train, X_val, y_val,
                                                       EPOCHS, update_encoder=False,
                                                       rng=np.random.default_rng(seed + 200))

    candidate_core = ratchet.hand_out_copy()
    candidate_external = ExternalLearner(candidate_core, n_classes, seed=seed,
                                          identity_dim=IDENTITY_DIM)
    candidate_acc, candidate_history = train_ratcheted(
        candidate_external, X_train, y_train, X_val, y_val, EPOCHS,
        rng=np.random.default_rng(seed + 300),
        drift_track=(imitation_sample, Z_original, IDENTITY_DIM))

    decision = ratchet.resolve(candidate_external.core, candidate_acc, baseline_acc)

    # --- distance-drift diagnostics: digits 0-4 ("low") vs 5-9 ("high") groups ---
    Z_working = np.stack([candidate_external.core.encode(x) for x in imitation_sample])
    low_mask = imitation_labels < 5
    high_mask = ~low_mask

    space_before = {
        "low_low": pairwise_distance_mean(Z_original[low_mask]),
        "high_high": pairwise_distance_mean(Z_original[high_mask]),
        "low_high": group_distance_mean(Z_original[low_mask], Z_original[high_mask]),
    }
    space_after = {
        "low_low": pairwise_distance_mean(Z_working[low_mask]),
        "high_high": pairwise_distance_mean(Z_working[high_mask]),
        "low_high": group_distance_mean(Z_working[low_mask], Z_working[high_mask]),
    }
    drift = per_sample_drift(Z_original, Z_working, identity_dim=IDENTITY_DIM)

    # --- Module B: attention combiner, self-anchored, read-only over Module A's memory ---
    combiner = AttentionCombiner(internal, candidate_external, seed=seed)
    confidences = []
    for _ in range(N_DREAM_ROUNDS):
        result = combiner.combine_once(confidence_threshold=0.6)
        if "confidence" in result:
            confidences.append(result["confidence"])
    half = N_DREAM_ROUNDS // 2
    conf_first, conf_second = confidences[:half], confidences[half:]

    # --- integrity checks ---
    memory_untouched = all(
        np.array_equal(internal.memory[internal._key(x)], Z_original[i])
        for i, x in enumerate(imitation_sample)
    )
    self_untouched = np.array_equal(internal.memory[SELF_KEY], self_vector_before)
    identity_weights_untouched = (
        np.array_equal(frozen_core.W_enc[:, :IDENTITY_DIM], candidate_external.core.W_enc[:, :IDENTITY_DIM])
        and np.array_equal(frozen_core.b_enc[:IDENTITY_DIM], candidate_external.core.b_enc[:IDENTITY_DIM])
    )

    return {
        "seed": seed,
        "cold_acc": cold_acc,
        "baseline_acc": baseline_acc,
        "candidate_acc": candidate_acc,
        "decision": decision,
        "space_before": space_before,
        "space_after": space_after,
        "drift_mean": drift["mean_drift"],
        "cosine_mean": drift["mean_cosine"],
        "identity_cosine": drift["identity_cosine"],
        "context_cosine": drift["context_cosine"],
        "conf_first_mean": float(np.mean(conf_first)),
        "conf_second_mean": float(np.mean(conf_second)),
        "conf_first_std": float(np.std(conf_first)),
        "conf_second_std": float(np.std(conf_second)),
        "memory_untouched": memory_untouched,
        "self_untouched": self_untouched,
        "identity_weights_untouched": identity_weights_untouched,
        "cold_history": cold_history,
        "candidate_history": candidate_history,
        "cold_curve": curve_metrics(cold_history),
        "candidate_curve": curve_metrics(candidate_history),
    }


def print_curve_comparison(results, label_a, key_a, label_b, key_b):
    """Prints mean+/-std of curve_metrics for two conditions side by side --
    this is the "angle/count/rate, not endpoint" comparison the final-accuracy
    numbers above deliberately don't make."""
    def agg_curve(curve_key, metric):
        vals = [r[curve_key][metric] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    metrics = ["first_jump", "mean_early_slope", "mean_late_slope", "checks_to_90pct_gain", "total_gain"]
    print(f"{'metric':>22} {label_a:>18} {label_b:>18}")
    for m in metrics:
        am, asd = agg_curve(key_a, m)
        bm, bsd = agg_curve(key_b, m)
        print(f"{m:>22} {am:>10.3f}+/-{asd:<5.3f} {bm:>10.3f}+/-{bsd:<5.3f}")


def main():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    n_classes = 10

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    results = []
    for seed in SEEDS:
        r = run_pipeline(seed, X_train, X_val, y_train, y_val, n_classes)
        results.append(r)
        print(f"[seed {seed}] cold={r['cold_acc']:.3f} baseline={r['baseline_acc']:.3f} "
              f"candidate={r['candidate_acc']:.3f} decision={r['decision']:<11} "
              f"memory_untouched={r['memory_untouched']} self_untouched={r['self_untouched']}")

    def agg(key):
        vals = [r[key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    print()
    print(f"=== multi-seed summary (n={len(SEEDS)}) -- FINAL accuracy, for reference only ===")
    for key in ["cold_acc", "baseline_acc", "candidate_acc"]:
        m, s = agg(key)
        print(f"{key}: {m:.3f} +/- {s:.3f}")
    commit_rate = np.mean([r["decision"] == "committed" for r in results])
    print(f"ratchet commit rate: {commit_rate:.0%}")
    print("(final numbers converge with enough epochs regardless of what's inherited --")
    print(" see the curve comparison below for what actually differs.)")

    print()
    print("=== accuracy CURVE shape: cold-start vs. candidate (inherits Module A) ===")
    print_curve_comparison(results, "cold-start", "cold_curve", "candidate", "candidate_curve")

    print()
    print("=== distance-structure drift (digits 0-4 vs 5-9), mean over seeds ===")
    for group_key, label in [("low_low", "within 0-4"), ("high_high", "within 5-9"),
                              ("low_high", "0-4 <-> 5-9")]:
        before = np.mean([r["space_before"][group_key] for r in results])
        after = np.mean([r["space_after"][group_key] for r in results])
        print(f"{label:12}: before={before:.3f} after={after:.3f} delta={after - before:+.3f}")

    dm, ds = agg("drift_mean")
    cm, cs = agg("cosine_mean")
    print(f"per-sample drift (L2 norm), whole vector: {dm:.3f} +/- {ds:.3f}")
    print(f"per-sample cosine, whole vector:          {cm:.3f} +/- {cs:.3f}")

    print()
    print(f"=== identity axes [:{ IDENTITY_DIM }] vs context axes [{ IDENTITY_DIM }:], mean over seeds ===")
    idm, ids = agg("identity_cosine")
    ctm, cts = agg("context_cosine")
    print(f"identity-axis cosine (vs Module A's own refined memory): {idm:.3f} +/- {ids:.3f}")
    print(f"context-axis cosine  (vs Module A's own refined memory): {ctm:.3f} +/- {cts:.3f}")
    id_weights_rate = np.mean([r["identity_weights_untouched"] for r in results])
    print(f"identity-axis ENCODER WEIGHTS byte-identical to Module A's core: {id_weights_rate:.0%} of seeds")

    print()
    print("=== Module B confidence convergence, mean over seeds (self-anchored) ===")
    f_m, f_s = agg("conf_first_mean")
    s_m, s_s = agg("conf_second_mean")
    within_first = np.mean([r["conf_first_std"] for r in results])
    within_second = np.mean([r["conf_second_std"] for r in results])
    print(f"first half : mean={f_m:.3f} +/- {f_s:.3f} (avg within-seed std {within_first:.3f})")
    print(f"second half: mean={s_m:.3f} +/- {s_s:.3f} (avg within-seed std {within_second:.3f})")

    untouched_rate = np.mean([r["memory_untouched"] for r in results])
    self_rate = np.mean([r["self_untouched"] for r in results])
    print()
    print(f"Module A memory integrity check passed in {untouched_rate:.0%} of seeds")
    print(f"Self-anchor integrity check passed in {self_rate:.0%} of seeds")

    print()
    print("=== fine-grained trajectory (first 2 epochs' worth of checkpoints): ===")
    print("=== cold vs candidate val_acc, candidate identity/context cosine        ===")
    print(f"{'step':>6} {'cold_acc':>9} {'cand_acc':>9} {'id_cos':>8} {'ctx_cos':>8}")
    n_show = min(17, len(results[0]["candidate_history"]))  # ~2 epochs at checks_per_epoch=8
    for i in range(n_show):
        step = results[0]["candidate_history"][i]["step"]
        cold_acc = np.mean([r["cold_history"][i]["val_acc"] for r in results])
        cand_acc = np.mean([r["candidate_history"][i]["val_acc"] for r in results])
        id_cos = np.mean([r["candidate_history"][i]["identity_cosine"] for r in results])
        ctx_cos = np.mean([r["candidate_history"][i]["context_cosine"] for r in results])
        print(f"{step:>6} {cold_acc:>9.3f} {cand_acc:>9.3f} {id_cos:>8.3f} {ctx_cos:>8.3f}")


if __name__ == "__main__":
    main()
