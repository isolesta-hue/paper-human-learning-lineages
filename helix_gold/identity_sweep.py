"""Sweep identity_dim from 0 (fully mutable, old all-free design) to LATENT_DIM
(fully frozen, old all-fixed / baseline design) and see how the CURVE SHAPE of
candidate training changes -- not final accuracy, which converges regardless.

Motivation: run_experiment.py showed candidate (identity_dim=8) converges
measurably SLOWER than cold-start (checks_to_90pct_gain ~6.2 vs ~4.8, smaller
first_jump). The natural explanation is that freezing identity_dim axes simply
removes that many trainable degrees of freedom from the encoder, independent
of whatever Module A actually put there. This sweep checks whether convergence
speed degrades roughly monotonically with identity_dim (consistent with "it's
just fewer free parameters") or does something more interesting (e.g. a sweet
spot, or a cliff at a particular fraction).

Reuses Module A's setup once per seed (identity_dim doesn't affect what Module
A stores) and re-clones a fresh candidate encoder from the ratchet for each
identity_dim value, so all identity_dim conditions for a given seed start from
byte-identical initial weights.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from imitation_mlp import (
    AutoencoderCore,
    InternalImitationLearner,
    ExternalLearner,
    RatchetMemory,
)
from run_experiment import train_ratcheted, curve_metrics

LATENT_DIM = 16
IDENTITY_DIMS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]  # finer near the new default (3)
EPOCHS = 15
IMITATION_N = 400
SEEDS = [0, 1, 2, 3, 4]


def main():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    n_classes = 10

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # {identity_dim: [curve_metrics dict per seed]}
    by_dim = {d: [] for d in IDENTITY_DIMS}

    for seed in SEEDS:
        frozen_core = AutoencoderCore(X_train.shape[1], LATENT_DIM, seed=seed)
        internal = InternalImitationLearner(frozen_core)
        imitation_sample = X_train[:IMITATION_N]
        for x in imitation_sample:
            internal.observe(x)
        ratchet = RatchetMemory(frozen_core)
        Z_original = np.stack([internal.memory[internal._key(x)] for x in imitation_sample])

        for identity_dim in IDENTITY_DIMS:
            candidate_core = ratchet.hand_out_copy()
            candidate_external = ExternalLearner(candidate_core, n_classes, seed=seed,
                                                  identity_dim=identity_dim)
            _, history = train_ratcheted(
                candidate_external, X_train, y_train, X_val, y_val, EPOCHS,
                rng=np.random.default_rng(seed + 300),
                drift_track=(imitation_sample, Z_original, identity_dim) if identity_dim else None)
            cm = curve_metrics(history)
            cm["final_acc"] = history[-1]["val_acc"]
            by_dim[identity_dim].append(cm)
        print(f"[seed {seed}] done")

    def agg(dim, metric):
        vals = [cm[metric] for cm in by_dim[dim]]
        return float(np.mean(vals)), float(np.std(vals))

    print()
    print(f"=== identity_dim sweep (n={len(SEEDS)} seeds each), CURVE SHAPE not final accuracy ===")
    header = f"{'identity_dim':>12} {'first_jump':>16} {'early_slope':>16} {'late_slope':>16} {'checks_to_90%':>16} {'final_acc(ref)':>16}"
    print(header)
    for d in IDENTITY_DIMS:
        fj_m, fj_s = agg(d, "first_jump")
        es_m, es_s = agg(d, "mean_early_slope")
        ls_m, ls_s = agg(d, "mean_late_slope")
        ck_m, ck_s = agg(d, "checks_to_90pct_gain")
        fa_m, fa_s = agg(d, "final_acc")
        print(f"{d:>12} {fj_m:>7.3f}+/-{fj_s:<5.3f} {es_m:>7.3f}+/-{es_s:<5.3f} "
              f"{ls_m:>7.4f}+/-{ls_s:<5.4f} {ck_m:>7.2f}+/-{ck_s:<5.2f} {fa_m:>7.3f}+/-{fa_s:<5.3f}")

    print()
    print("Reading guide: first_jump and mean_early_slope should be HIGHER (faster climb)")
    print("for a good identity_dim; checks_to_90pct_gain should be LOWER (fewer checks needed).")
    print("final_acc is printed for reference only -- it is expected to be roughly flat/noisy")
    print("across identity_dim, since enough epochs let the classifier converge regardless.")


if __name__ == "__main__":
    main()
