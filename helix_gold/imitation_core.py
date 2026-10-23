"""Module A + Module C only -- no attention, no blending (Module B removed).

Findings 12-15 traced the mode-collapse bug precisely to Module B's attention
weighting (a fixed query's softmax-weighted blend of several stored
vectors), not to anything in Module A or C. Rather than keep testing the
whole pipeline with a known-broken piece attached, this module strips
Module B out entirely so the original research questions (does the
identity/context axis split work? does self stay untouched? what does the
inherited-vs-cold-start convergence curve actually look like?) can be
re-examined on a codebase that is known to work correctly. Module B will be
revisited separately, once its weighting scheme is actually fixed.

  Module A (InternalImitationLearner) -- word/object vectorization only. Its
    AutoencoderCore's weights are fixed at construction and NEVER updated
    again; only a private per-word vector is optimized against that frozen
    decoder, so no two words can ever interfere with each other. Also
    hard-codes a permanent "self" anchor (SELF_KEY) into memory at
    construction, before any input is ever observed: the all-zero vector,
    literally the origin of the latent coordinate system. It has no consumer
    yet (Module B is gone for now) but stays, since Module A's own identity
    guarantees don't depend on Module B existing.

  Module C (RatchetMemory + ExternalLearner) -- ordinary supervised learning
    on a clone of Module A's starting weights, with an identity/context axis
    split: the first `identity_dim` latent axes never receive a gradient
    once cloned, so a word's "this is still the same thing" coordinates stay
    exactly what Module A computed, while the rest of the axes adapt freely
    under supervised fine-tuning. Commits/rolls back the clone based on
    downstream accuracy against a frozen-encoder baseline.
"""
import numpy as np

SELF_KEY = "__self__"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def pairwise_distance_mean(Z):
    if len(Z) < 2:
        return float("nan")
    diffs = Z[:, None, :] - Z[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(len(Z), k=1)
    return float(np.mean(d[iu]))


def group_distance_mean(Za, Zb):
    diffs = Za[:, None, :] - Zb[None, :, :]
    return float(np.mean(np.linalg.norm(diffs, axis=-1)))


def _drift_stats(Z_original, Z_working):
    diffs = Z_working - Z_original
    drift = np.linalg.norm(diffs, axis=1)
    denom = np.linalg.norm(Z_original, axis=1) * np.linalg.norm(Z_working, axis=1) + 1e-12
    cosine = np.sum(Z_original * Z_working, axis=1) / denom
    return float(np.mean(drift)), float(np.mean(cosine))


def per_sample_drift(Z_original, Z_working, identity_dim=None):
    mean_drift, mean_cosine = _drift_stats(Z_original, Z_working)
    result = {"mean_drift": mean_drift, "mean_cosine": mean_cosine}
    if identity_dim:
        id_drift, id_cos = _drift_stats(Z_original[:, :identity_dim], Z_working[:, :identity_dim])
        ctx_drift, ctx_cos = _drift_stats(Z_original[:, identity_dim:], Z_working[:, identity_dim:])
        result.update({
            "identity_drift": id_drift, "identity_cosine": id_cos,
            "context_drift": ctx_drift, "context_cosine": ctx_cos,
        })
    return result


class AutoencoderCore:
    """A set of weights defining a fixed encode/decode mapping."""

    def __init__(self, in_dim, latent_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W_enc = rng.normal(0, 1 / np.sqrt(in_dim), (in_dim, latent_dim))
        self.b_enc = np.zeros(latent_dim)
        self.W_dec = rng.normal(0, 1 / np.sqrt(latent_dim), (latent_dim, in_dim))
        self.b_dec = np.zeros(in_dim)

    def encode(self, x):
        return np.tanh(x @ self.W_enc + self.b_enc)

    def decode(self, z):
        return sigmoid(z @ self.W_dec + self.b_dec)

    def clone(self):
        other = AutoencoderCore.__new__(AutoencoderCore)
        other.W_enc, other.b_enc = self.W_enc.copy(), self.b_enc.copy()
        other.W_dec, other.b_dec = self.W_dec.copy(), self.b_dec.copy()
        return other

    def load_from(self, other):
        self.W_enc, self.b_enc = other.W_enc.copy(), other.b_enc.copy()
        self.W_dec, self.b_dec = other.W_dec.copy(), other.b_dec.copy()


class InternalImitationLearner:
    """Module A: pure per-word vectorization. core is fixed forever; only a
    private vector per input is optimized against core's frozen decoder.
    Hard-codes the permanent "self" anchor (all-zero, the literal origin)
    into memory at construction, before observe() is ever called."""

    def __init__(self, core, lr=0.5, max_iters=200, tol=1e-3):
        self.core = core  # never mutated after this point, by anyone
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.memory = {}
        latent_dim = core.W_enc.shape[1]
        self.memory[SELF_KEY] = np.zeros(latent_dim)

    @staticmethod
    def _key(x):
        return hash(np.round(x, 3).tobytes())

    def observe(self, x):
        key = self._key(x)
        if key in self.memory:
            return self.memory[key], 0

        core = self.core
        z = core.encode(x).copy()
        used_iters = 0
        for i in range(1, self.max_iters + 1):
            used_iters = i
            x_hat = core.decode(z)
            err = x_hat - x
            loss = np.mean(err ** 2)
            if loss < self.tol:
                break
            d_dec_pre = err * x_hat * (1 - x_hat) * (2.0 / x.size)
            dz = d_dec_pre @ core.W_dec.T
            z -= self.lr * dz

        self.memory[key] = z
        return z, used_iters


class ExternalLearner:
    """Module C: ordinary supervised classifier on a clone of Module A's
    starting core. identity_dim > 0 protects the first `identity_dim` latent
    axes: once handed a core, those columns of W_enc/b_enc never receive a
    gradient, so x -> z[:identity_dim] stays exactly what Module A computed,
    forever. Only z[identity_dim:] adapts under fine-tuning."""

    def __init__(self, core, n_classes, lr=0.1, seed=1, identity_dim=0):
        self.core = core
        rng = np.random.default_rng(seed)
        latent_dim = core.W_enc.shape[1]
        self.W_cls = rng.normal(0, 1 / np.sqrt(latent_dim), (latent_dim, n_classes))
        self.b_cls = np.zeros(n_classes)
        self.lr = lr
        self.identity_dim = identity_dim

    def forward(self, x):
        z = np.tanh(x @ self.core.W_enc + self.core.b_enc)
        probs = softmax(z @ self.W_cls + self.b_cls)
        return z, probs

    def train_step(self, x, y_index, update_encoder=True):
        z, probs = self.forward(x)
        y_onehot = np.zeros_like(probs)
        y_onehot[y_index] = 1.0
        d_logits = probs - y_onehot

        grad_W_cls = np.outer(z, d_logits)
        grad_b_cls = d_logits
        self.W_cls -= self.lr * grad_W_cls
        self.b_cls -= self.lr * grad_b_cls

        if update_encoder:
            d_z = d_logits @ self.W_cls.T
            d_enc_pre = d_z * (1 - z ** 2)
            grad_W_enc = np.outer(x, d_enc_pre)
            grad_b_enc = d_enc_pre
            if self.identity_dim > 0:
                grad_W_enc[:, :self.identity_dim] = 0.0
                grad_b_enc[:self.identity_dim] = 0.0
            self.core.W_enc -= self.lr * grad_W_enc
            self.core.b_enc -= self.lr * grad_b_enc

        return -np.log(probs[y_index] + 1e-12)

    def predict(self, X):
        Z = np.tanh(X @ self.core.W_enc + self.core.b_enc)
        logits = Z @ self.W_cls + self.b_cls
        return np.argmax(logits, axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def snapshot(self):
        return {"core": self.core.clone(), "W_cls": self.W_cls.copy(), "b_cls": self.b_cls.copy()}

    def restore(self, snapshot):
        self.core.load_from(snapshot["core"])
        self.W_cls = snapshot["W_cls"].copy()
        self.b_cls = snapshot["b_cls"].copy()


class RatchetMemory:
    """Hands the external learner a clone of a starting core, then commits or
    rolls back that (separate) clone based on downstream accuracy. Never
    reaches back into Module A's original core or memory."""

    def __init__(self, core):
        self.backup = core.clone()

    def hand_out_copy(self):
        return self.backup.clone()

    def resolve(self, candidate_core, candidate_acc, baseline_acc):
        if candidate_acc > baseline_acc:
            self.backup.load_from(candidate_core)
            return "committed"
        return "rolled_back"
