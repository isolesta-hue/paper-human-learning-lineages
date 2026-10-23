"""Small numpy implementation of the internal-imitation / external-learning idea.

Four strictly separate pieces of state exist -- this is the key fix from the
first version, which let a shared core drift under everyone's feet:

  Module A (InternalImitationLearner) -- word/object vectorization only. Its
    AutoencoderCore's weights are fixed at construction and NEVER updated again;
    only a private per-word vector is optimized against that frozen decoder. So
    learning word #400 cannot disturb the vector already assigned to word #1
    (step 3), and nothing else may touch internal.memory or internal.core.

    Module A also hard-codes a permanent "self" anchor (SELF_KEY) into memory
    at construction time, before any input is ever observed: the all-zero
    vector, i.e. literally the origin of the latent coordinate system that
    every other vector is eventually understood relative to. It is not derived
    from an input, is never touched by observe(), and its key is a distinct
    sentinel string so nothing else can ever collide with or overwrite it.

  Module C (RatchetMemory + ExternalLearner) -- ordinary supervised learning.
    It works on its OWN cloned copy of Module A's starting weights (step 4-9),
    and commits/rolls back that copy based on downstream accuracy (step 10-12).
    Module A's original core is never involved in this clone/compare/rollback --
    it is read once, at hand-out time, and never again.

    Within that clone, Module C now respects an identity/context split instead
    of an all-or-nothing freeze: the first `identity_dim` latent axes are the
    "this is still the same remembered thing" coordinates and never receive a
    gradient once cloned from Module A, while the remaining axes are free to
    move under supervised fine-tuning. This replaces treating "fixed" and
    "mutable" as a property of a whole module with treating it as a property of
    a subset of axes within one vector -- a single word's meaning can be partly
    permanent identity and partly situational, adjustable content.

  Module B (AttentionCombiner) -- relation/combination logic (step 14). It reads
    Module A's stored vectors READ-ONLY (never writes to internal.memory or
    internal.core) and keeps its own separate parameters (q_self, W_k, W_v) and
    its own separate storage (self.associations) for whatever it produces. Every
    combination is anchored on "self" as the query -- i.e. every relation is
    computed as "how does X relate to me", not as a relation between two
    arbitrary memories. Feedback from the external learner's confidence nudges
    q_self/W_k/W_v only.

    Note on why the query is q_self, a free parameter, rather than
    `self_vector @ W_q`: self_vector is exactly all-zeros (see Module A above).
    Multiplying a zero vector by any matrix W_q gives zero regardless of W_q,
    which would make W_q's gradient identically zero too -- a dead parameter
    that looks like it's training but never moves. q_self plays the same role
    (self's own fixed "attention perspective") without that trap, while
    self_vector itself stays exactly the literal origin in memory.
"""
import numpy as np

# Sentinel key for the permanent "self" entry in Module A's memory. A string so
# it can never collide with the integer hash keys observe() generates from
# quantized input arrays.
SELF_KEY = "__self__"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def pairwise_distance_mean(Z):
    """GPT-script-style diagnostic: average Euclidean distance within a group of vectors."""
    if len(Z) < 2:
        return float("nan")
    diffs = Z[:, None, :] - Z[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(len(Z), k=1)
    return float(np.mean(d[iu]))


def group_distance_mean(Za, Zb):
    """Average Euclidean distance between two groups of vectors."""
    diffs = Za[:, None, :] - Zb[None, :, :]
    return float(np.mean(np.linalg.norm(diffs, axis=-1)))


def _drift_stats(Z_original, Z_working):
    diffs = Z_working - Z_original
    drift = np.linalg.norm(diffs, axis=1)
    denom = np.linalg.norm(Z_original, axis=1) * np.linalg.norm(Z_working, axis=1) + 1e-12
    cosine = np.sum(Z_original * Z_working, axis=1) / denom
    return float(np.mean(drift)), float(np.mean(cosine))


def per_sample_drift(Z_original, Z_working, identity_dim=None):
    """How far (and in what direction) each vector moved between two snapshots
    of the same inputs -- e.g. Module A's frozen encoding vs. the external
    learner's fine-tuned encoding of the same samples.

    When identity_dim is given, also splits the same measurement into the
    identity axes [:identity_dim] and the context axes [identity_dim:], so the
    hypothesis "identity axes stay put, context axes are free to drift" can be
    checked directly instead of only looking at the whole-vector average.
    """
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
    private vector per input is optimized against core's frozen decoder, so no
    two words can ever interfere with each other (step 3).

    Also hard-codes the permanent "self" anchor into memory at construction,
    before observe() is ever called on anything: the all-zero vector, the
    literal origin of the latent coordinate system.
    """

    def __init__(self, core, lr=0.5, max_iters=200, tol=1e-3):
        self.core = core  # never mutated after this point, by anyone
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.memory = {}  # quantized-input key -> converged latent vector (also frozen once set)

        # "Self" is fixed independently of any input, before anything is learned:
        # the origin (every axis exactly 0) of the latent coordinate system that
        # everything else is eventually understood relative to. Never derived
        # from an input, never touched by observe(). (AttentionCombiner uses its
        # own q_self parameter rather than a projection of this vector -- see the
        # note at the top of this module for why.)
        latent_dim = core.W_enc.shape[1]
        self.memory[SELF_KEY] = np.zeros(latent_dim)

    @staticmethod
    def _key(x):
        return hash(np.round(x, 3).tobytes())

    def observe(self, x):
        key = self._key(x)
        if key in self.memory:
            return self.memory[key], 0  # already imitated -> skip, per spec

        core = self.core
        z = core.encode(x).copy()  # warm-start guess only; core.W_enc is never touched again
        used_iters = 0
        for i in range(1, self.max_iters + 1):
            used_iters = i
            x_hat = core.decode(z)
            err = x_hat - x
            loss = np.mean(err ** 2)
            if loss < self.tol:
                break

            d_dec_pre = err * x_hat * (1 - x_hat) * (2.0 / x.size)
            dz = d_dec_pre @ core.W_dec.T  # gradient flows only into z; W_dec/b_dec are read-only here
            z -= self.lr * dz

        self.memory[key] = z
        return z, used_iters


class ExternalLearner:
    """Module C: ordinary supervised classifier. Its encoder starts from a core
    HANDED TO IT (a clone -- see RatchetMemory), and is free to reshape that
    clone via backprop; it never has access to Module A's original core (step 6-9).

    identity_dim > 0 protects the first `identity_dim` latent axes of the clone:
    once handed a core, those columns of W_enc/b_enc never receive a gradient, so
    the mapping x -> z[:identity_dim] stays exactly what Module A computed for
    every input, forever. Only z[identity_dim:] -- the "context" axes -- adapt
    under fine-tuning. With identity_dim=0 this is the original all-or-nothing
    behavior.
    """

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
                # Identity axes are read-only from here on: zero their gradient
                # instead of letting fine-tuning touch them.
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
        """Gemini-style checkpoint: capture enough state to restore exactly."""
        return {"core": self.core.clone(), "W_cls": self.W_cls.copy(), "b_cls": self.b_cls.copy()}

    def restore(self, snapshot):
        self.core.load_from(snapshot["core"])
        self.W_cls = snapshot["W_cls"].copy()
        self.b_cls = snapshot["b_cls"].copy()


class RatchetMemory:
    """Hands the external learner a clone of a starting core, then commits or
    rolls back that (separate) clone based on downstream accuracy (step 4-5, 10-12).
    Never reaches back into Module A's original core or memory."""

    def __init__(self, core):
        self.backup = core.clone()

    def hand_out_copy(self):
        return self.backup.clone()

    def resolve(self, candidate_core, candidate_acc, baseline_acc):
        if candidate_acc > baseline_acc:
            self.backup.load_from(candidate_core)
            return "committed"
        return "rolled_back"


class AttentionCombiner:
    """Module B: relation/combination logic (step 14), kept strictly separate
    from Module A. Uses "self" as the fixed query ("situation cue"), attends
    over the OTHER stored vectors -- everything that isn't self -- via
    dot-product attention, and decodes the resulting mixture through Module
    A's frozen decoder -- read-only use.

    Every combination is therefore self-centered by construction: it answers
    "how does this stored thing relate to me", not "how do two arbitrary stored
    things relate to each other". This mirrors the idea that recall/experience
    is organized around a stable self-reference point, not a floating pair of
    unrelated memories.

    The query itself is q_self, a free learnable parameter, NOT a projection of
    Module A's self_vector (which is exactly all-zeros -- see SELF_KEY). Any
    matrix times a zero vector is zero, so a `self_vector @ W_q` query would
    make W_q's gradient identically zero too: a dead parameter. q_self takes
    over that role directly. W_k/W_v are the "separate storage" for whatever
    relational structure it discovers; all three start random (so early
    combinations can be arbitrary/nonsensical, matching a baby's early babble)
    and are nudged by self-training on the external learner's confidence in
    what got decoded (steps: only confident combinations reinforce; nothing
    here ever writes to internal.memory, internal.core, or the external
    learner's own weights).

    pool_size matters more than it looks: with a fixed query (q_self) and the
    FULL memory set attended over every time, every call to combine_once() is
    a deterministic function of the current parameters -- there is nothing
    that varies from round to round except q_self/W_k/W_v themselves. That
    turns "200 dream rounds" into 200 gradient-ascent steps on one single,
    fixed combination (confirmed empirically: label and top_partner never
    changed across 200 rounds when pool_size covered the whole memory).
    Sampling a random subset of `pool_size` other memories each call restores
    the diversity that made the pre-self-anchor design's confidence actually
    climb over training (0.974->0.991) instead of just riding a fixed point up
    to its ceiling.

    pool_size alone wasn't enough, though: measured directly, the attention
    scores (K @ q / sqrt(attn_dim)) had a total spread of only ~0.29 across
    all 400 stored memories -- softmax over spreads that small is
    indistinguishable from uniform (empirically: max weight 0.024 vs 0.0025
    uniform, out of 400). With near-uniform weights, `combined` is just the
    sample mean of V over whichever subset got drawn, which -- by the law of
    large numbers -- lands in nearly the same place regardless of which
    specific memories are in the subset. That explained the deeper collapse:
    top_partner varied (whoever has the marginally highest, nearly-arbitrary
    score in a given draw) while the predicted label stayed fixed (the mean
    barely moves). Two changes address this directly:
      - score_scale widens the INITIAL spread by scaling up q_self/W_k's init
        (not W_v, which sets the "content" magnitude, not the score spread).
      - temp is a learnable scalar multiplier on the score, separate from the
        fixed 1/sqrt(attn_dim) normalization -- so training can sharpen (or
        soften) attention on its own instead of being stuck with whatever
        spread initialization happens to produce.

    IMPORTANT: this fixes the flat-attention problem (verified: attention
    genuinely differentiates now, e.g. mean max-weight ~0.74 at temp=10,
    versus ~0.05/uniform before) but does NOT fix predicted-label collapse --
    that turned out to have a separate, deeper cause. Measured directly: the
    fixed q_self converges to a genuine, persistent per-class bias in score
    space (e.g. one seed's q_self scored class 6 at a mean of +0.89 vs. the
    next-highest class at +0.54, across the whole 400-memory population).
    Sharpening attention (higher temp) just makes that ALREADY-BIASED
    preference literal (a dominant single winner from the preferred class)
    instead of blurring it into an ensemble average that still leans the same
    way -- it doesn't introduce diversity. This is a structural property of a
    single fixed query trained ONLY to maximize its own output confidence: a
    classic rich-get-richer loop with nothing pushing back toward exploring
    other classes. Left as an open question (see README) rather than solved
    here.
    """

    def __init__(self, internal: InternalImitationLearner, external: ExternalLearner,
                 attn_dim=None, lr=0.05, seed=3, pool_size=20, score_scale=3.0):
        self.internal = internal
        self.external = external
        latent_dim = internal.core.W_enc.shape[1]
        self.attn_dim = attn_dim or latent_dim
        rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(latent_dim)
        # q_self/W_k get a wider init (score_scale) than W_v: they set the
        # attention SCORE's dynamic range, which measured far too small at the
        # default 1/sqrt(latent_dim) scale (see class docstring). W_v just
        # sets the mixed-in content's magnitude and doesn't need the same fix.
        self.q_self = rng.normal(0, scale * score_scale, self.attn_dim)
        self.W_k = rng.normal(0, scale * score_scale, (latent_dim, self.attn_dim))
        self.W_v = rng.normal(0, scale, (latent_dim, latent_dim))
        self.temp = 1.0  # learnable multiplier on the score, on top of the
        # fixed 1/sqrt(attn_dim) normalization -- lets training sharpen or
        # soften attention itself rather than being stuck with init-time spread.
        self.lr = lr
        self.rng = rng
        self.pool_size = pool_size
        self.associations = {}  # Module B's own separate record of validated combinations
        self._counter = 0

    def combine_once(self, confidence_threshold=0.6, learn=True):
        all_keys = [k for k in self.internal.memory.keys() if k != SELF_KEY]
        if len(all_keys) < 2:
            return {"used": False, "reason": "not enough memory yet"}

        # A fresh random subset each call -- see class docstring for why this
        # (not just the fixed query) is what makes Module B's self-training
        # a real learning problem instead of optimizing one fixed point.
        pool_n = min(self.pool_size, len(all_keys))
        pool_idx = self.rng.choice(len(all_keys), size=pool_n, replace=False)
        other_keys = [all_keys[i] for i in pool_idx]

        Z = np.stack([self.internal.memory[k] for k in other_keys])  # read-only

        q = self.q_self  # self's own learnable query -- see class docstring for why
        K = Z @ self.W_k
        V = Z @ self.W_v
        raw_scores = (K @ q) / np.sqrt(self.attn_dim)
        scores = raw_scores * self.temp
        weights = softmax(scores)
        combined = weights @ V

        x_imagined = self.internal.core.decode(combined)  # frozen decoder, read-only use

        z_ext, probs = self.external.forward(x_imagined)
        label = int(np.argmax(probs))
        confidence = float(probs[label])
        top_partner = other_keys[int(np.argmax(weights))]

        record = {
            "used": confidence >= confidence_threshold,
            "confidence": confidence,
            "label": label,
            "anchor": SELF_KEY,
            "top_partner": top_partner,
        }

        if confidence >= confidence_threshold:
            self._counter += 1
            self.associations[self._counter] = {
                "anchor": SELF_KEY,
                "top_partner": top_partner,
                "combined_vector": combined,
                "label": label,
                "confidence": confidence,
            }
            if learn:
                self._reinforce(q, Z, weights, V, combined, x_imagined, z_ext, probs, label)

        return record

    def _reinforce(self, q, Z, weights, V, combined, x, z_ext, probs, label):
        """Self-training step: increase this combination's confidence in its own
        predicted label. Backprop reaches q_self/W_k/W_v/temp only -- Z (Module
        A's vectors), the frozen decoder, and the external learner's own weights
        are used purely to route the gradient and are never updated here."""
        ext = self.external
        core = self.internal.core

        onehot = np.zeros_like(probs)
        onehot[label] = 1.0
        dlogits = probs - onehot
        dz_ext = dlogits @ ext.W_cls.T
        dx = (dz_ext * (1 - z_ext ** 2)) @ ext.core.W_enc.T

        dx_pre = dx * x * (1 - x)
        dcombined = dx_pre @ core.W_dec.T

        dweights = V @ dcombined
        dV = np.outer(weights, dcombined)
        dscores = weights * (dweights - np.sum(weights * dweights))  # dL/dscores

        K = Z @ self.W_k
        raw_scores = (K @ q) / np.sqrt(self.attn_dim)
        dtemp = float(np.dot(dscores, raw_scores))  # scores = raw_scores * temp
        draw_scores = dscores * self.temp  # dL/draw_scores

        dK = np.outer(draw_scores, q) / np.sqrt(self.attn_dim)
        dq = (draw_scores @ K) / np.sqrt(self.attn_dim)  # q = q_self directly, so this IS dq_self

        dW_k = Z.T @ dK
        dW_v = Z.T @ dV

        self.W_k -= self.lr * dW_k
        self.W_v -= self.lr * dW_v
        self.q_self -= self.lr * dq
        # Keep temp strictly positive: at temp <= 0 the score->softmax mapping
        # stops being a sensible "how peaked" control (0 flattens forever, a
        # sign flip would invert which key wins).
        self.temp = max(self.temp - self.lr * dtemp, 0.05)
