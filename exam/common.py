"""Seeds and the common reference feature rule (EXAM_DESIGN.md 2: object features = random by object id)."""
import hashlib

import numpy as np


def seed_of(seed):
    """an int seed as it is; a seed phrase (str) -> the first 8 bytes of its SHA-256."""
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")


def rng(seed, *salt):
    """an independent random stream for (seed, salt...): every part of the generator draws from its own stream."""
    words = [seed_of(seed) & 0xFFFFFFFF, seed_of(seed) >> 32]
    for s in salt:
        words.append(seed_of(str(s)) & 0xFFFFFFFF)
    return np.random.default_rng(np.random.SeedSequence(words))


def feature(name, dim=40):
    """the common reference feature vector of an id or a word: uniform(0, 1) seeded by the string."""
    return np.random.default_rng(seed_of("feature:" + name) & 0xFFFFFFFFFFFF).uniform(0, 1, dim)


def feature_of_kind(kind, name, dim=40, spread=0.15):
    """an object of a kind: the kind's vector plus a small id-seeded difference (things of one kind look alike)."""
    v = feature("kind:" + kind, dim) + np.random.default_rng(seed_of("inst:" + name) & 0xFFFFFFFFFFFF).normal(0, spread, dim)
    return np.clip(v, 0, 1)
