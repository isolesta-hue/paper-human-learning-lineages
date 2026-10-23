"""Voyager's hidden-state audit (2026-10-10; borrowed in principle from Pioneer statediff.py and New Horizons NH 96, which
took Voyager's record-diff analysis further): at fixed phases of every session, a fingerprint (hash) of EVERY reachable value
of the learner -- model parameters and buffers, all attributes (dicts, lists, numbers), the world model's state, the loop's
own unrecorded values (rpe, expect, acc_now, settings), and all random-number states (torch global, numpy global, python,
the loop's generators). Record only: run_long calls mark() only when VOY_STATEDIFF is set (a file path); nothing is changed.
diff(a, b) prints, for two recordings, the first (session, phase) where any fingerprint differs and which paths."""
import hashlib, os, pickle, random, sys

import numpy as np
import torch

SKIP = {"w", "learner", "world"} if not os.environ.get("VOY_STATEDIFF_DEEP") else {"learner"}            # back-references to the world / owner (fixed data, huge)


def _h(b):
    return hashlib.sha1(b).hexdigest()[:16]


def _leaf(x):
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.is_sparse:
            t = t.to_dense()
        return _h(t.numpy().tobytes() + str(t.shape).encode())
    if isinstance(x, np.ndarray):
        return _h(x.tobytes() + str(x.shape).encode())
    if isinstance(x, np.random.Generator):
        return _h(repr(x.bit_generator.state).encode())
    return None


def walk(x, path, out, seen, depth=0):
    lf = _leaf(x)
    if lf is not None:
        out[path] = lf
        return
    if isinstance(x, (int, float, str, bool, type(None), np.integer, np.floating)):
        out[path] = repr(x)
        return
    if id(x) in seen or depth > (14 if os.environ.get("VOY_STATEDIFF_DEEP") else 6):
        return
    seen.add(id(x))
    if isinstance(x, dict):
        if len(x) > 64:
            sub = {}
            for k in sorted(x, key=repr):
                walk(x[k], "", sub, seen, depth + 1)
            out[path] = _h(repr(sorted(sub.items())).encode() + repr(sorted(map(repr, x))).encode())
            return
        for k in x:
            walk(x[k], f"{path}[{k!r}]", out, seen, depth + 1)
        return
    if isinstance(x, (list, tuple, set)):
        items = list(x) if not isinstance(x, set) else sorted(x, key=repr)
        if len(items) > 16:
            sub = {}
            for j, v in enumerate(items):
                walk(v, str(j), sub, seen, depth + 1)
            out[path] = _h(repr(sorted(sub.items())).encode())
            return
        for j, v in enumerate(items):
            walk(v, f"{path}[{j}]", out, seen, depth + 1)
        return
    if isinstance(x, torch.nn.Module):
        for n, p in x.state_dict().items():
            out[f"{path}.sd.{n}"] = _leaf(p)
    d = getattr(x, "__dict__", None)
    if d:
        for k, v in d.items():
            if k in SKIP or k.startswith("_modules") or k == "_parameters" or k == "_buffers":
                continue
            walk(v, f"{path}.{k}", out, seen, depth + 1)


def fingerprint(me, extra):
    out = {}
    if os.environ.get("VOY_STATEDIFF_DEEP"):
        # deep mode (E14): the body first with the world model left out, so that objects the two share are recorded
        # under the body's paths (the seen-set records an object once, under the first path that reaches it)
        SKIP.add("wm")
        seen = set()
        walk(me, "me", out, seen)
        SKIP.discard("wm")
        if getattr(getattr(me, "model", None), "wm", None) is not None:
            walk(me.model.wm, "me.model.wm", out, seen)
    else:
        walk(me, "me", out, set())
    for k, v in extra.items():
        walk(v, f"loop.{k}", out, set())
    out["rng.torch"] = _h(torch.get_rng_state().numpy().tobytes())
    out["rng.numpy"] = _h(repr(np.random.get_state()).encode())
    out["rng.python"] = _h(repr(random.getstate()).encode())
    return out


RECORD = []


def mark(session, phase, me, extra):
    RECORD.append((session, phase, fingerprint(me, extra)))
    with open(os.environ["VOY_STATEDIFF"], "wb") as f:
        pickle.dump(RECORD, f)


def diff(pa, pb, show=25, ignore=()):
    """ignore: path fragments known to differ by design (the condition's own flags and the part's own caches)."""
    A, B = pickle.load(open(pa, "rb")), pickle.load(open(pb, "rb"))
    for (sa, ph, fa), (sb, _, fb) in zip(A, B):
        keys = sorted(set(fa) | set(fb))
        bad = [k for k in keys if fa.get(k) != fb.get(k) and not any(g in k for g in ignore)]
        print(f"session {sa + 1} {ph:13} {len(bad):5} differing of {len(keys)}")
        if bad:
            for k in bad[:show]:
                print(f"      {k}")
            return


if __name__ == "__main__":
    diff(sys.argv[1], sys.argv[2], ignore=tuple(sys.argv[3:]))
