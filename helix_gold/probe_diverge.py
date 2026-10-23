"""Where do two lives of the same seed first part? Compare the cached per-session records key by key."""
import pickle, sys, numpy as np
a_, b_ = sys.argv[1], sys.argv[2]
A = pickle.load(open(f"cache_long_v2/seed0_{a_}.pkl", "rb"))
B = pickle.load(open(f"cache_long_v2/seed0_{b_}.pkl", "rb"))
print("A =", a_, "\nB =", b_)
for i, (x, y) in enumerate(zip(A, B)):
    diff = []
    for k in sorted(set(x) & set(y)):
        u, v = x[k], y[k]
        try:
            if isinstance(u, (int, float, np.floating)) and isinstance(v, (int, float, np.floating)):
                if not (np.isnan(u) and np.isnan(v)) and abs(u - v) > 1e-12:
                    diff.append((k, u, v))
            elif u != v:
                diff.append((k, str(u)[:40], str(v)[:40]))
        except Exception:
            pass
    only = sorted(set(x) ^ set(y))
    print(f"session {i + 1}: {len(diff)} differing keys" + (f" (keys in one only: {len(only)})" if only and i == 0 else ""))
    for k, u, v in diff[:14]:
        print(f"    {k:28} A {u}   B {v}")
