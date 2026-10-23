"""Track A W2 checks (2026-10-13). reg: finding 143/144's Voyager B life (both@ascii+rediscover+chain+wm+wmlast, seed 0) under
the new code (the "@nw" option added to data_ascii / integrated / run_long) must equal its cache exactly. Smoke (seed 0) on
"@nw": run_newword.CONDS (P / body)."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
which, t0 = sys.argv[1], time.time()
if which == "reg":
    K = "both@ascii+rediscover+chain+wm+wmlast"
    out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
    d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
    same_keys = all(set(a) == set(b) for a, b in zip(old, out)) and len(old) == len(out)
    nan_same = all((isinstance(a[k], float) and np.isnan(a[k])) == (isinstance(b[k], float) and np.isnan(b[k]))
                   for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float))
    print(which, "regression max diff", d, "| same records/keys", same_keys, "| nan pattern same", nan_same)
else:
    import run_newword
    out = RL.one((0, run_newword.CONDS[which]))
    for k in ([f"nw{n} {t}" for n in (1, 2, 4, 8) for t in ("new", "early", "all")] + [f"name{n} same" for n in (1, 2, 4, 8)]
              + ["never-heard", "once all", "unseen all", "chain"]):
        print(f"{which:4} {k:12}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out))
print("time", round(time.time() - t0))
