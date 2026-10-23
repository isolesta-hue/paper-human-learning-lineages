"""D25 checks (2026-10-13). reg: finding 143/144's Voyager B life (seed 0) under the new code (d25.py hooks in integrated.py and
run_long.py, the piece deadline) must equal its cache exactly. Smoke (seed 0): run_d25 conditions by name (gain: N U F K M E,
safety: tN tU tF tM) -- only sane keys and no crash are looked at; the runner is fixed before."""
import pickle, sys, time
import numpy as np, torch
import run_long as RL
torch.set_num_threads(1)
t0 = time.time()
for which in sys.argv[1:]:
    if which == "reg":
        K = "both@ascii+rediscover+chain+wm+wmlast"
        out, old = RL.life(K, 0), pickle.load(open(f"{RL.CACHE}/seed0_{K}.pkl", "rb"))
        d = max(abs(a[k] - b[k]) for a, b in zip(old, out) for k in a if k in b and isinstance(a[k], float) and not np.isnan(a[k]))
        same_keys = all(set(a) == set(b) for a, b in zip(old, out)) and len(old) == len(out)
        print("reg regression max diff", d, "| same records/keys", same_keys, flush=True)
        continue
    import run_d25
    c = run_d25.SAFE[which[1:]] if which.startswith("t") else run_d25.GAIN[which]
    out = RL.one((0, c))
    keys = ["d25 chain body", "chain", "never-heard", "d25 never-heard body", "d25 candidates", "d25 transferred",
            "d25 transferred tale", "d25 new", "d25 k", "d25 mark keep", "d25 mark forget"]
    if which.startswith("t"):
        keys += ["grav body unturned", "tale preview marked body", "tale preview unmarked body", "tale in"]
    for k in keys:
        print(f"{which:3} {k:26}", " ".join(f"{st.get(k, np.nan):.3f}" for st in out), flush=True)
    print(which, "time", round(time.time() - t0), flush=True)
