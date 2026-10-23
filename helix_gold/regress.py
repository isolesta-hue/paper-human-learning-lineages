"""optimisation pass regression: the optimised code vs the original's cached results (cache_int, finding 69), same seeds.
Equal up to float rounding is not bit-equal, so the test is statistical: per-session never-heard differences should centre on
0 with no sign bias; also time per life."""
import os, pickle, sys, time, numpy as np, torch
torch.set_num_threads(1)
import run_integrated as RI
from run_unified_continual import sign_p
CACHE = "../rie/cache_int"
rows = []
SEEDS = range(int(sys.argv[1]), int(sys.argv[2]))
CONDS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["content", "both"]
for seed in SEEDS:
    ref = pickle.load(open(os.path.join(CACHE, f"seed{seed}.pkl"), "rb"))
    for wn in ("schema", "conj"):
        for c in CONDS:
            t = time.time()
            st = RI.life(wn, c, seed)
            a = np.array([s["never-heard"] for s in st]); b = np.array([s["never-heard"] for s in ref[(wn, c)]])
            rows.append((seed, wn, c, time.time() - t, a - b))
            print(seed, wn, c, "%.0fs" % (time.time() - t), "max|d| %.4f" % np.abs(a - b).max(), "end d %+.4f" % (a[-1] - b[-1]), flush=True)
for wn in ("schema", "conj"):
    for c in CONDS:
        d = np.array([r[4] for r in rows if r[1] == wn and r[2] == c])
        e = d[:, -1]; l = d.mean(1)
        print(f"{wn} {c}: exact sessions {np.mean(np.abs(d) < 1e-9):.2f}  life d {l.mean():+.4f} ({(l>0).sum()}/{(l<0).sum()})  end d {e.mean():+.4f} ({(e>0).sum()}/{(e<0).sum()})  p={sign_p((e>0).sum(), (e<0).sum()):.2f}")
