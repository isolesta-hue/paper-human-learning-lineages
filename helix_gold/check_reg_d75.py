"""Life regression after D72 / readiness step 2 / D75 (2026-10-17): an existing condition's seed-0 life with the current
code (resume off) against its cached result -- every key of every session, and the pickle bytes. The new code paths are
all guarded (d72=None, rehearse=False, no scene part): this checks it.  python3 check_reg_d75.py"""
import os, pickle, time
os.environ["RIE_NO_RESUME"] = "1"
import torch
import run_long as RL

torch.set_num_threads(1)
cond, seed = "both@ascii@multi+rediscover+chain+wm+bind+wmlast+track", 0
t0 = time.time()
new = RL.life(cond, seed)
ref = pickle.load(open(os.path.join(RL.CACHE, f"seed{seed}_{cond}.pkl"), "rb"))
diff = sorted({k for a, b in zip(new, ref) for k in set(a) | set(b) if repr(a.get(k)) != repr(b.get(k))})
same_bytes = pickle.dumps(new) == open(os.path.join(RL.CACHE, f"seed{seed}_{cond}.pkl"), "rb").read()
print(f"life regression {cond} seed {seed}: sessions {len(new)} / {len(ref)}, keys differing {diff or 'none'}; "
      f"pickle bytes equal to the cache: {same_bytes}")
print(f"exit 0 ({time.time() - t0:.0f} s)")
