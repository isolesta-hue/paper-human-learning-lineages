"""D11 check (2026-10-16; the user: "재시작을 이어도는 1번으로 해줘"): a life cut after session K (the process ends right after
the state of session K is written, as a container restart would end it) and started again must equal the uninterrupted life
exactly -- the same stat dicts (type-strict, NaN-aware deep compare) as the cached life from the queue. (Pickle bytes are
reported too; they can differ only through pickle's memo of shared objects -- the finished sessions' dicts come back from
the written state as new objects, so a key string shared by all sixteen dicts in one life is shared by two groups in the
other -- not through any value.)
Run in this dev copy: its cache_long_v2 is the queue's (read only here: life() is called directly, nothing is cached), its
_progress / _cpu / _resume directories are its own.
    python3 check_resume.py COND SEED K"""
import os
import pickle
import subprocess
import sys
import time

import numpy as np


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if type(a) is not type(b):                     # type-strict (np.float64 vs float would pickle differently)
        return False
    if isinstance(a, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


cond, seed, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import run_long as RL                                     # noqa: E402

cache = os.path.join(RL.CACHE, f"seed{seed}_{cond}.pkl")
assert os.path.exists(cache), f"no cached uninterrupted life: {cache}"
rp = RL.resume_path(cond, seed)
if os.path.exists(rp):
    os.remove(rp)
t0 = time.time()
cut = subprocess.run([sys.executable, "-c", f"import run_long as RL, torch; torch.set_num_threads(1); RL.life({cond!r}, {seed})"],
                     cwd=here, env=dict(os.environ, RIE_RESUME_STOP=str(k)), capture_output=True, text=True)
state = pickle.load(open(rp, "rb")) if os.path.exists(rp) else None
print(f"cut after session {k}: exit {cut.returncode} (17 expected), state written for session "
      f"{None if state is None else state['i'] + 1}, {os.path.getsize(rp) / 1e6 if state else 0:.0f} MB, "
      f"not written (made again): {None if state is None else state['skipped']} ({time.time() - t0:.0f} s)", flush=True)
out_path = os.path.join(here, f"resume_out_seed{seed}.pkl")
t1 = time.time()
res = subprocess.run([sys.executable, "-c", "import pickle, run_long as RL, torch; torch.set_num_threads(1); "
                      f"pickle.dump(RL.life({cond!r}, {seed}), open({out_path!r}, 'wb'))"],
                     cwd=here, capture_output=True, text=True)
print(f"resumed: exit {res.returncode}; {[l for l in res.stderr.splitlines() if 'D11' in l]} ({time.time() - t1:.0f} s)")
fresh = pickle.load(open(out_path, "rb"))
cached = pickle.load(open(cache, "rb"))
keys = sorted({kk for st in cached + fresh for kk in st})
diff = [kk for kk in keys if not all(same(a.get(kk), b.get(kk)) for a, b in zip(cached, fresh))]
print(f"resume check {cond} seed {seed} cut after {k}: sessions {len(fresh)} / {len(cached)}, keys differing "
      f"{diff if diff else 'none'} (type-strict, NaN-aware); pickle bytes equal to the cache: "
      f"{pickle.dumps(fresh) == open(cache, 'rb').read()} (bytes may differ only by shared-object memo: the first "
      f"sessions' dicts come back from the written state as new objects)")
import io                                               # noqa: E402


def flat(obj):
    """pickle bytes without pickle's memo (no shared references written): equal objects of equal types -> equal bytes."""
    b = io.BytesIO()
    p = pickle.Pickler(b, protocol=pickle.HIGHEST_PROTOCOL)
    p.fast = True
    p.dump(obj)
    return b.getvalue()


print(f"memo-free pickle bytes equal: {flat(fresh) == flat(cached)}")
os.remove(out_path)
if os.path.exists(rp):
    os.remove(rp)
print("exit 0")
