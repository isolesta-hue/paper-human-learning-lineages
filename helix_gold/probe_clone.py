"""Clone-cost probe (2026-10-12, speed pass): times each part copied by Learner.clone during the first N seconds of one life
(seed 0); read-only, no cache written. What this assumes: the first N seconds are representative."""
import copy, signal, sys, time, collections
import integrated, run_long as RL

T = collections.Counter(); N = collections.Counter(); S = {}
_dc = copy.deepcopy
def timed(name, x):
    t = time.perf_counter(); y = _dc(x); T[name] += time.perf_counter() - t; N[name] += 1
    if name not in S: S[name] = type(x).__name__ + (f" len={len(x)}" if hasattr(x, "__len__") else "")
    return y
def clone(self):
    t0 = time.perf_counter()
    c = copy.copy(self)
    self.model.chunks = None
    wm = getattr(self.model, "wm", None)
    c.model = timed("model", self.model)
    self.model.chunks = self.chunks
    c.hebb = timed("hebb", self.hebb); c.replay = timed("replay", self.replay); c.rec = c.hebb
    c.pattern = timed("pattern", self.pattern); c.chunks = timed("chunks", self.chunks); c.model.chunks = c.chunks
    c.known = set(self.known); c.FS, c.policy = dict(self.FS), dict(self.policy)
    if hasattr(self, "cued"): c.cued = timed("cued", self.cued)
    c.plas = timed("plas", self.plas); c.plas2 = timed("plas2", self.plas2)
    if getattr(self, "keep_order", False): c.order_log = list(self.order_log)
    if getattr(c.model, "wm", None) is not None: c.model.wm.learner = c
    c.st = None
    T["TOTAL clone"] += time.perf_counter() - t0
    return c
for cls in [v for v in vars(integrated).values() if isinstance(v, type) and "clone" in vars(v)]:
    cls.clone = clone
class Stop(Exception): pass
def h(*a): raise Stop
cond, secs = sys.argv[1], int(sys.argv[2])
signal.signal(signal.SIGALRM, h); signal.alarm(secs)
t = time.perf_counter()
try: RL.life(cond, 0)
except Stop: pass
print(cond, f"wall {time.perf_counter()-t:.0f}s")
for k, v in T.most_common(): print(f"  {k:12s} {v:8.1f}s  n={N[k]}  {S.get(k,'')}")
