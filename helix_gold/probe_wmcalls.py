"""E14: which code touches the world model DURING a session of the +wmmeasure life (it should be silent)? Every WorldBooks
method call made while Learner3.session runs is logged with its call stack (seed 0, 3 sessions). Record only."""
import os, sys, traceback, collections
sys.path.insert(0, ".")
os.environ["VOY_STATEDIFF"] = "/tmp/claude-0/-home-user/23cc98b2-a804-5343-bef4-6479ad5f8346/scratchpad/sd_tmp.pkl"
os.environ["VOY_STATEDIFF_STOP"] = "3"
import torch, worldmodel, integrated, run_long as RL
torch.set_num_threads(1)
STATE = {"in": False}
calls = collections.Counter()
orig_session = integrated.Learner3.session
def session(self, *a, **k):
    STATE["in"] = True
    try:
        return orig_session(self, *a, **k)
    finally:
        STATE["in"] = False
integrated.Learner3.session = session
for name in [n for n, f in vars(worldmodel.WorldBooks).items() if callable(f) and not n.startswith("__")]:
    f = getattr(worldmodel.WorldBooks, name)
    def wrap(f=f, name=name):
        def g(self, *a, **k):
            if STATE["in"]:
                st = "".join(traceback.format_stack(limit=6)[:-1])
                calls[(name, getattr(self, "silent", None), st)] += 1
            return f(self, *a, **k)
        return g
    setattr(worldmodel.WorldBooks, name, wrap())
RL.life("both@ascii@views+rediscover+chain+wm+bind+wmlast+views+wmmeasure", 0)
print("world-model calls inside sessions:", sum(calls.values()))
for (name, silent, st), n in calls.most_common(6):
    print(f"--- {name} x{n}, silent={silent}\n{st}")
