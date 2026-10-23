"""D69 diagnostic (a check, no outcome key): what do a life's heard sentences look like, and which of them does the world
model answer when given all but the last element? The @ten ALL+gate seed-0 life is cut after session K; from its written
state, the next session's new sentences are shown as token kinds, and the world model (every part on, no gate) is asked."""
import os, pickle, subprocess, sys
from collections import Counter
import torch
import run_long as RL
import lab2 as L_

k = int(sys.argv[1])
cond = ("both@ascii@ten+rediscover+chain+wm+wmctx+wmrev+bind+wmlast+track+count+contain"
        "+props+grav+wmframe+succ+scale+views+vturn+goal+self+gate")
rp = RL.resume_path(cond, 0)
if not os.path.exists(rp):
    subprocess.run([sys.executable, "-c", f"import run_long as RL, torch; torch.set_num_threads(1); RL.life({cond!r}, 0)"],
                   env=dict(os.environ, RIE_RESUME_STOP=str(k)), capture_output=True, text=True)
st = pickle.load(open(rp, "rb"))
V = st["vars"]
me, w, sents, sessions = V["me"], V["w"], V["sents"], V["sessions"]
wm = me.model.wm
kinds = list(w.kinds)
new = [s for c in sessions[st["i"] + 1] for s in sents[c]]
print(f"after session {st['i'] + 1}: next session has {len(new)} new sentences; kinds = {kinds}")
shape = Counter(" ".join(kinds[w.kind_of[x]][:4] for x in s) for s in new)
for sh, n in shape.most_common(12):
    print(f"  {n:4d}  {sh}")
items = [(list(x[:-1]), int(x[-1])) for x in new if len(x) >= 2]
cx, truth = L_.items_tensor(items)
with torch.no_grad():
    out = wm._term(cx)
fired = out.max(1).values > 0
right = fired & (out.argmax(1) == truth)
print(f"world model answers {int(fired.sum())} / {len(items)} truncated sentences, right {int(right.sum())}")
by = Counter()
for (c_, t_), f_, r_ in zip(items, fired.tolist(), right.tolist()):
    by[(" ".join(kinds[w.kind_of[x]][:4] for x in c_[-3:]) + " -> " + kinds[w.kind_of[t_]][:4], f_, r_)] += 1
for key, n in by.most_common(15):
    print(f"  {n:4d}  {key}")
print("trusted (tracker):", wm.trusted(), wm.track_trust, "| order_log", len(me.order_log), "| start particle", wm.start_particle)
