"""D69 mechanism peek (a check, never a result): the @ten ALL+gate life of seed 0 is cut after session K (RIE_RESUME_STOP;
this dev copy's own resume directory), and its written state is read for the gate's records only -- which parts speak,
how often each part changed a prediction of the session's new sentences and how often its prediction was the heard one.
No outcome key is read.  python3 peek_d69.py K"""
import os, pickle, subprocess, sys
import run_long as RL
import worldmodel as WM

k = int(sys.argv[1])
cond = ("both@ascii@ten+rediscover+chain+wm+wmctx+wmrev+bind+wmlast+track+count+contain"
        "+props+grav+wmframe+succ+scale+views+vturn+goal+self+gate")
rp = RL.resume_path(cond, 0)
if os.path.exists(rp):
    os.remove(rp)
r = subprocess.run([sys.executable, "-c", f"import run_long as RL, torch; torch.set_num_threads(1); RL.life({cond!r}, 0)"],
                   env=dict(os.environ, RIE_RESUME_STOP=str(k)), capture_output=True, text=True)
st = pickle.load(open(rp, "rb"))
wm = st["vars"]["me"].model.wm
fit = {"wmctx", "wmrev", "bind", "track", "count", "contain"}
print(f"peek {cond} seed 0 after session {st['i'] + 1} (cut exit {r.returncode}):")
for p, a in WM.GATED:
    if getattr(wm, a, False):
        h, n = wm.part_trust.get(p, (0, 0))
        print(f"  {p:8s} {'FIT' if p in fit else 'non'}  speaks {'yes' if wm.part_ok(p) else 'no '}  differed {n:5d}  right {h:5d}")
os.remove(rp)
print("exit 0")
