"""D83 (1) mechanism peek (a check, never a result): the @ten ALL+useful life of seed 0 is cut after session K
(RIE_RESUME_STOP; its own resume file, removed after), and its written state is read for the usefulness records only --
per part: its net and changed count over all question forms, and per form where it changed something, whether it speaks
there; the share of answer decisions that fell back to the all-forms record; wall time. No outcome key is read.
    python3 peek_d83.py K [world=ten]"""
import os, pickle, subprocess, sys, time
import run_long as RL
import run_w7pilot as P
import worldmodel as WM

k = int(sys.argv[1])
world = sys.argv[2] if len(sys.argv) > 2 else "ten"
cond = P.cond(world, "ALL") + "+useful"
fit = set(f for f in P.FIT[world].split("+") if f) - {"wm", "wmlast"}
rp = RL.resume_path(cond, 0)
if os.path.exists(rp):
    os.remove(rp)
t0 = time.time()
r = subprocess.run([sys.executable, "-c", f"import run_long as RL, torch; torch.set_num_threads(1); RL.life({cond!r}, 0)"],
                   env=dict(os.environ, RIE_RESUME_STOP=str(k)), capture_output=True, text=True)
st = pickle.load(open(rp, "rb"))
wm = st["vars"]["me"].model.wm
kinds = list(wm.learner.w.kinds)
print(f"peek {cond} seed 0 after session {st['i'] + 1} (cut exit {r.returncode}, {time.time() - t0:.0f} s):")
for p, a in WM.GATED:
    if getattr(wm, a, False):
        net, n = wm.use_all.get(p, (0, 0))
        forms = sorted((s, v) for (q, s), v in wm.use_rec.items() if q == p)
        say = [s for s, _ in forms if wm.use_say(p, s)]
        print(f"  {p:8s} {'FIT' if p in fit else 'non'}  net {net:+5d} changed {n:5d}  forms {len(forms):3d}  speaks in {len(say):3d}")
        for s, (fn, fc) in forms[:12]:
            print(f"      form {'/'.join(kinds[x] for x in s):40s} net {fn:+5d} changed {fc:4d} -> {'speaks' if fn > 0 else 'quiet'}")
c = wm.use_count
print(f"  answer rows {c.get('rows', 0)}, decisions by a form record {c.get('in form', 0)}, by the all-forms record {c.get('backoff', 0)}, "
      f"views {len(wm.use_views)}")
os.remove(rp)
print("exit 0")
