"""D75 step 1 smoke (a check, never a result): the "@scene" / "@sceneshuf" lives of seed 0 cut after session K
(RIE_RESUME_STOP, this dev copy's resume directory); prints that the scene keys exist and the part's event count per
session -- no score is read here.  python3 smoke_scene.py K"""
import os, pickle, subprocess, sys
import run_long as RL

k = int(sys.argv[1])
for world in ("scene", "sceneshuf"):
    cond = f"both@ascii@{world}+rediscover+scene"
    rp = RL.resume_path(cond, 0)
    if os.path.exists(rp):
        os.remove(rp)
    r = subprocess.run([sys.executable, "-c", f"import run_long as RL, torch; torch.set_num_threads(1); RL.life({cond!r}, 0)"],
                       env=dict(os.environ, RIE_RESUME_STOP=str(k)), capture_output=True, text=True)
    if not os.path.exists(rp):
        print(cond, "no state; exit", r.returncode, r.stderr[-3000:])
        continue
    st = pickle.load(open(rp, "rb"))
    hist = [v for v in st["vars"].values() if isinstance(v, list) and v and isinstance(v[0], dict)
            and any(str(key).startswith("scene") for key in v[0])]
    print(f"{cond} seed 0 after session {st['i'] + 1} (cut exit {r.returncode}):")
    for i, s in enumerate(hist[0] if hist else []):
        keys = sorted(x for x in s if str(x).startswith("scene"))
        print(f"  session {i + 1}: {len(keys)} scene keys, events {s.get('scene events')}, words {s.get('scene words')}")
    os.remove(rp)
print("exit 0")
