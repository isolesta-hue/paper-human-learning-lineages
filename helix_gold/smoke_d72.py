"""D72 smoke (a check, never a result): the "@d72cat" and "@d72verb" lives of seed 0 cut after session K (RIE_RESUME_STOP,
this dev copy's own resume directory); prints only that the D72 keys exist per session and the times heard -- no score is
read here (the scores are for the pre-registered runner).  python3 smoke_d72.py K"""
import os, pickle, subprocess, sys
import run_long as RL

k = int(sys.argv[1])
for subj in ("cat", "verb"):
    cond = f"both@ascii@d72{subj}+rediscover+chain+wm+wmctx+wmlast"
    rp = RL.resume_path(cond, 0)
    if os.path.exists(rp):
        os.remove(rp)
    r = subprocess.run([sys.executable, "-c", f"import run_long as RL, torch; torch.set_num_threads(1); RL.life({cond!r}, 0)"],
                       env=dict(os.environ, RIE_RESUME_STOP=str(k)), capture_output=True, text=True)
    if not os.path.exists(rp):
        print(cond, "no state; exit", r.returncode, r.stderr[-2000:])
        continue
    st = pickle.load(open(rp, "rb"))
    hist = [v for v in st["vars"].values() if isinstance(v, list) and v and isinstance(v[0], dict)
            and any(str(key).startswith("d72") for key in v[0])]
    print(f"{cond} seed 0 after session {st['i'] + 1} (cut exit {r.returncode}):")
    for i, s in enumerate(hist[0] if hist else []):
        keys = sorted(x for x in s if str(x).startswith("d72"))
        heard = [int(s[f"d72 k{j}"]) for j in range(4)]
        print(f"  session {i + 1}: {len(keys)} d72 keys, times heard {heard}, "
              f"forms {sorted({' '.join(x.split()[:2]) for x in keys if 'k' not in x.split()[1] and 'dens' not in x})}")
    os.remove(rp)
print("exit 0")
