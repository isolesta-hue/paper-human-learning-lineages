"""D11 memory probe (a check, never part of a result): what writing a resume state costs. Loads a written state (as the
life holds it at that session), then writes it once -- MODE "dumps" (bytes in memory, then the file: the queue's code) or
"dump" (straight into the file) -- and reports the process peak before and after the write.
    python3 probe_save_mem.py STATE.pkl dumps|dump"""
import os, pickle, resource, sys, time
import torch  # noqa: F401 (tensors in the state)


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) / 2 ** 20


path, mode = sys.argv[1], sys.argv[2]
t0 = time.time()
with open(path, "rb") as f:
    state = pickle.load(f)
a, pa = rss(), peak()
out = path + f".probe_{mode}"
t1 = time.time()
if mode == "dumps":
    data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out, "wb") as f:
        f.write(data)
    del data
else:
    with open(out, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
same = open(out, "rb").read() == pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL) if mode == "dump" else None
print(f"{mode}: file {os.path.getsize(path) / 1e6:.0f} MB; held after load {a:.2f} GB (peak {pa:.2f}); peak after write "
      f"{peak():.2f} GB -> the write adds {peak() - max(a, pa):+.2f} GB over the load peak ({time.time() - t1:.0f} s)"
      + (f"; bytes equal to dumps: {same}" if same is not None else ""), flush=True)
os.remove(out)
