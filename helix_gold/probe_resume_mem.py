"""D11 memory probe (a check, never part of a result): RSS of a resumed life around the restore -- start code done,
state loaded, frame restored, after gc.collect, after malloc_trim -- then the process ends. Needs a resume state file
written by a cut life (RIE_RESUME_STOP) in this dev copy.  python3 probe_resume_mem.py COND SEED"""
import ctypes, gc, os, sys, time
import torch
import run_long as RL


def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) / 2 ** 20
    return float("nan")


cond, seed = sys.argv[1], int(sys.argv[2])
_load, _restore = RL.resume_load, RL.resume_restore


def load(c, s):
    print(f"start code done (before load): {rss():.2f} GB", flush=True)
    st = _load(c, s)
    print(f"state loaded: {rss():.2f} GB", flush=True)
    return st


def restore(frame, state):
    _restore(frame, state)
    print(f"frame restored: {rss():.2f} GB", flush=True)
    gc.collect()
    print(f"after gc.collect: {rss():.2f} GB", flush=True)
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    print(f"after malloc_trim: {rss():.2f} GB", flush=True)
    os._exit(0)


RL.resume_load, RL.resume_restore = load, restore
torch.set_num_threads(1)
RL.life(cond, seed)
