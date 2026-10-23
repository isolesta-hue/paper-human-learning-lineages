"""D69 seed-0 smoke (a check, never a result): one life of "@ten" with every part on plus the self-trust gate ("+gate"),
run through life() directly (nothing cached). Prints, per session, which parts speak and their records (differed / right)
-- the mechanism only; no outcome key is read here.
    python3 smoke_d69.py"""
import time
import numpy as np
import torch
import run_long as RL
import worldmodel as WM

torch.set_num_threads(1)
cond = ("both@ascii@ten+rediscover+chain+wm+wmctx+wmrev+bind+wmlast+track+count+contain"
        "+props+grav+wmframe+succ+scale+views+vturn+goal+self+gate")
t0 = time.time()
stats = RL.life(cond, 0)
print(f"smoke {cond} seed 0: {len(stats)} sessions, {time.time() - t0:.0f} s")
for i, st in enumerate(stats):
    row = []
    for p, _ in WM.GATED:
        if f"gate {p} speaks" in st:
            row.append(f"{p}:{'ON' if st[f'gate {p} speaks'] else 'q'}({int(st[f'gate {p} right'])}/{int(st[f'gate {p} differed'])})")
    print(f"  session {i + 1:2d}: " + " ".join(row))
print("exit 0")
