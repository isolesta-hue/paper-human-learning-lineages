"""Readiness step 2 check (2026-10-17; a memory check, not an exam score): Module A with rehearsal (`+rehearse`,
run_seed_transfer.memorize_rehearse) on the exam's element tables (shared development seed dev1000, lives 000 / 001 --
element table only: no sentence heard, no item read) and on Dami's own worlds (seed 0): criterion 0's element check
(data_ascii.element_check, unchanged) without / with rehearsal, the rounds used, the birth time. Also: rehearsal's first
round equals memorize() to the bit (so a table recalled after one round is unchanged)."""
import os, sys, time
import numpy as np
import torch
import data_ascii as A
import run_seed_transfer as RST
from integrated import Learner3

torch.set_num_threads(1)
v = torch.tensor(np.random.default_rng(0).uniform(0.2, 0.8, (50, A.RAW_DIM)).astype(np.float32))
z0, _ = RST.memorize(v, 3)
z1, _, used, share = RST.memorize_rehearse(v, 3, rounds=1)
print(f"first round == memorize(): {bool(torch.equal(z0, z1))} (rounds {used}, recalled {share:.3f})", flush=True)
cases = ["conv:life_000", "conv:life_001"] + [f"{x}:0" for x in ("ascii", "ascii-kor", "ascii-multi", "ascii-ten", "ascii-sp",
                                                                   "ascii-bi", "ascii-c8", "ascii-nw", "ascii-when")]
for case in cases:
    kind, arg = case.split(":")
    if kind == "conv":
        import conv_v1 as CV
        w, _ = CV.convert_life(os.path.join(CV.EXAM_ROOT, "exam_dev", "dev1000", "lives", arg)); seed = 0
    else:
        import run_long as RL
        w = RL.build_world(kind, int(arg)); seed = int(arg)
    out = []
    for reh in (False, True):
        t0 = time.process_time()
        me = Learner3(w, seed, content=True, where="linear", rehearse=reh)
        t1 = time.process_time()
        res = A.element_check(w, me.memory, seed, linear_pos=me.where_decoder if hasattr(me, "where_decoder") else None)
        out.append((res["all"], t1 - t0, getattr(me, "rehearse_rounds", 1)))
    print(f"{case:16s} V {w.V:5d}  plain {out[0][0]:.4f} ({out[0][1]:.1f}s)  rehearse {out[1][0]:.4f} "
          f"({out[1][1]:.1f}s, rounds {out[1][2]})", flush=True)
print("exit 0")
