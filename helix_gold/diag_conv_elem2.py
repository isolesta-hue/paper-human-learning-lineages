"""Diagnosis (a check, never a result): exam-form element recall (life_000's element table) when Module A (a) practises
longer (MEMORIZE_STEPS) or (b) has a wider memory (LATENT) -- which of the two the shortfall is. Element table only."""
import os, sys
import numpy as np
import torch
import conv_v1 as CV
import data_ascii as A
import run_seed_transfer as RST
from integrated import Learner3

torch.set_num_threads(1)
w, _ = CV.convert_life(os.path.join(CV.EXAM_ROOT, "exam_dev", "dev1000", "lives", sys.argv[1] if len(sys.argv) > 1 else "life_000"))
for lat, steps in ((32, 400), (32, 2000), (40, 400), (48, 400), (64, 400)):
    RST.LATENT, RST.MEMORIZE_STEPS = lat, steps
    me = Learner3(w, 0, content=True, where="linear") if lat == 32 else None
    if me is None:                                  # wider memory: Module A alone (the where channel is unchanged by LATENT? no --
        z, mse = RST.memorize(torch.tensor(w.vecs), 0)   # its linear memory uses dims = where_dims), so read the plain recall
        class M: pass
        me = M(); me.memory = z; me.where_decoder = None
        res = A.element_check(w, me.memory, 0, linear_pos=None)
    else:
        res = A.element_check(w, me.memory, 0, linear_pos=me.where_decoder)
    print(f"LATENT {lat:3d} steps {steps:5d}: recall all {res['all']:.4f}  " +
          "  ".join(f"{k} {v:.4f}" for k, v in res.items() if k != "all"), flush=True)
