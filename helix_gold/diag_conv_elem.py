"""Diagnosis (a check, never a result): which exam-form pictures fail Dami's element check (criterion 0's gate) and why --
for each failing element, the nearest element the recall lands on, the RAW distance between the two (are they already
nearly the same picture before memory?), the recall error, and the two names. Element table only (no sentence, no item)."""
import os, sys
import numpy as np
import torch
import conv_v1 as CV
import data_ascii as A
from integrated import Learner3
from run_seed_transfer import LATENT

torch.set_num_threads(1)
case = sys.argv[1] if len(sys.argv) > 1 else "life_000"
w, _ = CV.convert_life(os.path.join(CV.EXAM_ROOT, "exam_dev", "dev1000", "lives", case))
me = Learner3(w, 0, content=True, where="linear")
g = torch.Generator().manual_seed(10_000 + 0)
Wd = torch.randn(LATENT, A.RAW_DIM, generator=g) / np.sqrt(LATENT)
rec = torch.sigmoid(me.memory @ Wd).numpy()
lin = me.where_decoder if hasattr(me, "where_decoder") else None
if lin is not None and lin is not False:
    pi = np.array(sorted(w.pos_of)); rec[pi] = 0.5; rec[pi, :w.where_dims] = (0.5 + me.memory[pi] @ lin).numpy()
V = w.vecs
d = ((rec[:, None, :] - V[None, :, :]) ** 2).sum(-1)
nn_ = d.argmin(1)
bad = np.flatnonzero(nn_ != np.arange(w.V))
raw = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(raw, np.inf)
print(f"{case}: V {w.V}, failing {len(bad)} ({1 - len(bad) / w.V:.4f} recalled)")
same = sum(raw[t, nn_[t]] < 1e-9 for t in bad)
print(f"  failing whose landing element has IDENTICAL raw features: {same} / {len(bad)}")
rr = np.array([raw[t, nn_[t]] for t in bad]); er = np.array([np.sqrt(d[t, t]) for t in bad])
print(f"  raw distance to the landing element: median {np.median(np.sqrt(rr)):.4f}, max {np.sqrt(rr).max():.4f}")
print(f"  recall error of the failing ones: median {np.median(er):.4f}; of all pictures: "
      f"{np.median([np.sqrt(d[t, t]) for t in range(w.V) if w.origin[t] in ('drawn', 'converted', 'picture')]):.4f}")
allnn = np.sqrt(raw.min(1))
pics = [t for t in range(w.V) if w.origin[t] not in ('element', 'word')]
print(f"  nearest-other raw distance, all pictures: median {np.median(allnn[pics]):.4f}, 1% {np.percentile(allnn[pics], 1):.4f}; "
      f"elements with an identical twin: {int((allnn < 1e-9).sum())}")
for t in bad[:12]:
    print(f"   {w.names[t]!s:28.28} -> {w.names[nn_[t]]!s:28.28} raw {np.sqrt(raw[t, nn_[t]]):.4f} err {np.sqrt(d[t, t]):.4f}")
