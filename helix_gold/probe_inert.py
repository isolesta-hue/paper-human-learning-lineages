"""D26 (1) S2 inertness audit (seed 0, record only): in the "+knownonly" life, every word- or room-indexed value of a word
(room) not yet heard (met) must stay exactly at its birth value, session after session -- then having its place from birth
is the same as appending it when first heard (growth for free). After each session, every tensor/array of the learner with
a dimension of size V (or n_cat) is compared, on the rows/columns of the still-unknown words (rooms), with the birth copy."""
import sys, copy
sys.path.insert(0, ".")
import numpy as np, torch
import integrated as I, run_long as RL
torch.set_num_threads(1)
C = sys.argv[1] if len(sys.argv) > 1 else "both@ascii+rediscover+chain+wm+wmlast+knownonly"
STOP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
state = {"birth": None, "i": 0}

def tensors(me, V, NC):
    out, seen = {}, set()
    def walk(x, path, d=0):
        if id(x) in seen or d > 6:
            return
        seen.add(id(x))
        if isinstance(x, torch.Tensor):
            if not x.is_sparse and (V in x.shape or NC in x.shape):
                out[path] = x.detach().clone()
            return
        if isinstance(x, np.ndarray):
            if V in x.shape or NC in x.shape:
                out[path] = x.copy()
            return
        if isinstance(x, torch.nn.Module):
            for n, p in x.state_dict().items():
                if V in p.shape or NC in p.shape:
                    out[f"{path}.sd.{n}"] = p.detach().clone()
        dd = x if isinstance(x, dict) else getattr(x, "__dict__", None)
        if dd:
            for k, v in list(dd.items()):
                if k in ("w", "learner", "world", "st") or str(k).startswith("_"):
                    continue
                walk(v, f"{path}.{k}", d + 1)
    walk(me, "me")
    return out

orig = I.Learner3.session
def session(self, *a, **k):
    V, NC = self.w.V, self.w.n_cat
    if state["birth"] is None:
        state["birth"] = tensors(self, V, NC)
    r = orig(self, *a, **k)
    state["i"] += 1
    unk = torch.tensor(sorted(set(range(V)) - set(self.known)))
    unc = torch.tensor(sorted(set(range(NC)) - {int(self.cats[x]) for x in self.known}))
    now = tensors(self, V, NC)
    moved = []
    for p, b in state["birth"].items():
        if p not in now or tuple(now[p].shape) != tuple(b.shape):
            continue
        a_, b_ = torch.as_tensor(np.asarray(now[p])), torch.as_tensor(np.asarray(b))
        for dim, n in enumerate(b_.shape):
            idx = unk if n == V else unc if n == NC else None
            if idx is None or len(idx) == 0:
                continue
            x, y = a_.index_select(dim, idx), b_.index_select(dim, idx)
            if not torch.equal(x, y):
                moved.append(f"{p} (dim {dim}, max |change| {float((x.double() - y.double()).abs().max()):.3g})")
    print(f"session {state['i']}: unknown words {len(unk)}, unmet rooms {len(unc)}; values of the unknown that moved: {len(moved)}")
    for m in moved[:12]:
        print("   ", m)
    sys.stdout.flush()
    if state["i"] >= STOP:
        raise SystemExit(0)
    return r
I.Learner3.session = session
RL.life(C, 0)
