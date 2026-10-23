"""benchmark: one learner in the long world after 6 sessions; time 200 epochs of one session's practice; fingerprint of output."""
import sys, time, copy, pickle, numpy as np, torch
torch.set_num_threads(1)
import integrated as I
from run_schema import PER_SESSION
def setup():
    w = I.build_world("life32", 0); torch.manual_seed(0); rng = np.random.default_rng(70000)
    me = I.Learner3(w, 0)
    sents = {c: [x for _, x, _ in w.categories[c]["heard"]] for c in range(w.n_cat)}
    order = [int(c) for ph in w.phases for c in rng.permutation(ph)]
    sess = [order[i:i + PER_SESSION] for i in range(0, len(order), PER_SESSION)]
    seen = []
    for nc in sess[:6]:
        old = list(seen); seen += nc
        me.prepare([x for c in nc for x in sents[c]], {c: sents[c] for c in old}, dict(I.START), rng)
        me.train_to(40); me.finish()
    nc = sess[6]; old = list(seen)
    return me, [x for c in nc for x in sents[c]], {c: sents[c] for c in old}
if __name__ == "__main__":
    me, new, old = setup()
    times = []
    for rep in range(3):
        b = me.clone(); b.prepare(new, old, dict(I.START), np.random.default_rng(1))
        t = time.time(); b.train_to(200); times.append(time.time() - t)
    with torch.no_grad():
        p = [x.detach().clone() for x in b.model.parameters()]
    print("train 200 epochs: %.2fs (min of 3)" % min(times), "items", len(b.st["phases"][0][1]), "V", b.w.V)
    pickle.dump(p, open(sys.argv[1] if len(sys.argv) > 1 else "bench_params.pkl", "wb"))
