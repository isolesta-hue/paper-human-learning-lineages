import sys; sys.path.insert(0, ".")
import numpy as np, torch, integrated as I, worldmodel as WM, data_ascii as D
for seed in (0, 1, 2):
    w = I.build_world("ascii-views", seed); torch.manual_seed(seed)
    me = I.Learner3(w, seed, content=True, where="linear")
    wm = WM.WorldBooks(me); wm.views = True
    for rand in (False, True):
        wm.views_rand = rand; wm.view_key = None
        books = {}
        for vt, k in ((v, w.verb_of[v]) for v in w.verbs):
            dx, dy = D.VERBS[k][1]
            mv = wm.mem[w.pos_to[(dx, dy)]] - wm.mem[w.pos_from[(0, 0)]] - wm.shift
            books[vt] = [[mv, 10, set()]]
        wm.books, wm.n_rows = books, 10 ** 9
        me.order_log = [s for c in w.categories[1:] for k, s, _ in c["heard"] if k == "views-heard"]
        res = {}
        for c in w.categories[1:]:
            for k, s, a in c.get("views", []):
                r = res.setdefault(k, [0, 0]); r[1] += 1; r[0] += int(wm.view_answer(s) == a[0])
        print(seed, "rand" if rand else "views", {k: round(v[0] / v[1], 3) for k, v in res.items()})
