import sys; sys.path.insert(0, ".")
import numpy as np, torch, integrated as I, worldmodel as WM, data_ascii as D
for seed in (0, 1, 2):
    w = I.build_world("ascii-goal", seed); torch.manual_seed(seed)
    me = I.Learner3(w, seed, content=True, where="linear")
    wm = WM.WorldBooks(me); wm.goal = True
    books = {}
    for vt in w.verbs:
        dx, dy = D.VERBS[w.verb_of[vt]][1]
        books[vt] = [[wm.mem[w.pos_to[(dx, dy)]] - wm.mem[w.pos_from[(0, 0)]] - wm.shift, 10, set()]]
    wm.books, wm.n_rows = books, 10 ** 9
    for rand in (False, True):
        wm.goal_rand = rand
        res = {}
        for c in w.categories[1:]:
            for k, s, a in c.get("goal", []):
                r = res.setdefault(k, [0, 0]); r[1] += 1; r[0] += int(wm.goal_answer(s) == a[0])
        print(seed, "goalrand" if rand else "goal", {k: round(v[0] / v[1], 3) for k, v in sorted(res.items())})
