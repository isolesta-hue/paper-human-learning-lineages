"""Which learning settings win in the small society (diagnostic for finding 43, 8 seeds)."""
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch

import data_v2 as D
import run_society as S


def one(seed):
    torch.set_num_threads(1)
    d = D.build(S.LEVEL, seed)
    return {c: S.life(c, d, seed) for c in ["alone", "strategy", "all+near"]}


if __name__ == "__main__":
    with Pool(4) as pool:
        res = pool.map(one, range(8))
    per = defaultdict(lambda: defaultdict(list))
    for r in res:
        for step, st in enumerate(r["alone"]):
            for strat, a in st["each"]:
                per[strat][step].append(a)
    print("alone: test accuracy by own settings (lr, replay), steps s1..s8 c1 c2")
    for strat in sorted(per):
        print(strat, " ".join(f"{np.mean(per[strat][i]):.3f}" if per[strat][i] else "  -  " for i in range(10)))
    for cond in ["strategy", "all+near"]:
        print(f"\n{cond}: settings held after each step (seed: steps)")
        for k, r in enumerate(res):
            print(k, [sorted({s for s, _ in st["each"]}) for st in r[cond]][:4], "... end", sorted({s for s, _ in r[cond][-1]["each"]}))
