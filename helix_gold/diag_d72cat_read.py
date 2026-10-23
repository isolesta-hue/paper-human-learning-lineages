"""Reads diag_d72cat.py's probe files (D72 diagnosis, 2026-10-18; Q14). Diagnosis only -- nothing here is a verdict.
Per direction (word = novel picture -> word, pic = word -> novel picture) and per part of the answer (finding 112's split):
  (1) right alone: the share of items where that part alone puts the right candidate strictly on top, by the item's k;
  (2) what the part rewards: a pooled least-squares fit over every candidate of every item in a k range, the part's score
      (centred within the item) on [the candidate is the right family] and [the candidate's family word heard k times,
      centred within the item]: b_family > 0 = the part points to the family, c_k < 0 = the part pushes a candidate down
      the more its family word was heard;
  (3) its weight: the part's mean spread over the four candidates (how much it can move the answer).
    python3 diag_d72cat_read.py DIR"""
import glob
import json
import os
import sys

import numpy as np

PARTS = ("full", "sim", "assoc", "pattern", "chunk", "side", "wm")
BUCKETS = [(0, 0), (1, 2), (3, 8), (9, 10 ** 6)]


def lab(lo, hi):
    return f"{lo}" if lo == hi else f"{lo}+" if hi > 100 else f"{lo}-{hi}"


def main(d):
    lives = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "seed*.json")))]
    print(f"lives {len(lives)}: " + ", ".join(f"seed {x['seed']} same as cached {x['same_as_cached']}, "
                                              f"{len(x['sessions'])} sessions" for x in lives))
    print(f"largest split error {max(s['split_err'] for x in lives for s in x['sessions']):.2e}")
    rows = [(x["seed"], i, r) for x in lives for i, s in enumerate(x["sessions"]) for r in s["rows"]]
    for kind in ("word", "pic"):
        rk = [(sd, i, r) for sd, i, r in rows if r["kind"] == kind]
        print(f"\n=== {kind}: {'novel picture -> word' if kind == 'word' else 'word -> novel picture'} "
              f"({len(rk)} item-sessions) ===")
        print("(1) right by this part alone, by the item's k: " + " | ".join(f"k {lab(*b)}" for b in BUCKETS))
        for p in PARTS:
            line = []
            for lo, hi in BUCKETS:
                v = [float(r[p][0] > max(r[p][1:])) for _, _, r in rk if p in r and lo <= r["cand_k"][0] <= hi]
                line.append(f"{np.mean(v):.3f} ({len(v)})" if v else "-")
            print(f"  {p:8s} " + " | ".join(line))
        print("(2) the part's score on [right family] and [candidate's k] (centred within the item), by the item's k range;"
              " (3) mean spread over the four candidates")
        for p in PARTS:
            out = []
            for lo, hi in ((0, 10 ** 6), (1, 10 ** 6)):
                X, y, sp = [], [], []
                for _, _, r in rk:
                    if p not in r or not (lo <= max(r["cand_k"]) <= hi):
                        continue
                    s = np.asarray(r[p], float)
                    if not np.all(np.isfinite(s)):
                        continue
                    k = np.asarray(r["cand_k"], float)
                    fam = np.array([1.0, 0, 0, 0])
                    y.extend(s - s.mean())
                    X.extend(np.stack([fam - fam.mean(), k - k.mean()], 1))
                    sp.append(s.max() - s.min())
                if len(y) < 8:
                    out.append("-")
                    continue
                coef = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)[0]
                out.append(f"b_family {coef[0]:+.3f}, c_k {coef[1]:+.4f} per hearing, spread {np.mean(sp):.3f}")
            print(f"  {p:8s} all items: {out[0]} || items with a heard candidate: {out[1]}")
    # the answer's top choice: which candidate wins, by its k rank among the four (most heard = rank 0)
    print("\n=== which candidate the full answer picks (word direction, items with distinct k): its k rank, 0 = most heard ===")
    for kind in ("word", "pic"):
        ranks = []
        for _, _, r in rows:
            if r["kind"] != kind:
                continue
            k = np.asarray(r["cand_k"])
            if len(set(k.tolist())) < 4:
                continue
            win = int(np.argmax(r["full"]))
            ranks.append(int((k > k[win]).sum()))
        if ranks:
            h = np.bincount(ranks, minlength=4) / len(ranks)
            print(f"  {kind}: picks the most heard {h[0]:.3f}, 2nd {h[1]:.3f}, 3rd {h[2]:.3f}, least heard {h[3]:.3f} "
                  f"({len(ranks)} items)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "diag_d72cat")
