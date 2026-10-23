"""Format checks of the W6b converter conv-v1 (W6_DESIGN.md 2-1), on the development seed's cases ONLY (exam_dev/dev1000;
never training material). Five checks:
  (1) the same records converted twice -> byte-identical (in this process and in a fresh process)
  (2) the records rebuilt from the converted moments == the original records (nothing lost, nothing added)
  (3) a fake body that knows the answers -> 1.000 in every format (through the shared grader's own rule)
  (4) a random body -> chance (the mean of 1 / number of candidates)
  (5) answering never changes the body or the converted world (fingerprints before / after)
    python3 check_conv.py [EDITION_DIR] [OUT_DIR]"""
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
from collections import defaultdict

import numpy as np

import conv_v1 as CV
from exam import grade as EG

ED = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else os.path.join(CV.EXAM_ROOT, "exam_dev", "dev1000")
OUT = sys.argv[2] if len(sys.argv) > 2 else "conv_check_answers"


class OracleBody:
    """knows every answer (reads the keys -- ONLY this fake body, for the format check)."""

    def __init__(self, keys, w):
        self.key = {k["id"]: k["answer"] for k in keys}
        self.w = w

    def score(self, q):
        a = self.key[q.id]
        if isinstance(a, list):
            a = f"{a[0]},{a[1]}"
        return [1.0 if k == str(a) else 0.0 for k, _ in q.cands]

    def state_probs(self, q, t):
        a = self.key[q.id]
        lab = a["possible"] if isinstance(a, dict) else a
        right = dict(dict(q.options)[lab])[t]
        others = list(self.w.pos_of) + [CV.ABSENT]
        return {s: (0.9 if s == right else 0.1 / (len(others) - 1)) for s in others}


class RandomBody:
    """random scores (its only state is the random stream, so its fingerprint is the empty learnable state)."""

    def __init__(self, seed, w):
        self.rng = np.random.default_rng(seed)
        self.states = list(w.pos_of) + [CV.ABSENT]

    def score(self, q):
        return list(self.rng.random(len(q.cands)))

    def state_probs(self, q, t):
        return dict(zip(self.states, self.rng.dirichlet(np.ones(len(self.states)))))


def fp(obj):
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


def run(body_name, make_body, lives, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    acc = defaultdict(lambda: [0, 0])
    chance = defaultdict(list)
    same = True
    for life in lives:
        w, qs = CV.convert_life(os.path.join(ED, "lives", life))
        keys = CV.load_jsonl(os.path.join(ED, "keys", f"{life}.jsonl"))
        items = {it["id"]: it for it in CV.load_jsonl(os.path.join(ED, "lives", life, "items.jsonl"))}
        body = make_body(keys, w)
        before = (fp(body.__dict__ if body_name == "oracle" else {}), CV.digest((w, qs)))
        rows = []
        for s in range(16):                          # a real body learns session s here; answering never learns
            for q in qs:
                if s in CV.sessions_to_answer(q.when):
                    rows.append(CV.answer(body, q, s))
        after = (fp(body.__dict__ if body_name == "oracle" else {}), CV.digest((w, qs)))
        same = same and before == after
        with open(os.path.join(out_dir, f"{life}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {(r["id"], r["after_session"]): r for r in rows}
        for k in keys:
            it = items[k["id"]]
            ok = EG.right(it, k, by.get((k["id"], EG.judged_session(k["when"]))))
            acc[it["format"]][0] += ok
            acc[it["format"]][1] += 1
            chance[it["format"]].append(1 / len(it["candidates"]))
    g = EG.grade(ED, out_dir)
    return acc, chance, same, g["missing"]


def main():
    lives = sorted(d for d in os.listdir(os.path.join(ED, "lives")) if d.startswith("life_"))
    print(f"=== conv-v1 format checks on {ED} ({len(lives)} lives; development cases only, never training) ===")
    ok_all = True
    for life in lives:
        d = os.path.join(ED, "lives", life)
        a, b = CV.digest(CV.convert_life(d)), CV.digest(CV.convert_life(d))
        c = subprocess.run([sys.executable, "-c", f"import conv_v1 as CV; print(CV.digest(CV.convert_life({d!r})))"],
                           capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        w, qs = CV.convert_life(d)
        expo = CV.load_jsonl(os.path.join(d, "exposure.jsonl"))
        items = CV.load_jsonl(os.path.join(d, "items.jsonl"))
        re_expo = CV.rebuild_exposure(w)
        re_items = [CV.rebuild_item(w, q) for q in qs]
        canon = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)
        r1 = a == b == c
        r2 = canon(re_expo) == canon(expo) and canon(re_items) == canon(items)
        n_mom = sum(len(cat["heard"]) for cat in w.categories)
        lines = sum(len(r["scene"]["lines"]) for r in expo)
        r2 = r2 and n_mom == len(expo) + lines
        bad_line = sum(1 for cat in w.categories for k_, t, _ in cat["heard"] if k_ == "conv line" and
                       not (len(t) == 2 and w.channel_of[t[0]] == 2 and w.channel_of[t[1]] == 0))
        print(f"{life}: {len(expo)} exposure lines -> {n_mom} moments ({len(expo)} sentences + {lines} scene lines; "
              f"line moments not [picture, at]: {bad_line}); {len(items)} items; elements {w.V} "
              f"(25 at + boundary + {len(w.word_of)} words + {len(w.id_of)} things; first met in the items: "
              f"{w.V - w.item_only_from})")
        print(f"  (1) twice the same: {r1} (digest {a[:16]}, fresh process {c[:16]})")
        print(f"  (2) records rebuilt from the moments == the originals: {r2}")
        ok_all = ok_all and r1 and r2 and bad_line == 0
    for name, mk in (("oracle", lambda keys, w: OracleBody(keys, w)), ("random", lambda keys, w: RandomBody(1000, w))):
        acc, chance, same, missing = run(name, mk, lives, os.path.join(OUT, name))
        print(f"  ({'3' if name == 'oracle' else '4'}) {name} body (grader's rule, judged answers; missing {missing}):")
        for f_ in ("blank", "slot", "choice", "pair"):
            a, n = acc[f_]
            ch = float(np.mean(chance[f_])) if f_ != "pair" else 0.5
            lo = ch - 2.58 * math.sqrt(ch * (1 - ch) / n)
            hi = ch + 2.58 * math.sqrt(ch * (1 - ch) / n)
            print(f"      {f_:7s} {a / n:.3f} ({a}/{n}); chance {ch:.3f} [99%: {lo:.3f}, {hi:.3f}]")
            if name == "oracle":
                ok_all = ok_all and a == n
            else:
                ok_all = ok_all and lo <= a / n <= hi
        print(f"  (5) answering left the body and the converted world unchanged ({name}): {same}")
        ok_all = ok_all and same and missing == 0
    print(f"ALL CHECKS PASS: {ok_all}")


if __name__ == "__main__":
    main()
