"""Q20 check (2026-10-21): with the target in the query (run_exam.READ_TARGET = True), the v1 exam life_000's first two
sessions are answered again by a plain run_exam.exam_life: every row of an item WITHOUT a target must equal the recorded v1
answer byte for byte; rows of items WITH a target may differ (report how many and which groups)."""
import gzip
import json
import os
import sys
from collections import Counter

import conv_v1 as CV
import run_exam as RE
sys.path.insert(0, "/home/user/shared-curriculum")
from exam.checks import group                      # noqa: E402
from exam.grade import load                        # noqa: E402

V1 = "/home/user/shared-curriculum/exam/v1"
life = "life_000"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q20_check", f"{life}.jsonl")
os.makedirs(os.path.dirname(out), exist_ok=True)
res = RE.exam_life(os.path.join(V1, "lives", life), out, seed=0, n_sessions=2)
new = [json.loads(l) for l in open(out)]
old = [json.loads(l) for l in gzip.open(f"exam_answers/v1/{life}.jsonl.gz", "rt") if json.loads(l)["after_session"] < 2]
_, qs = CV.convert_life(os.path.join(V1, "lives", life))
has_t = {q.id for q in qs if q.target is not None}
keys = {k["id"]: k for k in load(os.path.join(V1, "keys", f"{life}.jsonl"))}
key = lambda r: (r["id"], r["after_session"])
o = {key(r): r for r in old}
same_nt = sum(1 for r in new if r["id"] not in has_t and o.get(key(r)) == r)
n_nt = sum(1 for r in new if r["id"] not in has_t)
diff_t = Counter(group(keys[r["id"]])[1] for r in new if r["id"] in has_t and o.get(key(r)) != r)
n_t = sum(1 for r in new if r["id"] in has_t)
print(f"rows new {len(new)} old {len(old)}; items without a target identical {same_nt}/{n_nt}; with a target changed "
      f"{sum(diff_t.values())}/{n_t} {dict(diff_t)}; unchanged while answering {res['unchanged_while_answering']}")
print("Q20 regression:", "PASS" if same_nt == n_nt and len(new) == len(old) else "FAIL")
