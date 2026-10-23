"""Exam v1 reproduction condition (2026-10-20; Sia's sibling note of 10-19: the same body, seed and code answer differently
with OpenBLAS at 4 threads and at 1). Dami's v1 answers were made with torch at 1 thread per process (run_exam.py) and
numpy's OpenBLAS at its default (OMP_NUM_THREADS / OPENBLAS_NUM_THREADS unset, 4 CPUs). Read-only check: the first
sessions of one life are run again twice, in two processes -- (a) the exam's condition, (b) OpenBLAS, OpenMP and MKL at
1 thread, set in the environment before Python starts (so before numpy is imported) -- each into a scratch folder, and every
row of those sessions is compared with the recorded answer file: exact scores, and the top answer. The code must be the
exam's (fingerprint checked first); the edition is read only and nothing is written next to the answers.
    python3 check_exam_threads.py EDITION_DIR ANSWERS_DIR OUT_DIR [LIFE=life_000] [SESSIONS=2]"""
import json
import os
import subprocess
import sys

COND = {"exam": {}, "one": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}
FP = "0551dd3a140d"


def child(life_dir, out_path, n):
    import numpy as np
    import run_exam as RE
    import torch
    print(f"numpy {np.__version__}, torch {torch.__version__}, torch threads {torch.get_num_threads()}, "
          f"env {[(k, os.environ.get(k)) for k in COND['one']]}", flush=True)
    res = RE.exam_life(life_dir, out_path, seed=RE.life_seed(life_dir), n_sessions=n)
    print(json.dumps(res), flush=True)


def load(path, n):
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["after_session"] < n:
            rows[(r["id"], r["after_session"])] = r["scores"]
    return rows


def compare(ref, got):
    same_keys = set(ref) == set(got)
    exact = sum(ref[k] == got[k] for k in ref if k in got)
    top = sum(max(ref[k], key=ref[k].get) == max(got[k], key=got[k].get) for k in ref if k in got)
    diff = max((abs(ref[k][c] - got[k][c]) for k in ref if k in got for c in ref[k]), default=0.0)
    return same_keys, len(ref), exact, top, diff


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        sys.exit(0)
    edition, answers, out = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
    life = sys.argv[4] if len(sys.argv) > 4 else "life_000"
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    fp = subprocess.run([sys.executable, "exam_fingerprint.py"], capture_output=True, text=True).stdout.split()[0]
    if fp != FP:
        sys.exit(f"fingerprint {fp} != the exam's {FP}: not the exam's code")
    life_dir = os.path.join(edition, "lives", life)
    procs = {}
    for name, env in COND.items():
        os.makedirs(os.path.join(out, name), exist_ok=True)
        e = {k: v for k, v in os.environ.items() if k not in COND["one"]}
        e.update(env)
        log = open(os.path.join(out, name, "log.txt"), "w")
        procs[name] = subprocess.Popen([sys.executable, __file__, "--child", life_dir,
                                        os.path.join(out, name, f"{life}.jsonl"), str(n)], env=e, stdout=log,
                                       stderr=subprocess.STDOUT)
    for name, p in procs.items():
        if p.wait() != 0:
            sys.exit(f"{name}: child failed, see {out}/{name}/log.txt")
    ref = load(os.path.join(answers, f"{life}.jsonl"), n)
    print(f"=== {life}, sessions 0-{n - 1}: {len(ref)} recorded rows; fingerprint {fp} ===")
    for name in COND:
        print(open(os.path.join(out, name, "log.txt")).readline().strip())
        same_keys, total, exact, top, diff = compare(ref, load(os.path.join(out, name, f"{life}.jsonl"), n))
        print(f"{name}: same rows {same_keys}; scores exactly equal {exact}/{total}; top answer equal {top}/{total}; "
              f"largest score difference {diff:.3g}")
    print("exit 0")
