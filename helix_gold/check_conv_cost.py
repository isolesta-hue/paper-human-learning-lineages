"""Cost check of an exam-form life's FIRST step (the user, 10-16: "응, 시간과 메모리를 제봐도 좋아"): Dami's element memory
(Module A memorises every element at birth -- Learner3(w, seed, content=True, where="linear"), as run_long.life() and
run_elements.py) on the conv-v1 element table of the shared development seed's lives (exam_dev/dev1000), against her own
worlds. Only the element table is used: no sentence is heard, no item is read, no key is opened (development cases are never
training material). Measured per case in a fresh process: build time, birth (memorise) time and the peak memory after birth,
the element check's time and its
recall share (criterion 0's gate, data_ascii.element_check -- a memory check, not an exam score), peak memory.
    python3 check_conv_cost.py            (runs every case, one process each)
    python3 check_conv_cost.py CASE       (one case)
"""
import json
import os
import resource
import subprocess
import sys
import time

CASES = ["conv:life_000", "conv:life_001", "ascii-multi:0", "ascii-bi:0", "ascii-ten:0"]


def one(case):
    import torch
    torch.set_num_threads(1)
    kind, arg = case.split(":")
    t0 = time.process_time()
    if kind == "conv":
        import conv_v1 as CV
        w, _ = CV.convert_life(os.path.join(CV.EXAM_ROOT, "exam_dev", "dev1000", "lives", arg))
        seed = 0
    else:
        import run_long as RL
        w = RL.build_world(kind, int(arg))
        seed = int(arg)
    t1 = time.process_time()
    import data_ascii as A
    from integrated import Learner3
    me = Learner3(w, seed, content=True, where="linear")
    t2 = time.process_time()
    peak_birth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    res = A.element_check(w, me.memory, seed, linear_pos=me.where_decoder if hasattr(me, "where_decoder") else None)
    t3 = time.process_time()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return {"case": case, "V": int(w.V), "build_s": round(t1 - t0, 1), "birth_s": round(t2 - t1, 1),
            "check_s": round(t3 - t2, 1), "recall_all": round(res["all"], 4),
            "recall_by_type": {k: round(v, 4) for k, v in res.items() if k != "all"}, "peak_birth_gb": round(peak_birth, 2),
            "peak_gb": round(peak, 2)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(one(sys.argv[1]), ensure_ascii=False))
        sys.exit(0)
    print("=== exam-form cost check: element memory at birth (no sentence heard, no item read) ===")
    for c in CASES:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), c], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        line = (r.stdout.strip().splitlines() or [""])[-1]
        print(line if line.startswith("{") else f"{c}: FAILED {r.stderr.strip()[-400:]}", flush=True)
