"""Dami's opening-day runner (2026-10-17; the first sealed edition opens 10-21 UTC, user decision Q12): every life of an
edition through run_exam.exam_life, then the shared grader. Order only -- how the body learns and answers is run_exam.py's
(its fingerprint is checked here), and nothing in this file can change an answer.

  1. the edition: manifest.json's file hashes must match the lives' files (exposure.jsonl, items.jsonl); keys/ is not read
     until grading.
  2. the fingerprint: exam_fingerprint.py's output must equal --expect (the value in the shared STAGE1.md readiness line).
  3. the lives: one process per life (run_long.map_lives, D57: a dead worker fails loudly, memory gates new starts), the
     body's seed = the life's number (run_exam.life_seed); a finished life's answer file is kept and skipped on a rerun
     (written as .part, then renamed); after the queue piece's deadline (RIE_DEADLINE) no new life starts ("PIECE
     DEADLINE" on stderr ends the piece, and the next piece goes on).
  4. when every life has its answers: python3 -m exam.grade --edition EDITION --answers dami=OUT/answers --out
     OUT/grade.json, run in the shared repository (read only).
--smoke (development cases only, e.g. exam_dev/dev1000): each life is first replaced by run_exam.scrambled_copy (words
removed, nothing learnable -- the development cases are never training material); checks the mechanics only. The copy keeps
the life's folder name (OUT/smoke_input/life_XXX): conv-v1 seeds its position code with the folder's name (convert_life's
tag), so a copy under another name answers differently (10-17 dry run: "smoke_input" vs "life_000" -- top answer differed
in 158 of 1,092 rows; the same name gives the same rows).
    python3 run_exam_day.py EDITION_DIR OUT_DIR [WORKERS=4] --expect FP [--smoke] [--sessions N] [--shared DIR]"""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import run_exam as RE
import run_long as RL

HERE = os.path.dirname(os.path.abspath(__file__))
EXAM_PEAK_GB = float(os.environ.get("RIE_EXAM_GB", 3.0))   # a gate only; the recorded peaks replace it once lives finish


def check_edition(ed):
    with open(os.path.join(ed, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    lives = sorted(d for d in os.listdir(os.path.join(ed, "lives")) if d.startswith("life_"))
    bad = []
    for d in lives:
        for fn in ("exposure.jsonl", "items.jsonl"):
            rel = os.path.join("lives", d, fn)
            got = hashlib.sha256(open(os.path.join(ed, rel), "rb").read()).hexdigest()
            if m["files_sha256"].get(rel) != got:
                bad.append(rel)
    if bad or len(lives) != m["lives"]:
        sys.exit(f"EDITION CHECK FAILED: {len(bad)} file(s) differ from the manifest {bad[:3]}; lives {len(lives)} / "
                 f"manifest {m['lives']}")
    return m, lives


def fingerprint():
    out = subprocess.run([sys.executable, os.path.join(HERE, "exam_fingerprint.py")], capture_output=True, text=True,
                         check=True).stdout.split("\n")
    return out[0].strip()


def peak(task):
    """the expected peak of an exam life: 1.1 x the largest recorded (side files in OUT/runs), else RIE_EXAM_GB"""
    runs = os.path.join(task[2], "runs")
    seen = []
    for fn in os.listdir(runs) if os.path.isdir(runs) else []:
        try:
            with open(os.path.join(runs, fn)) as f:
                seen.append(json.load(f)["peak_rss_gb"])
        except Exception:
            pass
    return 1.1 * max(seen) if seen else EXAM_PEAK_GB


def one_life(task):
    life, life_dir, out, smoke, n = task
    ans = os.path.join(out, "answers", f"{life}.jsonl")
    if os.path.exists(ans):
        return "kept"
    dl = os.environ.get("RIE_DEADLINE")
    if dl and time.time() >= float(dl):
        print(f"PIECE DEADLINE: no new exam life after the piece's deadline ({life})", file=sys.stderr, flush=True)
        return None
    t_wall = time.time()
    src = RE.scrambled_copy(life_dir, os.path.join(out, "smoke_input", life)) if smoke else life_dir
    res = RE.exam_life(src, ans + ".part", seed=RE.life_seed(life_dir), n_sessions=n)
    res.update(life=life, seed=RE.life_seed(life_dir), smoke=smoke, wall_s=round(time.time() - t_wall, 1),
               host=os.uname().nodename, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    with open(os.path.join(out, "runs", f"{life}.json"), "w") as f:
        json.dump(res, f)
    os.replace(ans + ".part", ans)
    return "new"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
    smoke, n = "--smoke" in args, (int(opt("--sessions")) if "--sessions" in args else None)
    expect, shared = opt("--expect"), opt("--shared", "/home/user/shared-curriculum")
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in
                                                                        ("--expect", "--sessions", "--shared"))]
    ed, out = os.path.abspath(pos[0]), os.path.abspath(pos[1])
    workers = int(pos[2]) if len(pos) > 2 else 4
    m, lives = check_edition(ed)
    fp = fingerprint()
    print(f"edition {m['label']} generator {m['generator']} lives {m['lives']} seed_sha256 {m['seed_sha256'][:12]}; "
          f"Dami fingerprint {fp}; conv-v1 {RE.CV.VERSION}; smoke {smoke}; sessions {n or 'all'}; workers {workers}",
          flush=True)
    if expect is None or fp != expect:
        sys.exit(f"FINGERPRINT CHECK FAILED: {fp} != expected {expect} -- rewrite the STAGE1 readiness line first")
    if m["label"] == "edition" and smoke:
        sys.exit("--smoke is for development cases only")
    os.makedirs(os.path.join(out, "answers"), exist_ok=True)
    tasks = [(d, os.path.join(ed, "lives", d), out, smoke, n) for d in lives]
    got = RL.map_lives(tasks, workers, fn=one_life, peak=peak)
    print(f"lives: kept {got.count('kept')}, new {got.count('new')}, not started {got.count(None)}", flush=True)
    if None in got:
        sys.exit(0)
    for d in lives:
        with open(os.path.join(out, "runs", f"{d}.json")) as f:
            r = json.load(f)
        print(f"{d}: rows {r['rows']}/{r['due']} keys_match {r['keys_match']} unchanged {r['unchanged_while_answering']} "
              f"rehearse {r['rehearse_rounds']} recall {r['rehearse_recall']:.4f} cpu {r['cpu_s']:.0f}s "
              f"peak {r['peak_rss_gb']:.2f} GB", flush=True)
    g = subprocess.run([sys.executable, "-m", "exam.grade", "--edition", ed, "--answers", f"dami={out}/answers",
                        "--out", os.path.join(out, "grade.json")], cwd=shared, capture_output=True, text=True)
    print(g.stdout[-6000:], g.stderr[-3000:], sep="\n", flush=True)
    print(f"grader exit {g.returncode}")
