"""Exam v1 grading day (ledger Q16; the user's decisions 10-18/10-19: the keys stay sealed outside the repositories until all
three models' answers are in; then the keys and the seed phrase are published together and every answer set is graded with
the shared grader, unchanged). Written 10-20 while waiting for Ara; the phrase is given on the command line at grading time
only (never written in this file or its outputs). Each step must pass before the next:
  1. the phrase: sha256 of str(seed_of(phrase)) == the manifest's seed_sha256 (exam/edition.py hashes the int seed's decimal
     text, not the phrase itself -- fixed on grading day after the first run stopped here); the generator's code has the manifest's code hashes (the
     code did not change since the opening); the edition regenerated from the phrase (exam.edition, into a scratch folder)
     has every file hash of the manifest -- lives and keys -- so anyone can redo the check;
  2. the sealed keys: every scratchpad/sealed_keys_v1/keys/life_XXX.jsonl has the manifest's sha256 -> copied into shared
     exam/v1/keys/ (the publication; committed by hand after this script);
  3. the answer sets: Dami, Sia, Ara whole / alone (Ara's branch, pioneer/exam_answers/v1/{whole,alone}/*.jsonl.gz, read
     with git show and unzipped into scratchpad/exam_v1_answers/ara/), the six simple (M C N B L T) and four learned (P V MLP
     TF) baselines -- 30 files each; for the three models, every life's (id, after_session) rows are compared with what
     the edition asks (each item at each of its sessions) and the missing / extra rows are reported (the grader counts a
     missing answer as wrong);
  4. python3 -m exam.grade on all sets (report-only list exam_spec/report_only.json) -> OUT/grade.txt, OUT/grade.json;
  5. the score table -> OUT/score_table.md: per criterion line, accuracy and pass per answer set.
Note: Ara's answer manifest names the edition by sha256 of the manifest's sorted JSON dump (78e85ef3868d); the manifest
file's own sha256 is 4ac9e0977857 -- the same edition (checked 10-20).
    python3 grade_v1.py --phrase "..." --out OUT_DIR [--ara-ref REF] [--skip-regen]"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

SP = "/tmp/claude-0/-home-user/23cc98b2-a804-5343-bef4-6479ad5f8346/scratchpad"
SHARED = "/home/user/shared-curriculum"
ED = os.path.join(SHARED, "exam", "v1")
REPO = os.path.join(SP, "repo")
SIMPLE, LEARNED = ("M", "C", "N", "B", "L", "T"), ("P", "V", "MLP", "TF")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def step(msg):
    print(f"\n### {msg}", flush=True)


def due_rows(life):
    """(id, after_session) every row the edition asks of a life: "every" -> 0..15, "after_session_k" -> k, "end" -> 15."""
    out = set()
    for line in open(os.path.join(ED, "lives", life, "items.jsonl"), encoding="utf-8"):
        it = json.loads(line)
        w = it["when"]
        ss = range(16) if w == "every" else [int(w.rsplit("_", 1)[1])] if w.startswith("after_session_") else [15]
        out.update((it["id"], s) for s in ss)
    return out


def score_table(g, names):
    """markdown: per criterion, PASS per answer set, then each verdict line's accuracy (or ratio) with a check mark when it
    passes its line."""
    lines = ["| 기준 | 줄 | " + " | ".join(names) + " |", "|" + "---|" * (2 + len(names))]
    for c in range(10):
        c_ = str(c)
        lines.append(f"| **C{c}** | **통과** | " + " | ".join("**통과**" if g[n]["passed"][c_] else "–" for n in names) + " |")
        for k, d in g[names[0]]["verdicts"][c_].items():
            if not isinstance(d, dict) or k.endswith("(reported)"):
                continue
            cells = []
            for n in names:
                x = g[n]["verdicts"][c_][k]
                v = x.get("acc", x.get("ratio"))
                cells.append(("–" if v is None or v != v else f"{v:.3f}") + (" ✓" if x.get("pass") else ""))
            th = d.get("line")
            lines.append(f"| C{c} | {k}{f' (선 {th})' if th is not None else ''} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ara-ref", default="FETCH_HEAD")
    ap.add_argument("--skip-regen", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    man = json.load(open(os.path.join(ED, "manifest.json"), encoding="utf-8"))
    lives = sorted(os.listdir(os.path.join(ED, "lives")))
    assert len(lives) == man["lives"] == 30, lives

    step("1. the phrase and the edition")
    sys.path.insert(0, SHARED)
    from exam.common import seed_of                 # the generator's own: the phrase -> an int seed, whose decimal text
    ok_seed = hashlib.sha256(str(seed_of(a.phrase)).encode()).hexdigest() == man["seed_sha256"]   # the manifest hashes
    print(f"sha256(str(seed_of(phrase))) == seed_sha256 (the generator's rule, exam/edition.py): {ok_seed}")
    code_dir = os.path.join(SHARED, "exam")
    code_ok = {k: (sha(os.path.join(SHARED, "exam_spec", k)) if k == "closed_words.json" else sha(os.path.join(code_dir, k)))
               == v for k, v in man["code_sha256"].items()}
    print(f"generator code unchanged since the opening: {all(code_ok.values())} ({sum(code_ok.values())}/{len(code_ok)})")
    if not (ok_seed and all(code_ok.values())):
        sys.exit("step 1 failed")
    if not a.skip_regen:
        tmp = tempfile.mkdtemp(prefix="regen_v1_", dir=SP)
        subprocess.run([sys.executable, "-m", "exam.edition", "--phrase", a.phrase, "--lives", "30", "--out", tmp],
                       cwd=SHARED, check=True, capture_output=True)
        regen = json.load(open(os.path.join(tmp, "manifest.json"), encoding="utf-8"))
        same = regen["files_sha256"] == man["files_sha256"]
        print(f"regenerated from the phrase: {len(regen['files_sha256'])} files, every hash equal to the manifest: {same}")
        shutil.rmtree(tmp)
        if not same:
            sys.exit("step 1 failed: the phrase does not regenerate the edition")

    step("2. the sealed keys -> exam/v1/keys")
    sealed = os.path.join(SP, "sealed_keys_v1", "keys")
    bad = [lf for lf in lives if sha(os.path.join(sealed, f"{lf}.jsonl")) != man["files_sha256"][f"keys/{lf}.jsonl"]]
    print(f"sealed key files matching the manifest: {30 - len(bad)}/30")
    if bad:
        sys.exit(f"step 2 failed: {bad}")
    os.makedirs(os.path.join(ED, "keys"), exist_ok=True)
    for lf in lives:
        shutil.copyfile(os.path.join(sealed, f"{lf}.jsonl"), os.path.join(ED, "keys", f"{lf}.jsonl"))
    assert all(sha(os.path.join(ED, "keys", f"{lf}.jsonl")) == man["files_sha256"][f"keys/{lf}.jsonl"] for lf in lives)
    print("copied into exam/v1/keys (30 files, hashes checked)")

    step("3. the answer sets")
    ara = os.path.join(SP, "exam_v1_answers", "ara")
    for part in ("whole", "alone"):
        d = os.path.join(ara, part)
        os.makedirs(d, exist_ok=True)
        for lf in lives:
            blob = subprocess.run(["git", "-C", REPO, "show", f"{a.ara_ref}:pioneer/exam_answers/v1/{part}/{lf}.jsonl.gz"],
                                  capture_output=True, check=True).stdout
            open(os.path.join(d, f"{lf}.jsonl"), "wb").write(gzip.decompress(blob))
    sets = {"dami": os.path.join(SP, "exam_v1_dami", "answers"),
            "sia": os.path.join(SP, "exam_v1_answers", "sia", "exam_answers", "v1"),
            "ara_whole": os.path.join(ara, "whole"), "ara_alone": os.path.join(ara, "alone")}
    sets.update({b: os.path.join(SP, "exam_v1_baselines", "simple", b) for b in SIMPLE})
    sets.update({b: os.path.join(SP, "exam_v1_baselines", "learned", b) for b in LEARNED})
    due = {lf: due_rows(lf) for lf in lives}
    for name, d in sets.items():
        files = sorted(glob.glob(os.path.join(d, "life_*.jsonl")))
        if len(files) != 30:
            sys.exit(f"step 3 failed: {name} has {len(files)} files")
        if name in ("dami", "sia", "ara_whole", "ara_alone"):
            miss = extra = dup = 0
            for lf in lives:
                rows = [(r["id"], int(r["after_session"])) for r in map(json.loads, open(os.path.join(d, f"{lf}.jsonl")))]
                got = set(rows)
                dup += len(rows) - len(got)
                miss += len(due[lf] - got)
                extra += len(got - due[lf])
            print(f"{name}: 30 files, rows asked {sum(map(len, due.values()))}, missing {miss}, extra {extra}, duplicated {dup}")
        else:
            print(f"{name}: 30 files")

    step("4. the grader")
    cmd = [sys.executable, "-m", "exam.grade", "--edition", "exam/v1", "--report-only", "exam_spec/report_only.json",
           "--out", os.path.join(out, "grade.json"), "--answers"] + [f"{n}={d}" for n, d in sets.items()]
    res = subprocess.run(cmd, cwd=SHARED, capture_output=True, text=True)
    open(os.path.join(out, "grade.txt"), "w").write(res.stdout + res.stderr)
    print(f"grader exit {res.returncode}; report OUT/grade.txt ({len(res.stdout.splitlines())} lines)")
    if res.returncode != 0:
        sys.exit("step 4 failed:\n" + res.stderr[-2000:])

    step("5. the score table")
    g = json.load(open(os.path.join(out, "grade.json"), encoding="utf-8"))
    text = score_table(g, list(sets))
    open(os.path.join(out, "score_table.md"), "w").write(text)
    print(text)
    print("exit 0")


if __name__ == "__main__":
    main()
