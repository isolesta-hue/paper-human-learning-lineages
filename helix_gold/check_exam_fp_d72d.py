"""check (2026-10-22, D72 subject 4 ("next"), then D80 v2 ("@seq2")): the world builder gained the "next" branch and the
told-only walk option (data_ascii.py, integrated.py), so
the exam-code fingerprint changed (970dacab9e36 -> the new one printed by exam_fingerprint.py); the exam answers must not.
run_exam.exam_life with the new code on the v1 exam life_000, 2 sessions, == the answers made with the fingerprinted code
970dacab9e36 for the Q20 check (q20_check/life_000.jsonl, the same life, seed and sessions)."""
import json
import os
import sys

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
sys.path.insert(0, S + "repo")
import run_exam as RE

out = S + "fp_d72d_check/life_000.jsonl"
os.makedirs(os.path.dirname(out), exist_ok=True)
RE.exam_life("/home/user/shared-curriculum/exam/v1/lives/life_000", out, seed=0, n_sessions=2)
a = [json.loads(l) for l in open(out)]
b = [json.loads(l) for l in open(S + "q20_check/life_000.jsonl")]
print("rows", len(a), len(b), "new code == fingerprinted code (970dacab9e36):", a == b)
