"""check (2026-10-18, D72 subject 3 ("ice")): the world builder gained the "ice" branch (data_ascii.py, integrated.py),
so the exam fingerprint changed; the exam answers must not. run_exam.exam_life with the new code on the same smoke folder,
2 sessions, == the opening-day runner's dry-run answers made with the fingerprinted code 0d23b5934238 (exam_day_dry/)."""
import json, os, sys
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
sys.path.insert(0, S + "repo")
import run_exam as RE
os.makedirs(S + "exam_plain5", exist_ok=True)
res = RE.exam_life(S + "exam_day_dry/smoke_input/life_000", S + "exam_plain5/life_000.jsonl", seed=0, n_sessions=2)
a = [json.loads(l) for l in open(S + "exam_plain5/life_000.jsonl")]
b = [json.loads(l) for l in open(S + "exam_day_dry/answers/life_000.jsonl")]
print("rows", len(a), len(b), "new code == fingerprinted code (0d23b5934238):", a == b)
