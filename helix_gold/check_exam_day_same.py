"""check: the opening-day runner's answers == a plain process's run_exam.exam_life on the same input folder (same name)"""
import json, os, sys
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
sys.path.insert(0, S + "repo")
import run_exam as RE
os.makedirs(S + "exam_plain3", exist_ok=True)
res = RE.exam_life(S + "exam_day_dry/smoke_input/life_000", S + "exam_plain3/life_000.jsonl", seed=0, n_sessions=2)
a = [json.loads(l) for l in open(S + "exam_plain3/life_000.jsonl")]
b = [json.loads(l) for l in open(S + "exam_day_dry/answers/life_000.jsonl")]
print("rows", len(a), len(b), "runner == plain:", a == b)
