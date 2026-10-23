"""Dami's exam-taking code fingerprint (readiness, 2026-10-17): sha256 over every module of THIS repository that
run_exam.py loads (conv-v1, the body, Module A with rehearsal, ...), in name order; the first 12 hex digits are the
fingerprint written in the shared STAGE1.md readiness line. The shared exam package (read-only) is not included."""
import hashlib
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import run_exam  # noqa: E402,F401

mods = sorted((n, m.__file__) for n, m in list(sys.modules.items())
              if not n.startswith("__") and getattr(m, "__file__", None) and os.path.isabs(m.__file__) and os.path.dirname(m.__file__) == here)
h = hashlib.sha256()
for n, f in mods:
    h.update(n.encode())
    h.update(open(f, "rb").read())
print(h.hexdigest()[:12])
print(" ".join(n for n, _ in mods))
