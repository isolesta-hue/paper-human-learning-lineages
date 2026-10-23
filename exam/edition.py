"""Write an edition (or a dev instance) to files (exam_spec/GENERATOR.md 7).
  python3 -m exam.edition --dev 1000 --lives 2 --out exam_dev/dev1000     (development: integer seeds 1000-1999 only)
  python3 -m exam.edition --phrase "<the user's seed phrase>" --out exam/v1  (opening day only)
Model-facing files go to lives/ (exposure.jsonl, items.jsonl), answer keys to keys/ (opened only for grading); the manifest
holds the code hashes and every written file's hash."""
import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np

from . import GEN_VERSION
from .common import seed_of
from .items import Items
from .life import Life


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def code_hashes():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            out[name] = hashlib.sha256(open(os.path.join(here, name), "rb").read()).hexdigest()
    cw = os.path.join(here, "..", "exam_spec", "closed_words.json")
    out["closed_words.json"] = hashlib.sha256(open(cw, "rb").read()).hexdigest()
    return out


def build(seed, n_lives, out_dir, label):
    files = {}
    for i in range(n_lives):
        L = Life(seed, i)
        items, keys = Items(L).build()
        for rel, rows in ((os.path.join("lives", f"life_{i:03d}", "exposure.jsonl"), L.records),
                          (os.path.join("lives", f"life_{i:03d}", "items.jsonl"), items),
                          (os.path.join("keys", f"life_{i:03d}.jsonl"), keys)):
            write_jsonl(os.path.join(out_dir, rel), rows)
            files[rel] = hashlib.sha256(open(os.path.join(out_dir, rel), "rb").read()).hexdigest()
    manifest = {"generator": GEN_VERSION, "label": label, "lives": n_lives,
                "seed_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
                "code_sha256": code_hashes(), "files_sha256": files,
                "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", type=int)
    g.add_argument("--phrase", type=str)
    ap.add_argument("--lives", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.dev is not None and not 1000 <= a.dev <= 1999:
        sys.exit("development seeds are 1000-1999 only")
    seed = a.dev if a.dev is not None else seed_of(a.phrase)
    m = build(seed, a.lives, a.out, f"dev{a.dev}" if a.dev is not None else "edition")
    print(json.dumps({k: m[k] for k in ("generator", "label", "lives")}, ensure_ascii=False))
