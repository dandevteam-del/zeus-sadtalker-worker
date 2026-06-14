#!/usr/bin/env python3
"""Rewrite removed NumPy aliases (np.float/np.int/np.bool/...) to the Python builtins
across a source tree. Word-boundary aware so np.float64 / np.int32 are left intact.
SadTalker's vendored code (src/face3d/util/my_awing_arch.py et al.) still uses the
np.float alias removed in NumPy>=1.24, which crashes 3DMM landmark extraction.
"""
import os
import re
import sys

ALIASES = ("float", "int", "bool", "object", "str", "complex")
PAT = re.compile(r"\bnp\.(" + "|".join(ALIASES) + r")\b(?![0-9_])")

root = sys.argv[1] if len(sys.argv) > 1 else "."
changed = 0
for dirpath, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        path = os.path.join(dirpath, fn)
        src = open(path, encoding="utf-8", errors="ignore").read()
        new = PAT.sub(lambda m: m.group(1), src)
        if new != src:
            open(path, "w", encoding="utf-8").write(new)
            changed += 1
            print("patched np-alias:", path)
print(f"np-alias patch done: {changed} file(s) updated under {root}")
