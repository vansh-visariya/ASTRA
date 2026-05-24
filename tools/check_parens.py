"""Check paren balance in hf_models.py"""
from pathlib import Path

p = Path(__file__).parent.parent / "src/astra/core/models/hf_models.py"
text = p.read_bytes().decode("utf-8-sig")
lines = text.split("\n")

open_count = 0
for i, line in enumerate(lines):
    open_count += line.count("(") - line.count(")")
    print(f"{i+1:4d} [{open_count:3d}] {line[:100]}")
