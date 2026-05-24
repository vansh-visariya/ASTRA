from pathlib import Path
p = Path(__file__).parent.parent / "src/astra/core/models/hf_models.py"
text = p.read_bytes().decode("utf-8-sig")
lines = text.split("\n")
total_open = 0
for i, line in enumerate(lines):
    total_open += line.count("(") - line.count(")")
print(f"Total lines: {len(lines)}")
print(f"Paren balance: {total_open}")
for i, line in enumerate(lines):
    if line.count("(") != line.count(")"):
        print(f"Line {i+1}: +{line.count('(')} -{line.count(')')}: {line[:120]}")
