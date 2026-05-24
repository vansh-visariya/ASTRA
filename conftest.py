"""
Root conftest.py — adds src/ to sys.path so that `import astra` resolves
to src/astra/ for all pytest runs.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
