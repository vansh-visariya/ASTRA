"""Apply mypy type annotation fixes across ASTRA codebase."""
import re
from pathlib import Path

REPLACEMENTS = {
    # database.py
    "src/astra/app/database.py": [
        (r"from datetime import datetime",
         "from datetime import datetime\nfrom typing import Any, Generator, Optional"),
        (r"group_id: str = None", "group_id: Optional[str] = None"),
        (r"user_id: int = None", "user_id: Optional[int] = None"),
        (r"config: dict = None", "config: Optional[dict] = None"),
        (r"join_token: str = None", "join_token: Optional[str] = None"),
        (r"created_by: int = None", "created_by: Optional[int] = None"),
        (r"(\b    )client_id: str = None", r"\1client_id: Optional[str] = None"),
        (r"(\b    )accuracy: float = None", r"\1accuracy: Optional[float] = None"),
        (r"(\b    )loss: float = None", r"\1loss: Optional[float] = None"),
        (r"(\b    )num_clients: int = None", r"\1num_clients: Optional[int] = None"),
        (r"(\b    )metadata: dict = None", r"\1metadata: Optional[dict] = None"),
    ],
    # training_group.py
    "src/astra/app/training_group.py": [
        (r"from dataclasses import dataclass, field",
         "from dataclasses import dataclass, field\nfrom typing import Any, Optional"),
        (r"client_info: dict = None", "client_info: Optional[dict] = None"),
    ],
    # group_manager.py
    "src/astra/app/group_manager.py": [
        (r"from typing import Any, Callable, Dict, List, Optional, Set",
         "from typing import Any, Callable, Dict, List, Optional, Set, Union"),
        (r"group_id: str = None", "group_id: Optional[str] = None"),
        (r"details: dict = None", "details: Optional[dict] = None"),
        (r"event_type: str = None", "event_type: Optional[str] = None"),
        (r"client_info: dict = None", "client_info: Optional[dict] = None"),
    ],
    # notifications.py
    "src/astra/app/notifications.py": [
        (r"from typing import Any",
         "from typing import Any, Optional"),
        (r"reason: str = None", "reason: Optional[str] = None"),
    ],
    # robust.py
    "src/astra/core/aggregation/robust.py": [
        (r"from typing import Any",
         "from typing import Any, Optional"),
        (r"dataset_sizes: list\[int\] = None", "dataset_sizes: Optional[list[int]] = None"),
        # Fix type annotations on numpy arrays
        (r"similarity_scores: list\[Any\] =", "similarity_scores: list[float] ="),
    ],
    # routes/models.py
    "src/astra/app/routes/models.py": [
        (r"from typing import Optional",
         "from typing import Optional, Union"),
        (r"version: int = None", "version: Optional[int] = None"),
    ],
    # config.py - Path type fix
    "src/astra/core/config.py": [
        (r"from typing import Any, Dict, List, Optional, Union",
         "from typing import Any, Dict, List, Optional, Union"),
    ],
}


def strip_bom(p: Path) -> str:
    raw = p.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    return raw.decode("utf-8")


def apply_fixes():
    base = Path(__file__).parent.parent
    for relpath, fixes in REPLACEMENTS.items():
        fpath = base / relpath
        if not fpath.exists():
            print(f"SKIP (not found): {relpath}")
            continue
        content = strip_bom(fpath)
        changed = False
        for pattern, replacement in fixes:
            new_content = re.sub(pattern, replacement, content, count=1)
            if new_content != content:
                content = new_content
                changed = True
            else:
                print(f"  NO MATCH: {pattern[:60]} in {relpath}")
        if changed:
            fpath.write_text(content, encoding="utf-8")
            print(f"FIXED: {relpath} ({len(fixes)} replacements checked)")
        else:
            print(f"SKIP (no changes): {relpath}")


if __name__ == "__main__":
    apply_fixes()
