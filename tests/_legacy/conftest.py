from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_SRC_PATHS = (
    ROOT / "packages" / "wabizabi" / "src",
    ROOT / "packages" / "wazi" / "src",
)

for path in reversed(WORKSPACE_SRC_PATHS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
