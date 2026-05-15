"""Pytest configuration for local source-tree imports.

Purpose:
    The package uses a ``src`` layout. These tests should run both after an
    editable install and directly from a fresh checkout, so the repository's
    ``src`` directory is added to ``sys.path`` when needed.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
