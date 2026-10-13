"""Make `precomp` importable when a script is run straight from the repo.

The package is deliberately not installed: `python3 python/scripts/<name>.py`
should work in a fresh clone with nothing but the scientific stack present.
"""

from __future__ import annotations

import os
import sys

_PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

REPO_ROOT = os.path.dirname(_PYTHON_DIR)
