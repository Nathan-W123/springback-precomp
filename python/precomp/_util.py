"""Small shared helpers: the error type, JSON input/output and hashing.

Nothing here knows about forming; it exists so that every module reads and
writes JSON the same way (UTF-8, sorted keys, finite numbers only) and raises
the same kind of error.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

PathLike = Union[str, "os.PathLike[str]"]


class PrecompError(RuntimeError):
    """Raised when a file, a run or a data set is missing or inconsistent.

    Invalid argument values raise `ValueError` instead (the Python
    convention). Either way the message names the offending file, key or
    parameter; nothing in the package silently substitutes a default for a
    bad value.
    """


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays (recursively) to plain JSON types.

    Non-finite floats are refused: JSON has no NaN or infinity, and a NaN
    written as a bare token would make the file unreadable by strict parsers
    such as SparLab's.
    """
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            raise PrecompError(f"cannot write the non-finite number {v} to JSON")
        return v
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(doc: Any) -> str:
    """The canonical text of a JSON document: sorted keys, 2-space indent.

    Used both for files and for hashing, so two equal documents always give
    byte-identical files and the same content hash.
    """
    return json.dumps(to_jsonable(doc), sort_keys=True, indent=2, allow_nan=False) + "\n"


def write_json(path: PathLike, doc: Any) -> Path:
    """Write `doc` as canonical JSON (see `canonical_json`) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(doc), encoding="utf-8")
    return path


def read_json(path: PathLike) -> Dict[str, Any]:
    """Read a JSON document, with an error naming the file when it is absent."""
    path = Path(path)
    if not path.is_file():
        raise PrecompError(f"{path} does not exist")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise PrecompError(f"{path} is not valid JSON: {exc}") from exc


def sha256_bytes(*chunks: bytes) -> str:
    """Hex SHA-256 of the concatenation of `chunks`, each length-prefixed.

    The length prefix makes the hash of ("ab", "c") differ from ("a", "bc").
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(len(chunk).to_bytes(8, "little"))
        h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> str:
    """Mirror the file-name sanitisation of the C++ ResultWriter.

    Alphanumerics, '-' and '_' are kept, everything else becomes '_'; an empty
    name becomes 'unnamed' (the same rule as `sparlab_viz.loaders._safe`).
    """
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name) or "unnamed"


def require_positive(name: str, value: float) -> float:
    """Return `value` as float; ValueError naming `name` unless finite and > 0."""
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return v


def require_nonnegative(name: str, value: float) -> float:
    """Return `value` as float; ValueError naming `name` unless finite and >= 0."""
    v = float(value)
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return v
