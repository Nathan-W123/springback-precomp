"""Helpers shared by the workflow scripts (springback_fine_benchmark.py,
springback_fine_ml.py, combined_rim_ml.py): the deviation of a formed part by
region, a run's cost and physics checks, the design's train / calibration /
test split, JSON output and the software versions of a run.

Every number these helpers report comes from a SparLab simulation.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import _bootstrap  # noqa: F401  (puts python/ on sys.path)

import numpy as np
import pandas as pd

MM = 1e-3
LABEL = "SparLab simulation"
REPO = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def jdump(path: Path, doc: Any) -> None:
    """Write `doc` as JSON (sorted keys), atomically."""
    from precomp._util import to_jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(doc), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def split_of(point_id: str, cfg: Dict[str, Any]) -> str:
    """The split of a design point ("<family>-s<seed>-<index>"): the first
    `test_per_family` indices of a family are TEST, then every
    `calibration_every`-th is CALIBRATION, the rest TRAIN."""
    i = int(point_id.rsplit("-", 1)[1])
    if i < cfg["test_per_family"]:
        return "test"
    j = i - cfg["test_per_family"]
    return "calibration" if j % cfg["calibration_every"] == cfg["calibration_every"] - 1 \
        else "train"


def deviation(formed, target, band: float) -> Dict[str, float]:
    """Deviation of `formed` from `target` [mm]: vertical RMS / max / bias and
    normal RMS over the part, and the vertical RMS / bias / max of the upper
    band (the part less than `band` [m] deep, next to the clamp), the deep
    region and the flange, with the upper band's share of the squared error."""
    from precomp.metrology import (flange_mask, metrics, part_mask, signed_deviation,
                                   vertical_deviation)

    part = part_mask(target)
    v = vertical_deviation(formed, target)
    n = signed_deviation(formed, target)
    mv, mn = metrics(v, part), metrics(n, part)
    upper = part & (target.z > -band)
    out = {"vert_rms_mm": mv["rms"] / MM, "vert_max_mm": mv["max_abs"] / MM,
           "vert_bias_mm": mv["bias"] / MM, "normal_rms_mm": mn["rms"] / MM}
    tot = float(np.sum(v.z[part & v.mask] ** 2))
    for reg, m in {"upper_band": upper, "deep": part & ~upper,
                   "flange": flange_mask(target)}.items():
        sel = m & v.mask
        out[f"{reg}_rms_mm"] = float(np.sqrt(np.mean(v.z[sel] ** 2))) / MM
        out[f"{reg}_bias_mm"] = float(np.mean(v.z[sel])) / MM
        out[f"{reg}_max_mm"] = float(np.abs(v.z[sel]).max()) / MM
    out["upper_band_share"] = float(np.sum(v.z[upper & v.mask] ** 2)) / tot
    return out


def physics(res) -> Dict[str, Any]:
    """The run's cost and its physics checks, from summary.json and
    tool_forces.csv."""
    s = res.summary
    t = s.get("timing", {})
    steps = {st["name"]: st for st in s["steps"]}
    out: Dict[str, Any] = {
        "completed": bool(s["completed"]), "termination": s["termination"],
        "runtime_s": float(s["runtime_s"]), "increments": t.get("increments"),
        "iterations": t.get("iterations"), "cuts": t.get("cuts"),
        "max_plastic_strain": max(st.get("max_plastic_strain", 0.0) for st in s["steps"]),
        "warnings": len(s.get("warnings", [])),
        "warning_text": " | ".join(w[:160] for w in s.get("warnings", [])),
        "steps": ",".join(st["name"] for st in s["steps"]),
        "unload_clamp_reaction_N": steps.get("unload", {}).get("reaction_norm_N"),
        "release_reaction_N": steps.get("release", {}).get("reaction_norm_N"),
        "release_reference_force_N": steps.get("release", {}).get("reference_force_N"),
    }
    if "rim_unload" in steps:
        out["rim_unload_clamp_reaction_N"] = steps["rim_unload"].get("reaction_norm_N")
    return out


def tool_rows(case: str, res) -> List[Dict[str, Any]]:
    """Per tool of a run: contact and force statistics (tool_forces.csv)."""
    f = res.forming_forces()
    rows = []
    for tool in res.summary.get("tools", []):
        name = tool["name"]
        g = f[f["tool"] == name]
        row = {"case": case, "tool": name, "shape": tool.get("shape"),
               "max_active_nodes": tool.get("max_active_nodes"),
               "max_penetration_um": 1e6 * float(tool.get("max_penetration_m", 0.0)),
               "peak_force_N": float(np.linalg.norm(tool.get("peak_force_N", [0, 0, 0]))),
               "rows": int(len(g))}
        if len(g):
            touching = g[g["active_nodes"] > 0]
            row["increments_in_contact"] = int(len(touching))
            row["fz_max_N"] = float(g["fz"].max())
            row["fz_min_N"] = float(g["fz"].min())
            row["mean_abs_fz_in_contact_N"] = float(touching["fz"].abs().mean()) \
                if len(touching) else 0.0
            row["steps"] = ",".join(str(int(k)) for k in sorted(g["step"].unique()))
        rows.append(row)
    return rows


def versions(executable: Optional[str] = None) -> Dict[str, Any]:
    """The software a run used: precomp, Python and its scientific stack,
    torch, the platform, the sparlab_form build (when given) and the git
    revision."""
    import scipy
    import sklearn

    from precomp import __version__

    v: Dict[str, Any] = {"precomp": __version__, "python": platform.python_version(),
                         "numpy": np.__version__, "scipy": scipy.__version__,
                         "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                         "platform": platform.platform()}
    try:
        import torch
        v["torch"] = torch.__version__
    except ImportError:
        v["torch"] = None
    if executable is not None:
        from precomp.fea import sparlab_version
        v["sparlab_form"] = sparlab_version(executable)
    try:
        v["git_revision"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                           capture_output=True, text=True,
                                           cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        pass
    return v
