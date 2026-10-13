"""Process-realism lens of the physics audit: shared helpers.

Builds sparlab_form decks for process variants of benchmark test parts (the
commanded shape and FormingSetup of the springback benchmark's cached
uncompensated run), runs them with the audit worktree's Release build (one
thread, at most two at a time), and measures the formed / unloaded / released
shape against the target and against the part's baseline run.

Variants (spec keys, all optional):
  setup       : FormingSetup.replace(**setup) (blank_size, clamp_margin,
                step_down, tool_radius, friction, ...)
  offset      : [dx, dy, dz] [m] added to every in-contact path point (a
                constant path / calibration error)
  dsif        : {"thickness": "t0" | "sine" | float, "radius": R2, "friction": mu,
                "gap": g} - a second (support) sphere below the sheet opposite
                the primary tool (precomp.toolpath.dsif_support_path geometry)
Nothing here modifies tracked code.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from precomp.fea import build_deck, load_result, run_deck               # noqa: E402
from precomp.fea.deck import make_toolpath                              # noqa: E402
from precomp.fea.setup import FormingSetup                              # noqa: E402
from precomp.geometry.heightmap import HeightMap                        # noqa: E402
from precomp.metrology import flange_mask, part_mask, vertical_deviation  # noqa: E402
from precomp.toolpath import AIR, dsif_support_path, tool_center_surface  # noqa: E402

EXE = ROOT / "build" / "bin" / "sparlab_form"
CACHE = Path("/home/user/wt/bench/benchmarks/springback/work/runs/runs")
RUNS = HERE / "runs"
RESULTS = HERE / "results.jsonl"
MM = 1e-3

#: the 8 benchmark test parts -> deck hash of their cached uncompensated run
TEST_PARTS = {}
for _f in sorted(Path("/home/user/wt/bench/benchmarks/springback/work/da").glob("*.json")):
    TEST_PARTS[_f.stem] = json.loads(_f.read_text())["iterations"][0]["deck_hash"]


def cached_run_dir(part: str) -> Path:
    h = TEST_PARTS[part]
    hits = sorted((CACHE / h[:2]).glob(h + "*"))
    if len(hits) != 1:
        raise RuntimeError(f"cached run of {part} not found ({h})")
    return hits[0]


def part_inputs(part: str):
    d = cached_run_dir(part)
    prov = json.loads((d / "precomp_deck.json").read_text())
    setup = FormingSetup.from_dict(prov["setup"]).replace(executable=str(EXE), threads=1)
    return setup, HeightMap.load(d / "commanded.npz")


def write_traj(path: Path, t: np.ndarray, pts: np.ndarray) -> None:
    lines = ["t,x,y,z"] + [f"{a:.17g},{x:.17g},{y:.17g},{z:.17g}"
                           for a, (x, y, z) in zip(t, pts)]
    path.write_text("\n".join(lines) + "\n")


def forces_table(part: str, source: str) -> pd.DataFrame:
    if source == "cache":
        d = cached_run_dir(part) / "output"
    else:
        d = RUNS / part / source / "output"
    tf = pd.read_csv(d / "tool_forces.csv")
    return tf[tf["tool"] == "tool"]


def build_variant(part: str, name: str, spec: Dict[str, Any]) -> Path:
    setup, commanded = part_inputs(part)
    k = float(spec.get("scale", 1.0))
    if k != 1.0:
        # geometric scaling of the part and the fixture (tool, sheet, step-down,
        # mesh size unchanged): a k times larger part of the same shape
        g = commanded.grid
        from precomp.geometry.heightmap import Grid
        commanded = HeightMap(Grid(k * g.x0, k * g.y0, g.nx, g.ny, k * g.h), k * commanded.z,
                              commanded.mask, dict(commanded.metadata))
        setup = setup.replace(blank_size=k * setup.blank_size, clamp_margin=k * setup.clamp_margin)
    if spec.get("material"):
        setup = setup.replace(material=setup.material.replace(**spec["material"]))
    s = setup.replace(**spec.get("setup", {}))
    d = RUNS / part / name
    if (d / "COMPLETE").exists() or (d / "deck.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    path = make_toolpath(s, commanded)
    build_deck(s, commanded, d, toolpath=path)
    t = path.t
    pts = path.points.copy()
    contact = path.level != AIR
    info: Dict[str, Any] = {}
    if "offset" in spec:
        pts[contact] = pts[contact] + np.asarray(spec["offset"], float)
    write_traj(d / "toolpath.csv", t, pts)
    deck = json.loads((d / "deck.json").read_text())
    if "dsif" in spec:
        ds = spec["dsif"]
        th = ds.get("thickness", "t0")
        R2 = ds.get("radius", s.tool_radius)
        cz = tool_center_surface(commanded, path.tool_radius)
        n = cz.sample(cz.normals(), path.points[:, 0], path.points[:, 1])
        n = np.where(np.isfinite(n), n, np.array([0.0, 0.0, 1.0]))
        n[path.level == AIR] = np.array([0.0, 0.0, 1.0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        if th in ("t0", "nogouge"):          # precomp's dsif_support_path: initial thickness
            tl = np.full(len(t), s.thickness)
        elif th == "sine":      # thinned wall, t0 * n_z (sine law)
            tl = s.thickness * n[:, 2]
        else:
            tl = np.full(len(t), float(th))
        tl = tl + ds.get("gap", 0.0)
        spts = path.points - (path.tool_radius + tl + R2)[:, None] * n
        if th == "nogouge":
            # the support ball below the sheet, opposite the primary contact in
            # plan, lifted until it touches the TARGET's underside (target - t0,
            # a vertical offset = the sine-law thickness t0 cos(alpha) along the
            # normal) without penetrating it anywhere: a drop-cutter from below.
            # It cannot reach into the concave (from below) rim fillet, and never
            # penetrates the initial flat sheet (which lies above the target).
            under = commanded.with_z(-(commanded.z - s.thickness))
            S = -tool_center_surface(under, R2).z
            Shm = commanded.with_z(S)
            xy = path.points[:, :2] - (path.tool_radius + s.thickness + R2) * n[:, :2]
            spts = np.column_stack([xy, Shm.interpolate(xy[:, 0], xy[:, 1], masked=False)])
            spts[:, 2] += ds.get("squeeze", 0.0)
            info["support_primary_distance_mm"] = [
                float(np.linalg.norm(spts - path.points, axis=1)[~(path.level == AIR)].min() / MM),
                float(np.linalg.norm(spts - path.points, axis=1)[~(path.level == AIR)].max() / MM)]
        if th == "t0":
            ref = dsif_support_path(path, commanded, s.thickness, R2).points
            info["check_vs_dsif_support_path_mm"] = float(np.abs(ref - spts).max() / MM)
        # in the air (approach, retract) keep the support clear of the sheet
        air = path.level == AIR
        spts[air, 2] = np.minimum(spts[air, 2], -(s.thickness + R2) - 1.0e-3)
        write_traj(d / "support.csv", t, spts)
        e = s.free_half_width
        tool = copy.deepcopy(deck["forming"]["tools"][0])
        tool.update({"name": "support", "radius": float(R2),
                     "friction": float(ds.get("friction", s.friction)),
                     "surface": {"name": "bottom_side",
                                 "box": {"zmax": -s.thickness, "xmin": -e, "xmax": e,
                                         "ymin": -e, "ymax": e}},
                     "trajectory": {"file": "support.csv"}})
        deck["forming"]["tools"].append(tool)
        deck["forming"]["steps"][0]["tools"].append("support")
        info["support_min_z_mm"] = float(spts[:, 2].min() / MM)
    if spec.get("deck"):
        _merge(deck, spec["deck"])
    (d / "deck.json").write_text(json.dumps(deck, indent=1, sort_keys=True))
    (d / "variant.json").write_text(json.dumps({"part": part, "variant": name, "spec": spec,
                                                "info": info}, indent=1, sort_keys=True,
                                               default=lambda o: np.asarray(o).tolist()))
    return d


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def run_variant(part: str, name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    d = build_variant(part, name, spec)
    rec: Dict[str, Any] = {"part": part, "variant": name}
    if not (d / "COMPLETE").exists():
        t0 = time.time()
        try:
            run_deck(d, executable=EXE, threads=1)
            (d / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        except Exception as exc:
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:3000],
                       wall_s=time.time() - t0)
            (d / "FAILED").write_text(rec["error"])
            with open(RESULTS, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            return rec
    rec.update(ok=True, **measure(part, d))
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


# ---------------------------------------------------------------------------
def stats(v: np.ndarray) -> Dict[str, float]:
    return {"rms": float(np.sqrt(np.mean(v ** 2))), "max": float(np.abs(v).max()),
            "bias": float(np.mean(v))}


def plane_fit_residual(dev: np.ndarray, X: np.ndarray, Y: np.ndarray, mask: np.ndarray):
    """Least-squares z + tilt removal of dev over mask (a 3-DOF best fit)."""
    A = np.column_stack([np.ones(mask.sum()), X[mask], Y[mask]])
    c, *_ = np.linalg.lstsq(A, dev[mask], rcond=None)
    return dev - (c[0] + c[1] * X + c[2] * Y), c


def surfaces(d: Path, target: HeightMap) -> Dict[str, np.ndarray]:
    f = d / "surfaces.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in ("form", "unload", "release")}
    res = load_result(d / "output")
    out = {}
    for s in ("form", "unload", "release"):
        hm = res.formed_surface(s, grid=target.grid)
        out[s] = vertical_deviation(hm, target).z + target.z
    np.savez_compressed(f, **out, target=target.z)
    return out


def measure(part: str, d: Path) -> Dict[str, Any]:
    if (d / "commanded.npz").exists():
        target = HeightMap.load(d / "commanded.npz")
    else:
        _, target = part_inputs(part)
    S = surfaces(d, target)
    pm = part_mask(target)
    upper = pm & (target.z > -1e-3)
    deep = pm & ~upper
    fl = flange_mask(target)
    X, Y = target.grid.mesh()
    out: Dict[str, Any] = {}
    for st in ("release", "unload", "form"):
        dev = (S[st] - target.z) / MM
        out[f"{st}_part"] = stats(dev[pm])
        out[f"{st}_upper"] = stats(dev[upper])
        out[f"{st}_deep"] = stats(dev[deep])
        out[f"{st}_flange"] = stats(dev[fl])
        out[f"{st}_depth_mm"] = float(-S[st][pm].min() / MM)
    dev = (S["release"] - target.z) / MM
    r, c = plane_fit_residual(dev, X, Y, pm)
    out["release_part_bestfit3"] = stats(r[pm])
    # baseline comparison
    base = RUNS / part / "base"
    same_grid = target.grid.matches(part_inputs(part)[1].grid)
    if base.exists() and base != d and (base / "surfaces.npz").exists() and same_grid:
        B = np.load(base / "surfaces.npz")
        for st in ("release", "form"):
            dd = (S[st] - B[st]) / MM
            out[f"vs_base_{st}_part"] = stats(dd[pm])
            out[f"vs_base_{st}_upper"] = stats(dd[upper])
            out[f"vs_base_{st}_deep"] = stats(dd[deep])
    summ = json.loads((d / "output" / "summary.json").read_text())
    out["runtime_s"] = summ.get("runtime_s")
    out["increments"] = summ["timing"]["increments"]
    out["cuts"] = summ["timing"]["cuts"]
    out["max_plastic_strain"] = max(s.get("max_plastic_strain", 0) for s in summ["steps"])
    tf = pd.read_csv(d / "output" / "tool_forces.csv")
    for tool in tf["tool"].unique():
        c1 = tf[(tf["tool"] == tool) & (tf["step"] == 1) & (tf["active_nodes"] > 0)]
        if len(c1):
            out[f"force_{tool}"] = {"fz_mean": float(c1.fz.mean()), "fz_max": float(c1.fz.abs().max()),
                                    "fxy_mean": float(np.hypot(c1.fx, c1.fy).mean()),
                                    "fxy_max": float(np.hypot(c1.fx, c1.fy).max())}
    return out
