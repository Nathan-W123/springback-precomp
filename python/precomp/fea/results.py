"""Reader for the result directory of a sparlab_form run.

The contract is `docs/forming.md`, section 3 ("Output contract"):

* `summary.json` - `completed`, `termination`, `runtime_s`, `timing`,
  `analysis`, `steps` (per step: `name`, `type`, `completed`,
  `files_stem`, ...), `tools`, `mesh`, `warnings`, `files`, `provenance`;
* `config.json` (the deck) and `mesh.json`;
* per completed step: `<files_stem>_nodes.csv` with the columns
  `node,X,Y,Z,ux,uy,uz` (reference coordinates and displacement [m]) and
  `<files_stem>_elements.csv` with `element,eq_plastic_strain,von_mises_Pa`
  ([-], Pa). `files_stem` is `step_<k>_<s>`, k counting the steps from 1 and
  s the step name with every character other than letters, digits, `-` and
  `_` replaced by `_`; the loader takes it from the summary, so snapshot
  files (`<files_stem>_inc_<i>_nodes.csv`) are never mistaken for steps;
* `tool_forces.csv` - `step,increment,t,tool,cx,cy,cz,fx,fy,fz,active_nodes,
  max_penetration_m`: per converged increment and active tool, the step
  number (from 1), the pseudo-time, the tool's reference point [m], the
  force [N], the nodes in contact and the largest penetration [m].

**Force sign.** `fx, fy, fz` are the force the sheet exerts ON THE TOOL:
fz > 0 while the tool pushes down into the sheet. That is the load the
robot carries (`precomp.robot.forces_on_path` takes it as it is).

Every table is validated (columns present, no missing values, unique node
ids, the same nodes and reference coordinates in every step, a node table
for every completed step of the summary), and a violation raises
PrecompError naming the file. Nothing is recomputed from physics: the
formed surface is the deformed top surface of the mesh, interpolated
linearly on its own triangulation.

Where this fits: the results step of "Simulations": loads a run's output directory and gives the
    formed surface of any step as a HeightMap.
Called from: fea/runner.py:simulate (load_result), compensation.py:FEAPredictor.__call__ and
    ml/generate.py:SparlabSimulator.run (formed_surface).
Calls into: geometry.raycast (raycast_top), geometry.heightmap, pandas.
Start reading at: load_result, then FormingResult.formed_surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._util import PathLike, PrecompError, read_json, safe_name
from ..geometry.heightmap import Grid, HeightMap, _fill_nearest
from ..geometry.raycast import raycast_top

NODE_COLUMNS = ["node", "X", "Y", "Z", "ux", "uy", "uz"]
ELEMENT_COLUMNS = ["element", "eq_plastic_strain", "von_mises_Pa"]
FORCE_COLUMNS = ["step", "increment", "t", "tool", "cx", "cy", "cz", "fx", "fy", "fz",
                 "active_nodes", "max_penetration_m"]
_STEM_RE = re.compile(r"^step_(\d+)_[A-Za-z0-9_-]+$")


def _read_table(path: Path, columns: List[str], allow_empty: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas raises several types for malformed files
        raise PrecompError(f"{path}: cannot be read as CSV ({exc})") from exc
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PrecompError(f"{path}: missing columns {missing} (found {list(frame.columns)})")
    if frame.empty and not allow_empty:
        raise PrecompError(f"{path} contains no rows")
    numeric = [c for c in columns if c != "tool"]
    if frame[numeric].isna().any().any():
        bad = frame[numeric].columns[frame[numeric].isna().any()].tolist()
        raise PrecompError(f"{path}: missing values in columns {bad}")
    return frame


@dataclass
class StepResult:
    """Nodal (and optionally element) results at the end of one step.

    index : the step's number k in the file names (from 1); name : its name;
    info : its entry in summary.json's `steps`.
    """

    index: int
    name: str
    nodes: pd.DataFrame
    elements: Optional[pd.DataFrame] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def reference(self) -> np.ndarray:
        """(n, 3) reference coordinates [m], in node order."""
        return self.nodes[["X", "Y", "Z"]].to_numpy(dtype=float)

    @property
    def displacement(self) -> np.ndarray:
        """(n, 3) displacements [m]."""
        return self.nodes[["ux", "uy", "uz"]].to_numpy(dtype=float)

    @property
    def current(self) -> np.ndarray:
        """(n, 3) deformed coordinates reference + displacement [m]."""
        return self.reference + self.displacement


@dataclass
class _SurfaceLayout:
    """Top and bottom node indices of a structured sheet, as (ny, nx) arrays."""

    top: np.ndarray
    bottom: np.ndarray
    spacing: float


@dataclass
class FormingResult:
    """A loaded sparlab_form result directory (see the module docstring)."""

    directory: Path
    summary: Dict[str, Any]
    steps: List[StepResult]
    tool_forces: Optional[pd.DataFrame] = None
    mesh: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    _layout: Optional[_SurfaceLayout] = field(default=None, repr=False)

    # -- access ------------------------------------------------------------
    @property
    def step_names(self) -> List[str]:
        return [s.name for s in self.steps]

    @property
    def completed(self) -> bool:
        """The summary's completion flag (False when it is absent)."""
        return bool(self.summary.get("completed", False))

    @property
    def termination(self) -> str:
        """Why the run ended (summary.json's `termination`)."""
        return str(self.summary.get("termination", ""))

    def step(self, which: Union[int, str] = -1) -> StepResult:
        """A step by position in `steps` (negative counts from the end) or by name."""
        if isinstance(which, str):
            for s in self.steps:
                if s.name == which or s.name == safe_name(which):
                    return s
            raise PrecompError(f"{self.directory}: no step named {which!r}; steps are "
                               f"{self.step_names}")
        try:
            return self.steps[which]
        except IndexError:
            raise PrecompError(f"{self.directory}: step {which} out of range "
                               f"({len(self.steps)} steps)") from None

    # -- surface -----------------------------------------------------------
    def _surface_layout(self) -> _SurfaceLayout:
        if self._layout is not None:
            return self._layout
        ref = self.steps[0].reference
        span = float(np.ptp(ref, axis=0).max())
        tol = 1e-9 * max(span, 1e-12)
        ztop = ref[:, 2].max()
        zbot = ref[:, 2].min()
        top = np.flatnonzero(np.abs(ref[:, 2] - ztop) <= tol)
        bot = np.flatnonzero(np.abs(ref[:, 2] - zbot) <= tol)
        if abs(ztop) > 1e3 * tol:
            raise PrecompError(f"{self.directory}: the top surface of the reference mesh is "
                               f"at z = {ztop:.6g} m, not at the tool-side plane z = 0")

        def grid_of(ids: np.ndarray) -> Tuple[np.ndarray, float]:
            xs = np.unique(np.round(ref[ids, 0] / tol).astype(np.int64))
            ys = np.unique(np.round(ref[ids, 1] / tol).astype(np.int64))
            if len(xs) * len(ys) != len(ids):
                raise PrecompError(
                    f"{self.directory}: the surface nodes do not form a structured grid "
                    f"({len(ids)} nodes on {len(xs)} x {len(ys)} distinct x, y values)")
            ix = np.searchsorted(xs, np.round(ref[ids, 0] / tol).astype(np.int64))
            iy = np.searchsorted(ys, np.round(ref[ids, 1] / tol).astype(np.int64))
            out = np.full((len(ys), len(xs)), -1, dtype=np.int64)
            out[iy, ix] = ids
            if (out < 0).any():
                raise PrecompError(f"{self.directory}: duplicate surface nodes")
            spacing = float(np.median(np.diff(np.sort(np.unique(ref[ids, 0])))))
            return out, spacing

        top_grid, spacing = grid_of(top)
        bot_grid, _ = grid_of(bot)
        if top_grid.shape != bot_grid.shape:
            raise PrecompError(f"{self.directory}: top and bottom surfaces differ in layout")
        self._layout = _SurfaceLayout(top_grid, bot_grid, spacing)
        return self._layout

    def default_grid(self) -> Grid:
        """The grid of the reference top-surface nodes."""
        lay = self._surface_layout()
        ref = self.steps[0].reference
        x0 = float(ref[lay.top[0, 0], 0])
        y0 = float(ref[lay.top[0, 0], 1])
        ny, nx = lay.top.shape
        return Grid(x0, y0, nx, ny, lay.spacing)

    def formed_surface(self, step: Union[int, str] = -1,
                       grid: Optional[Grid] = None) -> HeightMap:
        """The deformed tool-side surface at the end of `step` on `grid` [m].

        The tool-side surface is the set of nodes with reference Z = 0; their
        deformed positions, triangulated as the reference grid (two triangles
        per cell), are interpolated linearly at the grid nodes (a vertical
        ray through each node). Grid nodes outside the deformed surface's plan
        are invalid in the result and carry the nearest valid height. The
        default grid is the reference node grid. Raises PrecompError if the
        deformed surface folds over in plan (it would not be a height field).
        """
        s = self.step(step)
        lay = self._surface_layout()
        cur = s.current
        xy = cur[lay.top, :2]
        z = cur[lay.top, 2]
        grid = self.default_grid() if grid is None else grid
        values, valid = _interpolate_structured(xy, z, grid, self.directory)
        return HeightMap(grid, values, valid, {"source": "sparlab_form", "step": s.name,
                                               "directory": str(self.directory),
                                               **{k: v for k, v in self.provenance.items()
                                                  if isinstance(v, (str, int, float, bool))}})

    def thickness_map(self, step: Union[int, str] = -1,
                      grid: Optional[Grid] = None) -> HeightMap:
        """Sheet thickness [m] at the end of `step`, on `grid`, located at the
        deformed tool-side surface.

        Thickness at a surface point is the distance between the deformed top
        node and the deformed bottom node of the same through-thickness node
        column (the column is straight in the structured mesh). With several
        layers the column's intermediate nodes are ignored. The result's
        `z` holds thickness values, metadata {"quantity": "thickness"}.
        """
        s = self.step(step)
        lay = self._surface_layout()
        cur = s.current
        t = np.linalg.norm(cur[lay.top] - cur[lay.bottom], axis=-1)
        grid = self.default_grid() if grid is None else grid
        values, valid = _interpolate_structured(cur[lay.top, :2], t, grid, self.directory)
        return HeightMap(grid, values, valid, {"quantity": "thickness", "unit": "m",
                                               "step": s.name})

    def plastic_strain(self, step: Union[int, str] = -1) -> Optional[pd.DataFrame]:
        """The element table of `step` (eq_plastic_strain [-], von_mises_Pa [Pa]) or None."""
        return self.step(step).elements

    def forming_forces(self, tool: Optional[str] = None) -> pd.DataFrame:
        """The tool force history (tool_forces.csv), optionally for one tool,
        with an added column `f` = |(fx, fy, fz)| [N]. The forces are those
        the sheet exerts on the tool (fz > 0 while pushing down)."""
        if self.tool_forces is None:
            raise PrecompError(f"{self.directory}: tool_forces.csv is absent")
        frame = self.tool_forces
        if tool is not None:
            frame = frame[frame["tool"] == tool]
            if frame.empty:
                raise PrecompError(f"{self.directory}: no forces for tool {tool!r}")
        frame = frame.copy()
        frame["f"] = np.linalg.norm(frame[["fx", "fy", "fz"]].to_numpy(float), axis=1)
        return frame.reset_index(drop=True)


def _interpolate_structured(xy: np.ndarray, values: np.ndarray, grid: Grid,
                            where: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of `values` (ny, nx) given at the positions `xy`
    (ny, nx, 2) of a structured surface, at the nodes of `grid`."""
    ny, nx = values.shape
    p = np.concatenate([xy, values[..., None]], axis=-1)
    a = p[:-1, :-1].reshape(-1, 3)
    b = p[:-1, 1:].reshape(-1, 3)
    c = p[1:, 1:].reshape(-1, 3)
    d = p[1:, :-1].reshape(-1, 3)
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], axis=0)
    e1 = tris[:, 1, :2] - tris[:, 0, :2]
    e2 = tris[:, 2, :2] - tris[:, 0, :2]
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.any(area <= 0):
        raise PrecompError(f"{where}: {int(np.sum(area <= 0))} surface triangles fold over in "
                           "plan view; the deformed surface is not a height field")
    z, hit = raycast_top(tris, grid)
    if not hit.any():
        raise PrecompError(f"{where}: the grid does not overlap the deformed surface")
    z = _fill_nearest(np.where(hit, z, 0.0), hit)
    return z, hit


def files_stem(k: int, name: str) -> str:
    """`step_<k>_<s>`: the stem of step k's files (k from 1), s the step name
    with every byte (of its UTF-8) but ASCII letters, digits, `-` and `_`
    replaced by `_`, "step" if empty - `forming_step_stem` of
    src/io/FormingWriter.cpp."""
    keep = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    clean = "".join(chr(b) if b in keep else "_" for b in name.encode("utf-8"))
    return f"step_{k}_{clean or 'step'}"


def _step_entries(d: Path, summary: Dict[str, Any]) -> List[Tuple[int, str, str, Dict]]:
    """(k, name, files_stem, entry) of every completed step of the summary."""
    steps = summary.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PrecompError(f"{d / 'summary.json'}: no 'steps' list")
    out = []
    for i, entry in enumerate(steps):
        if not isinstance(entry, dict) or "name" not in entry or "files_stem" not in entry:
            raise PrecompError(f"{d / 'summary.json'}: steps[{i}] lacks 'name' or "
                               "'files_stem'")
        if not entry.get("completed", False):
            continue                      # a stopped step has no files
        stem = str(entry["files_stem"])
        m = _STEM_RE.match(stem)
        if not m or stem != files_stem(i + 1, str(entry["name"])):
            raise PrecompError(f"{d / 'summary.json'}: steps[{i}] has files_stem {stem!r}, "
                               f"not step_{i + 1}_<name>")
        out.append((int(m.group(1)), str(entry["name"]), stem, entry))
    return out


def load_result(directory: PathLike, provenance: Optional[Dict[str, Any]] = None
                ) -> FormingResult:
    """Load and validate a sparlab_form result directory (see the module docstring).

    The steps are the completed steps of summary.json, in order; each must
    have its node table. Raises PrecompError when none is completed.
    """
    d = Path(directory)
    if not d.is_dir():
        raise PrecompError(f"{d} is not a directory")
    summary = read_json(d / "summary.json")
    entries = _step_entries(d, summary)
    if not entries:
        raise PrecompError(f"{d}: no step completed ({summary.get('termination', '')})")
    steps = []
    ref0 = None
    for k, name, stem, entry in entries:
        path = d / f"{stem}_nodes.csv"
        if not path.is_file():
            raise PrecompError(f"{path}: the node table of completed step {name!r} is missing")
        nodes = _read_table(path, NODE_COLUMNS)[NODE_COLUMNS]
        nodes = nodes.sort_values("node").reset_index(drop=True)
        if nodes["node"].duplicated().any():
            raise PrecompError(f"{path}: duplicate node ids")
        ref = nodes[["X", "Y", "Z"]].to_numpy(float)
        if ref0 is None:
            ref0 = (nodes["node"].to_numpy(), ref)
        else:
            if not np.array_equal(ref0[0], nodes["node"].to_numpy()):
                raise PrecompError(f"{path}: the node set differs from the first step's")
            scale = max(float(np.abs(ref0[1]).max()), 1e-300)
            if np.abs(ref - ref0[1]).max() > 1e-9 * scale:
                raise PrecompError(f"{path}: the reference coordinates differ from the "
                                   "first step's")
        epath = d / f"{stem}_elements.csv"
        elements = _read_table(epath, ELEMENT_COLUMNS) if epath.is_file() else None
        steps.append(StepResult(k, name, nodes, elements, dict(entry)))
    forces = None
    fpath = d / "tool_forces.csv"
    if fpath.is_file():
        forces = _read_table(fpath, FORCE_COLUMNS, allow_empty=True)
    mesh = read_json(d / "mesh.json") if (d / "mesh.json").is_file() else None
    return FormingResult(d, summary, steps, forces, mesh, dict(provenance or {}))
