"""Tool paths for single-point incremental forming with a ball tool.

Everything here is the path of the tool *centre* [m], which is what SparLab's
forming trajectory and a robot's tool-centre point take.

Drop cutter. `tool_center_surface` computes, on the grid of a height map, the
height at which a ball of radius R centred above each node first touches the
surface when lowered:

    c_z(x) = max over |d| <= R of [ z(x + d) + sqrt(R^2 - |d|^2) ],

as a grey-scale dilation with a non-flat spherical structuring element
(`scipy.ndimage.grey_dilation`). It is exact over the grid nodes - the ball
touches at least one node and no node lies inside it. Between nodes the
surface is only known through interpolation; a bilinearly interpolated surface
can rise above the node samples by up to about ``h^2 / (8 R cos^3 theta)``
at a wall of angle theta, which is the gouge the grid can hide (0.5 um for
h = 0.25 mm, R = 5 mm, theta = 45 deg). Outside the grid the surface is
taken as continuing flat at the border height (border mode 'nearest').

Paths. `spiral_toolpath` is built on z-level loops: the tool tip descends by
`step_down` per level, i.e. level k has the tool centre at the constant height
``R - k step_down``, and the loop is the level set of the drop-cutter surface
at that height, so every point of it touches the target without gouging. The
spiral descends continuously between them: along each revolution the height
falls linearly with arc length from one level to the next, and the x-y
position is found on the segment between corresponding points of the two
loops where the drop-cutter surface has exactly that height (bisection), so
the spiral is gouge-free as well and its height never increases.
Loops come from `contourpy`, are closed, oriented consistently, start near
the previous level's start and are resampled by arc length.

Where this fits: "Simulations", between the target height map and the forming deck: the spiral
    path, the two-tool support path and the edge pass (rim_pass_path).
Called from: fea/deck.py:make_toolpath, fea/support.py:dsif_trajectory, fea/support.py:rim_pass.
Calls into: geometry.heightmap (HeightMap, Grid), scipy.ndimage (drop cutter), contourpy.
Start reading at: spiral_toolpath, then Toolpath.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from ._util import PathLike, PrecompError, require_positive
from .geometry.heightmap import Grid, HeightMap

#: Level value of points where the tool moves in the air (approach, retract,
#: traverse between separate loops).
AIR = -1


# ---------------------------------------------------------------------------
# Drop cutter
# ---------------------------------------------------------------------------
def spherical_structure(tool_radius: float, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """(footprint, structure) of a ball of radius `tool_radius` on a grid of
    spacing `spacing` [m]: footprint is |d| <= R, structure sqrt(R^2 - |d|^2)."""
    R = require_positive("tool_radius", tool_radius)
    m = int(math.floor(R / spacing + 1e-12))
    k = np.arange(-m, m + 1) * spacing
    DX, DY = np.meshgrid(k, k, indexing="xy")
    r2 = DX * DX + DY * DY
    footprint = r2 <= R * R * (1 + 1e-12)
    structure = np.where(footprint, np.sqrt(np.maximum(R * R - r2, 0.0)), 0.0)
    return footprint, structure


def tool_center_surface(hm: HeightMap, tool_radius: float) -> HeightMap:
    """Drop-cutter tool-centre height c_z(x, y) [m] of a ball tool over `hm`.

    The tool centre is (x, y, c_z); the ball touches the surface (exact over
    the grid nodes, see the module docstring). On a flat region c_z = z + R.
    The result carries metadata {"quantity": "tool_centre_height",
    "tool_radius": R}. Invalid nodes of `hm` are treated as surface at their
    fill height.
    """
    footprint, structure = spherical_structure(tool_radius, hm.grid.h)
    cz = ndimage.grey_dilation(hm.z, footprint=footprint, structure=structure, mode="nearest")
    return HeightMap(hm.grid, cz, hm.mask, {"quantity": "tool_centre_height",
                                            "tool_radius": float(tool_radius)})


def tool_reach_surface(hm: HeightMap, tool_radius: float) -> HeightMap:
    """The surface a ball of radius `tool_radius` pressing from above forms
    at most under `hm` [m]: the lowest points of the balls at the drop-cutter
    heights, ``min over |d| <= R of [c_z(x + d) - sqrt(R^2 - |d|^2)]`` - the
    morphological closing of the surface by the ball. It is never below
    `hm` and equals it wherever the ball fits; a pit narrower than the ball
    or a concave corner tighter than R (a floor fillet) is filled to what
    the ball reaches. Its drop-cutter surface is `hm`'s (closing does not
    change the dilation), so a tool path made for either is the same.
    Metadata: the nodes raised and the largest raise [m]."""
    footprint, structure = spherical_structure(tool_radius, hm.grid.h)
    cz = tool_center_surface(hm, tool_radius).z
    z = ndimage.grey_erosion(cz, footprint=footprint, structure=structure, mode="nearest")
    z = np.maximum(z, hm.z)                 # round-off: never below the surface
    raised = z - hm.z
    meta = dict(hm.metadata)
    meta["tool_reach"] = {"tool_radius": float(tool_radius),
                          "nodes_raised": int(np.sum(raised > 1e-12)),
                          "max_raise_m": float(raised.max())}
    return HeightMap(hm.grid, z, hm.mask, meta)


# ---------------------------------------------------------------------------
# Toolpath container
# ---------------------------------------------------------------------------
@dataclass
class Toolpath:
    """An ordered tool-centre path.

    points : (n, 3) tool-centre positions [m].
    level : (n,) int, the z level a point belongs to (1 = first level below
        the sheet), `AIR` (-1) where the tool moves in the air.
    tool_radius : ball radius [m].
    metadata : style, step-down, spacing, level heights, ...

    Consecutive duplicate points are removed on construction, so the arc
    length `s` and the pseudo-time `t = s / total_length` are strictly
    increasing.
    """

    points: np.ndarray
    level: np.ndarray
    tool_radius: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = np.asarray(self.points, dtype=float)
        lv = np.asarray(self.level, dtype=int)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
            raise ValueError("a tool path needs an (n >= 2, 3) array of points")
        if lv.shape != (len(p),):
            raise ValueError("level must hold one entry per point")
        if not np.all(np.isfinite(p)):
            raise ValueError("tool path points must be finite")
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        keep = np.concatenate([[True], step > 1e-12])
        self.points = p[keep]
        self.level = lv[keep]
        if len(self.points) < 2:
            raise ValueError("a tool path needs at least two distinct points")
        self.tool_radius = float(self.tool_radius)
        self.metadata = dict(self.metadata)

    @property
    def s(self) -> np.ndarray:
        """(n,) cumulative 3-D arc length of the tool centre [m], s[0] = 0."""
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0),
                                                               axis=1))])

    @property
    def total_length(self) -> float:
        """Path length [m]."""
        return float(self.s[-1])

    @property
    def t(self) -> np.ndarray:
        """(n,) pseudo-time s / total_length in [0, 1], strictly increasing."""
        s = self.s
        return s / s[-1]

    @property
    def num_levels(self) -> int:
        return int(len(np.unique(self.level[self.level != AIR])))

    def summary(self, feed: Optional[float] = None) -> Dict[str, Any]:
        """Length [m], points, levels, height range [m], longest segment [m],
        in-contact length [m] and, with `feed` [m/s], the time at that feed [s]."""
        p = self.points
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        contact = (self.level[1:] != AIR) & (self.level[:-1] != AIR)
        out = {
            "length_m": self.total_length,
            "contact_length_m": float(seg[contact].sum()),
            "points": int(len(p)),
            "levels": self.num_levels,
            "z_min_m": float(p[:, 2].min()),
            "z_max_m": float(p[:, 2].max()),
            "max_segment_m": float(seg.max()),
            "tool_radius_m": self.tool_radius,
        }
        if feed is not None:
            out["feed_m_per_s"] = require_positive("feed", feed)
            out["time_s"] = self.total_length / feed
        return out

    def to_sparlab_csv(self, path: PathLike) -> Path:
        """Write the SparLab trajectory table `t,x,y,z` (t in [0, 1], metres).

        Values are written with 17 significant digits, so the file round-trips
        exactly and its content hash depends only on the path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        t = self.t
        lines = ["t,x,y,z"]
        for ti, (x, y, z) in zip(t, self.points):
            lines.append(f"{ti:.17g},{x:.17g},{y:.17g},{z:.17g}")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------
def _signed_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _loop_length(loop: np.ndarray) -> float:
    closed = np.vstack([loop, loop[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def _nearest_on_loop(loop: np.ndarray, point: np.ndarray) -> Tuple[int, float]:
    """(segment index, fraction) of the point of the closed polyline nearest `point`."""
    a = loop
    b = np.roll(loop, -1, axis=0)
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-300)
    u = np.clip(((point - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
    proj = a + u[:, None] * ab
    k = int(np.argmin(((proj - point) ** 2).sum(axis=1)))
    return k, float(u[k])


def _rotate_to(loop: np.ndarray, seg: int, frac: float) -> np.ndarray:
    """The closed polyline started at the point `frac` along segment `seg`."""
    a = loop[seg]
    b = loop[(seg + 1) % len(loop)]
    start = a + frac * (b - a)
    rest = np.roll(loop, -(seg + 1), axis=0)
    out = np.vstack([start, rest])
    step = np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0), axis=1)
    return out[np.concatenate([[True], step[:-1] > 1e-15])]


def resample_loop(loop: np.ndarray, spacing: float, n: Optional[int] = None) -> np.ndarray:
    """Resample a closed polyline (m, 2) [m] at equal arc-length steps.

    Returns `n` points (default ceil(length / spacing), at least 3) starting
    at the loop's first point; the loop stays implicitly closed (the last
    point is not repeated).
    """
    closed = np.vstack([loop, loop[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if n is None:
        n = max(3, int(math.ceil(total / spacing - 1e-9)))
    snew = np.arange(n) * (total / n)
    return np.stack([np.interp(snew, s, closed[:, 0]), np.interp(snew, s, closed[:, 1])], -1)


def extract_loops(cz: HeightMap, level: float, min_loop_length: float) -> List[np.ndarray]:
    """Closed contour loops (m, 2) [m] of the height map `cz` at height `level`.

    Uses `contourpy` (marching squares with linear interpolation along cell
    edges). A contour that ends on the grid border raises PrecompError: the
    part runs off the grid and the loop cannot be closed. Loops shorter than
    `min_loop_length` [m] are dropped (the tool cannot follow them).
    """
    from contourpy import LineType, contour_generator

    g = cz.grid
    gen = contour_generator(x=g.x, y=g.y, z=cz.z, line_type=LineType.Separate)
    loops = []
    for line in gen.lines(level):
        line = np.asarray(line, dtype=float)
        if len(line) < 3:
            continue
        if not np.allclose(line[0], line[-1], rtol=0.0, atol=1e-9 * g.h):
            raise PrecompError(
                f"the contour at height {level:.6g} m reaches the grid border; the grid "
                "must contain the part with a margin of at least the tool radius")
        loop = line[:-1]
        if _loop_length(loop) >= min_loop_length:
            loops.append(loop)
    return loops


def _centre_levels(cz: HeightMap, tool_radius: float, step_down: float) -> List[float]:
    """Tool-centre heights of the levels: R - k step_down above the deepest
    point, then one last level step_down / 1000 above it."""
    cmin = float(cz.z.min())
    delta = 1e-3 * step_down
    levels = []
    k = 1
    while tool_radius - k * step_down > cmin + delta:
        levels.append(tool_radius - k * step_down)
        k += 1
    if not levels or levels[-1] - (cmin + delta) > delta:
        levels.append(cmin + delta)
    if tool_radius - levels[0] < 0.5 * delta:
        raise PrecompError("the target is flat: there is nothing to form")
    return levels


DEEPEST_LEVEL_SEARCH = 64


def _resolvable_deepest(cz: HeightMap, levels: List[float], tool_radius: float,
                        min_loop_length: float, single: bool) -> Tuple[List[float], float]:
    """`levels` with the deepest ones the tool cannot follow replaced by the
    lowest height where it can (a loop `extract_loops` keeps; exactly one
    with `single`), searched on `DEEPEST_LEVEL_SEARCH` steps between the
    deepest level and the deepest followable one above it. Returns (levels,
    how far the new deepest level lies above the old one [m]).

    The last level lies just above the deepest point. Over a flat floor its
    contour is the floor's outline, but where the deepest point is a pole
    or a dimple (a dome, an uneven compensated floor) that contour - and a
    regular level only a little above it - is a loop too short to follow
    and would be dropped, leaving the tool up to a step-down or more short
    of the command."""
    def ok(h: float) -> bool:
        loops = extract_loops(cz, h, min_loop_length)
        return len(loops) == 1 if single else len(loops) > 0

    j = len(levels) - 1
    while j >= 0 and not ok(levels[j]):
        j -= 1
    if j == len(levels) - 1:
        return levels, 0.0
    last = levels[-1]
    above = levels[j] if j >= 0 else tool_radius
    dh = (above - last) / DEEPEST_LEVEL_SEARCH
    for k in range(DEEPEST_LEVEL_SEARCH):
        h = last + k * dh
        if ok(h):
            return levels[:j + 1] + [h], h - last
    return levels, 0.0


def _ordered_loops(cz: HeightMap, levels: Sequence[float], spacing: float, direction: str,
                   alternate: bool, start_angle_deg: float,
                   min_loop_length: float) -> List[Tuple[float, List[np.ndarray]]]:
    """(height, loops) of every level that has a loop: the loops oriented,
    ordered by nearest neighbour and started at the point nearest the current
    tool position; not yet resampled."""
    if direction not in ("ccw", "cw"):
        raise ValueError("direction must be 'ccw' or 'cw'")
    out: List[Tuple[float, List[np.ndarray]]] = []
    position: Optional[np.ndarray] = None
    for k, lev in enumerate(levels):
        loops = extract_loops(cz, lev, min_loop_length)
        if not loops:
            continue
        ccw = (direction == "ccw") != (alternate and k % 2 == 1)
        oriented = []
        for loop in loops:
            if (_signed_area(loop) > 0) != ccw:
                loop = loop[::-1]
            oriented.append(loop)
        if position is None:
            a = math.radians(start_angle_deg)
            d = np.array([math.cos(a), math.sin(a)])
            first = max(oriented, key=lambda lp: float((lp @ d).max()))
            position = first[int(np.argmax(first @ d))]
        ordered = []
        remaining = list(oriented)
        while remaining:
            dists = []
            for lp in remaining:
                seg, frac = _nearest_on_loop(lp, position)
                a0 = lp[seg]
                b0 = lp[(seg + 1) % len(lp)]
                dists.append(float(np.linalg.norm(a0 + frac * (b0 - a0) - position)))
            j = int(np.argmin(dists))
            lp = remaining.pop(j)
            seg, frac = _nearest_on_loop(lp, position)
            lp = _rotate_to(lp, seg, frac)
            ordered.append(lp)
            position = lp[0]
        out.append((float(lev), ordered))
    if not out:
        raise PrecompError("no tool path loop found; check the step-down and the target depth")
    return out


def _min_loop(target: HeightMap, spacing: float, given: Optional[float]) -> float:
    if given is not None:
        return float(given)
    return max(2.0 * spacing, 4.0 * math.pi * target.grid.h)


def spiral_toolpath(target: HeightMap, tool_radius: float, step_down: float,
                    spacing: float = 1e-3, *, direction: str = "ccw",
                    start_angle_deg: float = 0.0, clearance: float = 2e-3,
                    final_loop: bool = True,
                    min_loop_length: Optional[float] = None) -> Toolpath:
    """Continuous (helical) tool path: the height falls linearly with arc
    length through each revolution, one step_down per revolution.

    Revolution k runs from loop k - 1 to loop k (loop 0 is loop 1 at the
    sheet plane, where the tool tip touches z = 0). At normalised arc length
    lambda the height is ``c_(k-1) + lambda (c_k - c_(k-1))`` and the x-y
    position is found on the line from the point of loop k - 1 at lambda
    towards its nearest point on loop k, where the drop-cutter surface has
    exactly that height (bisection on its bilinear interpolant), so the tool
    touches the target all along the revolution and the height never
    increases. With `final_loop` a last
    revolution at the deepest level closes the path. `step_down` is the
    vertical step of the tool tip per revolution [m] and `spacing` the
    distance between path points [m]; every level must have exactly one loop
    (one pocket
    without islands), otherwise ValueError.
    """
    R = require_positive("tool_radius", tool_radius)
    step_down = require_positive("step_down", step_down)
    spacing = require_positive("spacing", spacing)
    # Drop-cutter surface of the target and the tool-centre height of each level
    cz = tool_center_surface(target, R)
    levels = _centre_levels(cz, R, step_down)
    min_loop = _min_loop(target, spacing, min_loop_length)
    levels, raised = _resolvable_deepest(cz, levels, R, min_loop, single=True)
    # One closed loop per level: the contour of the drop-cutter surface at that height
    per_level = _ordered_loops(cz, levels, spacing, direction, False, start_angle_deg,
                               min_loop)
    for k, (_, loops) in enumerate(per_level, start=1):
        if len(loops) != 1:
            raise ValueError(f"spiral_toolpath needs one loop per level; level {k} has "
                             f"{len(loops)} (a part with several pockets on one level).")
    levels = [h for h, _ in per_level]
    loops = [l[0] for _, l in per_level]
    # Revolution k descends from loop k - 1 to loop k (loop 0: loop 1 at the sheet plane)
    heights = [R] + list(levels)
    loops = [loops[0]] + loops
    pts = []
    lvl = []
    for k in range(1, len(loops)):
        la, lb = loops[k - 1], loops[k]
        n = max(3, int(math.ceil(max(_loop_length(la), _loop_length(lb)) / spacing)))
        A = resample_loop(la, spacing, n)
        B = A.copy() if la is lb else _nearest_on_polyline(lb, A, spacing)
        lam = np.arange(n) / n
        z = heights[k - 1] + lam * (heights[k] - heights[k - 1])
        xy = _blend_to_height(cz, A, B, z)
        pts.append(np.column_stack([xy, z]))
        lvl.append(np.full(n, k))
    # Optional last revolution at the deepest level closes the path
    if final_loop:
        la, lb = loops[-2], loops[-1]
        n = max(3, int(math.ceil(max(_loop_length(la), _loop_length(lb)) / spacing)))
        A = resample_loop(la, spacing, n)
        B = A.copy() if la is lb else _nearest_on_polyline(lb, A, spacing)
        z = np.full(n + 1, heights[-1])
        xy = _blend_to_height(cz, np.vstack([A, A[:1]]), np.vstack([B, B[:1]]), z)
        pts.append(np.column_stack([xy, z]))
        lvl.append(np.full(n + 1, len(loops) - 1))
    # Approach and retract points in the air (level AIR) above the start and the end
    body = np.vstack(pts)
    safe_z = R + clearance
    start = np.array([[body[0, 0], body[0, 1], safe_z]])
    end = np.array([[body[-1, 0], body[-1, 1], safe_z]])
    meta = {"style": "spiral", "step_down": step_down, "spacing": spacing,
            "direction": direction, "level_heights": [float(v) for v in levels],
            "clearance": clearance, "final_loop": bool(final_loop),
            "deepest_level_raised_m": raised}
    path = Toolpath(np.vstack([start, body, end]),
                    np.concatenate([[AIR], np.concatenate(lvl), [AIR]]), R, meta)
    path.metadata["max_gouge_m"] = max_gouge(path, cz)
    return path


def max_gouge(path: Toolpath, cz: HeightMap) -> float:
    """Largest depth [m] of a path point below the bilinear drop-cutter
    surface `cz` (0 when no point gouges); points off the grid are ignored."""
    cs = cz.interpolate(path.points[:, 0], path.points[:, 1], masked=False)
    d = cs - path.points[:, 2]
    d = d[np.isfinite(d)]
    return float(max(0.0, d.max())) if len(d) else 0.0


def _nearest_on_polyline(loop: np.ndarray, pts: np.ndarray, spacing: float) -> np.ndarray:
    """For each point (k, 2), the nearest point of the closed polyline `loop`
    (to within spacing / 20: the loop is densified and searched with a KD-tree)."""
    from scipy.spatial import cKDTree

    dense = resample_loop(loop, spacing / 20.0)
    _, idx = cKDTree(dense).query(pts)
    return dense[idx]


def _first_crossing(cz: HeightMap, A: np.ndarray, D: np.ndarray, z: np.ndarray,
                    mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(found, mu_root): per row, the first sign change of c_z(A + mu D) - z
    from >= 0 to < 0 over the sampled `mu` (k,), bisected 50 times, keeping
    the side where c_z < z."""
    xy = A[:, None, :] + mu[None, :, None] * D[:, None, :]
    f = cz.interpolate(xy[..., 0], xy[..., 1], masked=False) - z[:, None]
    f = np.where(np.isfinite(f), f, -np.inf)
    change = (f[:, :-1] >= 0) & (f[:, 1:] < 0)
    found = change.any(axis=1) & (np.linalg.norm(D, axis=1) > 0)
    idx = np.argmax(change, axis=1)
    lo = mu[idx].astype(float)
    hi = mu[idx + 1].astype(float)
    rows = np.flatnonzero(found)
    a, d, zz = A[rows], D[rows], z[rows]
    l, h = lo[rows], hi[rows]
    for _ in range(50):
        mid = 0.5 * (l + h)
        p = a + mid[:, None] * d
        up = cz.interpolate(p[:, 0], p[:, 1], masked=False) - zz >= 0
        l = np.where(up, mid, l)
        h = np.where(up, h, mid)
    root = np.zeros(len(A))
    root[rows] = h
    return found, root


#: `_blend_to_height`: a crossing more than this many |AB| from A is taken
#: only when there is none within three grid spacings of B.
FAR_CROSSING = 2.0

#: `_blend_to_height`: B stands for a far crossing with none next to it when
#: the surface there is at most this far [m] above z (B on the loop at z,
#: to the round-off of the densified loop it is taken from: nanometres).
GRAZE_TOL = 1e-6


def _blend_to_height(cz: HeightMap, A: np.ndarray, B: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per row, an x-y point where the bilinear drop-cutter surface equals z.

    The point is searched along the line from A through B: first from a
    quarter of |AB| behind A to six times |AB| beyond it on a grid of |AB| / 8,
    then, for rows without a crossing or with one more than `FAR_CROSSING`
    |AB| from A, within three grid spacings of B on a
    grid of h / 8 (a narrow crossing, such as the last small loop around the
    pole of a dome); a far crossing with none next to B gives way to B when
    B is on the surface at z (a line that grazes a small last loop at B). The first sign change of c_z - z from >= 0 to < 0 is
    bisected, keeping the side where c_z < z, so the tool never ends inside
    the surface. Where A and B coincide (the first revolution, or a single
    level) the direction is the downhill direction of c_z at A (smoothed over
    two grid spacings), over four grid spacings. A row with no crossing keeps
    A where the surface stays below z there (the tool is in the air); where
    it is above z (A inside the surface) the point walks towards B to the
    first point at z, or takes B (on its loop, at or below z).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float).copy()
    h = cz.grid.h
    same = np.linalg.norm(B - A, axis=1) < 1e-12
    if np.any(same):
        gx, gy = cz.gradient(sigma=2.0 * h)
        grad = np.column_stack([cz.sample(gx, A[same, 0], A[same, 1]),
                                cz.sample(gy, A[same, 0], A[same, 1])])
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        down = np.divide(-grad, norm, out=np.zeros_like(grad), where=norm > 0)
        B[same] = A[same] + 4.0 * h * down
    D = B - A
    found, root = _first_crossing(cz, A, D, z, np.arange(-2, 49) / 8.0)
    out = A.copy()
    out[found] = A[found] + root[found, None] * D[found]
    # a crossing far beyond B (a line that misses a small loop round B and
    # crosses the surface on the far side of the pocket, or off the grid)
    # gives way to one next to B when there is one
    missing = np.flatnonzero((~found | (root > FAR_CROSSING)) & (np.linalg.norm(D, axis=1) > 0))
    if len(missing):
        length = np.linalg.norm(D[missing], axis=1)
        unit = D[missing] / length[:, None]
        start = B[missing] - 3.0 * h * unit
        f2, r2 = _first_crossing(cz, start, unit, z[missing], np.arange(49) * (h / 8.0))
        rows = missing[f2]
        out[rows] = start[f2] + r2[f2, None] * unit[f2]
        # a line that only grazes the loop at B (c_z(B) = z, the last
        # revolution at B's own level) has no sign change near B either; B
        # itself is on the surface at z, the far crossing is not
        far = missing[~f2 & found[missing] & (root[missing] > FAR_CROSSING)]
        if len(far):
            on = cz.interpolate(B[far, 0], B[far, 1], masked=False) - z[far] <= GRAZE_TOL
            out[far[on]] = B[far[on]]
    # a row left inside the surface - no crossing at all, so it kept A,
    # where the surface is above z (the last revolution's first points, at
    # the previous loop over a DA-raised spot of the floor, up to 0.7 mm
    # below the drop-cutter surface) - walks from there towards B, which
    # lies on its loop at or below z, to the first point at z; B itself when
    # the surface comes down to z only at B (a graze)
    inside = np.flatnonzero(~same & (np.linalg.norm(D, axis=1) > 0)
                            & (cz.interpolate(out[:, 0], out[:, 1], masked=False) - z
                               > GRAZE_TOL))
    if len(inside):
        D3 = B[inside] - out[inside]
        f3, r3 = _first_crossing(cz, out[inside], D3, z[inside], np.arange(65) / 64.0)
        out[inside] = np.where(f3[:, None], out[inside] + r3[:, None] * D3, B[inside])
    return out


def lift_heights(surface: HeightMap, radius: float, x: np.ndarray, y: np.ndarray, *,
                 offset: float = 0.0, floor: Optional[np.ndarray] = None) -> np.ndarray:
    """The highest centre heights [m] of a ball of radius `radius` under a sheet.

    The sheet's underside is ``max(surface, floor) - offset`` (`offset` the
    vertical thickness; `floor` [m], one per point, the height below which
    the sheet has not been pushed yet - the in-process sheet of a path whose
    tip is at that height; None for the finished surface). At each point
    (x, y) the ball centre may rise to

        b = min over |d| <= R of [underside(x + d) - sqrt(R^2 - |d|^2)]

    and no higher without entering the underside - the drop cutter of
    `tool_center_surface` turned upside down ("lift cutter"), evaluated on
    the offsets of the surface's grid within R of the point, the surface
    interpolated bilinearly there (continued flat at 0 outside the grid).
    """
    R = require_positive("radius", radius)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    footprint, structure = spherical_structure(R, surface.grid.h)
    m = footprint.shape[0] // 2
    k = (np.arange(-m, m + 1) * surface.grid.h)
    DX, DY = np.meshgrid(k, k, indexing="xy")
    dx, dy, cap = DX[footprint], DY[footprint], structure[footprint]
    out = np.empty(len(x))
    chunk = max(1, 2_000_000 // len(dx))
    for a in range(0, len(x), chunk):
        b = slice(a, a + chunk)
        zz = surface.interpolate(x[b, None] + dx[None, :], y[b, None] + dy[None, :],
                                 masked=False, fill_value=0.0)
        if floor is not None:
            zz = np.maximum(zz, np.asarray(floor, dtype=float).ravel()[b, None])
        out[b] = (zz - offset - cap[None, :]).min(axis=1)
    return out


def dsif_support_points(primary: Toolpath, surface: HeightMap, thickness: float,
                        support_radius: Optional[float] = None, *, squeeze: float = 0.0,
                        thickness_law: str = "sine", clearance: float = 2e-3) -> np.ndarray:
    """The support (bottom) tool of double-sided incremental forming, point by
    point with the forming tool: (n, 3) ball centres [m], row i at the same
    pseudo-time as `primary.points[i]` (write them with `primary.t`, never
    with a path's own arc-length time).

    `surface` is the surface `primary` was made for. At a point in contact
    the support sits opposite the forming tool along the drop-cutter normal
    n (the target's normal at the contact point, for a smooth target): its
    centre is ``p - (R1 + t_n + R2) n``, so the two balls hold a sheet of
    thickness t_n between them - t_n = t cos(wall angle) = t n_z with
    `thickness_law` "sine" (the sine law of shear spinning: the vertical
    thickness stays t), t with "initial". That position assumes the sheet
    around the contact already has its final shape; where it has not been
    pushed down yet (the first revolutions, a wall still flat below the
    tool), it would lie inside the sheet. The support is therefore never
    raised above the lift cutter (`lift_heights`) of the in-process sheet:
    the surface above the tip's current height, flat at that height below
    it, vertically `thickness` thick. `squeeze` then moves it towards the
    forming tool by squeeze t_n along n (0: the balls just touch the sheet;
    > 0 squeezes it; < 0 leaves a gap). In the air (`AIR` points: approach,
    retract, traverses) the support waits below the whole part - its top
    `clearance` below the deepest point of `surface` minus `thickness`.
    Thinning beyond the sine law and the sheet's springback are ignored.
    """
    t = require_positive("thickness", thickness)
    R1 = primary.tool_radius
    R2 = R1 if support_radius is None else require_positive("support_radius", support_radius)
    if thickness_law not in ("sine", "initial"):
        raise ValueError("thickness_law must be 'sine' or 'initial'")
    if not -1.0 <= squeeze < 1.0:
        raise ValueError("squeeze must lie in [-1, 1)")
    clearance = require_positive("clearance", clearance)
    p = primary.points
    air = primary.level == AIR
    cz = tool_center_surface(surface, R1)
    n = cz.sample(cz.normals(), p[:, 0], p[:, 1])
    n = np.where(np.isfinite(n), n, np.array([0.0, 0.0, 1.0]))
    n[air] = np.array([0.0, 0.0, 1.0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    tn = t * n[:, 2] if thickness_law == "sine" else np.full(len(p), t)
    touch = p - (R1 + tn + R2)[:, None] * n
    bound = lift_heights(surface, R2, touch[:, 0], touch[:, 1], offset=t,
                         floor=p[:, 2] - R1)
    out = np.column_stack([touch[:, :2], np.minimum(touch[:, 2], bound)])
    out += (squeeze * tn)[:, None] * n
    low = -(t + R2 + clearance + max(0.0, -float(surface.z.min())))
    out[air] = np.column_stack([p[air, :2], np.full(int(air.sum()), low)])
    return out


def dsif_support_path(primary: Toolpath, target: HeightMap, thickness: float,
                      support_radius: Optional[float] = None, **kwargs: Any) -> Toolpath:
    """The support tool of double-sided incremental forming as a `Toolpath`
    (`dsif_support_points`; keyword arguments go there).

    Its own pseudo-time runs with its own arc length, which is NOT the
    forming tool's: to drive both tools in one analysis, write the points of
    `dsif_support_points` with the forming tool's `t` (the deck builder
    does, `precomp.fea.support`). Consecutive coincident points are dropped.
    """
    R1 = primary.tool_radius
    R2 = R1 if support_radius is None else require_positive("support_radius", support_radius)
    pts = dsif_support_points(primary, target, thickness, R2, **kwargs)
    meta = dict(primary.metadata)
    meta.update({"style": f"{primary.metadata.get('style', 'path')}_dsif_support",
                 "thickness": float(thickness), "primary_radius": R1,
                 "synchronised_with_primary": False})
    return Toolpath(pts, primary.level.copy(), R2, meta)


def lift_at_points(underside: HeightMap, radius: float, x: np.ndarray, y: np.ndarray
                   ) -> np.ndarray:
    """The highest centre heights [m] of a ball of radius R at points (x, y)
    with no node of `underside` inside it:

        b(p) = min over nodes x_k with |x_k - p| <= R of
               [underside(x_k) - sqrt(R^2 - |x_k - p|^2)],

    the lift cutter exact over the grid's nodes, as `tool_center_surface`
    is for the drop cutter, at any point (not only at nodes, and without
    interpolating the surface). +inf where no node lies within R. A ball at
    b(p) touches the underside at a node; a surface whose nodes all lie at
    or above the top of a set of such balls (`swept_ball_top`) lets every
    one of them rise to the same height again.
    """
    R = require_positive("radius", radius)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    gx, gy = underside.grid.x, underside.grid.y
    out = np.full(len(x), np.inf)
    for k, (px, py) in enumerate(zip(x, y)):
        i0, i1 = np.searchsorted(gx, px - R, "left"), np.searchsorted(gx, px + R, "right")
        j0, j1 = np.searchsorted(gy, py - R, "left"), np.searchsorted(gy, py + R, "right")
        if i0 >= i1 or j0 >= j1:
            continue
        d2 = (gx[None, i0:i1] - px) ** 2 + (gy[j0:j1, None] - py) ** 2
        inside = d2 <= R * R
        if inside.any():
            vals = underside.z[j0:j1, i0:i1] - np.sqrt(np.clip(R * R - d2, 0.0, None))
            out[k] = float(vals[inside].min())
    return out


def swept_ball_top(points: np.ndarray, radius: float, grid: Grid,
                   max_step: Optional[float] = None) -> np.ndarray:
    """The top of the volume a ball sweeps along a trajectory: (ny, nx) [m].

    The ball (radius R) moves in straight lines between consecutive rows of
    `points` ((n, 3) centres [m]), as sparlab_form moves a tool between its
    trajectory knots (docs/forming.md, 1.3). At every node x of `grid` the
    result is the highest point of any ball position c over it,

        top(x) = max over c of [c_z + sqrt(R^2 - |x - c_xy|^2)],  |x - c_xy| <= R,

    and -inf where no ball passes over the node. For a ball pushing a sheet
    up from below this is the highest the underside can be pushed. The
    segments are sampled every `max_step` [m] (default h / 4; between two
    samples the envelope is under-estimated by at most max_step^2 / (8 R)).
    """
    R = require_positive("radius", radius)
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("points must be a finite (n >= 1, 3) array")
    step = 0.25 * grid.h if max_step is None else require_positive("max_step", max_step)
    parts = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        m = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        parts.append(a + (np.arange(1, m + 1) / m)[:, None] * (b - a))
    x, y = grid.x, grid.y
    top = np.full(grid.shape, -np.inf)
    for cx, cy, cz in np.vstack(parts):
        i0, i1 = np.searchsorted(x, cx - R, "left"), np.searchsorted(x, cx + R, "right")
        j0, j1 = np.searchsorted(y, cy - R, "left"), np.searchsorted(y, cy + R, "right")
        if i0 >= i1 or j0 >= j1:
            continue
        d2 = (x[None, i0:i1] - cx) ** 2 + (y[j0:j1, None] - cy) ** 2
        cap = np.where(d2 <= R * R, cz + np.sqrt(np.clip(R * R - d2, 0.0, None)), -np.inf)
        win = top[j0:j1, i0:i1]
        np.maximum(win, cap, out=win)
    return top


def signed_outline_distance(reference: HeightMap, eps: float = 1e-6) -> HeightMap:
    """Signed distance [m] of every grid node to the outline of the part of
    `reference` (its nodes deeper than `eps`): negative inside the part,
    positive outside, the outline half-way between the last part node and
    the first flange node (Euclidean distance transforms of the two)."""
    part = reference.mask & (reference.z < -eps)
    if not part.any():
        raise PrecompError("the reference surface has no part (nothing below the sheet plane)")
    h = reference.grid.h
    d_out = ndimage.distance_transform_edt(~part) * h
    d_in = ndimage.distance_transform_edt(part) * h
    sd = np.where(part, -(d_in - 0.5 * h), d_out - 0.5 * h)
    return HeightMap(reference.grid, sd, None, {"quantity": "signed_outline_distance"})


def rim_pass_path(surface: HeightMap, reference: HeightMap, radius: float, thickness: float,
                  *, inside: float = 2e-3, outside: float = 1e-3, band_spacing: float = 1e-3,
                  spacing: float = 1e-3, clearance: float = 2e-3, direction: str = "ccw",
                  start_angle_deg: float = 0.0) -> Toolpath:
    """A support-tool-only pass along the rim, pushing up from below.

    Closed loops at signed distances ``outside, outside - band_spacing, ...``
    down to ``-inside`` from the outline of the part of `reference` (outside
    first, then inwards; `signed_outline_distance`), each point at the
    lift-cutter height of the underside of `surface` (`lift_at_points`,
    exact over the grid's nodes) - the highest the ball can rise there
    without entering the commanded sheet: where the formed sheet sagged below
    the command, the ball pushes it back up to it, and where the command lies
    above the sheet plane (a compensation that raises the rim), up to that.
    What the pass reaches is the top of the volume its ball sweeps
    (`swept_ball_top`). Loops are joined by straight moves kept
    below the lift cutter; the ball starts and ends in the air below the
    whole part (`AIR` points, its top `clearance` below the deepest point of
    `surface` minus `thickness`). `spacing` is the point spacing along a
    loop [m]. Every band level must have one loop (one pocket).
    """
    R = require_positive("radius", radius)
    t = require_positive("thickness", thickness)
    spacing = require_positive("spacing", spacing)
    band_spacing = require_positive("band_spacing", band_spacing)
    if inside < 0 or outside < 0 or inside + outside <= 0:
        raise ValueError("the band is empty: need inside, outside >= 0 and inside + outside > 0")
    sd = signed_outline_distance(reference)
    if not sd.grid.matches(surface.grid):
        sd = sd.resample(surface.grid)
    under = surface.with_z(surface.z - t)
    levels = []
    d = float(outside)
    while d >= -inside - 1e-12:
        levels.append(d)
        d -= band_spacing
    per_level = _ordered_loops(sd, levels, spacing, direction, False, start_angle_deg,
                               _min_loop(surface, spacing, None))
    pts: List[np.ndarray] = []
    lvl: List[np.ndarray] = []
    low = -(t + R + clearance + max(0.0, -float(surface.z.min())))

    def lift(xy: np.ndarray) -> np.ndarray:
        b = lift_at_points(under, R, xy[:, 0], xy[:, 1])
        return np.where(np.isfinite(b) & surface.grid.contains(xy[:, 0], xy[:, 1]), b, low)

    current: Optional[np.ndarray] = None
    for k, (level, loops) in enumerate(per_level, start=1):
        if len(loops) != 1:
            raise PrecompError(f"the rim pass needs one loop per band level; the level "
                               f"{level * 1e3:+.2f} mm from the outline has {len(loops)}")
        xy = resample_loop(loops[0], spacing)
        xy = np.vstack([xy, xy[:1]])
        ring = np.column_stack([xy, lift(xy)])
        if current is None:
            start = np.array([ring[0, 0], ring[0, 1], low])
            pts.append(start[None])
            lvl.append(np.array([AIR]))
        else:
            dist = float(np.linalg.norm(ring[0, :2] - current[:2]))
            m = max(1, int(math.ceil(dist / spacing)))
            u = (np.arange(1, m) / m)[:, None]
            link = current + u * (ring[0] - current)
            if len(link):
                link[:, 2] = np.minimum(link[:, 2], lift(link))
                pts.append(link)
                lvl.append(np.full(len(link), k))
        pts.append(ring)
        lvl.append(np.full(len(ring), k))
        current = ring[-1]
    end = np.array([current[0], current[1], low])
    pts.append(end[None])
    lvl.append(np.array([AIR]))
    meta = {"style": "rim_pass", "band_levels_m": [float(v) for v, _ in per_level],
            "inside": float(inside), "outside": float(outside),
            "band_spacing": band_spacing, "spacing": spacing, "clearance": clearance,
            "direction": direction, "thickness": t}
    return Toolpath(np.vstack(pts), np.concatenate(lvl), R, meta)
