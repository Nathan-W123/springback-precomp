"""The tool-side surface of a sheet as a height field on a regular grid.

Convention (see `docs/precomp.md`): the blank is flat in the x-y plane with
its tool-side (top) surface at z = 0 before forming and the material below it.
Single-point incremental forming pushes the tool down (-z), so a formed part
is a height field z = f(x, y) <= 0 of the tool-side surface. Targets,
commanded shapes, formed shapes and gridded scans all use this one class.

The grid is regular with one spacing `h` in x and y. Arrays are indexed
`[row, column] = [j, i]` with the row along y and x fastest (C order) - the
same layout as SparLab's structured grids, so a C-order reshape of a
row-major node list gives the same picture. Row 0 is the *lowest* y.

Every length is in metres, every angle in radians unless its name says `_deg`.

Where this fits: the surface type of every stage (targets, commands, formed shapes, scans);
    HeightMap.from_points (scans) serves benchmarks/physics_audit, not the three scripts.
Called from: geometry/parts.py:Part.heightmap, fea/results.py:FormingResult.formed_surface,
    ml/dataset.py:Sample.load.
Calls into: scipy (ndimage, interpolate, spatial), _util.
Start reading at: Grid, then HeightMap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.spatial import cKDTree

from .._util import PathLike, PrecompError, to_jsonable

#: Tolerance, relative to the spacing, used when two grids are compared.
GRID_MATCH_RTOL = 1e-6


@dataclass(frozen=True)
class Grid:
    """A regular grid of nodes `x0 + i h`, `y0 + j h`, `i < nx`, `j < ny` [m].

    `x0`, `y0` are the coordinates of node (0, 0), the lowest-x, lowest-y
    corner; `h` is the spacing in both directions. The grid has `nx * ny`
    nodes and spans `(nx - 1) h` by `(ny - 1) h`.
    """

    x0: float
    y0: float
    nx: int
    ny: int
    h: float

    def __post_init__(self) -> None:
        if not (isinstance(self.nx, (int, np.integer)) and isinstance(self.ny, (int, np.integer))):
            raise ValueError(f"Grid nx, ny must be integers, got {self.nx!r}, {self.ny!r}")
        if self.nx < 2 or self.ny < 2:
            raise ValueError(f"Grid needs at least 2 x 2 nodes, got {self.nx} x {self.ny}")
        if not (math.isfinite(self.h) and self.h > 0):
            raise ValueError(f"Grid spacing h must be finite and > 0, got {self.h!r}")
        if not (math.isfinite(self.x0) and math.isfinite(self.y0)):
            raise ValueError("Grid origin must be finite")
        object.__setattr__(self, "nx", int(self.nx))
        object.__setattr__(self, "ny", int(self.ny))
        object.__setattr__(self, "x0", float(self.x0))
        object.__setattr__(self, "y0", float(self.y0))
        object.__setattr__(self, "h", float(self.h))

    # -- construction ------------------------------------------------------
    @classmethod
    def centered(cls, size: float, spacing: float,
                 center: Tuple[float, float] = (0.0, 0.0)) -> "Grid":
        """A square grid of side `size` [m] centred on `center`, spacing `spacing` [m].

        The number of intervals is `round(size / spacing)`, so the side is
        `size` rounded to a whole number of spacings (the spacing is kept
        exactly); the node count per side is that plus one.
        """
        if not (size > 0 and spacing > 0):
            raise ValueError(f"size and spacing must be > 0, got {size!r}, {spacing!r}")
        n = int(round(size / spacing))
        if n < 1:
            raise ValueError(f"size {size} is smaller than the spacing {spacing}")
        half = 0.5 * n * spacing
        return cls(center[0] - half, center[1] - half, n + 1, n + 1, spacing)


    # -- coordinates -------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        """(ny, nx): the array shape of a field on this grid."""
        return (self.ny, self.nx)

    @property
    def x(self) -> np.ndarray:
        """(nx,) node x coordinates [m]."""
        return self.x0 + self.h * np.arange(self.nx)

    @property
    def y(self) -> np.ndarray:
        """(ny,) node y coordinates [m]."""
        return self.y0 + self.h * np.arange(self.ny)

    @property
    def xmax(self) -> float:
        return self.x0 + self.h * (self.nx - 1)

    @property
    def ymax(self) -> float:
        return self.y0 + self.h * (self.ny - 1)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) [m]."""
        return (self.x0, self.xmax, self.y0, self.ymax)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """(X, Y), each (ny, nx), the node coordinates [m]."""
        return np.meshgrid(self.x, self.y, indexing="xy")

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean array: True where (x, y) lies inside the closed grid extent."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        tol = 1e-9 * self.h
        return ((x >= self.x0 - tol) & (x <= self.xmax + tol)
                & (y >= self.y0 - tol) & (y <= self.ymax + tol))

    def matches(self, other: "Grid") -> bool:
        """True when `other` has the same nodes (origin and spacing equal to
        1e-6 of the spacing)."""
        tol = GRID_MATCH_RTOL * self.h
        return (self.nx == other.nx and self.ny == other.ny
                and abs(self.h - other.h) <= tol / max(self.nx, self.ny)
                and abs(self.x0 - other.x0) <= tol and abs(self.y0 - other.y0) <= tol)

    def to_dict(self) -> Dict[str, Any]:
        return {"x0": self.x0, "y0": self.y0, "nx": self.nx, "ny": self.ny, "h": self.h}

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "Grid":
        try:
            return cls(float(doc["x0"]), float(doc["y0"]), int(doc["nx"]), int(doc["ny"]),
                       float(doc["h"]))
        except KeyError as exc:
            raise PrecompError(f"grid description lacks the key {exc}") from exc


def _fill_nearest(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Copy of `z` whose invalid entries take the value of the nearest valid node."""
    if valid.all():
        return z.copy()
    if not valid.any():
        raise PrecompError("no valid node to fill from")
    _, (jj, ii) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return z[jj, ii]


def gaussian_derivative_kernels(sigma_cells: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-D Gaussian smoothing, first- and second-derivative kernels (per cell).

    Truncated at 4 sigma and normalised by their moments, not by the
    continuous formula: the smoothing kernel sums to 1, the first-derivative
    kernel k1 has sum(k1 x) = 1, the second-derivative kernel k2 has sum(k2) = 0
    and sum(k2 x^2) / 2 = 1. So a constant offset contributes nothing to a
    derivative and the derivatives of any quadratic are exact.
    (`scipy.ndimage.gaussian_filter` with `order=2` does not sum to zero, and
    on a surface far from z = 0 that leaks the height into the curvature.)
    """
    radius = max(1, int(4.0 * sigma_cells + 0.5))
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-0.5 * (x / sigma_cells) ** 2)
    k0 = g / g.sum()
    k1 = x * g
    k1 = k1 / np.sum(k1 * x)
    k2 = (x * x - np.sum(x * x * g) / g.sum()) * g
    k2 = k2 / (0.5 * np.sum(k2 * x * x))
    return k0, k1, k2


def _derivatives(z: np.ndarray, h: float, sigma: Optional[float], orders) -> list:
    """Derivatives of `z` [m] w.r.t. (x, y) [m] for each (ox, oy) in `orders`.

    Without `sigma`: second-order central differences (one-sided, first order
    at the border) via `np.gradient`, applied twice for second derivatives.
    With `sigma` [m]: derivatives of the Gaussian-smoothed field at that length
    scale (separable moment-normalised kernels, `gaussian_derivative_kernels`,
    border mode 'nearest').
    """
    out = []
    if sigma is None or sigma <= 0:
        gy, gx = np.gradient(z, h)
        cache: Dict[Tuple[int, int], np.ndarray] = {(0, 0): z, (1, 0): gx, (0, 1): gy}
        for ox, oy in orders:
            if (ox, oy) not in cache:
                if (ox, oy) == (2, 0):
                    cache[(2, 0)] = np.gradient(gx, h, axis=1)
                elif (ox, oy) == (0, 2):
                    cache[(0, 2)] = np.gradient(gy, h, axis=0)
                elif (ox, oy) == (1, 1):
                    cache[(1, 1)] = 0.5 * (np.gradient(gx, h, axis=0) + np.gradient(gy, h, axis=1))
                else:
                    raise ValueError(f"derivative order {(ox, oy)} is not supported")
            out.append(cache[(ox, oy)])
        return out
    kernels = gaussian_derivative_kernels(float(sigma) / h)
    for ox, oy in orders:
        if ox > 2 or oy > 2:
            raise ValueError(f"derivative order {(ox, oy)} is not supported")
        d = ndimage.correlate1d(z, kernels[ox], axis=1, mode="nearest")
        d = ndimage.correlate1d(d, kernels[oy], axis=0, mode="nearest")
        out.append(d / h ** (ox + oy))
    return out


@dataclass
class HeightMap:
    """A height field z(x, y) [m] of the tool-side surface on a `Grid`.

    Attributes
    ----------
    grid : Grid
    z : (ny, nx) float array [m]. Finite everywhere: nodes without data carry
        a fill value (the nearest valid height) so derivatives and
        interpolation stay defined, and are marked invalid in `mask`.
    mask : (ny, nx) bool array, True where `z` is a real value (measured,
        simulated or designed). Defaults to all True.
    metadata : free-form JSON-serialisable dictionary (provenance, the part
        description, units of a non-height field, ...).
    """

    grid: Grid
    z: np.ndarray
    mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z = np.array(self.z, dtype=float, copy=True)
        if self.z.shape != self.grid.shape:
            raise ValueError(f"z has shape {self.z.shape}, the grid needs {self.grid.shape}")
        if self.mask is None:
            self.mask = np.ones(self.grid.shape, dtype=bool)
        else:
            self.mask = np.array(self.mask, dtype=bool, copy=True)
            if self.mask.shape != self.grid.shape:
                raise ValueError(f"mask has shape {self.mask.shape}, the grid needs "
                                 f"{self.grid.shape}")
        if not np.all(np.isfinite(self.z)):
            raise ValueError("z must be finite at every node; mark missing data in `mask` "
                             "and give those nodes a fill value")
        self.metadata = dict(self.metadata)

    # -- basic views -------------------------------------------------------
    def copy(self) -> "HeightMap":
        return HeightMap(self.grid, self.z, self.mask, json.loads(json.dumps(
            to_jsonable(self.metadata))))

    def with_z(self, z: np.ndarray, mask: Optional[np.ndarray] = None,
               metadata: Optional[Dict[str, Any]] = None) -> "HeightMap":
        """A new map on the same grid with heights `z` [m] (mask and metadata
        kept unless given)."""
        return HeightMap(self.grid, z, self.mask if mask is None else mask,
                         dict(self.metadata) if metadata is None else metadata)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    @property
    def depth(self) -> float:
        """Largest depth below the sheet plane, max(-z) over valid nodes [m] (>= 0)."""
        if not self.mask.any():
            return 0.0
        return float(max(0.0, -self.z[self.mask].min()))

    # -- interpolation -----------------------------------------------------
    def _interpolator(self, values: np.ndarray, fill_value: float) -> RegularGridInterpolator:
        return RegularGridInterpolator((self.grid.y, self.grid.x), values, method="linear",
                                       bounds_error=False, fill_value=fill_value)

    def interpolate(self, x: np.ndarray, y: np.ndarray, *, masked: bool = True,
                    fill_value: float = np.nan) -> np.ndarray:
        """Bilinear interpolation of z at arbitrary points (x, y) [m].

        Points outside the grid get `fill_value`. With `masked`, a point whose
        bilinear stencil touches an invalid node (with non-zero weight) also
        gets `fill_value`: the interpolated mask indicator is 1 exactly when
        every contributing node is valid.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.stack([y.ravel(), x.ravel()], axis=-1)
        out = self._interpolator(self.z, fill_value)(pts)
        if masked and not self.mask.all():
            ind = self._interpolator(self.mask.astype(float), 0.0)(pts)
            out = np.where(ind > 1.0 - 1e-9, out, fill_value)
        return out.reshape(x.shape)

    def sample(self, values: np.ndarray, x: np.ndarray, y: np.ndarray,
               fill_value: float = np.nan) -> np.ndarray:
        """Bilinear interpolation of any field `values` on this grid
        ((ny, nx) or (ny, nx, k)) at points (x, y) [m]."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape[:2] != self.grid.shape:
            raise ValueError(f"field has shape {values.shape}, the grid needs {self.grid.shape}")
        pts = np.stack([y.ravel(), x.ravel()], axis=-1)
        out = self._interpolator(values, fill_value)(pts)
        return out.reshape(x.shape + values.shape[2:])

    # -- differential geometry --------------------------------------------
    def gradient(self, sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(dz/dx, dz/dy), each (ny, nx) [-].

        `sigma` [m] selects Gaussian-smoothed derivatives at that length scale;
        None uses central differences of the raw grid (second order inside,
        first order on the border).
        """
        gx, gy = _derivatives(self.z, self.grid.h, sigma, [(1, 0), (0, 1)])
        return gx, gy

    def normals(self, sigma: Optional[float] = None) -> np.ndarray:
        """(ny, nx, 3) unit normals (-z_x, -z_y, 1) / |.|, pointing +z (towards the tool)."""
        gx, gy = self.gradient(sigma)
        n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def wall_angle(self, sigma: Optional[float] = None) -> np.ndarray:
        """(ny, nx) wall angle arctan(|grad z|) [rad], 0 on a flat region."""
        gx, gy = self.gradient(sigma)
        return np.arctan(np.hypot(gx, gy))

    def curvature(self, sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(H, K): mean [1/m] and Gaussian [1/m^2] curvature of the graph surface.

        With p = z_x, q = z_y and w = 1 + p^2 + q^2:
        ``K = (z_xx z_yy - z_xy^2) / w^2`` and
        ``H = ((1 + p^2) z_yy - 2 p q z_xy + (1 + q^2) z_xx) / (2 w^(3/2))``.
        H is positive where the surface bends towards its +z normal: the bottom
        of the bowl z = a r^2 has H = 2a and K = 4a^2. Derivatives come from
        `sigma` [m] as in `gradient`; curvature from raw central differences is
        noisy on scanned data, so pass a length scale of a few grid spacings.
        """
        p, q, zxx, zyy, zxy = _derivatives(self.z, self.grid.h, sigma,
                                           [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)])
        w = 1.0 + p * p + q * q
        K = (zxx * zyy - zxy * zxy) / (w * w)
        H = ((1.0 + p * p) * zyy - 2.0 * p * q * zxy + (1.0 + q * q) * zxx) / (2.0 * w ** 1.5)
        return H, K


    # -- resampling and conversion ----------------------------------------
    def resample(self, grid: Grid) -> "HeightMap":
        """Bilinear resampling onto `grid`. Nodes outside this map's extent,
        or whose stencil touches invalid data, are invalid in the result and
        filled with the nearest valid height."""
        X, Y = grid.mesh()
        z = self.interpolate(X, Y, masked=True, fill_value=np.nan)
        valid = np.isfinite(z)
        if not valid.any():
            raise PrecompError("resample: the target grid does not overlap this height map")
        z = _fill_nearest(np.where(valid, z, 0.0), valid)
        return HeightMap(grid, z, valid, dict(self.metadata))

    def to_points(self, only_valid: bool = True) -> np.ndarray:
        """(n, 3) array of node coordinates (x, y, z) [m], row-major (x fastest)."""
        X, Y = self.grid.mesh()
        pts = np.stack([X.ravel(), Y.ravel(), self.z.ravel()], axis=-1)
        return pts[self.mask.ravel()] if only_valid else pts

    @classmethod
    def from_points(cls, points: np.ndarray, grid: Grid, method: str = "linear",
                    max_gap: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> "HeightMap":
        """Grid scattered points (n, 3) [m] onto `grid`.

        Linear interpolation on the Delaunay triangulation of the points' x-y
        positions (`scipy.interpolate.griddata`). Nodes outside the convex hull
        of the data, and with `max_gap` [m] nodes farther than that from every
        data point (holes in a scan), are invalid and filled with the nearest
        valid value. The points must form a height field: several points above
        one x-y position (an overhang) are not representable.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("from_points needs an (n >= 3, 3) array of x, y, z")
        if not np.all(np.isfinite(points)):
            raise ValueError("from_points: the points contain NaN or infinity")
        X, Y = grid.mesh()
        z = griddata(points[:, :2], points[:, 2], (X, Y), method=method)
        valid = np.isfinite(z)
        if max_gap is not None:
            dist, _ = cKDTree(points[:, :2]).query(np.stack([X.ravel(), Y.ravel()], -1))
            valid &= dist.reshape(grid.shape) <= max_gap
        if not valid.any():
            raise PrecompError("from_points: no grid node lies inside the data")
        z = _fill_nearest(np.where(valid, z, 0.0), valid)
        return cls(grid, z, valid, dict(metadata or {}))


    # -- persistence -------------------------------------------------------
    def save(self, path: PathLike) -> Path:
        """Write a compressed `.npz` with z, mask, the grid and the metadata (JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            np.savez_compressed(
                handle, z=self.z, mask=self.mask,
                grid=np.array(json.dumps(self.grid.to_dict())),
                metadata=np.array(json.dumps(to_jsonable(self.metadata), sort_keys=True)),
                format=np.array("precomp.HeightMap/1"))
        return path

    @classmethod
    def load(cls, path: PathLike) -> "HeightMap":
        """Read a map written by `save`."""
        path = Path(path)
        if not path.is_file():
            raise PrecompError(f"{path} does not exist")
        with np.load(path, allow_pickle=False) as data:
            for key in ("z", "mask", "grid", "metadata"):
                if key not in data:
                    raise PrecompError(f"{path} is not a precomp height map: no '{key}' array")
            grid = Grid.from_dict(json.loads(str(data["grid"])))
            return cls(grid, data["z"], data["mask"], json.loads(str(data["metadata"])))


def zeros(grid: Grid, metadata: Optional[Dict[str, Any]] = None) -> HeightMap:
    """The flat, unformed blank on `grid` (z = 0 everywhere)."""
    return HeightMap(grid, np.zeros(grid.shape), None, dict(metadata or {}))
