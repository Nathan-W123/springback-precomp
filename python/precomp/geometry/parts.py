"""Parametric part families: the targets of incremental sheet forming.

Every family is a frozen dataclass of physical parameters (metres, angles in
degrees with a `_deg` suffix), centred on the origin of the blank, with

* `height(x, y)` - the tool-side surface z <= 0 [m] at arbitrary points, and
  `heightmap(grid)` the same on a grid (refusing a grid that does not contain
  the part with one spacing to spare);
* `bounds` - the sampling range of every parameter, and `sample(rng)` a part
  drawn uniformly within them, rejecting draws that violate the family's
  constraints (so a sample is always a valid, formable part);
* `max_wall_angle_deg()` - exact where the geometry gives it in closed form,
  otherwise measured on a fine grid; every constructor refuses more than
  `MAX_WALL_ANGLE_DEG`;
* `footprint_radius()` - the radius about the origin outside which z = 0;
* `to_dict()` / `part_from_dict()` with a `family` key, for JSON.

Construction. The cone-like families are a *profile* g(w) of an inward
distance w from the nominal opening line: a polyline (flange, walls, ledges,
floor) whose corners are rounded by circular fillets tangent to both segments
(`FilletedProfile`), so the profile is C1. The distance function is exact for
each family (radial distance, distance to a rounded rectangle, distance to an
ellipse) and has unit gradient wherever the profile slopes, so the wall angle
is exactly the design angle on every straight wall. The constructors require
the sloping part of the profile to stay clear of the distance function's
medial axis (e.g. a rounded-rectangle corner radius at least the horizontal
run of the wall plus the bottom fillet's tangent length), which is what keeps
the surface C1 there; a sharp-cornered pyramid is therefore not in the family.
The flange - everything outside the opening - stays at z = 0.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Sequence, Tuple, Type

import numpy as np

from .heightmap import Grid, HeightMap

#: Forming limit of single-point incremental forming used by every family: the
#: largest wall angle a part may have [deg]. Beyond about 65-70 deg most sheet
#: alloys thin to fracture (the cosine law t = t0 cos(alpha)).
MAX_WALL_ANGLE_DEG = 65.0

#: Tries `sample` makes before giving up on drawing a valid part.
MAX_SAMPLE_TRIES = 2000


# ---------------------------------------------------------------------------
# Profiles and distance functions
# ---------------------------------------------------------------------------
class FilletedProfile:
    """A C1 profile z = g(w) [m]: a polyline with circular fillets at its corners.

    The polyline runs through `vertices` (w_k, z_k) with w strictly
    increasing, and continues horizontally before the first vertex (z = z_0)
    and after the last (z = z_last). Each vertex k is rounded by a circular arc
    of radius `radii[k]` (0 keeps it sharp) tangent to its two segments; the
    tangent length is ``T = r tan(|theta_out - theta_in| / 2)`` with theta the
    segment inclinations. The constructor refuses fillets whose tangent
    lengths overlap on a segment.
    """

    def __init__(self, vertices: Sequence[Tuple[float, float]], radii: Sequence[float]):
        v = np.asarray(vertices, dtype=float)
        r = np.asarray(radii, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2 or len(v) < 1:
            raise ValueError("vertices must be an (n >= 1, 2) array of (w, z)")
        if r.shape != (len(v),):
            raise ValueError("one fillet radius per vertex is required")
        if np.any(np.diff(v[:, 0]) <= 0):
            raise ValueError("profile vertices must have strictly increasing w "
                             "(a vertical or overhanging wall is not a height field)")
        if np.any(r < 0) or not np.all(np.isfinite(r)):
            raise ValueError("fillet radii must be finite and >= 0")
        self.w = v[:, 0]
        self.z = v[:, 1]
        n = len(v)
        # Segment slopes: 0 (flange ray), the n - 1 segments, 0 (floor ray).
        seg = np.diff(self.z) / np.diff(self.w)
        self.slopes = np.concatenate([[0.0], seg, [0.0]])
        theta = np.arctan(self.slopes)
        self.arcs = []
        tangent = np.zeros(n)
        for k in range(n):
            t_in, t_out = theta[k], theta[k + 1]
            delta = t_out - t_in
            if r[k] == 0 or abs(delta) < 1e-15:
                self.arcs.append(None)
                continue
            T = r[k] * math.tan(abs(delta) / 2.0)
            tangent[k] = T
            p_in = np.array([self.w[k], self.z[k]]) - T * np.array([math.cos(t_in), math.sin(t_in)])
            p_out = np.array([self.w[k], self.z[k]]) + T * np.array([math.cos(t_out),
                                                                   math.sin(t_out)])
            up = delta > 0  # slope increases: the centre lies above the curve
            nrm = np.array([-math.sin(t_in), math.cos(t_in)])
            centre = p_in + (r[k] if up else -r[k]) * nrm
            self.arcs.append((p_in[0], p_out[0], centre[0], centre[1], r[k], up))
        for k in range(1, n):
            length = math.hypot(self.w[k] - self.w[k - 1], self.z[k] - self.z[k - 1])
            if tangent[k - 1] + tangent[k] > length * (1 + 1e-12):
                raise ValueError(
                    f"fillets at profile corners {k - 1} and {k} overlap: tangent lengths "
                    f"{tangent[k - 1]:.4g} + {tangent[k]:.4g} m exceed the {length:.4g} m "
                    "segment between them; reduce the fillet radii")
        self.tangent_lengths = tangent

    @property
    def w_first(self) -> float:
        """The smallest w at which the profile departs from z_0 [m]."""
        return float(self.w[0] - self.tangent_lengths[0])

    @property
    def w_last(self) -> float:
        """The largest w at which the profile still varies [m]."""
        return float(self.w[-1] + self.tangent_lengths[-1])

    def __call__(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        z = np.interp(w, self.w, self.z)
        for arc in self.arcs:
            if arc is None:
                continue
            a, b, cw, cz, R, up = arc
            sel = (w >= a) & (w <= b)
            if np.any(sel):
                root = np.sqrt(np.maximum(R * R - (w[sel] - cw) ** 2, 0.0))
                z[sel] = cz - root if up else cz + root
        return z

    def slope(self, w: np.ndarray) -> np.ndarray:
        """dg/dw [-], continuous by construction."""
        w = np.asarray(w, dtype=float)
        idx = np.searchsorted(self.w, w, side="right")
        s = self.slopes[idx]
        for arc in self.arcs:
            if arc is None:
                continue
            a, b, cw, cz, R, up = arc
            sel = (w >= a) & (w <= b)
            if np.any(sel):
                dw = w[sel] - cw
                root = np.sqrt(np.maximum(R * R - dw * dw, 1e-300))
                s[sel] = dw / root if up else -dw / root
        return s


def rounded_rectangle_distance(x: np.ndarray, y: np.ndarray, half_x: float, half_y: float,
                               corner_radius: float) -> np.ndarray:
    """Signed inward distance [m] from the outline of a rounded rectangle.

    The outline has half-widths `half_x`, `half_y` and corner radius
    `corner_radius`, centred on the origin. The value is positive inside.
    It is exact outside the outline and inside it down to a depth of
    `corner_radius` (the region where the level sets are concentric rounded
    rectangles); deeper inside it is capped at `corner_radius` - the families
    only use that region as flat floor.
    """
    cx = half_x - corner_radius
    cy = half_y - corner_radius
    qx = np.maximum(np.abs(x) - cx, 0.0)
    qy = np.maximum(np.abs(y) - cy, 0.0)
    return corner_radius - np.hypot(qx, qy)


def ellipse_distance(x: np.ndarray, y: np.ndarray, a: float, b: float) -> np.ndarray:
    """Signed distance [m] from (x, y) to the ellipse (x/a)^2 + (y/b)^2 = 1,
    positive inside.

    Exact to round-off: the closest point is found by bisection (100 halvings)
    on the one-dimensional root of D. Eberly's formulation ("Distance from a
    point to an ellipse", Geometric Tools, 2013), vectorised, with the points
    on the axes handled in closed form.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    swap = b > a
    e0, e1 = (b, a) if swap else (a, b)
    y0 = np.abs(y if swap else x)
    y1 = np.abs(x if swap else y)
    dist = np.empty(np.broadcast(y0, y1).shape)
    y0, y1 = np.broadcast_arrays(y0, y1)
    general = (y1 > 0) & (y0 > 0)
    on_minor = (y1 > 0) & (y0 == 0)
    on_major = y1 == 0
    # general position
    if np.any(general):
        z0 = y0[general] / e0
        z1 = y1[general] / e1
        g = z0 * z0 + z1 * z1 - 1.0
        r0 = (e0 / e1) ** 2
        n0 = r0 * z0
        s0 = z1 - 1.0
        s1 = np.where(g < 0, 0.0, np.hypot(n0, z1) - 1.0)
        for _ in range(100):
            s = 0.5 * (s0 + s1)
            gs = (n0 / (s + r0)) ** 2 + (z1 / (s + 1.0)) ** 2 - 1.0
            s0 = np.where(gs > 0, s, s0)
            s1 = np.where(gs > 0, s1, s)
        s = 0.5 * (s0 + s1)
        x0 = r0 * y0[general] / (s + r0)
        x1 = y1[general] / (s + 1.0)
        dist[general] = np.hypot(x0 - y0[general], x1 - y1[general])
    if np.any(on_minor):
        dist[on_minor] = np.abs(y1[on_minor] - e1)
    if np.any(on_major):
        yy = y0[on_major]
        numer = e0 * yy
        denom = e0 * e0 - e1 * e1
        inner = numer < denom
        d = np.abs(yy - e0)
        if np.any(inner):
            xde0 = numer[inner] / denom
            x0 = e0 * xde0
            x1 = e1 * np.sqrt(np.maximum(1.0 - xde0 * xde0, 0.0))
            d[inner] = np.hypot(x0 - yy[inner], x1)
        dist[on_major] = d
    inside = (x / a) ** 2 + (y / b) ** 2 < 1.0
    return np.where(inside, dist, -dist)


def _smoothstep5(t: np.ndarray) -> np.ndarray:
    """C2 quintic step: 0 for t <= 0, 1 for t >= 1."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _cone_profile(wall_angle_deg: float, depth: float, top_fillet: float,
                  bottom_fillet: float) -> FilletedProfile:
    run = depth / math.tan(math.radians(wall_angle_deg))
    return FilletedProfile([(0.0, 0.0), (run, -depth)], [top_fillet, bottom_fillet])


def _check_angle(name: str, value: float, low: float = 0.0,
                 high: float = MAX_WALL_ANGLE_DEG) -> None:
    if not (math.isfinite(value) and low < value <= high):
        raise ValueError(f"{name} must lie in ({low}, {high}] deg, got {value!r}")


def _check_positive(**kwargs: float) -> None:
    for name, value in kwargs.items():
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a finite number > 0, got {value!r}")


def _check_nonnegative(**kwargs: float) -> None:
    for name, value in kwargs.items():
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type["Part"]] = {}


@dataclass(frozen=True)
class Part(abc.ABC):
    """A parametric target part (see the module docstring for the contract)."""

    family: ClassVar[str] = ""
    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.family:
            _REGISTRY[cls.family] = cls

    @abc.abstractmethod
    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tool-side surface z(x, y) <= 0 [m] at arbitrary points [m]."""

    @abc.abstractmethod
    def footprint_radius(self) -> float:
        """Radius about the origin [m] outside which the surface is at z = 0."""

    def max_wall_angle_deg(self) -> float:
        """Largest wall angle of the part [deg], measured numerically.

        Central differences on a grid of 1/600 of 1.1 footprint diameters,
        then around the 32 steepest nodes again at 1/20 of that spacing: a
        difference quotient under-reads the peak slope where the curvature
        changes sign (a fillet meeting a cap) by a fraction of a degree on the
        coarse grid, and the refinement brings that to about 0.02 deg. The
        families with a closed form override this.
        """
        r = self.footprint_radius()
        grid = Grid.centered(2.2 * r, 2.2 * r / 600.0)
        X, Y = grid.mesh()
        z = self.height(X, Y)
        gy, gx = np.gradient(z, grid.h)
        slope = np.hypot(gx, gy)
        best = float(slope.max())
        top = np.argsort(slope.ravel())[-32:]
        hf = grid.h / 20.0
        k = np.arange(-20, 21) * hf
        DX, DY = np.meshgrid(k, k, indexing="xy")
        for idx in top:
            cx, cy = X.ravel()[idx], Y.ravel()[idx]
            zz = self.height(cx + DX, cy + DY)
            fy, fx = np.gradient(zz, hf)
            best = max(best, float(np.hypot(fx, fy)[1:-1, 1:-1].max()))
        return float(np.degrees(np.arctan(best)))

    def max_depth(self) -> float:
        """Largest depth max(-z) [m]; measured on a fine grid unless the family
        overrides it with the exact value."""
        r = self.footprint_radius()
        grid = Grid.centered(2.2 * r, 2.2 * r / 600.0)
        X, Y = grid.mesh()
        return float(-self.height(X, Y).min())

    def heightmap(self, grid: Grid) -> HeightMap:
        """The part on `grid`: z [m], all nodes valid, metadata {"part": to_dict()}.

        Raises ValueError when the grid does not contain the footprint circle
        plus one spacing (the flange would be cut off).
        """
        r = self.footprint_radius() + grid.h
        x0, x1, y0, y1 = grid.extent
        if x0 > -r or x1 < r or y0 > -r or y1 < r:
            raise ValueError(
                f"the grid {grid.extent} does not contain the {self.family} footprint "
                f"(radius {self.footprint_radius():.4g} m plus one spacing)")
        X, Y = grid.mesh()
        z = self.height(X, Y)
        return HeightMap(grid, z, None, {"part": self.to_dict()})

    def to_dict(self) -> Dict[str, Any]:
        doc: Dict[str, Any] = {"family": self.family}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = [list(v) if isinstance(v, tuple) else v for v in value]
            doc[f.name] = value
        return doc


    @classmethod
    def _from_fields(cls, doc: Dict[str, Any]) -> "Part":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(doc) - names - {"family"}
        if unknown:
            raise ValueError(f"{cls.family}: unknown parameters {sorted(unknown)}")
        return cls(**{k: v for k, v in doc.items() if k != "family"})

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "Part":
        """A valid part with parameters drawn uniformly within `bounds`.

        Draws that violate a constraint of the family (fillets that do not
        fit, a floor that vanishes, ...) are rejected and redrawn; a
        RuntimeError after MAX_SAMPLE_TRIES failures means the bounds are
        inconsistent.
        """
        last = None
        for _ in range(MAX_SAMPLE_TRIES):
            params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in cls.bounds.items()}
            try:
                return cls(**params)
            except ValueError as exc:
                last = exc
        raise RuntimeError(f"{cls.family}: no valid sample in {MAX_SAMPLE_TRIES} draws; "
                           f"last rejection: {last}")


def part_from_dict(doc: Dict[str, Any]) -> Part:
    """Rebuild a part from `Part.to_dict()` output (dispatch on `family`)."""
    family = doc.get("family")
    if family not in _REGISTRY:
        raise ValueError(f"unknown part family {family!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[family]._from_fields(doc)


def families() -> Dict[str, Type[Part]]:
    """The registered part families by name."""
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TruncatedCone(Part):
    """A circular frustum with a flat floor, filleted at rim and floor.

    top_radius : radius [m] where the straight wall, extended, meets the
        sheet plane (the nominal opening of the drawing; the rim fillet makes
        the real opening larger by its tangent length).
    wall_angle_deg : wall angle from the sheet plane [deg].
    depth : floor depth below the sheet plane [m].
    top_fillet, bottom_fillet : fillet radii in the meridian plane [m].
    """

    top_radius: float
    wall_angle_deg: float
    depth: float
    top_fillet: float = 0.005
    bottom_fillet: float = 0.005

    family: ClassVar[str] = "truncated_cone"
    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {
        "top_radius": (0.035, 0.075), "wall_angle_deg": (30.0, 65.0),
        "depth": (0.01, 0.035), "top_fillet": (0.002, 0.01), "bottom_fillet": (0.002, 0.01)}

    def __post_init__(self) -> None:
        _check_positive(top_radius=self.top_radius, depth=self.depth)
        _check_nonnegative(top_fillet=self.top_fillet, bottom_fillet=self.bottom_fillet)
        _check_angle("wall_angle_deg", self.wall_angle_deg)
        prof = self.profile
        floor = self.top_radius - prof.w_last
        if floor < 0:
            raise ValueError(f"truncated_cone: the floor radius would be {floor:.4g} m < 0; "
                             "increase top_radius or the wall angle, or reduce depth or "
                             "bottom_fillet")

    @property
    def profile(self) -> FilletedProfile:
        return _cone_profile(self.wall_angle_deg, self.depth, self.top_fillet,
                             self.bottom_fillet)

    def height(self, x, y):
        r = np.hypot(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self.profile(self.top_radius - r)

    def footprint_radius(self) -> float:
        return self.top_radius - self.profile.w_first

    def max_wall_angle_deg(self) -> float:
        return float(self.wall_angle_deg)

    def max_depth(self) -> float:
        return float(self.depth)


@dataclass(frozen=True)
class Pyramid(Part):
    """A rectangular frustum ("pyramid") with rounded corners and fillets.

    half_width_x, half_width_y : half-widths of the nominal opening [m].
    wall_angle_deg : wall angle [deg]; depth : floor depth [m].
    corner_radius : plan-view corner radius of the opening [m]. It must be at
        least the horizontal run of the wall, depth / tan(angle), plus the
        bottom fillet's tangent length, so the walls meet at the corners as
        smooth cones rather than along a crease.
    top_fillet, bottom_fillet : fillet radii in the wall's section [m].
    """

    half_width_x: float
    half_width_y: float
    wall_angle_deg: float
    depth: float
    corner_radius: float
    top_fillet: float = 0.004
    bottom_fillet: float = 0.004

    family: ClassVar[str] = "pyramid"
    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {
        "half_width_x": (0.035, 0.07), "half_width_y": (0.035, 0.07),
        "wall_angle_deg": (35.0, 65.0), "depth": (0.008, 0.03),
        "corner_radius": (0.01, 0.05), "top_fillet": (0.002, 0.008),
        "bottom_fillet": (0.002, 0.008)}

    def __post_init__(self) -> None:
        _check_positive(half_width_x=self.half_width_x, half_width_y=self.half_width_y,
                        depth=self.depth, corner_radius=self.corner_radius)
        _check_nonnegative(top_fillet=self.top_fillet, bottom_fillet=self.bottom_fillet)
        _check_angle("wall_angle_deg", self.wall_angle_deg)
        if self.corner_radius > min(self.half_width_x, self.half_width_y):
            raise ValueError("pyramid: corner_radius exceeds the smaller half-width")
        need = self.profile.w_last
        if self.corner_radius < need:
            raise ValueError(
                f"pyramid: corner_radius {self.corner_radius:.4g} m is below the wall's "
                f"run plus bottom fillet tangent, {need:.4g} m; the surface would have a "
                "crease at the corners")

    @property
    def profile(self) -> FilletedProfile:
        return _cone_profile(self.wall_angle_deg, self.depth, self.top_fillet,
                             self.bottom_fillet)

    def height(self, x, y):
        w = rounded_rectangle_distance(np.asarray(x, float), np.asarray(y, float),
                                       self.half_width_x, self.half_width_y, self.corner_radius)
        return self.profile(w)

    def max_depth(self) -> float:
        return float(self.depth)

    def footprint_radius(self) -> float:
        t = -self.profile.w_first
        cx = self.half_width_x - self.corner_radius
        cy = self.half_width_y - self.corner_radius
        return math.hypot(cx, cy) + self.corner_radius + t

    def max_wall_angle_deg(self) -> float:
        return float(self.wall_angle_deg)


@dataclass(frozen=True)
class Dome(Part):
    """A spherical cap blended into the flange by a rim fillet; with
    `aspect` != 1 an ellipsoidal cap (the same shape scaled by `aspect` in y).

    opening_radius : radius along x where the rim fillet meets the sheet plane [m].
    depth : depth of the pole [m], < opening_radius.
    rim_fillet : radius of the fillet between the cap and the flange [m].
    aspect : y-to-x scale of the plan outline [-].

    With a the opening radius, d the depth and f the fillet, the sphere radius
    is R = (a^2 + d^2) / (2 d) - f and the steepest point (where cap and fillet
    meet) has the wall angle asin(2 a d / (a^2 + d^2)), divided in tangent by
    `aspect` when the plan is compressed in y.
    """

    opening_radius: float
    depth: float
    rim_fillet: float = 0.01
    aspect: float = 1.0

    family: ClassVar[str] = "dome"
    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {
        "opening_radius": (0.04, 0.075), "depth": (0.01, 0.035),
        "rim_fillet": (0.004, 0.02), "aspect": (0.75, 1.0)}

    def __post_init__(self) -> None:
        _check_positive(opening_radius=self.opening_radius, depth=self.depth,
                        aspect=self.aspect)
        _check_nonnegative(rim_fillet=self.rim_fillet)
        if self.depth >= self.opening_radius:
            raise ValueError("dome: depth must be below the opening radius (a cap, not a "
                             "hemisphere or deeper)")
        if self.sphere_radius <= 0:
            raise ValueError("dome: rim_fillet is too large for the opening and depth "
                             "(the sphere radius would be <= 0)")
        angle = self.max_wall_angle_deg()
        if angle > MAX_WALL_ANGLE_DEG:
            raise ValueError(f"dome: steepest wall {angle:.1f} deg exceeds "
                             f"{MAX_WALL_ANGLE_DEG} deg")

    @property
    def sphere_radius(self) -> float:
        a, d = self.opening_radius, self.depth
        return (a * a + d * d) / (2.0 * d) - self.rim_fillet

    def _profile(self, rho: np.ndarray) -> np.ndarray:
        a, d, f = self.opening_radius, self.depth, self.rim_fillet
        R = self.sphere_radius
        zc = R - d
        rho_t = a * R / (R + f)
        z = np.zeros_like(rho)
        cap = rho <= rho_t
        z[cap] = zc - np.sqrt(np.maximum(R * R - rho[cap] ** 2, 0.0))
        fil = (rho > rho_t) & (rho < a)
        z[fil] = -f + np.sqrt(np.maximum(f * f - (rho[fil] - a) ** 2, 0.0))
        return z

    def height(self, x, y):
        rho = np.hypot(np.asarray(x, float), np.asarray(y, float) / self.aspect)
        return self._profile(rho)

    def footprint_radius(self) -> float:
        return self.opening_radius * max(1.0, self.aspect)

    def max_depth(self) -> float:
        return float(self.depth)

    def max_wall_angle_deg(self) -> float:
        a, d = self.opening_radius, self.depth
        phi = math.asin(min(1.0, 2.0 * a * d / (a * a + d * d)))
        return float(math.degrees(math.atan(math.tan(phi) * max(1.0, 1.0 / self.aspect))))


@dataclass(frozen=True)
class EllipticCone(Part):
    """An elliptical frustum with a constant wall angle.

    semi_axis_x, semi_axis_y : semi-axes of the nominal opening ellipse [m].
    wall_angle_deg, depth, top_fillet, bottom_fillet : as for TruncatedCone.

    The level sets are exact parallel curves of the ellipse (Euclidean
    distance), so the wall angle is the same all round. The wall's run plus
    the bottom fillet's tangent length must not exceed the smallest radius of
    curvature of the ellipse, b^2 / a, beyond which parallel curves develop
    cusps.
    """

    semi_axis_x: float
    semi_axis_y: float
    wall_angle_deg: float
    depth: float
    top_fillet: float = 0.004
    bottom_fillet: float = 0.004

    family: ClassVar[str] = "elliptic_cone"
    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {
        "semi_axis_x": (0.045, 0.075), "semi_axis_y": (0.035, 0.06),
        "wall_angle_deg": (40.0, 65.0), "depth": (0.008, 0.025),
        "top_fillet": (0.002, 0.008), "bottom_fillet": (0.002, 0.008)}

    def __post_init__(self) -> None:
        _check_positive(semi_axis_x=self.semi_axis_x, semi_axis_y=self.semi_axis_y,
                        depth=self.depth)
        _check_nonnegative(top_fillet=self.top_fillet, bottom_fillet=self.bottom_fillet)
        _check_angle("wall_angle_deg", self.wall_angle_deg)
        a, b = max(self.semi_axis_x, self.semi_axis_y), min(self.semi_axis_x, self.semi_axis_y)
        rmin = b * b / a
        need = self.profile.w_last
        if need > rmin:
            raise ValueError(
                f"elliptic_cone: the wall's run plus bottom fillet tangent, {need:.4g} m, "
                f"exceeds the smallest radius of curvature of the opening, {rmin:.4g} m")

    @property
    def profile(self) -> FilletedProfile:
        return _cone_profile(self.wall_angle_deg, self.depth, self.top_fillet,
                             self.bottom_fillet)

    def height(self, x, y):
        w = ellipse_distance(np.asarray(x, float), np.asarray(y, float),
                             self.semi_axis_x, self.semi_axis_y)
        return self.profile(w)

    def max_depth(self) -> float:
        return float(self.depth)

    def footprint_radius(self) -> float:
        return max(self.semi_axis_x, self.semi_axis_y) - self.profile.w_first

    def max_wall_angle_deg(self) -> float:
        return float(self.wall_angle_deg)


@dataclass(frozen=True)
class TwoLevel(Part):
    """A stepped circular cone: an upper wall, a flat ledge, a lower wall.

    top_radius : nominal opening radius of the upper wall [m].
    wall_angle_1_deg, depth_1 : upper wall angle [deg] and its depth [m].
    ledge_width : radial width of the flat ledge between the walls [m].
    wall_angle_2_deg, depth_2 : lower wall angle [deg] and its additional depth [m].
    fillet_radius : one fillet radius for all four corners [m].
    """

    top_radius: float
    wall_angle_1_deg: float
    depth_1: float
    ledge_width: float
    wall_angle_2_deg: float
    depth_2: float
    fillet_radius: float = 0.003

    family: ClassVar[str] = "two_level"
    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {
        "top_radius": (0.05, 0.075), "wall_angle_1_deg": (30.0, 60.0),
        "depth_1": (0.006, 0.015), "ledge_width": (0.006, 0.015),
        "wall_angle_2_deg": (30.0, 65.0), "depth_2": (0.006, 0.015),
        "fillet_radius": (0.002, 0.005)}

    def __post_init__(self) -> None:
        _check_positive(top_radius=self.top_radius, depth_1=self.depth_1,
                        depth_2=self.depth_2, ledge_width=self.ledge_width)
        _check_nonnegative(fillet_radius=self.fillet_radius)
        _check_angle("wall_angle_1_deg", self.wall_angle_1_deg)
        _check_angle("wall_angle_2_deg", self.wall_angle_2_deg)
        floor = self.top_radius - self.profile.w_last
        if floor < 0:
            raise ValueError(f"two_level: the floor radius would be {floor:.4g} m < 0")

    @property
    def profile(self) -> FilletedProfile:
        r1 = self.depth_1 / math.tan(math.radians(self.wall_angle_1_deg))
        r2 = self.depth_2 / math.tan(math.radians(self.wall_angle_2_deg))
        d1, d2 = self.depth_1, self.depth_1 + self.depth_2
        v = [(0.0, 0.0), (r1, -d1), (r1 + self.ledge_width, -d1),
             (r1 + self.ledge_width + r2, -d2)]
        return FilletedProfile(v, [self.fillet_radius] * 4)

    def height(self, x, y):
        r = np.hypot(np.asarray(x, float), np.asarray(y, float))
        return self.profile(self.top_radius - r)

    def footprint_radius(self) -> float:
        return self.top_radius - self.profile.w_first

    def max_wall_angle_deg(self) -> float:
        return float(max(self.wall_angle_1_deg, self.wall_angle_2_deg))

    def max_depth(self) -> float:
        return float(self.depth_1 + self.depth_2)


@dataclass(frozen=True)
class Freeform(Part):
    """A smooth free-form pocket: a sum of Gaussian dents under a radial window.

    ``z = -W(r) sum_k A_k exp(-|p - c_k|^2 / (2 s_k^2))`` with W = 1 inside
    `inner_radius`, falling to 0 at `outer_radius` along a C2 quintic step,
    and 0 beyond (the flange).

    dents : tuple of (cx [m], cy [m], amplitude [m] > 0, sigma [m] > 0).
    inner_radius, outer_radius : the window [m].
    seed : the generator seed a sampled part came from (informational).

    The wall angle is measured on a fine grid; the constructor refuses more
    than MAX_WALL_ANGLE_DEG, and `sample` scales the amplitudes down to at
    most `SAMPLE_WALL_ANGLE_DEG` (the "slope limit") and, in `sample`, the
    depth to at most `SAMPLE_MAX_DEPTH` [m].
    """

    dents: Tuple[Tuple[float, float, float, float], ...]
    inner_radius: float = 0.05
    outer_radius: float = 0.07
    seed: int = -1

    family: ClassVar[str] = "freeform"
    #: Sampling hyper-parameters (not constructor arguments): the dent count,
    #: amplitude [m], sigma [m], the dent centre's radius as a fraction of
    #: the inner radius, the inner radius [m] and the window width [m].
    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {
        "n_dents": (1, 4), "amplitude": (0.005, 0.025), "sigma": (0.01, 0.025),
        "centre_radius_fraction": (0.0, 0.5), "inner_radius": (0.045, 0.06),
        "window_width": (0.012, 0.02)}
    SAMPLE_WALL_ANGLE_DEG: ClassVar[float] = 55.0
    SAMPLE_MAX_DEPTH: ClassVar[float] = 0.035

    def __post_init__(self) -> None:
        dents = tuple(tuple(float(v) for v in d) for d in self.dents)
        object.__setattr__(self, "dents", dents)
        object.__setattr__(self, "seed", int(self.seed))
        if not dents:
            raise ValueError("freeform: at least one dent is required")
        for d in dents:
            if len(d) != 4:
                raise ValueError("freeform: each dent is (cx, cy, amplitude, sigma)")
            _check_positive(amplitude=d[2], sigma=d[3])
        _check_positive(inner_radius=self.inner_radius)
        if not self.outer_radius > self.inner_radius:
            raise ValueError("freeform: outer_radius must exceed inner_radius")
        angle = self.max_wall_angle_deg()
        if angle > MAX_WALL_ANGLE_DEG:
            raise ValueError(f"freeform: steepest wall {angle:.1f} deg exceeds "
                             f"{MAX_WALL_ANGLE_DEG} deg")

    def height(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        s = np.zeros(np.broadcast(x, y).shape)
        for cx, cy, amp, sig in self.dents:
            s += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sig * sig))
        r = np.hypot(x, y)
        window = 1.0 - _smoothstep5((r - self.inner_radius) / (self.outer_radius -
                                                                self.inner_radius))
        return -window * s

    def footprint_radius(self) -> float:
        return float(self.outer_radius)

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "Freeform":
        """Draw 1-4 dents within `bounds`, then scale the amplitudes so the
        steepest wall is at most SAMPLE_WALL_ANGLE_DEG and the depth at most
        SAMPLE_MAX_DEPTH."""
        b = cls.bounds
        n = int(rng.integers(int(b["n_dents"][0]), int(b["n_dents"][1]) + 1))
        inner = float(rng.uniform(*b["inner_radius"]))
        outer = inner + float(rng.uniform(*b["window_width"]))
        dents = []
        for _ in range(n):
            rad = float(rng.uniform(*b["centre_radius_fraction"])) * inner
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            dents.append((rad * math.cos(ang), rad * math.sin(ang),
                          float(rng.uniform(*b["amplitude"])), float(rng.uniform(*b["sigma"]))))
        seed = int(rng.integers(0, 2**31 - 1))
        # Measure the slope of the unscaled shape with the constructor's check
        # bypassed, then scale: the slope is linear in the amplitudes.
        probe = object.__new__(cls)
        object.__setattr__(probe, "dents", tuple(dents))
        object.__setattr__(probe, "inner_radius", inner)
        object.__setattr__(probe, "outer_radius", outer)
        object.__setattr__(probe, "seed", seed)
        slope = math.tan(math.radians(Part.max_wall_angle_deg(probe)))
        limit = math.tan(math.radians(cls.SAMPLE_WALL_ANGLE_DEG))
        factor = min(1.0, limit / slope) if slope > 0 else 1.0
        factor = min(factor, cls.SAMPLE_MAX_DEPTH / Part.max_depth(probe))
        dents = [(cx, cy, a * factor, s) for cx, cy, a, s in dents]
        return cls(tuple(dents), inner, outer, seed)

    @classmethod
    def from_seed(cls, seed: int) -> "Freeform":
        """The part `sample` draws from `numpy.random.default_rng(seed)`."""
        return cls.sample(np.random.default_rng(seed))


@dataclass(frozen=True)
class Saddle(Part):
    """An anticlastic tray: a rounded-rectangle pocket whose floor is a saddle.

    ``z = -D(x, y) phi(w)`` with w the inward distance from the rounded
    rectangle outline, phi the unit-depth filleted wall profile (0 on the
    flange, 1 on the floor) and ``D = depth + saddle_amplitude ((y/hy)^2 -
    (x/hx)^2)``, so the floor rises along x and falls along y (negative
    Gaussian curvature). The wall angle is `wall_angle_deg` where D = depth
    and varies with D elsewhere; the largest is measured on a fine grid.

    half_width_x, half_width_y, corner_radius : the opening [m] (corner radius
        as for Pyramid).
    wall_angle_deg : nominal wall angle at D = depth [deg].
    depth : floor depth at the centre [m].
    saddle_amplitude : depth change at the floor edges [m], < depth.
    """

    half_width_x: float
    half_width_y: float
    corner_radius: float
    wall_angle_deg: float
    depth: float
    saddle_amplitude: float
    top_fillet: float = 0.004
    bottom_fillet: float = 0.004

    family: ClassVar[str] = "saddle"
    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {
        "half_width_x": (0.045, 0.07), "half_width_y": (0.04, 0.065),
        "corner_radius": (0.02, 0.04), "wall_angle_deg": (30.0, 50.0),
        "depth": (0.01, 0.02), "saddle_amplitude": (0.001, 0.006),
        "top_fillet": (0.002, 0.006), "bottom_fillet": (0.002, 0.006)}

    def __post_init__(self) -> None:
        _check_positive(half_width_x=self.half_width_x, half_width_y=self.half_width_y,
                        corner_radius=self.corner_radius, depth=self.depth)
        _check_nonnegative(saddle_amplitude=self.saddle_amplitude, top_fillet=self.top_fillet,
                           bottom_fillet=self.bottom_fillet)
        _check_angle("wall_angle_deg", self.wall_angle_deg)
        if self.saddle_amplitude >= self.depth:
            raise ValueError("saddle: saddle_amplitude must be below depth")
        if self.corner_radius > min(self.half_width_x, self.half_width_y):
            raise ValueError("saddle: corner_radius exceeds the smaller half-width")
        need = self.profile.w_last
        if self.corner_radius < need:
            raise ValueError(f"saddle: corner_radius {self.corner_radius:.4g} m is below the "
                             f"wall's run plus bottom fillet tangent, {need:.4g} m")
        angle = self.max_wall_angle_deg()
        if angle > MAX_WALL_ANGLE_DEG:
            raise ValueError(f"saddle: steepest wall {angle:.1f} deg exceeds "
                             f"{MAX_WALL_ANGLE_DEG} deg")

    @property
    def profile(self) -> FilletedProfile:
        return _cone_profile(self.wall_angle_deg, self.depth, self.top_fillet,
                             self.bottom_fillet)

    def height(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        w = rounded_rectangle_distance(x, y, self.half_width_x, self.half_width_y,
                                       self.corner_radius)
        phi = -self.profile(w) / self.depth
        D = self.depth + self.saddle_amplitude * ((y / self.half_width_y) ** 2
                                                  - (x / self.half_width_x) ** 2)
        return -D * phi

    def footprint_radius(self) -> float:
        t = -self.profile.w_first
        cx = self.half_width_x - self.corner_radius
        cy = self.half_width_y - self.corner_radius
        return math.hypot(cx, cy) + self.corner_radius + t
