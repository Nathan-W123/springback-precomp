"""Point features of a commanded surface: the versioned input schema of precomp.ml.

The learning target of the package is the vertical deviation
dz(x, y) = z_formed(x, y) - z_commanded(x, y) [m] of the tool-side surface after
springback, as a function of the COMMANDED surface (during compensation the
model is queried at compensated shapes), the process and the material. This
module turns a commanded `HeightMap` and a `FormingSetup` into one feature
vector per grid node.

Feature kinds
-------------
local      functions of the surface around the node that do not change when the
           whole part is moved in x-y on the grid: depth, wall angle and
           curvature at three length scales, signed distance to the rim,
           position relative to the part's centroid, the pseudo-time at which
           the tool passes, and annular ring descriptors;
placement  where the node lies on the blank (distance to the clamped frame,
           radius from the blank centre): what the unclamping springback of the
           whole sheet depends on;
global     one value per part (depth, area, volume, wall angles, perimeter,
           aspect, second moments of the depth field);
process    tool radius, step-down, thickness, friction;
material   elasticity, hardening descriptors, r-values, back-stress saturation.

The list, the units and the descriptions are `FeatureSpec` records;
`FeatureConfig.schema_hash()` fingerprints the version, the names, the units
and the configuration, and a saved model refuses to load under another schema.
Every feature is finite for any valid commanded surface with a part on it.

Lengths are metres, angles radians, stresses pascals; curvatures are made
dimensionless by their smoothing length (H sigma, K sigma^2) so the three scales
are comparable.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage, signal
from scipy.spatial import cKDTree

from .._util import canonical_json, sha256_bytes
from ..fea.setup import FormingSetup
from ..geometry.heightmap import HeightMap
from ..metrology import wall_mask
from ..toolpath import AIR, Toolpath

#: Version of the feature schema. Bump it whenever a feature's definition
#: changes (not only its name): a model trained under one version must not be
#: fed features computed under another. Version 2 renamed yield_over_E to
#: flow_stress_20_over_E (the value, the flow stress at 20 % plastic strain
#: over E, was never the yield stress over E).
FEATURE_SCHEMA_VERSION = "2"

#: Feature kinds, in the order they appear in the feature vector.
KINDS = ("local", "placement", "global", "process", "material")

#: Region codes returned by `region_labels` (index = code).
REGION_NAMES = ("flange", "rim", "wall", "base")

#: Name of the extra column a physics prior adds (see models.ResidualModel).
PRIOR_FEATURE = "prior_dz"

#: Plastic strains [-] at which the flow stress is reported.
FLOW_STRAINS = (0.2, 0.5)

#: Plastic strain [-] at which the monotonic back stress stands for its saturation.
BACKSTRESS_STRAIN = 0.5


@dataclass(frozen=True)
class FeatureSpec:
    """One feature: `name`, `unit` ("m", "rad", "Pa", "-", ...), a one-line
    `description` and its `kind` (one of KINDS, or "prior")."""

    name: str
    unit: str
    description: str
    kind: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FeatureConfig:
    """How the features are computed; part of the schema of a trained model.

    scales : smoothing lengths of the wall-angle and curvature features, as
        multiples of the tool radius (three by default: 1, 2.5, 6).
    rings : number of annular rings K around each node; ring k spans radii
        ((k - 1) r, k r], r = ring_max * tool radius / K.
    ring_max : outer radius of the last ring, in tool radii.
    time_source : "toolpath" - the pseudo-time of the tool path at the point
        where the tool passes nearest the node (the setup's own path,
        `precomp.fea.make_toolpath`, when none is given); "depth" - the depth
        fraction instead (cheap, and monotone in the pseudo-time of a
        z-level or spiral path to first order). Chosen once per model; never
        substituted silently.
    part_eps : depth [m] below which a node belongs to the flange.
    rim_width : width of the rim band inside the part, in tool radii
        (`region_labels`).
    """

    scales: Tuple[float, ...] = (1.0, 2.5, 6.0)
    rings: int = 4
    ring_max: float = 4.0
    time_source: str = "toolpath"
    part_eps: float = 1e-6
    rim_width: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scales", tuple(float(s) for s in self.scales))
        if not self.scales or any(not (s > 0 and math.isfinite(s)) for s in self.scales):
            raise ValueError(f"scales must be positive multiples of the tool radius, "
                             f"got {self.scales!r}")
        if not (isinstance(self.rings, (int, np.integer)) and self.rings >= 1):
            raise ValueError(f"rings must be an integer >= 1, got {self.rings!r}")
        object.__setattr__(self, "rings", int(self.rings))
        if not self.ring_max > 0:
            raise ValueError("ring_max must be > 0")
        if self.time_source not in ("toolpath", "depth"):
            raise ValueError(f"time_source must be 'toolpath' or 'depth', got "
                             f"{self.time_source!r}")
        if not self.part_eps > 0 or not self.rim_width > 0:
            raise ValueError("part_eps and rim_width must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["scales"] = list(self.scales)
        return d

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "FeatureConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(doc) - names
        if unknown:
            raise ValueError(f"FeatureConfig: unknown keys {sorted(unknown)}")
        return cls(**{k: (tuple(v) if k == "scales" else v) for k, v in doc.items()})

    # -- schema --------------------------------------------------------------
    def specs(self) -> List[FeatureSpec]:
        """The features in vector order."""
        out: List[FeatureSpec] = []
        L = out.append
        L(FeatureSpec("depth", "m", "depth below the sheet plane, -z (>= 0)", "local"))
        L(FeatureSpec("depth_frac", "-", "depth / the part's maximum depth", "local"))
        for k, s in enumerate(self.scales, 1):
            L(FeatureSpec(f"wall_angle_s{k}", "rad",
                          f"wall angle arctan|grad z| after Gaussian smoothing at "
                          f"{s:g} tool radii", "local"))
        for k, s in enumerate(self.scales, 1):
            L(FeatureSpec(f"mean_curv_s{k}", "-",
                          f"mean curvature H times the smoothing length sigma = {s:g} tool "
                          f"radii (H > 0 where the surface bends towards the tool)", "local"))
        for k, s in enumerate(self.scales, 1):
            L(FeatureSpec(f"gauss_curv_s{k}", "-",
                          f"Gaussian curvature K times sigma^2, sigma = {s:g} tool radii",
                          "local"))
        L(FeatureSpec("rim_dist", "m", "signed distance to the part boundary (the rim where "
                      "the depth departs from 0), positive inside the part", "local"))
        L(FeatureSpec("radial_norm", "-", "distance from the part's area centroid / the "
                      "part's equivalent radius sqrt(area / pi)", "local"))
        L(FeatureSpec("polar_sin", "-", "sine of the polar angle about the part centroid, "
                      "regularised: dy / sqrt(r^2 + R^2), R the tool radius (0 at the "
                      "centroid, where the angle is undefined)", "local"))
        L(FeatureSpec("polar_cos", "-", "cosine of the polar angle about the part centroid, "
                      "regularised: dx / sqrt(r^2 + R^2)", "local"))
        L(FeatureSpec("time_frac", "-",
                      "pseudo-time t in [0, 1] of the tool path where the tool passes nearest"
                      if self.time_source == "toolpath" else
                      "depth fraction standing for the tool-path pseudo-time "
                      "(time_source = 'depth')", "local"))
        for k in range(1, self.rings + 1):
            for stat in ("mean", "min", "max"):
                L(FeatureSpec(f"ring{k}_{stat}", "m",
                              f"{stat} of z(neighbour) - z(node) over ring {k} of {self.rings} "
                              f"(radii up to {self.ring_max:g} tool radii)", "local"))
        L(FeatureSpec("clamp_dist", "m", "distance to the clamped frame (L-infinity, positive "
                      "inside the free window)", "placement"))
        L(FeatureSpec("blank_radius", "-", "distance from the blank centre / the free "
                      "half-width", "placement"))
        L(FeatureSpec("max_depth", "m", "maximum depth of the part", "global"))
        L(FeatureSpec("area", "m^2", "plan area of the part (depth > part_eps)", "global"))
        L(FeatureSpec("volume", "m^3", "volume under the sheet plane, sum of depth x cell "
                      "area", "global"))
        L(FeatureSpec("mean_wall_angle", "rad", "mean wall angle over the part (central "
                      "differences)", "global"))
        L(FeatureSpec("max_wall_angle", "rad", "largest wall angle over the part (central "
                      "differences)", "global"))
        L(FeatureSpec("perimeter_over_area", "1/m", "rim length / plan area", "global"))
        L(FeatureSpec("aspect", "-", "principal-axis aspect ratio of the part's plan area "
                      "(>= 1)", "global"))
        L(FeatureSpec("depth_gyration_major", "m", "larger radius of gyration of the depth "
                      "field about its centroid", "global"))
        L(FeatureSpec("depth_gyration_minor", "m", "smaller radius of gyration of the depth "
                      "field about its centroid", "global"))
        L(FeatureSpec("tool_radius", "m", "ball tool radius", "process"))
        L(FeatureSpec("step_down", "m", "vertical step per level/revolution", "process"))
        L(FeatureSpec("thickness", "m", "initial sheet thickness", "process"))
        L(FeatureSpec("friction", "-", "Coulomb friction coefficient tool/sheet", "process"))
        L(FeatureSpec("elastic_curvature", "1/m", "elastic unloading curvature of a fully "
                      "plastic bend, 3 sigma_f(0.2) / (E t), sigma_f(0.2) the flow stress at "
                      "20 % plastic strain: the classical springback scale of the sheet",
                      "process"))
        L(FeatureSpec("youngs_modulus", "Pa", "Young's modulus E", "material"))
        L(FeatureSpec("poisson_ratio", "-", "Poisson's ratio", "material"))
        L(FeatureSpec("yield_stress", "Pa", "initial yield stress", "material"))
        for eps in FLOW_STRAINS:
            L(FeatureSpec(f"flow_stress_{int(round(eps * 100)):02d}", "Pa",
                          f"isotropic flow stress at plastic strain {eps:g}", "material"))
        L(FeatureSpec("hardening_modulus", "Pa", "linear isotropic hardening modulus H",
                      "material"))
        L(FeatureSpec("voce_Q", "Pa", "Voce saturation stress Q", "material"))
        L(FeatureSpec("voce_rate", "-", "Voce saturation rate delta", "material"))
        L(FeatureSpec("r0", "-", "Lankford r-value at 0 deg (1 when isotropic)", "material"))
        L(FeatureSpec("r45", "-", "Lankford r-value at 45 deg (1 when isotropic)", "material"))
        L(FeatureSpec("r90", "-", "Lankford r-value at 90 deg (1 when isotropic)", "material"))
        L(FeatureSpec("backstress_sat", "Pa",
                      f"monotonic back stress at plastic strain {BACKSTRESS_STRAIN:g} "
                      "(Prager + Armstrong-Frederick; tends to sum C / gamma)", "material"))
        L(FeatureSpec("flow_stress_20_over_E", "-", "flow stress at plastic strain 0.2 "
                      "(20 %, not the 0.2 % proof stress) / E: the elastic springback strain "
                      "scale", "material"))
        return out

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.specs()]

    def names_of_kind(self, *kinds: str) -> List[str]:
        return [s.name for s in self.specs() if s.kind in kinds]

    def schema_hash(self) -> str:
        """SHA-256 of the schema version, the feature names and units, and this
        configuration: equal hashes mean identical inputs."""
        doc = {"version": FEATURE_SCHEMA_VERSION, "config": self.to_dict(),
               "features": [[s.name, s.unit] for s in self.specs()]}
        return sha256_bytes(canonical_json(doc).encode("utf-8"))


DEFAULT_CONFIG = FeatureConfig()


def names_hash(names: Sequence[str]) -> str:
    """SHA-256 of a list of feature names (order matters)."""
    return sha256_bytes(*[n.encode("utf-8") for n in names])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def as_setup(setup: Union[FormingSetup, Mapping[str, Any]]) -> FormingSetup:
    """A FormingSetup from a setup or its `to_dict()` / `physics_dict()`."""
    if isinstance(setup, FormingSetup):
        return setup
    if isinstance(setup, Mapping):
        return FormingSetup.from_dict(dict(setup))
    raise TypeError(f"setup must be a FormingSetup or its dict, got {type(setup).__name__}")


def _part_mask(commanded: HeightMap, eps: float) -> np.ndarray:
    part = commanded.mask & (commanded.z < -eps)
    if not part.any():
        raise ValueError("the commanded surface has no part (no node deeper than "
                         f"{eps:g} m): nothing to featurise")
    return part


def _pad_convolve(z: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlation of z with a symmetric 2-D kernel, border mode 'nearest' (FFT)."""
    ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
    zp = np.pad(z, ((ry, ry), (rx, rx)), mode="edge")
    return signal.fftconvolve(zp, kernel[::-1, ::-1], mode="valid")


def _ring_footprints(radius_cells: Sequence[float]) -> List[np.ndarray]:
    rmax = int(math.floor(radius_cells[-1] + 1e-9))
    k = np.arange(-rmax, rmax + 1)
    DX, DY = np.meshgrid(k, k, indexing="xy")
    d = np.hypot(DX, DY)
    out = []
    inner = 0.0
    for r in radius_cells:
        fp = (d > inner + 1e-9) & (d <= r + 1e-9)
        out.append(fp)
        inner = r
    return out


def _row_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """(first, last) column of every run of True in a boolean row."""
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[cuts + 1]].tolist(), np.r_[idx[cuts], idx[-1]].tolist()))


def _footprint_extrema(z: np.ndarray, footprints: Sequence[np.ndarray]
                       ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(min, max) of z over each footprint around every node, border mode
    'nearest' - equal to ndimage.minimum_filter / maximum_filter with that
    footprint, bit for bit, at a cost linear in the footprint radius rather
    than its area: every row of a footprint is a few runs of columns, and the
    extreme over a run is a 1-D running extreme (O(n) whatever its length),
    computed once per run length on the padded array and shifted. The
    footprints share one odd shape."""
    ry, rx = footprints[0].shape[0] // 2, footprints[0].shape[1] // 2
    ny, nx = z.shape
    zp = np.pad(z, ((ry, ry), (rx, rx)), mode="edge")
    cache: Dict[Tuple[str, int], np.ndarray] = {}
    out = []
    for fp in footprints:
        pair = []
        for op, filt, red in (("min", ndimage.minimum_filter1d, np.minimum),
                              ("max", ndimage.maximum_filter1d, np.maximum)):
            acc: Optional[np.ndarray] = None
            for i in range(fp.shape[0]):
                for a, b in _row_runs(fp[i]):
                    w = b - a + 1
                    if (op, w) not in cache:
                        cache[(op, w)] = filt(zp, w, axis=1, mode="nearest")
                    # column c of the filtered row holds the extreme over
                    # [c - w // 2, c - w // 2 + w - 1]; the run [x + a, x + b]
                    # of the padded row is centred on c = x + a + w // 2
                    v = cache[(op, w)][i:i + ny, a + w // 2:a + w // 2 + nx]
                    acc = v.copy() if acc is None else red(acc, v, out=acc)
            pair.append(acc)
        out.append((pair[0], pair[1]))
    return out


def material_features(setup: FormingSetup) -> Dict[str, float]:
    """The material descriptors of `setup.material` (names as in the schema)."""
    m = setup.material
    out = {"youngs_modulus": m.youngs_modulus, "poisson_ratio": m.poisson_ratio,
           "yield_stress": m.yield_stress}
    for eps in FLOW_STRAINS:
        out[f"flow_stress_{int(round(eps * 100)):02d}"] = float(m.flow_stress(eps))
    out.update(hardening_modulus=m.hardening_modulus, voce_Q=m.saturation_stress,
               voce_rate=m.saturation_rate,
               r0=1.0 if m.r0 is None else m.r0, r45=1.0 if m.r45 is None else m.r45,
               r90=1.0 if m.r90 is None else m.r90)
    a = BACKSTRESS_STRAIN
    out["backstress_sat"] = float(m.uniaxial_stress(a) - m.flow_stress(a))
    out["flow_stress_20_over_E"] = float(m.flow_stress(FLOW_STRAINS[0]) / m.youngs_modulus)
    return {k: float(v) for k, v in out.items()}


def process_features(setup: FormingSetup) -> Dict[str, float]:
    """The process descriptors of `setup` (names as in the schema)."""
    m = setup.material
    kappa = 3.0 * float(m.flow_stress(FLOW_STRAINS[0])) / (m.youngs_modulus * setup.thickness)
    return {"tool_radius": float(setup.tool_radius), "step_down": float(setup.step_down),
            "thickness": float(setup.thickness), "friction": float(setup.friction),
            "elastic_curvature": kappa}


def _principal(cov: np.ndarray) -> Tuple[float, float]:
    ev = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    return float(max(ev[-1], 0.0)), float(max(ev[0], 0.0))


def _perimeter(commanded: HeightMap, depth: np.ndarray, eps: float) -> float:
    from contourpy import LineType, contour_generator

    g = commanded.grid
    gen = contour_generator(x=g.x, y=g.y, z=depth, line_type=LineType.Separate)
    total = 0.0
    for line in gen.lines(eps):
        line = np.asarray(line, dtype=float)
        if len(line) >= 2:
            total += float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    return total


def geometry_globals(commanded: HeightMap, config: FeatureConfig = DEFAULT_CONFIG
                     ) -> Dict[str, float]:
    """The global (per-part) geometric descriptors of the commanded surface."""
    g = commanded.grid
    h = g.h
    part = _part_mask(commanded, config.part_eps)
    depth = np.maximum(-commanded.z, 0.0)
    X, Y = g.mesh()
    cell = h * h
    area = float(part.sum() * cell)
    wa = commanded.wall_angle()[part]
    # plan-area principal axes
    xs, ys = X[part], Y[part]
    cov = np.cov(np.stack([xs, ys])) if xs.size > 1 else np.zeros((2, 2))
    floor = h * h / 12.0                       # a one-cell-wide strip still has this
    lmax, lmin = _principal(cov)
    aspect = math.sqrt((lmax + floor) / (lmin + floor))
    # depth-weighted moments
    w = np.where(part, depth, 0.0)
    wsum = float(w.sum())
    cx, cy = float((w * X).sum() / wsum), float((w * Y).sum() / wsum)
    dx, dy = X - cx, Y - cy
    cd = np.array([[(w * dx * dx).sum(), (w * dx * dy).sum()],
                   [(w * dx * dy).sum(), (w * dy * dy).sum()]]) / wsum
    gmax, gmin = _principal(cd)
    perim = _perimeter(commanded, depth, config.part_eps)
    if not perim > 0:
        perim = 2.0 * math.sqrt(math.pi * area)          # a part too small to contour
    return {"max_depth": float(depth[part].max()), "area": area,
            "volume": float(depth[part].sum() * cell),
            "mean_wall_angle": float(wa.mean()), "max_wall_angle": float(wa.max()),
            "perimeter_over_area": perim / area, "aspect": aspect,
            "depth_gyration_major": math.sqrt(gmax + floor),
            "depth_gyration_minor": math.sqrt(gmin + floor)}


def global_features(commanded: HeightMap, setup: Union[FormingSetup, Mapping[str, Any]],
                    config: FeatureConfig = DEFAULT_CONFIG) -> Tuple[np.ndarray, List[str]]:
    """The part-level descriptor vector: global, process and material features
    (in schema order), and their names."""
    setup = as_setup(setup)
    vals = {**geometry_globals(commanded, config), **process_features(setup),
            **material_features(setup)}
    names = config.names_of_kind("global", "process", "material")
    return np.array([vals[n] for n in names], dtype=float), names


def _time_map(commanded: HeightMap, setup: FormingSetup, toolpath: Optional[Toolpath],
              depth_frac: np.ndarray, config: FeatureConfig) -> np.ndarray:
    if config.time_source == "depth":
        return depth_frac
    if toolpath is None:
        from ..fea.deck import make_toolpath
        toolpath = make_toolpath(setup, commanded)
    contact = toolpath.level != AIR
    if contact.sum() < 2:
        raise ValueError("the tool path has fewer than two points in contact; cannot "
                         "compute time_frac")
    pts = toolpath.points[contact]
    t = toolpath.t[contact]
    # the tool centre that shaped a node sits one tool radius along its normal
    n = commanded.normals()
    X, Y = commanded.grid.mesh()
    R = toolpath.tool_radius
    q = np.stack([X + R * n[..., 0], Y + R * n[..., 1], commanded.z + R * n[..., 2]], -1)
    _, idx = cKDTree(pts).query(q.reshape(-1, 3))
    return t[idx].reshape(commanded.grid.shape)


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------
@dataclass
class FeatureMaps:
    """All features at every node: `values` (ny, nx, F), `names`, `config`,
    the part mask and the region codes (`REGION_NAMES`)."""

    values: np.ndarray
    names: List[str]
    config: FeatureConfig
    part: np.ndarray
    region: np.ndarray

    def column(self, name: str) -> np.ndarray:
        return self.values[..., self.names.index(name)]

    def gather(self, points: Any = None, grid=None) -> np.ndarray:
        """Rows of features at `points` (see `point_features`)."""
        ny, nx, F = self.values.shape
        flat = self.values.reshape(-1, F)
        if points is None:
            return flat.copy()
        p = np.asarray(points)
        if p.dtype == bool:
            if p.shape != (ny, nx):
                raise ValueError(f"a boolean point mask must have the grid shape {(ny, nx)}, "
                                 f"got {p.shape}")
            return flat[p.ravel()]
        if np.issubdtype(p.dtype, np.integer):
            if p.ndim != 1 or (p.size and (p.min() < 0 or p.max() >= ny * nx)):
                raise ValueError("integer points must be flat node indices in [0, ny * nx)")
            return flat[p]
        if p.ndim == 2 and p.shape[1] == 2:
            if grid is None:
                raise ValueError("coordinates need the grid to interpolate on")
            hm = HeightMap(grid, np.zeros(grid.shape))
            out = hm.sample(self.values, p[:, 0], p[:, 1])
            if not np.all(np.isfinite(out)):
                raise ValueError("some points lie outside the grid")
            return out
        raise ValueError("points must be None, a boolean grid mask, flat node indices or an "
                         "(n, 2) array of x, y [m]")


def feature_maps(commanded: HeightMap, setup: Union[FormingSetup, Mapping[str, Any]],
                 toolpath: Optional[Toolpath] = None,
                 config: FeatureConfig = DEFAULT_CONFIG) -> FeatureMaps:
    """Every feature of the schema at every node of the commanded grid."""
    setup = as_setup(setup)
    g = commanded.grid
    h = g.h
    R = float(setup.tool_radius)
    ring_step = config.ring_max * R / config.rings
    if ring_step < h * (1 - 1e-9):
        raise ValueError(f"grid spacing {h:g} m is coarser than the ring width "
                         f"{ring_step:g} m (ring_max x tool radius / rings); refine the grid")
    z = commanded.z
    part = _part_mask(commanded, config.part_eps)
    depth = np.maximum(-z, 0.0)
    dmax = float(depth[part].max())
    depth_frac = depth / dmax
    X, Y = g.mesh()
    cols: Dict[str, np.ndarray] = {"depth": depth, "depth_frac": depth_frac}
    for k, s in enumerate(config.scales, 1):
        sigma = s * R
        gx, gy = commanded.gradient(sigma)
        cols[f"wall_angle_s{k}"] = np.arctan(np.hypot(gx, gy))
        H, K = commanded.curvature(sigma)
        cols[f"mean_curv_s{k}"] = H * sigma
        cols[f"gauss_curv_s{k}"] = K * sigma * sigma
    d_in = ndimage.distance_transform_edt(part) * h
    d_out = ndimage.distance_transform_edt(~part) * h
    cols["rim_dist"] = np.where(part, d_in - 0.5 * h, -(d_out - 0.5 * h))
    area = float(part.sum()) * h * h
    r_eq = math.sqrt(area / math.pi)
    cx, cy = float(X[part].mean()), float(Y[part].mean())
    dx, dy = X - cx, Y - cy
    r = np.hypot(dx, dy)
    cols["radial_norm"] = r / r_eq
    reg = np.sqrt(r * r + R * R)
    cols["polar_sin"], cols["polar_cos"] = dy / reg, dx / reg
    cols["time_frac"] = _time_map(commanded, setup, toolpath, depth_frac, config)
    radii = [ring_step * k / h for k in range(1, config.rings + 1)]
    fps = _ring_footprints(radii)
    for k, fp in enumerate(fps, 1):
        if not fp.any():
            raise ValueError(f"ring {k} contains no grid node at spacing {h:g} m")
    for k, (fp, (lo, hi)) in enumerate(zip(fps, _footprint_extrema(z, fps)), 1):
        kern = fp.astype(float) / fp.sum()
        cols[f"ring{k}_mean"] = _pad_convolve(z, kern) - z
        cols[f"ring{k}_min"] = lo - z
        cols[f"ring{k}_max"] = hi - z
    e = float(setup.free_half_width)
    cols["clamp_dist"] = e - np.maximum(np.abs(X), np.abs(Y))
    cols["blank_radius"] = np.hypot(X, Y) / e
    scalars = {**geometry_globals(commanded, config), **process_features(setup),
               **material_features(setup)}
    names = config.names
    values = np.empty(g.shape + (len(names),), dtype=float)
    for i, name in enumerate(names):
        values[..., i] = cols[name] if name in cols else scalars[name]
    if not np.all(np.isfinite(values)):
        bad = [n for i, n in enumerate(names) if not np.all(np.isfinite(values[..., i]))]
        raise ValueError(f"non-finite features {bad}; the commanded surface is degenerate")
    region = region_labels(commanded, setup, config, part=part, rim_dist=cols["rim_dist"])
    return FeatureMaps(values, names, config, part, region)


def region_labels(commanded: HeightMap, setup: Union[FormingSetup, Mapping[str, Any]],
                  config: FeatureConfig = DEFAULT_CONFIG, *, part: Optional[np.ndarray] = None,
                  rim_dist: Optional[np.ndarray] = None) -> np.ndarray:
    """(ny, nx) int8 region codes (`REGION_NAMES`): 0 flange (not part), 1 rim
    (part nodes within `rim_width` tool radii of the boundary), 2 wall (other
    part nodes with a wall angle above 5 deg, `metrology.wall_mask`), 3 base
    (the rest: floors, pole regions)."""
    setup = as_setup(setup)
    if part is None:
        part = _part_mask(commanded, config.part_eps)
    if rim_dist is None:
        rim_dist = ndimage.distance_transform_edt(part) * commanded.grid.h \
            - 0.5 * commanded.grid.h
    rim = part & (rim_dist <= config.rim_width * setup.tool_radius)
    wall = part & ~rim & wall_mask(commanded, 5.0, config.part_eps)
    out = np.zeros(commanded.grid.shape, dtype=np.int8)
    out[rim] = 1
    out[wall] = 2
    out[part & ~rim & ~wall] = 3
    return out
