"""Design of experiments, commanded-shape variants, simulators and data generation.

Design. `design_points` draws, per part family, a scrambled Sobol sequence
(`scipy.stats.qmc`, seeded by the seed and the family's NAME, so a family's
points do not depend on which other families are drawn) over the family's
parameter bounds, the process parameters (tool radius, step-down, thickness,
friction) and the material choice; a draw that the family refuses, or whose
part does not fit the unclamped window with the tool, is skipped and the
sequence continues, so the design is deterministic for a seed and the first
n points of a family are the same for any `n_per_family` >= n. (Freeform
parts are drawn by `Freeform.sample` from a seed taken from the Sobol point.)

Variants. Every design point yields up to three commanded surfaces:

* "uncompensated" - commanded = target;
* "perturbed" - target + a smooth random field of the amplitude of a typical
  springback, zero on the flange and faded in over the rim, projected onto the
  formable set (`precomp.compensation.limit_wall_angle`);
* "compensated" - one displacement-adjustment step from a previous prediction:
  by default the simulated uncompensated part itself (c = t - (f(t) - t),
  conditioned as in `precomp.compensation`), or a given compensator such as a
  surrogate from an earlier training round;

so a model learns the commanded -> formed map around compensated shapes, where
it is queried during compensation.

Simulators. `SparlabSimulator` runs `precomp.fea.simulate_many` through the
content-addressed run cache (so generation is resumable and never runs a deck
twice) and records every failure with its reason. `ProxySimulator` is a fast
analytic deviation generator - NOT physics-validated, labelled "proxy - not
physics" wherever it appears - for unit tests, CI and smoke runs of the
pipeline. Like the deck builder, it builds the setup's tool path of every job
and fails the job (recorded, with the reason) when there is none - e.g. a
spiral on a commanded surface with several pockets on one level.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple,
                    runtime_checkable)

import numpy as np
from scipy import ndimage
from scipy.stats import qmc

from .._util import PrecompError, canonical_json, read_json, to_jsonable
from ..compensation import displacement_adjustment, limit_wall_angle
from ..fea.setup import FormingSetup
from ..geometry.heightmap import Grid, HeightMap
from ..geometry.parts import MAX_WALL_ANGLE_DEG, Freeform, Part, families
from ..materials import get_material
from ..metrology import part_mask
from .dataset import Dataset, Sample
from .features import as_setup

#: Version of the proxy's formulas; part of every proxy sample's fidelity tag.
PROXY_VERSION = "proxy-v1"
PROXY_LABEL = "proxy - not physics"

#: Process parameters varied by default and their ranges (SI).
DEFAULT_PROCESS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "tool_radius": (4e-3, 8e-3), "step_down": (2e-4, 1e-3),
    "thickness": (6e-4, 1.5e-3), "friction": (0.05, 0.2)}

#: Materials drawn by default (the library's sheet alloys).
DEFAULT_MATERIALS = ("AA5754-O", "AA2024-O", "AA6061-T6", "AA7075-O", "DC04")

VARIANTS = ("uncompensated", "perturbed", "compensated")


# ---------------------------------------------------------------------------
# Proxy simulator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProxyParams:
    """Coefficients of the proxy deviation (SI; lengths in m unless noted).

    rim_amplitude : under-forming near the rim at the reference material and
        thickness for a vertical wall; rim_decay : its decay length inside the
        part [tool radii]; rim_depth_scale : depth over which it builds up.
    pillow_amplitude, pillow_length : the bulge of flat bases and how fast it
        grows with the distance from the base edge; flat_angle_deg : wall
        angle below which a region counts as flat.
    curvature : unclamping curvature [1/m] of the whole sheet for a part
        `curvature_depth` deep of the reference material and thickness.
    strain_ref : flow stress at plastic strain 0.2 / E of the reference
        material (AA5754-O); thickness_ref, tool_radius_ref, step_down_ref,
        friction_ref : the reference process; friction_gain : relative change
        of the rim term per unit of friction.
    smoothing : Gaussian smoothing of the local terms [tool radii].
    noise, noise_seed : optional white noise [m] (e.g. to mimic a scan).
    """

    rim_amplitude: float = 8e-4
    rim_decay: float = 2.5
    rim_depth_scale: float = 0.01
    pillow_amplitude: float = 5e-4
    pillow_length: float = 0.012
    flat_angle_deg: float = 12.0
    curvature: float = 0.1
    curvature_depth: float = 0.025
    strain_ref: float = 3.8e-3
    thickness_ref: float = 1e-3
    tool_radius_ref: float = 5e-3
    step_down_ref: float = 5e-4
    friction_ref: float = 0.1
    friction_gain: float = 1.0
    smoothing: float = 0.5
    noise: float = 0.0
    noise_seed: int = 0


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class SimOutcome:
    """One simulated (or proxied) forming: `formed` on the commanded grid, or
    `error` (the termination reason); `provenance` holds the source, fidelity,
    and for SparLab the version, deck hash, runtime and cache hit."""

    ok: bool
    formed: Optional[HeightMap] = None
    error: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


def sim_job(setup: Any, commanded: HeightMap, target: Any) -> Tuple[Any, ...]:
    """A `Simulator` job: (setup, commanded), and the target (a HeightMap, or
    a callable making it) as a third item only when the setup has a support
    strategy whose fixture follows it - so a simulator written for pairs
    keeps working for every setup without support."""
    if as_setup(setup).support == "none":
        return (setup, commanded)
    return (setup, commanded, target() if callable(target) else target)


@runtime_checkable
class Simulator(Protocol):
    """Anything that forms commanded surfaces: `source` ("sim" or "proxy")
    and ``run(jobs) -> [SimOutcome]`` in job order, failures included. A job
    is (setup, commanded) or, for a setup with a support strategy,
    (setup, commanded, target) - `target` the part its fixture is made for
    (`precomp.fea.simulate`; `sim_job`)."""

    source: str

    def run(self, jobs: Sequence[Tuple[Any, ...]]) -> List[SimOutcome]: ...


class ProxySimulator:
    """Analytic springback stand-in. NOT physics-validated.

    dz = S_R(rim + pillow) + global (+ noise), with, for commanded depth d,
    wall angle theta and signed rim distance s (positive inside):

    * rim: under-forming (dz > 0, the part too shallow) near the rim,
      A sin(theta_2.5R) exp(-s / (rim_decay R)) inside and a short bending
      zone exp(-(s / R)^2) outside, A growing with the depth and with
      (sigma_f(0.2) / E)^0.75 (t_ref / t)^0.5 (1 + gain (mu - mu_ref));
    * pillow: flat bases bulge up by P (1 - exp(-(e / pillow_length)^2)),
      e the distance inside the flat region from its edge, P growing with
      (sigma_f(0.2) / E)^0.5, sqrt(R / R_ref), sqrt(step_down / step_ref);
    * global: after a 3-2-1 release the whole sheet bends,
      dz = (kx (a^2 - x^2) + ky (a^2 - y^2)) / 2 with a the support position
      (zero at the supports), kx + ky = 2 kappa, kappa proportional to depth,
      sigma_f(0.2) / E and t_ref / t, split between x and y by the depth field's
      second moments; with "clamped_only" a quarter of it inside the window.

    It has the global + local structure the models must capture, depends on
    the commanded shape (so displacement adjustment on it is non-trivial) and
    on every process and material descriptor that matters to it. Every
    sample, metric and plot made from it is labelled "proxy - not physics".
    It knows single-point forming only: as a simulator or a model it refuses
    a setup with support from below (`FormingSetup.support`) rather than give
    single-point numbers under the supported setup's name.

    `run` (the `Simulator` protocol) also builds the setup's tool path of
    every job, as the deck builder does, and fails the job without one;
    `check_toolpath=False` skips that (the deviation does not depend on it).
    """

    source = "proxy"
    label = PROXY_LABEL
    #: as a model: what its numbers are
    data_source = PROXY_LABEL

    def __init__(self, params: Optional[ProxyParams] = None, *, check_toolpath: bool = True,
                 **changes: Any):
        p = params or ProxyParams()
        self.params = dataclasses.replace(p, **changes) if changes else p
        self.check_toolpath = bool(check_toolpath)

    @property
    def fidelity(self) -> str:
        return PROXY_VERSION

    def shifted(self, **changes: Any) -> "ProxySimulator":
        """A proxy with some coefficients changed (e.g. a 'real' process that
        differs from the simulated one, for transfer-learning tests)."""
        return ProxySimulator(dataclasses.replace(self.params, **changes),
                              check_toolpath=self.check_toolpath)

    def describe(self) -> Dict[str, Any]:
        return {"name": "ProxySimulator", "version": PROXY_VERSION, "label": PROXY_LABEL,
                "params": dataclasses.asdict(self.params)}

    def __getstate__(self) -> Dict[str, Any]:
        return {"params": dataclasses.asdict(self.params),
                "check_toolpath": self.check_toolpath}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.params = ProxyParams(**state["params"])
        self.check_toolpath = bool(state.get("check_toolpath", True))

    def deviation(self, commanded: HeightMap, setup: Any) -> np.ndarray:
        """(ny, nx) dz [m] of forming `commanded` with `setup` (single-point
        forming; PrecompError for a setup with a support)."""
        p = self.params
        s = as_setup(setup)
        if s.support != "none":
            raise PrecompError(
                f"the proxy has no model of support from below (support {s.support!r}); its "
                "dz would be single-point forming's under the supported setup's name - "
                "simulate with sparlab_form (SparlabSimulator)")
        m = s.material
        g = commanded.grid
        h = g.h
        R = s.tool_radius
        depth = np.maximum(-commanded.z, 0.0)
        part = commanded.mask & (commanded.z < -1e-6)
        if not part.any():
            return np.zeros(g.shape)
        D = float(depth[part].max())
        strain = float(m.flow_stress(0.2)) / m.youngs_modulus / p.strain_ref
        thick = p.thickness_ref / s.thickness
        X, Y = g.mesh()
        # rim under-forming
        d_in = ndimage.distance_transform_edt(part) * h - 0.5 * h
        d_out = ndimage.distance_transform_edt(~part) * h - 0.5 * h
        theta_rim = commanded.wall_angle(2.5 * R)
        amp = (p.rim_amplitude * strain ** 0.75 * thick ** 0.5
               * max(0.0, 1.0 + p.friction_gain * (s.friction - p.friction_ref))
               * (1.0 - math.exp(-D / p.rim_depth_scale)))
        shape = np.where(part, np.exp(-d_in / (p.rim_decay * R)), np.exp(-(d_out / R) ** 2))
        rim = amp * np.sin(theta_rim) * shape
        # pillow on flat bases
        theta = commanded.wall_angle(R)
        flat_limit = math.radians(p.flat_angle_deg)
        flat = part & (theta < flat_limit)
        pillow = np.zeros(g.shape)
        if flat.any():
            e = ndimage.distance_transform_edt(flat) * h
            pamp = (p.pillow_amplitude * strain ** 0.5 * math.sqrt(R / p.tool_radius_ref)
                    * math.sqrt(s.step_down / p.step_down_ref) * thick ** 0.25
                    * min(D / 0.01, 1.0))
            pillow = np.where(flat, pamp * (1.0 - np.exp(-(e / p.pillow_length) ** 2))
                              * _smoothstep((flat_limit - theta) / flat_limit), 0.0)
        local = ndimage.gaussian_filter(rim + pillow, p.smoothing * R / h, mode="nearest")
        # global bending on unclamping
        kappa = p.curvature * strain * (D / p.curvature_depth) * thick
        w = np.where(part, depth, 0.0)
        ixx, iyy = float((w * X * X).sum()), float((w * Y * Y).sum())
        kx = 2.0 * kappa * ixx / (ixx + iyy)
        ky = 2.0 * kappa - kx
        if s.release == "321":
            a = 0.5 * s.meshed_blank_size - 0.5 * s.clamp_margin
            glob = 0.5 * (kx * (a * a - X * X) + ky * (a * a - Y * Y))
        else:
            e_ = s.free_half_width
            inside = np.clip(e_ - np.maximum(np.abs(X), np.abs(Y)), 0.0, None)
            glob = 0.25 * 0.5 * kappa * inside * (2 * e_ - inside)
        dz = local + glob
        if p.noise > 0:
            rng = np.random.default_rng(p.noise_seed)
            dz = dz + rng.normal(0.0, p.noise, g.shape)
        return dz

    def formed(self, commanded: HeightMap, setup: Any) -> HeightMap:
        """commanded + dz, labelled as a proxy result in its metadata."""
        dz = self.deviation(commanded, setup)
        return commanded.with_z(commanded.z + dz,
                                metadata={"source": "proxy", "label": PROXY_LABEL,
                                          "version": PROXY_VERSION})

    # protocols: a FieldModel, a Simulator
    def predict_deviation(self, commanded: HeightMap, setup: Any
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.deviation(commanded, setup), None

    def __call__(self, commanded: HeightMap, setup: Any) -> HeightMap:
        return self.formed(commanded, setup)

    def run(self, jobs: Sequence[Tuple[Any, ...]]) -> List[SimOutcome]:
        out = []
        for job in jobs:
            setup, commanded = job[0], job[1]
            try:
                if self.check_toolpath:          # a deck needs one; so does time_frac
                    from ..fea.deck import make_toolpath
                    try:
                        make_toolpath(as_setup(setup), commanded)
                    except (ValueError, PrecompError) as exc:
                        raise PrecompError(f"no {as_setup(setup).toolpath_style} tool path: "
                                           f"{exc}") from exc
                out.append(SimOutcome(True, self.formed(commanded, setup), None,
                                      {"source": "proxy", "fidelity": self.fidelity,
                                       "simulator": self.describe()}))
            except Exception as exc:  # recorded, never dropped
                out.append(SimOutcome(False, None, f"{type(exc).__name__}: {exc}",
                                      {"source": "proxy", "fidelity": self.fidelity}))
        return out


class SparlabSimulator:
    """Forming by sparlab_form through `precomp.fea.simulate_many` (cached).

    work_dir : the run cache; max_workers, executor : as for simulate_many
    (keep max_workers x setup.threads within the cores you may use);
    step : the result step whose surface is the formed one (default the last,
    after release); retry_failed : re-run decks that failed before.
    """

    source = "sim"

    def __init__(self, work_dir: Any, *, max_workers: Optional[int] = 2,
                 executor: str = "process", step: Any = -1, retry_failed: bool = False):
        self.work_dir = work_dir
        self.max_workers = max_workers
        self.executor = executor
        self.step = step
        self.retry_failed = retry_failed

    @staticmethod
    def fidelity_of(setup: FormingSetup) -> str:
        element = setup.element
        if setup.element_formulation != "standard":
            element += "-im" if setup.element_formulation == "incompatible_modes" \
                else f"-{setup.element_formulation}"
        if setup.thickness_points:
            element += f"-tp{setup.thickness_points}"
        label = (f"sparlab:{element}:{setup.element_size * 1e3:g}mm:{setup.layers}L:"
                 f"{setup.kinematics}:{setup.release}")
        return label if setup.support == "none" else f"{label}:{setup.support}"

    def run(self, jobs: Sequence[Tuple[Any, ...]]) -> List[SimOutcome]:
        from ..fea.runner import simulate_many, sparlab_version

        outcomes = simulate_many(jobs, self.work_dir, max_workers=self.max_workers,
                                 executor=self.executor, retry_failed=self.retry_failed)
        out = []
        for job, oc in zip(jobs, outcomes):
            setup, commanded = job[0], job[1]
            prov: Dict[str, Any] = {"source": "sim", "fidelity": self.fidelity_of(setup),
                                    "deck_hash": oc.key, "runtime_s": oc.runtime_s,
                                    "cache_hit": oc.cache_hit}
            if not oc.ok:
                out.append(SimOutcome(False, None, oc.error or "unknown failure", prov))
                continue
            try:
                res = oc.load()
                formed = res.formed_surface(self.step, grid=commanded.grid)
                prov["sparlab_version"] = sparlab_version(setup.resolved_executable())
                prov["result_dir"] = oc.result_dir
                out.append(SimOutcome(True, formed, None, prov))
            except Exception as exc:  # an unreadable result is a failure with its reason
                out.append(SimOutcome(False, None, f"{type(exc).__name__}: {exc}", prov))
        return out


# ---------------------------------------------------------------------------
# Design of experiments
# ---------------------------------------------------------------------------
@dataclass
class DesignSpace:
    """What `design_points` varies.

    families : part families (default: all registered).
    process : ranges of FormingSetup fields drawn uniformly (SI).
    materials : library material names, drawn uniformly.
    base_setup : `FormingSetup.to_dict()`-style fields not varied (blank,
        mesh, solver, ...); a material given here is replaced by the draw.
    part_bounds : per family, overrides of `Part.bounds` entries.
    grid_spacing : spacing of the part grid [m]; the grid covers the meshed
        blank (`Grid.centered(meshed_blank_size, grid_spacing)`).
    """

    families: Tuple[str, ...] = tuple(sorted(families()))
    process: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_PROCESS_BOUNDS))
    materials: Tuple[str, ...] = DEFAULT_MATERIALS
    base_setup: Dict[str, Any] = field(default_factory=dict)
    part_bounds: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    grid_spacing: float = 2e-3

    def __post_init__(self) -> None:
        known = families()
        for f in self.families:
            if f not in known:
                raise ValueError(f"unknown family {f!r}; known: {sorted(known)}")
        fields = {f.name for f in dataclasses.fields(FormingSetup)}
        for key, (lo, hi) in self.process.items():
            if key not in fields:
                raise ValueError(f"process parameter {key!r} is not a FormingSetup field")
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError(f"process range {key}: ({lo}, {hi}) is not a finite interval")
        for name in self.materials:
            get_material(name)
        if not self.materials:
            raise ValueError("at least one material is required")
        if not self.grid_spacing > 0:
            raise ValueError("grid_spacing must be > 0")

    def bounds_of(self, family: str) -> Dict[str, Tuple[float, float]]:
        b = dict(families()[family].bounds)
        b.update(self.part_bounds.get(family, {}))
        return b

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "DesignSpace":
        d = dict(doc)
        if "families" in d:
            d["families"] = tuple(d["families"])
        if "materials" in d:
            d["materials"] = tuple(d["materials"])
        if "process" in d:
            d["process"] = {k: tuple(v) for k, v in d["process"].items()}
        if "part_bounds" in d:
            d["part_bounds"] = {f: {k: tuple(v) for k, v in b.items()}
                                for f, b in d["part_bounds"].items()}
        return cls(**d)


@dataclass
class DesignPoint:
    """One design point: a target part, a setup and the grid of the part."""

    point_id: str
    family: str
    part: Part
    setup: FormingSetup
    grid: Grid
    sobol_index: int
    u: List[float]

    def target(self) -> HeightMap:
        return self.part.heightmap(self.grid)

    def describe(self) -> Dict[str, Any]:
        return {"point_id": self.point_id, "family": self.family,
                "part": self.part.to_dict(), "sobol_index": self.sobol_index,
                "u": list(self.u), "material": self.setup.material.name}


def stable_hash(text: str) -> int:
    """A 32-bit integer of `text` that is the same in every process and
    version (unlike hash()): seeds that follow a name, not a list position."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def _sobol(d: int, seed_seq: np.random.SeedSequence) -> qmc.Sobol:
    gen = np.random.default_rng(seed_seq)
    try:
        return qmc.Sobol(d, scramble=True, rng=gen)
    except TypeError:                                   # scipy < 1.15
        return qmc.Sobol(d, scramble=True, seed=gen)


def design_points(space: DesignSpace, n_per_family: int, seed: int, *,
                  max_draws: int = 4096) -> List[DesignPoint]:
    """`n_per_family` valid design points per family of `space` (see the
    module docstring). ValueError when a family yields fewer valid points than
    asked within `max_draws` Sobol draws (the bounds do not fit the setup)."""
    if n_per_family < 1:
        raise ValueError("n_per_family must be >= 1")
    base = dict(space.base_setup)
    base.setdefault("material", get_material(space.materials[0]))
    proc_keys = sorted(space.process)
    out: List[DesignPoint] = []
    for fam in space.families:
        cls = families()[fam]
        pbounds = space.bounds_of(fam)
        pkeys = [] if cls is Freeform else sorted(pbounds)
        d = len(pkeys) + len(proc_keys) + 1 + (1 if cls is Freeform else 0)
        sampler = _sobol(d, np.random.SeedSequence([int(seed), stable_hash(fam)]))
        found: List[DesignPoint] = []
        draws = 0
        rejects: Dict[str, int] = {}
        while len(found) < n_per_family:
            if draws >= max_draws:
                top = dict(sorted(rejects.items(), key=lambda kv: -kv[1])[:5])
                raise ValueError(
                    f"{fam}: only {len(found)} of {n_per_family} valid design points in "
                    f"{max_draws} draws; most frequent rejections: {top}. Adjust "
                    "part_bounds, the process ranges or the blank size")
            batch = sampler.random(1 << max(4, int(math.ceil(math.log2(n_per_family * 2)))))
            for u in batch:
                idx = draws
                draws += 1
                k = 0
                try:
                    if cls is Freeform:
                        part = Freeform.sample(np.random.default_rng(int(u[0] * (2 ** 31 - 1))))
                        k = 1
                    else:
                        params = {key: pbounds[key][0] + u[k + j] * (pbounds[key][1]
                                                                     - pbounds[key][0])
                                  for j, key in enumerate(pkeys)}
                        k += len(pkeys)
                        part = cls(**params)
                    proc = {key: space.process[key][0] + u[k + j] * (space.process[key][1]
                                                                   - space.process[key][0])
                            for j, key in enumerate(proc_keys)}
                    k += len(proc_keys)
                    mat = space.materials[min(int(u[k] * len(space.materials)),
                                              len(space.materials) - 1)]
                    setup = FormingSetup.from_dict({**base, **proc,
                                                    "material": get_material(mat)})
                    setup.check_part_fits(part.footprint_radius())
                    grid = Grid.centered(setup.meshed_blank_size, space.grid_spacing)
                    part.heightmap(grid)
                except (ValueError, RuntimeError) as exc:
                    key = type(exc).__name__ + ": " + str(exc).split(";")[0][:60]
                    rejects[key] = rejects.get(key, 0) + 1
                    continue
                found.append(DesignPoint(f"{fam}-s{seed}-{len(found):04d}", fam, part, setup,
                                         grid, idx, [float(v) for v in u]))
                if len(found) == n_per_family:
                    break
        out.extend(found)
    return out


# ---------------------------------------------------------------------------
# Commanded-shape variants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PerturbationSpec:
    """Random smooth perturbations of the commanded surface.

    amplitude : range of the RMS over the part [m] (drawn uniformly);
    correlation_length : range of the Gaussian correlation length [m];
    taper : the perturbation fades in over this many tool radii inside the
        rim (so the flange and the rim stay continuous);
    angle_margin_deg : it also fades out where the target's wall (smoothed
        at one tool radius) comes within this many degrees of the forming
        limit, so it does not push steep walls past it.
    """

    amplitude: Tuple[float, float] = (2e-4, 1e-3)
    correlation_length: Tuple[float, float] = (0.008, 0.025)
    taper: float = 2.0
    angle_margin_deg: float = 10.0


def perturbed_commanded(target: HeightMap, setup: FormingSetup, rng: np.random.Generator,
                        spec: PerturbationSpec = PerturbationSpec()
                        ) -> Tuple[HeightMap, Dict[str, Any]]:
    """target + a smooth random field (see `PerturbationSpec`), held at the
    target on the flange, kept at z <= 0 and within the wall-angle limit."""
    g = target.grid
    part = part_mask(target)
    amp = float(rng.uniform(*spec.amplitude))
    corr = float(rng.uniform(*spec.correlation_length))
    field_ = ndimage.gaussian_filter(rng.standard_normal(g.shape), corr / g.h, mode="nearest")
    inside = ndimage.distance_transform_edt(part) * g.h
    field_ = field_ * _smoothstep(inside / (spec.taper * setup.tool_radius))
    margin = np.radians(MAX_WALL_ANGLE_DEG) - target.wall_angle(setup.tool_radius)
    field_ = field_ * _smoothstep(margin / np.radians(spec.angle_margin_deg))
    rms = float(np.sqrt(np.mean(field_[part] ** 2)))
    if not rms > 0:
        raise PrecompError("the perturbation vanished on the part (a part thinner than the "
                           "taper?)")
    z = np.where(part, np.minimum(target.z + amp * field_ / rms, 0.0), target.z)
    hm = limit_wall_angle(target.with_z(z), MAX_WALL_ANGLE_DEG)
    info = {"amplitude_rms_m": amp, "correlation_length_m": corr,
            "taper_tool_radii": spec.taper}
    hm.metadata["commanded"] = {"kind": "perturbed", **info}
    return hm, info


def compensated_commanded(target: HeightMap, formed: HeightMap, alpha: float = 1.0,
                          **da_options: Any) -> HeightMap:
    """One displacement-adjustment step from a predicted (or simulated)
    formed surface of the target: t - alpha (f - t), conditioned as in
    `precomp.compensation` (flange held, z <= 0, wall angle <= 65 deg).
    `da_options`: further keywords of `displacement_adjustment` - a
    support's `upper_bound`, `hold_mask` and `adjust_mask`, the tool's reach
    (`tool_radius`)."""
    res = displacement_adjustment(target, lambda c: formed, iterations=1, alpha=alpha,
                                  **da_options)
    out = res.proposed
    out.metadata["commanded"] = {"kind": "compensated", "alpha": alpha}
    if da_options.get("tool_radius") is not None:
        out.metadata["commanded"]["tool_reach_radius_m"] = float(da_options["tool_radius"])
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@dataclass
class GenerationReport:
    """What `generate` did: counts and the failures (with their reasons)."""

    requested: int = 0
    created: List[str] = field(default_factory=list)
    skipped_existing: List[str] = field(default_factory=list)
    failed: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {"requested": self.requested, "created": len(self.created),
                "skipped_existing": len(self.skipped_existing), "failed": len(self.failed),
                "failures": [{k: f[k] for k in ("sample_id", "reason")} for f in self.failed]}


def sample_id(point: DesignPoint, kind: str) -> str:
    return f"{point.point_id}-{kind[:4]}"


def generate(dataset: Dataset, points: Sequence[DesignPoint], simulator: Simulator, *,
             created_at: str, kinds: Sequence[str] = VARIANTS, seed: int = 0,
             perturbation: PerturbationSpec = PerturbationSpec(),
             compensator: Optional[Callable[[HeightMap, FormingSetup], HeightMap]] = None,
             da_options: Optional[Callable[["DesignPoint"], Mapping[str, Any]]] = None,
             store_toolpath: bool = False) -> GenerationReport:
    """Simulate the requested variants of every design point into `dataset`.

    Resumable: samples already in the data set are skipped (and a SparLab run
    is fetched from its cache) - after checking that the stored sample is
    the same design point: the same part, setup (physics) and grid. A sample
    id that holds another part (a design drawn from another space under the
    same seed, say) raises PrecompError before anything runs: a data set
    never mixes two designs under one id. Every failure is recorded in the
    data set's failures.jsonl with its reason and returned in the report -
    never dropped; a compensated variant whose uncompensated run failed is
    recorded as a failure of its own ("dependency failed").

    seed : seeds the perturbations; that of a design point is drawn from
        SeedSequence([seed, hash of its point id, 1]), whatever the list it
        comes in.
    compensator : ``(target, setup) -> commanded`` for the "compensated"
        variant (e.g. surrogate DA with an earlier model); default: one DA
        step from the simulated uncompensated part.
    da_options : ``point -> {keyword: value}`` of `displacement_adjustment`
        for that default step (a support's masks and command bound, the
        tool's reach `tool_radius`); None: the defaults.
    store_toolpath : also store the setup's tool path of every commanded
        surface (features rebuild it deterministically otherwise).
    """
    for k in kinds:
        if k not in VARIANTS:
            raise ValueError(f"unknown variant {k!r}; known: {VARIANTS}")
    if not created_at:
        raise ValueError("created_at must be supplied by the caller")
    report = GenerationReport()
    sim_desc = (simulator.describe() if hasattr(simulator, "describe")
                else {"name": type(simulator).__name__})

    def make_sample(point: DesignPoint, kind: str, commanded: HeightMap, outcome: SimOutcome,
                    info: Dict[str, Any]) -> Sample:
        prov = {"created_at": created_at, "simulator": sim_desc,
                "design": {"seed": seed, **point.describe()}, "variant": info}
        prov.update({k: v for k, v in outcome.provenance.items()
                     if k in ("sparlab_version", "deck_hash", "runtime_s", "cache_hit",
                              "result_dir")})
        path = None
        if store_toolpath:
            from ..fea.deck import make_toolpath
            path = make_toolpath(point.setup, commanded)
        return Sample(sample_id(point, kind), commanded, outcome.formed, point.setup.to_dict(),
                      outcome.provenance.get("source", simulator.source),
                      outcome.provenance.get("fidelity", getattr(simulator, "fidelity", "?")),
                      kind, point.part.to_dict(), point.point_id, point.target(), path, prov)

    def fail(point: DesignPoint, kind: str, reason: str, extra: Optional[Dict] = None) -> None:
        rec = {"sample_id": sample_id(point, kind), "point_id": point.point_id, "kind": kind,
               "reason": reason.splitlines()[0][:500] if reason else "unknown",
               "detail": reason, "created_at": created_at, **(extra or {})}
        dataset.record_failure(rec)
        report.failed.append(rec)

    # samples already present must be the same design point
    index = {r["sample_id"]: r for r in dataset.index().to_dict("records")}
    for point in points:
        for kind in kinds:
            sid = sample_id(point, kind)
            if sid in index:
                _check_resumed(dataset, index[sid], point)
    # phase 1: variants that need no simulation result
    jobs: List[Tuple[DesignPoint, str, HeightMap, Dict[str, Any]]] = []
    for point in points:
        target = point.target()
        for kind in ("uncompensated", "perturbed"):
            if kind not in kinds:
                continue
            report.requested += 1
            sid = sample_id(point, kind)
            if dataset.has(sid):
                report.skipped_existing.append(sid)
                continue
            if kind == "uncompensated":
                cmd, info = target.copy(), {"kind": kind}
            else:
                rng = np.random.default_rng(np.random.SeedSequence(
                    [int(seed), stable_hash(point.point_id), 1]))
                try:
                    cmd, pinfo = perturbed_commanded(target, point.setup, rng, perturbation)
                except (PrecompError, ValueError) as exc:
                    fail(point, kind, f"perturbation failed: {exc}")
                    continue
                info = {"kind": kind, **pinfo}
            jobs.append((point, kind, cmd, info))
    results = simulator.run([sim_job(p.setup, c, p.target) for p, _, c, _ in jobs]) \
            if jobs else []
    uncomp: Dict[str, HeightMap] = {}
    uncomp_error: Dict[str, str] = {}
    for (point, kind, cmd, info), oc in zip(jobs, results):
        if oc.ok:
            dataset.append(make_sample(point, kind, cmd, oc, info))
            report.created.append(sample_id(point, kind))
            if kind == "uncompensated":
                uncomp[point.point_id] = oc.formed
        else:
            fail(point, kind, oc.error or "unknown failure",
                 {k: oc.provenance.get(k) for k in ("deck_hash", "runtime_s")})
            if kind == "uncompensated":
                uncomp_error[point.point_id] = oc.error or "unknown failure"
    # phase 2: compensated variants
    if "compensated" in kinds:
        jobs = []
        for point in points:
            report.requested += 1
            sid = sample_id(point, "compensated")
            if dataset.has(sid):
                report.skipped_existing.append(sid)
                continue
            target = point.target()
            try:
                if compensator is not None:
                    cmd = compensator(target, point.setup)
                    info = {"kind": "compensated", "from": "compensator"}
                else:
                    f0 = uncomp.get(point.point_id)
                    if f0 is None:
                        usid = sample_id(point, "uncompensated")
                        if dataset.has(usid):
                            f0 = dataset.load(usid).formed
                        elif point.point_id in uncomp_error:
                            fail(point, "compensated", "dependency failed: the uncompensated "
                                 f"run failed ({uncomp_error[point.point_id].splitlines()[0]})")
                            continue
                        else:
                            fail(point, "compensated", "dependency failed: no uncompensated "
                                 "sample (request the 'uncompensated' variant too)")
                            continue
                    cmd = compensated_commanded(target, f0,
                                                **(dict(da_options(point)) if da_options
                                                   else {}))
                    info = {"kind": "compensated", "from": "one DA step on the simulated "
                            "uncompensated part"}
            except (PrecompError, ValueError) as exc:
                fail(point, "compensated", f"compensation failed: {exc}")
                continue
            jobs.append((point, "compensated", cmd, info))
        results = simulator.run([sim_job(p.setup, c, p.target) for p, _, c, _ in jobs]) \
            if jobs else []
        for (point, kind, cmd, info), oc in zip(jobs, results):
            if oc.ok:
                dataset.append(make_sample(point, kind, cmd, oc, info))
                report.created.append(sample_id(point, kind))
            else:
                fail(point, kind, oc.error or "unknown failure",
                     {k: oc.provenance.get(k) for k in ("deck_hash", "runtime_s")})
    return report


def _check_resumed(dataset: Dataset, row: Mapping[str, Any], point: DesignPoint) -> None:
    """PrecompError unless the stored sample `row` was made from `point`
    (same part, setup physics and grid)."""
    from ..fea.setup import EXECUTION_FIELDS

    sid = row["sample_id"]
    doc = read_json(dataset.root / "samples" / f"{sid}.json")
    def phys(d: Mapping[str, Any]) -> Dict[str, Any]:
        # through FormingSetup, so a sample stored before a field existed
        # compares at that field's default
        full = FormingSetup.from_dict(dict(d)).to_dict() if d else {}
        return {k: v for k, v in full.items() if k not in EXECUTION_FIELDS}

    diffs = []
    if canonical_json(doc.get("part")) != canonical_json(point.part.to_dict()):
        diffs.append("part")
    if canonical_json(phys(doc.get("setup") or {})) != canonical_json(
            phys(point.setup.to_dict())):
        diffs.append("setup")
    g = point.grid
    if (int(row["nx"]), int(row["ny"])) != (g.nx, g.ny) or not math.isclose(
            float(row["h"]), g.h, rel_tol=1e-9):
        diffs.append("grid")
    if diffs:
        raise PrecompError(
            f"{dataset.root}: sample {sid} exists but was made from another design point "
            f"(its {', '.join(diffs)} differ{'s' if len(diffs) == 1 else ''}); resuming "
            "would mix two designs under one id. Generate into a new data set, or use "
            "another seed")


def simulate_samples(points: Sequence[DesignPoint], simulator: Simulator, *, created_at: str,
                     kinds: Sequence[str] = ("uncompensated",), seed: int = 0) -> List[Sample]:
    """`generate` into a temporary data set, for small experiments and tests:
    the samples in memory, in the data set's order. Raises PrecompError
    listing the failures if any job failed (use `generate` to keep going)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ds = Dataset.create(tmp, created_at=created_at)
        rep = generate(ds, points, simulator, created_at=created_at, kinds=kinds, seed=seed)
        if rep.failed:
            raise PrecompError(f"{len(rep.failed)} job(s) failed: "
                               + "; ".join(f"{f['sample_id']}: {f['reason']}"
                                           for f in rep.failed[:5]))
        return ds.samples()


__all__ = ["ProxyParams", "ProxySimulator", "SparlabSimulator", "SimOutcome", "Simulator",
           "DesignSpace", "DesignPoint", "design_points", "PerturbationSpec",
           "perturbed_commanded", "compensated_commanded", "generate", "GenerationReport",
           "simulate_samples", "sample_id", "PROXY_VERSION", "PROXY_LABEL", "VARIANTS",
           "DEFAULT_PROCESS_BOUNDS", "DEFAULT_MATERIALS"]
