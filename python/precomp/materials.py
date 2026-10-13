"""Sheet materials: elasticity, plasticity, and the SparLab material block.

`Material` holds the data of SparLab's rate-independent plasticity
(`docs/configuration.md`, `material.plasticity`) - E, nu, density, the
initial yield stress, linear plus Voce isotropic hardening, Prager's linear
kinematic hardening and, optionally, Hill's 1948 anisotropic yield criterion
calibrated by r-values and one Armstrong-Frederick backstress:

    sigma_y(a) = yield_stress + hardening_modulus a
                 + saturation_stress (1 - exp(-saturation_rate a)),
    d beta   = (2/3) kinematic_hardening_modulus d eps_p,
    d beta_1 = (2/3) af_C d eps_p - af_gamma beta_1 d a,

with a the accumulated plastic strain. `to_sparlab` writes them under the
keys SparLab reads: the r-values as ``"yield_criterion": "hill48"`` with an
``anisotropy {r0, r45, r90, rolling_direction, sheet_normal}`` block (RD
along the deck's x axis, the sheet normal along z, as `precomp.fea` meshes
the blank), and the Armstrong-Frederick term as ``backstresses: [{modulus:
af_C, recovery: af_gamma}]``. With Hill48, `yield_stress` is the uniaxial
yield stress along the rolling direction (SparLab's convention).

Power laws are converted, not passed through: `Material.from_swift` fits
the linear + Voce form to a Swift curve ``sigma = K (eps0 + a)^n`` (Hollomon
is Swift with eps0 = (sigma_y / K)^(1/n)) by least squares over a stated
strain range and records the law, the range and the fit error in
`hardening_source`. Units: Pa, kg/m^3, dimensionless strains and r-values.

Where this fits: material data for "Simulations"; a FormingSetup holds a Material and the deck
    carries its to_sparlab block.
Called from: fea/setup.py:FormingSetup.__post_init__, fea/deck.py:deck_document,
    ml/generate.py:design_points.
Calls into: scipy.optimize (power-law fits), _util.
Start reading at: Material, then Material.to_sparlab.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar, nnls

from ._util import require_nonnegative, require_positive

#: Plastic strain range [-] over which power laws are fitted by default: SPIF
#: reaches equivalent plastic strains of order 0.3-1 in the wall.
DEFAULT_FIT_RANGE = (0.0, 0.6)

#: The material frame of the Hill48 block: the rolling direction along the
#: deck's x axis and the sheet normal along z (`precomp.fea` meshes the blank
#: in the x-y plane, tool side up).
ROLLING_DIRECTION = (1.0, 0.0, 0.0)
SHEET_NORMAL = (0.0, 0.0, 1.0)

#: The caveat carried by every library material.
NOMINAL_SOURCE = ("nominal handbook-order values, not certified data; measure the "
                  "batch before trusting a prediction")


@dataclass(frozen=True)
class Material:
    """An isotropic elastic, elastoplastic sheet material (SI units).

    name : label carried into the deck.
    youngs_modulus [Pa] > 0; poisson_ratio in (-1, 0.5); density [kg/m^3] >= 0.
    yield_stress [Pa] > 0 : initial uniaxial yield stress.
    hardening_modulus [Pa] >= 0 : linear isotropic hardening H.
    saturation_stress [Pa] >= 0, saturation_rate [-] : Voce term Q, delta
        (delta > 0 required when Q > 0).
    kinematic_hardening_modulus [Pa] >= 0 : Prager's linear kinematic hardening.
    af_C [Pa] > 0, af_gamma [-] >= 0 : optional Armstrong-Frederick back
        stress (both or neither): SparLab's backstress `modulus` and
        `recovery` (af_gamma = 0 is a second linear, Prager, term).
    r0, r45, r90 [-] > 0 : optional Lankford coefficients at 0, 45 and 90 deg
        to the rolling direction, which select Hill's 1948 yield criterion
        (all three or none; all equal to 1 is von Mises).
    hardening_source : provenance of the hardening parameters (e.g. the
        Swift law they were fitted to and the fit error).
    source : provenance of the whole data set.
    """

    name: str
    youngs_modulus: float
    poisson_ratio: float
    density: float
    yield_stress: float
    hardening_modulus: float = 0.0
    saturation_stress: float = 0.0
    saturation_rate: float = 0.0
    kinematic_hardening_modulus: float = 0.0
    af_C: Optional[float] = None
    af_gamma: Optional[float] = None
    r0: Optional[float] = None
    r45: Optional[float] = None
    r90: Optional[float] = None
    hardening_source: Dict[str, Any] = field(default_factory=dict, compare=False)
    source: str = ""

    def __post_init__(self) -> None:
        require_positive("youngs_modulus", self.youngs_modulus)
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError(f"poisson_ratio must lie in (-1, 0.5), got {self.poisson_ratio}")
        require_nonnegative("density", self.density)
        require_positive("yield_stress", self.yield_stress)
        require_nonnegative("hardening_modulus", self.hardening_modulus)
        require_nonnegative("saturation_stress", self.saturation_stress)
        require_nonnegative("saturation_rate", self.saturation_rate)
        require_nonnegative("kinematic_hardening_modulus", self.kinematic_hardening_modulus)
        if self.saturation_stress > 0 and not self.saturation_rate > 0:
            raise ValueError("saturation_rate must be > 0 when saturation_stress is given")
        if (self.af_C is None) != (self.af_gamma is None):
            raise ValueError("Armstrong-Frederick needs both af_C and af_gamma, or neither")
        if self.af_C is not None:
            require_positive("af_C", self.af_C)            # SparLab: modulus > 0
            require_nonnegative("af_gamma", self.af_gamma)
        rs = (self.r0, self.r45, self.r90)
        if any(r is None for r in rs) and not all(r is None for r in rs):
            raise ValueError("Hill48 needs all three of r0, r45, r90, or none")
        if self.r0 is not None:
            for key, r in zip(("r0", "r45", "r90"), rs):
                require_positive(key, r)

    # -- derived -----------------------------------------------------------

    @property
    def has_hill48(self) -> bool:
        return self.r0 is not None

    @property
    def has_armstrong_frederick(self) -> bool:
        return self.af_C is not None

    def flow_stress(self, plastic_strain: np.ndarray) -> np.ndarray:
        """Isotropic yield stress sigma_y(a) [Pa] at accumulated plastic strain a [-]."""
        a = np.asarray(plastic_strain, dtype=float)
        return (self.yield_stress + self.hardening_modulus * a
                + self.saturation_stress * (1.0 - np.exp(-self.saturation_rate * a)))

    def uniaxial_stress(self, plastic_strain: np.ndarray) -> np.ndarray:
        """Monotonic uniaxial stress [Pa] at plastic strain a [-]: the isotropic
        flow stress plus the saturated or linear back stresses (Prager adds
        H_kin a; Armstrong-Frederick adds (C / gamma)(1 - exp(-gamma a)))."""
        a = np.asarray(plastic_strain, dtype=float)
        s = self.flow_stress(a) + self.kinematic_hardening_modulus * a
        if self.af_C is not None:
            if self.af_gamma > 0:
                s = s + self.af_C / self.af_gamma * (1.0 - np.exp(-self.af_gamma * a))
            else:
                s = s + self.af_C * a
        return s

    # -- output ------------------------------------------------------------
    def to_sparlab(self) -> Dict[str, Any]:
        """The SparLab `material` block (`docs/configuration.md`).

        Always: name, youngs_modulus, poisson_ratio, density and a
        `plasticity` block with yield_stress, hardening_modulus,
        saturation_stress, saturation_rate, kinematic_hardening_modulus.
        With r-values: ``"yield_criterion": "hill48"`` and ``"anisotropy":
        {"r0", "r45", "r90", "rolling_direction": [1, 0, 0], "sheet_normal":
        [0, 0, 1]}``. With an Armstrong-Frederick term: ``"backstresses":
        [{"modulus": af_C, "recovery": af_gamma}]``.
        """
        plasticity: Dict[str, Any] = {
            "yield_stress": float(self.yield_stress),
            "hardening_modulus": float(self.hardening_modulus),
            "saturation_stress": float(self.saturation_stress),
            "saturation_rate": float(self.saturation_rate),
            "kinematic_hardening_modulus": float(self.kinematic_hardening_modulus),
        }
        if self.has_hill48:
            plasticity["yield_criterion"] = "hill48"
            plasticity["anisotropy"] = {"r0": float(self.r0), "r45": float(self.r45),
                                        "r90": float(self.r90),
                                        "rolling_direction": list(ROLLING_DIRECTION),
                                        "sheet_normal": list(SHEET_NORMAL)}
        if self.has_armstrong_frederick:
            plasticity["backstresses"] = [{"modulus": float(self.af_C),
                                           "recovery": float(self.af_gamma)}]
        return {"name": self.name, "youngs_modulus": float(self.youngs_modulus),
                "poisson_ratio": float(self.poisson_ratio), "density": float(self.density),
                "plasticity": plasticity}

    def to_dict(self) -> Dict[str, Any]:
        """Every field, for JSON (`from_dict` inverts it)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "Material":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(doc) - names
        if unknown:
            raise ValueError(f"Material: unknown keys {sorted(unknown)}")
        return cls(**doc)

    def replace(self, **changes: Any) -> "Material":
        """A copy with some fields changed (validated again)."""
        return dataclasses.replace(self, **changes)

    # -- construction from power laws --------------------------------------
    @classmethod
    def from_swift(cls, name: str, youngs_modulus: float, poisson_ratio: float,
                   density: float, K: float, eps0: float, n: float, *,
                   fit_range: Tuple[float, float] = DEFAULT_FIT_RANGE,
                   kinematic_hardening_modulus: float = 0.0,
                   points: int = 200, **extra: Any) -> "Material":
        """A material whose hardening approximates ``sigma = K (eps0 + a)^n``.

        The initial yield stress is exact, ``K eps0^n``; H, Q and delta
        minimise the squared stress error over `points` equally spaced plastic
        strains in `fit_range` (H, Q >= 0 by non-negative least squares for
        each delta, delta by a bounded scalar search on a log scale).
        `hardening_source` records the law, the range and the RMS and largest
        errors relative to the Swift stress. Other fields pass through
        `extra` (e.g. r-values, `source`).
        """
        K = require_positive("K", K)
        eps0 = require_positive("eps0", eps0)
        n = require_nonnegative("n", n)
        lo, hi = float(fit_range[0]), float(fit_range[1])
        if not (0.0 <= lo < hi):
            raise ValueError("fit_range must satisfy 0 <= low < high")
        a = np.linspace(lo, hi, points)
        target = K * (eps0 + a) ** n
        sy0 = K * eps0 ** n
        rhs = target - sy0

        def solve(log_delta: float):
            delta = math.exp(log_delta)
            A = np.column_stack([a, 1.0 - np.exp(-delta * a)])
            coef, res = nnls(A, rhs)
            return coef, float(res)

        best = minimize_scalar(lambda ld: solve(ld)[1], bounds=(math.log(0.1), math.log(500.0)),
                               method="bounded", options={"xatol": 1e-6})
        coef, _ = solve(best.x)
        delta = math.exp(best.x)
        H, Q = float(coef[0]), float(coef[1])
        if Q <= 0.0:
            delta = 0.0
        fitted = sy0 + H * a + Q * (1.0 - np.exp(-delta * a))
        rel = (fitted - target) / target
        info = {"law": "swift", "K": K, "eps0": eps0, "n": n, "fit_range": [lo, hi],
                "rms_relative_error": float(np.sqrt(np.mean(rel ** 2))),
                "max_relative_error": float(np.abs(rel).max())}
        return cls(name=name, youngs_modulus=youngs_modulus, poisson_ratio=poisson_ratio,
                   density=density, yield_stress=sy0, hardening_modulus=H,
                   saturation_stress=Q, saturation_rate=delta,
                   kinematic_hardening_modulus=kinematic_hardening_modulus,
                   hardening_source=info, **extra)

    @classmethod
    def from_hollomon(cls, name: str, youngs_modulus: float, poisson_ratio: float,
                      density: float, K: float, n: float, yield_stress: float,
                      **kwargs: Any) -> "Material":
        """Hollomon ``sigma = K eps^n`` from the yield point on: the Swift law
        with eps0 = (yield_stress / K)^(1/n), so the curve starts at the
        given yield stress (see `from_swift` for the fit)."""
        K = require_positive("K", K)
        n = require_positive("n", n)
        sy = require_positive("yield_stress", yield_stress)
        eps0 = (sy / K) ** (1.0 / n)
        mat = cls.from_swift(name, youngs_modulus, poisson_ratio, density, K, eps0, n, **kwargs)
        src = dict(mat.hardening_source)
        src["law"] = "hollomon (as swift with eps0 = (yield_stress / K)^(1/n))"
        return dataclasses.replace(mat, hardening_source=src)


def _library() -> Dict[str, Material]:
    lib = {}

    def add(mat: Material) -> None:
        lib[mat.name] = mat

    # Hollomon K, n and yield stress of the tempers, and r-values, at handbook
    # order of magnitude. Kinematic hardening is left at zero: its parameters
    # come from reverse-loading tests, which handbooks do not give.
    add(Material.from_hollomon("AA5754-O", 70.0e9, 0.33, 2670.0, K=420e6, n=0.30,
                               yield_stress=100e6, r0=0.75, r45=0.70, r90=0.80,
                               source=NOMINAL_SOURCE))
    add(Material.from_hollomon("AA2024-O", 73.1e9, 0.33, 2780.0, K=330e6, n=0.20,
                               yield_stress=75e6, r0=0.70, r45=0.80, r90=0.70,
                               source=NOMINAL_SOURCE))
    add(Material.from_hollomon("AA6061-T6", 68.9e9, 0.33, 2700.0, K=410e6, n=0.06,
                               yield_stress=276e6, r0=0.60, r45=0.70, r90=0.65,
                               source=NOMINAL_SOURCE))
    add(Material.from_hollomon("AA7075-O", 71.7e9, 0.33, 2810.0, K=400e6, n=0.17,
                               yield_stress=103e6, r0=0.70, r45=0.90, r90=0.80,
                               source=NOMINAL_SOURCE))
    add(Material.from_hollomon("DC04", 210.0e9, 0.30, 7850.0, K=530e6, n=0.22,
                               yield_stress=170e6, r0=1.90, r45=1.50, r90=2.20,
                               source=NOMINAL_SOURCE))
    add(Material.from_hollomon("Ti-6Al-4V-annealed", 113.8e9, 0.342, 4430.0, K=1100e6,
                               n=0.04, yield_stress=880e6, source=NOMINAL_SOURCE))
    return lib


_LIBRARY: Optional[Dict[str, Material]] = None


def library() -> Dict[str, Material]:
    """The built-in materials by name (AA5754-O, AA2024-O, AA6061-T6,
    AA7075-O, DC04, Ti-6Al-4V-annealed).

    Every entry is marked in `source`: nominal handbook-order values, not
    certified data. The hardening is a linear + Voce fit to a Hollomon law
    over plastic strains 0-0.6 (`hardening_source` gives the fit error).
    """
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = _library()
    return dict(_LIBRARY)


def get_material(name: str) -> Material:
    """A library material by name (case-insensitive); KeyError lists the names."""
    lib = library()
    for key, mat in lib.items():
        if key.lower() == name.lower():
            return mat
    raise KeyError(f"unknown material {name!r}; the library has {sorted(lib)}")
