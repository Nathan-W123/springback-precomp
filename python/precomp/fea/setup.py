"""FormingSetup: everything about a forming run except the commanded shape.

A setup is JSON-serialisable (`to_dict` / `from_dict`) so that a data set can
record exactly how every sample was simulated. Fields that change the physics
go into the deck and hence into the content hash of a run; the fields that
only change how the run is executed (`executable`, `timeout`, `threads`) do
not.

`support` selects how the sheet is supported from below while it is formed
(`SUPPORTS`; the geometry is `precomp.fea.support`): "none" (single-point
incremental forming, the default), "backing_plate" (a rigid plate under the
sheet outside an opening around the part) or "dsif" (a second, support tool
under the sheet, double-sided incremental forming). `support_settings` holds
that strategy's parameters (`SUPPORT_SETTINGS`); keys left out take the
defaults of `resolved_support`.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .._util import PrecompError, require_nonnegative, require_positive
from ..materials import Material, get_material

#: Environment variable naming the sparlab_form executable.
EXECUTABLE_ENV = "PRECOMP_SPARLAB_FORM"

#: Default executable, relative to the working directory or the repository root.
DEFAULT_EXECUTABLE = "build/bin/sparlab_form"

#: Fields that only affect how a run is executed, not its result.
EXECUTION_FIELDS = ("executable", "timeout", "threads")

#: The kinematics of the deck's `forming` block (docs/forming.md, section 2):
#: "finite_logarithmic" is the large-strain formulation (the plastic return in
#: the logarithmic strain) and the default, since incremental forming reaches
#: plastic strains of order one; "finite" (Green-Lagrange strain) and
#: "small_strain" are the small-strain laws.
KINEMATICS = ("finite_logarithmic", "finite", "small_strain")

#: `contact` keys: the tool's penalty settings in the deck (docs/forming.md):
#: `penalty` the scale s of kappa = s E / h (default 10), `tangential_penalty`
#: kappa_T / kappa (default 1).
CONTACT_KEYS = ("penalty", "tangential_penalty")

#: `solver` keys that go into the `forming.newton` block ...
NEWTON_KEYS = ("max_iterations", "residual_tolerance", "displacement_tolerance",
               "line_search", "max_cuts", "max_increments")
#: ... and those that go into the `forming` block itself.
FORMING_SOLVER_KEYS = ("friction_tangent", "solver", "mean_dilatation")

#: Hex8 formulations (`FormingSetup.element_formulation`, the deck's
#: `model.element_formulation`, docs/configuration.md): "standard" (the
#: isoparametric Hex8; with a plastic material sparlab_form's
#: `mean_dilatation: auto` gives it B-bar) or "incompatible_modes" (the Hex8
#: with Wilson-Taylor incompatible modes, which does not lock in bending; Hex8
#: only, and not with `mean_dilatation` "all").
ELEMENT_FORMULATIONS = ("standard", "incompatible_modes")

#: Rim-support strategies (`FormingSetup.support`; `precomp.fea.support`).
SUPPORTS = ("none", "backing_plate", "dsif")

#: The `support_settings` keys of each strategy and their defaults; None
#: means "taken from the setup" (see `FormingSetup.resolved_support`).
#:
#: backing_plate
#:   clearance [m] - the plate's opening is the part's outline (on the target
#:       the fixture is made for) grown by this; the plate carries the bottom
#:       faces wholly outside it (on the mesh, up to one element more).
#:   friction [-] - Coulomb coefficient sheet / plate (default: `friction`).
#:   penalty [-] - penalty scale of the plate contact (default: the tool's).
#: dsif
#:   radius [m] - the support ball's radius (default: `tool_radius`).
#:   squeeze [-] - how far the support is pushed towards the forming tool,
#:       as a fraction of the local sheet thickness: the gap between the
#:       two balls along the surface normal is (1 - squeeze) t_n. 0 (the
#:       default) makes them just touch a sheet of thickness t_n.
#:   thickness_law - t_n: "sine" (t cos(wall angle), the sine law; the
#:       default) or "initial" (t).
#:   friction [-], penalty [-] - of the support contact (defaults as above).
#:   clearance [m] - how far below the part the support waits while the
#:       forming tool is in the air (default 2 mm).
#:   rim_pass (bool) - after forming, the support alone traces the rim band
#:       from below, pushing the sheet up to the commanded surface
#:       (`precomp.fea.support`, "rim pass"); off by default.
#:   rim_inside, rim_outside [m] - the band the rim pass sweeps: from this far
#:       inside the part's outline to this far outside it (defaults 2 mm, 1 mm).
#:   rim_spacing [m] - distance between its loops (default 1 mm).
#:   rim_max_raise [m] - the highest a command may rise above the sheet plane
#:       in the band the rim pass realises (default 2 mm).
SUPPORT_SETTINGS: Dict[str, Dict[str, Any]] = {
    "none": {},
    "backing_plate": {"clearance": 1.0e-3, "friction": None, "penalty": None},
    "dsif": {"radius": None, "squeeze": 0.0, "thickness_law": "sine", "friction": None,
             "penalty": None, "clearance": 2.0e-3, "rim_pass": False, "rim_inside": 2.0e-3,
             "rim_outside": 1.0e-3, "rim_spacing": 1.0e-3, "rim_max_raise": 2.0e-3},
}


#: The process of the springback benchmark (benchmarks/springback): a 40 x 40
#: x 1 mm blank, 5 mm clamp, 4 mm tool, 1 mm spiral, 3-2-1 release.
_BENCHMARK_PROCESS: Dict[str, Any] = {
    "material": "AA5754-O", "blank_size": 0.04, "clamp_margin": 0.005, "thickness": 1e-3,
    "element": "hex8", "tool_radius": 4e-3, "step_down": 1e-3, "friction": 0.1,
    "toolpath_style": "spiral", "toolpath_spacing": 1e-3, "max_tool_travel": 1e-3,
    "release": "321", "kinematics": "finite_logarithmic", "threads": 1}

#: Named setups (`FormingSetup.preset`, `precomp setup --preset`), the
#: FormingSetup fields each one sets. Presets are never changed once a record
#: uses them; a new discretisation gets a new name.
#:
#: springback_2mm
#:   the springback benchmark as it was run (benchmarks/springback,
#:   protocol 1): 20 x 20 x 2 standard Hex8 (2 mm in plane, mean dilatation),
#:   2 x 2 x 2 points, penalty 10. Not converged: the physics audit
#:   (benchmarks/physics_audit) puts its released shape 0.08-0.12 mm RMS from
#:   the converged answer.
#: springback_fine
#:   the physics audit's recommended production setup on the same process
#:   (benchmarks/physics_audit/README.md, section 5): 48 x 48 x 1 Hex8 with
#:   incompatible modes (in-plane 40/48 = 0.833 mm, which divides the blank
#:   and the 5 mm clamp), 5 Gauss points through the thickness, penalty 10
#:   (the default, so `contact` stays empty and the deck is the audit's) -
#:   for validation and compensation runs. About 0.02-0.03 mm RMS from
#:   converged; 7-11 times the cost of springback_2mm.
#: springback_fine_bulk
#:   springback_fine with penalty 3, for bulk surrogate data: +0.005-0.009 mm
#:   RMS (always towards more sag) at about half the cost.
PRESETS: Dict[str, Dict[str, Any]] = {
    "springback_2mm": {**_BENCHMARK_PROCESS, "element_size": 2e-3, "layers": 2},
    "springback_fine": {**_BENCHMARK_PROCESS, "element_size": 0.04 / 48, "layers": 1,
                        "element_formulation": "incompatible_modes", "thickness_points": 5,
                        "contact": {}},
    "springback_fine_bulk": {**_BENCHMARK_PROCESS, "element_size": 0.04 / 48, "layers": 1,
                             "element_formulation": "incompatible_modes",
                             "thickness_points": 5, "contact": {"penalty": 3.0}},
}


def preset_fields(name: str) -> Dict[str, Any]:
    """A copy of preset `name`'s FormingSetup fields (`PRESETS`; the material
    by library name). Raises ValueError for an unknown name."""
    import copy

    if name not in PRESETS:
        raise ValueError(f"unknown setup preset {name!r}; presets: {', '.join(PRESETS)}")
    return copy.deepcopy(PRESETS[name])


def repository_root() -> Optional[Path]:
    """The SparLab checkout this package lives in (an editable install), or None."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "CMakeLists.txt").is_file() and (parent / "python" / "precomp").is_dir():
            return parent
    return None


def _check_keys(name: str, doc: Any, allowed: Tuple[str, ...]) -> None:
    if not isinstance(doc, dict):
        raise ValueError(f"{name} must be a dict")
    unknown = sorted(set(doc) - set(allowed))
    if unknown:
        raise ValueError(f"{name}: unknown keys {unknown}; sparlab_form reads "
                         f"{', '.join(allowed)}")


@dataclass
class FormingSetup:
    """Process, discretisation and solver settings of one SPIF simulation (SI).

    Blank and mesh
      blank_size : side of the square blank [m], centred on the origin.
      clamp_margin : width of the clamped frame along the blank edges [m]
          (at least one element).
      thickness : initial sheet thickness t [m]; the blank occupies
          z in [-t, 0], its tool side at z = 0.
      element_size : in-plane element size [m] (the blank gets
          round(blank_size / element_size) elements per side).
      layers : elements through the thickness.
      element : "hex8" (structured_hex), the only one.
      element_formulation : "standard" (the default) or "incompatible_modes"
          (Hex8 only; `ELEMENT_FORMULATIONS`), the deck's
          `model.element_formulation`.
      thickness_points : Gauss points through the thickness of each Hex8
          layer, 1-7 (the deck's `model.integration.thickness_points`, along
          z); 0 (the default) leaves the solver's rule (2) and writes nothing.
    Material
      material : a `Material`.
    Tool and path
      tool_radius [m], friction (Coulomb coefficient) [-], step_down [m],
      toolpath_style ("spiral", the only one), toolpath_spacing [m],
      toolpath_direction ("ccw" | "cw").
      contact : the tool's penalty contact settings, written into the tool
          object: `penalty` (the scale s of the contact stiffness
          kappa = s E / h, default 10) and `tangential_penalty` (the friction
          stiffness over kappa, default 1); no other keys.
    Steps
      max_tool_travel : increment control, the largest tool-centre travel per
          increment of the "form" step [m].
      release : "321" - after the tool is withdrawn (step "unload", still
          clamped) the clamp is replaced by a statically determinate 3-2-1
          support (step "release"), so the part springs back freely;
          "clamped_only" - springback within the fixture only.
      kinematics : the `forming` block's kinematics, "finite_logarithmic"
          (large strain, the default), "finite" or "small_strain".
      solver : Newton and linear-solver settings: the keys of the deck's
          `forming.newton` block (max_iterations, residual_tolerance,
          displacement_tolerance, line_search, max_cuts, max_increments) and
          friction_tangent, solver, mean_dilatation of the `forming` block.
    Support (see the module docstring)
      support : "none" | "backing_plate" | "dsif" (`SUPPORTS`).
      support_settings : that strategy's parameters (`SUPPORT_SETTINGS`);
          a key of another strategy is refused.
    Execution (not part of the physics hash)
      executable : path of sparlab_form; None means $PRECOMP_SPARLAB_FORM,
          else build/bin/sparlab_form relative to the working directory or
          the repository root.
      timeout : wall-clock limit per run [s]; threads : OMP_NUM_THREADS.
    """

    material: Material
    blank_size: float = 0.20
    clamp_margin: float = 0.02
    thickness: float = 1.0e-3
    element_size: float = 2.5e-3
    layers: int = 2
    element: str = "hex8"
    element_formulation: str = "standard"
    thickness_points: int = 0
    tool_radius: float = 5.0e-3
    friction: float = 0.1
    step_down: float = 0.5e-3
    toolpath_style: str = "spiral"
    toolpath_spacing: float = 1.0e-3
    toolpath_direction: str = "ccw"
    contact: Dict[str, Any] = field(default_factory=dict)
    max_tool_travel: float = 1.0e-3
    release: str = "321"
    kinematics: str = "finite_logarithmic"
    solver: Dict[str, Any] = field(default_factory=dict)
    support: str = "none"
    support_settings: Dict[str, Any] = field(default_factory=dict)
    name: str = "spif"
    executable: Optional[str] = None
    timeout: float = 24 * 3600.0
    threads: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.material, dict):
            self.material = Material.from_dict(self.material)
        elif isinstance(self.material, str):
            self.material = get_material(self.material)
        if not isinstance(self.material, Material):
            raise ValueError("material must be a Material, a Material dict or a library name")
        for key in ("blank_size", "thickness", "element_size", "tool_radius", "step_down",
                    "toolpath_spacing", "max_tool_travel", "timeout"):
            require_positive(key, getattr(self, key))
        require_positive("clamp_margin", self.clamp_margin)
        require_nonnegative("friction", self.friction)
        if not (isinstance(self.layers, int) and self.layers >= 1):
            raise ValueError(f"layers must be an integer >= 1, got {self.layers!r}")
        if not (isinstance(self.threads, int) and self.threads >= 1):
            raise ValueError(f"threads must be an integer >= 1, got {self.threads!r}")
        if self.element != "hex8":
            raise ValueError(f"element must be 'hex8', got {self.element!r}")
        if self.element_formulation not in ELEMENT_FORMULATIONS:
            raise ValueError("element_formulation must be one of "
                             f"{', '.join(ELEMENT_FORMULATIONS)}; got "
                             f"{self.element_formulation!r}")
        tp = self.thickness_points
        if not (isinstance(tp, int) and not isinstance(tp, bool) and 0 <= tp <= 7):
            raise ValueError(f"thickness_points must be an integer 0-7 (0: the solver's "
                             f"default), got {tp!r}")
        if tp == 1 and self.element_formulation == "incompatible_modes":
            raise ValueError("thickness_points 1 is refused with incompatible modes (the "
                             "mode gradients vanish at the element centre)")
        if self.toolpath_style != "spiral":
            raise ValueError(f"toolpath_style must be 'spiral', got {self.toolpath_style!r}")
        if self.toolpath_direction not in ("ccw", "cw"):
            raise ValueError("toolpath_direction must be 'ccw' or 'cw'")
        if self.release not in ("321", "clamped_only"):
            raise ValueError("release must be '321' or 'clamped_only'")
        if self.kinematics not in KINEMATICS:
            raise ValueError(f"kinematics must be one of {', '.join(KINEMATICS)}; got "
                             f"{self.kinematics!r}")
        _check_keys("contact", self.contact, CONTACT_KEYS)
        _check_keys("solver", self.solver, NEWTON_KEYS + FORMING_SOLVER_KEYS)
        for key in CONTACT_KEYS:
            if key in self.contact:
                require_positive(f"contact[{key!r}]", self.contact[key])
        md = self.solver.get("mean_dilatation")
        if self.element_formulation == "incompatible_modes" and (md is True or md == "all"):
            raise ValueError("solver['mean_dilatation'] 'all' is refused with incompatible "
                             "modes (the modes relax the constraint themselves)")
        if self.clamp_margin < self.element_size * (1 - 1e-9):
            raise ValueError("clamp_margin must be at least one element_size")
        if 2 * self.clamp_margin >= self.blank_size:
            raise ValueError("the clamped frame covers the whole blank")
        self.resolved_support()                 # validates support and its settings

    # -- derived -----------------------------------------------------------
    @property
    def elements_per_side(self) -> int:
        """In-plane elements per blank side, round(blank_size / element_size)."""
        return max(1, int(round(self.blank_size / self.element_size)))

    @property
    def meshed_blank_size(self) -> float:
        """Side of the meshed blank [m]: `blank_size` rounded to a whole number
        of elements (elements_per_side * element_size)."""
        return self.elements_per_side * self.element_size

    @property
    def free_half_width(self) -> float:
        """Half-width of the unclamped window [m]: meshed_blank_size / 2 - clamp_margin."""
        return 0.5 * self.meshed_blank_size - self.clamp_margin

    def resolved_support(self) -> Dict[str, Any]:
        """The support strategy's settings with every default filled in
        (`SUPPORT_SETTINGS`; friction and penalty from the forming tool's,
        the dsif radius the tool radius). Raises ValueError for an unknown
        strategy, a key it does not take or a value out of range."""
        if self.support not in SUPPORTS:
            raise ValueError(f"support must be one of {', '.join(SUPPORTS)}; got "
                             f"{self.support!r}")
        defaults = SUPPORT_SETTINGS[self.support]
        if not isinstance(self.support_settings, dict):
            raise ValueError("support_settings must be a dict")
        unknown = sorted(set(self.support_settings) - set(defaults))
        if unknown:
            takes = ", ".join(defaults) if defaults else "no settings"
            raise ValueError(f"support_settings: {unknown} are not settings of support "
                             f"{self.support!r} (it takes {takes})")
        out = dict(defaults)
        out.update(self.support_settings)
        if self.support == "none":
            return out
        if out.get("friction") is None:
            out["friction"] = float(self.friction)
        if out.get("penalty") is None:
            out["penalty"] = float(self.contact.get("penalty", 10.0))
        require_nonnegative("support_settings['friction']", out["friction"])
        require_positive("support_settings['penalty']", out["penalty"])
        if self.support == "backing_plate":
            require_nonnegative("support_settings['clearance']", out["clearance"])
            return out
        if out.get("radius") is None:
            out["radius"] = float(self.tool_radius)
        for key in ("radius", "clearance", "rim_spacing", "rim_max_raise"):
            require_positive(f"support_settings[{key!r}]", out[key])
        for key in ("rim_inside", "rim_outside"):
            require_nonnegative(f"support_settings[{key!r}]", out[key])
        sq = float(out["squeeze"])
        if not -1.0 <= sq < 1.0:
            raise ValueError("support_settings['squeeze'] must lie in [-1, 1) (a fraction of "
                             f"the sheet thickness); got {out['squeeze']!r}")
        if out["thickness_law"] not in ("sine", "initial"):
            raise ValueError("support_settings['thickness_law'] must be 'sine' or 'initial'")
        if not isinstance(out["rim_pass"], bool):
            raise ValueError("support_settings['rim_pass'] must be true or false")
        if out["rim_inside"] + out["rim_outside"] <= 0.0:
            raise ValueError("the rim pass band is empty (rim_inside + rim_outside = 0)")
        return out

    def resolved_executable(self) -> Path:
        """The sparlab_form executable this setup runs (see the class docstring).

        Raises PrecompError naming every place looked at when none exists.
        """
        tried = []
        candidates = []
        if self.executable:
            candidates.append(Path(self.executable))
        elif os.environ.get(EXECUTABLE_ENV):
            candidates.append(Path(os.environ[EXECUTABLE_ENV]))
        else:
            candidates.append(Path.cwd() / DEFAULT_EXECUTABLE)
            root = repository_root()
            if root is not None:
                candidates.append(root / DEFAULT_EXECUTABLE)
        for c in candidates:
            tried.append(str(c))
            if c.is_file() and os.access(c, os.X_OK):
                return c.resolve()
        raise PrecompError(
            "the SparLab forming executable was not found (looked at: " + ", ".join(tried)
            + f"); build it with `make build`, or set {EXECUTABLE_ENV} or "
            "FormingSetup.executable")

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        doc = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        doc["material"] = self.material.to_dict()
        doc["contact"] = dict(self.contact)
        doc["solver"] = dict(self.solver)
        doc["support_settings"] = dict(self.support_settings)
        return doc

    def physics_dict(self) -> Dict[str, Any]:
        """`to_dict` without the execution-only fields."""
        doc = self.to_dict()
        for key in EXECUTION_FIELDS:
            doc.pop(key)
        return doc

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "FormingSetup":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(doc) - names
        if unknown:
            raise ValueError(f"FormingSetup: unknown keys {sorted(unknown)}")
        if "material" not in doc:
            raise ValueError("FormingSetup: 'material' is required")
        return cls(**doc)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "FormingSetup":
        """The named setup `name` (`PRESETS`) with `overrides` (FormingSetup
        fields; e.g. a support, an executable) applied on top."""
        doc = preset_fields(name)
        doc.update(overrides)
        return cls.from_dict(doc)

    def replace(self, **changes: Any) -> "FormingSetup":
        return dataclasses.replace(self, **changes)

    def check_part_fits(self, footprint_radius: float) -> None:
        """Raise ValueError unless a part of that footprint radius [m], plus
        the tool radius, fits inside the unclamped window."""
        need = footprint_radius + self.tool_radius
        if need > self.free_half_width:
            raise ValueError(
                f"the part (footprint radius {footprint_radius:.4g} m plus tool radius "
                f"{self.tool_radius:.4g} m) does not fit in the unclamped window of "
                f"half-width {self.free_half_width:.4g} m")
