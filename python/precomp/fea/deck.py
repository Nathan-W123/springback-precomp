"""The sparlab_form input deck.

`build_deck` writes, into one directory,

* `deck.json` - a SparLab JSON deck (`docs/configuration.md`): mesh,
  material, model and the `forming` block of `docs/forming.md` (section 2)
  with the tool, the ordered steps and their constraints, the kinematics,
  the Newton settings and the output. It has no top-level
  `boundary_conditions`, `load_cases` or `nonlinear` block: every step lists
  its own constraints, the forming analysis applies no load case, and
  sparlab_form reads no `nonlinear` block. Every key is one sparlab_form
  reads - the runner runs it with `--strict-config`, which refuses any other;
* `toolpath.csv` - the tool-centre trajectory, columns `t,x,y,z`, t strictly
  increasing from 0 to 1;
* `commanded.npz`, `precomp_deck.json` - the commanded surface and the
  setup, for provenance. The solver does not read them and they are not part
  of the content hash.

The `forming` keys are written in one place, `forming_block`, so a change of
the C++ contract is a change here only.

Geometry: the blank is a structured mesh of the square [-L/2, L/2]^2 by
[-t, 0], L the blank size rounded to whole elements; the clamp fixes every
node (through the thickness) whose x or y lies within `clamp_margin` of the
blank edge; the tool contacts the top faces (z = 0) inside the clamp; the
3-2-1 release support fixes x, y, z of the top node nearest (-a, -a, 0), y
and z of the one nearest (a, -a, 0) and z of the one nearest (-a, a, 0),
with a = L/2 - clamp_margin / 2 (three flange points in the former clamp):
six constraints, no rigid-body mode left and no constraint on the
springback.

Steps: "form" (the tool follows the trajectory from t = 0 to its last point
in contact with the part, clamp held: the shape under the tool), "unload"
(a release: the tool is removed and its force ramped out, clamp held - the
springback in the fixture) and, with release "321", "release" (the clamp
replaced by the 3-2-1 support - the free springback). The final retract of
the trajectory is not simulated: the "unload" step takes the tool away.

Support (`FormingSetup.support`, `precomp.fea.support`): a backing plate
("plate", a still plane under the sheet outside an opening around the part)
or a DSIF support ball ("support", under the sheet, its trajectory
`support_path.csv` on the forming tool's pseudo-time) is active in "form"
beside the tool and removed with it in "unload". With the DSIF rim pass,
"rim_pass" (the support alone, pseudo-time from 2) and "rim_unload" (its
removal) follow "unload", clamp held. The plate's opening and the rim pass
band follow the outline of `target` (the part the fixture is made for),
the commanded surface's when none is given.

Every step constraint is `"mode": "hold"`: its DOFs stay where the step
finds them - the clamp at the reference position, the three support nodes
where the clamp left them, so the released part keeps its place on the
fixture. sparlab_form warns that the clamp of "unload" holds reactions (it
is not statically determinate): that is the springback the fixture
prevents, which "release" lets go.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .. import __version__
from .._util import (PathLike, PrecompError, canonical_json, read_json, sha256_bytes,
                     to_jsonable, write_json)
from ..geometry.heightmap import HeightMap
from ..toolpath import AIR, Toolpath, spiral_toolpath
from .setup import CONTACT_KEYS, FORMING_SOLVER_KEYS, NEWTON_KEYS, FormingSetup
from .support import (COMMAND_TOLERANCE, RIM_PASS_WINDOW, SUPPORT_TOOL, SupportPlan,
                      check_command, plan_support)

DECK_FILE = "deck.json"
TOOLPATH_FILE = "toolpath.csv"
COMMANDED_FILE = "commanded.npz"
PROVENANCE_FILE = "precomp_deck.json"
TOOL_NAME = "tool"


def forming_surface(commanded: HeightMap) -> HeightMap:
    """The surface the forming tool's path is made for: the command where it
    lies below the sheet plane, the plane where it rises above it. The tool
    presses from above and cannot lift the sheet; only a support's rim pass
    realises a command above the plane (`precomp.fea.support`). A command
    that rises nowhere more than `COMMAND_TOLERANCE` above the plane (1 um of
    round-off) is taken as it is, as before supports existed."""
    if not (commanded.z > COMMAND_TOLERANCE).any():
        return commanded
    return commanded.with_z(np.minimum(commanded.z, 0.0))


def make_toolpath(setup: FormingSetup, commanded: HeightMap) -> Toolpath:
    """The spiral tool path the setup prescribes for `commanded` (step-down,
    spacing, direction and tool radius from the setup), made for its part
    below the sheet plane (`forming_surface`)."""
    return spiral_toolpath(forming_surface(commanded), setup.tool_radius, setup.step_down,
                           setup.toolpath_spacing, direction=setup.toolpath_direction)


def _clamp_region(setup: FormingSetup) -> Dict[str, Any]:
    e = setup.free_half_width
    return {"name": "clamp", "any_of": [
        {"box": {"xmax": -e}}, {"box": {"xmin": e}},
        {"box": {"ymax": -e}}, {"box": {"ymin": e}}]}


def clamp_condition(setup: FormingSetup) -> Dict[str, Any]:
    """The clamped frame, a step constraint: x, y, z held (at the reference
    position, where the analysis starts)."""
    return {"name": "clamp", "fix": ["x", "y", "z"], "mode": "hold",
            "region": _clamp_region(setup)}


def support_321(setup: FormingSetup) -> List[Dict[str, Any]]:
    """The 3-2-1 release support (see the module docstring), as held step
    constraints."""
    a = 0.5 * setup.meshed_blank_size - 0.5 * setup.clamp_margin
    return [
        {"name": "support_xyz", "fix": ["x", "y", "z"], "mode": "hold",
         "region": {"nearest_node": [-a, -a, 0.0]}},
        {"name": "support_yz", "fix": ["y", "z"], "mode": "hold",
         "region": {"nearest_node": [a, -a, 0.0]}},
        {"name": "support_z", "fix": ["z"], "mode": "hold",
         "region": {"nearest_node": [-a, a, 0.0]}},
    ]


def form_end(path: Toolpath) -> float:
    """The pseudo-time of the last point of `path` in contact with the part
    (not `AIR`) - where the "form" step ends; 1 if every point is in the air."""
    contact = np.flatnonzero(path.level != AIR)
    return float(path.t[contact[-1]]) if len(contact) else 1.0


def forming_block(setup: FormingSetup, toolpath_file: str = TOOLPATH_FILE,
                  t_form_end: float = 1.0,
                  support: Optional[SupportPlan] = None) -> Dict[str, Any]:
    """The deck's `forming` object (docs/forming.md, section 2): the
    kinematics, the spherical tool (and the support's tools, `support`), the
    ordered steps (see the module docstring; "form" runs from t = 0 to
    `t_form_end`), the Newton settings of `setup.solver` and the output."""
    if support is None:
        if setup.support != "none":
            raise PrecompError(f"support {setup.support!r} needs its plan "
                               "(precomp.fea.support.plan_support); build_deck makes it")
        support = SupportPlan()
    e = setup.free_half_width
    surface = {"name": "tool_side", "box": {"zmin": 0.0, "xmin": -e, "xmax": e,
                                            "ymin": -e, "ymax": e}}
    tool: Dict[str, Any] = {"name": TOOL_NAME, "shape": "sphere",
                            "radius": float(setup.tool_radius), "surface": surface,
                            "friction": float(setup.friction)}
    tool.update({k: setup.contact[k] for k in CONTACT_KEYS if k in setup.contact})
    tool["trajectory"] = {"file": toolpath_file}
    clamp = clamp_condition(setup)
    steps: List[Dict[str, Any]] = [
        {"name": "form", "type": "form", "tools": [TOOL_NAME] + list(support.form_tools),
         "time": [0.0, float(t_form_end)],
         "max_tool_travel": float(setup.max_tool_travel), "boundary_conditions": [clamp]},
        {"name": "unload", "type": "release", "tools": [], "boundary_conditions": [clamp]},
    ]
    if support.rim_pass_end is not None:
        steps += [
            {"name": "rim_pass", "type": "form", "tools": [SUPPORT_TOOL],
             "time": [float(RIM_PASS_WINDOW[0]), float(support.rim_pass_end)],
             "max_tool_travel": float(setup.max_tool_travel), "boundary_conditions": [clamp]},
            {"name": "rim_unload", "type": "release", "tools": [],
             "boundary_conditions": [clamp]},
        ]
    if setup.release == "321":
        steps.append({"name": "release", "type": "release", "tools": [],
                      "boundary_conditions": support_321(setup)})
    block: Dict[str, Any] = {"kinematics": setup.kinematics}
    block.update({k: setup.solver[k] for k in FORMING_SOLVER_KEYS if k in setup.solver})
    block["tools"] = [tool] + [dict(t) for t in support.tools]
    block["steps"] = steps
    newton = {k: setup.solver[k] for k in NEWTON_KEYS if k in setup.solver}
    if newton:
        block["newton"] = newton
    block["output"] = {"vtk": True, "snapshots": "steps"}
    return block


def model_block(setup: FormingSetup) -> Dict[str, Any]:
    """The deck's `model` object: the stress state and, when the setup asks
    for them, the Hex8 formulation (`element_formulation`) and the Gauss
    points through the thickness (`integration.thickness_points`, along z).
    A setup at the defaults ("standard", 0) writes neither key, so its deck -
    and its content hash - is the one it was before these settings existed."""
    model: Dict[str, Any] = {"stress_state": "three_dimensional"}
    if setup.element_formulation != "standard":
        model["element_formulation"] = setup.element_formulation
    if setup.thickness_points:
        model["integration"] = {"thickness_points": int(setup.thickness_points),
                                "thickness_direction": "z"}
    return model


def deck_document(setup: FormingSetup, t_form_end: float = 1.0,
                  support: Optional[SupportPlan] = None) -> Dict[str, Any]:
    """The full deck.json content for `setup`; the commanded shape enters
    through the trajectory file and the end of the "form" step, `t_form_end`
    (`form_end` of the tool path), and the support's plan (`support`, needed
    unless the setup has none)."""
    n = setup.elements_per_side
    L = setup.meshed_blank_size
    return {
        "name": setup.name,
        "description": (f"{_PROCESS[setup.support]} of a {setup.material.name} blank, "
                        f"written by precomp {__version__}"),
        "units": "SI",
        "mesh": {"type": "structured_hex", "nx": n, "ny": n, "nz": setup.layers,
                 "lx": L, "ly": L, "lz": setup.thickness,
                 "x0": -0.5 * L, "y0": -0.5 * L, "z0": -setup.thickness},
        "material": setup.material.to_sparlab(),
        "model": model_block(setup),
        "forming": forming_block(setup, t_form_end=t_form_end, support=support),
    }


_PROCESS = {"none": "Single-point incremental forming",
            "backing_plate": "Single-point incremental forming on a backing plate",
            "dsif": "Double-sided incremental forming"}


def check_toolpath(setup: FormingSetup, path: Toolpath) -> None:
    """Raise PrecompError if the tool would reach the clamped frame: every
    in-contact tool centre must stay a tool radius inside the free window."""
    contact = path.level != AIR
    xy = np.abs(path.points[contact, :2])
    reach = float(xy.max()) + setup.tool_radius if len(xy) else 0.0
    if reach > setup.free_half_width:
        raise PrecompError(
            f"the tool path reaches {reach:.4g} m from the centre (tool radius included), "
            f"beyond the unclamped half-width {setup.free_half_width:.4g} m; enlarge the "
            "blank or reduce the clamp margin")


def build_deck(setup: FormingSetup, commanded: HeightMap, out_dir: PathLike,
               toolpath: Optional[Toolpath] = None,
               target: Optional[HeightMap] = None) -> Path:
    """Write the deck for forming `commanded` with `setup` into `out_dir`.

    The tool path is generated from `commanded` (its part below the sheet
    plane, `forming_surface`) with the setup's path settings unless given.
    `target` is the part the fixture is made for: with a support strategy
    the backing plate's opening and the rim pass band follow its outline -
    fixed hardware, the same for every compensated command of that part
    (None: the commanded surface's outline). Returns the directory. Raises
    PrecompError when a path would reach the clamp, or when the command
    rises above the sheet plane where no tool can push it
    (`precomp.fea.support.check_command`).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    check_command(setup, commanded, target)
    path = toolpath if toolpath is not None else make_toolpath(setup, commanded)
    if abs(path.tool_radius - setup.tool_radius) > 1e-12:
        raise PrecompError("the tool path was made for a different tool radius")
    check_toolpath(setup, path)
    plan = plan_support(setup, commanded, forming_surface(commanded), path, target)
    write_json(out / DECK_FILE, deck_document(setup, form_end(path), plan))
    path.to_sparlab_csv(out / TOOLPATH_FILE)
    for name, text in plan.files.items():
        (out / name).write_text(text, encoding="ascii")
    commanded.save(out / COMMANDED_FILE)
    prov = {
        "precomp_version": __version__,
        "setup": setup.to_dict(),
        "commanded_grid": commanded.grid.to_dict(),
        "commanded_metadata": to_jsonable(commanded.metadata),
        "toolpath": path.summary(),
        "toolpath_metadata": to_jsonable(path.metadata),
    }
    if setup.support != "none":
        prov["support"] = to_jsonable(plan.info)
    write_json(out / PROVENANCE_FILE, prov)
    return out


def trajectory_files(deck: Dict[str, Any]) -> List[str]:
    """The trajectory files the deck's tools read, in tool order (each once)."""
    out: List[str] = []
    for tool in deck.get("forming", {}).get("tools", []):
        name = tool.get("trajectory", {}).get("file")
        if name and name not in out:
            out.append(name)
    return out


def deck_hash(deck_dir: PathLike, solver_version: str) -> str:
    """Content hash of a deck: SHA-256 of the canonical deck JSON, the
    trajectory CSVs its tools read (in tool order: `toolpath.csv`, then a
    support's) and the solver's `--version` text. Two decks with the same
    hash are the same run."""
    d = Path(deck_dir)
    try:
        doc = read_json(d / DECK_FILE)
        deck = canonical_json(doc).encode("utf-8")
        files = trajectory_files(doc) or [TOOLPATH_FILE]
        trajectories = [(d / name).read_bytes() for name in files]
    except FileNotFoundError as exc:
        raise PrecompError(f"{d} is not a complete deck directory: {exc}") from exc
    return sha256_bytes(deck, *trajectories, solver_version.strip().encode("utf-8"))
