"""Rim support in the forming deck: a backing plate, or a DSIF support tool.

A single tool pushing from above cannot keep the sheet between the clamp and
the first contour from bending down: the released part comes out too deep
near its rim (`benchmarks/springback`). `FormingSetup.support` adds support
from below, with the existing rigid tools of `sparlab_form`
(docs/forming.md):

* ``"backing_plate"`` - a rigid plate under the sheet whose opening is the
  part's outline grown by `clearance` (partial support die). It is a
  `plane` tool (normal +z, through the sheet's bottom z = -t), held still,
  whose slave surface is the bottom faces lying wholly outside the opening:
  their nodes are listed by id (`node_ids`, the structured mesh's documented
  numbering, include/sparlab/mesh/StructuredMesh.hpp), since no geometric
  region primitive describes an arbitrary outline. The sheet outside the
  opening cannot go below the plate; nothing holds it down onto the plate.
  On the mesh the opening is the union of the elements it touches, up to
  one element wider than asked: `plate_nodes` reports the clearance the
  mesh realises, the smallest distance of a carried node to the outline
  (`outline_distance`). The plate is active in "form" and removed with the
  tool in "unload" (clamp still on), before the 3-2-1 release.
* ``"dsif"`` - double-sided incremental forming: a second ball (the
  support) under the sheet, on the bottom faces of the free window,
  following `precomp.toolpath.dsif_support_points` - opposite the forming
  tool along the surface normal, never above the sheet not yet formed -
  written with the forming tool's pseudo-time, so both tools move together
  in one "form" step; "unload" removes both. With `rim_pass`, after that
  unload the support alone runs `precomp.toolpath.rim_pass_path` (step
  "rim_pass", pseudo-time 2 to 3, clamp on) and is removed in "rim_unload",
  before the release.

What a command above the sheet plane means (`command_upper_bound`): the
forming tool presses from above and its path is made for the part of the
command below the plane (`precomp.fea.deck.make_toolpath`); a backing
plate at the sheet's bottom holds the sheet up to the plane outside the
opening but pushes nothing higher; the DSIF support pushes up only
opposite the forming tool, whose path never rises above the plane. Only
the rim pass pushes the sheet up by itself: in the band it sweeps (signed
distance to the target's outline in [-rim_inside, rim_outside], the flange
strip included, which displacement adjustment then adjusts instead of
holding - `compensation_masks`) the command may rise above the plane, by at
most `rim_max_raise` and no further than the rim pass written for it pushes
the sheet: the top of the volume its ball sweeps along its loops
(`precomp.toolpath.swept_ball_top`), plus the sheet thickness. The loops sit
at the lift cutter of the commanded underside, so a raise narrower than the
ball, a corner seen from below tighter than it, or a raise between two loops
is out of reach; `command_upper_bound` cuts the command to what the ball
reaches, to a fixed point. Every other command is kept at z <= 0 (to within
`COMMAND_TOLERANCE`), and `build_deck` refuses one above the bound.

All lengths are metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .._util import PrecompError
from ..geometry.heightmap import HeightMap
from ..toolpath import (AIR, Toolpath, dsif_support_points, rim_pass_path,
                        signed_outline_distance, swept_ball_top)
from .setup import CONTACT_KEYS, FormingSetup

#: Names of the support tools in the deck (fields of tool_forces.csv).
PLATE_TOOL = "plate"
SUPPORT_TOOL = "support"
#: The support tool's trajectory file, beside toolpath.csv.
SUPPORT_PATH_FILE = "support_path.csv"
#: The rim pass runs over this pseudo-time window, after "unload" (which
#: ends at most at 2: the forming tool's path spans [0, 1]).
RIM_PASS_WINDOW = (2.0, 3.0)
#: A command up to this far above what the tools realise is accepted [m]
#: (round-off of the conditioning; far below the mesh's resolution): above
#: the sheet plane without a rim pass, above `command_upper_bound` with one.
COMMAND_TOLERANCE = 1e-6
#: The fixed point of `command_upper_bound` is reached when an iteration
#: changes the command by at most this [m] ...
BOUND_CONVERGENCE = 1e-8
#: ... or after this many iterations.
BOUND_ITERATIONS = 8
#: Largest move of the DSIF support between two trajectory knots while both
#: tools are in contact [m]: the solver moves a tool in a straight line
#: between knots, so where the support swings round the forming tool (a
#: corner of the path) it would cut the chord, towards the forming tool, and
#: pinch the sheet; knots are added there (`dsif_trajectory`).
SUPPORT_KNOT_STEP = 0.5e-3
#: Rounds of knot insertion (each recomputes the support on the new knots).
SUPPORT_KNOT_ROUNDS = 4


@dataclass
class SupportPlan:
    """What a support strategy adds to the deck (`plan_support`).

    tools : tool objects after the forming tool; form_tools : their names
    active in "form"; rim_pass_end : the pseudo-time the "rim_pass" step
    ends at (None: no rim pass); files : {name: text} written beside the
    deck (trajectory files - part of the content hash); info : provenance.
    """

    tools: List[Dict[str, Any]] = field(default_factory=list)
    form_tools: List[str] = field(default_factory=list)
    rim_pass_end: Optional[float] = None
    files: Dict[str, str] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


def _contact(setup: FormingSetup, settings: Dict[str, Any]) -> Dict[str, Any]:
    out = {"friction": float(settings["friction"])}
    out.update({k: setup.contact[k] for k in CONTACT_KEYS if k in setup.contact})
    out["penalty"] = float(settings["penalty"])
    return out


def bottom_node_grid(setup: FormingSetup) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ids, x, y), each (n + 1, n + 1) [row j, column i], of the bottom
    nodes (z = -t) of the structured blank mesh: node (i, j, 0) has id
    j (n + 1) + i and sits at (-L/2 + i dx, -L/2 + j dx), n elements per
    side, dx = L / n (include/sparlab/mesh/StructuredMesh.hpp)."""
    n = setup.elements_per_side
    L = setup.meshed_blank_size
    c = -0.5 * L + L * np.arange(n + 1) / n
    X, Y = np.meshgrid(c, c, indexing="xy")
    ids = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    return ids, X, Y


def outline_distance(reference: HeightMap, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed distance [m] from points (x, y) to the outline of the part of
    `reference` (its nodes deeper than 1 um): negative inside, positive
    outside, the outline half-way between the last part node and the first
    node off it - `precomp.toolpath.signed_outline_distance` at the grid's
    nodes, and between them from the nearest part and non-part nodes (a
    point is inside when its nearest node is a part node)."""
    part = reference.mask & (reference.z < -1e-6)
    if not part.any():
        raise PrecompError("the reference surface has no part: a backing plate needs an "
                           "outline to follow")
    X, Y = reference.grid.mesh()
    q = np.column_stack([np.ravel(x), np.ravel(y)])
    d_part, _ = cKDTree(np.column_stack([X[part], Y[part]])).query(q)
    if (~part).any():
        d_off, _ = cKDTree(np.column_stack([X[~part], Y[~part]])).query(q)
    else:
        d_off = np.full(len(q), np.inf)
    half = 0.5 * reference.grid.h
    sd = np.where(d_part < d_off, -(d_off - half), d_part - half)
    return sd.reshape(np.shape(x))


def plate_nodes(setup: FormingSetup, reference: HeightMap,
                clearance: Optional[float] = None) -> Tuple[List[int], Dict[str, Any]]:
    """The bottom nodes the backing plate carries, and a report.

    A bottom face (cell) is carried when its four corners lie farther than
    `clearance` (default the setup's) outside the outline of the part of
    `reference` (`outline_distance`); the nodes are the corners of the
    carried cells (so every listed node is a slave node of a carried face
    and sparlab_form's face selection - every node of a face in the region -
    picks exactly those faces). The report gives the clearance asked, the
    one the mesh realises (the smallest distance of a carried node to the
    outline, never less than asked), and the counts.
    """
    s = setup.resolved_support()
    c = float(s["clearance"] if clearance is None else clearance)
    ids, X, Y = bottom_node_grid(setup)
    d = outline_distance(reference, X, Y)
    out = d > c
    cell = out[:-1, :-1] & out[1:, :-1] & out[:-1, 1:] & out[1:, 1:]
    carried = np.zeros_like(out)
    carried[:-1, :-1] |= cell
    carried[1:, :-1] |= cell
    carried[:-1, 1:] |= cell
    carried[1:, 1:] |= cell
    if not cell.any():
        raise PrecompError("the backing plate carries no face: the opening covers the blank")
    info = {"clearance_m": c, "realised_clearance_m": float(d[carried].min()),
            "carried_faces": int(cell.sum()), "carried_nodes": int(carried.sum()),
            "bottom_faces": int(cell.size)}
    return sorted(int(i) for i in ids[carried]), info


def plate_tool(setup: FormingSetup, reference: HeightMap) -> Tuple[Dict[str, Any],
                                                                   Dict[str, Any]]:
    """The backing plate's tool object (a still plane through the sheet's
    bottom, normal +z) and its report (`plate_nodes`)."""
    s = setup.resolved_support()
    nodes, info = plate_nodes(setup, reference)
    t = float(setup.thickness)
    tool: Dict[str, Any] = {"name": PLATE_TOOL, "shape": "plane", "normal": [0.0, 0.0, 1.0],
                            "surface": {"name": "plate_side", "node_ids": nodes}}
    tool.update(_contact(setup, s))
    tool["trajectory"] = {"times": [0.0, 1.0], "points": [[0.0, 0.0, -t], [0.0, 0.0, -t]]}
    return tool, info


def support_tool(setup: FormingSetup) -> Dict[str, Any]:
    """The DSIF support tool object: a ball on the bottom faces of the free
    window, driven by `SUPPORT_PATH_FILE`."""
    s = setup.resolved_support()
    e = setup.free_half_width
    t = float(setup.thickness)
    tool: Dict[str, Any] = {"name": SUPPORT_TOOL, "shape": "sphere", "radius": float(s["radius"]),
                            "surface": {"name": "support_side",
                                        "box": {"zmax": -t, "xmin": -e, "xmax": e,
                                                "ymin": -e, "ymax": e}}}
    tool.update(_contact(setup, s))
    tool["trajectory"] = {"file": SUPPORT_PATH_FILE}
    return tool


def _check_reach(setup: FormingSetup, centres: np.ndarray, normals: np.ndarray,
                 radius: float, what: str) -> None:
    """PrecompError unless every contact point (centre + radius * normal)
    lies inside the free window."""
    if not len(centres):
        return
    tip = centres[:, :2] + radius * normals[:, :2]
    reach = float(np.abs(tip).max())
    if reach > setup.free_half_width + 1e-12:
        raise PrecompError(f"the {what} reaches {reach:.4g} m from the centre, beyond the "
                           f"unclamped half-width {setup.free_half_width:.4g} m; enlarge the "
                           "blank or reduce the clamp margin")


def _csv(t: np.ndarray, pts: np.ndarray) -> str:
    lines = ["t,x,y,z"]
    for ti, (x, y, z) in zip(t, pts):
        lines.append(f"{ti:.17g},{x:.17g},{y:.17g},{z:.17g}")
    return "\n".join(lines) + "\n"


def rim_pass(setup: FormingSetup, commanded: HeightMap, reference: HeightMap) -> Toolpath:
    """The rim pass the deck writes for `commanded` (its full command, above
    the plane included) with the fixture made for `reference`: the
    setup's `precomp.toolpath.rim_pass_path` (the support ball, the band
    and loop spacing of `support_settings`, the point spacing and direction
    of the forming path)."""
    s = setup.resolved_support()
    return rim_pass_path(commanded, reference, float(s["radius"]), float(setup.thickness),
                         inside=float(s["rim_inside"]), outside=float(s["rim_outside"]),
                         band_spacing=float(s["rim_spacing"]),
                         spacing=float(setup.toolpath_spacing), clearance=float(s["clearance"]),
                         direction=setup.toolpath_direction)


def rim_pass_reach(setup: FormingSetup, commanded: HeightMap, path: Toolpath) -> np.ndarray:
    """(ny, nx) [m] on `commanded`'s grid: the highest the rim pass `path`
    pushes the sheet's top surface - the top of the volume its ball sweeps
    up to its last point in contact (`precomp.toolpath.swept_ball_top`;
    the steps "rim_pass" simulates), plus the sheet thickness (the vertical
    thickness stays t under the sine law). -inf where the ball never passes
    under a node."""
    on = np.flatnonzero(path.level != AIR)
    if not len(on):
        return np.full(commanded.grid.shape, -np.inf)
    pts = path.points[:on[-1] + 1]
    return swept_ball_top(pts, path.tool_radius, commanded.grid) + float(setup.thickness)


def dsif_trajectory(setup: FormingSetup, forming_surface: HeightMap, path: Toolpath,
                    commanded: Optional[HeightMap] = None,
                    reference: Optional[HeightMap] = None
                    ) -> Tuple[np.ndarray, np.ndarray, Optional[float], Dict[str, Any]]:
    """(t, points, rim_pass_end, info): the support tool's trajectory knots.

    On [0, 1] the knots of the forming tool's `path` (its `t`) with
    `dsif_support_points` over `forming_surface` (the surface the path was
    made for), plus knots on the path's segments wherever the support would
    otherwise move more than `SUPPORT_KNOT_STEP` between two knots in
    contact (swinging round the forming tool at a corner, a straight move
    would cut towards it and pinch the sheet); with `rim_pass`, the knots of `rim_pass_path` over
    `commanded` (its full command, above the plane included) and the
    outline of `reference`, on `RIM_PASS_WINDOW` by arc length, the pass
    ending (`rim_pass_end`) at its last point in contact."""
    s = setup.resolved_support()
    R2 = float(s["radius"])
    t0 = float(setup.thickness)

    def support(p: Toolpath) -> np.ndarray:
        return dsif_support_points(p, forming_surface, t0, R2, squeeze=float(s["squeeze"]),
                                   thickness_law=s["thickness_law"],
                                   clearance=float(s["clearance"]))

    # knots where the support moves more than SUPPORT_KNOT_STEP between two
    # contact knots: the forming tool's segment split evenly (collinear
    # points, so its motion and pseudo-time are unchanged)
    knots, tk = path, path.t
    pts = support(knots)
    added = 0
    for _ in range(SUPPORT_KNOT_ROUNDS):
        move = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        both = (knots.level[:-1] != AIR) & (knots.level[1:] != AIR)
        split = np.where(both, np.ceil(move / SUPPORT_KNOT_STEP - 1e-9), 1).astype(int)
        if not (split > 1).any():
            break
        P, L, Tn = [knots.points[:1]], [knots.level[:1]], [tk[:1]]
        for i, m in enumerate(split):
            a, b = knots.points[i], knots.points[i + 1]
            u = np.arange(1, m) / m
            P += [a + u[:, None] * (b - a), b[None]]
            L.append(np.full(m, knots.level[i + 1]))
            # the forming tool's knots keep their pseudo-time exactly
            Tn += [tk[i] + u * (tk[i + 1] - tk[i]), tk[i + 1:i + 2]]
        added += int((split - 1).sum())
        knots = Toolpath(np.vstack(P), np.concatenate(L), path.tool_radius, path.metadata)
        tk = np.concatenate(Tn)
        if len(tk) != len(knots.points):
            raise PrecompError("support knots: a repeated point on the tool path")
        pts = support(knots)
    contact = knots.level != AIR
    n = knots.points - pts
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    _check_reach(setup, pts[contact], n[contact], R2, "support tool")
    info: Dict[str, Any] = {"support_radius_m": R2, "squeeze": float(s["squeeze"]),
                            "thickness_law": s["thickness_law"],
                            "support_z_min_m": float(pts[:, 2].min()),
                            # how far the ball's body reaches in plan while in contact
                            # (a lower clamp frame starts at the free half-width)
                            "support_body_reach_m": float(np.abs(pts[contact, :2]).max() + R2)
                            if contact.any() else None,
                            "support_z_max_in_contact_m":
                                float(pts[contact, 2].max()) if contact.any() else None,
                            "support_knots_added": added}
    t = tk
    if not s["rim_pass"]:
        return t, pts, None, info
    if commanded is None or reference is None:
        raise PrecompError("the rim pass needs the commanded surface and the reference part")
    rim = rim_pass(setup, commanded, reference)
    up = np.array([0.0, 0.0, 1.0])
    rc = rim.level != AIR
    _check_reach(setup, rim.points[rc], np.tile(up, (int(rc.sum()), 1)), R2, "rim pass")
    a, b = RIM_PASS_WINDOW
    tr = a + (b - a) * rim.t
    last = int(np.flatnonzero(rc)[-1])
    info["rim_pass"] = {"band_levels_m": rim.metadata["band_levels_m"],
                        "length_m": rim.total_length, "points": int(len(rim.points)),
                        "z_max_m": float(rim.points[rc, 2].max()),
                        "z_min_m": float(rim.points[rc, 2].min())}
    return (np.concatenate([t, tr]), np.vstack([pts, rim.points]), float(tr[last]), info)


def plan_support(setup: FormingSetup, commanded: HeightMap, forming_surface: HeightMap,
                 path: Toolpath, reference: Optional[HeightMap]) -> SupportPlan:
    """The tools, steps and files the setup's support strategy adds (see the
    module docstring). `reference` is the part the fixture is made for (the
    plate's opening and the rim pass band follow its outline); None takes
    the commanded surface."""
    plan = SupportPlan(info={"support": setup.support})
    if setup.support == "none":
        return plan
    ref = commanded if reference is None else reference
    if not ref.grid.matches(commanded.grid):
        ref = ref.resample(commanded.grid)
    plan.info["outline_from"] = "commanded" if reference is None else "target"
    if setup.support == "backing_plate":
        tool, info = plate_tool(setup, ref)
        plan.tools.append(tool)
        plan.form_tools.append(PLATE_TOOL)
        plan.info["plate"] = info
        return plan
    t, pts, rim_end, info = dsif_trajectory(setup, forming_surface, path, commanded, ref)
    plan.tools.append(support_tool(setup))
    plan.form_tools.append(SUPPORT_TOOL)
    plan.files[SUPPORT_PATH_FILE] = _csv(t, pts)
    plan.rim_pass_end = rim_end
    plan.info["dsif"] = info
    return plan


def rim_band(setup: FormingSetup, target: HeightMap) -> Tuple[np.ndarray, np.ndarray]:
    """(band, strip), (ny, nx) masks on `target`'s grid: the part within
    `rim_inside` of the outline and the flange within `rim_outside` of it -
    what the rim pass sweeps. Both empty without a DSIF rim pass."""
    band = np.zeros(target.grid.shape, dtype=bool)
    strip = band.copy()
    if setup.support != "dsif" or not setup.resolved_support()["rim_pass"]:
        return band, strip
    s = setup.resolved_support()
    sd = signed_outline_distance(target).z
    part = target.mask & (target.z < -1e-6)
    band = part & (sd >= -float(s["rim_inside"]))
    strip = target.mask & ~part & (sd <= float(s["rim_outside"]))
    return band, strip


def compensation_masks(setup: FormingSetup, target: HeightMap
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(hold, adjust) for displacement adjustment with this setup, or
    (None, None) for the defaults (hold the flange, adjust the part). The rim
    pass pushes the flange strip it sweeps from below, so that strip is
    adjusted like the part instead of held at the sheet plane."""
    band, strip = rim_band(setup, target)
    if not strip.any():
        return None, None
    part = target.mask & (target.z < -1e-6)
    flange = target.mask & ~part
    return flange & ~strip, part | strip


def command_upper_bound(setup: FormingSetup, target: HeightMap):
    """The highest command the setup's tools can realise, as a function of
    the command: ``bound(z) -> (ny, nx)`` [m] on `target`'s grid, for
    `displacement_adjustment(upper_bound=...)` and `check_command`.

    0 (the sheet plane) everywhere for every strategy but the DSIF rim
    pass: the forming tool only pushes down, a plate at the sheet's bottom
    and the synchronised support push nothing above the plane. With the rim
    pass, on the band and strip it sweeps (`rim_band`), the command may rise
    above the plane by at most `rim_max_raise`, and only as far as the rim
    pass written for it pushes the sheet: `rim_pass_reach` of `rim_pass` -
    the top of the volume the ball sweeps along its loops, plus t. The loops
    sit at the lift cutter of the command's underside, so a raised band
    narrower than the ball, a corner seen from below tighter than it (the
    part's rim), the outermost loop held down by the flange beside it, or a
    raise between two loops cannot be pushed up by it and is cut to what the
    ball reaches (never below the plane). The cut lowers the underside the
    loops follow, so it is repeated on the cut command until an iteration
    changes it by at most `BOUND_CONVERGENCE` (at most `BOUND_ITERATIONS`
    times): the rim pass made for a command at or below the bound then
    reaches it (`check_command` verifies exactly that). Where nothing is
    raised the bound is `rim_max_raise` on the band (nothing to check).
    """
    band, strip = rim_band(setup, target)
    if not (band.any() or strip.any()):
        zero = np.zeros(target.grid.shape)
        return lambda z: zero.copy()
    s = setup.resolved_support()
    zone = band | strip
    cap = np.where(zone, float(s["rim_max_raise"]), 0.0)

    def bound(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if z.shape != cap.shape:
            raise ValueError(f"the command has shape {z.shape}, the target's grid {cap.shape}")
        out = cap.copy()
        cur = np.minimum(z, cap)
        for _ in range(BOUND_ITERATIONS):
            if not (cur > COMMAND_TOLERANCE).any():
                break
            cmd = target.with_z(cur)
            reach = rim_pass_reach(setup, cmd, rim_pass(setup, cmd, target))
            out = np.where(zone, np.clip(reach, 0.0, cap), 0.0)
            new = np.minimum(z, out)
            if np.abs(new - cur).max() <= BOUND_CONVERGENCE:
                break
            cur = new
        return out

    return bound


def check_command(setup: FormingSetup, commanded: HeightMap,
                  reference: Optional[HeightMap]) -> None:
    """PrecompError if `commanded` rises more than `COMMAND_TOLERANCE` above
    what the setup's tools can realise (`command_upper_bound`): above the
    sheet plane anywhere without a DSIF rim pass; with one, above what the
    rim pass the deck writes for it pushes the sheet to. A command above the
    plane with a rim pass needs the target the fixture is made for."""
    raised = commanded.mask & (commanded.z > COMMAND_TOLERANCE)
    if not raised.any():
        return
    top = float(commanded.z[raised].max())
    if not (setup.support == "dsif" and setup.resolved_support()["rim_pass"]):
        raise PrecompError(
            f"the commanded surface rises above what the tools can realise at "
            f"{int(raised.sum())} nodes (up to {top * 1e3:.3f} mm above the sheet plane): the "
            f"forming tool only pushes down, and with support {setup.support!r} nothing "
            "pushes the sheet up (only a DSIF rim pass does, in a band around the target's "
            "outline; precomp.fea.support.command_upper_bound)")
    if reference is None:
        raise PrecompError(
            f"the commanded surface rises {top * 1e3:.3f} mm above the sheet plane; only the "
            "rim pass can realise that, in a band around the target's outline - pass the "
            "target to build the deck")
    ref = reference if reference.grid.matches(commanded.grid) else \
        reference.resample(commanded.grid)
    bound = command_upper_bound(setup, ref)(commanded.z)
    over = commanded.mask & (commanded.z > bound + COMMAND_TOLERANCE)
    if over.any():
        worst = float((commanded.z - bound)[over].max())
        raise PrecompError(
            f"the commanded surface rises above what the tools can realise at {int(over.sum())} "
            f"nodes (up to {worst * 1e3:.3f} mm too high): the forming tool only pushes down, "
            "and the rim pass written for this command does not push the sheet up that far "
            "(precomp.fea.support.command_upper_bound)")


__all__ = ["PLATE_TOOL", "SUPPORT_TOOL", "SUPPORT_PATH_FILE", "RIM_PASS_WINDOW",
           "COMMAND_TOLERANCE", "SupportPlan", "plan_support", "plate_nodes", "plate_tool",
           "support_tool", "dsif_trajectory", "rim_pass", "rim_pass_reach",
           "command_upper_bound", "check_command", "compensation_masks", "rim_band",
           "bottom_node_grid", "outline_distance"]
