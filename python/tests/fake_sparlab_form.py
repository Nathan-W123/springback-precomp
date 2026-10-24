"""A test double of sparlab_form that writes its output contract exactly.

It is *not* a solver. It reads the deck as the real executable does - with
`--strict-config` it refuses a key sparlab_form does not read (exit 2), and
it refuses the values the real parser refuses for the keys precomp writes -
and writes exactly the files, CSV columns and summary keys of
docs/forming.md, section 3, in the same order: config.json and mesh.json
first, per completed step `step_<k>_<s>_nodes.csv`, `_elements.csv` and
`.vtk` (k from 1) and tool_forces.csv (a row per increment of a step the tool
is active in: the force of the sheet ON THE TOOL, fz > 0), summary.json
last. `test_integration_sparlab.py` compares it with the real executable,
file by file, column by column and key by key.

The numbers are made up. The deck's mesh must be `structured_hex`, and - a
test-only shortcut - the double reads the provenance file `commanded.npz`
that `precomp.fea.build_deck` writes beside the deck. Every node of a
through-thickness column moves vertically by the commanded height at its
(X, Y), times a springback factor S after a "form" step:

    form step:           u_z = z_cmd(X, Y)
    release steps:       u_z = S z_cmd(X, Y),   S = $FAKE_SPARLAB_SPRINGBACK (0.9)

so the formed tool-side surface is exactly S times the commanded one at the
nodes, a linear "springback" whose displacement-adjustment inverse is known.
Environment switches for the tests: FAKE_SPARLAB_COUNTER (append one line per
run to this file), FAKE_SPARLAB_FAIL (the first step stops: exit 3 with the
reason in summary.json and on stderr, as the real executable does).
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

VERSION = "sparlab 0.0.0-fake (test double)"

# The keys sparlab_form reads (src/io/Config.cpp) in the blocks precomp
# writes. A dict is the schema of an object, a one-item list that of every
# item of an array, None a leaf (or a block the double does not look into).
_BOX = {k: None for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
_REGION = {"name": None, "invert": None, "tolerance": None, "all": None, "box": _BOX,
           "circle": None, "annulus": None, "sphere": None, "node_ids": None,
           "element_ids": None, "nearest_node": None, "group": None}
_REGION["any_of"] = [_REGION]
SCHEMA = {
    "name": None, "description": None, "units": None,
    "mesh": {k: None for k in ("type", "nx", "ny", "nz", "lx", "ly", "lz", "x0", "y0", "z0",
                               "path", "format", "scale", "merge_duplicate_nodes",
                               "duplicate_tolerance")},
    "material": {
        "name": None, "youngs_modulus": None, "poisson_ratio": None, "density": None,
        "thermal_expansion": None, "reference_temperature": None, "conductivity": None,
        "plasticity": {
            "yield_stress": None, "hardening_modulus": None, "saturation_stress": None,
            "saturation_rate": None, "kinematic_hardening_modulus": None,
            "yield_criterion": None, "kinematic_integration": None,
            "anisotropy": {k: None for k in ("r0", "r45", "r90", "stress_ratios",
                                             "coefficients", "out_of_plane_shear",
                                             "rolling_direction", "sheet_normal",
                                             "rolling_angle")},
            "backstresses": [{"modulus": None, "recovery": None}]}},
    "material_regions": None,
    "model": {"thickness": None, "stress_state": None, "integration": None},
    "boundary_conditions": [{"name": None, "fix": None, "value": None, "region": _REGION}],
    "forming": {
        "kinematics": None, "mean_dilatation": None,
        "friction_tangent": None, "solver": None,
        "tools": [{"name": None, "shape": None, "radius": None, "normal": None, "axis": None,
                   "surface": _REGION, "friction": None, "penalty": None,
                   "tangential_penalty": None,
                   "trajectory": {"file": None, "times": None, "points": None}}],
        "steps": [{"name": None, "type": None, "tools": None, "time": None,
                   "max_tool_travel": None, "increments": None,
                   "boundary_conditions": [{"name": None, "fix": None, "value": None,
                                            "region": _REGION, "mode": None}]}],
        "newton": {k: None for k in ("max_iterations", "residual_tolerance",
                                     "displacement_tolerance", "line_search", "max_cuts",
                                     "max_increments")},
        "output": {"vtk": None, "snapshots": None}},
}


class ConfigError(Exception):
    pass


def unknown_keys(doc, schema, path=""):
    """Dotted paths of the keys of `doc` that `schema` does not know."""
    out = []
    if isinstance(schema, dict) and isinstance(doc, dict):
        for key, value in doc.items():
            p = f"{path}.{key}" if path else key
            out += [p] if key not in schema else unknown_keys(value, schema[key], p)
    elif isinstance(schema, list) and isinstance(doc, list):
        for i, item in enumerate(doc):
            out += unknown_keys(item, schema[0], f"{path}[{i}]")
    return out


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_values(deck):
    """The refusals of the real parser, for the values precomp writes."""
    fm = deck.get("forming")
    if not isinstance(fm, dict):
        raise ConfigError("the deck has no 'forming' block; sparlab_form runs the "
                          "incremental-forming analysis it describes")
    kinematics = fm.get("kinematics", "finite")
    if kinematics not in ("finite", "finite_logarithmic", "small_strain"):
        raise ConfigError(f"unknown kinematics '{kinematics}'; expected \"finite\", "
                          "\"finite_logarithmic\" or \"small_strain\"")
    plastic = deck.get("material", {}).get("plasticity", {})
    if ("anisotropy" in plastic) != (plastic.get("yield_criterion") == "hill48"):
        raise ConfigError("'material.plasticity': \"yield_criterion\": \"hill48\" and the "
                          "\"anisotropy\" block go together")
    for i, b in enumerate(plastic.get("backstresses", [])):
        if not (_number(b.get("modulus")) and b["modulus"] > 0):
            raise ConfigError(f"'material.plasticity.backstresses[{i}].modulus' must be > 0")
    names = []
    for i, tool in enumerate(fm.get("tools", [])):
        names.append(tool.get("name", f"tool{i}"))
        for key in ("radius", "friction", "penalty", "tangential_penalty"):
            if key in tool and not _number(tool[key]):
                raise ConfigError(f"'forming.tools[{i}].{key}' must be a number, got "
                                  f"{type(tool[key]).__name__}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", names[-1]):
            raise ConfigError(f"tool name '{names[-1]}' must be non-empty and use only "
                              "letters, digits, '_', '-' and '.'")
    steps = fm.get("steps") or []
    if not steps:
        raise ConfigError("'forming.steps' is missing or empty")
    for i, st in enumerate(steps):
        label = f"'forming.steps[{i}]'"
        if st.get("type", "form") not in ("form", "release"):
            raise ConfigError(f"{label}: unknown step type '{st['type']}'")
        if "increments" in st and not (isinstance(st["increments"], int)
                                       and not isinstance(st["increments"], bool)):
            raise ConfigError(f"'forming.steps[{i}].increments' must be a number, got "
                              f"{type(st['increments']).__name__}")
        if "max_tool_travel" in st and not _number(st["max_tool_travel"]):
            raise ConfigError(f"'forming.steps[{i}].max_tool_travel' must be a number, got "
                              f"{type(st['max_tool_travel']).__name__}")
        for t in st.get("tools", []):
            if t not in names:
                raise ConfigError(f"{label} names tool '{t}', which 'forming.tools' does "
                                  "not define")
        for j, bc in enumerate(st.get("boundary_conditions", [])):
            mode = bc.get("mode", "hold")
            if mode not in ("hold", "absolute"):
                raise ConfigError(f"{label}.boundary_conditions[{j}]: unknown mode '{mode}'")
            if mode == "hold" and any(v != 0 for v in bc.get("value", [])):
                raise ConfigError(f"'forming.steps[{i}].boundary_conditions[{j}].value' is not "
                                  "zero, but the constraint's mode is \"hold\"")


def structured_mesh(m):
    """Nodes (x fastest, then y, then z) and Hex8 connectivity of a
    structured_hex mesh, as SparLab numbers them."""
    if m.get("type") != "structured_hex":
        raise ConfigError("the test double reads structured_hex meshes only")
    nx, ny, nz = m["nx"], m["ny"], m["nz"]
    xs = m.get("x0", 0.0) + m["lx"] * np.arange(nx + 1) / nx
    ys = m.get("y0", 0.0) + m["ly"] * np.arange(ny + 1) / ny
    zs = m.get("z0", 0.0) + m["lz"] * np.arange(nz + 1) / nz
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def n(i, j, k):
        return (k * (ny + 1) + j) * (nx + 1) + i

    elements = [[n(i, j, k), n(i + 1, j, k), n(i + 1, j + 1, k), n(i, j + 1, k),
                 n(i, j, k + 1), n(i + 1, j, k + 1), n(i + 1, j + 1, k + 1), n(i, j + 1, k + 1)]
                for k in range(nz) for j in range(ny) for i in range(nx)]
    return nodes, np.array(elements, dtype=int)


def files_stem(k, name):
    """step_<k>_<name> (k from 0 here, from 1 in the name), sanitised as SparLab does."""
    keep = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    clean = "".join(chr(b) if b in keep else "_" for b in name.encode("utf-8"))
    return f"step_{k + 1}_{clean or 'step'}"


def write_csv(path, header, rows, fmt):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(f % v for f, v in zip(fmt, row)) + "\n")


def write_vtk(path, nodes, elements, u, plastic, von_mises):
    lines = ["# vtk DataFile Version 3.0", "SparLab forming", "ASCII",
             "DATASET UNSTRUCTURED_GRID", f"POINTS {len(nodes)} double"]
    lines += [f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in nodes]
    lines.append(f"CELLS {len(elements)} {len(elements) * 9}")
    lines += ["8 " + " ".join(str(i) for i in e) for e in elements]
    lines.append(f"CELL_TYPES {len(elements)}")
    lines += ["12"] * len(elements)
    lines += [f"POINT_DATA {len(nodes)}", "SCALARS displacement_magnitude double 1",
              "LOOKUP_TABLE default"]
    lines += [f"{v:.9g}" for v in np.linalg.norm(u, axis=1)]
    lines.append("VECTORS displacement double")
    lines += [f"{a:.9g} {b:.9g} {c:.9g}" for a, b, c in u]
    lines.append(f"CELL_DATA {len(elements)}")
    for name, values in (("eq_plastic_strain", plastic), ("von_mises", von_mises)):
        lines += [f"SCALARS {name} double 1", "LOOKUP_TABLE default"]
        lines += [f"{v:.9g}" for v in values]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def load_trajectory(deck_dir, tool):
    traj = tool["trajectory"]
    if "file" in traj:
        data = np.loadtxt(os.path.join(deck_dir, traj["file"]), delimiter=",", comments="#",
                          skiprows=1, ndmin=2)
        return data[:, 0], data[:, 1:4]
    return np.asarray(traj["times"], float), np.asarray(traj["points"], float)


def run(args, deck, deck_dir, out):
    start = time.perf_counter()
    fm = deck["forming"]
    newton = fm.get("newton", {})
    nodes, elements = structured_mesh(deck["mesh"])
    X, Y = nodes[:, 0], nodes[:, 1]
    with np.load(os.path.join(deck_dir, "commanded.npz")) as data:
        grid = json.loads(str(data["grid"]))
        zc = data["z"]
    gx = grid["x0"] + grid["h"] * np.arange(grid["nx"])
    gy = grid["y0"] + grid["h"] * np.arange(grid["ny"])
    interp = RegularGridInterpolator((gy, gx), zc, bounds_error=False, fill_value=0.0)
    zcmd = interp(np.column_stack([Y, X]))
    springback = float(os.environ.get("FAKE_SPARLAB_SPRINGBACK", "0.9"))
    fail = bool(os.environ.get("FAKE_SPARLAB_FAIL"))
    output = fm.get("output", {})
    vtk = output.get("vtk", True) and not args.no_vtk
    step_files = output.get("snapshots", "steps") != "none"
    tools = fm.get("tools", [])
    names = [t.get("name", f"tool{i}") for i, t in enumerate(tools)]
    trajectories = [load_trajectory(deck_dir, t) for t in tools]
    name = deck.get("name", "case")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(deck, handle, indent=2)
    with open(os.path.join(out, "mesh.json"), "w", encoding="utf-8") as handle:
        json.dump({"case": name, "element_type": "Hex8", "dim": 3, "nodes_per_element": 8,
                   "nodes_m": nodes.tolist(), "elements": elements.tolist(),
                   "dofs_per_node": 3, "prescribed_dofs": [],
                   "load_cases": [{"name": "forming", "weight": 1, "nodal_forces": []}]},
                  handle)

    files = ["summary.json", "config.json", "mesh.json"]
    rows = []                                 # tool_forces.csv
    steps = []
    t_end = None
    termination = "completed every step"
    for k, st in enumerate(fm["steps"]):
        sname = st.get("name", f"step{k + 1}")
        kind = st.get("type", "form")
        active = [names.index(t) for t in st.get("tools", [])]
        if "time" in st:
            t0, t1 = st["time"]
        elif kind == "form" and active:
            t0 = min(trajectories[i][0][0] for i in active)
            t0 = t0 if t_end is None else max(t0, t_end)
            t1 = max(trajectories[i][0][-1] for i in active)
        else:
            t0 = 0.0 if t_end is None else t_end
            t1 = t0 + 1.0
        t_end = t1
        completed = not (fail and k == 0)
        increments = st.get("increments", 10 if kind == "release" else 1)
        if completed and kind == "form" and active:
            ts = np.linspace(t0, t1, 6)[1:]
            increments = len(ts)
            for inc, t in enumerate(ts, 1):
                for i in active:
                    times, points = trajectories[i]
                    c = [float(np.interp(t, times, points[:, a])) for a in range(3)]
                    touching = c[2] < tools[i].get("radius", 0.0)
                    fz = 200.0 + 1.0e5 * max(0.0, -c[2]) if touching else 0.0
                    rows.append((k + 1, inc, t, names[i], c[0], c[1], c[2], 0.0, 0.0, fz,
                                 3 if touching else 0, 1e-6 if touching else 0.0))
        elif completed and active:            # a release that keeps tools
            for inc in range(1, increments + 1):
                t = t0 + (t1 - t0) * inc / increments
                for i in active:
                    rows.append((k + 1, inc, t, names[i], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0))
        if not completed:
            increments = 0
        u = np.column_stack([np.zeros_like(X), np.zeros_like(X),
                             (1.0 if kind == "form" else springback) * zcmd])
        plastic = np.full(len(elements), 0.1)
        von_mises = np.full(len(elements), 1.0e8 if kind == "form" else 1.0e7)
        stem = files_stem(k, sname)
        if completed and step_files:
            write_csv(os.path.join(out, stem + "_nodes.csv"),
                      ["node", "X", "Y", "Z", "ux", "uy", "uz"],
                      [(n,) + tuple(nodes[n]) + tuple(u[n]) for n in range(len(nodes))],
                      ["%d"] + ["%.17g"] * 6)
            write_csv(os.path.join(out, stem + "_elements.csv"),
                      ["element", "eq_plastic_strain", "von_mises_Pa"],
                      [(e, plastic[e], von_mises[e]) for e in range(len(elements))],
                      ["%d", "%.17g", "%.17g"])
            files += [stem + "_nodes.csv", stem + "_elements.csv"]
            if vtk:
                write_vtk(os.path.join(out, stem + ".vtk"), nodes, elements, u, plastic,
                          von_mises)
                files.append(stem + ".vtk")
        write_csv(os.path.join(out, "tool_forces.csv"),
                  ["step", "increment", "t", "tool", "cx", "cy", "cz", "fx", "fy", "fz",
                   "active_nodes", "max_penetration_m"], rows,
                  ["%d", "%d", "%.17g", "%s"] + ["%.17g"] * 6 + ["%d", "%.17g"])
        reason = ("released" if kind == "release" else "reached the end of its window")
        if not completed:
            reason = "failure requested by FAKE_SPARLAB_FAIL"
        steps.append({
            "name": sname, "type": kind, "completed": completed, "increments": increments,
            "iterations": increments, "cuts": 0, "max_plastic_strain": 0.1,
            "reaction_norm_N": 0.0, "warnings": [], "termination": reason,
            "files_stem": stem, "t_begin_s": t0, "t_end_s": t1,
            "tools": [names[i] for i in active], "constrained_dofs": 6,
            "start_imbalance_N": 0.0, "reference_force_N": 1.0,
            "max_displacement_change_m": float(np.abs(u[:, 2]).max()),
            "max_displacement_m": float(np.abs(u[:, 2]).max())})
        if not completed:
            termination = f"step '{sname}' ({k + 1} of {len(fm['steps'])}): {reason}"
            break
    completed = termination == "completed every step"
    files.append("tool_forces.csv")

    tool_docs = []
    for i, tool in enumerate(tools):
        times, points = trajectories[i]
        mine = [r for r in rows if r[3] == names[i]]
        peak = max(mine, key=lambda r: abs(r[9]), default=None)
        doc = {"name": names[i], "shape": tool.get("shape", "sphere")}
        if doc["shape"] != "plane":
            doc["radius_m"] = tool["radius"]
        else:
            doc["normal"] = tool["normal"]
        if doc["shape"] == "cylinder":
            doc["axis"] = tool.get("axis", [0.0, 0.0, 1.0])
        doc.update({
            "friction": tool.get("friction", 0.0), "penalty": tool.get("penalty", 10.0),
            "tangential_penalty": tool.get("tangential_penalty", 1.0),
            "trajectory_knots": len(times), "trajectory_t_start_s": float(times[0]),
            "trajectory_t_end_s": float(times[-1]),
            "trajectory_length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            "peak_force_N": [0.0, 0.0, peak[9] if peak else 0.0],
            "peak_force_time_s": peak[2] if peak else 0.0,
            "max_active_nodes": max((r[10] for r in mine), default=0),
            "max_penetration_m": max((r[11] for r in mine), default=0.0)})
        tool_docs.append(doc)

    runtime = time.perf_counter() - start
    increments = sum(s["increments"] for s in steps)
    m = deck["mesh"]
    summary = {
        "case": name, "sparlab_version": VERSION[len("sparlab "):], "completed": completed,
        "termination": termination, "runtime_s": runtime,
        "timing": {"contact_s": 0.0, "element_residual_s": 0.0, "element_tangent_s": 0.0,
                   "factor_cholmod_llt_s": 0.0, "factorisation_s": 0.0, "output_s": 0.0,
                   "solve_s": 0.0, "total_s": runtime, "increments": increments,
                   "iterations": increments, "cuts": 0,
                   "linear_solver": f"CHOLMOD supernodal LLT x {increments}",
                   "failed_factorisations": "", "suitesparse": True},
        "analysis": {"kinematics": fm.get("kinematics", "finite"), "plastic": True,
                     "mean_dilatation": True,
                     "friction_tangent": fm.get("friction_tangent", "exact"),
                     "residual_tolerance": newton.get("residual_tolerance", 1e-6),
                     "displacement_tolerance": newton.get("displacement_tolerance", 1e-6),
                     "max_iterations": newton.get("max_iterations", 30),
                     "max_cuts": newton.get("max_cuts", 10),
                     "line_search": newton.get("line_search", True)},
        "steps": steps, "tools": tool_docs,
        "mesh": {"source": f"structured_hex {m['nx']} x {m['ny']} x {m['nz']}",
                 "element_type": "Hex8", "element_formulation": "standard",
                 "stiffness_rule_points": 8, "dim": 3, "nodes": len(nodes),
                 "elements": len(elements), "dofs": 3 * len(nodes),
                 "bounding_box_min_m": nodes.min(axis=0).tolist(),
                 "bounding_box_max_m": nodes.max(axis=0).tolist()},
        "warnings": [], "files": files,
        "provenance": {"code": "SparLab", "version": "0.0.0-fake", "build_type": "fake",
                       "compiler": "none", "eigen_version": "none",
                       "config_source": args.config, "case": name,
                       "description": deck.get("description", ""),
                       "units": "SI: m, N, Pa, kg, kg/m^3, Hz, J",
                       "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"case: {name}")
    if not completed:
        print(f"[error] the forming analysis stopped: {termination}", file=sys.stderr)
        return 3
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--version" in argv:
        print(VERSION)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--verbosity")
    ap.add_argument("--strict-config", action="store_true")
    ap.add_argument("--no-vtk", action="store_true")
    args = ap.parse_args(argv)
    counter = os.environ.get("FAKE_SPARLAB_COUNTER")
    if counter:
        with open(counter, "a", encoding="utf-8") as handle:
            handle.write(args.config + "\n")
    try:
        with open(args.config, encoding="utf-8") as handle:
            deck = json.load(handle)
        unknown = unknown_keys(deck, SCHEMA)
        if unknown:
            message = (f"{args.config} contains {len(unknown)} key(s) that SparLab does not "
                       "recognise (a misspelled key silently takes its default, so this is "
                       "reported):" + "".join(f"\n  - {k}" for k in unknown))
            if args.strict_config:
                raise ConfigError(message)
            print(f"[warn ] {message}", file=sys.stderr)
        check_values(deck)
        return run(args, deck, os.path.dirname(os.path.abspath(args.config)), args.output)
    except ConfigError as exc:
        print(f"[error] configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
