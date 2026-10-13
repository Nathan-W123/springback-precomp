"""Tests 1 + 2: a clamped sheet formed by a rigid punch, unloaded, then
released onto statically determinate supports - in SparLab and CalculiX.

kind = strip: plane-strain strip (u_y = 0 everywhere) x in [-20, 20] mm,
       1 mm thick, clamped for |x| >= 15 mm, cylindrical punch R = 4 mm
       (axis y) pushed down `depth` at x = 0.
kind = plate: 40 x 40 x 1 mm plate clamped outside the 30 x 30 mm window
       (the benchmark's blank and clamp), spherical punch R = 4 mm pushed
       down `depth` at the centre, optionally then dragged `drag` mm along +x
       at depth (friction engaged).
Steps: form (punch in) -> unload (punch away, still clamped) -> release
(clamps off, 3-2-1 / plane 2-1 supports at (+-18, +-18, top), as the
benchmark's supports at the nearest nodes of (+-17.5, +-17.5, 0)).

Material: the benchmark's AA5754-O (J2 here; Hill48 has no built-in CalculiX
counterpart), E = 70 GPa, nu = 0.33, linear + Voce hardening.
SparLab: finite_logarithmic, penalty sphere/cylinder contact, kappa = s E/h.
CalculiX: NLGEOM (multiplicative J2), *CONTACT PAIR NODE TO SURFACE, linear
penalty of the same nominal stiffness, Coulomb friction with the same
stick stiffness; the punch a C3D8 shell whose every node is prescribed.

usage: python3 case_punch.py NAME key=value ...
  kind=strip|plate h=2 nz=2 elem=std|im|std_nobbar mu=0.1 depth=4 drag=0
  travel=0.1 (SparLab max_tool_travel, mm) ccx_inc=0.1 (ccx punch travel per
  increment, mm) only=sparlab|ccx  tp=0 (thickness points)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (E_PA, ROOT, Grid, ccx_material_cards, last_per_step, parse_dat, run_ccx,
                    run_sparlab, sparlab_material)
from punch_geometry import cylinder_ring, sphere_shell

R_TOOL = 4.0  # mm
CLAMP = 15.0  # mm, |x| or |y| beyond this is clamped
SUP = 18.0    # mm, support positions
PENALTY_S = 10.0


def parse_args(argv):
    p = dict(kind="strip", h=2.0, nz=2, elem="std", mu=0.1, depth=4.0, drag=0.0, travel=0.1,
             ccx_inc=0.1, only="", tp=0, ccx_elem="", sparlab_unload="release", ccx_unload="remove")
    name = argv[0]
    for a in argv[1:]:
        k, v = a.split("=")
        p[k] = type(p[k])(v) if not isinstance(p[k], str) else v
    return name, p


def grid_mm(p):
    h = p["h"]
    n = int(round(40.0 / h))
    if p["kind"] == "strip":
        return Grid(-20.0, 0.0, -1.0, 40.0, h, 1.0, n, 1, p["nz"])
    return Grid(-20.0, -20.0, -1.0, 40.0, 40.0, 1.0, n, n, p["nz"])


def tool_path_mm(p):
    """(times, centre points) of the punch in mm; t in [0, 1] form, the
    unload is a separate step."""
    D, R = p["depth"], R_TOOL
    pts = [[0.0, 0.0, R], [0.0, 0.0, R - D]]
    times = [0.0, 1.0]
    if p["drag"] > 0:
        pts.append([p["drag"], 0.0, R - D])
        times.append(1.0 + p["drag"] / D)
    return np.array(times), np.array(pts)


# ----------------------------------------------------------------------------
def sparlab_deck(p):
    g = grid_mm(p)
    h = p["h"] * 1e-3
    strip = p["kind"] == "strip"
    times, pts = tool_path_mm(p)
    pts_m = pts * 1e-3
    if strip:
        pts_m[:, 1] = 0.5 * h
    tool = {"name": "punch", "shape": "cylinder" if strip else "sphere", "radius": R_TOOL * 1e-3,
            "surface": {"box": {"zmin": 0.0}}, "friction": p["mu"], "penalty": PENALTY_S,
            "trajectory": {"times": times.tolist(), "points": pts_m.tolist()}}
    if strip:
        tool["axis"] = [0.0, 1.0, 0.0]
    c = CLAMP * 1e-3
    if strip:
        clamp_region = {"any_of": [{"box": {"xmax": -c}}, {"box": {"xmin": c}}]}
        clamp = [{"name": "clamp", "fix": ["x", "z"], "mode": "hold", "region": clamp_region},
                 {"name": "plane", "fix": ["y"], "mode": "hold", "region": {"all": True}}]
        s = SUP * 1e-3
        tol = 1e-5
        sup = [{"name": "plane", "fix": ["y"], "mode": "hold", "region": {"all": True}},
               {"name": "A", "fix": ["x", "z"], "mode": "hold",
                "region": {"box": {"xmin": -s - tol, "xmax": -s + tol, "zmin": -tol}}},
               {"name": "B", "fix": ["z"], "mode": "hold",
                "region": {"box": {"xmin": s - tol, "xmax": s + tol, "zmin": -tol}}}]
    else:
        clamp_region = {"any_of": [{"box": {"xmax": -c}}, {"box": {"xmin": c}},
                                   {"box": {"ymax": -c}}, {"box": {"ymin": c}}]}
        clamp = [{"name": "clamp", "fix": ["x", "y", "z"], "mode": "hold", "region": clamp_region}]
        s = SUP * 1e-3
        sup = [{"name": "A", "fix": ["x", "y", "z"], "mode": "hold", "region": {"nearest_node": [-s, -s, 0.0]}},
               {"name": "B", "fix": ["y", "z"], "mode": "hold", "region": {"nearest_node": [s, -s, 0.0]}},
               {"name": "C", "fix": ["z"], "mode": "hold", "region": {"nearest_node": [-s, s, 0.0]}}]
    steps = [{"name": "form", "type": "form", "tools": ["punch"], "time": [0.0, float(times[-1])],
              "max_tool_travel": p["travel"] * 1e-3, "boundary_conditions": clamp}]
    if p["sparlab_unload"] == "release":
        steps.append({"name": "unload", "type": "release", "tools": [], "increments": 20,
                      "boundary_conditions": clamp})
    else:  # retract the punch along +z by depth, as CalculiX does
        steps.append({"name": "unload", "type": "form", "tools": ["punch"],
                      "time": [float(times[-1]), float(times[-1]) + 1.0],
                      "max_tool_travel": p["travel"] * 1e-3, "boundary_conditions": clamp})
        tool["trajectory"]["times"].append(float(times[-1]) + 1.0)
        tool["trajectory"]["points"].append((pts_m[-1] + np.array([0, 0, p["depth"] * 1e-3])).tolist())
    steps.append({"name": "release", "type": "release", "tools": [], "increments": 20,
                  "boundary_conditions": sup})
    model = {"stress_state": "three_dimensional"}
    if p["elem"] == "im":
        model["element_formulation"] = "incompatible_modes"
    if p["tp"]:
        model["integration"] = {"thickness_points": p["tp"]}
    forming = {"kinematics": "finite_logarithmic", "tools": [tool], "steps": steps,
               "output": {"vtk": False, "snapshots": "steps"}}
    if p["elem"] == "std_nobbar":
        forming["mean_dilatation"] = "none"
    return {"name": "punch", "units": "SI",
            "mesh": {"type": "structured_hex", "x0": g.xyz[:, 0].min() * 1e-3, "y0": g.xyz[:, 1].min() * 1e-3,
                     "z0": -1e-3, "lx": 0.04, "ly": (g.xyz[:, 1].max() - g.xyz[:, 1].min()) * 1e-3,
                     "lz": 1e-3, "nx": g.nx, "ny": g.ny, "nz": g.nz},
            "model": model, "material": sparlab_material(), "forming": forming}


# ----------------------------------------------------------------------------
def ccx_deck(p):
    g = grid_mm(p)
    strip = p["kind"] == "strip"
    times, pts = tool_path_mm(p)
    h = p["h"]
    if strip:
        nodes_p, conn_p, faces_p = cylinder_ring((0.0, R_TOOL), R_TOOL, -1.0, h + 1.0)
    else:
        nodes_p, conn_p, faces_p = sphere_shell(np.array([0.0, 0.0, R_TOOL]), R_TOOL, n=32)
    nsheet = len(g.xyz)
    L = ["*HEADING", "punch cross-check", "*NODE, NSET=NSHEET"]
    for i, q in enumerate(g.xyz):
        L.append(f"{i+1}, {q[0]:.12g}, {q[1]:.12g}, {q[2]:.12g}")
    L.append("*NODE, NSET=PUNCH")
    for i, q in enumerate(nodes_p):
        L.append(f"{nsheet+i+1}, {q[0]:.12g}, {q[1]:.12g}, {q[2]:.12g}")
    etype = p["ccx_elem"] or {"std": "C3D8", "std_nobbar": "C3D8", "im": "C3D8I"}[p["elem"]]
    L.append(f"*ELEMENT, TYPE={etype}, ELSET=SHEET")
    for e, c in enumerate(g.conn):
        L.append(f"{e+1}, " + ", ".join(str(n + 1) for n in c))
    ne = len(g.conn)
    L.append("*ELEMENT, TYPE=C3D8, ELSET=TOOL")
    for e, c in enumerate(conn_p):
        L.append(f"{ne+e+1}, " + ", ".join(str(nsheet + n + 1) for n in c))
    x = g.xyz

    def nset(name, mask):
        ids = [str(i + 1) for i in np.nonzero(mask)[0]]
        L.append(f"*NSET, NSET={name}")
        for k in range(0, len(ids), 12):
            L.append(", ".join(ids[k:k + 12]))

    tol = 1e-6
    if strip:
        nset("CLAMP", np.abs(x[:, 0]) >= CLAMP - tol)
        nset("SUPA", (np.abs(x[:, 0] + SUP) < tol) & (x[:, 2] > -tol))
        nset("SUPB", (np.abs(x[:, 0] - SUP) < tol) & (x[:, 2] > -tol))
    else:
        nset("CLAMP", (np.abs(x[:, 0]) >= CLAMP - tol) | (np.abs(x[:, 1]) >= CLAMP - tol))
        def nearest(q):
            return np.argmin(np.linalg.norm(x - np.array(q), axis=1))
        for nm, q in (("SUPA", (-SUP, -SUP, 0)), ("SUPB", (SUP, -SUP, 0)), ("SUPC", (-SUP, SUP, 0))):
            m = np.zeros(len(x), bool); m[nearest(q)] = True
            nset(nm, m)
    nset("NTOP", x[:, 2] > -tol)
    # slave: top faces of the top layer (CalculiX face 2 = nodes 5-8-7-6)
    L.append("*ELSET, ELSET=ETOPL")
    top = [e + 1 for e, c in enumerate(g.conn) if np.all(x[c[4:], 2] > -tol)]
    for k in range(0, len(top), 12):
        L.append(", ".join(str(v) for v in top[k:k + 12]))
    L.append("*SURFACE, NAME=SSLAVE, TYPE=ELEMENT")
    L.append("ETOPL, S2")
    L.append("*SURFACE, NAME=SMASTER, TYPE=ELEMENT")
    for e, f in faces_p:
        L.append(f"{ne+e+1}, S{f}")
    L.append(ccx_material_cards("AL"))
    L.append("*MATERIAL, NAME=STEEL\n*ELASTIC\n210000., 0.3")
    L.append("*SOLID SECTION, ELSET=SHEET, MATERIAL=AL")
    L.append("*SOLID SECTION, ELSET=TOOL, MATERIAL=STEEL")
    kappa = PENALTY_S * E_PA / 1e6 / h  # MPa/mm, SparLab's kappa for an interior node
    L += ["*SURFACE INTERACTION, NAME=SI", "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
          f"{kappa:.6g}, 0.01"]
    if p["mu"] > 0:
        L += ["*FRICTION", f"{p['mu']}, {kappa:.6g}"]
    L += ["*CONTACT PAIR, INTERACTION=SI, TYPE=NODE TO SURFACE", "SSLAVE, SMASTER"]

    def bc_common():
        out = ["*BOUNDARY"]
        if strip:
            out += ["NSHEET, 2, 2, 0.0"]
        return out

    def punch_at(c):
        d = c - pts[0]
        return ["*BOUNDARY", f"PUNCH, 1, 1, {d[0]:.12g}", f"PUNCH, 2, 2, {d[1]:.12g}",
                f"PUNCH, 3, 3, {d[2]:.12g}"]

    def clamp_bc():
        return ["*BOUNDARY", "CLAMP, 1, 1, 0.0", "CLAMP, 3, 3, 0.0"] + ([] if strip else ["CLAMP, 2, 2, 0.0"])

    def outputs():
        # CalculiX's print FREQUENCY is one setting for all print requests,
        # so the top-surface displacements are printed every increment too.
        return ["*NODE PRINT, NSET=PUNCH, TOTALS=ONLY", "RF",
                "*NODE PRINT, NSET=NTOP", "U",
                "*EL PRINT, ELSET=SHEET", "PEEQ"]

    inc = p["ccx_inc"]
    # step 1: punch in (and drag)
    seg = [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
    for si, (a, b) in enumerate(seg):
        dist = np.linalg.norm(b - a)
        f = min(1.0, inc / dist)
        L += ["*STEP, NLGEOM, INC=100000", "*STATIC", f"{f:.6g}, 1.0, {f*1e-4:.6g}, {f:.6g}"]
        L += bc_common() + clamp_bc() + punch_at(b) + outputs()
        L += ["*END STEP"]
    if p["ccx_unload"] == "retract":
        # unload: punch straight up by depth + 1 mm (the strip runs of the
        # first auditor; diverges at the first loss of contact in the plate
        # runs and with C3D20R slaves)
        up = pts[-1] + np.array([0, 0, p["depth"] + 1.0])
        f = min(1.0, inc / (p["depth"] + 1.0))
        L += ["*STEP, NLGEOM, INC=100000", "*STATIC", f"{f:.6g}, 1.0, {f*1e-4:.6g}, {f:.6g}"]
        L += bc_common() + clamp_bc() + punch_at(up) + outputs() + ["*END STEP"]
    else:
        # unload: the contact pair is removed (SparLab's 'release' unload
        # step removes the tool the same way); the punch stays where it is
        up = pts[-1]
        L += ["*STEP, NLGEOM, INC=100000", "*STATIC", "0.05, 1.0, 1e-7, 0.05",
              "*MODEL CHANGE, TYPE=CONTACT PAIR, REMOVE", "SSLAVE, SMASTER"]
        L += bc_common() + clamp_bc() + punch_at(up) + outputs() + ["*END STEP"]
    # release: clamp off, determinate supports
    L += ["*STEP, NLGEOM, INC=100000", "*STATIC", "0.05, 1.0, 1e-6, 0.05"]
    L += ["*BOUNDARY, OP=NEW"] + (["NSHEET, 2, 2, 0.0"] if strip else [])
    L += punch_at(up)[1:]
    if strip:
        L += ["SUPA, 1, 1, 0.0", "SUPA, 3, 3, 0.0", "SUPB, 3, 3, 0.0"]
    else:
        L += ["SUPA, 1, 3, 0.0", "SUPB, 2, 3, 0.0", "SUPC, 3, 3, 0.0"]
    L += outputs() + ["*END STEP"]
    return "\n".join(L) + "\n", len(seg)


# ----------------------------------------------------------------------------
def collect(name, p, case):
    """Comparison tables of a case whose runs both finished."""
    sdir, cdir = case / "sparlab", case / "ccx"
    res = {"name": name, "params": p}
    strip = p["kind"] == "strip"
    W = p["h"] if strip else 1.0
    # SparLab force history (N on the tool; per mm of width for the strip)
    tf = list(csv.DictReader(open(sdir / "out" / "tool_forces.csv")))
    s_force = [(float(r["t"]), -float(r["cz"]) * 1e3 + R_TOOL, float(r["fz"]) / W, float(r["fx"]) / W)
               for r in tf if r["step"] in ("1", "form")]
    # displacement of the punch below the initial touch = R - cz(mm)
    s_force = [(t, R_TOOL - (-d + R_TOOL), fz, fx) for t, d, fz, fx in s_force]
    blocks = parse_dat(cdir / "job.dat")
    nseg = p["_nseg"]
    c_force = []
    _, pts = tool_path_mm(p)
    for b in blocks:
        if b[0].startswith("total force") and b[4] <= nseg:
            a, bb = pts[b[4] - 1], pts[b[4]]
            c = a + (b[2] - (b[4] - 1)) * (bb - a)  # printed time is the total time
            rf = np.array(b[3][0])
            c_force.append((b[4] - 1 + b[2], c, -rf[2] / W, -rf[0] / W))
    # SparLab: the trajectory's pseudo-time t; travel along the path
    times, _ = tool_path_mm(p)
    # depth at each record
    s_tab = []
    for r in tf:
        if r["step"] not in ("1",):
            continue
        cz = float(r["cz"]) * 1e3
        cx = float(r["cx"]) * 1e3
        s_tab.append((R_TOOL - cz, cx, float(r["fz"]) / W, float(r["fx"]) / W))
    c_tab = [(R_TOOL - c[2], c[0], fz, fx) for _, c, fz, fx in c_force]
    with open(case / "force.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "depth_mm", "x_mm", "fz", "fx"])
        for r in s_tab:
            w.writerow(["sparlab"] + [f"{v:.6g}" for v in r])
        for r in c_tab:
            w.writerow(["ccx"] + [f"{v:.6g}" for v in r])
    # peak force comparison (along the plunge: interpolate SparLab at ccx depths)
    s_arr = np.array([r for r in s_tab if abs(r[1]) < 1e-9 or True])
    c_arr = np.array(c_tab)
    plunge_s = s_arr[s_arr[:, 1] < 1e-6]
    plunge_c = c_arr[c_arr[:, 1] < 1e-6]
    res["F_end_plunge_sparlab"] = float(plunge_s[np.argmax(plunge_s[:, 0]), 2])
    res["F_end_plunge_ccx"] = float(plunge_c[np.argmax(plunge_c[:, 0]), 2])
    dd = np.linspace(0.5, p["depth"], 8)
    fs = np.interp(dd, plunge_s[:, 0], plunge_s[:, 2])
    fc = np.interp(dd, plunge_c[:, 0], plunge_c[:, 2])
    res["F_profile_depths"] = dd.tolist()
    res["F_sparlab"] = fs.tolist()
    res["F_ccx"] = fc.tolist()
    res["F_rel_max"] = float(np.max(np.abs(fs - fc) / np.abs(fc)))
    if p["drag"] > 0:
        drag_s = s_arr[s_arr[:, 1] > 1e-6]
        drag_c = c_arr[c_arr[:, 1] > 1e-6]
        res["drag_fx_mean_sparlab"] = float(np.mean(drag_s[drag_s[:, 1] > p["drag"] / 2, 3]))
        res["drag_fx_mean_ccx"] = float(np.mean(drag_c[drag_c[:, 1] > p["drag"] / 2, 3]))
        res["drag_fz_mean_sparlab"] = float(np.mean(drag_s[drag_s[:, 1] > p["drag"] / 2, 2]))
        res["drag_fz_mean_ccx"] = float(np.mean(drag_c[drag_c[:, 1] > p["drag"] / 2, 2]))
    # shapes: top-surface nodes, z displacement, at the end of each step
    g = grid_mm(p)
    top = np.nonzero(g.xyz[:, 2] > -1e-6)[0]
    ulast = last_per_step(blocks, "displacements")
    nsteps_c = nseg + 2
    sname = {"form": 1, "unload": 2, "release": 3}
    cstep = {"form": nseg, "unload": nseg + 1, "release": nseg + 2}
    shapes = {}
    for st in ("form", "unload", "release"):
        sn = np.loadtxt(sdir / "out" / f"step_{sname[st]}_{st}_nodes.csv", delimiter=",", skiprows=1)
        sx = sn[:, 1:4] * 1e3
        su = sn[:, 4:7] * 1e3
        cb = np.array(ulast[cstep[st]][3])
        cu = {int(r[0]): r[1:4] for r in cb}
        # match SparLab nodes to grid nodes by coordinates
        key = {tuple(np.round(q, 6)): i for i, q in enumerate(sx)}
        rows = []
        for i in top:
            j = key[tuple(np.round(g.xyz[i], 6))]
            rows.append([g.xyz[i, 0], g.xyz[i, 1], su[j, 0], su[j, 1], su[j, 2], *cu[i + 1]])
        rows = np.array(rows)
        shapes[st] = rows
        np.savetxt(case / f"shape_{st}.csv", rows, delimiter=",",
                   header="X_mm,Y_mm,ux_sparlab,uy_sparlab,uz_sparlab,ux_ccx,uy_ccx,uz_ccx", comments="")
        dz = rows[:, 4] - rows[:, 7]
        d3 = np.linalg.norm(rows[:, 2:5] - rows[:, 5:8], axis=1)
        res[f"{st}_uz_min_sparlab"] = float(rows[:, 4].min())
        res[f"{st}_uz_min_ccx"] = float(rows[:, 7].min())
        res[f"{st}_dz_rms"] = float(np.sqrt(np.mean(dz ** 2)))
        res[f"{st}_dz_max"] = float(np.max(np.abs(dz)))
        res[f"{st}_d3_max"] = float(d3.max())
        res[f"{st}_uz_range_sparlab"] = float(np.ptp(rows[:, 4]))
        res[f"{st}_uz_range_ccx"] = float(np.ptp(rows[:, 7]))
    # springback between the formed (loaded) and released shapes
    for code, col in (("sparlab", 4), ("ccx", 7)):
        res[f"springback_unload_max_{code}"] = float(np.max(np.abs(shapes["unload"][:, col] - shapes["form"][:, col])))
    # equivalent plastic strain
    el = np.loadtxt(sdir / "out" / "step_1_form_elements.csv", delimiter=",", skiprows=1)
    res["peeq_max_sparlab"] = float(el[:, 1].max())
    pe = last_per_step(blocks, "equivalent plastic")[nseg]
    res["peeq_max_ccx"] = float(np.max(np.array(pe[3])[:, 2]))
    ss = json.load(open(sdir / "out" / "summary.json"))
    res["sparlab_runtime_s"] = ss["runtime_s"]
    res["sparlab_increments"] = [st["increments"] for st in ss["steps"]]
    res["sparlab_iterations"] = [st["iterations"] for st in ss["steps"]]
    res["sparlab_max_penetration_mm"] = ss["tools"][0].get("max_penetration_m", float("nan")) * 1e3 \
        if isinstance(ss["tools"][0].get("max_penetration_m"), (int, float)) else None
    return res


def main():
    name, p = parse_args(sys.argv[1:])
    case = ROOT / "punch" / name
    case.mkdir(parents=True, exist_ok=True)
    deck = sparlab_deck(p)
    inp, nseg = ccx_deck(p)
    p["_nseg"] = nseg
    (case / "params.json").write_text(json.dumps(p, indent=1))
    if p["only"] != "ccx" and not (case / "sparlab" / "out" / "summary.json").exists():
        rc, wall = run_sparlab(deck, case / "sparlab")
        print(name, "sparlab rc", rc, f"{wall:.1f} s", flush=True)
    if p["only"] != "sparlab" and not (case / "ccx" / "job.dat").exists() or \
            (p["only"] != "sparlab" and "Job finished" not in (case / "ccx" / "ccx.log").read_text(errors="replace")):
        rc, wall = run_ccx(inp, case / "ccx")
        (case / "ccx" / "wall_s.txt").write_text(f"{wall:.1f}\n")
        print(name, "ccx rc", rc, f"{wall:.1f} s", flush=True)
    if not p["only"]:
        res = collect(name, p, case)
        (case / "result.json").write_text(json.dumps(res, indent=1))
        for k, v in res.items():
            if k != "params":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
