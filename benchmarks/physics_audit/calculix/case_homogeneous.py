"""Test 3: homogeneous large-strain J2 plasticity, one Hex8 / C3D8.

Paths (each a chain of steps, the state at every step end compared):
  uniaxial: x-stretch to log strain 0.5, then back to 0 (reversed into
            compression), lateral faces free (symmetry planes);
  shear:    simple shear u_x = gamma z with u_y = u_z = 0 on every node, gamma
            to 1.0 (the axes of stretch rotate 0 -> 31.7 deg);
SparLab: sparlab_form, finite_logarithmic (Hencky, additive in log strain,
Kirchhoff stress), J2, linear + Voce isotropic hardening of the benchmark.
CalculiX: C3D8, *STEP NLGEOM, *PLASTIC table (multiplicative finite-strain
J2), same E, nu and flow curve. Quantities: Cauchy von Mises stress and
equivalent plastic strain at the end of every step.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, last_per_step, Grid, ccx_material_cards, flow_stress, parse_dat, read_sparlab_elements,
                    run_ccx, run_sparlab, sparlab_material, table_interp_error)

L = 1.0e-3  # cube edge [m]
OUT = ROOT / "homogeneous"

PATHS = {
    "uniaxial": [0.002, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],  # log strain
    "shear": [0.002, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],  # gamma
}
INC_PER_STEP = 20
CCX_INC = int(__import__("os").environ.get("CCX_INC", "20"))
TAG = __import__("os").environ.get("TAG", "")


def sparlab_bcs(path, v):
    """Boundary conditions of SparLab at path value v (absolute, SI)."""
    if path == "uniaxial":
        d = L * (math.exp(v) - 1.0)
        return [
            {"name": "x0", "fix": ["x"], "mode": "absolute", "value": [0, 0, 0], "region": {"box": {"xmax": 0.0}}},
            {"name": "y0", "fix": ["y"], "mode": "absolute", "value": [0, 0, 0], "region": {"box": {"ymax": 0.0}}},
            {"name": "z0", "fix": ["z"], "mode": "absolute", "value": [0, 0, 0], "region": {"box": {"zmax": 0.0}}},
            {"name": "pull", "fix": ["x"], "mode": "absolute", "value": [d, 0, 0], "region": {"box": {"xmin": L}}},
        ]
    return [
        {"name": "bottom", "fix": ["x", "y", "z"], "mode": "absolute", "value": [0, 0, 0],
         "region": {"box": {"zmax": 0.0}}},
        # u_y of the top nodes is left free (zero by symmetry): with every DOF
        # prescribed sparlab_form crashes (SIGSEGV, 0 free DOFs; README).
        {"name": "top", "fix": ["x", "z"], "mode": "absolute", "value": [v * L, 0, 0],
         "region": {"box": {"zmin": L}}},
    ]


def sparlab_deck(path):
    steps = []
    for k, v in enumerate(PATHS[path]):
        steps.append({"name": f"s{k:02d}", "type": "form", "increments": INC_PER_STEP,
                      "boundary_conditions": sparlab_bcs(path, v)})
    return {
        "name": f"homog_{path}", "units": "SI",
        "mesh": {"type": "structured_hex", "x0": 0, "y0": 0, "z0": 0, "lx": L, "ly": L, "lz": L,
                 "nx": 1, "ny": 1, "nz": 1},
        "model": {"stress_state": "three_dimensional"},
        "material": sparlab_material(),
        "forming": {"kinematics": "finite_logarithmic", "steps": steps,
                    "newton": {"max_iterations": 50, "residual_tolerance": 1e-9,
                               "displacement_tolerance": 1e-9, "max_cuts": 12},
                    "output": {"vtk": False, "snapshots": "steps"}},
    }


def ccx_deck(path):
    g = Grid(0, 0, 0, 1.0, 1.0, 1.0, 1, 1, 1)  # mm
    lines = ["*HEADING", f"homogeneous {path}", "*NODE, NSET=NALL"]
    for i, p in enumerate(g.xyz):
        lines.append(f"{i+1}, {p[0]:.12g}, {p[1]:.12g}, {p[2]:.12g}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    for e, c in enumerate(g.conn):
        lines.append(f"{e+1}, " + ", ".join(str(n + 1) for n in c))
    x = g.xyz
    def nset(name, mask):
        ids = [str(i + 1) for i in np.nonzero(mask)[0]]
        lines.append(f"*NSET, NSET={name}")
        lines.append(", ".join(ids))
    nset("X0", x[:, 0] < 1e-9); nset("X1", x[:, 0] > 1 - 1e-9)
    nset("Y0", x[:, 1] < 1e-9); nset("Z0", x[:, 2] < 1e-9); nset("Z1", x[:, 2] > 1 - 1e-9)
    lines.append(ccx_material_cards())
    lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=AL")
    for k, v in enumerate(PATHS[path]):
        lines += ["*STEP, NLGEOM, INC=10000", "*STATIC", f"{1.0/CCX_INC}, 1.0, 1e-6, {1.0/CCX_INC}"]
        if path == "uniaxial":
            d = math.exp(v) - 1.0
            lines += ["*BOUNDARY", "X0, 1, 1, 0.0", "Y0, 2, 2, 0.0", "Z0, 3, 3, 0.0", f"X1, 1, 1, {d:.12g}"]
        else:
            lines += ["*BOUNDARY", "Z0, 1, 3, 0.0", f"Z1, 1, 1, {v:.12g}", "Z1, 3, 3, 0.0"]
        lines += ["*EL PRINT, ELSET=EALL", "S, PEEQ", "*NODE PRINT, NSET=X1, TOTALS=ONLY", "RF",
                  "*END STEP"]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in PATHS:
        sdir, cdir = OUT / path / "sparlab", OUT / path / f"ccx{TAG}"
        rc_s, w_s = run_sparlab(sparlab_deck(path), sdir) if not (sdir / "out" / "summary.json").exists() else (0, 0.0)
        rc_c, w_c = run_ccx(ccx_deck(path), cdir)
        print(path, "sparlab rc", rc_s, f"{w_s:.1f}s", "ccx rc", rc_c, f"{w_c:.1f}s")
        blocks = parse_dat(cdir / "job.dat")
        sblocks = last_per_step(blocks, "stresses")
        pblocks = last_per_step(blocks, "equivalent plastic strain")
        for k, v in enumerate(PATHS[path]):
            el = read_sparlab_elements(sdir / "out" / f"step_{k+1}_s{k:02d}_elements.csv").reshape(-1, 3)
            s_mises, s_peeq = el[0, 2] / 1e6, el[0, 1]
            st = np.array(sblocks[k+1][3])[:, 2:8]  # sxx syy szz sxy sxz syz per int point
            sm = st.mean(axis=0)
            sxx, syy, szz, sxy, sxz, syz = sm
            c_mises = math.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                                + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))
            c_peeq = float(np.mean(np.array(pblocks[k+1][3])[:, 2]))
            rows.append(dict(path=path, step=k, value=v, sparlab_mises_MPa=s_mises, ccx_mises_MPa=c_mises,
                             rel_mises=(s_mises - c_mises) / c_mises, sparlab_peeq=s_peeq, ccx_peeq=c_peeq,
                             diff_peeq=s_peeq - c_peeq, ccx_sxx=sxx, ccx_szz=szz, ccx_sxz=sxz,
                             flow_at_ccx_peeq_MPa=float(flow_stress(c_peeq)) / 1e6))
    import csv
    with open(OUT / f"results{TAG}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['path']:9s} {r['value']:6.3f}  mises S={r['sparlab_mises_MPa']:8.3f} C={r['ccx_mises_MPa']:8.3f} "
              f"rel={r['rel_mises']:+.2e}  peeq S={r['sparlab_peeq']:.5f} C={r['ccx_peeq']:.5f}")
    print("table interpolation error", table_interp_error())


if __name__ == "__main__":
    main()
