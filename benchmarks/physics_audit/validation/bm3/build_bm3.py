"""Build a SparLab forming deck replicating NUMISHEET 2014 Benchmark 3 (SPIF truncated cone,
AA7075-O, t = 1.63 mm, 45 deg, tool D = 12.66 mm) as described by Neto et al., IJAMT 85 (2016)
521-534, doi:10.1007/s00170-015-7954-9, section 2-3.

Reductions (all stated in README.md): step-down dz (paper: 0.5 mm), O-grid mesh of a disc of
radius 79 mm clamped on its rim (paper: 158 x 158 mm square window clamped on its edges),
backing plate = rigid plane under the ring r >= 70 mm without its 4 mm edge radius, von Mises
instead of Yld91 (paper's c_i ~ 1, m = 8), Swift law fitted by linear + Voce.

usage: python3 build_bm3.py <case_dir> --dz 2 --nc 120 --hr 2.0 [--penalty 3] [--travel 3]
       [--layers 1] [--form im|std] [--depth 44] [--mu 0.01]
"""
import argparse, json, math, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('case'); ap.add_argument('--dz', type=float, default=2.0)
ap.add_argument('--nc', type=int, default=120, help='circumferential segments (multiple of 4)')
ap.add_argument('--hr', type=float, default=2.0, help='radial element size in the tool zone [mm]')
ap.add_argument('--penalty', type=float, default=3.0); ap.add_argument('--travel', type=float, default=3.0)
ap.add_argument('--form', default='im'); ap.add_argument('--tp', type=int, default=5)
ap.add_argument('--depth', type=float, default=44.0); ap.add_argument('--mu', type=float, default=0.01)
ap.add_argument('--chord', type=float, default=2.5, help='toolpath chord length [mm]')
ap.add_argument('--rout', type=float, default=79.0)
ap.add_argument('--path', default='spiral', help='spiral | zlevel')
ap.add_argument('--kin', action='store_true', help='Voce part of the fitted curve as an Armstrong-Frederick backstress (C = Q b, gamma = b): same monotonic curve, Bauschinger effect')
ap.add_argument('--split', type=float, default=0.0, help='implicit: split the form step every this many mm of depth (files per step)')
# --explicit writes a form_explicit step, removed in the cleanup: such decks need a build of commit 7d571e6
ap.add_argument('--explicit', type=float, default=0.0, help='tool speed [m/s] of a form_explicit step (0: implicit)')
ap.add_argument('--dt', type=float, default=1e-6, help='explicit mass-scaling target time step [s]')
a = ap.parse_args()
os.makedirs(a.case, exist_ok=True)

T = 1.63; R = 12.66 / 2; RC1 = 62.0          # first contour: tool-centre diameter 124 mm
N = a.nc; assert N % 4 == 0
# ---------------- mesh (mm) ----------------
m = N // 4; acore = 8.0; r1 = 16.0; ntr = 4
radii = [r1]
r = r1
while r < 66.0 - 1e-9:
    r = min(66.0, r + a.hr); radii.append(r)
nout = max(2, round((a.rout - 66.0) / 2.2))
radii += list(np.linspace(66.0, a.rout, nout + 1)[1:])
# boundary loop of the core square, counter-clockwise, starting at corner (a,-a)
def sq(i):
    s = (i % N) / m; side = int(s); f = s - side
    c = [(acore, -acore), (acore, acore), (-acore, acore), (-acore, -acore), (acore, -acore)]
    x0, y0 = c[side]; x1, y1 = c[side + 1]
    return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
def circ(i, rr):
    th = -math.pi / 4 + 2 * math.pi * (i % N) / N
    return (rr * math.cos(th), rr * math.sin(th))
pts2 = []; idx = {}
def add(key, p):
    idx[key] = len(pts2); pts2.append(p)
for i in range(m + 1):
    for j in range(m + 1):
        add(('c', i, j), (-acore + 2 * acore * i / m, -acore + 2 * acore * j / m))
def core_key_of_loop(i):
    i %= N; s = i // m; f = i % m
    if s == 0: return ('c', m, f)
    if s == 1: return ('c', m - f, m)
    if s == 2: return ('c', 0, m - f)
    return ('c', f, 0)
for k in range(1, ntr + 1):
    s = k / ntr
    for i in range(N):
        p, q = sq(i), circ(i, r1)
        add(('t', k, i), ((1 - s) * p[0] + s * q[0], (1 - s) * p[1] + s * q[1]))
for k, rr in enumerate(radii[1:], start=1):
    for i in range(N): add(('r', k, i), circ(i, rr))
def ring_key(k, i):  # k=0 core boundary, 1..ntr transition, ntr+k polar
    i %= N
    if k == 0: return core_key_of_loop(i)
    if k <= ntr: return ('t', k, i)
    return ('r', k - ntr, i)
quads = []
for i in range(m):
    for j in range(m):
        quads.append([('c', i, j), ('c', i + 1, j), ('c', i + 1, j + 1), ('c', i, j + 1)])
nring = ntr + len(radii) - 1
for k in range(nring):
    for i in range(N):
        quads.append([ring_key(k, i), ring_key(k + 1, i), ring_key(k + 1, i + 1), ring_key(k, i + 1)])
P = np.array(pts2); n2 = len(P)
# orientation check (counter-clockwise in plan)
el = []
for q in quads:
    ids = [idx[x] for x in q]
    xy = P[ids]; area = 0.5 * sum(xy[j, 0] * xy[(j + 1) % 4, 1] - xy[(j + 1) % 4, 0] * xy[j, 1] for j in range(4))
    if area < 0: ids = ids[::-1]
    el.append(ids)
L = 1 if a.form == 'im' else 2
zs = np.linspace(-T, 0.0, L + 1)
with open(os.path.join(a.case, 'mesh.inp'), 'w') as f:
    f.write('*NODE\n')
    for l, z in enumerate(zs):
        for i, (x, y) in enumerate(P):
            f.write(f'{l * n2 + i + 1}, {x:.6f}, {y:.6f}, {z:.6f}\n')
    f.write('*ELEMENT, TYPE=C3D8\n')
    e = 1
    for l in range(L):
        for ids in el:
            b = [l * n2 + i + 1 for i in ids]; t = [(l + 1) * n2 + i + 1 for i in ids]
            f.write(f'{e}, ' + ', '.join(map(str, b + t)) + '\n'); e += 1
nel = (e - 1)
# ---------------- toolpath (z-level contours, clockwise, step-down at angle 0) ----------------
dz = a.dz; K = int(round(a.depth / dz))
pts = []
def push(x, y, z):
    pts.append((x, y, z))
d1 = dz; rc = RC1 - (d1 - 0.5)   # tool-centre radius so that the wall is the paper's (r_c = 62 at tip depth 0.5)
pts.append((62.5, 0.0, R + 0.3))   # tip 0.3 mm above the sheet
if a.path == 'zlevel':
    for k in range(1, K + 1):
        d = k * dz; rc = RC1 - (d - 0.5); zc = -d + R
        push(rc, 0.0, zc)                         # step down (and in) to the contour start
        nch = max(8, math.ceil(2 * math.pi * rc / a.chord))
        for j in range(1, nch + 1):
            th = -2 * math.pi * j / nch            # clockwise
            push(rc * math.cos(th), rc * math.sin(th), zc)
else:
    # spiral: the tip descends dz per revolution (clockwise) from the surface to the full depth, the tool-centre
    # radius following the same 45-degree wall (r_c = 62.5 - d), then one flat revolution at the full depth
    push(62.5, 0.0, R)
    th = 0.0; d = 0.0
    while d < K * dz - 1e-9:
        rc = 62.5 - d
        dth = a.chord / rc
        th -= dth; d = min(K * dz, -th / (2 * math.pi) * dz)
        push((62.5 - d) * math.cos(th), (62.5 - d) * math.sin(th), -d + R)
    rc = 62.5 - d; nch = max(8, math.ceil(2 * math.pi * rc / a.chord)); th0 = th
    for j in range(1, nch + 1):
        push(rc * math.cos(th0 - 2 * math.pi * j / nch), rc * math.sin(th0 - 2 * math.pi * j / nch), -d + R)
push(pts[-1][0], pts[-1][1], R + 5.0)       # retract
t = [0.0]
for p0, p1 in zip(pts[:-1], pts[1:]): t.append(t[-1] + math.dist(p0, p1) * 1e-3)
with open(os.path.join(a.case, 'toolpath.csv'), 'w') as f:
    f.write('t,x,y,z\n')
    for ti, p in zip(t, pts): f.write(f'{ti:.9f},{p[0]*1e-3:.9e},{p[1]*1e-3:.9e},{p[2]*1e-3:.9e}\n')
tform = t[-1]
zc_all = [p[2] for p in pts]
# ---------------- material: AA7075-O, Swift K=343.3 MPa, n=0.184, eps0=0.0015 (Neto Table 1) --------
fit = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'swift_fit.json')))
clamp = {"name": "clamp", "fix": ["x", "y", "z"], "mode": "hold",
         "region": {"annulus": {"center": [0, 0, 0], "inner_radius": (a.rout - 0.3) * 1e-3, "radius": (a.rout + 1) * 1e-3}}}
backing = {"name": "backing", "shape": "plane", "normal": [0, 0, 1], "friction": 0.0, "penalty": a.penalty,
           "surface": {"annulus": {"center": [0, 0, 0], "inner_radius": 0.070, "radius": (a.rout + 1) * 1e-3}},
           "trajectory": {"times": [0.0, tform], "points": [[0, 0, -T * 1e-3], [0, 0, -T * 1e-3]]}}
tool = {"name": "tool", "shape": "sphere", "radius": R * 1e-3, "friction": a.mu, "penalty": a.penalty,
        "surface": {"circle": {"center": [0, 0, 0], "radius": 0.0695}}, "trajectory": {"file": "toolpath.csv"}}
# tool surface: whole disc r < 69.5 mm; the tool only ever touches the top, but the region picks all faces;
# the bottom faces never come within reach of the sphere (sheet in between).
rs = 74.0e-3
sup = [{"name": "support_xyz", "fix": ["x", "y", "z"], "mode": "hold", "region": {"nearest_node": [rs, 0, 0]}},
       {"name": "support_yz", "fix": ["y", "z"], "mode": "hold", "region": {"nearest_node": [-rs, 0, 0]}},
       {"name": "support_z", "fix": ["z"], "mode": "hold", "region": {"nearest_node": [0, rs, 0]}}]
if a.explicit > 0:
    formstep = {"name": "form", "type": "form_explicit", "tools": ["tool", "backing"], "time": [0.0, tform],
                "boundary_conditions": [clamp],
                "explicit": {"tool_speed": a.explicit,
                             "mass_scaling": {"mode": "selective", "target_time_step": a.dt, "max_added_mass_fraction": 1000.0, "dynamic": True},
                             "contact_stiffness": 0.5, "history_every": 200}}
    unload_tools = []   # an implicit step keeping a tool would see the explicit penetration with the implicit penalty
else:
    formstep = {"name": "form", "type": "form", "tools": ["tool", "backing"], "time": [0.0, tform],
                "max_tool_travel": a.travel * 1e-3, "boundary_conditions": [clamp]}
    unload_tools = ["backing"]
    if a.split > 0:
        cuts = []; nxt = a.split
        for ti, zc in zip(t, zc_all):
            if R - zc >= nxt - 1e-9 and nxt < K * dz - 1e-9:
                cuts.append((ti, nxt)); nxt += a.split
        formsteps = []; t0 = 0.0
        for ti, dd in cuts:
            formsteps.append(dict(formstep, name=f"form_d{int(round(dd)):02d}", time=[t0, ti])); t0 = ti
        formsteps.append(dict(formstep, name="form", time=[t0, tform]))
model = {"stress_state": "three_dimensional"}
if a.form == 'im':
    model["element_formulation"] = "incompatible_modes"
if a.tp: model["integration"] = {"thickness_points": a.tp, "thickness_direction": "z"}
deck = {
  "name": os.path.basename(os.path.abspath(a.case)), "units": "SI",
  "description": f"NUMISHEET 2014 BM3 replica (Neto et al. 2016): dz={dz} mm, O-grid N={N} hr={a.hr} mm, {a.form}, {a.path}" + (", kinematic AF" if a.kin else "") + (f", explicit v={a.explicit} m/s dt={a.dt}" if a.explicit else ""),
  "mesh": {"type": "file", "path": "mesh.inp", "scale": 0.001},
  "model": model,
  "material": {"name": "AA7075-O", "youngs_modulus": 72e9, "poisson_ratio": 0.33, "density": 2810,
               "plasticity": ({"yield_stress": fit["sigma0"], "hardening_modulus": fit["H"],
                              "saturation_stress": fit["Q"], "saturation_rate": fit["b"]} if not a.kin else
                              {"yield_stress": fit["sigma0"], "hardening_modulus": fit["H"],
                               "backstresses": [{"modulus": fit["Q"] * fit["b"], "recovery": fit["b"]}]})},
  "forming": {
    "kinematics": "finite_logarithmic",
    "tools": [tool, backing],
    "steps": [
      *(formsteps if (a.explicit <= 0 and a.split > 0) else [formstep]),
      {"name": "unload", "type": "release", "tools": unload_tools, "boundary_conditions": [clamp]},
      {"name": "release", "type": "release", "tools": [], "boundary_conditions": sup}],
    "output": {"vtk": False, "snapshots": "steps"}}}
json.dump(deck, open(os.path.join(a.case, 'deck.json'), 'w'), indent=1)
print(f'{a.case}: {nel} elements, {len(zs)*n2} nodes, {3*len(zs)*n2} dofs, {K} contours, path {tform:.3f} m, '
      f'>= {int(tform/(a.travel*1e-3))} increments')
