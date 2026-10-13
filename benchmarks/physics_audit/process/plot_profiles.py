"""Azimuthally averaged radial profiles of the truncated cone (released
surface after the 3-2-1 release, and loaded surface at the end of forming)
for the process variants: fig_profiles.png, profiles.csv. scale2 is shown
divided by 2 (geometrically similar part)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proclens as P              # noqa: E402
from precomp.geometry.heightmap import HeightMap  # noqa: E402

PART = "truncated_cone-s2026-0000"
VARS = [("base", "benchmark (one tool, rigid path)"), ("dsif_ng", "second robot as support (DSIF)"),
        ("dsif_ng_gap02", "DSIF support 0.2 mm clear"), ("clamp8", "backing plate 12 mm (clamp 8)"),
        ("blank60", "wider frame (blank 60)"), ("scale2", "part x2 (shown /2)"),
        ("stepdown05", "step-down 0.5 mm"), ("tool6", "tool R 6 mm")]


def radial(z, X, Y, k=1.0, rmax=15.0):
    r = np.hypot(X, Y).ravel() / 1e-3 / k
    zz = z.ravel() / 1e-3 / k
    edges = np.arange(0, rmax + 0.25, 0.25)
    idx = np.digitize(r, edges)
    rc, zc = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.any():
            rc.append(0.5 * (edges[i - 1] + edges[i]))
            zc.append(zz[m].mean())
    return np.array(rc), np.array(zc)


rows = []
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
tgt_done = False
for name, label in VARS:
    d = P.RUNS / PART / name
    if not (d / "surfaces.npz").exists():
        continue
    S = np.load(d / "surfaces.npz")
    tgt = HeightMap.load(d / "commanded.npz") if (d / "commanded.npz").exists() else P.part_inputs(PART)[1]
    k = 2.0 if name == "scale2" else 1.0
    X, Y = tgt.grid.mesh()
    if not tgt_done:
        r, z = radial(tgt.z, X, Y)
        for a in ax:
            a.plot(r, z, "k--", lw=2, label="target")
        tgt_done = True
    for j, st in enumerate(("release", "form")):
        r, z = radial(S[st], X, Y, k)
        ax[j].plot(r, z, lw=1.3, label=label)
        for ri, zi in zip(r, z):
            rows.append({"variant": name, "state": st, "r_mm": ri, "z_mm": zi})
for a, t in zip(ax, ("released (3-2-1)", "loaded (end of forming)")):
    a.set_xlabel("r [mm]")
    a.set_ylabel("z [mm] (azimuthal mean)")
    a.set_title(f"{PART}: {t}")
    a.grid(alpha=0.3)
    a.set_xlim(0, 15)
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "fig_profiles.png", dpi=130)
pd.DataFrame(rows).to_csv(HERE / "profiles.csv", index=False, float_format="%.4f")
print("wrote fig_profiles.png")
