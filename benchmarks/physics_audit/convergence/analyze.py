#!/usr/bin/env python3
"""Evaluate every finished case of the convergence study.

For each case (cases/<name>/output, plus the benchmark's own cached run as
"bench_cache"), on the target's 0.25 mm grid (the benchmark's metric):

* deviation of the released part from the target over the part (target
  z < 0): vertical RMS / max / bias, normal RMS (precomp.metrology, as
  springback_benchmark.deviation_metrics);
* regions as the benchmark's regions.csv: upper band (part < 1 mm deep, the
  rim; its bias is the "rim sag"), deeper part, flange (z = 0 outside the
  part), and the free flange ring (flange inside the clamp window);
* springback fields: release - unload (the springback on unclamping, "sb_rel")
  and release - form (from the end of forming, tool still on the part,
  "sb_tot"); RMS and largest |.| over the part;
* the formed shape (end of `form`) vs target, final depth, thinning, tool force,
  cost and solver counters from summary.json.

Pairwise differences of the released / formed / springback fields between
each case and its reference(s) (REFS below) go to diffs.csv: RMS and largest
|.| over the part and over the free window.

Writes results.csv, diffs.csv and surfaces/<case>.npz (the three step surfaces
on the target grid, small, for re-analysis).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(HERE))

from cases import PARTS  # noqa: E402

# CONV_PART=c1 evaluates the second check part (cases "c1_*", outputs *_c1.*)
PART_KEY = os.environ.get("CONV_PART", "t0")
PART_RUN = PARTS[PART_KEY]
PREFIX = "" if PART_KEY == "t0" else PART_KEY + "_"
SUFFIX = "" if PART_KEY == "t0" else "_" + PART_KEY
REFCASE = "h0.625_L1_im_tp7" if PART_KEY == "t0" else "c1_h0.625_L1_im_tp5"
from precomp.fea import load_result  # noqa: E402
from precomp.fea.results import _interpolate_structured  # noqa: E402
from precomp.geometry.heightmap import HeightMap  # noqa: E402
from precomp.metrology import flange_mask, metrics, part_mask, signed_deviation  # noqa: E402

MM = 1e-3
TARGET = HeightMap.load(PART_RUN / "commanded.npz")
GRID = TARGET.grid
X, Y = GRID.mesh()
PART = part_mask(TARGET)
UPPER = PART & (TARGET.z > -1e-3)
DEEP = PART & ~UPPER
FLANGE = flange_mask(TARGET)
WINDOW = (np.abs(X) < 0.015 - 1e-9) & (np.abs(Y) < 0.015 - 1e-9)
FREE_FLANGE = FLANGE & WINDOW
FRAME = ~WINDOW                     # the clamped frame (|x| or |y| >= 15 mm)


def frame_plane(z):
    """Least-squares plane a + b x + c y through z over the clamped frame
    (the released part's rigid position as its 3-2-1 supports left it)."""
    m = FRAME & np.isfinite(z)
    A = np.column_stack([np.ones(m.sum()), X[m], Y[m]])
    coef, *_ = np.linalg.lstsq(A, z[m], rcond=None)
    return coef[0] + coef[1] * X + coef[2] * Y


def case_dirs():
    out = {"bench_cache": PART_RUN / "output"}
    for d in sorted((HERE / "cases").glob("*")):
        is_other = any(d.name.startswith(k + "_") for k in PARTS if k != "t0")
        if d.is_dir() and (d / "DONE").exists() and (
                d.name.startswith(PREFIX) if PREFIX else not is_other):
            out[d.name] = d / "output"
    return out


def surfaces(res, name):
    """{'form','unload','release'} -> z on GRID (np arrays) + validity; cached in surfaces/."""
    cache = HERE / "surfaces" / f"{name}{SUFFIX if name == 'bench_cache' else ''}.npz"
    if cache.exists() and cache.stat().st_mtime > (res.directory / "summary.json").stat().st_mtime:
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    s = {}
    for st in ("form", "unload", "release"):
        hm = res.formed_surface(st, grid=GRID)
        s[st] = np.where(hm.mask, hm.z, np.nan)
        th = res.thickness_map(st, grid=GRID)
        s[st + "_thick"] = np.where(th.mask, th.z, np.nan)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **s)
    return s


def subsampled_release(res, k):
    """The released tool-side surface represented by every k-th node only
    (the coarse mesh's interpolation of the same deformed surface)."""
    lay = res._surface_layout()
    cur = res.step("release").current
    top = lay.top[::k, ::k]
    z, hit = _interpolate_structured(cur[top, :2], cur[top, 2], GRID, res.directory)
    return np.where(hit, z, np.nan)


def cubic_release(res, k=1, refine=8, step="release"):
    """The tool-side surface of `step` from every k-th node, the nodal
    displacements interpolated by bicubic splines over the reference plane
    (RectBivariateSpline) on a grid `refine` times finer, then triangulated
    and ray-cast like `formed_surface` - a smooth reconstruction of the nodal
    solution instead of flat triangles between the nodes."""
    from scipy.interpolate import RectBivariateSpline

    lay = res._surface_layout()
    st = res.step(step)
    top = lay.top[::k, ::k]
    ref = st.reference[top]
    cur = st.current[top]
    xs = ref[0, :, 0]
    ys = ref[:, 0, 1]
    xf = np.linspace(xs[0], xs[-1], (len(xs) - 1) * refine + 1)
    yf = np.linspace(ys[0], ys[-1], (len(ys) - 1) * refine + 1)
    comp = []
    for i in range(3):
        sp = RectBivariateSpline(ys, xs, cur[:, :, i], kx=3, ky=3)
        comp.append(sp(yf, xf))
    xy = np.stack(comp[:2], axis=-1)
    z, hit = _interpolate_structured(xy, comp[2], GRID, res.directory)
    return np.where(hit, z, np.nan)


def rms(a, m):
    v = a[m & np.isfinite(a)]
    return float(np.sqrt(np.mean(v * v))) / MM if v.size else np.nan


def mx(a, m):
    v = a[m & np.isfinite(a)]
    return float(np.max(np.abs(v))) / MM if v.size else np.nan


def bias(a, m):
    v = a[m & np.isfinite(a)]
    return float(np.mean(v)) / MM if v.size else np.nan


def case_row(name, d):
    res = load_result(d)
    s = surfaces(res, name)
    summ = res.summary
    row = {"case": name}
    p = {}
    cj = d.parent / "case.json"
    if cj.exists():
        p = json.loads(cj.read_text())
    elif name == "bench_cache":
        p = dict(h=2.0, L=2, form="std", tp=0, travel=1.0, pen=10.0)
    row.update({k: p.get(k) for k in ("h", "L", "form", "tp", "travel", "pen")})
    row["explicit"] = json.dumps(p["explicit"]) if p.get("explicit") else ""
    rel = HeightMap(GRID, np.nan_to_num(s["release"]), np.isfinite(s["release"]))
    dv = s["release"] - TARGET.z
    row["vert_rms_mm"] = rms(dv, PART)
    row["vert_max_mm"] = mx(dv, PART)
    row["vert_bias_mm"] = bias(dv, PART)
    row["normal_rms_mm"] = metrics(signed_deviation(rel, TARGET), PART)["rms"] / MM
    row["upper_rms_mm"] = rms(dv, UPPER)
    row["rim_sag_mm"] = bias(dv, UPPER)            # upper-band bias (benchmark's "rim sag")
    row["deep_rms_mm"] = rms(dv, DEEP)
    row["deep_bias_mm"] = bias(dv, DEEP)
    row["flange_rms_mm"] = rms(dv, FLANGE)
    row["free_flange_bias_mm"] = bias(dv, FREE_FLANGE)
    pl = frame_plane(s["release"])
    row["frame_plane_max_mm"] = mx(pl, WINDOW)
    row["frame_flatness_rms_mm"] = rms(s["release"] - pl, FRAME)
    dvr = s["release"] - pl - TARGET.z
    row["vert_rms_framereg_mm"] = rms(dvr, PART)
    row["rim_sag_framereg_mm"] = bias(dvr, UPPER)
    row["depth_release_mm"] = -float(np.nanmin(s["release"][PART])) / MM
    df = s["form"] - TARGET.z
    row["form_vert_rms_mm"] = rms(df, PART)
    row["form_rim_mm"] = bias(df, UPPER)
    row["depth_form_mm"] = -float(np.nanmin(s["form"][PART])) / MM
    sb_rel = s["release"] - s["unload"]
    sb_unl = s["unload"] - s["form"]
    sb_tot = s["release"] - s["form"]
    row["sb_rel_rms_mm"] = rms(sb_rel, PART)
    row["sb_rel_max_mm"] = mx(sb_rel, PART)
    row["sb_unload_rms_mm"] = rms(sb_unl, PART)
    row["sb_tot_rms_mm"] = rms(sb_tot, PART)
    row["sb_tot_max_mm"] = mx(sb_tot, PART)
    row["sb_tot_bias_mm"] = bias(sb_tot, PART)
    row["min_thickness_mm"] = float(np.nanmin(s["release_thick"][PART])) / MM
    st = {x["name"]: x for x in summ.get("steps", [])}
    tim = summ.get("timing", {})
    row["runtime_s"] = summ.get("runtime_s")
    row["increments"] = tim.get("increments")
    row["iterations"] = tim.get("iterations")
    row["cuts"] = tim.get("cuts")
    row["max_peeq"] = st.get("form", {}).get("max_plastic_strain")
    row["dofs"] = summ.get("mesh", {}).get("dofs")
    ex = st.get("form", {}).get("explicit")
    if ex:
        for k in ("steps", "max_kinetic_ratio", "peak_kinetic_ratio", "added_mass_fraction",
                  "max_penetration_ratio", "max_energy_error", "time_step_s"):
            row["x_" + k] = ex.get(k)
    try:
        f = res.forming_forces()
        f = f[f["step"] == 1]
        row["fz_mean_N"] = float(f.loc[f["active_nodes"] > 0, "fz"].mean())
        row["fz_max_N"] = float(f["fz"].max())
    except Exception:
        pass
    row["warnings"] = " | ".join(w for x in summ.get("steps", []) for w in x.get("warnings", [])
                                 if "over-constrained" not in w)[:300]
    return row, s, res


# reference pairs: (case, reference) - the reference is the finer / better setting
REFS = [
    ("bench_cache", "h2_L2_std"),
    ("h2_L2_std", "h1_L2_std"), ("h1_L2_std", "h0.5_L2_std"), ("h1_L2_std", "h0.625_L2_std"),
    ("h2_L2_std", "h0.5_L2_std"),
    ("h2_L2_std", "h2_L4_std"), ("h1_L2_std", "h1_L4_std"),
    ("h2_L2_std", "h2_L2_im"), ("h2_L2_std", "h2_L1_im_tp5"), ("h2_L2_std", "h2_L1_im_tp7"),
    ("h2_L2_std", "h2_L2_std_tp5"), ("h2_L2_im", "h2_L2_im_tp3"),
    ("h2_L1_im_tp5", "h2_L1_im_tp7"), ("h1_L1_im_tp5", "h1_L1_im_tp7"),
    ("h2_L1_im_tp7", "h1_L1_im_tp7"), ("h1_L1_im_tp7", "h0.5_L1_im_tp7"),
    ("h1_L1_im_tp7", "h0.625_L1_im_tp7"),
    ("h2_L2_im", "h1_L2_im"), ("h1_L2_im", "h1_L2_im_tp3"), ("h1_L2_im_tp3", "h1_L1_im_tp7"),
    ("h1_L2_im_tp3", "h0.625_L2_im_tp3"),
    ("h1_L2_std", "h1_L2_im_tp3"), ("h1_L2_std", "h1_L1_im_tp7"),
    ("h2_L2_std", "h2_L2_std_t0.5"), ("h1_L2_std", "h1_L2_std_t0.5"),
    ("h2_L2_std", "h2_L2_std_p30"), ("h2_L2_std", "h2_L2_std_tol1e-8"),
    ("h2_L2_std", "h2_L2_std_rel40"), ("h2_L2_std_p3", "h2_L2_std"), ("h1_L2_std", "h1_L2_std_p30"),
    ("h1_L2_std_x1", "h1_L2_std"), ("h1_L2_std_x05", "h1_L2_std"),
    ("h1_L2_std_x1", "h1_L2_std_x05"), ("h2_L2_std_x05", "h2_L2_std"),
    ("h1_L2_std_x05", "h0.5_L2_std_x05"), ("h0.625_L2_std_x05", "h0.625_L1_im_tp7"),
    ("h1_L2_std_x05", "h0.625_L2_std_x05"), ("h2_L2_std_x05", "h1_L2_std_x05"),
    ("h1_L1_im_tp7", "h0.625_L1_im_tp7"), ("h2_L2_std", "h0.625_L1_im_tp7"),
    ("h1_L2_std", "h0.625_L1_im_tp7"), ("h2_L1_im_tp7", "h0.625_L1_im_tp7"),
    ("h2_L2_std_x05", "h0.625_L1_im_tp7"), ("h1_L2_std_x05", "h0.625_L1_im_tp7"),
    ("h1_L1_im_tp5", "h0.625_L1_im_tp7"), ("h0.833_L1_im_tp5", "h0.625_L1_im_tp7"),
    ("h1_L1_im_tp5", "h0.833_L1_im_tp5"), ("h0.833_L1_im_tp5_p3", "h0.833_L1_im_tp5"), ("h2_L2_im", "h0.625_L1_im_tp7"), ("h0.5_L2_std_x05", "h0.5_L4_std_x05"),
    ("h2_L2_std", "h1_L2_im_tp3"), ("h2_L2_std", "h1_L1_im_tp7"),
    ("h2_L2_std", "h0.5_L1_im_tp7"), ("h2_L2_std", "h0.625_L1_im_tp7"),
    ("h0.714_L1_im_tp5_p3", "h0.625_L1_im_tp7"), ("h0.714_L1_im_tp5", "h0.625_L1_im_tp7"),
    ("h0.714_L1_im_tp5_p3", "h0.714_L1_im_tp5"), ("h0.714_L1_im_tp5_p3", "h0.833_L1_im_tp5_p3"),
]
if PART_KEY != "t0":
    REFS = [("bench_cache", "c1_h2_L2_std"),
            ("c1_h2_L2_std", "c1_h2_L1_im_tp5"), ("c1_h2_L1_im_tp5", "c1_h1_L1_im_tp5"),
            ("c1_h1_L1_im_tp5", "c1_h0.833_L1_im_tp5"), ("c1_h1_L1_im_tp5", "c1_h0.625_L1_im_tp5"),
            ("c1_h0.833_L1_im_tp5", "c1_h0.625_L1_im_tp5"),
            ("c1_h0.833_L1_im_tp5_p3", "c1_h0.833_L1_im_tp5"),
            ("c1_h2_L2_std", "c1_h1_L1_im_tp5"), ("c1_h2_L2_std", "c1_h0.833_L1_im_tp5")]


def main():
    rows, surf, results = [], {}, {}
    for name, d in case_dirs().items():
        try:
            r, s, res = case_row(name, d)
        except Exception as exc:
            print(f"{name}: {exc}")
            continue
        rows.append(r)
        surf[name] = s
        results[name] = res
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"results{SUFFIX}.csv", index=False, float_format="%.5g")
    drows = []
    pairs = list(REFS) + [(c, REFCASE) for c in surf
                          if c != REFCASE and (c, REFCASE) not in REFS]
    for a, b in pairs:
        if a not in surf or b not in surf:
            continue
        sa, sb = surf[a], surf[b]
        r = {"case": a, "reference": b}
        for key, fa, fb in (("release", sa["release"], sb["release"]),
                            ("form", sa["form"], sb["form"]),
                            ("sb_rel", sa["release"] - sa["unload"], sb["release"] - sb["unload"]),
                            ("sb_tot", sa["release"] - sa["form"], sb["release"] - sb["form"])):
            dd = fa - fb
            r[f"{key}_rms_part_mm"] = rms(dd, PART)
            r[f"{key}_max_part_mm"] = mx(dd, PART)
            r[f"{key}_rms_window_mm"] = rms(dd, WINDOW)
            r[f"{key}_max_window_mm"] = mx(dd, WINDOW)
        ra = sa["release"] - frame_plane(sa["release"])
        rb = sb["release"] - frame_plane(sb["release"])
        r["release_framereg_rms_part_mm"] = rms(ra - rb, PART)
        r["release_framereg_max_part_mm"] = mx(ra - rb, PART)
        r["vert_rms_change_mm"] = rms(sa["release"] - TARGET.z, PART) - rms(sb["release"] - TARGET.z, PART)
        r["vert_max_change_mm"] = mx(sa["release"] - TARGET.z, PART) - mx(sb["release"] - TARGET.z, PART)
        dr = sa["release"] - sb["release"]
        r["release_upper_bias_diff_mm"] = bias(dr, UPPER)
        r["release_deep_bias_diff_mm"] = bias(dr, DEEP)
        drows.append(r)
    # interpolation error of a coarse representation of a fine surface
    for name, k in (("h1_L2_std", 2), ("h0.5_L2_std", 4), ("h0.5_L2_std", 2),
                    ("h1_L1_im_tp7", 2), ("h0.5_L1_im_tp7", 4), ("h0.5_L1_im_tp7", 2),
                    ("h0.5_L2_std_x05", 4), ("h0.5_L2_std_x05", 2)):
        if name not in results:
            continue
        zc = subsampled_release(results[name], k)
        dd = zc - surf[name]["release"]
        drows.append({"case": f"{name}[every {k}th node]", "reference": name,
                      "release_rms_part_mm": rms(dd, PART), "release_max_part_mm": mx(dd, PART),
                      "release_rms_window_mm": rms(dd, WINDOW),
                      "release_max_window_mm": mx(dd, WINDOW),
                      "release_upper_bias_diff_mm": bias(dd, UPPER),
                      "release_deep_bias_diff_mm": bias(dd, DEEP),
                      "vert_rms_mm_coarse_repr": rms(zc - TARGET.z, PART),
                      "rim_sag_mm_coarse_repr": bias(zc - TARGET.z, UPPER)})
    # azimuthally averaged profiles (the part is axisymmetric)
    R = np.hypot(X, Y)
    edges = np.arange(0.0, 15.0001e-3, 0.25e-3)
    prow = []
    for name, s in surf.items():
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (R >= lo) & (R < hi)
            prow.append({"case": name, "r_mm": 0.5 * (lo + hi) / MM,
                         "target_z_mm": float(np.nanmean(TARGET.z[m])) / MM,
                         "release_dev_mm": float(np.nanmean((s["release"] - TARGET.z)[m])) / MM,
                         "form_dev_mm": float(np.nanmean((s["form"] - TARGET.z)[m])) / MM,
                         "unload_dev_mm": float(np.nanmean((s["unload"] - TARGET.z)[m])) / MM,
                         "sb_tot_mm": float(np.nanmean((s["release"] - s["form"])[m])) / MM,
                         "sb_rel_mm": float(np.nanmean((s["release"] - s["unload"])[m])) / MM,
                         "release_dev_sd_mm": float(np.nanstd((s["release"] - TARGET.z)[m])) / MM})
    pd.DataFrame(prow).to_csv(HERE / f"profiles{SUFFIX}.csv", index=False, float_format="%.5g")
    # smooth (bicubic) reconstruction of the nodal surface: of a subsampled
    # fine run against the fine run itself, and of the coarse runs
    for name, k, ref in (("h1_L1_im_tp7", 2, "h1_L1_im_tp7"), ("h1_L2_std", 2, "h1_L2_std"),
                         ("h0.625_L1_im_tp7", 2, "h0.625_L1_im_tp7"),
                         ("h2_L2_std", 1, "h1_L1_im_tp7"), ("h2_L2_std", 1, "h0.625_L1_im_tp7"),
                         ("h2_L1_im_tp7", 1, "h0.625_L1_im_tp7"),
                         ("h1_L1_im_tp7", 1, "h0.625_L1_im_tp7")):
        if name not in results or ref not in surf:
            continue
        zc = cubic_release(results[name], k)
        dd = zc - surf[ref]["release"]
        drows.append({"case": f"{name}[every {k}th node, bicubic]", "reference": ref,
                      "release_rms_part_mm": rms(dd, PART), "release_max_part_mm": mx(dd, PART),
                      "release_rms_window_mm": rms(dd, WINDOW),
                      "release_max_window_mm": mx(dd, WINDOW),
                      "release_upper_bias_diff_mm": bias(dd, UPPER),
                      "release_deep_bias_diff_mm": bias(dd, DEEP),
                      "vert_rms_mm_coarse_repr": rms(zc - TARGET.z, PART),
                      "rim_sag_mm_coarse_repr": bias(zc - TARGET.z, UPPER)})
    dfd = pd.DataFrame(drows)
    dfd.to_csv(HERE / f"diffs{SUFFIX}.csv", index=False, float_format="%.5g")
    cols = ["case", "vert_rms_mm", "vert_max_mm", "rim_sag_mm", "deep_bias_mm",
            "free_flange_bias_mm", "depth_release_mm", "sb_rel_rms_mm", "sb_tot_rms_mm",
            "sb_tot_max_mm", "runtime_s", "increments", "iterations", "cuts"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if not dfd.empty:
            print(dfd[[c for c in ("case", "reference", "release_rms_part_mm",
                                   "release_max_part_mm", "release_upper_bias_diff_mm",
                                   "sb_tot_rms_part_mm", "form_rms_part_mm")
                       if c in dfd]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
