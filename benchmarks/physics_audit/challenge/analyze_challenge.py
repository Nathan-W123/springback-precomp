"""Measure the challenge reruns with the convergence lens' own metric code
(convergence/analyze.py case_row: target grid, part / rim band / deep masks)
and compare each to its base on the same 0.833 mm IM5 mesh
(convergence/cases/h0.833_L1_im_tp5) and to the 2 mm-mesh effect the
material / process lens reported. Writes results.csv."""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(A / "convergence"))
import analyze as C            # noqa: E402
C.HERE = HERE                  # surfaces cache -> challenge/surfaces
MM = 1e-3

def surf_of(name, outdir):
    row, s, res = C.case_row(name, outdir)
    return row, s

rows = []
base_row, base = surf_of("h0.833_L1_im_tp5", A / "convergence/cases/h0.833_L1_im_tp5/output")
base_row["case"] = "base_h0833 (convergence lens run)"
rows.append(base_row)
for name in ["kin_h0833", "dsif_h0833"]:
    d = HERE / "cases" / name
    if not (d / "DONE").exists():
        continue
    row, s = surf_of(name, d / "output")
    for st in ("release", "form"):
        diff = s[st] - base[st]
        row[f"d_{st}_rms_mm"] = C.rms(diff, C.PART)
        row[f"d_{st}_max_mm"] = C.mx(diff, C.PART)
        row[f"d_{st}_rim_rms_mm"] = C.rms(diff, C.UPPER)
        row[f"d_{st}_deep_rms_mm"] = C.rms(diff, C.DEEP)
    row["springback_change_pct"] = 100 * (row["sb_tot_rms_mm"] / base_row["sb_tot_rms_mm"] - 1)
    row["d_rim_sag_mm"] = row["rim_sag_mm"] - base_row["rim_sag_mm"]
    rows.append(row)
# the same deltas at 2 mm, recomputed from the lenses' raw outputs with this code
M = A / "material/runs/truncated_cone-s2026-0000"
PR = A / "process/runs/truncated_cone-s2026-0000"
pairs = [("kin_shutov (2 mm std, material lens)", M / "kin_shutov/output", "m_base", M / "base/output"),
         ("kin_shutov_im5 (2 mm IM5, material lens)", M / "kin_shutov_im5/output", "m_base_im5", M / "base_im5/output"),
         ("dsif_ng (2 mm std, process lens)", PR / "dsif_ng/output", "m_base", M / "base/output")]
cache = {}
for name, out, bname, bout in pairs:
    if bname not in cache:
        cache[bname] = surf_of(bname, bout)
    brow, b = cache[bname]
    row, s = surf_of(name.split()[0] + ("_p" if "process" in name else "_m"), out)
    row["case"] = name
    for st in ("release", "form"):
        diff = s[st] - b[st]
        row[f"d_{st}_rms_mm"] = C.rms(diff, C.PART)
        row[f"d_{st}_max_mm"] = C.mx(diff, C.PART)
        row[f"d_{st}_rim_rms_mm"] = C.rms(diff, C.UPPER)
        row[f"d_{st}_deep_rms_mm"] = C.rms(diff, C.DEEP)
    row["springback_change_pct"] = 100 * (row["sb_tot_rms_mm"] / brow["sb_tot_rms_mm"] - 1)
    row["d_rim_sag_mm"] = row["rim_sag_mm"] - brow["rim_sag_mm"]
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(HERE / "results.csv", index=False, float_format="%.5g")
cols = [c for c in ["case", "vert_rms_mm", "vert_max_mm", "rim_sag_mm", "deep_bias_mm", "depth_release_mm",
                    "sb_tot_rms_mm", "springback_change_pct", "d_release_rms_mm", "d_release_max_mm",
                    "d_release_rim_rms_mm", "d_release_deep_rms_mm", "d_rim_sag_mm", "fz_mean_N", "fz_max_N",
                    "runtime_s"] if c in df]
print(df[cols].to_string())
