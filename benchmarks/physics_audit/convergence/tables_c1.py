#!/usr/bin/env python3
"""Table 5 (second check part, truncated_cone-s2026-0001) -> tables_c1.md.
Needs results_c1.csv / diffs_c1.csv (CONV_PART=c1 analyze.py) and optionally
richardson_fit_c1.csv (CONV_PART=c1 richardson_fit.py)."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
R = pd.read_csv(HERE / "results_c1.csv").set_index("case")
D = pd.read_csv(HERE / "diffs_c1.csv")
ORDER = [("c1_h2_L2_std", "2 mm, 2 std layers (**benchmark**)"),
         ("c1_h2_L1_im_tp5", "2 mm, 1 IM layer 2x2x5"),
         ("c1_h1_L1_im_tp5", "1 mm, 1 IM layer 2x2x5"),
         ("c1_h0.833_L1_im_tp5", "0.833 mm, 1 IM layer 2x2x5"),
         ("c1_h0.833_L1_im_tp5_p3", "0.833 mm, 1 IM layer 2x2x5, penalty 3"),
         ("c1_h0.625_L1_im_tp5", "0.625 mm, 1 IM layer 2x2x5 (finest)")]
REF = next(c for c in ("c1_h0.625_L1_im_tp5", "c1_h0.833_L1_im_tp5", "c1_h1_L1_im_tp5")
           if c in R.index)
lim = {}
fp = HERE / "richardson_fit_c1.csv"
if fp.exists():
    F = pd.read_csv(fp).set_index("quantity")
    lim = {q: (F.loc[q, "limit_min"], F.loc[q, "limit_max"]) for q in F.index}


def err(q, v):
    if q not in lim:
        return "-"
    a, b = sorted([v - lim[q][0], v - lim[q][1]], key=abs)
    return f"{a:+.3f}..{b:+.3f}"


out = ["### Table 5 - second check part truncated_cone-s2026-0001 (wall 54 deg) [mm]\n",
       f"Released-shape difference against `{REF}`; errors against the band of "
       "Richardson limits of `richardson_fit_c1.csv` (empty until three IM meshes exist).\n",
       "| Case | vert. RMS | vert. max | rim sag | springback RMS | released vs finest RMS / max "
       "| err. RMS | err. max | err. rim sag | wall [s] (load) |",
       "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
for c, lab in ORDER:
    if c not in R.index:
        continue
    r = R.loc[c]
    q = D[(D.case == c) & (D.reference == REF)]
    dv = "0 (finest)" if c == REF else (
        f"{q.release_rms_part_mm.iloc[0]:.3f} / {q.release_max_part_mm.iloc[0]:.3f}" if len(q) else "-")
    j = json.loads((HERE / "cases" / c / "run.json").read_text())
    out.append(f"| {lab} | {r.vert_rms_mm:.3f} | {r.vert_max_mm:.3f} | {r.rim_sag_mm:.3f} | "
               f"{r.sb_tot_rms_mm:.3f} | {dv} | {err('vert_rms_mm', r.vert_rms_mm)} | "
               f"{err('vert_max_mm', r.vert_max_mm)} | {err('rim_sag_mm', r.rim_sag_mm)} | "
               f"{j['wall_s']:.0f} ({j['loadavg_start'][0]:.1f}-{j['loadavg_end'][0]:.1f}) |")
(HERE / "tables_c1.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
