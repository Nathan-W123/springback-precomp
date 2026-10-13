#!/usr/bin/env python3
"""The springback benchmark's held-out test parts on the physics audit's
recommended mesh, with and without support from below.

usage:
    OPENBLAS_NUM_THREADS=1 python3 python/scripts/springback_fine_benchmark.py \
        --out benchmarks/springback_fine --work benchmarks/springback_fine/work \
        [--workers 4] [--strategies none dsif backing_plate] [--parts ...]

Every formed shape is a `sparlab_form` simulation (implicit, quasi-static,
finite_logarithmic kinematics, Hill48 AA5754-O with nominal handbook data):
**simulation, not experiment**.

Protocol (config.json, written from `PROTOCOL` when absent):
  * parts: the 8 test parts of benchmarks/springback (design indices 0-1 of
    each family, the same Sobol design, seed 2026);
  * setup: the named setup `springback_fine` (precomp.fea.PRESETS) - the
    benchmark's process on the audit's 0.833 mm incompatible-mode Hex8 mesh,
    one layer, 5 points through the thickness, penalty 10;
  * strategies: single-point forming ("none"), a backing plate with a 1 mm
    clearance to the target's outline, and two-tool forming ("dsif": an
    underside support tool follows the forming tool, sine-law gap);
  * methods: uncompensated (1 FE run), FE-DA-1 (2), FE-DA-2 (3): vertical
    displacement adjustment with the FE model in the loop, alpha = 1, no
    smoothing, 65 deg wall limit, the support-aware masks and command bound
    (`compensation_masks`, `command_upper_bound`) - the fixture made for the
    target.

Protocol revision 2 (the deck fixes listed at `PROTOCOL`).

Resumable: every run goes through precomp's content-addressed cache in
`<work>/runs` (an identical deck is never run twice), so a rerun after an
interruption repeats only the runs that were in flight. The tables are
rewritten each time a DA chain finishes, from the chains finished so far.

Outputs in --out: config.json, cases.csv (one row per part, strategy and FE
run: deviation by region, cost, physics checks), tools.csv, summary.csv
(per strategy and method, mean over the parts), summary_common_parts.csv (the
same over the parts every strategy and method finished), tables.md, run.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import common  # noqa: E402  (deviation, physics, tool_rows, versions)
from common import LABEL, MM, REPO  # noqa: E402

OLD = REPO / "benchmarks" / "springback"

PROTOCOL: Dict[str, Any] = {
    # 2: the same protocol on the fixed deck generation - the deepest tool
    # path level no longer dropped under a dimpled floor or a dome's pole,
    # and DSIF support knots between the tool's where it swings round it
    # (revision 1 had both defects; its outputs are at tag pre-cleanup)
    "protocol_revision": 2,
    "data_source": LABEL,
    "parts_from": "benchmarks/springback (config.json design, test split: indices 0-1 per family)",
    "preset": "springback_fine",
    "setup_overrides": {},
    "strategies": {
        "none": {"support": "none"},
        "backing_plate": {"support": "backing_plate", "support_settings": {"clearance": 0.001}},
        "dsif": {"support": "dsif", "support_settings": {}},
    },
    "da": {"fe_runs": 3, "alpha": 1.0, "direction": "vertical", "smoothing": None,
           "max_wall_angle_deg": 65.0},
    "upper_band_depth": 0.001,
    "timeout_s": 8 * 3600.0,
}

METHODS = {1: "uncompensated", 2: "FE-DA-1", 3: "FE-DA-2", 4: "FE-DA-3"}
STRATEGY_ORDER = {"none": 0, "backing_plate": 1, "dsif": 2}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def test_points(old_cfg: Dict[str, Any]):
    """The springback benchmark's test parts, drawn from its own design."""
    from precomp.ml import DesignSpace, design_points
    from precomp.materials import get_material

    s = dict(old_cfg["solver"]["setup"])
    s["material"] = get_material(s["material"])
    space = DesignSpace(families=tuple(old_cfg["families"]), process={},
                        materials=(old_cfg["solver"]["setup"]["material"],), base_setup=s,
                        part_bounds={f: {k: tuple(v) for k, v in b.items()}
                                     for f, b in old_cfg["part_bounds"].items()},
                        grid_spacing=old_cfg["grid_spacing"])
    n = int(old_cfg["test_per_family"])
    return [p for p in design_points(space, n, old_cfg["seed"])
            if int(p.point_id.rsplit("-", 1)[1]) < n]


def make_setup(cfg: Dict[str, Any], strategy: str, exe: str):
    from precomp.fea import FormingSetup

    over = copy.deepcopy(cfg.get("setup_overrides", {}))
    over.update(copy.deepcopy(cfg["strategies"][strategy]))
    return FormingSetup.preset(cfg["preset"], executable=exe, timeout=cfg["timeout_s"],
                               **over)


def run_chain(cfg: Dict[str, Any], point, strategy: str, work: Path,
              exe: str) -> List[Dict[str, Any]]:
    """The DA chain of one part and strategy (`da.fe_runs` FE runs, through
    the cache); one record per FE run, or a failure record."""
    from precomp.compensation import FEAPredictor, displacement_adjustment
    from precomp.fea.support import command_upper_bound, compensation_masks

    setup = make_setup(cfg, strategy, exe)
    target = point.target()
    pred = FEAPredictor(setup, work / "runs", target=target)
    upper = command_upper_bound(setup, target) if setup.support != "none" else None
    hold, adjust = compensation_masks(setup, target)
    records: List[Dict[str, Any]] = []
    t_chain = time.time()

    def cb(k, commanded, formed, error):
        res = pred.results[-1]
        rec: Dict[str, Any] = {
            "point_id": point.point_id, "family": point.family, "strategy": strategy,
            "method": METHODS.get(k + 1, f"FE-DA-{k}"), "fe_run": k + 1,
            "commanded_max_mm": float(commanded.z.max()) / MM,
            "commanded_depth_mm": commanded.depth / MM,
            "deck_hash": res.provenance.get("key"), "cache_hit": res.provenance.get("cache_hit"),
            "wall_runtime_s": res.provenance.get("runtime_s")}
        rec.update(common.deviation(formed, target, cfg["upper_band_depth"]))
        rec.update(common.physics(res))
        theta = target.wall_angle()
        wall = (target.z < -3e-4) & (np.degrees(theta) > 30.0)
        th = res.thickness_map("form", grid=target.grid)
        sel = wall & th.mask
        rec["wall_thickness_mm"] = float(np.median(th.z[sel])) / MM if sel.any() else None
        rec["sine_law_thickness_mm"] = (float(np.median(setup.thickness * np.cos(theta[sel])))
                                        / MM if sel.any() else None)
        prov = json.loads((res.directory.parent / "precomp_deck.json").read_text())
        plate = prov.get("support", {}).get("plate")
        if plate:
            rec["plate_realised_clearance_mm"] = plate["realised_clearance_m"] / MM
        rec["_tools"] = [{"point_id": point.point_id, "strategy": strategy,
                          "method": rec["method"], **t} for t in common.tool_rows(rec["method"], res)]
        records.append(rec)
        log(f"{point.point_id} {strategy} {rec['method']}: RMS {rec['vert_rms_mm']:.3f} mm, "
            f"rim sag {rec['upper_band_bias_mm']:+.3f} mm, {rec['runtime_s']:.0f} s"
            + (" (cached)" if rec["cache_hit"] else ""))

    da = cfg["da"]
    try:
        displacement_adjustment(target, pred, iterations=int(da["fe_runs"]), alpha=da["alpha"],
                                direction=da["direction"], smoothing=da["smoothing"],
                                max_wall_angle_deg=da["max_wall_angle_deg"],
                                upper_bound=upper, hold_mask=hold, adjust_mask=adjust,
                                tool_radius=setup.tool_radius if da.get("tool_reach") else None,
                                callback=cb)
    except Exception as exc:  # recorded, reported
        k = len(records) + 1
        records.append({"point_id": point.point_id, "family": point.family,
                        "strategy": strategy, "method": METHODS.get(k, f"FE-DA-{k - 1}"),
                        "fe_run": k, "completed": False,
                        "error": f"{type(exc).__name__}: {exc}".splitlines()[0][:400]})
        log(f"{point.point_id} {strategy}: FAILED at FE run {k} - {records[-1]['error']}")
    for r in records:
        r["chain_elapsed_s"] = time.time() - t_chain
    return records


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def summarise(cases: pd.DataFrame) -> pd.DataFrame:
    ok = cases[cases["completed"].fillna(False).astype(bool)]
    rows = []
    unc = ok[ok["method"] == "uncompensated"].set_index(["strategy", "point_id"])["vert_rms_mm"]
    none_unc = ok[(ok["method"] == "uncompensated") & (ok["strategy"] == "none")] \
        .set_index("point_id")["vert_rms_mm"]
    for (strategy, method), g in ok.groupby(["strategy", "method"]):
        fe = int(g["fe_run"].max())
        # FE runs and CPU time the method costs per part: its own run and the chain's before
        chain = ok[(ok["strategy"] == strategy) & (ok["fe_run"] <= fe)
                   & ok["point_id"].isin(g["point_id"])]
        failed = cases[(cases["strategy"] == strategy) & (cases["method"] == method)
                       & ~cases["completed"].fillna(False).astype(bool)]
        r = {"strategy": strategy, "method": method, "fe_runs_per_part": fe,
             "n_parts": int(len(g)), "n_failed": int(len(failed)),
             "vert_rms_mean_mm": g["vert_rms_mm"].mean(),
             "vert_rms_median_mm": g["vert_rms_mm"].median(),
             "vert_max_mean_mm": g["vert_max_mm"].mean(),
             "normal_rms_mean_mm": g["normal_rms_mm"].mean(),
             "rim_sag_mean_mm": g["upper_band_bias_mm"].mean(),
             "upper_band_rms_mean_mm": g["upper_band_rms_mm"].mean(),
             "interior_rms_mean_mm": g["deep_rms_mm"].mean(),
             "interior_bias_mean_mm": g["deep_bias_mm"].mean(),
             "flange_bias_mean_mm": g["flange_bias_mm"].mean(),
             "upper_band_share_mean": g["upper_band_share"].mean(),
             "runtime_s_mean": g["runtime_s"].mean(),
             "runtime_s_max": g["runtime_s"].max(),
             "cpu_h_per_part": chain["runtime_s"].sum() / 3600.0 / max(len(g), 1),
             "newton_iterations_mean": g["iterations"].mean(),
             "cuts_mean": g["cuts"].mean()}
        u = unc.reindex([(strategy, p) for p in g["point_id"]]).to_numpy()
        r["parts_better_than_own_uncompensated"] = int((g["vert_rms_mm"].to_numpy() < u).sum()) \
            if method != "uncompensated" else None
        nu = none_unc.reindex(g["point_id"]).to_numpy()
        r["parts_better_than_spif_uncompensated"] = int((g["vert_rms_mm"].to_numpy() < nu).sum())
        rows.append(r)
    sm = pd.DataFrame(rows)
    if len(sm):
        sm["o"] = sm["strategy"].map(STRATEGY_ORDER) * 10 + sm["fe_runs_per_part"]
        sm = sm.sort_values("o").drop(columns="o").reset_index(drop=True)
        sm["data_source"] = LABEL
    return sm


def write_tables(out: Path, cfg: Dict[str, Any], cases: pd.DataFrame, tools: pd.DataFrame,
                 meta: Dict[str, Any]) -> None:
    cases = cases.copy()
    if "completed" not in cases:
        cases["completed"] = False
    cases["data_source"] = LABEL
    cases = cases.sort_values(["point_id", "strategy", "fe_run"],
                              key=lambda s: s.map(STRATEGY_ORDER) if s.name == "strategy" else s)
    cases.to_csv(out / "cases.csv", index=False, float_format="%.6g")
    if len(tools):
        tools.assign(data_source=LABEL).to_csv(out / "tools.csv", index=False,
                                               float_format="%.6g")
    sm = summarise(cases)
    sm.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    # the same over the parts where every strategy and method has a result, so
    # that a failed chain does not change the set of parts a mean is taken over
    okc = cases[cases["completed"].fillna(False).astype(bool)]
    n_combos = okc.groupby("point_id").apply(lambda g: len(set(zip(g["strategy"],
                                                                 g["method"]))))
    full = n_combos.max() if len(n_combos) else 0
    common = sorted(n_combos[n_combos == full].index)
    smc = summarise(cases[cases["point_id"].isin(common)])
    smc.to_csv(out / "summary_common_parts.csv", index=False, float_format="%.4f")
    from precomp.fea.setup import PRESETS

    penalty = PRESETS[cfg["preset"]].get("contact", {}).get("penalty", 10.0)
    L = ["# Springback benchmark test parts on the fine mesh: SparLab simulations, not "
         "experiments", "",
         f"Setup `{cfg['preset']}` (0.833 mm incompatible-mode Hex8, 1 layer, 5 thickness "
         f"points, penalty {penalty:g}); the 8 test parts of benchmarks/springback. Deviation "
         "of the released part from the target over the part [mm]; rim sag = mean vertical "
         "deviation "
         "of the part less than 1 mm deep (negative = too deep); interior = the part deeper "
         "than that.", "",
         f"{int(cases['completed'].fillna(False).astype(bool).sum())} simulations complete, "
         f"{int((~cases['completed'].fillna(False).astype(bool)).sum())} failed records.", "",
         "| strategy | method | FE runs / part | parts | vertical RMS mean | median | max mean | "
         "normal RMS | rim sag | upper band RMS | interior RMS (bias) | flange bias | "
         "runtime / run [s] | CPU h / part | Newton its | failed |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in sm.iterrows():
        L.append(f"| {r['strategy']} | {r['method']} | {r['fe_runs_per_part']} | "
                 f"{r['n_parts']} | **{fmt(r['vert_rms_mean_mm'])}** | "
                 f"{fmt(r['vert_rms_median_mm'])} | {fmt(r['vert_max_mean_mm'])} | "
                 f"{fmt(r['normal_rms_mean_mm'])} | {fmt(r['rim_sag_mean_mm'])} | "
                 f"{fmt(r['upper_band_rms_mean_mm'])} | {fmt(r['interior_rms_mean_mm'])} "
                 f"({fmt(r['interior_bias_mean_mm'], 2)}) | {fmt(r['flange_bias_mean_mm'])} | "
                 f"{fmt(r['runtime_s_mean'], 0)} | {fmt(r['cpu_h_per_part'], 2)} | "
                 f"{fmt(r['newton_iterations_mean'], 0)} | {r['n_failed']} |")
    L += ["", f"The same over the {len(common)} parts where every strategy and method has a "
          "result (" + ", ".join(common) + "):", "",
          "| strategy | method | vertical RMS mean | max mean | rim sag | interior RMS (bias) |",
          "|---|---|--:|--:|--:|--:|"]
    for _, r in smc.iterrows():
        L.append(f"| {r['strategy']} | {r['method']} | **{fmt(r['vert_rms_mean_mm'])}** | "
                 f"{fmt(r['vert_max_mean_mm'])} | {fmt(r['rim_sag_mean_mm'])} | "
                 f"{fmt(r['interior_rms_mean_mm'])} ({fmt(r['interior_bias_mean_mm'], 2)}) |")
    # per part
    ok_cases = cases[cases["completed"].fillna(False).astype(bool)]
    L += ["", "Per part, vertical RMS [mm] (max |dev| in brackets; rim sag after the "
          "slash):", ""]
    combos = [(s, m) for s in cfg["strategies"] for m in ["uncompensated", "FE-DA-1", "FE-DA-2"]]
    L.append("| part | " + " | ".join(f"{s} {m}" for s, m in combos) + " |")
    L.append("|---|" + "--:|" * len(combos))
    for pid in sorted(cases["point_id"].unique()):
        cells = []
        for s, m in combos:
            x = ok_cases[(ok_cases["point_id"] == pid) & (ok_cases["strategy"] == s)
                         & (ok_cases["method"] == m)]
            f = cases[(cases["point_id"] == pid) & (cases["strategy"] == s)
                      & (cases["method"] == m) & ~cases["completed"].fillna(False).astype(bool)]
            cells.append(f"{x['vert_rms_mm'].iloc[0]:.3f} ({x['vert_max_mm'].iloc[0]:.2f}) / "
                         f"{x['upper_band_bias_mm'].iloc[0]:+.2f}" if len(x)
                         else ("FAILED" if len(f) else "-"))
        L.append(f"| {pid} | " + " | ".join(cells) + " |")
    fails = cases[~cases["completed"].fillna(False).astype(bool)]
    L += ["", "Failures:", ""]
    if len(fails):
        for _, r in fails.iterrows():
            L.append(f"* {r['point_id']} {r['strategy']} {r['method']}: {r.get('error', '')}")
    else:
        L.append("none")
    (out / "tables.md").write_text("\n".join(L) + "\n")
    (out / "run.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True, help="run cache (large, git-ignored)")
    ap.add_argument("--workers", type=int, default=4, help="DA chains (= simulations) at a time")
    ap.add_argument("--strategies", nargs="*", help="subset of the protocol's strategies")
    ap.add_argument("--parts", nargs="*", help="subset of the test parts (point ids)")
    ap.add_argument("--alpha", type=float,
                    help="DA relaxation factor instead of the protocol's (a study: give it its "
                         "own --out; the run cache can be shared)")
    ap.add_argument("--tool-reach", action="store_true",
                    help="clip every DA command to the tool's reach (a study: its own --out)")
    ap.add_argument("--fe-runs", type=int,
                    help="FE runs per DA chain instead of the protocol's (a new --out only)")
    ap.add_argument("--executable",
                    help="sparlab_form; default: a copy of build/bin/sparlab_form kept in "
                         "<work>/bin, so a rebuild during the benchmark cannot change the "
                         "solver version in the run hashes")
    args = ap.parse_args(argv)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if args.alpha is not None and args.alpha != cfg["da"]["alpha"]:
            ap.error(f"{cfg_path} has alpha {cfg['da']['alpha']}; use another --out")
        if args.tool_reach and not cfg["da"].get("tool_reach"):
            ap.error(f"{cfg_path} does not clip to the tool's reach; use another --out")
        if args.fe_runs is not None:
            cfg["da"]["fe_runs"] = args.fe_runs   # a longer chain resumes from the cache
            cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    else:
        cfg = copy.deepcopy(PROTOCOL)
        if args.alpha is not None:
            cfg["da"]["alpha"] = args.alpha
        if args.tool_reach:
            cfg["da"]["tool_reach"] = True
        if args.fe_runs is not None:
            cfg["da"]["fe_runs"] = args.fe_runs
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    if args.executable:
        exe = str(Path(args.executable).resolve())
    else:
        exe_copy = work / "bin" / "sparlab_form"
        if not exe_copy.exists():
            exe_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / "build" / "bin" / "sparlab_form", exe_copy)
        exe = str(exe_copy.resolve())
    old_cfg = json.loads((OLD / "config.json").read_text())
    points = test_points(old_cfg)
    if args.parts:
        points = [p for p in points if p.point_id in set(args.parts)]
    strategies = args.strategies or list(cfg["strategies"])
    # the costliest chains first (plate, then DSIF, then single-point forming)
    jobs = [(p, s) for s in strategies for p in points]
    jobs.sort(key=lambda j: ({"backing_plate": 0, "dsif": 1, "none": 2}.get(j[1], 3),
                             j[0].point_id))
    t0 = time.time()
    load0 = os.getloadavg()
    log(f"{len(points)} parts x {len(strategies)} strategies = {len(jobs)} DA chains of "
        f"{cfg['da']['fe_runs']} FE runs, {args.workers} at a time; solver {exe}")
    lock = threading.Lock()
    done: Dict[tuple, List[Dict[str, Any]]] = {}

    def meta() -> Dict[str, Any]:
        return {"command": " ".join([Path(sys.argv[0]).name] + list(argv or sys.argv[1:])),
                "versions": common.versions(exe), "data_source": LABEL,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "elapsed_s": time.time() - t0, "load_average_start": load0,
                "load_average_end": os.getloadavg(), "workers": args.workers,
                "cpu_count": os.cpu_count(), "chains_done": len(done),
                "chains_total": len(jobs), "config": cfg}

    def flush() -> None:
        rows, trows = [], []
        for recs in done.values():
            for rec in recs:
                rec = dict(rec)
                trows += rec.pop("_tools", [])
                rows.append(rec)
        write_tables(out, cfg, pd.DataFrame(rows), pd.DataFrame(trows), meta())

    def one(job):
        p, s = job
        recs = run_chain(cfg, p, s, work, exe)
        with lock:
            done[(p.point_id, s)] = recs
            try:
                flush()
            except Exception as exc:  # the tables are rewritten at the end anyway
                log(f"tables not written: {type(exc).__name__}: {exc}")
        return recs

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, jobs))
    flush()
    log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
