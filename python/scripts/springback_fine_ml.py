#!/usr/bin/env python3
"""Surrogate pre-compensation on the fine mesh with a backing plate:
training data at penalty 3, the ML first shot verified at penalty 10 on the
fine benchmark's 8 test parts.

usage:
    OPENBLAS_NUM_THREADS=1 python3 python/scripts/springback_fine_ml.py \
        --out benchmarks/springback_fine_ml --work benchmarks/springback_fine_ml/work \
        --stages 18 26 [--workers 4] [--threads 4] [--data-only]

Every formed shape is a `sparlab_form` simulation (implicit, quasi-static,
finite_logarithmic kinematics, Hill48 AA5754-O with nominal handbook data):
**simulation, not experiment**.

Protocol (config.json, written from `PROTOCOL` when absent):
  * design: benchmarks/springback's (four families, its part bounds, Sobol
    seed 2026, 0.25 mm grid) and its nested split: indices 0-1 per family
    TEST (the fine benchmark's 8 parts), then every 4th CALIBRATION, the
    rest TRAIN; stage n<N> holds the first N points per family;
  * process: the backing plate (1 mm clearance to the target's outline);
    training and calibration data at the `springback_fine_bulk` preset
    (penalty 3), test runs at `springback_fine` (penalty 10);
  * data per train / calibration part: the target and one FE-DA step from
    its simulated part (vertical, alpha 1, no smoothing, 65 deg wall limit,
    the plate's masks and bound - the defaults - and the command clipped to
    the tool's reach), both simulated (`precomp.ml.generate`);
  * model: MLPEnsemble, five networks of 128 x 128 (57 point features, 2000
    nodes per sample, splits grouped by part, conformal calibration on the
    calibration parts, training envelope enforced);
  * ML first shot: `surrogate_compensate` (<= 8 predictions, stop at < 2 %
    predicted improvement, keep the best predicted iterate, reach clip) of
    each test target with the training setup (penalty 3: the setup the model
    knows; the penalty is not a feature), then ONE simulation of the
    compensated shape at penalty 10 - the part the method delivers; a target
    outside the envelope is compensated with the override and flagged;
  * baselines: the fine benchmark's plate runs of the same parts at penalty
    10 (uncompensated, FE-DA-1, FE-DA-2: benchmarks/springback_fine) and
    FE-DA-1 rebuilt on the fixed decks (<out>/fe_da_fixed, made by
    springback_fine_benchmark.py --tool-reach);
  * dz evaluation on the test parts' two penalty-10 runs (uncompensated and
    fixed FE-DA-1), queried with the training setup.

Resumable: simulations go through the content-addressed run cache in
`<work>/runs`, the data set skips the samples it has, model bundles are
reused when their training and calibration samples are the same, and a
compensated shape is kept in `<work>/ml/<stage>/<model>/`. Data runs are
made `--workers` at a time, one part's two runs in a row per worker.

Outputs in --out/stage_n<N>/: headline.csv, summary.csv, dz_per_part.csv,
ml_compensation_<model>.csv, design.csv, simulations.csv, failures.csv,
run.json, tables.md; --out/data_status.csv (every data part's runs).
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
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import _bootstrap  # noqa: F401  (puts python/ on sys.path)

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402  (deviation by region, split, versions, JSON)
from common import LABEL, MM, REPO, log  # noqa: E402

OLD = REPO / "benchmarks" / "springback"
FINE = REPO / "benchmarks" / "springback_fine"

PROTOCOL: Dict[str, Any] = {
    "protocol_revision": 1,
    "data_source": LABEL,
    "design_from": "benchmarks/springback/config.json (families, part bounds, seed, grid, "
                   "split)",
    "preset_data": "springback_fine_bulk",
    "preset_test": "springback_fine",
    "support": "backing_plate",
    "support_settings": {"clearance": 0.001},
    "train_kinds": ["uncompensated", "compensated"],
    "da": {"alpha": 1.0, "direction": "vertical", "smoothing": None,
           "max_wall_angle_deg": 65.0, "tool_reach": True},
    "surrogate": {"iterations": 8, "stagnation": 0.02, "points_per_sample": 2000,
                  "mlp": {"n_members": 5, "hidden": [128, 128], "epochs": 40}},
    "baseline": "benchmarks/springback_fine/cases.csv, strategy backing_plate (penalty 10)",
    "fe_da_fixed": "fe_da_fixed/cases.csv (penalty 10, fixed decks, reach clip)",
    "upper_band_depth": 0.001,
    "timeout_s": 8 * 3600.0,
}

METHOD_ORDER = ["uncompensated", "FE-DA-1", "FE-DA-2", "FE-DA-1 fixed", "ML-MLP"]
#: The model trained: the five-network MLP ensemble ("mlp" names its outputs:
#: ml_compensation_mlp.csv, method ML-MLP, <work>/models/<stage>/mlp).
MODELS = ("mlp",)


# ---------------------------------------------------------------------------
# setups and design
# ---------------------------------------------------------------------------
def make_setup(cfg: Dict[str, Any], preset: str, exe: str):
    from precomp.fea import FormingSetup

    return FormingSetup.preset(preset, executable=exe, timeout=cfg["timeout_s"], threads=1,
                               support=cfg["support"],
                               support_settings=copy.deepcopy(cfg["support_settings"]))


def design(cfg: Dict[str, Any], old_cfg: Dict[str, Any], n: int, exe: str):
    """The first `n` design points per family of benchmarks/springback's
    design, each with the data setup (penalty 3, plate)."""
    from precomp.materials import get_material
    from precomp.ml import DesignSpace, design_points

    base = make_setup(cfg, cfg["preset_data"], exe).to_dict()
    base["material"] = get_material(old_cfg["solver"]["setup"]["material"])
    space = DesignSpace(families=tuple(old_cfg["families"]), process={},
                        materials=(old_cfg["solver"]["setup"]["material"],), base_setup=base,
                        part_bounds={f: {k: tuple(v) for k, v in b.items()}
                                     for f, b in old_cfg["part_bounds"].items()},
                        grid_spacing=old_cfg["grid_spacing"])
    return design_points(space, n, old_cfg["seed"])


def da_options(cfg: Dict[str, Any]):
    """point -> the DA keywords of the compensated variant: the plate's masks
    and bound (`compensation_masks`, `command_upper_bound`) and the reach."""
    from precomp.fea.support import command_upper_bound, compensation_masks

    da = cfg["da"]

    def opts(point) -> Dict[str, Any]:
        target = point.target()
        hold, adjust = compensation_masks(point.setup, target)
        o: Dict[str, Any] = {"alpha": da["alpha"], "direction": da["direction"],
                             "smoothing": da["smoothing"],
                             "max_wall_angle_deg": da["max_wall_angle_deg"],
                             "upper_bound": command_upper_bound(point.setup, target),
                             "hold_mask": hold, "adjust_mask": adjust}
        if da.get("tool_reach"):
            o["tool_radius"] = point.setup.tool_radius
        return o

    return opts


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def generate_data(cfg, points, ds, work: Path, workers: int, out: Path) -> List[Dict[str, Any]]:
    """Both runs of every train / calibration point not yet in the data set:
    `workers` parts at a time, each worker its part's uncompensated run and
    then its FE-DA step through the run cache; the samples are then written
    by `generate` (a cache hit for both runs), one part at a time."""
    from precomp.ml import SparlabSimulator, generate
    from precomp.ml.generate import compensated_commanded, sim_job

    opts = da_options(cfg)
    kinds = tuple(cfg["train_kinds"])
    sim = SparlabSimulator(work / "runs", max_workers=1, executor="serial")
    lock = threading.Lock()
    status: List[Dict[str, Any]] = []
    have = set(ds.ids())
    todo = [p for p in points if common.split_of(p.point_id, cfg_split(cfg)) != "test"
            and any(f"{p.point_id}-{k[:4]}" not in have for k in kinds)]
    log(f"data: {len(todo)} parts to simulate ({2 * len(todo)} runs at most), "
        f"{workers} at a time")

    def one(point):
        try:
            return chain(point)
        except Exception as exc:  # logged; the other parts go on
            log(f"data {point.point_id}: ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None

    def chain(point):
        t0 = time.time()
        target = point.target()
        row: Dict[str, Any] = {"point_id": point.point_id,
                               "split": common.split_of(point.point_id, cfg_split(cfg))}
        oc0 = sim.run([sim_job(point.setup, target, target)])[0]
        row.update(unc_ok=oc0.ok, unc_hash=oc0.provenance.get("deck_hash"),
                   unc_runtime_s=oc0.provenance.get("runtime_s"),
                   unc_cache_hit=oc0.provenance.get("cache_hit"))
        if oc0.ok and "compensated" in kinds:
            try:
                cmd = compensated_commanded(target, oc0.formed, **opts(point))
                oc1 = sim.run([sim_job(point.setup, cmd, target)])[0]
                row.update(comp_ok=oc1.ok, comp_hash=oc1.provenance.get("deck_hash"),
                           comp_runtime_s=oc1.provenance.get("runtime_s"),
                           comp_cache_hit=oc1.provenance.get("cache_hit"),
                           comp_depth_mm=cmd.depth / MM,
                           comp_reach_raise_mm=cmd.metadata.get("tool_reach", {})
                           .get("max_raise_m", 0.0) / MM)
            except Exception as exc:  # recorded by generate below as well
                row["comp_error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
        with lock:
            rep = generate(ds, [point], sim, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                      time.gmtime()),
                           kinds=kinds, seed=cfg_split(cfg)["seed"], da_options=opts)
            row["created"] = len(rep.created)
            row["failed"] = "; ".join(f"{f['sample_id']}: {f['reason']}" for f in rep.failed)
            row["elapsed_s"] = time.time() - t0
            status.append(row)
            f = out / "data_status.csv"
            old = pd.read_csv(f) if f.exists() else pd.DataFrame()
            old = old[old["point_id"] != point.point_id] if len(old) else old
            pd.concat([old, pd.DataFrame([row])]).sort_values("point_id").to_csv(
                f, index=False, float_format="%.6g")
        log(f"data {point.point_id}: {row['created']} samples"
            + (f", FAILED {row['failed']}" if row["failed"] else "")
            + f" [{row['elapsed_s']:.0f} s; unc {row.get('unc_runtime_s') or 0:.0f} s"
            + f", comp {row.get('comp_runtime_s') or 0:.0f} s]")
        return row

    # in design order (a smaller stage's parts first), the largest parts of
    # each index first
    todo.sort(key=lambda p: (int(p.point_id.rsplit("-", 1)[1]), -p.part.footprint_radius()))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return status


def cfg_split(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The split rules and seed of benchmarks/springback (kept in our config)."""
    return cfg["split"]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def make_model(name: str, cfg: Dict[str, Any]):
    """The surrogate's model from the protocol: the MLP ensemble."""
    from precomp.ml import MLPEnsemble

    if name != "mlp":
        raise ValueError(name)
    m = cfg["surrogate"]["mlp"]
    return MLPEnsemble(m["n_members"], hidden=tuple(m["hidden"]), epochs=m["epochs"],
                       seed=cfg["seed"])


def train_or_load(name, train, cal, cfg, bundle: Path, threads: int):
    """Load the bundle when it was trained on exactly these training and
    calibration samples, else train, calibrate and save it. Returns (surrogate,
    training time [s] or None when loaded)."""
    from precomp.ml import FeatureConfig, load_model, save_model, train_surrogate

    ids = sorted(s.sample_id for s in train)
    cal_ids = sorted(s.sample_id for s in cal)
    if (bundle / "manifest.json").exists():
        sur = load_model(bundle)
        if (sorted(sur.training["sample_ids"]) == ids
                and sorted(sur.training["calibration_sample_ids"]) == cal_ids):
            log(f"{name}: loaded {bundle}")
            return sur, None
    t0 = time.perf_counter()
    sur = train_surrogate(make_model(name, cfg), train, calibration=cal or None,
                          features=FeatureConfig(),
                          points_per_sample=cfg["surrogate"]["points_per_sample"],
                          seed=cfg["seed"], n_jobs=threads)
    dt = time.perf_counter() - t0
    save_model(sur, bundle, {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "notes": "springback benchmark (SparLab simulation)",
                             "training_time_s": dt}, overwrite=True)
    log(f"{name}: trained on {len({s.part_id for s in train})} parts ({len(train)} samples), "
        f"calibrated on {len({s.part_id for s in cal})} parts [{dt:.0f} s]")
    return sur, dt


# ---------------------------------------------------------------------------
# test samples and baselines (penalty 10)
# ---------------------------------------------------------------------------
def baseline_rows(cfg: Dict[str, Any], out: Path) -> pd.DataFrame:
    fine = pd.read_csv(FINE / "cases.csv")
    fine = fine[(fine["strategy"] == "backing_plate")].copy()
    fine["source"] = "benchmarks/springback_fine"
    fixed_path = out / "fe_da_fixed" / "cases.csv"
    rows = [fine]
    if fixed_path.exists():
        fx = pd.read_csv(fixed_path)
        fx = fx[(fx["strategy"] == "backing_plate") & (fx["method"] == "FE-DA-1")].copy()
        fx["method"] = "FE-DA-1 fixed"
        fx["source"] = "benchmarks/springback_fine_ml/fe_da_fixed"
        rows.append(fx)
    return pd.concat(rows, ignore_index=True)


def test_samples(cfg, test_points, base: pd.DataFrame, work: Path) -> List[Any]:
    """The test parts' penalty-10 runs as samples (uncompensated, and the
    fixed-deck FE-DA-1 with its clipped command), with the training setup
    (the features do not see the penalty; the envelope would refuse it)."""
    from precomp.fea import load_result
    from precomp.ml import Sample
    from precomp.ml.generate import compensated_commanded

    opts = da_options(cfg)
    out = []
    for p in test_points:
        target = p.target()
        sel = base[(base["point_id"] == p.point_id) & base["completed"].fillna(False)]
        u = sel[sel["method"] == "uncompensated"]
        f = sel[sel["method"] == "FE-DA-1 fixed"]
        if not len(u):
            continue
        formed_u = load_result(run_dir(work, u["deck_hash"].iloc[0])).formed_surface(
            -1, grid=target.grid)
        items = [("unco", "uncompensated", target.copy(), formed_u, u["deck_hash"].iloc[0])]
        if len(f):
            cmd = compensated_commanded(target, formed_u, **opts(p))
            formed_c = load_result(run_dir(work, f["deck_hash"].iloc[0])).formed_surface(
                -1, grid=target.grid)
            items.append(("comp", "compensated", cmd, formed_c, f["deck_hash"].iloc[0]))
        for tag, kind, cmd, formed, h in items:
            out.append(Sample(f"{p.point_id}-{tag}", cmd, formed, p.setup.to_dict(), "sim",
                              "sparlab:penalty10-test", kind, p.part.to_dict(), p.point_id,
                              target, None,
                              {"created_at": "from the run cache", "deck_hash": h,
                               "sparlab_version": "sparlab 1.0.0 (bd26230795aa)",
                               "note": "formed at penalty 10 (springback_fine); setup "
                                       "recorded as the training setup"}))
    return out


def run_dir(work: Path, h: str) -> Path:
    return work / "runs" / "runs" / h[:2] / h / "output"


# ---------------------------------------------------------------------------
# models and the ML first shot
# ---------------------------------------------------------------------------
def ml_shot(name, sur, cfg, test_points, work: Path, stage: str, exe: str,
            workers: int) -> List[Dict[str, Any]]:
    """Surrogate DA of each test target (training setup), then one verifying
    run at the test setup (penalty 10)."""
    from precomp._util import PrecompError
    from precomp.geometry.heightmap import HeightMap
    from precomp.ml import SparlabSimulator, surrogate_compensate
    from precomp.ml.generate import sim_job

    sc = cfg["surrogate"]
    opts = da_options(cfg)
    root = work / "ml" / stage / name
    test_setup = make_setup(cfg, cfg["preset_test"], exe)
    rows, jobs = [], []
    for p in test_points:
        f, meta = root / f"{p.point_id}.npz", root / f"{p.point_id}.json"
        target = p.target()
        if f.exists() and meta.exists():
            comp, info = HeightMap.load(f), json.loads(meta.read_text())
        else:
            t0 = time.perf_counter()
            o = opts(p)
            kw = dict(iterations=sc["iterations"], stagnation=sc["stagnation"],
                      alpha=o["alpha"], direction=o["direction"],
                      smoothing=o["smoothing"], max_wall_angle_deg=o["max_wall_angle_deg"],
                      upper_bound=o["upper_bound"], hold_mask=o["hold_mask"],
                      adjust_mask=o["adjust_mask"], tool_radius=o.get("tool_radius"))
            override = False
            try:
                res = surrogate_compensate(target, p.setup, sur, **kw)
            except PrecompError as exc:
                if "envelope" not in str(exc):
                    raise
                override = True
                res = surrogate_compensate(target, p.setup, sur, allow_out_of_envelope=True,
                                           **kw)
            pm = res.predicted_metrics()
            info = {"point_id": p.point_id, "stopped": res.stopped,
                    "predictions": len([h for h in res.history if h.get("predicted")]),
                    "best_iteration": res.best_iteration,
                    "predicted_rms_mm": pm["rms"] / MM,
                    "interval_halfwidth_mm": (pm["interval_halfwidth_mean_m"] / MM
                                              if pm.get("interval_halfwidth_mean_m")
                                              is not None else None),
                    "target_in_envelope": bool(res.ood_target["in_envelope"])
                    if res.ood_target else None,
                    "compensated_in_envelope": bool(res.ood["in_envelope"])
                    if res.ood else None,
                    "envelope_override": override,
                    "target_ood_reasons": (res.ood_target or {}).get("reasons", [])[:3],
                    "history_predicted_rms_mm": [h["error"]["rms"] / MM for h in res.history
                                                 if h.get("predicted")],
                    "compensation_time_s": time.perf_counter() - t0}
            comp = res.compensated
            root.mkdir(parents=True, exist_ok=True)
            comp.save(f)
            common.jdump(meta, info)
        rows.append(info)
        jobs.append((p, comp))
    t0 = time.perf_counter()
    sim = SparlabSimulator(work / "runs", max_workers=workers, executor="process")
    outs = sim.run([sim_job(test_setup, c, p.target()) for p, c in jobs])
    for info, (p, c), oc in zip(rows, jobs, outs):
        info["commanded_depth_mm"] = c.depth / MM
        info["commanded_reach_raise_mm"] = c.metadata.get("tool_reach", {}).get(
            "max_raise_m", 0.0) / MM
        info["deck_hash"] = oc.provenance.get("deck_hash")
        if "runtime_s" not in info:
            info["runtime_s"] = oc.provenance.get("runtime_s")
            info["cache_hit"] = oc.provenance.get("cache_hit")
            if oc.ok:
                common.jdump(root / f"{p.point_id}.json", info)
        info["verify_ok"] = oc.ok
        if oc.ok:
            info.update(common.deviation(oc.formed, p.target(), cfg["upper_band_depth"]))
        else:
            info["error"] = (oc.error or "").splitlines()[0][:500]
    log(f"{name}: {len(jobs)} ML first shots verified at penalty 10 "
        f"[{time.perf_counter() - t0:.0f} s]")
    return rows


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


METRICS = ["vert_rms_mm", "vert_max_mm", "vert_bias_mm", "normal_rms_mm", "upper_band_rms_mm",
           "upper_band_bias_mm", "deep_rms_mm", "deep_bias_mm", "flange_rms_mm",
           "flange_bias_mm", "upper_band_share"]


def headline(test_points, base: pd.DataFrame, ml_rows: Dict[str, List[Dict[str, Any]]]
             ) -> pd.DataFrame:
    rows = []
    for p in test_points:
        b = {"point_id": p.point_id, "family": p.family,
             "footprint_mm": 2 * p.part.footprint_radius() / MM,
             "depth_mm": p.part.max_depth() / MM, "wall_deg": p.part.max_wall_angle_deg()}
        for _, r in base[base["point_id"] == p.point_id].iterrows():
            ok = bool(r.get("completed", False))
            rows.append({**b, "method": r["method"],
                         "fe_runs": 2 if r["method"] == "FE-DA-1 fixed" else int(r["fe_run"]),
                         **{m: r.get(m) for m in METRICS}, "deck_hash": r.get("deck_hash"),
                         "source": r["source"], "note": "" if ok else "FAILED"})
        for name, rs in ml_rows.items():
            info = next((x for x in rs if x["point_id"] == p.point_id), None)
            if info is None:
                continue
            note = []
            if info.get("envelope_override"):
                note.append("target outside the training envelope (override)")
            if info.get("best_iteration") == 0:
                note.append("the best predicted iterate is the target itself")
            if not info.get("verify_ok"):
                note.append("FAILED: " + str(info.get("error")))
            rows.append({**b, "method": f"ML-{name.upper()}", "fe_runs": 1,
                         **{m: info.get(m) for m in METRICS},
                         "predicted_rms_mm": info.get("predicted_rms_mm"),
                         "deck_hash": info.get("deck_hash"), "source": "this stage",
                         "note": "; ".join(note)})
    hd = pd.DataFrame(rows)
    hd["data_source"] = LABEL
    hd["o"] = hd["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return hd.sort_values(["point_id", "o"]).drop(columns="o").reset_index(drop=True)


def summarise(hd: pd.DataFrame) -> pd.DataFrame:
    ok = hd[hd["vert_rms_mm"].notna()]
    piv = ok.pivot_table(index="point_id", columns="method", values="vert_rms_mm")
    complete = piv.dropna()
    rows = []
    for m in [x for x in METHOD_ORDER if x in piv.columns]:
        v = complete[m]
        g = ok[(ok["method"] == m) & ok["point_id"].isin(complete.index)]
        r = {"method": m, "fe_runs_per_part": int(hd[hd["method"] == m]["fe_runs"].max()),
             "n_parts": int(len(v)),
             "n_failed": int(hd[(hd["method"] == m)]["note"].fillna("")
                             .str.contains("FAILED").sum()),
             "vert_rms_mean_mm": v.mean(), "vert_rms_median_mm": v.median(),
             **{f"{c}_mean": g[c].mean() for c in METRICS if c != "vert_rms_mm"}}
        if "uncompensated" in complete:
            r["rms_over_uncompensated_mean"] = (v / complete["uncompensated"]).mean()
            r["parts_better_than_uncompensated"] = int((v < complete["uncompensated"]).sum())
        if "FE-DA-1" in complete:
            r["parts_better_than_DA1"] = int((v < complete["FE-DA-1"]).sum())
            d = v - complete["FE-DA-1"]
            r["minus_DA1_mean_mm"], r["minus_DA1_std_mm"] = d.mean(), d.std()
        rows.append(r)
    sm = pd.DataFrame(rows)
    sm["data_source"] = LABEL
    return sm


def simulation_table(work: Path) -> pd.DataFrame:
    """Every run of the cache: runtime, increments, status, failure reason."""
    rows = []
    for d in sorted((work / "runs" / "runs").glob("??/*")):
        if not d.is_dir():
            continue
        done, failed = d / "COMPLETE", d / "FAILED"
        r: Dict[str, Any] = {"deck_hash": d.name,
                             "status": "complete" if done.exists() else
                             ("failed" if failed.exists() else "incomplete")}
        try:
            run = json.loads((d / "run.json").read_text())
            r["runtime_s"] = run.get("runtime_s")
            r["exit_code"] = run.get("exit_code", run.get("returncode"))
        except (OSError, ValueError):
            pass
        if failed.exists():
            try:
                fr = json.loads(failed.read_text())
                r["reason"] = str(fr.get("reason", fr))[:300]
            except ValueError:
                r["reason"] = failed.read_text()[:300]
        try:
            s = json.loads((d / "output" / "summary.json").read_text())
            r["increments"] = s["timing"]["increments"]
            r["newton_iterations"] = s["timing"]["iterations"]
            r["cuts"] = s["timing"]["cuts"]
            r["warnings"] = len(s.get("warnings", []))
            r["max_plastic_strain"] = max(st["max_plastic_strain"] for st in s["steps"])
        except (OSError, ValueError, KeyError):
            pass
        rows.append(r)
    return pd.DataFrame(rows)


def write_stage(out: Path, stage: str, cfg, points, test_points, base, ml_rows, dz_frames,
                dz_summary, ds, work: Path, meta: Dict[str, Any]) -> pd.DataFrame:
    sdir = out / stage
    sdir.mkdir(parents=True, exist_ok=True)
    spl = cfg_split(cfg)
    pd.DataFrame([{"point_id": p.point_id, "family": p.family,
                   "split": common.split_of(p.point_id, spl),
                   "footprint_mm": 2 * p.part.footprint_radius() / MM,
                   "depth_mm": p.part.max_depth() / MM,
                   "max_wall_angle_deg": p.part.max_wall_angle_deg(),
                   "params": json.dumps({k: v for k, v in p.part.to_dict().items()
                                         if k != "family"}, sort_keys=True)}
                  for p in points]).to_csv(sdir / "design.csv", index=False,
                                           float_format="%.4g")
    hd = headline(test_points, base, ml_rows)
    hd.to_csv(sdir / "headline.csv", index=False, float_format="%.4f")
    sm = summarise(hd)
    sm.to_csv(sdir / "summary.csv", index=False, float_format="%.4f")
    for name, rows in ml_rows.items():
        pd.DataFrame(rows).assign(model=name, data_source=LABEL).to_csv(
            sdir / f"ml_compensation_{name}.csv", index=False, float_format="%.5g")
    if dz_frames:
        pd.concat(dz_frames).to_csv(sdir / "dz_per_part.csv", index=False,
                                    float_format="%.6g")
    # the stage's runs: its data samples and the ML verifying runs
    pids = {p.point_id for p in points}
    idx = ds.index()
    idx = idx[idx["part_id"].isin(pids)] if len(idx) else idx
    hashes = {}
    for sid in idx["sample_id"] if len(idx) else []:
        doc = json.loads((ds.root / "samples" / f"{sid}.json").read_text())
        hashes[(doc.get("provenance") or {}).get("deck_hash")] = f"data {sid}"
    for name, rows in ml_rows.items():
        for x in rows:
            hashes.setdefault(x.get("deck_hash"), f"ML-{name.upper()} {x['point_id']}")
    fr = ds.failures()
    fails = fr[fr["point_id"].isin(pids)].to_dict("records") if len(fr) else []
    present = set(ds.ids())
    frows = [{"stage": "generate", "sample_id": f["sample_id"], "reason": f["reason"],
              "resolved_later": f["sample_id"] in present} for f in fails]
    for name, rows in ml_rows.items():
        for x in rows:
            if not x.get("verify_ok"):
                frows.append({"stage": f"verify {name}", "sample_id": x["point_id"],
                              "reason": x.get("error"), "resolved_later": False})
    pd.DataFrame(frows, columns=["stage", "sample_id", "reason", "resolved_later"]).to_csv(
        sdir / "failures.csv", index=False)
    sims = simulation_table(work)
    if len(sims):
        sims = sims[sims["deck_hash"].isin(set(hashes))].copy()
        sims["role"] = sims["deck_hash"].map(hashes)
    sims.to_csv(sdir / "simulations.csv", index=False, float_format="%.4g")
    okd = sims[(sims["status"] == "complete") & sims["role"].str.startswith("data")] \
        if len(sims) else sims
    meta = dict(meta)
    by_split = {s: int(sum(common.split_of(pid, spl) == s for pid in idx["part_id"].unique()))
                for s in ("train", "calibration", "test")} if len(idx) else {}
    meta["dataset"] = {"samples": int(len(idx)), "parts": int(idx["part_id"].nunique())
                       if len(idx) else 0, "by_split": by_split,
                       "generation_failures": len(fails),
                       "data_runs_complete": int(len(okd)),
                       "data_runtime_s_mean": float(okd["runtime_s"].mean()) if len(okd)
                       else None,
                       "data_runtime_s_total": float(okd["runtime_s"].sum()) if len(okd)
                       else None}
    meta["dz"] = dz_summary
    common.jdump(sdir / "run.json", meta)
    # markdown
    L = [f"# {stage}: SparLab simulations (not experiments)", "",
         f"Backing plate. {by_split.get('train', 0)} train / "
         f"{by_split.get('calibration', 0)} calibration parts, {len(idx)} samples at "
         "penalty 3 (springback_fine_bulk); 8 test parts verified at penalty 10 "
         "(springback_fine).", "",
         "| method | FE runs / part | parts | vertical RMS mean | median | max mean | "
         "rim sag | interior RMS (bias) | RMS / uncomp. | better than uncomp. | "
         "better than FE-DA-1 | minus FE-DA-1 (sd) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in sm.iterrows():
        L.append(f"| {r['method']} | {r['fe_runs_per_part']} | {r['n_parts']} | "
                 f"**{fmt(r['vert_rms_mean_mm'])}** | {fmt(r['vert_rms_median_mm'])} | "
                 f"{fmt(r['vert_max_mm_mean'])} | {fmt(r['upper_band_bias_mm_mean'])} | "
                 f"{fmt(r['deep_rms_mm_mean'])} ({fmt(r['deep_bias_mm_mean'], 2)}) | "
                 f"{fmt(r.get('rms_over_uncompensated_mean'), 2)} | "
                 f"{r.get('parts_better_than_uncompensated', '-')} | "
                 f"{r.get('parts_better_than_DA1', '-')} | "
                 f"{fmt(r.get('minus_DA1_mean_mm'))} ({fmt(r.get('minus_DA1_std_mm'))}) |")
    methods = list(sm["method"])
    L += ["", "Per part, vertical RMS [mm] (max |dev| in brackets):", "",
          "| part | " + " | ".join(methods) + " |", "|---|" + "--:|" * len(methods)]
    for pid in sorted(hd["point_id"].unique()):
        cells = []
        for m in methods:
            x = hd[(hd["point_id"] == pid) & (hd["method"] == m)]
            cells.append("-" if x.empty or pd.isna(x["vert_rms_mm"].iloc[0]) else
                         f"{x['vert_rms_mm'].iloc[0]:.3f} ({x['vert_max_mm'].iloc[0]:.2f})"
                         + (" *" if "envelope" in str(x["note"].iloc[0]) else "")
                         + (" =" if "target itself" in str(x["note"].iloc[0]) else ""))
        L.append(f"| {pid} | " + " | ".join(cells) + " |")
    L += ["", "`*` target outside the training envelope (override); `=` the surrogate kept "
          "the target (no compensation).", ""]
    if dz_summary:
        L += ["dz prediction error on the test samples (penalty-10 runs, uncompensated and "
              "fixed FE-DA-1):", "", "| model | samples | dz error RMS [mm] | dz RMS [mm] | "
              "relative | 90 % coverage | flagged by the envelope |",
              "|---|--:|--:|--:|--:|--:|--:|"]
        for name, s in dz_summary.items():
            t = s.get("test", {})
            L.append(f"| {name} | {t.get('n_samples')} | {fmt(1e3 * t['err_rms_mean_m'])} | "
                     f"{fmt(1e3 * t['dz_rms_mean_m'])} | {fmt(t.get('rel_rms_mean'), 2)} | "
                     f"{fmt(t.get('coverage_mean'), 2)} | "
                     f"{fmt(t.get('ood_flagged_share'), 2)} |")
    for name, rows in ml_rows.items():
        L += ["", f"ML-{name.upper()} compensation per test part:", "",
              "| part | stopped | predictions | best iterate | predicted RMS | verified RMS | "
              "commanded depth | reach raise | envelope (target / shot) |",
              "|---|---|--:|--:|--:|--:|--:|--:|---|"]
        for x in rows:
            L.append(f"| {x['point_id']} | {x['stopped']} | {x['predictions']} | "
                     f"{x['best_iteration']} | {fmt(x['predicted_rms_mm'])} | "
                     f"{fmt(x.get('vert_rms_mm'))} | {fmt(x.get('commanded_depth_mm'), 2)} | "
                     f"{fmt(x.get('commanded_reach_raise_mm'), 2)} | "
                     f"{x.get('target_in_envelope')} / {x.get('compensated_in_envelope')}"
                     + (" (override)" if x.get("envelope_override") else "") + " |")
    L += ["", "Failures:", ""]
    L += [f"* {f['stage']} {f['sample_id']}: {f['reason']}" for f in frows] or ["none"]
    (sdir / "tables.md").write_text("\n".join(L) + "\n")
    return sm


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True, help="run cache, data set, models (large)")
    ap.add_argument("--stages", type=int, nargs="+", required=True,
                    help="design points per family, one stage each, in order (e.g. 18 26)")
    ap.add_argument("--workers", type=int, default=4, help="simulations at a time")
    ap.add_argument("--threads", type=int, default=4, help="threads for training")
    ap.add_argument("--data-only", action="store_true", help="generate the data, no ML")
    ap.add_argument("--executable", help="sparlab_form (default: <work>/bin/sparlab_form, "
                                         "copied from build/bin when absent)")
    args = ap.parse_args(argv)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["PRECOMP_ML_THREADS"] = str(args.threads)
    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass
    from precomp.ml import Dataset, evaluate_surrogate

    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    old_cfg = json.loads((OLD / "config.json").read_text())
    cfg_path = out / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        cfg = copy.deepcopy(PROTOCOL)
        cfg["split"] = {k: old_cfg[k] for k in ("seed", "test_per_family",
                                                "calibration_every")}
        cfg["seed"] = old_cfg["seed"]          # the models' seed (the design's)
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    if args.executable:
        exe = str(Path(args.executable).resolve())
    else:
        exe_copy = work / "bin" / "sparlab_form"
        if not exe_copy.exists():
            exe_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / "build" / "bin" / "sparlab_form", exe_copy)
        exe = str(exe_copy.resolve())
    ds = Dataset.create(work / "data", created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                time.gmtime()),
                        exist_ok=True,
                        description="fine-mesh springback, backing plate, penalty 3 - "
                                    "SparLab simulation")
    for n in args.stages:
        t_start = time.time()
        stage = f"stage_n{n}"
        load0 = os.getloadavg()
        points = design(cfg, old_cfg, n, exe)
        spl = cfg_split(cfg)
        test_points = [p for p in points if common.split_of(p.point_id, spl) == "test"]
        log(f"{stage}: {len(points)} design points, {len(test_points)} test")
        t0 = time.time()
        generate_data(cfg, points, ds, work, args.workers, out)
        timings = {"data_s": time.time() - t0}
        if args.data_only:
            continue
        samples = {s.sample_id: s for s in ds}
        by_split: Dict[str, List[Any]] = {"train": [], "calibration": []}
        pids = {p.point_id for p in points}
        for s in samples.values():
            if s.part_id in pids:
                by_split.setdefault(common.split_of(s.part_id, spl), []).append(s)
        base = baseline_rows(cfg, out)
        tests = test_samples(cfg, test_points, base, work)
        ml_rows, dz_frames, dz_summary, train_times = {}, [], {}, {}
        for name in MODELS:
            t0 = time.time()
            sur, dt = train_or_load(name, by_split["train"], by_split["calibration"], cfg,
                                    work / "models" / stage / name, args.threads)
            train_times[name] = dt
            if tests:
                ev = evaluate_surrogate(sur, tests, split="test")
                f = ev.per_part.copy()
                f.insert(0, "model", name)
                dz_frames.append(f)
                dz_summary[name] = ev.summary()
                s = dz_summary[name]["test"]
                log(f"{name}: dz error on {s['n_samples']} test samples: "
                    f"{1e3 * s['err_rms_mean_m']:.3f} mm (dz {1e3 * s['dz_rms_mean_m']:.3f} mm)")
            ml_rows[name] = ml_shot(name, sur, cfg, test_points, work, stage, exe,
                                    args.workers)
            timings[f"{name}_s"] = time.time() - t0
        meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_source": LABEL, "stage": stage, "n_per_family": n,
                "command": "python3 python/scripts/springback_fine_ml.py "
                           + " ".join(argv if argv is not None else sys.argv[1:]),
                "versions": common.versions(exe),
                "workers": args.workers, "training_threads": args.threads,
                "cpu_count": os.cpu_count(), "load_average_start": load0,
                "load_average_end": os.getloadavg(), "timings_s": timings,
                "elapsed_s": time.time() - t_start, "training_time_s": train_times,
                "config": cfg}
        sm = write_stage(out, stage, cfg, points, test_points, base, ml_rows, dz_frames,
                         dz_summary, ds, work, meta)
        log("\n" + sm[["method", "n_parts", "vert_rms_mean_mm", "vert_rms_median_mm",
                       "parts_better_than_uncompensated"]].to_string(index=False))
    log("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
