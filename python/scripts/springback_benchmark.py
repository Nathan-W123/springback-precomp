#!/usr/bin/env python3
"""Springback pre-compensation benchmark on SparLab forming simulations:
uncompensated vs displacement adjustment with the FE model in the loop vs
a learned surrogate's first shot.

usage:
    python3 python/scripts/springback_benchmark.py --out benchmarks/springback \
        --work benchmarks/springback/work --n-per-family 6 [--models gbm mlp] \
        [--workers 2] [--config benchmarks/springback/config.json]

Every formed shape is a `sparlab_form` simulation (implicit, quasi-static,
finite_logarithmic kinematics, Hill48 AA5754-O with nominal handbook data):
**simulation, not experiment**. The script is resumable and idempotent:
simulations go through precomp's content-addressed run cache in
`<work>/runs` (an identical deck is never run twice), the data set skips the
samples it has, and model bundles are reused when their training parts are
the same. Re-running with a larger `--n-per-family` adds design points to the
same data set (the first n Sobol points of a family do not depend on n),
re-trains and re-evaluates; the test parts stay the same.

Protocol (fixed in config.json before the first run):
  * design: four families (truncated cone, pyramid, dome, elliptic cone),
    small parts (footprint 14-22 mm, 2-4 mm deep) on a 40 x 40 x 1 mm blank,
    one material and one process (config.json); a scrambled Sobol sequence
    per family (`precomp.ml.design_points`);
  * split by design-point index within each family, so it is stable as the
    design grows: the first `test_per_family` points are the TEST parts; of
    the others, every `calibration_every`-th is a CALIBRATION part (conformal
    intervals), the rest TRAIN;
  * data: train and calibration parts get the commanded variants of
    `train_kinds` (uncompensated and one FE displacement-adjustment step;
    optionally a perturbed one), all simulated (`precomp.ml.generate` with the real SparlabSimulator); test
    parts get the uncompensated and the one-step variant (for the dz
    evaluation - never for training);
  * methods on each test part, each formed shape simulated:
      - uncompensated: the target commanded as it is (1 FE run);
      - FE-DA-k: displacement adjustment with the FE model in the loop,
        alpha = 1, flange held, z <= 0, <= 65 deg (as `precomp compensate
        --method fea`): DA-1 is the part formed from the once-compensated
        shape (2 FE runs in total), DA-2 from the twice-compensated one (3);
      - ML (GBM, MLP): displacement adjustment on the surrogate
        (`precomp.ml.surrogate_compensate`, <= 8 predictions, stop at < 2 %
        predicted improvement) - no FE run - then ONE simulation of the
        compensated shape, which is the part the method delivers (1 FE run
        per part, plus the training data set, shared by all parts);
    a target outside the model's training envelope is compensated anyway
    (`allow_out_of_envelope`) and flagged in the tables;
  * metrics: vertical deviation of the released part from the target over
    the part (nodes where the target is below the sheet plane): RMS and max
    |.|, and the RMS of the normal deviation;
  * dz prediction error of each model on the test parts' simulated samples
    (`precomp.ml.evaluate_surrogate`).

The solver sits behind the `precomp.ml.Simulator` protocol (`run(jobs)`):
the forming model is `config["solver"]` - a simulator name and the
`FormingSetup` fields - and nothing else in this script knows about it, so
another step type (explicit forming) is a change of that block only.

Outputs in --out/stage_n<N>/: headline.csv (per test part and method),
summary.csv, regions.csv, dz_per_part.csv, ml_compensation_<model>.csv,
failures.csv, design.csv, simulations.csv, run.json and tables.md
(benchmarks/springback/README.md describes them).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import _bootstrap  # noqa: F401  (puts python/ on sys.path)

import numpy as np
import pandas as pd

MM = 1e-3
LABEL = "SparLab simulation"

DEFAULT_CONFIG: Dict[str, Any] = {
    "protocol_revision": 1,
    "seed": 2026,
    "families": ["truncated_cone", "pyramid", "dome", "elliptic_cone"],
    # part bounds [m, deg]: footprints 14-22 mm, 2-4 mm deep, walls 30-55 deg
    "part_bounds": {
        "truncated_cone": {"top_radius": [6.5e-3, 9.5e-3], "wall_angle_deg": [30.0, 55.0],
                           "depth": [2e-3, 4e-3], "top_fillet": [1e-3, 2.5e-3],
                           "bottom_fillet": [1e-3, 2.5e-3]},
        "pyramid": {"half_width_x": [6.5e-3, 8.5e-3], "half_width_y": [6.5e-3, 8.5e-3],
                    "wall_angle_deg": [30.0, 50.0], "depth": [2e-3, 3.5e-3],
                    "corner_radius": [3.5e-3, 6.5e-3], "top_fillet": [1e-3, 2e-3],
                    "bottom_fillet": [1e-3, 2e-3]},
        "dome": {"opening_radius": [7.5e-3, 10.5e-3], "depth": [2e-3, 4e-3],
                 "rim_fillet": [1.5e-3, 4e-3], "aspect": [0.75, 1.0]},
        "elliptic_cone": {"semi_axis_x": [7.5e-3, 10e-3], "semi_axis_y": [5.5e-3, 8e-3],
                          "wall_angle_deg": [30.0, 50.0], "depth": [2e-3, 3.5e-3],
                          "top_fillet": [1e-3, 2e-3], "bottom_fillet": [1e-3, 2e-3]},
    },
    "grid_spacing": 2.5e-4,
    # The forming model. Everything the solver needs is here; swapping the
    # solver (e.g. an explicit "form_explicit" step) changes this block only.
    "solver": {
        "simulator": "sparlab_form",
        "setup": {"material": "AA5754-O", "blank_size": 0.04, "clamp_margin": 0.005,
                  "thickness": 1e-3, "element_size": 2e-3, "layers": 2, "element": "hex8",
                  "tool_radius": 4e-3, "step_down": 1e-3, "friction": 0.1,
                  "toolpath_style": "spiral", "toolpath_spacing": 1e-3,
                  "max_tool_travel": 1e-3, "release": "321",
                  "kinematics": "finite_logarithmic", "threads": 1},
    },
    # perturbed variants: RMS 0.1-0.5 mm, correlation 4-12 mm (scaled to the
    # small parts; the springback here is ~0.5 mm RMS)
    "perturbation": {"amplitude": [1e-4, 5e-4], "correlation_length": [4e-3, 12e-3],
                     "taper": 1.0, "angle_margin_deg": 10.0},
    "test_per_family": 2,
    "calibration_every": 4,
    # commanded variants simulated per train / calibration part ("perturbed"
    # can be added later into the same data set); test parts always get
    # uncompensated + compensated (= the first FE-DA step)
    "train_kinds": ["uncompensated", "compensated"],
    "da_fe_runs": 3,
    "surrogate": {"iterations": 8, "stagnation": 0.02, "points_per_sample": 2000,
                  "gbm": {"n_members": 8, "max_iter": 300},
                  "mlp": {"n_members": 5, "hidden": [128, 128], "epochs": 40}},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# configuration: design, solver
# ---------------------------------------------------------------------------
def load_config(path: Optional[Path]) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None and path.exists():
        cfg.update(json.loads(path.read_text()))
    return cfg


def base_setup(cfg: Dict[str, Any]):
    """The FormingSetup fields of the solver block (material resolved)."""
    from precomp.materials import get_material

    s = dict(cfg["solver"]["setup"])
    s["material"] = get_material(s["material"])
    return s


def make_space(cfg: Dict[str, Any]):
    from precomp.ml import DesignSpace

    return DesignSpace(
        families=tuple(cfg["families"]), process={},
        materials=(cfg["solver"]["setup"]["material"],), base_setup=base_setup(cfg),
        part_bounds={f: {k: tuple(v) for k, v in b.items()}
                     for f, b in cfg["part_bounds"].items()},
        grid_spacing=cfg["grid_spacing"])


def make_simulator(cfg: Dict[str, Any], work: Path, workers: int):
    """The solver behind the `Simulator` protocol - the one place that knows
    which solver forms the parts."""
    name = cfg["solver"]["simulator"]
    if name == "sparlab_form":
        from precomp.ml import SparlabSimulator

        return SparlabSimulator(work / "runs", max_workers=workers,
                                executor="serial" if workers == 1 else "process")
    raise ValueError(f"unknown simulator {name!r}")


class SimulatorPredictor:
    """commanded -> formed by one run of any `Simulator` (for displacement
    adjustment with the FE model in the loop); keeps every outcome."""

    def __init__(self, simulator, setup):
        self.simulator = simulator
        self.setup = setup
        self.outcomes: List[Any] = []

    def __call__(self, commanded):
        from precomp._util import PrecompError

        oc = self.simulator.run([(self.setup, commanded)])[0]
        self.outcomes.append(oc)
        if not oc.ok:
            raise PrecompError(f"simulation failed: {oc.error}")
        return oc.formed


def split_of(point_id: str, cfg: Dict[str, Any]) -> str:
    i = int(point_id.rsplit("-", 1)[1])
    if i < cfg["test_per_family"]:
        return "test"
    j = i - cfg["test_per_family"]
    return "calibration" if j % cfg["calibration_every"] == cfg["calibration_every"] - 1 \
        else "train"


def deviation_metrics(formed, target) -> Dict[str, float]:
    from precomp.metrology import metrics, part_mask, signed_deviation, vertical_deviation

    part = part_mask(target)
    v = metrics(vertical_deviation(formed, target), part)
    n = metrics(signed_deviation(formed, target), part)
    return {"vert_rms_mm": v["rms"] / MM, "vert_max_mm": v["max_abs"] / MM,
            "vert_bias_mm": v["bias"] / MM, "normal_rms_mm": n["rms"] / MM,
            "normal_max_mm": n["max_abs"] / MM}


def jdump(path: Path, doc: Any) -> None:
    from precomp._util import to_jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(doc), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_generate(ds, points, cfg, sim, chunk: int) -> List[Dict[str, Any]]:
    """Simulate the variants of every point into the data set, a few points
    at a time (so samples land on disk as they finish)."""
    from precomp.ml import PerturbationSpec, generate

    pert = PerturbationSpec(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in cfg["perturbation"].items()})
    reports = []
    groups = {"test": ("uncompensated", "compensated"),
              "train": tuple(cfg["train_kinds"])}
    todo = [(p, groups["test" if split_of(p.point_id, cfg) == "test" else "train"])
            for p in points]
    # the test parts first: their DA and the baseline need them whatever the size
    todo.sort(key=lambda pk: (split_of(pk[0].point_id, cfg) != "test", pk[0].point_id))
    have = set(ds.ids())
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        for kinds in {k for _, k in part}:
            pts = [p for p, k in part if k == kinds]
            need = [p for p in pts if any(f"{p.point_id}-{k[:4]}" not in have for k in kinds)]
            if not need:
                continue
            t0 = time.perf_counter()
            rep = generate(ds, pts, sim, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                   time.gmtime()),
                           kinds=kinds, seed=cfg["seed"], perturbation=pert)
            reports.append(rep.summary())
            have = set(ds.ids())
            log(f"generate {', '.join(p.point_id for p in pts)}: {len(rep.created)} created, "
                f"{len(rep.skipped_existing)} present, {len(rep.failed)} failed "
                f"[{time.perf_counter() - t0:.0f} s]")
            for f in rep.failed:
                log(f"  FAILED {f['sample_id']}: {f['reason']}")
    return reports


def fe_da(point, cfg, work: Path) -> Dict[str, Any]:
    """Displacement adjustment with the FE model in the loop on one test
    part (`da_fe_runs` simulations; the first two are the data set's runs,
    fetched from the cache)."""
    from precomp.compensation import displacement_adjustment

    out = work / "da" / f"{point.point_id}.json"
    sim = make_simulator(cfg, work, 1)
    pred = SimulatorPredictor(sim, point.setup)
    target = point.target()
    rows: List[Dict[str, Any]] = []

    def cb(k, commanded, formed, error):
        oc = pred.outcomes[-1]
        rows.append({"fe_run": k + 1, "commanded_depth_mm": commanded.depth / MM,
                     **deviation_metrics(formed, target),
                     "deck_hash": oc.provenance.get("deck_hash"),
                     "runtime_s": oc.provenance.get("runtime_s"),
                     "cache_hit": oc.provenance.get("cache_hit")})

    rec: Dict[str, Any] = {"point_id": point.point_id, "iterations": rows}
    try:
        displacement_adjustment(target, pred, iterations=cfg["da_fe_runs"], alpha=1.0,
                                callback=cb)
        rec["ok"] = True
    except Exception as exc:  # recorded, reported
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:500]
    jdump(out, rec)
    return rec


def stage_da(test_points, cfg, work: Path, workers: int) -> Dict[str, Dict[str, Any]]:
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(lambda p: fe_da(p, cfg, work), test_points))
    for r in recs:
        it = r["iterations"]
        log(f"FE-DA {r['point_id']}: " + ", ".join(f"{x['vert_rms_mm']:.3f}" for x in it)
            + " mm RMS" + ("" if r["ok"] else f"; FAILED: {r['error']}"))
    log(f"FE-DA on {len(test_points)} test parts [{time.perf_counter() - t0:.0f} s]")
    return {r["point_id"]: r for r in recs}


def make_model(name: str, cfg: Dict[str, Any]):
    from precomp.ml import MLPEnsemble

    sc = cfg["surrogate"]
    if name == "mlp":
        m = sc["mlp"]
        return MLPEnsemble(m["n_members"], hidden=tuple(m["hidden"]), epochs=m["epochs"],
                           seed=cfg["seed"])
    raise ValueError(name)


def train_or_load(name, train, cal, cfg, bundle: Path, threads: int):
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


def ml_compensate(name, sur, test_points, cfg, work: Path, stage: str, sim):
    """Surrogate DA of every test target, then one verifying simulation each."""
    from precomp._util import PrecompError
    from precomp.geometry.heightmap import HeightMap
    from precomp.ml import surrogate_compensate

    sc = cfg["surrogate"]
    root = work / "ml" / stage / name
    rows, jobs = [], []
    for p in test_points:
        f = root / f"{p.point_id}.npz"
        meta = root / f"{p.point_id}.json"
        target = p.target()
        if f.exists() and meta.exists():
            comp, info = HeightMap.load(f), json.loads(meta.read_text())
        else:
            t0 = time.perf_counter()
            override = False
            try:
                res = surrogate_compensate(target, p.setup, sur, iterations=sc["iterations"],
                                           stagnation=sc["stagnation"])
            except PrecompError as exc:
                if "envelope" not in str(exc):
                    raise
                override = True
                res = surrogate_compensate(target, p.setup, sur, iterations=sc["iterations"],
                                           stagnation=sc["stagnation"],
                                           allow_out_of_envelope=True)
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
                    "compensation_time_s": time.perf_counter() - t0}
            comp = res.compensated
            root.mkdir(parents=True, exist_ok=True)
            comp.save(f)
            jdump(meta, info)
        rows.append(info)
        jobs.append((p, comp))
    t0 = time.perf_counter()
    outs = sim.run([(p.setup, c) for p, c in jobs])
    for info, (p, c), oc in zip(rows, jobs, outs):
        info["commanded_depth_mm"] = c.depth / MM
        info["deck_hash"] = oc.provenance.get("deck_hash")
        # the verifying run as first made (a rerun fetches it from the cache)
        if "runtime_s" not in info:
            info["runtime_s"] = oc.provenance.get("runtime_s")
            info["cache_hit"] = oc.provenance.get("cache_hit")
            if oc.ok:
                jdump(root / f"{p.point_id}.json", info)
        info["verify_ok"] = oc.ok
        if oc.ok:
            info.update(deviation_metrics(oc.formed, p.target()))
        else:
            info["error"] = (oc.error or "").splitlines()[0][:500]
    log(f"{name}: surrogate compensation of {len(jobs)} test targets verified "
        f"[{time.perf_counter() - t0:.0f} s]")
    return rows


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
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


def region_table(work: Path, points, cfg, da, ml_rows) -> pd.DataFrame:
    """Per test part and method, the RMS vertical deviation in three regions
    of the target: the upper band of the part (less than `band` deep - the
    rim, where the sheet sags between clamp and first contour and the
    command cannot rise above the sheet plane), the rest of the part, and
    the flange (outside the part; held at the target by the compensation).
    The formed surfaces are read back from the run cache by deck hash."""
    from precomp.fea import load_result
    from precomp.metrology import flange_mask, part_mask, vertical_deviation

    band = 1e-3
    rows = []
    for p in points:
        if split_of(p.point_id, cfg) != "test":
            continue
        target = p.target()
        part = part_mask(target)
        upper = part & (target.z > -band)
        regions = {"upper_band": upper, "deep": part & ~upper, "flange": flange_mask(target)}
        runs = []
        for it in da.get(p.point_id, {}).get("iterations", []):
            k = it["fe_run"] - 1
            runs.append(("uncompensated" if k == 0 else f"FE-DA-{k}", it.get("deck_hash")))
        for name, rs in ml_rows.items():
            info = next((x for x in rs if x["point_id"] == p.point_id), None)
            if info and info.get("verify_ok"):
                runs.append((f"ML-{name.upper()}", info.get("deck_hash")))
        for method, h in runs:
            if not h:
                continue
            d = work / "runs" / "runs" / h[:2] / h / "output"
            try:
                dev = vertical_deviation(load_result(d).formed_surface(-1, grid=target.grid),
                                         target)
            except Exception as exc:  # reported, not hidden
                rows.append({"point_id": p.point_id, "method": method, "error": str(exc)[:200]})
                continue
            r = {"point_id": p.point_id, "method": method}
            tot = float(np.sum(dev.z[part] ** 2))
            for reg, m in regions.items():
                sel = m & dev.mask
                r[f"{reg}_rms_mm"] = float(np.sqrt(np.mean(dev.z[sel] ** 2))) / MM
                r[f"{reg}_bias_mm"] = float(np.mean(dev.z[sel])) / MM
                r[f"{reg}_nodes"] = int(sel.sum())
            r["upper_band_share_of_part_sq_error"] = float(np.sum(dev.z[upper] ** 2)) / tot
            rows.append(r)
    return pd.DataFrame(rows)


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_tables(out: Path, stage: str, points, cfg, da, ml_rows, dz_frames, fails, meta,
                 ds, work: Path) -> pd.DataFrame:
    sdir = out / stage
    sdir.mkdir(parents=True, exist_ok=True)
    # design
    drows = []
    for p in points:
        drows.append({"point_id": p.point_id, "family": p.family,
                      "split": split_of(p.point_id, cfg),
                      "footprint_mm": 2 * p.part.footprint_radius() / MM,
                      "depth_mm": p.part.max_depth() / MM,
                      "max_wall_angle_deg": p.part.max_wall_angle_deg(),
                      "params": json.dumps({k: v for k, v in p.part.to_dict().items()
                                            if k != "family"}, sort_keys=True)})
    pd.DataFrame(drows).to_csv(sdir / "design.csv", index=False, float_format="%.4g")
    # headline: per test part and method
    test = [p for p in points if split_of(p.point_id, cfg) == "test"]
    head = []
    for p in test:
        base = {"point_id": p.point_id, "family": p.family,
                "footprint_mm": 2 * p.part.footprint_radius() / MM,
                "depth_mm": p.part.max_depth() / MM,
                "wall_deg": p.part.max_wall_angle_deg()}
        r = da.get(p.point_id, {"iterations": [], "ok": False, "error": "not run"})
        for it in r["iterations"]:
            k = it["fe_run"] - 1
            method = "uncompensated" if k == 0 else f"FE-DA-{k}"
            head.append({**base, "method": method, "fe_runs": it["fe_run"],
                         **{m: it[m] for m in ("vert_rms_mm", "vert_max_mm", "vert_bias_mm",
                                               "normal_rms_mm", "normal_max_mm")},
                         "note": ""})
        if not r["ok"]:
            head.append({**base, "method": f"FE-DA-{len(r['iterations'])}",
                         "fe_runs": len(r["iterations"]) + 1, "note": "FAILED: " + r["error"]})
        for name, rows in ml_rows.items():
            info = next((x for x in rows if x["point_id"] == p.point_id), None)
            if info is None:
                continue
            note = []
            if info.get("envelope_override"):
                note.append("target outside the training envelope (override)")
            if not info.get("verify_ok"):
                note.append("FAILED: " + info.get("error", "?"))
            head.append({**base, "method": f"ML-{name.upper()}", "fe_runs": 1,
                         **{m: info.get(m) for m in ("vert_rms_mm", "vert_max_mm",
                                                     "vert_bias_mm", "normal_rms_mm",
                                                     "normal_max_mm")},
                         "predicted_rms_mm": info.get("predicted_rms_mm"),
                         "note": "; ".join(note)})
    hd = pd.DataFrame(head)
    hd["data_source"] = LABEL
    hd.to_csv(sdir / "headline.csv", index=False, float_format="%.4f")
    # summary per method (over the test parts where every method has a result)
    piv = hd.pivot_table(index="point_id", columns="method", values="vert_rms_mm")
    complete = piv.dropna()
    summ = []
    for m in piv.columns:
        col = hd[hd["method"] == m]
        v = col.set_index("point_id").loc[complete.index] if len(complete) else col.iloc[:0]
        unc = complete["uncompensated"] if "uncompensated" in complete else None
        row = {"method": m, "n_parts": len(v), "fe_runs_per_part": int(col["fe_runs"].max()),
               "vert_rms_mean_mm": v["vert_rms_mm"].mean(),
               "vert_rms_median_mm": v["vert_rms_mm"].median(),
               "vert_max_mean_mm": v["vert_max_mm"].mean(),
               "normal_rms_mean_mm": v["normal_rms_mm"].mean(),
               "n_failed": int(col["note"].fillna("").str.startswith("FAILED").sum())}
        if unc is not None and len(v):
            ratio = v["vert_rms_mm"] / unc
            row["rms_over_uncompensated_mean"] = ratio.mean()
            row["parts_better_than_uncompensated"] = int((ratio < 1).sum())
        if "FE-DA-1" in complete and len(v):
            row["parts_better_than_DA1"] = int((v["vert_rms_mm"] < complete["FE-DA-1"]).sum())
        summ.append(row)
    order = {"uncompensated": 0, "FE-DA-1": 1, "FE-DA-2": 2, "FE-DA-3": 3}
    sm = pd.DataFrame(summ)
    sm["o"] = sm["method"].map(lambda m: order.get(m, 10))
    sm = sm.sort_values(["o", "method"]).drop(columns="o")
    sm["data_source"] = LABEL
    sm.to_csv(sdir / "summary.csv", index=False, float_format="%.4f")
    for name, rows in ml_rows.items():
        pd.DataFrame(rows).assign(model=name, data_source=LABEL).to_csv(
            sdir / f"ml_compensation_{name}.csv", index=False, float_format="%.5g")
    if dz_frames:
        pd.concat(dz_frames).to_csv(sdir / "dz_per_part.csv", index=False, float_format="%.6g")
    # failures: generation (data set), FE-DA, verification
    frows = [{"stage": "generate", "sample_id": f["sample_id"], "reason": f["reason"],
              "resolved_later": f.get("resolved_later")} for f in fails]
    for r in da.values():
        if not r["ok"]:
            frows.append({"stage": "FE-DA", "sample_id": r["point_id"], "reason": r["error"]})
    for name, rows in ml_rows.items():
        for info in rows:
            if not info.get("verify_ok"):
                frows.append({"stage": f"verify {name}", "sample_id": info["point_id"],
                              "reason": info.get("error")})
    pd.DataFrame(frows, columns=["stage", "sample_id", "reason", "resolved_later"]).to_csv(
        sdir / "failures.csv", index=False)
    reg = region_table(work, points, cfg, da, ml_rows)
    reg["data_source"] = LABEL
    reg.to_csv(sdir / "regions.csv", index=False, float_format="%.4f")
    # the stage's own runs: its samples, FE-DA runs and verifying runs
    pids = {p.point_id for p in points}
    idx = ds.index()
    idx = idx[idx["part_id"].isin(pids)] if len(idx) else idx
    hashes = set()
    for sid in idx["sample_id"] if len(idx) else []:
        doc = json.loads((ds.root / "samples" / f"{sid}.json").read_text())
        hashes.add((doc.get("provenance") or {}).get("deck_hash"))
    for r in da.values():
        hashes.update(it.get("deck_hash") for it in r["iterations"])
    for rows in ml_rows.values():
        hashes.update(x.get("deck_hash") for x in rows)
    fail_hashes = {f.get("deck_hash") for f in fails}
    sims = simulation_table(work)
    cache_total = int(len(sims))
    if len(sims):
        sims = sims[sims["deck_hash"].isin(hashes | fail_hashes)]
    sims.to_csv(sdir / "simulations.csv", index=False, float_format="%.4g")
    meta = dict(meta)
    ok = sims[sims["status"] == "complete"] if len(sims) else sims
    meta["simulations"] = {
        "total": int(len(sims)), "complete": int(len(ok)),
        "failed": int((sims["status"] == "failed").sum()) if len(sims) else 0,
        "runtime_s_median": float(ok["runtime_s"].median()) if len(ok) else None,
        "runtime_s_mean": float(ok["runtime_s"].mean()) if len(ok) else None,
        "runtime_s_min": float(ok["runtime_s"].min()) if len(ok) else None,
        "runtime_s_max": float(ok["runtime_s"].max()) if len(ok) else None,
        "runtime_s_total": float(ok["runtime_s"].sum()) if len(ok) else None}
    meta["simulations"]["run_cache_total"] = cache_total
    meta["dataset"] = {"samples": int(len(idx)), "parts": int(idx["part_id"].nunique()),
                       "by_split": {s: int(sum(split_of(pid, cfg) == s
                                               for pid in idx["part_id"].unique()))
                                    for s in ("train", "calibration", "test")},
                       "generation_failures": len(fails)}
    prev = sdir / "run.json"
    if prev.exists():                 # a rerun: keep the record of the run that did the work
        old = json.loads(prev.read_text())
        reruns = old.pop("reruns", [])
        reruns.append({k: meta[k] for k in ("created_at", "timings_s", "load_average_start",
                                            "load_average_end", "command")})
        meta = {**meta, **{k: old[k] for k in ("created_at", "timings_s", "training_time_s",
                                               "load_average_start", "load_average_end")
                           if k in old}, "reruns": reruns}
    jdump(sdir / "run.json", meta)
    # markdown
    lines = [f"# {stage}: SparLab simulations (not experiments)", "",
             f"{meta['dataset']['by_split']['train']} train / "
             f"{meta['dataset']['by_split']['calibration']} calibration / "
             f"{meta['dataset']['by_split']['test']} test parts; "
             f"{meta['dataset']['samples']} samples; {meta['simulations']['complete']} "
             f"simulations complete, {meta['simulations']['failed']} failed.", "",
             "| Method | FE runs per part | n parts | vertical RMS mean [mm] | median [mm] | "
             "vertical max mean [mm] | normal RMS mean [mm] | RMS / uncompensated | "
             "better than uncompensated | better than DA-1 |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in sm.iterrows():
        lines.append(f"| {r['method']} | {r['fe_runs_per_part']} | {r['n_parts']} | "
                     f"{fmt(r['vert_rms_mean_mm'])} | {fmt(r['vert_rms_median_mm'])} | "
                     f"{fmt(r['vert_max_mean_mm'])} | {fmt(r['normal_rms_mean_mm'])} | "
                     f"{fmt(r.get('rms_over_uncompensated_mean'), 2)} | "
                     f"{r.get('parts_better_than_uncompensated', '-')} | "
                     f"{r.get('parts_better_than_DA1', '-')} |")
    lines += ["", "Per part, vertical RMS [mm] (max |dev| in brackets):", ""]
    methods = list(sm["method"])
    lines.append("| part | " + " | ".join(methods) + " |")
    lines.append("|---|" + "--:|" * len(methods))
    for pid in piv.index:
        cells = []
        for m in methods:
            x = hd[(hd["point_id"] == pid) & (hd["method"] == m)]
            cells.append("-" if x.empty or pd.isna(x["vert_rms_mm"].iloc[0]) else
                         f"{x['vert_rms_mm'].iloc[0]:.3f} ({x['vert_max_mm'].iloc[0]:.2f})"
                         + (" *" if "envelope" in str(x["note"].iloc[0]) else ""))
        lines.append(f"| {pid} | " + " | ".join(cells) + " |")
    lines += ["", "`*` target outside the model's training envelope (compensated with the "
              "override).", "",
              "Mean over the test parts of the RMS vertical deviation per region [mm] "
              "(upper band: part nodes less than 1 mm deep; share: of the squared error "
              "over the part):", "",
              "| Method | upper band RMS | bias | deep part RMS | bias | flange RMS | "
              "share of error in upper band |", "|---|--:|--:|--:|--:|--:|--:|"]
    if len(reg) and "upper_band_rms_mm" in reg:
        g = reg.groupby("method").mean(numeric_only=True)
        for m in methods:
            if m in g.index:
                r = g.loc[m]
                lines.append(f"| {m} | {fmt(r['upper_band_rms_mm'])} | "
                             f"{fmt(r['upper_band_bias_mm'])} | {fmt(r['deep_rms_mm'])} | "
                             f"{fmt(r['deep_bias_mm'])} | {fmt(r['flange_rms_mm'])} | "
                             f"{fmt(r['upper_band_share_of_part_sq_error'], 2)} |")
    (sdir / "tables.md").write_text("\n".join(lines) + "\n")
    return sm


# ---------------------------------------------------------------------------
def versions() -> Dict[str, Any]:
    import scipy
    import sklearn

    from precomp import __version__

    v = {"precomp": __version__, "python": platform.python_version(),
         "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
         "scikit-learn": sklearn.__version__, "platform": platform.platform()}
    try:
        import torch
        v["torch"] = torch.__version__
    except ImportError:
        v["torch"] = None
    try:
        import subprocess
        v["git_revision"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                           capture_output=True, text=True,
                                           cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        pass
    return v


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="where the tables go")
    ap.add_argument("--work", required=True, help="run cache, data set, models (large)")
    ap.add_argument("--config", help="protocol (default: <out>/config.json, written from "
                                     "the defaults when absent)")
    ap.add_argument("--n-per-family", type=int, required=True,
                    help="design points per family (test + calibration + train)")
    ap.add_argument("--models", nargs="*", default=["mlp"], choices=["mlp"])
    ap.add_argument("--workers", type=int, default=2, help="concurrent simulations")
    ap.add_argument("--threads", type=int, default=2, help="threads for training")
    ap.add_argument("--chunk", type=int, default=4, help="design points per generate call")
    ap.add_argument("--da-fe-runs", type=int,
                    help="FE runs of the FE-DA baseline (default: config da_fe_runs); 2 "
                         "costs nothing beyond the data set, 3 one more run per test part")
    ap.add_argument("--data-only", action="store_true",
                    help="generate the data and FE-DA runs, no training")
    args = ap.parse_args(argv)
    os.environ["PRECOMP_ML_THREADS"] = str(args.threads)
    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass

    from precomp.fea import sparlab_version
    from precomp.fea.setup import FormingSetup
    from precomp.ml import Dataset, design_points, evaluate_surrogate

    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config) if args.config else out / "config.json"
    cfg = load_config(cfg_path)
    if not cfg_path.exists():
        jdump(cfg_path, cfg)
    t_start = time.perf_counter()
    stage = f"stage_n{args.n_per_family}"
    timings: Dict[str, float] = {}
    load0 = list(os.getloadavg())

    space = make_space(cfg)
    points = design_points(space, args.n_per_family, cfg["seed"])
    test_points = [p for p in points if split_of(p.point_id, cfg) == "test"]
    sim = make_simulator(cfg, work, args.workers)
    ds = Dataset.create(work / "data", created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                time.gmtime()),
                        exist_ok=True, description="springback benchmark - SparLab simulation")
    log(f"{stage}: {len(points)} design points ({len(test_points)} test), "
        f"solver {cfg['solver']['simulator']}")

    t0 = time.perf_counter()
    stage_generate(ds, points, cfg, sim, args.chunk)
    timings["generate_s"] = time.perf_counter() - t0

    samples = {s.sample_id: s for s in ds}
    by_split: Dict[str, List[Any]] = {"train": [], "calibration": [], "test": []}
    point_ids = {p.point_id for p in points}
    for s in samples.values():
        if s.part_id in point_ids:
            by_split[split_of(s.part_id, cfg)].append(s)
    ml_rows: Dict[str, List[Dict[str, Any]]] = {}
    dz_frames = []
    train_times = {}
    if not args.data_only:
        for name in args.models:
            t0 = time.perf_counter()
            sur, dt = train_or_load(name, by_split["train"], by_split["calibration"], cfg,
                                    work / "models" / stage / name, args.threads)
            train_times[name] = dt
            ev = evaluate_surrogate(sur, by_split["test"], split="test")
            f = ev.per_part.copy()
            f.insert(0, "model", name)
            dz_frames.append(f)
            s = ev.summary()["test"]
            log(f"{name}: dz error on the test samples: RMS {1e3 * s['err_rms_mean_m']:.3f} mm "
                f"(dz RMS {1e3 * s['dz_rms_mean_m']:.3f} mm, relative "
                f"{s['rel_rms_mean']:.2f}); flagged {s['ood_flagged_share']:.2f}")
            ml_rows[name] = ml_compensate(name, sur, test_points, cfg, work, stage, sim)
            timings[f"{name}_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if args.da_fe_runs:
        cfg = {**cfg, "da_fe_runs": args.da_fe_runs}
    da = stage_da(test_points, cfg, work, args.workers)
    timings["fe_da_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_start
    fr = ds.failures()
    fails = fr.to_dict("records") if len(fr) else []
    present = set(ds.ids())
    for f in fails:                   # a failure a later run resolved stays listed, marked
        f["resolved_later"] = f["sample_id"] in present
    setup0 = FormingSetup.from_dict(base_setup(cfg))
    meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_source": LABEL, "stage": stage, "n_per_family": args.n_per_family,
            "command": "python3 python/scripts/springback_benchmark.py "
                       + " ".join(a for a in (argv if argv is not None else sys.argv[1:])),
            "versions": versions(),
            "sparlab_version": sparlab_version(setup0.resolved_executable()),
            "workers": args.workers, "training_threads": args.threads,
            "cpu_count": os.cpu_count(), "load_average_start": load0,
            "load_average_end": list(os.getloadavg()), "timings_s": timings,
            "training_time_s": train_times, "models": args.models,
            "config": cfg}
    if not args.data_only:
        sm = write_tables(out, stage, points, cfg, da, ml_rows, dz_frames, fails, meta, ds,
                          work)
        log("\n" + sm.to_string(index=False))
    log(f"done in {timings['total_s']:.0f} s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
