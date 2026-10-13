#!/usr/bin/env python3
"""The headline method (SparLab simulation): two-tool forming with an edge
pass holds the clamp-side band, and the ML surrogate + optimiser compensates
the smooth deep-region error the edge pass leaves. (In the code "DSIF" is the
two-tool process and the "rim pass" its edge pass.)

  1. data: 12 TRAIN-split parts of the stage_n26 design (3 per family,
     `select_parts`), the UNCOMPENSATED target of each simulated with exactly
     the rim-pass setup (`edge_setup(True)`:
     benchmarks/springback_fine's dsif strategy + the edge pass,
     `springback_fine` preset, penalty 10), 4 runs at a time, 1 thread each;
     one Sample per run (source "sim", fidelity "sparlab:dsif_rim") in its own
     Dataset, <work>/data_rim (never mixed into the plate data);
  2. model: `transfer_surrogate` - the stage_n26 MLP (loaded as the driver
     does), frozen, plus the default ("linear") TransferModel correction,
     fitted on the 8 FIT parts (2 per family); the other 4 (1 per family) are
     held out to score base vs transfer. No conformal calibration: the API
     refuses calibration parts the base was trained on (every TRAIN part is),
     and DA / the optimiser do not use intervals;
  3. combined compensation of pyramid-s2026-0001 with the rim-pass setup:
     surrogate DA on the deep region (the part minus the band the rim pass
     sweeps, `support.rim_band`; the band and the flange are held at the
     target), then `surrogate_optimize` from it, both with the transfer
     model; ONE verifying SparLab run with the rim-pass setup (penalty 10).

The verification starts as soon as the 8 fit parts are done; the 4 held-out
runs are only for scoring and run in the remaining slots.

usage (worktree root; OPENBLAS_NUM_THREADS=1 PYTHONPATH=python):
    python3 python/scripts/combined_rim_ml.py all      # steps 1-3, resumable
    COMBINED_PART=<test part> python3 python/scripts/combined_rim_ml.py one
                                                       # step 3 for another test part
    python3 python/scripts/combined_rim_ml.py status
    python3 python/scripts/combined_rim_ml.py report   # results.csv, README.md (one part)

The published 8-part result ran `all` once, then `one` for each of the 8
test parts (benchmarks/springback_fine_ml/combined_rim/run_all8.sh); its
table (combined_rim/all8_results.csv and the "All 8 test parts" section of
combined_rim/README.md) was assembled from the per-part verification records
in <work>/combined_rim/ by hand. `report` writes the one-part table only,
and rewrites combined_rim/README.md: keep the 8-part section when you run it.

Resumable: every SparLab run goes through the content-addressed run cache
(<work>/runs); each step writes a JSON in <work>/combined_rim/ and is skipped
when it exists (a recorded failure is not retried).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402  (deviation by region, physics checks, split, JSON)
import springback_fine_ml as fm  # noqa: E402  (the design, the split rules)
from common import LABEL, MM, REPO, log  # noqa: E402

OUT_ML = REPO / "benchmarks" / "springback_fine_ml"
WORK = OUT_ML / "work"
#: the training stage whose MLP ensemble is the base model
STAGE, N = "stage_n26", 26
OUT = OUT_ML / "combined_rim"
CW = WORK / "combined_rim"
DATA = WORK / "data_rim"
#: the test part (env COMBINED_PART selects another one, for `one`)
PART = os.environ.get("COMBINED_PART", "pyramid-s2026-0001")
FIDELITY = "sparlab:dsif_rim"
SLOTS = 4
COLS = ["vert_rms_mm", "upper_band_rms_mm", "upper_band_bias_mm", "deep_rms_mm",
        "deep_bias_mm"]
SEL_COLS = ["footprint_mm", "depth_mm", "max_wall_angle_deg"]


# ---------------------------------------------------------------------------
# configuration, setup and the base model
# ---------------------------------------------------------------------------
def cfgs():
    """(benchmarks/springback_fine_ml/config.json, benchmarks/springback/config.json)."""
    cfg = json.loads((OUT_ML / "config.json").read_text())
    old = json.loads((fm.OLD / "config.json").read_text())
    return cfg, old


def exe() -> str:
    return str((WORK / "bin" / "sparlab_form").resolve())


#: The edge pass: after forming, the support tool alone sweeps loops around
#: the part's outline from below (FormingSetup support_settings rim_pass).
EDGE_PASS = {"support": "dsif", "support_settings": {"rim_pass": True}}


def edge_setup(rim: bool):
    """benchmarks/springback_fine's dsif strategy (two-tool forming), with
    the edge pass (`EDGE_PASS`) when `rim`."""
    from precomp.fea import FormingSetup

    fcfg = json.loads((fm.FINE / "config.json").read_text())
    over = copy.deepcopy(fcfg.get("setup_overrides", {}))
    over.update(copy.deepcopy(fcfg["strategies"]["dsif"]))
    if rim:
        over.update(copy.deepcopy(EDGE_PASS))
    return FormingSetup.preset(fcfg["preset"], executable=exe(), timeout=fcfg["timeout_s"],
                               **over)


def load_mlp(cfg, old, pts):
    """The stage_n26 MLP ensemble exactly as springback_fine_ml.py loads it
    (`train_or_load` with the stage's train / calibration samples: a load,
    never a retrain)."""
    from precomp.ml import Dataset

    ds = Dataset.create(WORK / "data", created_at="", exist_ok=True)
    spl = fm.cfg_split(cfg)
    pids = {p.point_id for p in pts}
    by: Dict[str, List[Any]] = {"train": [], "calibration": []}
    for s in ds:
        if s.part_id in pids:
            by.setdefault(common.split_of(s.part_id, spl), []).append(s)
    bundle = WORK / "models" / STAGE / "mlp"
    sur, dt = fm.train_or_load("mlp", by["train"], by["calibration"], cfg, bundle, 1)
    if dt is not None:
        raise SystemExit("the MLP was retrained: the bundle did not match the stage")
    return sur


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def select_parts() -> pd.DataFrame:
    """3 TRAIN parts per family of stage_n26 (the driver's split): among the
    parts whose penalty-3 data run took at most the family's median (a cost
    cap - the rim-pass runs at penalty 10 take ~3x as long), the 3 farthest
    apart in standardised (footprint, depth, wall angle) - the triple with
    the largest smallest pairwise distance (ties: largest sum). Of the three,
    the one closest to their centroid is HELD OUT, the other two FIT the
    correction."""
    f = CW / "selection.csv"
    if f.exists():
        return pd.read_csv(f)
    d = pd.read_csv(OUT_ML / STAGE / "design.csv")
    st = pd.read_csv(OUT_ML / "data_status.csv")[["point_id", "unc_runtime_s"]]
    cfg, _ = cfgs()
    spl = fm.cfg_split(cfg)
    d = d[[common.split_of(p, spl) == "train" for p in d.point_id]]
    assert (d.split == "train").all()
    m = d.merge(st, on="point_id")
    rows = []
    for fam, g in m.groupby("family"):
        z = (g[SEL_COLS] - g[SEL_COLS].mean()) / g[SEL_COLS].std()
        c = g[g.unc_runtime_s <= g.unc_runtime_s.median()]
        zc = z.loc[c.index].to_numpy()

        def score(t):
            ds = [np.linalg.norm(zc[a] - zc[b]) for a, b in itertools.combinations(t, 2)]
            return (min(ds), sum(ds))

        best = max(itertools.combinations(range(len(c)), 3), key=score)
        zz = zc[list(best)]
        hold = int(np.argmin(np.linalg.norm(zz - zz.mean(0), axis=1)))
        for i, k in enumerate(best):
            r = c.iloc[k]
            rows.append({"point_id": r.point_id, "family": fam, "split": "train",
                         "role": "holdout" if i == hold else "fit",
                         **{col: r[col] for col in SEL_COLS},
                         "data_runtime_p3_s": r.unc_runtime_s})
    sel = pd.DataFrame(rows).sort_values(["family", "point_id"]).reset_index(drop=True)
    CW.mkdir(parents=True, exist_ok=True)
    sel.to_csv(f, index=False, float_format="%.6g")
    return sel


# ---------------------------------------------------------------------------
# context (loaded once per process)
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self) -> None:
        self.cfg, self.old = cfgs()
        self.points = {p.point_id: p for p in fm.design(self.cfg, self.old, N, exe())}
        self.setup = edge_setup(True)
        self._mlp = None
        self.lock = threading.Lock()

    def mlp(self):
        with self.lock:
            if self._mlp is None:
                self._mlp = load_mlp(self.cfg, self.old, list(self.points.values()))
            return self._mlp

    def dataset(self):
        from precomp.ml import Dataset

        return Dataset.create(DATA, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              exist_ok=True,
                              description="fine-mesh springback, DSIF + rim pass, penalty 10 "
                                          "(springback_fine), uncompensated targets - "
                                          "SparLab simulation")


def sample_id(pid: str) -> str:
    return f"{pid}-rimu"


# ---------------------------------------------------------------------------
# step 1: one rim-pass run of an uncompensated training target
# ---------------------------------------------------------------------------
def data_run(ctx: Ctx, pid: str, role: str) -> Dict[str, Any]:
    from precomp.fea.runner import FormingError, simulate
    from precomp.ml import Sample

    f = CW / "runs" / f"{pid}.json"
    if f.exists():
        return json.loads(f.read_text())
    p = ctx.points[pid]
    target = p.target()
    rec: Dict[str, Any] = {"point_id": pid, "family": p.family, "role": role,
                           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.perf_counter()
    log(f"data {pid} ({role}): start")
    try:
        res = simulate(ctx.setup, target, WORK / "runs", target=target)
        formed = res.formed_surface(-1, grid=target.grid)
        prov = res.provenance
        rec.update(ok=True, deck_hash=prov.get("key"), cache_hit=prov.get("cache_hit"))
        rec.update(common.physics(res))
        rec.update(common.deviation(formed, target, ctx.cfg["upper_band_depth"]))
        s = Sample(sample_id(pid), target.copy(), formed, ctx.setup.to_dict(), "sim", FIDELITY,
                   "uncompensated", p.part.to_dict(), pid, target, None,
                   {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "deck_hash": prov.get("key"),
                    "sparlab_version": prov.get("sparlab_version"),
                    "runtime_s": rec.get("runtime_s"),
                    "note": "uncompensated target, DSIF + rim pass, penalty 10 "
                            "(combined_rim_ml.edge_setup(True))"})
        with ctx.lock:
            ds = ctx.dataset()
            if not ds.has(s.sample_id):
                ds.append(s)
    except FormingError as exc:
        r = {k: v for k, v in (exc.record or {}).items()
             if isinstance(v, (str, int, float, bool))}
        rec.update(ok=False, error=str(exc)[:2000], termination=r.get("termination")
                   or r.get("reason"), record=r)
        with ctx.lock:
            ctx.dataset().record_failure({"sample_id": sample_id(pid), "point_id": pid,
                                          "kind": "uncompensated",
                                          "reason": str(exc).splitlines()[0][:500],
                                          "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                      time.gmtime())})
    rec["wall_s"] = time.perf_counter() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    common.jdump(f, rec)
    log(f"data {pid}: {'ok' if rec['ok'] else 'FAILED ' + str(rec.get('termination'))}"
        f" [{rec['wall_s']:.0f} s] vert RMS {rec.get('vert_rms_mm', float('nan')):.3f} mm")
    return rec


# ---------------------------------------------------------------------------
# step 2: the transfer model
# ---------------------------------------------------------------------------
def rim_samples(ctx: Ctx, pids: List[str]) -> List[Any]:
    ds = ctx.dataset()
    return [ds.load(sample_id(p)) for p in pids if ds.has(sample_id(p))]


def transfer_model(ctx: Ctx, sel: pd.DataFrame):
    from precomp.ml import load_model, save_model
    from precomp.ml.surrogate import transfer_surrogate

    fit = rim_samples(ctx, list(sel[sel.role == "fit"].point_id))
    bundle = CW / "model" / "transfer_mlp"
    ids = sorted(s.sample_id for s in fit)
    if (bundle / "manifest.json").exists():
        sur = load_model(bundle)
        if sorted(sur.training["transfer"]["real_sample_ids"]) == ids:
            return sur
    t0 = time.perf_counter()
    sur = transfer_surrogate(ctx.mlp(), fit, correction="linear",
                             points_per_sample=ctx.cfg["surrogate"]["points_per_sample"],
                             seed=ctx.cfg["seed"])
    save_model(sur, bundle, {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "notes": "stage_n26 MLP (backing plate, penalty 3) + linear "
                                      "TransferModel correction on DSIF + rim-pass runs "
                                      "(penalty 10) - SparLab simulation",
                             "fit_time_s": time.perf_counter() - t0}, overwrite=True)
    v = sur.training["transfer"]["validation"]
    log(f"transfer: {len(fit)} fit parts, accepted {sur.training['transfer']['accepted']}, "
        f"LOPO base {1e3 * v.get('lopo_base_rms_m', np.nan):.4f} mm, corrected "
        f"{1e3 * v.get('lopo_corrected_rms_m', np.nan):.4f} mm, theta {v.get('theta')}")
    return sur


# ---------------------------------------------------------------------------
# step 3: the combined compensation and its verification
# ---------------------------------------------------------------------------
def masks(ctx: Ctx, target):
    """hold = everything but the deep region; adjust = the deep region (the
    part minus the band the rim pass sweeps)."""
    from precomp.fea.support import rim_band

    band, strip = rim_band(ctx.setup, target)
    part = target.mask & (target.z < -1e-6)
    deep = part & ~band
    return target.mask & ~deep, deep, band, strip


def combined(ctx: Ctx, sel: pd.DataFrame, max_time: float) -> Dict[str, Any]:
    from precomp._util import PrecompError
    from precomp.fea.support import command_upper_bound
    from precomp.geometry.heightmap import HeightMap
    from precomp.ml import surrogate_compensate
    from precomp.ml.compensate import surrogate_optimize

    fpred = CW / f"{PART}-combined.json"
    fcmd = CW / f"{PART}-combined.npz"
    if fpred.exists() and fcmd.exists():
        info = json.loads(fpred.read_text())
        comp = HeightMap.load(fcmd)
    else:
        sur = transfer_model(ctx, sel)
        p = ctx.points[PART]
        target = p.target()
        hold, adjust, band, strip = masks(ctx, target)
        sc = ctx.cfg["surrogate"]
        da = ctx.cfg["da"]
        kw = dict(max_wall_angle_deg=da["max_wall_angle_deg"],
                  upper_bound=command_upper_bound(ctx.setup, target),
                  hold_mask=hold, adjust_mask=adjust, tool_radius=ctx.setup.tool_radius)
        override = False
        t0 = time.perf_counter()
        try:
            dres = surrogate_compensate(target, ctx.setup, sur, iterations=sc["iterations"],
                                        stagnation=sc["stagnation"], alpha=da["alpha"],
                                        direction=da["direction"], smoothing=da["smoothing"],
                                        **kw)
        except PrecompError as exc:
            if "envelope" not in str(exc):
                raise
            override = True
            dres = surrogate_compensate(target, ctx.setup, sur, iterations=sc["iterations"],
                                        stagnation=sc["stagnation"], alpha=da["alpha"],
                                        direction=da["direction"], smoothing=da["smoothing"],
                                        allow_out_of_envelope=True, **kw)
        t_da = time.perf_counter() - t0
        dres.compensated.save(CW / f"{PART}-combined-da.npz")
        log(f"combined DA: predicted RMS {dres.predicted_metrics()['rms'] / MM:.4f} mm "
            f"({dres.stopped}, best {dres.best_iteration}) [{t_da:.0f} s]")
        t0 = time.perf_counter()
        ores = surrogate_optimize(target, ctx.setup, sur, start=dres, max_time_s=max_time,
                                  allow_out_of_envelope=override, **kw)
        t_opt = time.perf_counter() - t0
        comp = ores.compensated
        ores.save(CW / f"{PART}-combined-optimizer")

        def pred(res):
            d = common.deviation(res.predicted, target, ctx.cfg["upper_band_depth"])
            return {k: d[k] for k in COLS}

        v = sur.training["transfer"]
        info = {"point_id": PART, "data_source": "transfer-model prediction (stage_n26 MLP "
                "on plate data, penalty 3, + linear correction on DSIF rim-pass runs, "
                "penalty 10; SparLab simulation)",
                "setup": "combined_rim_ml.edge_setup(True) (dsif + rim pass, penalty 10)",
                "transfer_accepted": v["accepted"], "transfer_validation": v["validation"],
                "envelope_override": override,
                "masks": {"band_nodes": int(band.sum()), "strip_nodes": int(strip.sum()),
                          "adjust_nodes": int(adjust.sum()), "hold_nodes": int(hold.sum())},
                "target_predicted": None,
                "da": {**pred(dres), "stopped": dres.stopped, "best_iteration":
                       dres.best_iteration, "time_s": t_da,
                       "commanded_depth_mm": dres.compensated.depth / MM},
                "optimizer": {**pred(ores), "stopped": ores.stopped, "time_s": t_opt,
                              "evaluations": len(ores.history),
                              "in_envelope": ores.in_envelope,
                              "stop_reason": ores.history[-1].get("stop_reason")
                              if ores.history else None,
                              "commanded_depth_mm": comp.depth / MM,
                              "max_abs_change_from_da_mm":
                                  float(np.abs(comp.z - dres.compensated.z).max()) / MM}}
        # the transfer model's prediction for the UNCOMPENSATED target (rim pass)
        mu, _ = sur.predict_deviation(target, ctx.setup)
        info["target_predicted"] = {k: v for k, v in common.deviation(
            target.with_z(target.z + mu), target, ctx.cfg["upper_band_depth"]).items()
            if k in COLS}
        comp.save(fcmd)
        common.jdump(fpred, info)
        log("combined prediction: " + json.dumps({k: info[k] for k in ("da", "optimizer")},
                                                 default=str))
    return verify(ctx, comp)


def verify(ctx: Ctx, comp) -> Dict[str, Any]:
    from precomp.fea.runner import FormingError, simulate

    f = CW / f"{PART}-verify.json"
    if f.exists():
        return json.loads(f.read_text())
    target = ctx.points[PART].target()
    rec: Dict[str, Any] = {"case": "combined", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log(f"verify {PART}: start (dsif + rim pass, penalty 10)")
    t0 = time.perf_counter()
    try:
        res = simulate(ctx.setup, comp, WORK / "runs", target=target)
        formed = res.formed_surface(-1, grid=target.grid)
        rec.update(ok=True, deck_hash=res.provenance.get("key"),
                   cache_hit=res.provenance.get("cache_hit"))
        rec.update(common.physics(res))
        rec.update(common.deviation(formed, target, ctx.cfg["upper_band_depth"]))
        formed.save(CW / f"{PART}-verify-formed.npz")
    except FormingError as exc:
        r = {k: v for k, v in (exc.record or {}).items()
             if isinstance(v, (str, int, float, bool))}
        rec.update(ok=False, error=str(exc)[:2000], termination=r.get("termination")
                   or r.get("reason"), record=r)
    rec["wall_s"] = time.perf_counter() - t0
    common.jdump(f, rec)
    log(f"verify {PART}: " + json.dumps({k: rec.get(k) for k in ["ok"] + COLS}, default=str))
    return rec


# ---------------------------------------------------------------------------
# evaluation: base vs transfer on the rim-pass data
# ---------------------------------------------------------------------------
def evaluate(ctx: Ctx, sel: pd.DataFrame) -> Dict[str, Any]:
    from precomp.ml import evaluate_surrogate
    from precomp.ml.models import TransferModel
    from precomp.ml.dataset import build_table

    f = CW / "evaluation.json"
    hold_ids = list(sel[sel.role == "holdout"].point_id)
    held = rim_samples(ctx, hold_ids)
    fit = rim_samples(ctx, list(sel[sel.role == "fit"].point_id))
    allr = fit + held
    key = sorted(s.sample_id for s in allr)
    if f.exists():
        old = json.loads(f.read_text())
        if old.get("samples") == key:
            return old
    base = ctx.mlp()
    tr = transfer_model(ctx, sel)
    out: Dict[str, Any] = {"samples": key,
                           "note": "the 12 parts are TRAIN parts of the base MLP (with the "
                                   "backing plate at penalty 3): 'held out' means held out "
                                   "from the transfer correction, not unseen geometry"}
    # held out from the correction (4 parts): base vs transfer
    if held:
        per = []
        for name, sur in (("base", base), ("transfer", tr)):
            ev = evaluate_surrogate(sur, held, split="held_out", assess=False,
                                    check_unseen=False)
            s = ev.summary()["held_out"]
            out[f"heldout_{name}"] = {"n_samples": s["n_samples"],
                                      "err_rms_mean_mm": 1e3 * s["err_rms_mean_m"],
                                      "dz_rms_mean_mm": 1e3 * s["dz_rms_mean_m"]}
            pp = ev.per_part.copy()
            pp.insert(0, "model", name)
            per.append(pp)
        pd.concat(per).to_csv(CW / "heldout_per_part.csv", index=False, float_format="%.6g")
    # leave-one-part-out over the 8 fit parts (TransferModel's own validation)
    v = tr.training["transfer"]["validation"]
    out["lopo_fit"] = {"n_parts": v.get("n_parts"),
                       "base_rms_mm": 1e3 * v["lopo_base_rms_m"] if "lopo_base_rms_m" in v
                       else None,
                       "transfer_rms_mm": 1e3 * v["lopo_corrected_rms_m"]
                       if "lopo_corrected_rms_m" in v else None,
                       "accepted": tr.training["transfer"]["accepted"],
                       "theta": v.get("theta"), "per_part": v.get("per_part")}
    # leave-one-part-out over all 12 (a separate fit, for the CV number only)
    if len(allr) > len(fit):
        tab = build_table(allr, base.features,
                          points_per_sample=ctx.cfg["surrogate"]["points_per_sample"],
                          mask=base.domain, rng=ctx.cfg["seed"])
        m = TransferModel(base.model, "linear", seed=ctx.cfg["seed"])
        m.fit(tab.X, tab.y, tab.part_id, feature_names=tab.feature_names)
        out["lopo_all"] = {"n_parts": m.validation.get("n_parts"),
                           "base_rms_mm": 1e3 * m.validation.get("lopo_base_rms_m", np.nan),
                           "transfer_rms_mm": 1e3 * m.validation.get("lopo_corrected_rms_m",
                                                                     np.nan),
                           "accepted": m.accepted, "theta": m.validation.get("theta"),
                           "per_part": m.validation.get("per_part")}
    common.jdump(f, out)
    log("evaluation: " + json.dumps({k: v for k, v in out.items()
                                     if k.startswith(("heldout", "lopo"))}, default=str)[:2000])
    return out


# ---------------------------------------------------------------------------
# the scheduler: 4 slots; fit runs, then the combined job, then held-out runs
# ---------------------------------------------------------------------------
def cmd_all(args) -> None:
    t_start = time.time()
    sel = select_parts()
    log("selection:\n" + sel.to_string(index=False))
    ctx = Ctx()
    fit = sel[sel.role == "fit"].sort_values("data_runtime_p3_s", ascending=False)
    hold = sel[sel.role == "holdout"].sort_values("data_runtime_p3_s", ascending=False)
    jobs: List[Dict[str, Any]] = (
        [{"name": f"data {r.point_id}", "kind": "data", "pid": r.point_id, "role": "fit"}
         for r in fit.itertuples()]
        + [{"name": "combined", "kind": "combined"}]
        + [{"name": f"data {r.point_id}", "kind": "data", "pid": r.point_id,
            "role": "holdout"} for r in hold.itertuples()])
    fit_names = {f"data {p}" for p in fit.point_id}
    done: set = set()
    running: set = set()
    cv = threading.Condition()

    def runnable(j):
        return j["kind"] != "combined" or fit_names <= done

    def worker(i: int) -> None:
        while True:
            with cv:
                while True:
                    todo = [j for j in jobs if j["name"] not in done | running]
                    if not todo:
                        return
                    nxt = next((j for j in todo if runnable(j)), None)
                    if nxt is not None:
                        running.add(nxt["name"])
                        break
                    cv.wait(30)
            try:
                if nxt["kind"] == "data":
                    data_run(ctx, nxt["pid"], nxt["role"])
                else:
                    combined(ctx, sel, args.max_time)
            except Exception as exc:  # logged; the others go on
                log(f"{nxt['name']}: ERROR {type(exc).__name__}: {exc}")
                traceback.print_exc()
            with cv:
                running.discard(nxt["name"])
                done.add(nxt["name"])
                cv.notify_all()

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(SLOTS)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    evaluate(ctx, sel)
    meta = {"finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "this_invocation_s": time.time() - t_start}
    common.jdump(CW / "all_done.json", meta)
    cmd_report(args)
    log("done")


def cmd_status(args) -> None:
    sel = select_parts()
    for r in sel.itertuples():
        f = CW / "runs" / f"{r.point_id}.json"
        x = json.loads(f.read_text()) if f.exists() else {}
        print(f"{r.point_id:28s} {r.role:8s} "
              + (("ok" if x.get("ok") else "FAILED") + f" {x.get('wall_s', 0):.0f} s"
                 if x else "pending"))
    for n in ("combined.json", "verify.json"):
        f = CW / f"{PART}-{n}"
        print(f"{f.name}: {'present' if f.exists() else 'pending'}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(args) -> None:
    sel = select_parts()
    rows: List[Dict[str, Any]] = []
    pf = CW / f"{PART}-combined.json"
    info = json.loads(pf.read_text()) if pf.exists() else {}
    if info:
        rows.append({"case": "DSIF + rim pass uncompensated", "kind": "predicted (transfer)",
                     "fe_runs": 0, **info["target_predicted"], "source": "this run"})
        rows.append({"case": "combined: rim pass + transfer DA", "kind": "predicted (transfer)",
                     "fe_runs": 0, **{c: info["da"][c] for c in COLS}, "source": "this run"})
        rows.append({"case": "combined: rim pass + transfer DA + optimiser",
                     "kind": "predicted (transfer)", "fe_runs": 0,
                     **{c: info["optimizer"][c] for c in COLS}, "source": "this run"})
    vf = CW / f"{PART}-verify.json"
    if vf.exists():
        x = json.loads(vf.read_text())
        rows.append({"case": "combined: rim pass + transfer DA + optimiser",
                     "kind": "simulated", "fe_runs": 1, **{c: x.get(c) for c in COLS},
                     "deck_hash": x.get("deck_hash"),
                     "note": "" if x.get("ok") else "FAILED: " + str(x.get("termination")
                                                                     or x.get("error"))[:300],
                     "source": "this run"})
    else:
        rows.append({"case": "combined: rim pass + transfer DA + optimiser",
                     "kind": "simulated", "fe_runs": 1, "note": "not run yet",
                     "source": "this run"})
    df = pd.DataFrame(rows)
    df.insert(0, "point_id", PART)
    df["data_source"] = LABEL
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "results.csv", index=False, float_format="%.4f")
    # training runs
    tr = []
    for r in sel.itertuples():
        f = CW / "runs" / f"{r.point_id}.json"
        x = json.loads(f.read_text()) if f.exists() else {}
        tr.append({"point_id": r.point_id, "family": r.family, "role": r.role,
                   **{c: getattr(r, c) for c in SEL_COLS},
                   "status": ("ok" if x.get("ok") else "FAILED") if x else "pending",
                   "termination": x.get("termination"), "runtime_s": x.get("runtime_s"),
                   **{c: x.get(c) for c in COLS}, "deck_hash": x.get("deck_hash")})
    trd = pd.DataFrame(tr)
    trd["data_source"] = LABEL
    trd.to_csv(OUT / "training_runs.csv", index=False, float_format="%.4f")
    ev = json.loads((CW / "evaluation.json").read_text()) if (CW / "evaluation.json").exists() \
        else {}
    if ev:
        common.jdump(OUT / "evaluation.json", ev)
    if info:
        common.jdump(OUT / "combined_prediction.json", info)

    def f3(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:+.3f}" if isinstance(v, float) and v < 0 else \
            f"{v:.3f}" if isinstance(v, float) else str(v)

    L = [f"# Combined compensation on {PART}: DSIF rim pass + transfer model + optimiser", "",
         f"**{LABEL}** (sparlab_form). **One test part: not statistically meaningful.** "
         "Predictions are the surrogate's (the stage_n26 MLP, or that MLP with the transfer "
         "correction), not results.", "",
         "All simulated rows at penalty 10 (`springback_fine` preset). Deviation formed - "
         "target, vertical, in mm; upper band = the top 1 mm of depth (next to the clamp), "
         "deep = the rest of the part.", "",
         "| case | kind | FE runs | vertical RMS | upper-band RMS | upper-band bias | "
         "deep RMS | deep bias | note |", "|---|---|--:|--:|--:|--:|--:|--:|---|"]
    for _, x in df.iterrows():
        note = x.get("note")
        L.append(f"| {x['case']} | {x['kind']} | {x['fe_runs']} | "
                 + " | ".join(f3(x.get(c)) for c in COLS)
                 + f" | {'' if pd.isna(note) else note} |")
    L += ["", "FE runs: SparLab runs of this part for that row (the combined row also "
          "needed the 12 rim-pass training runs below).", ""]
    if info:
        o = info["optimizer"]
        L += [f"Combined: surrogate DA ({info['da']['stopped']}, {info['da']['time_s']:.0f} s), "
              f"then the optimiser ({o['evaluations']} evaluation batches, {o['time_s']:.0f} s, "
              f"stop '{o.get('stop_reason')}', in envelope {o['in_envelope']}, largest change "
              f"from the DA command {o['max_abs_change_from_da_mm']:.3f} mm). Adjusted: the "
              f"deep region ({info['masks']['adjust_nodes']} nodes, the part minus the "
              f"{info['masks']['band_nodes']}-node band the rim pass sweeps); the band and "
              "the flange are held at the target. Envelope override: "
              f"{info['envelope_override']}.", ""]
    if ev:
        L += ["## Base vs transfer model on the rim-pass data", "",
              "dz prediction error (RMS over the part, mean over parts) of the stage_n26 MLP "
              "(base, backing plate, penalty 3) and of the base + linear TransferModel "
              "correction, on the rim-pass runs (penalty 10):", "",
              "| evaluation | parts | base [mm] | transfer [mm] |", "|---|--:|--:|--:|"]
        for k, lab in (("lopo_fit", "leave-one-part-out over the fit parts (the model used)"),
                       ("lopo_all", "leave-one-part-out, all 12 parts (separate fit)")):
            if k in ev:
                L.append(f"| {lab} | {ev[k]['n_parts']} | {f3(ev[k]['base_rms_mm'])} | "
                         f"{f3(ev[k]['transfer_rms_mm'])} |")
        if "heldout_base" in ev:
            L.append(f"| held out from the correction | {len(sel[sel.role == 'holdout'])} | "
                     f"{f3(ev['heldout_base']['err_rms_mean_mm'])} | "
                     f"{f3(ev['heldout_transfer']['err_rms_mean_mm'])} |")
            L += ["", f"(dz RMS of the held-out runs: "
                  f"{f3(ev['heldout_base']['dz_rms_mean_mm'])} mm.) "
                  f"Correction (offset a [m], gain b; y = a + (1 + b) mu): "
                  f"{ev['lopo_fit'].get('theta')}. {ev['note']}.", ""]
    L += ["## The 12 training parts (TRAIN split of stage_n26, 3 per family)", "",
          "Picked per family among the TRAIN parts whose penalty-3 data run took at most "
          "the family median (cost cap), as the 3 farthest apart in standardised "
          "(footprint, depth, wall angle); the one nearest their centroid is held out from "
          "the correction. Uncompensated target, DSIF + rim pass, penalty 10.", "",
          "| part | role | footprint | depth | wall angle | status | runtime [s] | "
          "vertical RMS | upper-band bias | deep bias |",
          "|---|---|--:|--:|--:|---|--:|--:|--:|--:|"]
    for _, r in trd.iterrows():
        st = r.status + (f" ({r.termination})" if r.status == "FAILED" else "")
        L.append(f"| {r.point_id} | {r.role} | {r.footprint_mm:.2f} | {r.depth_mm:.2f} | "
                 f"{r.max_wall_angle_deg:.1f} | {st} | {f3(r.runtime_s)} | "
                 f"{f3(r.vert_rms_mm)} | {f3(r.upper_band_bias_mm)} | {f3(r.deep_bias_mm)} |")
    L += ["", "Made by `python/scripts/combined_rim_ml.py` (rim-pass setup = "
          "`edge_setup(True)`: benchmarks/springback_fine's dsif strategy "
          "with the edge pass). Data set: "
          "benchmarks/springback_fine_ml/work/data_rim (kept apart from the plate data).", ""]
    (OUT / "README.md").write_text("\n".join(L))
    print(df.to_string(index=False))


def main() -> int:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PRECOMP_ML_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["all", "one", "select", "status", "report"])
    ap.add_argument("--max-time", type=float, default=240.0,
                    help="optimiser wall-time cap [s]")
    a = ap.parse_args()
    if a.what == "select":
        print(select_parts().to_string(index=False))
        return 0
    if a.what == "one":
        # the combined compensation + verification of PART only (the rim-pass
        # data and the transfer model must exist: run `all` first)
        rec = combined(Ctx(), select_parts(), a.max_time)
        log(f"one {PART}: " + json.dumps(rec, default=str))
        return 0
    {"all": cmd_all, "status": cmd_status, "report": cmd_report}[a.what](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
