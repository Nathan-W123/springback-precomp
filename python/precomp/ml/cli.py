"""The machine-learning commands of `precomp` (dispatched here by precomp.cli).

    precomp dataset generate   design of experiments -> simulated data set
    precomp dataset info       what a data set holds (counts, failures)
    precomp train              fit a surrogate, calibrate it, save the bundle
    precomp evaluate           score a bundle on held-out parts (CSV tables)
    precomp active             rank candidate parts for the next simulations

`precomp compensate --method surrogate --model DIR [--verify-fea]` is part
of the main command and loads bundles through `precomp.ml.registry`.

Every table and summary names its data source. `created_at` defaults to the
current UTC time, read here - the command line is the caller that supplies it
to the library. Exit status 0, or 2 with the cause on stderr.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .._util import PrecompError, canonical_json, read_json, to_jsonable, write_json


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # raise instead of exiting the interpreter
        raise _UsageError(f"{self.prog}: {message}")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _assignments(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v.strip()
    return out


def _features(args: argparse.Namespace):
    from .features import FeatureConfig
    return FeatureConfig(time_source=args.time_source)


# -- dataset ------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> int:
    from .dataset import Dataset
    from .generate import (DesignSpace, ProxySimulator, SparlabSimulator, design_points,
                           generate)

    base = read_json(args.setup) if args.setup else {}
    base.pop("material", None)
    process = {}
    for item in args.process or []:
        k, v = item.split("=", 1)
        lo, hi = (float(x) for x in v.split(","))
        process[k] = (lo, hi)
    kw: Dict[str, Any] = {"grid_spacing": args.grid_spacing, "base_setup": base}
    if args.families:
        kw["families"] = tuple(args.families)
    if args.materials:
        kw["materials"] = tuple(args.materials)
    if args.process is not None:
        kw["process"] = process
    space = DesignSpace(**kw)
    points = design_points(space, args.n_per_family, args.seed)
    if args.simulator == "proxy":
        sim: Any = ProxySimulator()
        print("simulator: ProxySimulator - proxy, NOT physics; for tests and smoke runs only",
              file=sys.stderr)
    else:
        if not args.work_dir:
            raise ValueError("--simulator sparlab needs --work-dir (the run cache)")
        sim = SparlabSimulator(args.work_dir, max_workers=args.max_workers)
    created = args.created_at or _now()
    ds = Dataset.create(args.out, created_at=created, exist_ok=True,
                        description=f"design seed {args.seed}, {args.simulator}")
    write_json(Path(args.out) / f"design_seed{args.seed}.json",
               {"space": space.to_dict(), "seed": args.seed,
                "n_per_family": args.n_per_family, "simulator": args.simulator,
                "points": [p.describe() for p in points]})
    rep = generate(ds, points, sim, created_at=created, kinds=args.kinds, seed=args.seed)
    print(canonical_json(rep.summary()), end="")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .dataset import Dataset, source_label

    ds = Dataset(args.data)
    idx = ds.index()
    fail = ds.failures()
    doc = {"samples": int(len(idx)), "parts": int(idx["part_id"].nunique()) if len(idx) else 0,
           "data_source": source_label(idx["source"]) if len(idx) else None,
           "by_family": idx.groupby("family").size().to_dict() if len(idx) else {},
           "by_kind": idx.groupby("kind").size().to_dict() if len(idx) else {},
           "failures": int(len(fail)),
           "failure_reasons": fail["reason"].value_counts().to_dict() if len(fail) else {}}
    print(canonical_json(doc), end="")
    return 0


# -- train --------------------------------------------------------------------------
def _make_model(kind: str, params: Dict[str, Any], args: argparse.Namespace):
    from .models import FEAPrior, GBMEnsemble, MLPEnsemble, ResidualModel

    seed = args.seed
    if kind == "gbm":
        return GBMEnsemble(**{"seed": seed, **params})
    if kind == "mlp":
        return MLPEnsemble(**{"seed": seed, **params})
    if kind == "residual":
        if args.prior == "proxy":
            from .generate import ProxySimulator
            prior: Any = ProxySimulator()
            print("prior: ProxySimulator - proxy, NOT physics", file=sys.stderr)
        elif args.prior == "fea":
            if not args.prior_work_dir:
                raise ValueError("--prior fea needs --prior-work-dir")
            prior = FEAPrior(args.prior_work_dir, _assignments(args.prior_set))
        else:
            raise ValueError("--model residual needs --prior proxy|fea")
        return ResidualModel(prior, GBMEnsemble(**{"seed": seed, **params}))
    raise ValueError(f"unknown model {kind!r}")


def cmd_train(args: argparse.Namespace) -> int:
    from .dataset import Dataset, family_split, grouped_split
    from .evaluate import evaluate_surrogate
    from .registry import load_model, save_model
    from .surrogate import train_surrogate, transfer_surrogate

    ds = Dataset(args.data)
    idx = ds.index()
    if args.families:
        idx = idx[idx["family"].isin(args.families)]
    excluded: List[str] = []
    if args.holdout_family:
        keep, excluded = family_split(idx, args.holdout_family)
        idx = idx[idx["sample_id"].isin(keep)]
    if len(idx) == 0:
        raise ValueError("no samples selected")
    rest, test = grouped_split(idx, test_fraction=args.test_fraction, seed=args.seed) \
        if args.test_fraction > 0 else (list(idx["sample_id"]), [])
    rest_idx = idx[idx["sample_id"].isin(rest)]
    if args.calibration_fraction > 0:
        train_ids, cal_ids = grouped_split(rest_idx, test_fraction=args.calibration_fraction,
                                           seed=args.seed + 1)
    else:                                       # no intervals: said so by predict_interval
        train_ids, cal_ids = list(rest_idx["sample_id"]), []
    params = _assignments(args.param)
    features = _features(args)
    created = args.created_at or _now()
    if args.model == "transfer":
        if not args.base:
            raise ValueError("--model transfer needs --base MODEL_DIR (the simulation-trained "
                             "bundle)")
        base = load_model(args.base)
        extra = {k: v for k, v in params.items() if k != "correction"}
        sur = transfer_surrogate(base, ds.samples(train_ids), calibration=ds.samples(cal_ids),
                                 correction=params.get("correction", "linear"),
                                 points_per_sample=args.points_per_sample, seed=args.seed,
                                 **extra)
    else:
        model = _make_model(args.model, params, args)
        sur = train_surrogate(model, ds.samples(train_ids), calibration=ds.samples(cal_ids),
                              features=features, points_per_sample=args.points_per_sample,
                              seed=args.seed, n_jobs=args.n_jobs, group_by=args.group_by)
    metrics: Dict[str, Any] = {}
    if test:
        rep = evaluate_surrogate(sur, ds.samples(test), tolerance=args.tolerance,
                                 level=args.level, split="held_out")
        metrics["held_out"] = {**rep.summary()["held_out"], "data_source": rep.data_source}
    out = save_model(sur, args.out, {"created_at": created, "metrics": metrics,
                                     "notes": args.notes or "",
                                     "dataset": str(Path(args.data).resolve())},
                     overwrite=args.overwrite)
    write_json(out / "split.json", {"seed": args.seed, "by": "part", "train": train_ids,
                                    "calibration": cal_ids, "test": test,
                                    "excluded_family": args.holdout_family,
                                    "excluded": excluded})
    print(canonical_json({"model": str(out), "model_class": sur.model_class,
                          "data_source": sur.data_source, "n_train": len(train_ids),
                          "n_calibration": len(cal_ids), "n_test": len(test),
                          "metrics": metrics}), end="")
    return 0


# -- evaluate -----------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    from .dataset import Dataset
    from .evaluate import EvaluationReport, evaluate_surrogate
    from .registry import load_model
    from .surrogate import seen_ids

    sur = load_model(args.model)
    ds = Dataset(args.data)
    split_file = Path(args.model) / "split.json"
    if args.ids:
        ids = [line.strip() for line in Path(args.ids).read_text().splitlines()
               if line.strip()]
    elif split_file.is_file():
        ids = list(read_json(split_file)["test"])
    else:                                   # every sample of a part the model never saw
        seen_s, seen_p = seen_ids(sur.training)
        idx = ds.index()
        ids = [i for i, pid in zip(idx["sample_id"], idx["part_id"])
               if i not in seen_s and pid not in seen_p]
    reports = []
    if ids:
        reports.append(evaluate_surrogate(sur, ds.samples(ids), tolerance=args.tolerance,
                                          level=args.level, split="held_out"))
    if args.family:
        if args.family in sur.training.get("families", []):
            raise ValueError(f"the model was trained on family {args.family!r}; a held-out "
                             "family evaluation needs a model that has not seen it")
        fam_ids = ds.filter(family=args.family)
        if not fam_ids:
            raise ValueError(f"no sample of family {args.family!r} in {args.data}")
        reports.append(evaluate_surrogate(sur, ds.samples(fam_ids), tolerance=args.tolerance,
                                          level=args.level, split="held_out_family"))
    if not reports:
        raise ValueError("nothing to evaluate: no held-out ids")
    rep = EvaluationReport.concatenate(reports)
    paths = rep.to_csv(args.out)
    summary = rep.summary()
    write_json(Path(args.out) / "summary.json", to_jsonable(summary))
    print(canonical_json({"tables": {k: str(v) for k, v in paths.items()},
                          "summary": summary}), end="")
    return 0


# -- active -------------------------------------------------------------------------
def _process_ranges(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    out = {}
    for item in items or []:
        if "=" not in item or "," not in item:
            raise ValueError(f"expected KEY=LO,HI, got {item!r}")
        k, v = item.split("=", 1)
        lo, hi = (float(x) for x in v.split(","))
        out[k.strip()] = (lo, hi)
    return out


def _active_space(training: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """DesignSpace arguments of `precomp active`: by default the setup the
    model was trained on - its materials, the ranges of the process fields it
    saw (a field that did not vary is fixed at its value), every other setup
    field as in training (the smallest value of one that varied), its grid
    spacing - so candidates differ in geometry, not in a process the model
    never saw. --materials, --process, --setup and --grid-spacing override."""
    env = training.get("setup") or {}
    base: Dict[str, Any] = {}
    for key, rule in (env.get("fields") or {}).items():
        base[key] = rule["values"][0] if "values" in rule else rule["min"]
    process: Dict[str, Any] = {}
    for key, rule in (env.get("process") or {}).items():
        if rule["max"] > rule["min"]:
            process[key] = (rule["min"], rule["max"])
        else:
            base[key] = rule["min"]
    if args.setup:
        doc = read_json(args.setup)
        doc.pop("material", None)
        base.update(doc)
        for key in doc:
            process.pop(key, None)
    if args.process is not None:            # given without values: fix the process
        given = _process_ranges(args.process)
        if not given:
            for key, (lo, _) in process.items():
                base.setdefault(key, lo)
            process = {}
        for key in given:
            base.pop(key, None)
        process.update(given)
    kw: Dict[str, Any] = {"base_setup": base, "process": process}
    materials = args.materials or env.get("materials")
    if materials:
        kw["materials"] = tuple(materials)
    if args.families:
        kw["families"] = tuple(args.families)
    spacings = training.get("grid_spacings") or []
    kw["grid_spacing"] = args.grid_spacing or (spacings[0] if spacings else 2e-3)
    return kw


def cmd_active(args: argparse.Namespace) -> int:
    from .active import Candidate, rank_candidates
    from .generate import DesignSpace, design_points
    from .registry import load_model

    sur = load_model(args.model)
    space = DesignSpace(**_active_space(sur.training, args))
    n_fam = len(space.families)
    per = max(1, -(-args.n_candidates // n_fam))
    points = design_points(space, per, args.seed)[:max(args.n_candidates, 1)]
    cands = [Candidate(p.target(), p.setup, p.point_id,
                       {"family": p.family, "material": p.setup.material.name,
                        "part": json.dumps(to_jsonable(p.part.to_dict()), sort_keys=True)})
             for p in points]
    table = rank_candidates(sur, cands, n_select=args.select, diversity=args.diversity)
    table.insert(0, "model_data_source", sur.data_source)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.6g")
    top = table.head(args.select)[["rank", "label", "mean_std_m", "ood_part_score",
                                   "acquisition"]]
    print(top.to_string(index=False))
    return 0


# -- parser -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="precomp", description="precomp machine-learning commands")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)

    d = sub.add_parser("dataset", help="data sets")
    dsub = d.add_subparsers(dest="dataset_command", required=True, parser_class=_Parser)
    g = dsub.add_parser("generate", help="design of experiments -> data set")
    g.add_argument("--out", required=True, help="data set directory (created or resumed)")
    g.add_argument("--simulator", choices=["sparlab", "proxy"], required=True,
                   help="sparlab: sparlab_form; proxy: the analytic ProxySimulator (NOT "
                        "physics; tests and smoke runs)")
    g.add_argument("--work-dir", help="run cache of sparlab_form")
    g.add_argument("--families", nargs="*", help="part families (default: all)")
    g.add_argument("--materials", nargs="*", help="library materials (default: the sheet "
                                                  "alloys; one per model is recommended)")
    g.add_argument("--process", nargs="*", metavar="KEY=LO,HI",
                   help="process ranges [SI], e.g. thickness=6e-4,1.5e-3 (give none to fix "
                        "the process at --setup)")
    g.add_argument("--setup", help="FormingSetup JSON with the fields not varied")
    g.add_argument("--n-per-family", type=int, required=True)
    g.add_argument("--kinds", nargs="*", default=["uncompensated", "perturbed", "compensated"])
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--grid-spacing", type=float, default=2e-3, help="[m]")
    g.add_argument("--max-workers", type=int, default=2)
    g.add_argument("--created-at", help="ISO time recorded in the provenance (default: now)")
    g.set_defaults(func=cmd_generate)
    i = dsub.add_parser("info", help="summary of a data set")
    i.add_argument("--data", required=True)
    i.set_defaults(func=cmd_info)

    t = sub.add_parser("train", help="train, calibrate and save a surrogate")
    t.add_argument("--data", required=True)
    t.add_argument("--model", choices=["gbm", "mlp", "residual", "transfer"],
                   default="gbm")
    t.add_argument("--out", required=True, help="model bundle directory")
    t.add_argument("--param", nargs="*", metavar="KEY=VALUE", help="model hyper-parameters")
    t.add_argument("--families", nargs="*", help="train on these families only")
    t.add_argument("--holdout-family", help="exclude a family entirely (to evaluate it later)")
    t.add_argument("--test-fraction", type=float, default=0.2,
                   help="grouped share of parts held out for testing (0: none)")
    t.add_argument("--calibration-fraction", type=float, default=0.25,
                   help="grouped share of the remaining parts for conformal calibration")
    t.add_argument("--points-per-sample", type=int, default=2000)
    t.add_argument("--time-source", choices=["toolpath", "depth"], default="toolpath",
                   help="how the time_frac feature is computed")
    t.add_argument("--group-by", choices=["sample", "part"], default="sample",
                   help="groups of the GBM bootstrap / MLP early-stopping split")
    t.add_argument("--prior", choices=["proxy", "fea"], help="residual model: the prior")
    t.add_argument("--prior-work-dir", help="residual model with --prior fea: run cache")
    t.add_argument("--prior-set", nargs="*", metavar="KEY=VALUE",
                   help="setup overrides of the coarse FE prior (e.g. element_size=5e-3)")
    t.add_argument("--base", help="transfer model: the simulation-trained bundle")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--tolerance", type=float, default=2e-4, help="[m]")
    t.add_argument("--level", type=float, default=0.9)
    t.add_argument("--n-jobs", type=int, default=1, help="featurisation processes (<= 2)")
    t.add_argument("--notes")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--created-at")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="score a model bundle on held-out parts")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--ids", help="file of sample ids (default: the bundle's test split)")
    e.add_argument("--family", help="also score this held-out family")
    e.add_argument("--tolerance", type=float, default=2e-4, help="[m]")
    e.add_argument("--level", type=float, default=0.9)
    e.add_argument("--out", required=True, help="directory of the CSV tables")
    e.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("active", help="rank candidate parts for the next runs")
    a.add_argument("--model", required=True)
    a.add_argument("--n-candidates", type=int, default=64)
    a.add_argument("--select", type=int, default=8)
    a.add_argument("--families", nargs="*", help="part families (default: all)")
    a.add_argument("--materials", nargs="*", help="default: the model's training materials")
    a.add_argument("--process", nargs="*", metavar="KEY=LO,HI",
                   help="process ranges [SI] (default: the ranges the model was trained on)")
    a.add_argument("--setup", help="FormingSetup JSON of the fields not varied (default: "
                                   "the model's training setup)")
    a.add_argument("--grid-spacing", type=float,
                   help="[m] (default: the training grid spacing)")
    a.add_argument("--diversity", type=float, default=0.5)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--out", required=True, help="ranking CSV")
    a.set_defaults(func=cmd_active)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one ML command; returns the exit status (errors: 2, message on stderr)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv)
        return int(args.func(args) or 0)
    except _UsageError as exc:
        print(f"precomp (precomp.ml): usage error: {exc}", file=sys.stderr)
        return 2
    except (PrecompError, ValueError, KeyError, FileNotFoundError, TypeError,
            ImportError) as exc:
        print(f"precomp (precomp.ml): error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
