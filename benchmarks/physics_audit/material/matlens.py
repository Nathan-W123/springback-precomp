"""Material-model-fidelity lens of the physics audit: shared helpers.

Builds sparlab_form decks for material / friction variants of one benchmark
test part (the commanded shape and FormingSetup of the springback
benchmark's cached uncompensated run), runs them with the audit worktree's
Release build (one thread, at most two at a time), and measures the released
shape.

Nothing here modifies tracked code; every deck is written by precomp.fea's
`build_deck` and, where a variant needs a key FormingSetup does not expose
(material_regions, element formulation, thickness points), patched after.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                   # /home/user/wt/audit
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from precomp.fea import build_deck, load_result, run_deck          # noqa: E402
from precomp.fea.setup import FormingSetup                          # noqa: E402
from precomp.geometry.heightmap import HeightMap                    # noqa: E402
from precomp.materials import Material, get_material                # noqa: E402
from precomp.metrology import (flange_mask, part_mask,              # noqa: E402
                               vertical_deviation)

EXE = ROOT / "build" / "bin" / "sparlab_form"
CACHE = Path("/home/user/wt/bench/benchmarks/springback/work/runs/runs")
RUNS = HERE / "runs"
RESULTS = HERE / "results.jsonl"
MM = 1e-3

#: benchmark test parts used here -> deck hash of their cached uncompensated run
PARTS = {
    "truncated_cone-s2026-0000": "3c6e05ba7b508ae158f25c53446bbff3020340ee70ad512751f2d5cbed58660b",
    "pyramid-s2026-0001": "49a7d65dcec9",
    "elliptic_cone-s2026-0001": "ad8669ed976a",
}


def cached_run_dir(part: str) -> Path:
    h = PARTS[part]
    hits = sorted((CACHE / h[:2]).glob(h + "*"))
    if len(hits) != 1:
        raise RuntimeError(f"cached run of {part} not found ({h})")
    return hits[0]


def part_inputs(part: str):
    """(setup, commanded HeightMap) of the benchmark's uncompensated run of `part`."""
    d = cached_run_dir(part)
    prov = json.loads((d / "precomp_deck.json").read_text())
    setup = FormingSetup.from_dict(prov["setup"])
    setup = setup.replace(executable=str(EXE), threads=1)
    commanded = HeightMap.load(d / "commanded.npz")
    return setup, commanded


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def library_material() -> Material:
    return get_material("AA5754-O")


def apply_variant(setup: FormingSetup, spec: Dict[str, Any]):
    """A setup and a deck patch for a variant spec:
    {"material": {field: value, ...}  (Material.replace),
     "setup": {field: value, ...}     (FormingSetup.replace),
     "deck": {...}                    (merged into deck.json after build_deck),
     "e_of_ep": {...}                 (per-element E from a baseline strain map)}"""
    mat = setup.material
    if spec.get("material"):
        mat = mat.replace(**spec["material"])
    s = setup.replace(material=mat, **spec.get("setup", {}))
    return s


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def e_of_ep_regions(material_block: Dict[str, Any], ep: np.ndarray, law: Dict[str, Any],
                    nbins: int = 8) -> List[Dict[str, Any]]:
    """material_regions giving each element E(ep) = E0 - (E0 - Ea)(1 - exp(-xi ep))
    (Yoshida-Uemori's chord-modulus law), binned into `nbins` levels of E."""
    E0 = float(material_block["youngs_modulus"])
    Ea, xi = float(law["Ea"]), float(law["xi"])
    E = E0 - (E0 - Ea) * (1.0 - np.exp(-xi * ep))
    # bin edges in E between Ea and E0; elements within 0.25 % of E0 keep the base
    lo = E.min()
    edges = np.linspace(lo, E0, nbins + 1)
    regions = []
    for b in range(nbins):
        sel = (E >= edges[b]) & ((E < edges[b + 1]) if b < nbins - 1 else (E <= E0))
        ids = np.flatnonzero(sel)
        Eb = float(np.mean(E[ids])) if len(ids) else None
        if not len(ids) or Eb > E0 * (1 - 2.5e-3):
            continue
        m = copy.deepcopy(material_block)
        m["name"] = f"{material_block['name']}-E{Eb / 1e9:.2f}"
        m["youngs_modulus"] = Eb
        regions.append({"name": f"e_bin_{b}", "region": {"element_ids": [int(i) for i in ids]},
                        "material": m})
    return regions


def build_variant(part: str, name: str, spec: Dict[str, Any]) -> Path:
    setup, commanded = part_inputs(part)
    s = apply_variant(setup, spec)
    d = RUNS / part / name
    if (d / "COMPLETE").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    build_deck(s, commanded, d)
    deck = json.loads((d / "deck.json").read_text())
    if spec.get("deck"):
        deep_merge(deck, spec["deck"])
    if spec.get("e_of_ep"):
        law = spec["e_of_ep"]
        base = law.get("from", "base")
        el = np.loadtxt(RUNS / part / base / "output" / "step_3_release_elements.csv",
                        delimiter=",", skiprows=1)
        ep = el[np.argsort(el[:, 0]), 1]
        deck["material_regions"] = e_of_ep_regions(deck["material"], ep, law)
    (d / "deck.json").write_text(json.dumps(deck, indent=1, sort_keys=True))
    (d / "variant.json").write_text(json.dumps({"part": part, "variant": name, "spec": spec},
                                               indent=1, sort_keys=True))
    return d


def run_variant(part: str, name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    d = build_variant(part, name, spec)
    rec: Dict[str, Any] = {"part": part, "variant": name}
    if not (d / "COMPLETE").exists():
        t0 = time.time()
        try:
            run_deck(d, executable=EXE, threads=1)
            (d / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        except Exception as exc:                              # recorded, reported
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:2000],
                       wall_s=time.time() - t0)
            (d / "FAILED").write_text(rec["error"])
            return rec
    rec.update(ok=True, **measure(part, d))
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def region_stats(dev: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    v = dev[mask]
    return {"rms": float(np.sqrt(np.mean(v ** 2))), "max": float(np.abs(v).max()),
            "bias": float(np.mean(v))}


def measure(part: str, d: Path) -> Dict[str, Any]:
    _, target = part_inputs(part)
    res = load_result(d / "output")
    grid = target.grid
    formed = {s: res.formed_surface(s, grid=grid) for s in ("form", "unload", "release")}
    np.savez_compressed(d / "surfaces.npz", **{s: h.z for s, h in formed.items()},
                        **{f"{s}_mask": h.mask for s, h in formed.items()},
                        target=target.z, grid=json.dumps(grid.to_dict()))
    part_m = part_mask(target)
    upper = part_m & (target.z > -1e-3)
    deep = part_m & ~upper
    fl = flange_mask(target)
    out: Dict[str, Any] = {}
    dev = vertical_deviation(formed["release"], target).z / MM
    out["dev_part"] = region_stats(dev, part_m)
    out["dev_upper"] = region_stats(dev, upper)
    out["dev_deep"] = region_stats(dev, deep)
    out["dev_flange"] = region_stats(dev, fl)
    sb = (formed["release"].z - formed["form"].z) / MM      # free springback (released - loaded)
    out["springback_part"] = region_stats(sb, part_m)
    sb2 = (formed["unload"].z - formed["form"].z) / MM      # springback in the clamp
    out["springback_clamped_part"] = region_stats(sb2, part_m)
    out["loaded_depth_mm"] = float(-formed["form"].z[part_m].min() / MM)
    out["released_depth_mm"] = float(-formed["release"].z[part_m].min() / MM)
    summ = json.loads((d / "output" / "summary.json").read_text())
    out["runtime_s"] = summ.get("runtime_s")
    out["increments"] = summ["timing"]["increments"]
    out["cuts"] = summ["timing"]["cuts"]
    out["max_plastic_strain"] = max(s.get("max_plastic_strain", 0) for s in summ["steps"])
    out["completed"] = summ.get("completed")
    tf = res.forming_forces()
    out["peak_fz_N"] = float(tf["fz"].max())
    th = res.thickness_map("release", grid=grid)
    out["min_thickness_mm"] = float(th.z[part_m & th.mask].min() / MM)
    return out


def released_surface(part: str, name: str) -> np.ndarray:
    z = np.load(RUNS / part / name / "surfaces.npz")
    return z["release"]
