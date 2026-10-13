"""Samples of the commanded -> formed map, their on-disk data set, tables and splits.

A `Sample` is one forming of one commanded surface: the commanded and the formed
tool-side surfaces on the same grid, the setup, the target part it came from,
and its provenance. The learning target is dz = z_formed - z_commanded [m].

On disk a `Dataset` is a directory

    dataset.json            format, description, created_at (given by the caller)
    index.jsonl             one line per sample (append-only; the last line of an
                            id wins): id, part_id, family, kind, source, fidelity
    samples/<id>.npz        commanded, formed (and target) heights and masks, the
                            grid, the tool path when recorded
    samples/<id>.json       the full record: part, setup, provenance
    failures.jsonl          jobs that produced no sample, with their reason

Every sample states its data source: "sim" (a SparLab simulation), "scan"
(a measured part) or "proxy" (the analytic ProxySimulator - not physics). Tables
and metrics carry that label (`SOURCE_LABELS`) wherever they are reported.

Splits are grouped. The variants of one design point (the uncompensated,
perturbed and compensated commanded shapes of one target part) share a
`part_id` and are split together by default, so a part never appears on both
sides of a split; `by="sample"` and `by="family"` (leave-family-out) are the
other groupings.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional,
                    Sequence, Tuple, Union)

import numpy as np
import pandas as pd

from .._util import (PathLike, PrecompError, canonical_json, read_json,
                     safe_name, to_jsonable)
from ..geometry.heightmap import Grid, HeightMap
from ..toolpath import Toolpath
from .features import (DEFAULT_CONFIG, REGION_NAMES, FeatureConfig, as_setup,
                       feature_maps)

DATASET_FORMAT = "precomp.ml.Dataset/1"
SAMPLE_FORMAT = "precomp.ml.Sample/1"

#: Data sources and how every metric computed on them is labelled.
SOURCES = ("sim", "scan", "proxy")
SOURCE_LABELS = {"sim": "SparLab simulation", "scan": "scan",
                 "proxy": "proxy - not physics"}

#: How the commanded surface of a sample was obtained.
KINDS = ("uncompensated", "perturbed", "compensated", "measured")

#: Training masks of `to_table`.
MASKS = ("part", "all")


def source_label(sources: Iterable[str]) -> str:
    """The label of a set of data sources: one label, or "mixed: a + b"."""
    s = sorted(set(sources))
    if not s:
        raise ValueError("no data source")
    for x in s:
        if x not in SOURCE_LABELS:
            raise ValueError(f"unknown data source {x!r}; known: {SOURCES}")
    if len(s) == 1:
        return SOURCE_LABELS[s[0]]
    return "mixed: " + " + ".join(SOURCE_LABELS[x] for x in s)


@dataclass
class Sample:
    """One commanded surface and the surface formed from it.

    sample_id : unique, file-name safe ([A-Za-z0-9_-]).
    commanded, formed : HeightMaps on the same grid [m].
    setup : `FormingSetup.to_dict()` of the run.
    source : "sim" | "scan" | "proxy"; fidelity : a tag of the model fidelity
        (e.g. "sparlab:hex8:2.5mm:2L" or "proxy-v1").
    kind : "uncompensated" | "perturbed" | "compensated" | "measured".
    part : `Part.to_dict()` of the target part, or None (family "unknown").
    part_id : the design point; variants of one target share it (default:
        the sample id).
    target : the target surface the commanded one was derived from, or None.
    toolpath : the tool path that formed it, when recorded.
    provenance : at least `created_at` (a string supplied by the caller, never
        read from the clock here); for "sim" also `sparlab_version` and
        `deck_hash`, and `runtime_s` when known.
    """

    sample_id: str
    commanded: HeightMap
    formed: HeightMap
    setup: Dict[str, Any]
    source: str
    fidelity: str
    kind: str = "uncompensated"
    part: Optional[Dict[str, Any]] = None
    part_id: Optional[str] = None
    target: Optional[HeightMap] = None
    toolpath: Optional[Toolpath] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or safe_name(self.sample_id) != self.sample_id:
            raise ValueError(f"sample_id {self.sample_id!r} must be non-empty and use only "
                             "letters, digits, '-' and '_'")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not self.fidelity:
            raise ValueError("fidelity must be a non-empty tag")
        if not self.formed.grid.matches(self.commanded.grid):
            raise ValueError(f"sample {self.sample_id}: the formed surface is not on the "
                             "commanded grid; resample it (HeightMap.resample) first")
        if self.target is not None and not self.target.grid.matches(self.commanded.grid):
            raise ValueError(f"sample {self.sample_id}: the target is not on the commanded grid")
        if hasattr(self.setup, "to_dict"):
            self.setup = self.setup.to_dict()
        self.setup = dict(self.setup)
        if "material" not in self.setup:
            raise ValueError(f"sample {self.sample_id}: setup has no material")
        if self.part_id is None:
            self.part_id = self.sample_id
        prov = dict(self.provenance)
        if not isinstance(prov.get("created_at"), str) or not prov["created_at"]:
            raise ValueError(f"sample {self.sample_id}: provenance needs 'created_at', a "
                             "string supplied by the caller")
        if self.source == "sim":
            for key in ("sparlab_version", "deck_hash"):
                if not prov.get(key):
                    raise ValueError(f"sample {self.sample_id}: a simulated sample needs "
                                     f"provenance['{key}']")
        self.provenance = prov

    # -- derived -------------------------------------------------------------
    @property
    def family(self) -> str:
        return str(self.part.get("family", "unknown")) if self.part else "unknown"

    @property
    def grid(self) -> Grid:
        return self.commanded.grid

    @property
    def valid(self) -> np.ndarray:
        """Nodes where both surfaces have data."""
        return self.commanded.mask & self.formed.mask

    @property
    def dz(self) -> np.ndarray:
        """(ny, nx) vertical deviation z_formed - z_commanded [m] (0 where invalid)."""
        return np.where(self.valid, self.formed.z - self.commanded.z, 0.0)

    def deviation(self) -> HeightMap:
        """dz as a HeightMap (invalid where either surface is)."""
        return HeightMap(self.grid, self.dz, self.valid,
                         {"quantity": "vertical_deviation", "unit": "m",
                          "sample_id": self.sample_id})

    def forming_setup(self):
        return as_setup(self.setup)

    @property
    def data_source(self) -> str:
        return SOURCE_LABELS[self.source]

    def index_entry(self) -> Dict[str, Any]:
        return {"sample_id": self.sample_id, "part_id": self.part_id, "family": self.family,
                "kind": self.kind, "source": self.source, "fidelity": self.fidelity,
                "material": self.setup["material"].get("name")
                if isinstance(self.setup["material"], Mapping) else str(self.setup["material"]),
                "nx": self.grid.nx, "ny": self.grid.ny, "h": self.grid.h,
                "created_at": self.provenance["created_at"]}

    # -- files ---------------------------------------------------------------
    def save(self, directory: PathLike) -> Tuple[Path, Path]:
        """Write samples/<id>.npz and <id>.json into `directory` (atomically)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, Any] = {
            "format": np.array(SAMPLE_FORMAT),
            "grid": np.array(json.dumps(self.grid.to_dict())),
            "commanded_z": self.commanded.z, "commanded_mask": self.commanded.mask,
            "formed_z": self.formed.z, "formed_mask": self.formed.mask}
        if self.target is not None:
            arrays["target_z"] = self.target.z
            arrays["target_mask"] = self.target.mask
        if self.toolpath is not None:
            arrays["toolpath_points"] = self.toolpath.points
            arrays["toolpath_level"] = self.toolpath.level
            arrays["toolpath_radius"] = np.array(self.toolpath.tool_radius)
        npz = d / f"{self.sample_id}.npz"
        tmp = d / f".{self.sample_id}.{uuid.uuid4().hex}.npz"
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, npz)
        doc = {"format": SAMPLE_FORMAT, "sample_id": self.sample_id, "part_id": self.part_id,
               "source": self.source, "data_source": self.data_source,
               "fidelity": self.fidelity, "kind": self.kind, "part": self.part,
               "setup": self.setup, "provenance": self.provenance,
               "metadata": {"commanded": self.commanded.metadata,
                            "formed": self.formed.metadata,
                            "target": None if self.target is None else self.target.metadata,
                            "toolpath": None if self.toolpath is None
                            else self.toolpath.metadata}}
        js = d / f"{self.sample_id}.json"
        tmp = d / f".{self.sample_id}.{uuid.uuid4().hex}.json"
        tmp.write_text(canonical_json(doc), encoding="utf-8")
        os.replace(tmp, js)
        return npz, js

    @classmethod
    def load(cls, directory: PathLike, sample_id: str) -> "Sample":
        d = Path(directory)
        doc = read_json(d / f"{sample_id}.json")
        npz = d / f"{sample_id}.npz"
        if not npz.is_file():
            raise PrecompError(f"{npz} does not exist")
        if doc.get("format") != SAMPLE_FORMAT:
            raise PrecompError(f"{d / sample_id}.json: format {doc.get('format')!r}, expected "
                               f"{SAMPLE_FORMAT!r}")
        meta = doc.get("metadata") or {}
        with np.load(npz, allow_pickle=False) as data:
            grid = Grid.from_dict(json.loads(str(data["grid"])))
            commanded = HeightMap(grid, data["commanded_z"], data["commanded_mask"],
                                  meta.get("commanded") or {})
            formed = HeightMap(grid, data["formed_z"], data["formed_mask"],
                               meta.get("formed") or {})
            target = (HeightMap(grid, data["target_z"], data["target_mask"],
                                meta.get("target") or {}) if "target_z" in data else None)
            path = (Toolpath(data["toolpath_points"], data["toolpath_level"],
                             float(data["toolpath_radius"]), meta.get("toolpath") or {})
                    if "toolpath_points" in data else None)
        return cls(doc["sample_id"], commanded, formed, doc["setup"], doc["source"],
                   doc["fidelity"], doc["kind"], doc.get("part"), doc.get("part_id"), target,
                   path, doc.get("provenance") or {})


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
@dataclass
class Table:
    """Rows of features and targets, one per selected node.

    X (n, F), y (n,) dz [m], groups (n,) sample ids; per row also the region
    code (`REGION_NAMES`), the flat node index, the part id, family and data
    source. Unpacks as ``X, y, groups = table``.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]
    region: np.ndarray
    node: np.ndarray
    part_id: np.ndarray
    family: np.ndarray
    source: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.X
        yield self.y
        yield self.groups

    def __len__(self) -> int:
        return int(len(self.y))

    @property
    def data_source(self) -> str:
        return source_label(np.unique(self.source))

    def subset(self, rows: np.ndarray) -> "Table":
        rows = np.asarray(rows)
        return Table(self.X[rows], self.y[rows], self.groups[rows], list(self.feature_names),
                     self.region[rows], self.node[rows], self.part_id[rows],
                     self.family[rows], self.source[rows])

    @staticmethod
    def concatenate(tables: Sequence["Table"]) -> "Table":
        if not tables:
            raise ValueError("no tables to concatenate")
        names = tables[0].feature_names
        for t in tables[1:]:
            if t.feature_names != names:
                raise ValueError("tables with different feature names cannot be concatenated")
        cat = lambda a: np.concatenate([getattr(t, a) for t in tables])  # noqa: E731
        return Table(cat("X"), cat("y"), cat("groups"), list(names), cat("region"),
                     cat("node"), cat("part_id"), cat("family"), cat("source"))


def stratified_nodes(candidates: np.ndarray, region: np.ndarray, n: Optional[int],
                     rng: np.random.Generator, stratify: bool = True) -> np.ndarray:
    """Flat node indices: `n` of the `candidates` (bool, grid shape), equally
    shared among the regions present when `stratify` (a region with fewer
    nodes than its share gives all it has and the rest is redistributed), or
    uniformly at random otherwise. All candidates when n is None or larger."""
    idx = np.flatnonzero(candidates.ravel())
    if n is None or n >= idx.size:
        return idx
    if n <= 0:
        raise ValueError("points_per_sample must be > 0")
    if not stratify:
        return np.sort(rng.choice(idx, size=n, replace=False))
    reg = region.ravel()[idx]
    strata = [idx[reg == r] for r in np.unique(reg)]
    strata.sort(key=len)
    chosen = []
    remaining = n
    for k, s in enumerate(strata):
        share = remaining // (len(strata) - k)
        take = min(share, s.size)
        chosen.append(rng.choice(s, size=take, replace=False) if take < s.size else s)
        remaining -= take
    return np.sort(np.concatenate(chosen))


def _mask_of(sample: Sample, part: np.ndarray, mask: str) -> np.ndarray:
    if mask == "part":
        return part & sample.valid
    if mask == "all":
        return sample.valid.copy()
    raise ValueError(f"mask must be one of {MASKS}, got {mask!r}")


def sample_table(sample: Sample, config: FeatureConfig = DEFAULT_CONFIG, *,
                 target: str = "dz", points_per_sample: Optional[int] = 2000,
                 mask: str = "part", rng: Optional[np.random.Generator] = None,
                 stratify: bool = True) -> Table:
    """The table of one sample (see `build_table`)."""
    if target != "dz":
        raise ValueError(f"target must be 'dz' (the vertical deviation), got {target!r}")
    rng = np.random.default_rng(0) if rng is None else rng
    try:
        fm = feature_maps(sample.commanded, sample.setup, sample.toolpath, config)
    except (ValueError, PrecompError) as exc:
        raise PrecompError(f"sample {sample.sample_id}: {exc}") from exc
    cand = _mask_of(sample, fm.part, mask)
    if not cand.any():
        raise PrecompError(f"sample {sample.sample_id}: no valid node in mask {mask!r}")
    nodes = stratified_nodes(cand, fm.region, points_per_sample, rng, stratify)
    X = fm.gather(nodes)
    names = list(fm.names)
    y = sample.dz.ravel()[nodes]
    n = len(nodes)
    return Table(X, y, np.full(n, sample.sample_id, dtype=object), names,
                 fm.region.ravel()[nodes], nodes, np.full(n, sample.part_id, dtype=object),
                 np.full(n, sample.family, dtype=object), np.full(n, sample.source, dtype=object))


def build_table(samples: Iterable[Sample], config: FeatureConfig = DEFAULT_CONFIG, *,
                target: str = "dz", points_per_sample: Optional[int] = 2000,
                mask: str = "part", rng: Union[None, int, np.random.Generator] = None,
                stratify: bool = True, n_jobs: int = 1) -> Table:
    """Features and targets of many samples.

    points_per_sample : nodes drawn per sample (None: every node in the mask).
        With `stratify` they are shared equally among the regions present
        (rim, wall, base, and flange for mask "all"), so large flat regions do
        not dominate; without, uniformly.
    mask : "part" (commanded depth > part_eps) or "all" (every valid node).
    rng : a Generator or a seed; each sample gets its own child stream in
        order, so the table is reproducible for the same samples and seed.
    n_jobs : processes for featurisation (joblib); keep <= 2 on shared machines.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("no samples")
    seed_seq = (rng.bit_generator.seed_seq if isinstance(rng, np.random.Generator)
                else np.random.SeedSequence(0 if rng is None else int(rng)))
    children = seed_seq.spawn(len(samples))
    kw = dict(target=target, points_per_sample=points_per_sample, mask=mask,
              stratify=stratify)
    if n_jobs == 1 or len(samples) == 1:
        tables = [sample_table(s, config, rng=np.random.default_rng(c), **kw)
                  for s, c in zip(samples, children)]
    else:
        from joblib import Parallel, delayed
        tables = Parallel(n_jobs=n_jobs)(
            delayed(sample_table)(s, config, rng=np.random.default_rng(c), **kw)
            for s, c in zip(samples, children))
    return Table.concatenate(tables)


# ---------------------------------------------------------------------------
# The data set on disk
# ---------------------------------------------------------------------------
class Dataset:
    """A directory of samples (see the module docstring). Single writer."""

    def __init__(self, root: PathLike):
        self.root = Path(root)
        meta = self.root / "dataset.json"
        if not meta.is_file():
            raise PrecompError(f"{self.root} is not a precomp.ml data set (no dataset.json); "
                               "create one with Dataset.create")
        self.meta = read_json(meta)
        if self.meta.get("format") != DATASET_FORMAT:
            raise PrecompError(f"{meta}: format {self.meta.get('format')!r}, expected "
                               f"{DATASET_FORMAT!r}")

    @classmethod
    def create(cls, root: PathLike, *, created_at: str, description: str = "",
               exist_ok: bool = False) -> "Dataset":
        """Create an empty data set (or open an existing one with `exist_ok`)."""
        root = Path(root)
        if (root / "dataset.json").is_file():
            if not exist_ok:
                raise PrecompError(f"{root} already holds a data set")
            return cls(root)
        if not created_at:
            raise ValueError("created_at must be supplied by the caller")
        (root / "samples").mkdir(parents=True, exist_ok=True)
        (root / "index.jsonl").touch()
        (root / "dataset.json").write_text(canonical_json(
            {"format": DATASET_FORMAT, "created_at": created_at, "description": description}),
            encoding="utf-8")
        return cls(root)

    @classmethod
    def open(cls, root: PathLike) -> "Dataset":
        """Open an existing data set (PrecompError when `root` holds none)."""
        return cls(root)

    # -- index ---------------------------------------------------------------
    def _index_rows(self) -> List[Dict[str, Any]]:
        path = self.root / "index.jsonl"
        rows: Dict[str, Dict[str, Any]] = {}
        if path.is_file():
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PrecompError(f"{path}:{n} is not valid JSON: {exc}") from exc
                rows[row["sample_id"]] = row
        return list(rows.values())

    def index(self) -> pd.DataFrame:
        """One row per sample (the latest index line of each id)."""
        rows = self._index_rows()
        cols = ["sample_id", "part_id", "family", "kind", "source", "fidelity", "material",
                "nx", "ny", "h", "created_at"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def ids(self) -> List[str]:
        return [r["sample_id"] for r in self._index_rows()]

    def __len__(self) -> int:
        return len(self._index_rows())

    def has(self, sample_id: str) -> bool:
        return (self.root / "samples" / f"{sample_id}.json").is_file() and \
            sample_id in set(self.ids())

    def append(self, sample: Sample, *, overwrite: bool = False) -> None:
        """Store a sample. An existing id is refused unless `overwrite`."""
        if self.has(sample.sample_id) and not overwrite:
            raise PrecompError(f"sample {sample.sample_id} is already in {self.root}; pass "
                               "overwrite=True to replace it")
        sample.save(self.root / "samples")
        with open(self.root / "index.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(sample.index_entry()), sort_keys=True) + "\n")

    def load(self, sample_id: str) -> Sample:
        return Sample.load(self.root / "samples", sample_id)

    def __iter__(self) -> Iterator[Sample]:
        for sid in self.ids():
            yield self.load(sid)

    def samples(self, ids: Optional[Sequence[str]] = None) -> List[Sample]:
        return [self.load(s) for s in (self.ids() if ids is None else ids)]

    def filter(self, predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
               **criteria: Any) -> List[str]:
        """Ids whose index entry matches every criterion (a value, or a
        collection of accepted values, per column: source, family, kind,
        fidelity, material, part_id) and the predicate, if given."""
        out = []
        for row in self._index_rows():
            ok = True
            for key, want in criteria.items():
                if key not in row:
                    raise KeyError(f"unknown index column {key!r}")
                accepted = set(want) if isinstance(want, (list, tuple, set, frozenset)) \
                    else {want}
                if row[key] not in accepted:
                    ok = False
                    break
            if ok and (predicate is None or predicate(row)):
                out.append(row["sample_id"])
        return out

    # -- failures ------------------------------------------------------------
    def record_failure(self, record: Mapping[str, Any]) -> None:
        """Append a failed job (`sample_id`, `reason`, ...) to failures.jsonl."""
        if "sample_id" not in record or "reason" not in record:
            raise ValueError("a failure record needs 'sample_id' and 'reason'")
        with open(self.root / "failures.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(dict(record)), sort_keys=True) + "\n")

    def failures(self) -> pd.DataFrame:
        """Every recorded failure (latest per sample id), minus samples that
        have since succeeded."""
        path = self.root / "failures.jsonl"
        rows: Dict[str, Dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[row["sample_id"]] = row
        done = set(self.ids())
        rows = {k: v for k, v in rows.items() if k not in done}
        return pd.DataFrame(list(rows.values()))

    # -- tables --------------------------------------------------------------
    def to_table(self, features: FeatureConfig = DEFAULT_CONFIG, target: str = "dz", *,
                 points_per_sample: Optional[int] = 2000, mask: str = "part",
                 rng: Union[None, int, np.random.Generator] = None,
                 ids: Optional[Sequence[str]] = None, stratify: bool = True,
                 n_jobs: int = 1) -> Table:
        """X, y, groups (and more, see `Table`) of the samples `ids` (default:
        all); see `build_table` for the arguments."""
        return build_table(self.samples(ids), features, target=target,
                           points_per_sample=points_per_sample, mask=mask, rng=rng,
                           stratify=stratify, n_jobs=n_jobs)


__all__ = ["Sample", "Dataset", "Table", "build_table", "sample_table", "stratified_nodes",
           "source_label", "SOURCES", "SOURCE_LABELS", "KINDS", "MASKS", "REGION_NAMES",
           "DATASET_FORMAT"]
