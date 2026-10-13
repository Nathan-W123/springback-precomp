"""Running sparlab_form: one deck, a content-addressed cache, many in parallel.

Cache layout under a work directory W:

    W/runs/<h[:2]>/<h>/            h = deck_hash(deck, `sparlab_form --version`)
        deck.json, toolpath.csv, commanded.npz, precomp_deck.json
        output/                    the solver's result directory
        run.json                   command, version, runtime, exit code
        COMPLETE                   written last: the entry is a finished run
        FAILED                     instead of COMPLETE: why the run failed (JSON)
    W/staging/<uuid>/              decks being built or run

A run is built and executed in staging and then published with one atomic
rename, so a reader never sees a half-written entry and two processes racing
on the same deck do not corrupt it (the second publisher discards its copy).
An identical deck - same physics, same trajectory, same solver build - is
never run twice: `simulate` returns the cached result. A recorded failure is
raised again without re-running unless `retry_failed` is set, so a
deterministic failure costs its runtime once.

Each run is a subprocess with OMP_NUM_THREADS set to the setup's `threads`
(1 by default, so a pool of N workers uses N cores), and with
`--strict-config`: a deck key sparlab_form does not read is an error (exit
2), never a default taken silently. sparlab_form exits with 0 when every step
completed, 3 when one stopped (`summary.json` says why), 2 for a
configuration error and 4 for an I/O error (docs/forming.md, section 3);
anything but 0 is a FormingError.

Where this fits: the runner step of "Simulations": runs sparlab_form through the content-addressed
    run cache, one deck (simulate) or many in parallel (simulate_many).
Called from: compensation.py:FEAPredictor.__call__, ml/generate.py:SparlabSimulator.run,
    combined_rim_ml.py:verify.
Calls into: fea.deck (build_deck, deck_hash), fea.results (load_result), sparlab_form.
Start reading at: simulate, then run_deck.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._util import PathLike, PrecompError, read_json, write_json
from ..geometry.heightmap import HeightMap
from ..toolpath import Toolpath
from .deck import DECK_FILE, build_deck, deck_hash
from .results import FormingResult, load_result
from .setup import FormingSetup

OUTPUT_DIR = "output"
#: sparlab_form's exit statuses (docs/forming.md, section 3).
EXIT_CODES = {0: "completed", 2: "configuration error", 3: "a step stopped",
              4: "I/O error"}
LOG_FILE = "sparlab_form.log"
_VERSION_CACHE: Dict[Tuple[str, float], str] = {}


class FormingError(PrecompError):
    """A sparlab_form run failed; `record` holds the failure record."""

    def __init__(self, message: str, record: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.record = dict(record or {})


def sparlab_version(executable: PathLike) -> str:
    """The stripped output of `<executable> --version` (cached per file and mtime).

    Raises PrecompError when the executable is missing or the call fails: a
    run whose solver build is unknown cannot be cached safely.
    """
    exe = Path(executable)
    if not exe.is_file():
        raise PrecompError(f"{exe} does not exist")
    key = (str(exe.resolve()), exe.stat().st_mtime)
    if key in _VERSION_CACHE:
        return _VERSION_CACHE[key]
    try:
        proc = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PrecompError(f"{exe} --version failed: {exc}") from exc
    if proc.returncode != 0:
        raise PrecompError(f"{exe} --version exited with {proc.returncode}: "
                           f"{(proc.stderr or proc.stdout).strip()[-500:]}")
    text = (proc.stdout.strip() or proc.stderr.strip())
    if not text:
        raise PrecompError(f"{exe} --version printed nothing")
    _VERSION_CACHE[key] = text
    return text


def _tail(path: Path, chars: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-chars:]
    except FileNotFoundError:
        return ""


def run_deck(deck_dir: PathLike, output_dir: Optional[PathLike] = None, *,
             executable: PathLike, timeout: Optional[float] = None, threads: int = 1,
             provenance: Optional[Dict[str, Any]] = None) -> FormingResult:
    """Run sparlab_form on `deck_dir/deck.json` and load the result.

    The command is ``<executable> --config deck.json --output <output_dir>
    --strict-config`` (default `deck_dir/output`), run in `deck_dir` with
    OMP_NUM_THREADS = `threads`; stdout and stderr go to
    `deck_dir/sparlab_form.log`. A non-zero exit, a timeout [s] or an
    unreadable result raises FormingError with the tail of the log (and, for
    a run that stopped, the reason from summary.json). `run.json` in
    `deck_dir` records the command, exit code and runtime.
    """
    d = Path(deck_dir).resolve()
    if not (d / DECK_FILE).is_file():
        raise PrecompError(f"{d} holds no {DECK_FILE}")
    out = Path(output_dir).resolve() if output_dir is not None else d / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    exe = str(Path(executable).resolve())
    cmd = [exe, "--config", str(d / DECK_FILE), "--output", str(out), "--strict-config"]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(int(threads))
    log = d / LOG_FILE
    start = time.perf_counter()
    record: Dict[str, Any] = {"command": cmd, "threads": int(threads), "timeout_s": timeout}
    try:
        with open(log, "w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=d, stdout=handle, stderr=subprocess.STDOUT,
                                  env=env, timeout=timeout)
        record["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        record.update(returncode=None, reason=f"timed out after {timeout} s",
                      runtime_s=time.perf_counter() - start, log_tail=_tail(log))
        write_json(d / "run.json", record)
        raise FormingError(f"sparlab_form timed out after {timeout} s on {d}", record)
    except OSError as exc:
        record.update(returncode=None, reason=f"could not start: {exc}")
        write_json(d / "run.json", record)
        raise FormingError(f"sparlab_form could not be started: {exc}", record) from exc
    record["runtime_s"] = time.perf_counter() - start
    if proc.returncode != 0:
        meaning = EXIT_CODES.get(proc.returncode, "failed")
        record.update(reason=f"exit code {proc.returncode} ({meaning})", log_tail=_tail(log))
        if (out / "summary.json").is_file():
            try:
                termination = read_json(out / "summary.json").get("termination")
            except PrecompError:
                termination = None
            if termination:
                record["termination"] = str(termination)
                record["reason"] += f": {termination}"
        write_json(d / "run.json", record)
        raise FormingError(f"sparlab_form exited with {proc.returncode} ({meaning}) on {d}:\n"
                           f"{record['log_tail']}", record)
    write_json(d / "run.json", record)
    try:
        prov = dict(provenance or {})
        prov.setdefault("runtime_s", record["runtime_s"])
        return load_result(out, prov)
    except PrecompError as exc:
        record.update(reason=f"unreadable result: {exc}")
        write_json(d / "run.json", record)
        raise FormingError(f"sparlab_form finished but its result is unreadable: {exc}",
                           record) from exc


def _publish(staging: Path, final: Path) -> Path:
    """Move a finished staging directory to its cache entry atomically.

    If a completed entry is already there (another process won the race),
    the staging copy is discarded; a FAILED entry is replaced.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    if (final / "COMPLETE").is_file():
        shutil.rmtree(staging, ignore_errors=True)
        return final
    if final.exists():
        trash = final.with_name(final.name + f".old-{uuid.uuid4().hex[:8]}")
        os.replace(final, trash)
        shutil.rmtree(trash, ignore_errors=True)
    try:
        os.replace(staging, final)
    except OSError:
        if (final / "COMPLETE").is_file():
            shutil.rmtree(staging, ignore_errors=True)
        else:
            raise
    return final


def cache_entry(work_dir: PathLike, key: str) -> Path:
    """The cache directory of a run with content hash `key`."""
    return Path(work_dir) / "runs" / key[:2] / key


def simulate(setup: FormingSetup, commanded: HeightMap, work_dir: PathLike, *,
             cache: bool = True, retry_failed: bool = False,
             toolpath: Optional[Toolpath] = None,
             target: Optional[HeightMap] = None) -> FormingResult:
    """Simulate forming `commanded` with `setup`, through the run cache in `work_dir`.

    With `cache` an identical earlier run is returned without running (its
    provenance says `cache_hit: True`); without it the run is repeated and
    replaces the entry. A failure is recorded in the entry and raised as
    FormingError; later calls raise the recorded failure again unless
    `retry_failed`. The result's provenance holds the content hash (`key`),
    the solver version, the runtime and the cache directory. `target`: the
    part the fixture is made for (`build_deck`; only a support strategy
    reads it).
    """
    exe = setup.resolved_executable()
    version = sparlab_version(exe)
    work = Path(work_dir).resolve()
    staging = work / "staging" / uuid.uuid4().hex
    try:
        # Build the deck in staging; its content hash (with the solver version) is the cache key
        build_deck(setup, commanded, staging, toolpath=toolpath, target=target)
        key = deck_hash(staging, version)
        final = cache_entry(work, key)
        prov = {"key": key, "sparlab_version": version, "executable": str(exe),
                "cache_dir": str(final)}
        # Cache hit: return the stored result without running
        if cache and (final / "COMPLETE").is_file():
            shutil.rmtree(staging, ignore_errors=True)
            run = read_json(final / "run.json") if (final / "run.json").is_file() else {}
            prov.update(cache_hit=True, runtime_s=run.get("runtime_s"))
            return load_result(final / OUTPUT_DIR, prov)
        # A recorded failure is raised again unless retry_failed
        if cache and (final / "FAILED").is_file() and not retry_failed:
            shutil.rmtree(staging, ignore_errors=True)
            record = read_json(final / "FAILED")
            raise FormingError(f"this deck failed before ({record.get('reason', 'unknown')}); "
                               f"pass retry_failed=True to run it again. Entry: {final}",
                               record)
        # Run sparlab_form in staging; a failure is recorded and published as well
        prov["cache_hit"] = False
        try:
            result = run_deck(staging, executable=exe, timeout=setup.timeout,
                              threads=setup.threads, provenance=prov)
        except FormingError as exc:
            record = dict(exc.record)
            record.update(key=key, sparlab_version=version)
            write_json(staging / "FAILED", record)
            _publish(staging, final)
            raise
        # Mark the run complete and move it into the cache in one atomic rename
        (staging / "COMPLETE").write_text(key + "\n", encoding="ascii")
        entry = _publish(staging, final)
        return load_result(entry / OUTPUT_DIR, result.provenance)
    finally:
        if staging.exists() and not (staging / "COMPLETE").exists() \
                and not (staging / "FAILED").exists():
            shutil.rmtree(staging, ignore_errors=True)


@dataclass
class SimulationOutcome:
    """The outcome of one job of `simulate_many` (a light, picklable record).

    ok : whether a result exists; result_dir : its output directory;
    key : the content hash (None if the deck could not be built);
    error : the failure reason; cache_hit; runtime_s [s] (None for a cache hit
    whose runtime was not recorded).
    """

    index: int
    ok: bool
    result_dir: Optional[str] = None
    key: Optional[str] = None
    error: Optional[str] = None
    cache_hit: bool = False
    runtime_s: Optional[float] = None

    def load(self) -> FormingResult:
        """Load the result (PrecompError for a failed job)."""
        if not self.ok or self.result_dir is None:
            raise PrecompError(f"job {self.index} failed: {self.error}")
        return load_result(self.result_dir, {"key": self.key, "cache_hit": self.cache_hit,
                                             "runtime_s": self.runtime_s})


def _simulate_job(args: Tuple[int, Dict[str, Any], HeightMap, str, bool, bool,
                              Optional[HeightMap]]) -> SimulationOutcome:
    index, setup_doc, commanded, work_dir, cache, retry_failed, target = args
    try:
        setup = FormingSetup.from_dict(setup_doc)
        res = simulate(setup, commanded, work_dir, cache=cache, retry_failed=retry_failed,
                       target=target)
        p = res.provenance
        return SimulationOutcome(index, True, str(res.directory), p.get("key"), None,
                                 bool(p.get("cache_hit")), p.get("runtime_s"))
    except FormingError as exc:
        return SimulationOutcome(index, False, None, exc.record.get("key"), str(exc),
                                 False, exc.record.get("runtime_s"))
    except Exception as exc:  # recorded with its type, never dropped: one bad job must
        # not take the batch down (a data set records every failure and its reason)
        return SimulationOutcome(index, False, None, None, f"{type(exc).__name__}: {exc}")


def simulate_many(jobs: Sequence[Tuple[Any, ...]], work_dir: PathLike, *,
                  max_workers: Optional[int] = None, executor: str = "process",
                  cache: bool = True, retry_failed: bool = False) -> List[SimulationOutcome]:
    """Run many (setup, commanded) - or (setup, commanded, target), `target`
    the part a support's fixture is made for (`simulate`) - jobs in
    parallel through the cache.

    `executor` is "process" (a `ProcessPoolExecutor`: the deck building and
    tool-path generation in Python run in parallel too), "thread" or
    "serial". Every job returns a `SimulationOutcome` in input order; a failed
    job is recorded with its reason and never dropped. `max_workers` defaults
    to the CPU count; each solver run uses the setup's `threads`.
    """
    payload = []
    for i, job in enumerate(jobs):
        if len(job) not in (2, 3):
            raise ValueError("a job is (setup, commanded) or (setup, commanded, target)")
        setup, cmd = job[0], job[1]
        target = job[2] if len(job) == 3 else None
        payload.append((i, setup.to_dict(), cmd, str(Path(work_dir).resolve()), cache,
                        retry_failed, target))
    if executor == "serial" or len(payload) <= 1:
        return [_simulate_job(p) for p in payload]
    if executor == "process":
        pool: cf.Executor = cf.ProcessPoolExecutor(max_workers=max_workers)
    elif executor == "thread":
        pool = cf.ThreadPoolExecutor(max_workers=max_workers)
    else:
        raise ValueError("executor must be 'process', 'thread' or 'serial'")
    with pool:
        return list(pool.map(_simulate_job, payload))
