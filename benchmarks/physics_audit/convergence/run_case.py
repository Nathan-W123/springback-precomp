#!/usr/bin/env python3
"""Build and run one case of the numerical-convergence study.

A case is the benchmark's representative test part (truncated_cone-s2026-0000,
formed as commanded = target) with the benchmark setup
(benchmarks/springback/config.json, solver.setup) and a few discretisation
overrides. The deck is written by precomp (`FormingSetup`, `build_deck`) and
then patched for the settings `FormingSetup` does not expose:

  h        in-plane element size [mm]            (FormingSetup.element_size)
  L        element layers through the thickness  (FormingSetup.layers)
  form     "std" (Hex8, mean dilatation) | "im" (incompatible modes)
                                                  (model.element_formulation)
  tp       Gauss points through each layer, 0 = the default 2
                                                  (model.integration.thickness_points)
  travel   max tool travel per increment [mm]    (FormingSetup.max_tool_travel)
  pen      penalty scale s                       (FormingSetup.contact.penalty)
  solver_extra  keys merged into the deck's forming block (e.g. "newton")
  release_increments  increments of the unload and release steps (default 10)
  explicit None, or {"speed": m/s, "dt": target step s, "sc": contact_stiffness}
           (form_explicit was removed in the cleanup: such decks need a build of
           commit 7d571e6)
           -> the "form" step becomes a form_explicit step (selective,
              dynamic mass scaling) and the implicit unload / release follow.

Usage: run_case.py <case-name>      (cases are defined in cases.py)
Writes cases/<name>/{deck/, output/, run.json, DONE | FAILED}.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(HERE))

from cases import CASES, PARTS  # noqa: E402

EXE = REPO / "build" / "bin" / "sparlab_form"


def base_setup():
    from precomp.materials import get_material

    cfg = json.loads((REPO / "benchmarks" / "springback" / "config.json").read_text())
    s = dict(cfg["solver"]["setup"])
    s["material"] = get_material(s["material"])
    return s


def build(name: str, c: dict, out: Path) -> Path:
    from precomp.fea import FormingSetup, build_deck
    from precomp.geometry.heightmap import HeightMap

    commanded = HeightMap.load(PARTS[c.get("part", "t0")] / "commanded.npz")
    s = base_setup()
    s["element_size"] = c["h"] * 1e-3
    s["layers"] = c["L"]
    s["max_tool_travel"] = c.get("travel", 1.0) * 1e-3
    if c.get("pen", 10.0) != 10.0:
        s["contact"] = {"penalty": float(c["pen"])}
    s["executable"] = str(EXE)
    setup = FormingSetup(**s)
    deck_dir = out / "deck"
    build_deck(setup, commanded, deck_dir)
    deck = json.loads((deck_dir / "deck.json").read_text())
    model = deck.setdefault("model", {})
    if c.get("form", "std") == "im":
        model["element_formulation"] = "incompatible_modes"
    if c.get("tp", 0):
        model.setdefault("integration", {})["thickness_points"] = int(c["tp"])
        model["integration"]["thickness_direction"] = "z"
    fb = deck["forming"]
    fb["output"] = {"vtk": False, "snapshots": "steps"}
    if c.get("solver_extra"):
        fb.update(c["solver_extra"])
    if c.get("release_increments"):
        for st in fb["steps"][1:]:
            st["increments"] = int(c["release_increments"])
    ex = c.get("explicit")
    if ex:
        st = fb["steps"][0]
        assert st["name"] == "form" and st["type"] == "form"
        st.pop("max_tool_travel", None)
        st["type"] = "form_explicit"
        st["explicit"] = {
            "tool_speed": float(ex["speed"]),
            "mass_scaling": {"mode": "selective", "target_time_step": float(ex["dt"]),
                             "max_added_mass_fraction": 1000.0, "dynamic": True},
            "stable_step": {"method": "element_eigenvalue", "safety": 0.9,
                            "update_every": 1000},
            "damping": 0.0,
            "contact_stiffness": float(ex.get("sc", 0.5)),
            "history_every": 200,
        }
    (deck_dir / "deck.json").write_text(json.dumps(deck, indent=2) + "\n")
    return deck_dir


def main() -> int:
    name = sys.argv[1]
    c = CASES[name]
    out = HERE / "cases" / name
    if (out / "DONE").exists():
        print(f"{name}: done already")
        return 0
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "case.json").write_text(json.dumps(c, indent=2) + "\n")
    deck_dir = build(name, c, out)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               OMP_WAIT_POLICY="PASSIVE")
    cmd = [str(EXE), "--config", str(deck_dir / "deck.json"), "--output", str(out / "output"),
           "--threads", "1", "--strict-config"]
    load0 = os.getloadavg()
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out / "sparlab_form.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    rec = {"case": name, "params": c, "command": cmd, "returncode": rc, "wall_s": wall,
           "started_utc": started, "loadavg_start": load0, "loadavg_end": os.getloadavg(),
           "version": subprocess.run([str(EXE), "--version"], capture_output=True,
                                     text=True).stdout.strip()}
    (out / "run.json").write_text(json.dumps(rec, indent=2) + "\n")
    (out / ("DONE" if rc == 0 else "FAILED")).write_text(f"{rc}\n")
    print(f"{name}: rc={rc} wall={wall:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
