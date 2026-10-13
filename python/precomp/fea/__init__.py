"""The bridge to SparLab's forming solver (sparlab_form).

setup    FormingSetup - process, mesh, material and solver settings
deck     build_deck - deck.json + toolpath.csv; deck_hash - the content hash
runner   run_deck, simulate (content-addressed cache), simulate_many (pool)
results  load_result, FormingResult - the validated result loader

Where this fits: the public API of "Simulations" (setup -> deck -> runner -> results).
Called from: springback_fine_ml.py:make_setup, combined_rim_ml.py:edge_setup (FormingSetup),
    common.py:versions (sparlab_version).
Calls into: fea.setup, fea.deck, fea.runner, fea.results.
Start reading at: runner.simulate, then deck.build_deck.
"""

from .deck import (
    DECK_FILE,
    TOOLPATH_FILE,
    build_deck,
    clamp_condition,
    deck_document,
    deck_hash,
    forming_block,
    make_toolpath,
    support_321,
)
from .results import (
    ELEMENT_COLUMNS,
    FORCE_COLUMNS,
    NODE_COLUMNS,
    FormingResult,
    StepResult,
    load_result,
)
from .runner import (
    FormingError,
    SimulationOutcome,
    cache_entry,
    run_deck,
    simulate,
    simulate_many,
    sparlab_version,
)
from .setup import DEFAULT_EXECUTABLE, EXECUTABLE_ENV, PRESETS, FormingSetup

__all__ = [
    "FormingSetup", "EXECUTABLE_ENV", "DEFAULT_EXECUTABLE", "PRESETS",
    "build_deck", "deck_document", "deck_hash", "forming_block", "clamp_condition",
    "support_321", "make_toolpath", "DECK_FILE", "TOOLPATH_FILE",
    "FormingResult", "StepResult", "load_result", "NODE_COLUMNS", "ELEMENT_COLUMNS",
    "FORCE_COLUMNS",
    "run_deck", "simulate", "simulate_many", "SimulationOutcome", "FormingError",
    "sparlab_version", "cache_entry",
]
