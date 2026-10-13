"""precomp - springback pre-compensation for robotic incremental sheet forming.

The Python layer around the SparLab forming solver: part geometry, tool paths,
the forming deck and its results, metrology and displacement-adjustment
compensation. Machine-learned surrogates live
in `precomp.ml`, which is built on the public API of the modules below.

Units are strict SI everywhere (m, N, Pa, s), as in SparLab
(`docs/conventions.md`); angles carry their unit in their name (`_deg`,
`_rad`). A millimetre value appears only in a human-facing report string,
labelled as such.

Layout
------
geometry      HeightMap (the tool-side surface as a height field), Grid,
              parametric part families
materials     Material (elasticity, J2 hardening, optional Hill48 and
              Armstrong-Frederick data), a nominal alloy library, SparLab JSON
toolpath      drop-cutter tool-centre surface, the spiral tool path, the
              DSIF support and edge-pass paths, the SparLab trajectory file
fea           FormingSetup, the sparlab_form deck, a content-addressed run
              cache, parallel runs, and the result loader
metrology     robust point-to-plane ICP, signed normal and vertical
              deviation, error metrics and masks
compensation  displacement adjustment over any predictor, predictor adapters

Where this fits: the package root; the workflow stages live in the modules above and precomp.ml.
Called from: every `import precomp...`; __version__ is read by fea/deck.py:build_deck,
    ml/registry.py:package_versions and common.py:versions.
Calls into: _util (PrecompError).
Start reading at: fea.runner.simulate, then compensation.displacement_adjustment.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._util import PrecompError

__all__ = ["PrecompError", "__version__"]
