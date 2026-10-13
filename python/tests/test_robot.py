"""Robot compliance: the Cartesian compliance and the forces along a path."""

import numpy as np
import pandas as pd
import pytest

from precomp.robot import CartesianCompliance, forces_on_path
from precomp.toolpath import Toolpath


def _path(n=50):
    s = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([0.03 * np.cos(s), 0.03 * np.sin(s), -0.01 * s / (2 * np.pi)])


def test_compliance_from_joint_stiffness():
    J = np.array([[0.5, 0.2, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    K = np.array([1e5, 2e5, 5e4])
    C = CartesianCompliance.from_joint_stiffness(J, K)
    assert np.allclose(C.C, J @ np.diag(1 / K) @ J.T)
    assert np.allclose(C.C, C.C.T)
    with pytest.raises(ValueError):
        CartesianCompliance(np.eye(2))


def test_forces_on_path_interpolate_the_history_with_the_stated_sign():
    pts = _path()
    path = Toolpath(pts, np.ones(len(pts), int), 0.005)
    # sparlab_form's convention: the force the sheet exerts on the tool, fz > 0
    hist = pd.DataFrame({"step": 1, "increment": range(3), "t": [0.0, 0.5, 1.0], "tool": "tool",
                         "fx": [0.0, -10.0, -20.0], "fy": 0.0, "fz": [100.0, 300.0, 500.0]})
    f = forces_on_path(path, hist, tool="tool")
    assert np.allclose(f[:, 2], 100.0 + 400.0 * path.t)       # taken as it is
    g = forces_on_path(path, hist.assign(fx=-hist["fx"], fz=-hist["fz"]), sign="on_sheet")
    assert np.allclose(g, f)                                  # a force on the sheet, negated
    C = CartesianCompliance(np.diag([1e-6, 1e-6, 2e-6]))
    assert np.allclose(C.deflection(f), f * [1e-6, 1e-6, 2e-6])   # delta = C f
