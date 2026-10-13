"""Robot compliance: how far the forming force pushes the tool off its path.

An industrial robot holding a forming tool is compliant: the forming force
deflects the tool-centre point by ``delta = C f`` with C the 3 x 3 Cartesian
compliance at the working pose [m/N] and f the force the sheet exerts on the
tool [N] (the reaction of the forming force). The physics audit
(benchmarks/physics_audit/process) uses it to bound the error the simulation
leaves out by modelling the robot as rigid.

Forces from a simulation: `forces_on_path` interpolates the tool force
history of a sparlab_form result at the path's pseudo-time. sparlab_form's
tool_forces.csv holds the force the sheet exerts on the tool (fz > 0 while
the tool pushes down; docs/forming.md, section 3) - the load on the robot -
which is taken as it is; `sign="on_sheet"` negates a history that holds the
force on the sheet instead (another solver's convention).

A single constant C is a first-order model: the compliance of a serial robot
changes with its pose. `CartesianCompliance.from_joint_stiffness` computes C
for one pose; a pose-dependent map needs one C per path point (pass an
(n, 3, 3) array to `deflection`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ._util import PrecompError
from .toolpath import Toolpath


@dataclass
class CartesianCompliance:
    """Translational compliance C [m/N] of the tool-centre point (3 x 3, or
    (n, 3, 3) for one matrix per path point)."""

    C: np.ndarray

    def __post_init__(self) -> None:
        C = np.asarray(self.C, dtype=float)
        if C.shape[-2:] != (3, 3) or C.ndim not in (2, 3):
            raise ValueError(f"compliance must be 3 x 3 or (n, 3, 3), got {C.shape}")
        if not np.all(np.isfinite(C)):
            raise ValueError("compliance must be finite")
        self.C = C


    @classmethod
    def from_joint_stiffness(cls, jacobian: np.ndarray,
                             joint_stiffness: np.ndarray) -> "CartesianCompliance":
        """C = J_v K^-1 J_v^T from the translational rows J_v (3 x n) [m/rad]
        of the robot Jacobian at the pose and the joint stiffnesses K (n,)
        [N m/rad] - the classical joint-compliance model (links rigid)."""
        J = np.asarray(jacobian, dtype=float)
        if J.shape[0] == 6:
            J = J[:3]
        K = np.asarray(joint_stiffness, dtype=float)
        if J.ndim != 2 or J.shape[0] != 3 or K.shape != (J.shape[1],):
            raise ValueError("jacobian must be 3 x n (or 6 x n) and joint_stiffness (n,)")
        if np.any(K <= 0):
            raise ValueError("joint stiffnesses must be > 0")
        return cls(J @ np.diag(1.0 / K) @ J.T)

    def deflection(self, forces: np.ndarray) -> np.ndarray:
        """(n, 3) tool-centre deflections C f [m] under forces (n, 3) [N]
        acting on the tool."""
        f = np.asarray(forces, dtype=float)
        if self.C.ndim == 2:
            return f @ self.C.T
        if self.C.shape[0] != len(f):
            raise ValueError("one compliance matrix per force is required")
        return np.einsum("nij,nj->ni", self.C, f)


def forces_on_path(path: Toolpath, tool_forces: pd.DataFrame, *, tool: Optional[str] = None,
                   step: Optional[int] = None, sign: str = "on_tool") -> np.ndarray:
    """Forces on the tool (n, 3) [N] at the points of `path`, from a force history.

    `tool_forces` is a FormingResult's tool force table (columns t, tool,
    step, fx, fy, fz); rows are filtered by `tool` and `step` (default: the
    first step, the forming one) and interpolated linearly in the pseudo-time
    t at the path's `t`. `sign`: "on_tool" (default, sparlab_form's
    convention) when fx, fy, fz are the force the sheet exerts on the tool,
    "on_sheet" when they are the force the tool exerts on the sheet (the
    result is then their negative, the load on the robot).
    """
    frame = tool_forces
    if tool is not None:
        frame = frame[frame["tool"] == tool]
    if step is None:
        step = int(frame["step"].min())
    frame = frame[frame["step"] == step].sort_values("t")
    if len(frame) < 2:
        raise PrecompError("the force history has fewer than two rows for this tool and step")
    t = frame["t"].to_numpy(float)
    if np.any(np.diff(t) < 0):
        raise PrecompError("the force history's t is not monotone")
    f = frame[["fx", "fy", "fz"]].to_numpy(float)
    if sign == "on_sheet":
        f = -f
    elif sign != "on_tool":
        raise ValueError("sign must be 'on_sheet' or 'on_tool'")
    tp = path.t
    return np.column_stack([np.interp(tp, t, f[:, i]) for i in range(3)])
