"""Process-lens variants (see proclens.py for the spec keys) and their queue order."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_rc = json.loads((HERE / "robot_compliance.json").read_text())


def _C(install, tool=0.30, reach=1.8, h=None):
    for p in _rc["poses"]:
        if p["install"] == install and p["tool_m"] == tool and p["reach_m"] == reach and \
                (h is None or p["height_m"] == h):
            return p["C_sheet_m_per_N"]
    raise KeyError(install)


C_VERT = _C("vertical_sheet", h=1.3)      # Machina-style cell, centre of the frame
C_HORIZ = _C("horizontal_sheet", h=0.3)   # sheet flat below the robot (lab cells)

TC = "truncated_cone-s2026-0000"
PY = "pyramid-s2026-0001"
EC = "elliptic_cone-s2026-0001"

# (part, name, spec, depends_on)
QUEUE = [
    (TC, "rob_iso1", {"robot": {"iso_um_per_N": 1.0}}, None),
    (TC, "scale2", {"scale": 2.0}, None),
    (TC, "rob_irb_vert", {"robot": {"C": C_VERT}}, None),
    (TC, "dsif_ng", {"dsif": {"thickness": "nogouge"}}, None),
    (TC, "rob_irb_horiz", {"robot": {"C": C_HORIZ}}, None),
    (TC, "rob_iso1_it2", {"robot": {"iso_um_per_N": 1.0, "forces_from": "rob_iso1"}}, "rob_iso1"),
    # --- second batch (priority order) ---
    # under-relaxed fixed point of the robot-sheet equilibrium p = p_cmd + C f(p):
    # it3 takes the mean of the rigid-path forces and it1's, it4 the mean of it3's
    # input and output forces; forces smoothed over 3 mm of path (contact noise)
    (TC, "rob_iso1_it3", {"robot": {"iso_um_per_N": 1.0, "smooth_m": 3e-3,
                                    "forces_mix": [["cache", 0.5], ["rob_iso1", 0.5]]}}, None),
    (TC, "rob_iso1_it4", {"robot": {"iso_um_per_N": 1.0, "smooth_m": 3e-3,
                                    "forces_mix": [["rob_iso1_it3:in", 0.5], ["rob_iso1_it3", 0.5]]}},
     "rob_iso1_it3"),
    (TC, "rob_iso1_it5", {"robot": {"iso_um_per_N": 1.0, "smooth_m": 3e-3,
                                    "forces_mix": [["rob_iso1_it4:in", 0.5], ["rob_iso1_it4", 0.5]]}},
     "rob_iso1_it4"),
    (PY, "dsif_ng", {"dsif": {"thickness": "nogouge"}}, None),
    (TC, "dsif_ng_gap02", {"dsif": {"thickness": "nogouge", "squeeze": -2.0e-4}}, None),
    (TC, "clamp8", {"setup": {"clamp_margin": 0.008}}, None),
    (TC, "blank60", {"setup": {"blank_size": 0.06}}, None),
    (EC, "dsif_ng", {"dsif": {"thickness": "nogouge"}}, None),
    (TC, "stepdown05", {"setup": {"step_down": 0.0005}}, None),
    (TC, "tool6", {"setup": {"tool_radius": 0.006}}, None),
    (PY, "rob_iso1", {"robot": {"iso_um_per_N": 1.0}}, None),
    (TC, "offset_z03", {"offset": [0.0, 0.0, 3.0e-4]}, None),
    (TC, "dsif_ng_gap05", {"dsif": {"thickness": "nogouge", "squeeze": -5.0e-4}}, None),
    (TC, "mat_5052H32", {"material": {"yield_stress": 193e6, "hardening_modulus": 50e6,
                                      "saturation_stress": 50e6, "saturation_rate": 25.0}}, None),
    (EC, "rob_iso1", {"robot": {"iso_um_per_N": 1.0}}, None),
    (PY, "blank60", {"setup": {"blank_size": 0.06}}, None),
]
