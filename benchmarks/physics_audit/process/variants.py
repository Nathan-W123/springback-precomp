"""Process-lens variants (see proclens.py for the spec keys) and their queue order.

The robot-compliance variants (rob_*) and their scripts were removed with
precomp.robot; their committed results (robot_deflection.csv,
robot_compliance.json, ../challenge/robot_fixedpoint.json) remain as the
record, and the code is in the history at commit fe29724.
"""

TC = "truncated_cone-s2026-0000"
PY = "pyramid-s2026-0001"
EC = "elliptic_cone-s2026-0001"

# (part, name, spec, depends_on)
QUEUE = [
    (TC, "scale2", {"scale": 2.0}, None),
    (TC, "dsif_ng", {"dsif": {"thickness": "nogouge"}}, None),
    # --- second batch (priority order) ---
    (PY, "dsif_ng", {"dsif": {"thickness": "nogouge"}}, None),
    (TC, "dsif_ng_gap02", {"dsif": {"thickness": "nogouge", "squeeze": -2.0e-4}}, None),
    (TC, "clamp8", {"setup": {"clamp_margin": 0.008}}, None),
    (TC, "blank60", {"setup": {"blank_size": 0.06}}, None),
    (EC, "dsif_ng", {"dsif": {"thickness": "nogouge"}}, None),
    (TC, "stepdown05", {"setup": {"step_down": 0.0005}}, None),
    (TC, "tool6", {"setup": {"tool_radius": 0.006}}, None),
    (TC, "offset_z03", {"offset": [0.0, 0.0, 3.0e-4]}, None),
    (TC, "dsif_ng_gap05", {"dsif": {"thickness": "nogouge", "squeeze": -5.0e-4}}, None),
    (TC, "mat_5052H32", {"material": {"yield_stress": 193e6, "hardening_modulus": 50e6,
                                      "saturation_stress": 50e6, "saturation_rate": 25.0}}, None),
    (PY, "blank60", {"setup": {"blank_size": 0.06}}, None),
]
