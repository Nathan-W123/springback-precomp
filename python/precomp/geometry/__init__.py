"""Geometry: the tool-side surface as a height field, and the part families.

See `heightmap` for the surface convention (flat blank at z = 0, formed part
z <= 0, arrays indexed [y, x] with x fastest) and `parts` for the parametric
target families.

Where this fits: re-exports the geometry of "Design of the parts" (Grid, HeightMap, the part
    families).
Called from: the Python tests only (python/tests); package code imports geometry.heightmap
    and geometry.parts directly.
Calls into: geometry.heightmap, geometry.parts.
Start reading at: heightmap.HeightMap, then parts.Part.
"""

from .heightmap import Grid, HeightMap, zeros
from .parts import (
    MAX_WALL_ANGLE_DEG,
    Dome,
    EllipticCone,
    FilletedProfile,
    Part,
    Pyramid,
    TruncatedCone,
    ellipse_distance,
    families,
    part_from_dict,
    rounded_rectangle_distance,
)

__all__ = [
    "Grid", "HeightMap", "zeros",
    "MAX_WALL_ANGLE_DEG", "Part", "TruncatedCone", "Pyramid", "Dome", "EllipticCone",
    "FilletedProfile", "ellipse_distance",
    "rounded_rectangle_distance", "families", "part_from_dict",
]
