"""A triangulated surface as a height field.

`raycast_top` casts a ray along -z through each node of a grid and keeps the
highest hit: the tool-side surface of a part, whatever the triangulation.
`precomp.fea.results` uses it to read a formed sheet's top surface off the
deformed mesh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .heightmap import Grid

#: Triangle pairs per chunk of the ray cast: bounds its memory (~100 MB).
_RAYCAST_CHUNK_PAIRS = 2_000_000


def raycast_top(triangles: np.ndarray, grid: "Grid") -> Tuple[np.ndarray, np.ndarray]:
    """Top-most intersection of a vertical ray through every grid node.

    Parameters
    ----------
    triangles : (m, 3, 3) corners [m].
    grid : the nodes to cast through.

    Returns
    -------
    z : (ny, nx) height of the highest triangle above each node [m] (-inf
        where nothing is hit).
    hit : (ny, nx) bool.

    Each triangle is tested against the nodes inside its x-y bounding box
    with barycentric coordinates (a node on a shared edge or vertex is inside
    both neighbours, to a relative 1e-6 - enough for the float32 corners of a
    binary STL); triangles seen edge-on from above
    (zero x-y area) are skipped - their edges belong to neighbouring facets.
    The work is vectorised over (triangle, node) pairs in bounded chunks.
    """
    tri = np.asarray(triangles, dtype=float)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangles must be an (m, 3, 3) array")
    h = grid.h
    zbuf = np.full(grid.nx * grid.ny, -np.inf)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0 = b[:, :2] - a[:, :2]
    v1 = c[:, :2] - a[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    scale = np.maximum(np.abs(v0).max(axis=1), np.abs(v1).max(axis=1))
    keep = np.abs(den) > 1e-12 * np.maximum(scale, 1e-300) ** 2
    xmin = tri[:, :, 0].min(axis=1)
    xmax = tri[:, :, 0].max(axis=1)
    ymin = tri[:, :, 1].min(axis=1)
    ymax = tri[:, :, 1].max(axis=1)
    eps = 1e-6
    i0 = np.clip(np.ceil((xmin - grid.x0) / h - eps), 0, grid.nx - 1).astype(np.int64)
    i1 = np.clip(np.floor((xmax - grid.x0) / h + eps), -1, grid.nx - 1).astype(np.int64)
    j0 = np.clip(np.ceil((ymin - grid.y0) / h - eps), 0, grid.ny - 1).astype(np.int64)
    j1 = np.clip(np.floor((ymax - grid.y0) / h + eps), -1, grid.ny - 1).astype(np.int64)
    wi = np.maximum(i1 - i0 + 1, 0)
    wj = np.maximum(j1 - j0 + 1, 0)
    wi = np.where((xmax < grid.x0) | (xmin > grid.xmax), 0, wi)
    wj = np.where((ymax < grid.y0) | (ymin > grid.ymax), 0, wj)
    counts = np.where(keep, wi * wj, 0)
    ids = np.flatnonzero(counts)
    if len(ids) == 0:
        return zbuf.reshape(grid.shape), np.zeros(grid.shape, dtype=bool)
    csum = np.cumsum(counts[ids])
    start = 0
    while start < len(ids):
        base = csum[start - 1] if start else 0
        stop = int(np.searchsorted(csum, base + _RAYCAST_CHUNK_PAIRS, side="right"))
        stop = max(stop, start + 1)
        sel = ids[start:stop]
        cnt = counts[sel]
        t = np.repeat(sel, cnt)
        offs = np.repeat(np.cumsum(cnt) - cnt, cnt)
        k = np.arange(int(cnt.sum())) - offs
        ii = i0[t] + k % wi[t]
        jj = j0[t] + k // wi[t]
        px = grid.x0 + ii * h - a[t, 0]
        py = grid.y0 + jj * h - a[t, 1]
        d = den[t]
        u = (px * v1[t, 1] - v1[t, 0] * py) / d
        v = (v0[t, 0] * py - px * v0[t, 1]) / d
        tol = 1e-6
        inside = (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol)
        zz = a[t, 2] + u * (b[t, 2] - a[t, 2]) + v * (c[t, 2] - a[t, 2])
        flat = (jj * grid.nx + ii)[inside]
        np.maximum.at(zbuf, flat, zz[inside])
        start = stop
    z = zbuf.reshape(grid.shape)
    return z, np.isfinite(z)
