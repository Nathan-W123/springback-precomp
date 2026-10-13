"""Measured parts: alignment, signed deviation, error metrics and masks.

Units. Every function works in metres (a scan in millimetres is scaled by
0.001 before it gets here).

Alignment. `align` registers a scan to the target height field with
point-to-plane ICP: each point's residual is its signed distance along the
target normal to the tangent plane at the point of the target vertically
below it, ``r = n . (p - (x, y, z_t(x, y)))``; zero exactly on the surface.
The update minimises the robustly weighted sum of squared residuals
linearised in a small rotation about the weighted centroid and a translation
(iteratively re-weighted least squares with Tukey's biweight or Huber's
weights on a MAD scale, plus optional trimming). Degrees of freedom the data
cannot fix - x, y and the rotation about z of a flat flange - get a zero
update (truncated least squares, `OBSERVABILITY_RCOND`), and the number of
fixed ones is reported as `rank`. The rotation of an axisymmetric part about
its axis is such a direction: a scan of a truncated cone cannot fix it.

Deviation. `signed_deviation` measures, at every target grid node, the
distance along the target normal to the formed surface: positive when the
formed surface lies on the tool side (+normal) of the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.spatial import cKDTree

from ._util import PrecompError
from .geometry.heightmap import HeightMap, _fill_nearest


#: Relative singular value of the ICP least-squares matrix (rotation columns
#: scaled by the cloud size) below which a rigid direction counts as
#: unobservable and is not updated. Measured: the rotation of an
#: axisymmetric cone about its axis sits at 9e-5 (the grid breaks the
#: symmetry only slightly), in-plane motion over a flat flange at 5e-5 to
#: 8e-4, while an ellipse of axis ratio 1.06 still fixes its rotation at 5e-3.
OBSERVABILITY_RCOND = 2e-3


# ---------------------------------------------------------------------------
# Rigid transforms
# ---------------------------------------------------------------------------
def rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector `axis_angle` [rad] (Rodrigues)."""
    w = np.asarray(axis_angle, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-300:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of the rotation matrix R [deg]."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(math.degrees(math.acos(min(1.0, max(-1.0, c)))))


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """The 4x4 matrix of x -> R x + t."""
    M = np.eye(4)
    M[:3, :3] = rotation
    M[:3, 3] = translation
    return M


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
@dataclass
class Alignment:
    """Result of `align`: aligned = rotation @ p + translation, all in metres.

    residuals : (n,) final point-to-plane residuals [m] (NaN off the target or
        outside the fixture region); weights : (n,) final robust weights.
    rms : weighted RMS residual of the points used [m]; rms_inliers : RMS over
        points with non-zero weight [m]; n_used, n_inliers : counts.
    rank : number of rigid degrees of freedom the data fixed (6 for a full
        rigid fit on a curved surface, 3 on a flat flange, 1 for mode
        translation_z).
    history : weighted RMS per iteration [m].
    """

    rotation: np.ndarray
    translation: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray
    rms: float
    rms_inliers: float
    n_used: int
    n_inliers: int
    iterations: int
    converged: bool
    rank: int
    mode: str
    history: List[float] = field(default_factory=list)

    @property
    def matrix(self) -> np.ndarray:
        """The 4x4 homogeneous transform."""
        return make_transform(self.rotation, self.translation)

    @property
    def rotation_deg(self) -> float:
        return rotation_angle_deg(self.rotation)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform (n, 3) points into the target frame."""
        return np.asarray(points, float) @ self.rotation.T + self.translation

    def summary(self) -> Dict[str, Any]:
        return {"mode": self.mode, "rms_m": self.rms, "rms_inliers_m": self.rms_inliers,
                "n_used": self.n_used, "n_inliers": self.n_inliers,
                "iterations": self.iterations, "converged": self.converged,
                "rank": self.rank, "rotation_deg": self.rotation_deg,
                "translation_m": [float(v) for v in self.translation],
                "matrix": self.matrix.tolist()}


def _robust_weights(r: np.ndarray, robust: str, floor: float) -> np.ndarray:
    med = np.median(r)
    sigma = max(1.4826 * float(np.median(np.abs(r - med))), floor)
    a = np.abs(r)
    if robust == "tukey":
        c = 4.685 * sigma
        w = np.where(a < c, (1.0 - (a / c) ** 2) ** 2, 0.0)
    elif robust == "huber":
        k = 1.345 * sigma
        w = np.where(a <= k, 1.0, k / np.maximum(a, 1e-300))
    elif robust == "none":
        w = np.ones_like(a)
    else:
        raise ValueError("robust must be 'tukey', 'huber' or 'none'")
    return w


class _Target:
    """Target height field with its normals and an optional fixture mask."""

    def __init__(self, target: HeightMap, fixture_mask: Optional[np.ndarray],
                 normal_sigma: Optional[float]):
        self.hm = target
        self.normals = target.normals(normal_sigma)
        use = target.mask.copy()
        if fixture_mask is not None:
            fm = np.asarray(fixture_mask, dtype=bool)
            if fm.shape != target.grid.shape:
                raise ValueError(f"fixture_mask has shape {fm.shape}, the target grid "
                                 f"{target.grid.shape}")
            use &= fm
            if not use.any():
                raise ValueError("fixture_mask selects no valid target node")
        self.use = use.astype(float)

    def residuals(self, q: np.ndarray):
        zt = self.hm.interpolate(q[:, 0], q[:, 1], masked=True)
        ok = np.isfinite(zt)
        if not np.all(self.use == 1.0):
            ind = self.hm.sample(self.use, q[:, 0], q[:, 1], fill_value=0.0)
            ok &= ind > 1.0 - 1e-9
        n = self.hm.sample(self.normals, q[:, 0], q[:, 1])
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        r = np.full(len(q), np.nan)
        r[ok] = n[ok, 2] * (q[ok, 2] - zt[ok])
        return r, n, ok


def _icp(points: np.ndarray, tgt: _Target, R: np.ndarray, t: np.ndarray, mode: str,
         robust: str, trim_fraction: float, max_iterations: int, tolerance: float,
         length: float):
    history: List[float] = []
    converged = False
    rank = 0
    floor = 1e-9 * length
    it = 0
    w_full = np.zeros(len(points))
    r = np.full(len(points), np.nan)
    for it in range(1, max_iterations + 1):
        q = points @ R.T + t
        r, n, ok = tgt.residuals(q)
        if ok.sum() < (6 if mode == "rigid" else 1):
            raise PrecompError(f"alignment: only {int(ok.sum())} points lie on the target; "
                               "improve the initial guess or check the units")
        ro = r[ok]
        w = _robust_weights(ro, robust, floor)
        if trim_fraction > 0:
            cut = np.quantile(np.abs(ro), 1.0 - trim_fraction)
            w = np.where(np.abs(ro) <= cut, w, 0.0)
        w_full = np.zeros(len(points))
        w_full[ok] = w
        wsum = float(w.sum())
        if wsum <= 0:
            raise PrecompError("alignment: every point was rejected as an outlier")
        history.append(float(np.sqrt(np.sum(w * ro * ro) / wsum)))
        if mode == "none":
            converged = True
            break
        sw = np.sqrt(w)
        if mode == "translation_z":
            J = n[ok, 2:3]
            scale = np.array([1.0])
        else:
            qo = q[ok]
            c = (w[:, None] * qo).sum(axis=0) / wsum
            J = np.column_stack([np.cross(qo - c, n[ok]), n[ok]])
            scale = np.array([length, length, length, 1.0, 1.0, 1.0])
        A = sw[:, None] * J / scale
        b = -sw * ro
        sol, _, rank, sv = np.linalg.lstsq(A, b, rcond=OBSERVABILITY_RCOND)
        delta = sol / scale
        if mode == "translation_z":
            t = t + np.array([0.0, 0.0, delta[0]])
            step = abs(delta[0])
        else:
            dR = rotation_matrix(delta[:3])
            R = dR @ R
            t = dR @ (t - c) + c + delta[3:]
            step = max(float(np.linalg.norm(delta[:3])) * length, float(np.linalg.norm(delta[3:])))
        if step < tolerance:
            converged = True
            break
    q = points @ R.T + t
    r, n, ok = tgt.residuals(q)
    return R, t, r, w_full, it, converged, int(rank) if mode != "none" else 0, history


def _principal_frame(pts: np.ndarray):
    c = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - c).T))
    return c, vecs[:, ::-1]


def align(points: np.ndarray, target: HeightMap, mode: str = "rigid", *,
          fixture_mask: Optional[np.ndarray] = None, robust: str = "tukey",
          trim_fraction: float = 0.0, max_iterations: int = 100, tolerance: float = 1e-10,
          initial: Union[str, np.ndarray] = "auto",
          normal_sigma: Optional[float] = None) -> Alignment:
    """Register a scan (n, 3) [m] to `target` by robust point-to-plane ICP.

    mode : "rigid" (6 DOF), "translation_z" (the z offset only - a part
        scanned in its fixture) or "none" (residuals of the scan as it is).
    fixture_mask : (ny, nx) bool on the target grid; only points above it are
        used (e.g. `flange_mask(target)` to align on the clamped flange, as the
        forming cell fixtures the part). Unobservable DOFs are left at the
        initial guess and `rank` says how many were fixed.
    robust : "tukey" (default; zero weight beyond 4.685 MAD-sigmas), "huber"
        or "none"; trim_fraction additionally drops that share of the largest
        residuals each iteration.
    max_iterations, tolerance : stop when the update moves no point by more
        than `tolerance` [m] (rotation measured at the cloud's size).
    initial : "identity", "centroid" (translate the centroids together),
        "pca" (principal axes, the best of the four proper sign choices),
        "auto" (identity and the PCA candidates, each refined 10 iterations;
        the lowest median |residual| wins, and a PCA candidate must halve
        identity's - on a symmetric part, or with a flat fixture, they tie
        and identity must win) or a 4x4 matrix.
    normal_sigma : smoothing length [m] of the target normals (None: raw).

    Assumes the scan is of the tool-side surface, as a height field, in
    metres. Returns an `Alignment`.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("points must be an (n >= 3, 3) array")
    if mode not in ("rigid", "translation_z", "none"):
        raise ValueError("mode must be 'rigid', 'translation_z' or 'none'")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")
    tgt = _Target(target, fixture_mask, normal_sigma)
    length = float(np.linalg.norm(np.ptp(pts, axis=0))) or 1.0

    candidates = []
    if isinstance(initial, np.ndarray):
        M = np.asarray(initial, dtype=float)
        if M.shape != (4, 4):
            raise ValueError("an initial transform must be a 4x4 matrix")
        candidates.append((M[:3, :3].copy(), M[:3, 3].copy()))
    elif initial in ("identity", "auto") or mode == "none":
        candidates.append((np.eye(3), np.zeros(3)))
    if isinstance(initial, str) and initial in ("centroid", "pca", "auto") and mode == "rigid":
        tp = target.to_points()
        if fixture_mask is not None:
            tp = tp[np.asarray(fixture_mask, bool)[target.mask]]
        cp, Vp = _principal_frame(pts)
        ct, Vt = _principal_frame(tp)
        if initial == "centroid":
            candidates.append((np.eye(3), ct - cp))
        else:
            for signs in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)):
                Rc = Vt @ np.diag(signs) @ Vp.T
                if np.linalg.det(Rc) < 0:
                    Rc = Vt @ np.diag(signs) @ np.diag([1, 1, -1]) @ Vp.T
                candidates.append((Rc, ct - Rc @ cp))
    elif isinstance(initial, str) and initial not in ("identity", "auto", "centroid", "pca"):
        raise ValueError(f"unknown initial guess {initial!r}")
    if mode == "translation_z" and isinstance(initial, str):
        zt = target.interpolate(pts[:, 0], pts[:, 1])
        ok = np.isfinite(zt)
        if ok.any():
            candidates = [(np.eye(3), np.array([0.0, 0.0, float(np.median(zt[ok] - pts[ok, 2]))]))]

    if len(candidates) > 1:
        scored = []
        for i, (Rc, tc) in enumerate(candidates):
            try:
                Rr, tr, r, *_ = _icp(pts, tgt, Rc, tc, mode, robust, trim_fraction, 10,
                                     tolerance, length)
            except PrecompError:
                continue
            ok = np.isfinite(r)
            score = float(np.median(np.abs(r[ok])))
            if i == 0 and initial == "auto":
                score *= 0.5      # a PCA candidate must beat identity clearly (ties: symmetry)
            scored.append((score, int(ok.sum()), Rr, tr))
        if not scored:
            raise PrecompError("alignment: no initial guess puts enough points on the target "
                               "(or its fixture region); check the units and the extent")
        # A candidate that slides most points off the target (or off the
        # fixture region) can fit the few left perfectly: require at least
        # half the best coverage.
        best_cover = max(c for _, c, _, _ in scored)
        scored = [s for s in scored if s[1] >= 0.5 * best_cover]
        scored.sort(key=lambda s: s[0])
        R0, t0 = scored[0][2], scored[0][3]
    else:
        R0, t0 = candidates[0]
    R, t, r, w, it, conv, rank, hist = _icp(pts, tgt, R0, t0, mode, robust, trim_fraction,
                                            max_iterations, tolerance, length)
    ok = np.isfinite(r)
    wi = w > 0
    rms = float(np.sqrt(np.sum(w[ok] * r[ok] ** 2) / max(w[ok].sum(), 1e-300)))
    rms_in = float(np.sqrt(np.mean(r[wi] ** 2))) if wi.any() else float("nan")
    return Alignment(R, t, r, w, rms, rms_in, int(ok.sum()), int(wi.sum()), it, conv, rank,
                     mode, hist)


# ---------------------------------------------------------------------------
# Deviation
# ---------------------------------------------------------------------------
def signed_deviation(formed: Union[HeightMap, np.ndarray], target: HeightMap, *,
                     max_distance: Optional[float] = None,
                     normal_sigma: Optional[float] = None,
                     mask: Optional[np.ndarray] = None, neighbours: int = 8,
                     iterations: int = 20) -> HeightMap:
    """Signed normal deviation [m] of `formed` from `target` at the target nodes.

    For each valid target node P with unit normal n (pointing +z, towards the
    tool; smoothed at `normal_sigma` [m] if given) the deviation is the s for
    which P + s n lies on the formed surface - positive when the formed
    surface is on the tool side of the target.

    * `formed` a HeightMap: s solves ``z_P + s n_z = z_f(x_P + s n_x, y_P +
      s n_y)`` by Newton's method on the bilinear formed height field, from
      the first-order guess s = n_z (z_f - z_P), with `iterations` steps.
      Nodes whose line leaves the formed data or does not converge (|residual|
      above 1e-10 m) are invalid.
    * `formed` an (m, 3) point cloud: the `neighbours` nearest cloud points
      of P are fitted with a plane (least squares) and the line is
      intersected with it; where the line runs nearly parallel to that plane
      (|cos| < 0.2) the signed nearest-point distance is used instead. Nodes
      farther than 5 median point spacings from the cloud (a hole) are
      invalid. The plane misses a surface of curvature kappa by about
      kappa s^2 / 2 at point spacing s - a few um on gently curved walls,
      tens of um in tight fillets - so for dense scans gridding the cloud
      first (`HeightMap.from_points`) is the more accurate route.

    `max_distance` [m] invalidates larger |s|; `mask` (ny, nx) restricts the
    nodes. The result is on the target grid, metadata {"quantity":
    "signed_normal_deviation", "unit": "m"}; the discretisation error is that
    of the interpolation (bilinear, or planar over the neighbours).
    """
    grid = target.grid
    X, Y = grid.mesh()
    n = target.normals(normal_sigma)
    valid = target.mask.copy()
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(valid.ravel())
    P = np.column_stack([X.ravel()[idx], Y.ravel()[idx], target.z.ravel()[idx]])
    N = n.reshape(-1, 3)[idx]
    s = np.full(len(idx), np.nan)
    ok = np.zeros(len(idx), dtype=bool)
    meta: Dict[str, Any] = {"quantity": "signed_normal_deviation", "unit": "m",
                            "sign": "positive: formed surface on the tool side of the target"}
    if isinstance(formed, HeightMap):
        gx, gy = formed.gradient()
        zf = formed.interpolate(P[:, 0], P[:, 1])
        s = N[:, 2] * (zf - P[:, 2])
        for _ in range(iterations):
            x = P[:, 0] + s * N[:, 0]
            y = P[:, 1] + s * N[:, 1]
            zf = formed.interpolate(x, y)
            G = P[:, 2] + s * N[:, 2] - zf
            dG = N[:, 2] - (formed.sample(gx, x, y) * N[:, 0] + formed.sample(gy, x, y) * N[:, 1])
            s = s - G / np.where(np.abs(dG) > 1e-6, dG, np.nan)
        x = P[:, 0] + s * N[:, 0]
        y = P[:, 1] + s * N[:, 1]
        G = P[:, 2] + s * N[:, 2] - formed.interpolate(x, y)
        ok = np.isfinite(s) & np.isfinite(G) & (np.abs(G) <= 1e-10)
        meta["unconverged"] = int(np.sum(np.isfinite(s) & ~ok))
    else:
        cloud = np.asarray(formed, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < neighbours:
            raise ValueError("a formed point cloud must be an (m >= neighbours, 3) array")
        tree = cKDTree(cloud)
        spacing = float(np.median(tree.query(cloud, k=2)[0][:, 1]))
        dist, nb = tree.query(P, k=neighbours)
        Q = cloud[nb]                                      # (k, neighbours, 3)
        c = Q.mean(axis=1)
        D = Q - c[:, None, :]
        cov = np.einsum("kni,knj->kij", D, D)
        _, vecs = np.linalg.eigh(cov)
        m = vecs[:, :, 0]
        m = np.where((m[:, 2:3] < 0), -m, m)
        cosang = np.einsum("ki,ki->k", m, N)
        s_plane = np.einsum("ki,ki->k", m, c - P) / np.where(np.abs(cosang) > 0.2, cosang, np.nan)
        near = Q[:, 0, :]
        s_near = np.sign(np.einsum("ki,ki->k", N, near - P)) * dist[:, 0]
        s = np.where(np.abs(cosang) > 0.2, s_plane, s_near)
        ok = dist[:, 0] <= 5.0 * spacing
        meta["cloud_spacing_m"] = spacing
    if max_distance is not None:
        ok &= np.abs(s) <= max_distance
    ok &= np.isfinite(s)
    dev = np.zeros(grid.shape)
    good = np.zeros(grid.shape, dtype=bool)
    dev.ravel()[idx[ok]] = s[ok]
    good.ravel()[idx[ok]] = True
    if not good.any():
        raise PrecompError("signed_deviation: no target node has a defined deviation")
    return HeightMap(grid, _fill_nearest(dev, good), good, meta)


def vertical_deviation(formed: HeightMap, target: HeightMap) -> HeightMap:
    """z_formed - z_target [m] at the target nodes (formed interpolated
    bilinearly); invalid where either is."""
    X, Y = target.grid.mesh()
    zf = formed.interpolate(X, Y)
    good = np.isfinite(zf) & target.mask
    if not good.any():
        raise PrecompError("vertical_deviation: the surfaces do not overlap")
    dev = np.where(good, zf - target.z, 0.0)
    return HeightMap(target.grid, _fill_nearest(dev, good), good,
                     {"quantity": "vertical_deviation", "unit": "m"})


# ---------------------------------------------------------------------------
# Metrics and regions
# ---------------------------------------------------------------------------
def metrics(dev: Union[HeightMap, np.ndarray], mask: Optional[np.ndarray] = None,
            tolerance: Optional[float] = None) -> Dict[str, float]:
    """Error statistics of a deviation field [m].

    Uses the valid nodes of a HeightMap (or the finite entries of an array),
    restricted to `mask`. Returns rms, mae, max_abs, p95_abs (95th percentile
    of |d|), bias (mean), std [m], n (count) and, with `tolerance` [m],
    pct_within_tol - the percentage (0-100) of nodes with |d| <= tolerance.
    ValueError when no node is selected.
    """
    if isinstance(dev, HeightMap):
        values = dev.z
        sel = dev.mask.copy()
    else:
        values = np.asarray(dev, dtype=float)
        sel = np.isfinite(values)
    if mask is not None:
        sel = sel & np.asarray(mask, dtype=bool)
    v = values[sel]
    if v.size == 0:
        raise ValueError("metrics: no node selected")
    a = np.abs(v)
    out = {"rms": float(np.sqrt(np.mean(v * v))), "mae": float(a.mean()),
           "max_abs": float(a.max()), "p95_abs": float(np.percentile(a, 95)),
           "bias": float(v.mean()), "std": float(v.std()), "n": int(v.size)}
    if tolerance is not None:
        out["tolerance"] = float(tolerance)
        out["pct_within_tol"] = float(100.0 * np.mean(a <= tolerance))
    return out


def part_mask(target: HeightMap, eps: float = 1e-6) -> np.ndarray:
    """Valid nodes below the sheet plane by more than `eps` [m]: the part."""
    return target.mask & (target.z < -eps)


def flange_mask(target: HeightMap, eps: float = 1e-6) -> np.ndarray:
    """Valid nodes within `eps` [m] of the sheet plane: the flange."""
    return target.mask & (target.z >= -eps)


def wall_mask(target: HeightMap, min_angle_deg: float = 5.0, eps: float = 1e-6,
              sigma: Optional[float] = None) -> np.ndarray:
    """Part nodes whose wall angle exceeds `min_angle_deg` [deg] (derivatives
    at length scale `sigma` [m], raw central differences by default)."""
    return part_mask(target, eps) & (target.wall_angle(sigma) > math.radians(min_angle_deg))
