#include "sparlab/material/LogarithmicStrain.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace sparlab {
namespace {

/// Tensor indices of Voigt component k, {11, 22, 33, 12, 23, 31}.
constexpr int kRow[6] = {0, 1, 2, 0, 1, 2};
constexpr int kCol[6] = {0, 1, 2, 1, 2, 0};

/// Below this spread of three eigenvalues of C, relative to the middle one,
/// the second divided difference of ln is its Taylor series (see the file
/// comment of LogarithmicStrain.hpp).
constexpr Scalar kSpread = 5.0e-3;

/// ln[x, y] = (ln x - ln y) / (x - y) of x = 1 + 2 mx, y = 1 + 2 my: from
/// the larger one, log1p((x - y) / y) / (x - y), exact to round-off at any
/// gap (the limit 1 / y at none).
Scalar log_divided(Scalar mx, Scalar my) {
  if (mx > my) std::swap(mx, my);
  const Scalar y = 1.0 + 2.0 * my;
  const Scalar d = 2.0 * (mx - my);
  if (d == 0.0) return 1.0 / y;
  return std::log1p(d / y) / d;
}

/// ln[x, y, z] of the eigenvalues with indices a, b, c, from the first
/// divided differences: the difference quotient for a wide spread, the
/// Taylor series about the middle eigenvalue for a narrow one.
Scalar log_divided2(const Vector3& mu, const Matrix3& first, int a, int b, int c) {
  int i = a;
  int j = b;
  int k = c;
  // Sort so that mu_i >= mu_j >= mu_k.
  if (mu(i) < mu(j)) std::swap(i, j);
  if (mu(j) < mu(k)) std::swap(j, k);
  if (mu(i) < mu(j)) std::swap(i, j);
  const Scalar y = 1.0 + 2.0 * mu(j);
  const Scalar spread = 2.0 * (mu(i) - mu(k));
  if (spread > kSpread * y) return (first(i, j) - first(j, k)) / spread;
  // y^-2 sum_{n=0}^{5} (-1)^(n+1) h_n(u, w) / (n + 2), h_n the complete
  // homogeneous symmetric polynomial of degree n: h_n = u h_(n-1) + w^n.
  const Scalar u = 2.0 * (mu(i) - mu(j)) / y;
  const Scalar w = 2.0 * (mu(k) - mu(j)) / y;
  Scalar h = 1.0;
  Scalar wn = 1.0;
  Scalar sum = -0.5;
  for (int n = 1; n <= 5; ++n) {
    wn *= w;
    h = u * h + wn;
    sum += (n % 2 == 1 ? 1.0 : -1.0) * h / static_cast<Scalar>(n + 2);
  }
  return sum / (y * y);
}

Matrix3 tensor_of(const Vector6& v, Scalar shear_factor) {
  Matrix3 t;
  t(0, 0) = v(0);
  t(1, 1) = v(1);
  t(2, 2) = v(2);
  t(0, 1) = t(1, 0) = shear_factor * v(3);
  t(1, 2) = t(2, 1) = shear_factor * v(4);
  t(2, 0) = t(0, 2) = shear_factor * v(5);
  return t;
}

Vector6 voigt_of(const Matrix3& t, Scalar shear_factor) {
  Vector6 v;
  v << t(0, 0), t(1, 1), t(2, 2), shear_factor * t(0, 1), shear_factor * t(1, 2),
      shear_factor * t(2, 0);
  return v;
}

/// The map of tensorial Voigt components into the principal axes,
/// t(Q^T A Q) = M t(A), a shear component standing for both of its entries.
Matrix6 principal_rotation(const Matrix3& q) {
  Matrix6 m;
  for (int r = 0; r < 6; ++r) {
    const int a = kRow[r];
    const int b = kCol[r];
    for (int c = 0; c < 6; ++c) {
      const int i = kRow[c];
      const int j = kCol[c];
      m(r, c) = i == j ? q(i, a) * q(i, b) : q(i, a) * q(j, b) + q(j, a) * q(i, b);
    }
  }
  return m;
}

/// diag(1, 1, 1, 1/2, 1/2, 1/2) applied on the right (S^-1): engineering to
/// tensorial columns.
Matrix6 tensorial_columns(Matrix6 m) {
  m.rightCols<3>() *= 0.5;
  return m;
}

}  // namespace

LogarithmicStrain logarithmic_strain(const Vector6& green) {
  if (!green.allFinite()) {
    throw SolverError("the logarithmic strain met a non-finite Green-Lagrange strain");
  }
  LogarithmicStrain out;
  const Eigen::SelfAdjointEigenSolver<Matrix3> eigen(tensor_of(green, 0.5));
  out.green = eigen.eigenvalues();
  out.axes = eigen.eigenvectors();
  Vector3 principal;
  for (int a = 0; a < 3; ++a) {
    const Scalar stretch2 = 1.0 + 2.0 * out.green(a);
    if (!(stretch2 > 0.0) || !std::isfinite(stretch2)) {
      std::ostringstream os;
      os << "a point is degenerate: a principal stretch squared of C is "
         << stretch2 << "; the load step is too large or the mesh too coarse";
      throw SolverError(os.str());
    }
    principal(a) = 0.5 * std::log1p(2.0 * out.green(a));
  }
  const Matrix3& q = out.axes;
  out.strain = voigt_of(q * principal.asDiagonal() * q.transpose(), 2.0);
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      out.first(a, b) = out.first(b, a) = log_divided(out.green(a), out.green(b));
    }
  }
  // P = M^T S D M S^-1, D the first divided differences of each principal
  // component: [dE_log]'_ab = l[a, b] [dE]'_ab in the principal axes.
  const Matrix6 m = principal_rotation(q);
  Vector6 sd;
  for (int k = 0; k < 6; ++k) sd(k) = (k < 3 ? 1.0 : 2.0) * out.first(kRow[k], kCol[k]);
  out.projection = tensorial_columns(m.transpose() * sd.asDiagonal() * m);
  return out;
}

Vector6 logarithmic_stress(const LogarithmicStrain& log, const Vector6& t) {
  // S' = l o T' in the principal axes.
  const Matrix3& q = log.axes;
  const Matrix3 principal = (q.transpose() * tensor_of(t, 1.0) * q).cwiseProduct(log.first);
  return voigt_of(q * principal * q.transpose(), 1.0);
}

Matrix6 logarithmic_curvature(const LogarithmicStrain& log, const Vector6& t) {
  const Matrix3& q = log.axes;
  const Matrix3 tp = q.transpose() * tensor_of(t, 1.0) * q;
  // l[a, b, c], symmetric in its indices: ten distinct values.
  Scalar second[27];
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      for (int c = b; c < 3; ++c) {
        const Scalar v = log_divided2(log.green, log.first, a, b, c);
        const int perm[6][3] = {{a, b, c}, {a, c, b}, {b, a, c}, {b, c, a}, {c, a, b}, {c, b, a}};
        for (const auto& i : perm) second[i[0] + 3 * i[1] + 9 * i[2]] = v;
      }
    }
  }
  // The bilinear form 2 sum_abc T'_ab l[a, c, b] (X_ac Y_cb + Y_ac X_cb) on
  // the principal unit tensors X, Y of each tensorial Voigt component (a
  // shear one with both of its entries).
  const auto entries = [](int k, int (&rows)[2], int (&cols)[2]) {
    rows[0] = kRow[k];
    cols[0] = kCol[k];
    rows[1] = kCol[k];
    cols[1] = kRow[k];
    return k < 3 ? 1 : 2;
  };
  const auto half = [&](int kx, int ky) {
    int xr[2], xc[2], yr[2], yc[2];
    const int nx = entries(kx, xr, xc);
    const int ny = entries(ky, yr, yc);
    Scalar sum = 0.0;
    for (int s = 0; s < nx; ++s) {
      const int a = xr[s];
      const int c = xc[s];
      for (int r = 0; r < ny; ++r) {
        if (yr[r] != c) continue;
        const int b = yc[r];
        sum += tp(a, b) * second[a + 3 * b + 9 * c];
      }
    }
    return sum;
  };
  Matrix6 hp;
  for (int k = 0; k < 6; ++k) {
    for (int l = k; l < 6; ++l) hp(k, l) = hp(l, k) = 2.0 * (half(k, l) + half(l, k));
  }
  const Matrix6 m = tensorial_columns(principal_rotation(q));  // M S^-1
  return m.transpose() * hp * m;
}

}  // namespace sparlab
