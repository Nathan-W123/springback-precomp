// Where this fits: see include/sparlab/fem/TotalLagrangian.hpp

#include "sparlab/fem/TotalLagrangian.hpp"

namespace sparlab {
namespace {

/// Tensor index pair of Voigt component c ({11, 22, 12} or
/// {11, 22, 33, 12, 23, 31}).
void voigt_pair(int dim, int c, int& i, int& j) {
  static constexpr int pairs3[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {2, 0}};
  static constexpr int pairs2[3][2] = {{0, 0}, {1, 1}, {0, 1}};
  i = dim == 3 ? pairs3[c][0] : pairs2[c][0];
  j = dim == 3 ? pairs3[c][1] : pairs2[c][1];
}

}  // namespace

Matrix reference_gradients(const StrainOperator& op, int dim, int nodes) {
  // The normal-strain rows of the strain operator hold dN_a/dX_i.
  Matrix g(dim, nodes);
  for (int a = 0; a < nodes; ++a) {
    for (int i = 0; i < dim; ++i) g(i, a) = op.b(i, dim * a + i);
  }
  return g;
}

Matrix green_lagrange_operator(const Matrix& f, const Matrix& g) {
  const int dim = static_cast<int>(g.rows());
  const int nodes = static_cast<int>(g.cols());
  const int nv = voigt_components(dim);
  Matrix b(nv, dim * nodes);
  for (int c = 0; c < nv; ++c) {
    int i = 0;
    int j = 0;
    voigt_pair(dim, c, i, j);
    for (int a = 0; a < nodes; ++a) {
      for (int k = 0; k < dim; ++k) {
        b(c, dim * a + k) =
            i == j ? f(k, i) * g(i, a) : f(k, i) * g(j, a) + f(k, j) * g(i, a);
      }
    }
  }
  return b;
}

Vector green_lagrange_voigt(const Matrix& grad_u) {
  const int dim = static_cast<int>(grad_u.rows());
  const Matrix e = 0.5 * (grad_u + grad_u.transpose() + grad_u.transpose() * grad_u);
  const int nv = voigt_components(dim);
  Vector out(nv);
  for (int c = 0; c < nv; ++c) {
    int i = 0;
    int j = 0;
    voigt_pair(dim, c, i, j);
    out(c) = i == j ? e(i, j) : 2.0 * e(i, j);
  }
  return out;
}

}  // namespace sparlab
