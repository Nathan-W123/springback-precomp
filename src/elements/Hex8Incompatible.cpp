#include "sparlab/elements/Hex8Incompatible.hpp"

#include "Hex8Kernels.hpp"
#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Quadrature.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <sstream>

namespace sparlab {
namespace {

using detail::Hex8Coords;
using detail::Hex8Mapping;
using detail::hex8_coords;
using detail::hex8_jacobian_inverse;
using detail::hex8_mapping;
using detail::hex8_stiffness_rule;
using detail::hex8_strain_matrix;
using detail::require_unit_thickness;

/// The Jacobian inverse and determinant at the element centre.
struct CentreMapping {
  Matrix3 inverse;
  Scalar det = 0.0;
};

CentreMapping centre_mapping(const Hex8Coords& coords) {
  const Matrix3 jac = coords * hex8_shape_gradients_natural(0.0, 0.0, 0.0);
  CentreMapping c;
  c.inverse = hex8_jacobian_inverse(jac, 0.0, 0.0, 0.0, "at the element centre", c.det);
  return c;
}

/// The small-strain operator of the three modes (6 x 9), columns 3k..3k+2
/// for mode k, from their gradients (columns of g).
Eigen::Matrix<Scalar, 6, 9> mode_strain_matrix(const Matrix3& g) {
  Eigen::Matrix<Scalar, 6, 9> b = Eigen::Matrix<Scalar, 6, 9>::Zero();
  for (int k = 0; k < 3; ++k) {
    const Scalar dx = g(0, k);
    const Scalar dy = g(1, k);
    const Scalar dz = g(2, k);
    const int c = 3 * k;
    b(0, c + 0) = dx;
    b(1, c + 1) = dy;
    b(2, c + 2) = dz;
    b(3, c + 0) = dy;
    b(3, c + 1) = dx;
    b(4, c + 1) = dz;
    b(4, c + 2) = dy;
    b(5, c + 0) = dz;
    b(5, c + 2) = dx;
  }
  return b;
}

Matrix6 constitutive_6(const Matrix& d) {
  if (d.rows() != 6 || d.cols() != 6) {
    std::ostringstream os;
    os << "Hex8 expects a 6 x 6 constitutive matrix, received " << d.rows() << " x "
       << d.cols();
    throw ModelError(os.str());
  }
  return d;
}

}  // namespace

Matrix3 hex8_incompatible_gradients(const Matrix3& centre_inverse, Scalar centre_det,
                                    Scalar det, Scalar xi, Scalar eta, Scalar zeta) {
  const Scalar ratio = centre_det / det;
  const Scalar natural[3] = {xi, eta, zeta};
  Matrix3 g;
  for (int k = 0; k < 3; ++k) {
    // d phi_k / d xi = -2 xi_k e_k, mapped with J_0^{-T}: row k of J_0^{-1}.
    g.col(k) = (ratio * -2.0 * natural[k]) * centre_inverse.row(k).transpose();
  }
  return g;
}

Matrix Hex8IncompatibleElement::internal_mode_gradients(const Matrix& coords_in,
                                                        const NaturalPoint& point) const {
  const Hex8Coords coords = hex8_coords(coords_in);
  const CentreMapping centre = centre_mapping(coords);
  const Hex8Mapping map = hex8_mapping(coords, point.xi, point.eta, point.zeta, "");
  return hex8_incompatible_gradients(centre.inverse, centre.det, map.det, point.xi, point.eta,
                                     point.zeta);
}

Matrix Hex8IncompatibleElement::stiffness(const Matrix& coords_in, const Matrix& d_in,
                                          Scalar thickness,
                                          const IntegrationOptions& opts) const {
  require_unit_thickness(thickness);
  const Matrix6 d = constitutive_6(d_in);
  const Hex8Coords coords = hex8_coords(coords_in);
  const CentreMapping centre = centre_mapping(coords);
  Eigen::Matrix<Scalar, 24, 24> kuu = Eigen::Matrix<Scalar, 24, 24>::Zero();
  Eigen::Matrix<Scalar, 9, 24> kau = Eigen::Matrix<Scalar, 9, 24>::Zero();
  Eigen::Matrix<Scalar, 9, 9> kaa = Eigen::Matrix<Scalar, 9, 9>::Zero();
  for (const auto& gp : hex8_stiffness_rule(opts)) {
    const Hex8Mapping map = hex8_mapping(coords, gp.xi, gp.eta, gp.zeta,
                                         "during stiffness integration");
    const Eigen::Matrix<Scalar, 6, 24> b = hex8_strain_matrix(map.dn_dx);
    const Eigen::Matrix<Scalar, 6, 9> bt = mode_strain_matrix(hex8_incompatible_gradients(
        centre.inverse, centre.det, map.det, gp.xi, gp.eta, gp.zeta));
    const Scalar w = map.det * gp.weight;
    const Eigen::Matrix<Scalar, 6, 24> db = d * b;
    kuu.noalias() += w * (b.transpose() * db);
    kau.noalias() += w * (bt.transpose() * db);
    kaa.noalias() += w * (bt.transpose() * (d * bt));
  }
  kaa = 0.5 * (kaa + kaa.transpose()).eval();
  const Eigen::LLT<Eigen::Matrix<Scalar, 9, 9>> llt(kaa);
  if (llt.info() != Eigen::Success) {
    throw SolverError("Hex8 incompatible modes: the stiffness of the modes is not positive "
                      "definite: the constitutive matrix is not positive definite, or the rule has a "
                      "single point along a natural axis (where the mode gradients vanish)");
  }
  const Eigen::Matrix<Scalar, 9, 24> coupling = llt.solve(kau);
  Eigen::Matrix<Scalar, 24, 24> ke = kuu - kau.transpose() * coupling;
  // Symmetrise: the condensed matrix is symmetric, any asymmetry is round-off.
  return 0.5 * (ke + ke.transpose());
}

Matrix Hex8IncompatibleElement::consistent_mass(const Matrix& coords, Scalar density,
                                                Scalar thickness,
                                                const IntegrationOptions& opts) const {
  return standard_.consistent_mass(coords, density, thickness, opts);
}

StrainOperator Hex8IncompatibleElement::strain_operator(const Matrix& coords,
                                                        const NaturalPoint& point) const {
  return standard_.strain_operator(coords, point);
}

Vector Hex8IncompatibleElement::shape_functions(const NaturalPoint& point) const {
  return standard_.shape_functions(point);
}

std::vector<IntegrationPoint> Hex8IncompatibleElement::integration_rule(
    const IntegrationOptions& opts) const {
  return standard_.integration_rule(opts);
}

}  // namespace sparlab
