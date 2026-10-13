#include "sparlab/elements/Hex8.hpp"

#include "Hex8Kernels.hpp"
#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Quadrature.hpp"

#include <Eigen/Dense>

#include <sstream>

namespace sparlab {
namespace {

/// Natural coordinates of the eight corners in VTK order.
constexpr Scalar kCorner[8][3] = {{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0},
                                  {-1.0, 1.0, -1.0},  {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
                                  {1.0, 1.0, 1.0},    {-1.0, 1.0, 1.0}};

using detail::Hex8Coords;
using detail::Hex8Mapping;
using detail::hex8_coords;
using detail::hex8_mapping;
using detail::hex8_stiffness_rule;
using detail::hex8_strain_matrix;
using detail::require_unit_thickness;

/// Natural derivatives of the bilinear shape functions of a quadrilateral
/// face, (dN/dxi, dN/deta) per node in counter-clockwise order.
Eigen::Matrix<Scalar, 4, 2> quad4_shape_gradients_natural(Scalar xi, Scalar eta) {
  Eigen::Matrix<Scalar, 4, 2> g;
  // dN/dxi                          dN/deta
  g(0, 0) = -0.25 * (1.0 - eta);  g(0, 1) = -0.25 * (1.0 - xi);
  g(1, 0) =  0.25 * (1.0 - eta);  g(1, 1) = -0.25 * (1.0 + xi);
  g(2, 0) =  0.25 * (1.0 + eta);  g(2, 1) =  0.25 * (1.0 + xi);
  g(3, 0) = -0.25 * (1.0 + eta);  g(3, 1) =  0.25 * (1.0 - xi);
  return g;
}

}  // namespace

Eigen::Matrix<Scalar, 8, 1> hex8_shape_functions(Scalar xi, Scalar eta, Scalar zeta) {
  Eigen::Matrix<Scalar, 8, 1> n;
  for (int a = 0; a < 8; ++a) {
    n(a) = 0.125 * (1.0 + xi * kCorner[a][0]) * (1.0 + eta * kCorner[a][1]) *
           (1.0 + zeta * kCorner[a][2]);
  }
  return n;
}

Eigen::Matrix<Scalar, 8, 3> hex8_shape_gradients_natural(Scalar xi, Scalar eta,
                                                         Scalar zeta) {
  Eigen::Matrix<Scalar, 8, 3> g;
  for (int a = 0; a < 8; ++a) {
    const Scalar fx = 1.0 + xi * kCorner[a][0];
    const Scalar fy = 1.0 + eta * kCorner[a][1];
    const Scalar fz = 1.0 + zeta * kCorner[a][2];
    g(a, 0) = 0.125 * kCorner[a][0] * fy * fz;
    g(a, 1) = 0.125 * kCorner[a][1] * fx * fz;
    g(a, 2) = 0.125 * kCorner[a][2] * fx * fy;
  }
  return g;
}

Scalar hex8_volume(const Matrix& coords_in, Scalar* min_det) {
  const Hex8Coords coords = hex8_coords(coords_in);
  Scalar volume = 0.0;
  Scalar lowest = std::numeric_limits<Scalar>::max();
  for (const auto& gp : gauss_legendre_cube(2)) {
    const Eigen::Matrix<Scalar, 8, 3> dn_dxi =
        hex8_shape_gradients_natural(gp.xi, gp.eta, gp.zeta);
    const Matrix3 jac = coords * dn_dxi;
    const Scalar det = jac.determinant();
    lowest = std::min(lowest, det);
    volume += gp.weight * det;
  }
  if (min_det != nullptr) *min_det = lowest;
  return volume;
}

Scalar hex8_face_area(const Matrix& face) {
  if (face.rows() != 3 || face.cols() != 4) {
    throw MeshError("hex8_face_area expects a 3 x 4 coordinate matrix");
  }
  Scalar area = 0.0;
  for (const auto& gp : gauss_legendre_square(2)) {
    const Eigen::Matrix<Scalar, 4, 2> dn = quad4_shape_gradients_natural(gp.xi, gp.eta);
    const Vector3 xs = face * dn.col(0);
    const Vector3 xt = face * dn.col(1);
    area += gp.weight * xs.cross(xt).norm();
  }
  return area;
}

Vector Hex8Element::shape_functions(const NaturalPoint& point) const {
  return hex8_shape_functions(point.xi, point.eta, point.zeta);
}

StrainOperator Hex8Element::strain_operator(const Matrix& coords,
                                            const NaturalPoint& point) const {
  const Hex8Mapping map =
      hex8_mapping(hex8_coords(coords), point.xi, point.eta, point.zeta, "");
  StrainOperator op;
  op.detJ = map.det;
  op.b = hex8_strain_matrix(map.dn_dx);
  return op;
}

Matrix Hex8Element::stiffness(const Matrix& coords_in, const Matrix& d_in,
                              Scalar thickness, const IntegrationOptions& opts) const {
  require_unit_thickness(thickness);
  if (d_in.rows() != 6 || d_in.cols() != 6) {
    std::ostringstream os;
    os << "Hex8 expects a 6 x 6 constitutive matrix, received " << d_in.rows() << " x "
       << d_in.cols();
    throw ModelError(os.str());
  }
  const Hex8Coords coords = hex8_coords(coords_in);
  const Matrix6 d = d_in;
  Eigen::Matrix<Scalar, 24, 24> ke = Eigen::Matrix<Scalar, 24, 24>::Zero();
  for (const auto& gp : hex8_stiffness_rule(opts)) {
    const Hex8Mapping map = hex8_mapping(coords, gp.xi, gp.eta, gp.zeta,
                                         "during stiffness integration");
    const Eigen::Matrix<Scalar, 6, 24> b = hex8_strain_matrix(map.dn_dx);
    ke.noalias() += (map.det * gp.weight) * (b.transpose() * d * b);
  }
  // Symmetrise: the integrand is symmetric, so any asymmetry is round-off.
  return 0.5 * (ke + ke.transpose());
}

Matrix Hex8Element::consistent_mass(const Matrix& coords_in, Scalar density,
                                    Scalar thickness,
                                    const IntegrationOptions& opts) const {
  require_unit_thickness(thickness);
  const Hex8Coords coords = hex8_coords(coords_in);
  Eigen::Matrix<Scalar, 24, 24> me = Eigen::Matrix<Scalar, 24, 24>::Zero();
  for (const auto& gp : gauss_legendre_cube(opts.mass_points)) {
    const Eigen::Matrix<Scalar, 8, 1> n = hex8_shape_functions(gp.xi, gp.eta, gp.zeta);
    const Hex8Mapping map =
        hex8_mapping(coords, gp.xi, gp.eta, gp.zeta, "during mass integration");
    // N_mat is 3 x 24 with the node-major DOF ordering.
    Eigen::Matrix<Scalar, 3, 24> nmat = Eigen::Matrix<Scalar, 3, 24>::Zero();
    for (int a = 0; a < 8; ++a) {
      nmat(0, 3 * a + 0) = n(a);
      nmat(1, 3 * a + 1) = n(a);
      nmat(2, 3 * a + 2) = n(a);
    }
    me.noalias() += (density * map.det * gp.weight) * (nmat.transpose() * nmat);
  }
  return 0.5 * (me + me.transpose());
}

std::vector<IntegrationPoint> Hex8Element::integration_rule(
    const IntegrationOptions& opts) const {
  std::vector<IntegrationPoint> rule;
  for (const auto& gp : hex8_stiffness_rule(opts)) {
    IntegrationPoint ip;
    ip.point.xi = gp.xi;
    ip.point.eta = gp.eta;
    ip.point.zeta = gp.zeta;
    ip.weight = gp.weight;
    rule.push_back(ip);
  }
  return rule;
}

}  // namespace sparlab
