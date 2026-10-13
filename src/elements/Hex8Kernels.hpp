/// \file Hex8Kernels.hpp
/// \brief The fixed-size mapping and strain-operator kernels of the
///        trilinear hexahedron, shared by the standard (Hex8.cpp) and the
///        incompatible-mode (Hex8Incompatible.cpp) elements. Internal to the
///        library.
///
/// The code is that of the standard Hex8 kernels, moved here unchanged:
/// the element's results stay bit-for-bit what they were.
///
/// Where this fits: Simulations stage - the Hex8 mapping and strain-matrix kernels shared
///   by the two Hex8 elements.
/// Called from: elements/Hex8.cpp:Hex8Element::stiffness, Hex8Element::strain_operator,
///   elements/Hex8Incompatible.cpp:Hex8IncompatibleElement::stiffness
/// Calls into: elements/Hex8 (shape gradients), elements/Quadrature
/// Start reading at: hex8_mapping, hex8_strain_matrix
#pragma once

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Types.hpp"
#include "sparlab/elements/Element.hpp"
#include "sparlab/elements/Hex8.hpp"
#include "sparlab/elements/Quadrature.hpp"

#include <sstream>
#include <vector>

namespace sparlab {
namespace detail {

using Hex8Coords = Eigen::Matrix<Scalar, 3, 8>;

inline Hex8Coords hex8_coords(const Matrix& coords) {
  if (coords.rows() != 3 || coords.cols() != 8) {
    std::ostringstream os;
    os << "Hex8 expects a 3 x 8 nodal coordinate matrix, received " << coords.rows()
       << " x " << coords.cols();
    throw MeshError(os.str());
  }
  return coords;
}

inline void require_unit_thickness(Scalar thickness) {
  if (thickness != 1.0) {
    std::ostringstream os;
    os << "a solid (Hex8) element has no thickness; received " << thickness
       << " m. Leave model.thickness at its default of 1 for a 3-D mesh";
    throw ConfigError(os.str());
  }
}

/// Jacobian, its determinant and the Cartesian shape gradients at one point.
struct Hex8Mapping {
  Eigen::Matrix<Scalar, 8, 3> dn_dx;
  Scalar det = 0.0;
};

/// The explicit inverse of a 3 x 3 Jacobian J(i,j) = dx_i/dxi_j through its
/// adjugate, and its determinant.
/// \throws MeshError when det J <= 0.
inline Matrix3 hex8_jacobian_inverse(const Matrix3& jac, Scalar xi, Scalar eta, Scalar zeta,
                                     const char* context, Scalar& det_out) {
  // Explicit 3x3 determinant and adjugate: this runs at every Gauss point of
  // every element and needs no LU machinery.
  const Scalar c00 = jac(1, 1) * jac(2, 2) - jac(1, 2) * jac(2, 1);
  const Scalar c01 = jac(1, 2) * jac(2, 0) - jac(1, 0) * jac(2, 2);
  const Scalar c02 = jac(1, 0) * jac(2, 1) - jac(1, 1) * jac(2, 0);
  const Scalar det = jac(0, 0) * c00 + jac(0, 1) * c01 + jac(0, 2) * c02;
  if (!(det > 0.0)) {
    std::ostringstream os;
    os << "Hex8 Jacobian determinant is " << det << " m^3 at (xi, eta, zeta) = (" << xi
       << ", " << eta << ", " << zeta << ") " << context
       << "; the element is inverted, folded or its nodes do not follow the VTK "
          "hexahedron ordering";
    throw MeshError(os.str());
  }
  Matrix3 jinv;
  jinv(0, 0) = c00 / det;
  jinv(1, 0) = c01 / det;
  jinv(2, 0) = c02 / det;
  jinv(0, 1) = (jac(0, 2) * jac(2, 1) - jac(0, 1) * jac(2, 2)) / det;
  jinv(1, 1) = (jac(0, 0) * jac(2, 2) - jac(0, 2) * jac(2, 0)) / det;
  jinv(2, 1) = (jac(0, 1) * jac(2, 0) - jac(0, 0) * jac(2, 1)) / det;
  jinv(0, 2) = (jac(0, 1) * jac(1, 2) - jac(0, 2) * jac(1, 1)) / det;
  jinv(1, 2) = (jac(0, 2) * jac(1, 0) - jac(0, 0) * jac(1, 2)) / det;
  jinv(2, 2) = (jac(0, 0) * jac(1, 1) - jac(0, 1) * jac(1, 0)) / det;
  det_out = det;
  return jinv;
}

inline Hex8Mapping hex8_mapping(const Hex8Coords& coords, Scalar xi, Scalar eta, Scalar zeta,
                                const char* context) {
  const Eigen::Matrix<Scalar, 8, 3> dn_dxi = hex8_shape_gradients_natural(xi, eta, zeta);
  // J(i,j) = dx_i / dxi_j
  const Matrix3 jac = coords * dn_dxi;
  Hex8Mapping map;
  const Matrix3 jinv = hex8_jacobian_inverse(jac, xi, eta, zeta, context, map.det);
  // dN/dx = dN/dxi * J^{-1}  (row a holds grad N_a).
  map.dn_dx = dn_dxi * jinv;
  return map;
}

inline Eigen::Matrix<Scalar, 6, 24> hex8_strain_matrix(const Eigen::Matrix<Scalar, 8, 3>& dn_dx) {
  Eigen::Matrix<Scalar, 6, 24> b = Eigen::Matrix<Scalar, 6, 24>::Zero();
  for (int a = 0; a < 8; ++a) {
    const Scalar dx = dn_dx(a, 0);
    const Scalar dy = dn_dx(a, 1);
    const Scalar dz = dn_dx(a, 2);
    const int c = 3 * a;
    b(0, c + 0) = dx;  // eps_xx
    b(1, c + 1) = dy;  // eps_yy
    b(2, c + 2) = dz;  // eps_zz
    b(3, c + 0) = dy;  // gamma_xy = du/dy + dv/dx
    b(3, c + 1) = dx;
    b(4, c + 1) = dz;  // gamma_yz = dv/dz + dw/dy
    b(4, c + 2) = dy;
    b(5, c + 0) = dz;  // gamma_zx = du/dz + dw/dx
    b(5, c + 2) = dx;
  }
  return b;
}

/// The stiffness rule of a Hex8: the n x n x n Gauss-Legendre cube, or with
/// `opts.thickness_points` set (and different from n) the n x n x m box rule
/// through the thickness axis `opts.thickness_axis`.
inline const std::vector<QuadraturePoint3D>& hex8_stiffness_rule(const IntegrationOptions& opts) {
  if (opts.thickness_points <= 0 || opts.thickness_points == opts.stiffness_points) {
    return gauss_legendre_cube(opts.stiffness_points);
  }
  return gauss_legendre_box(opts.stiffness_points, opts.thickness_points, opts.thickness_axis);
}

}  // namespace detail
}  // namespace sparlab
