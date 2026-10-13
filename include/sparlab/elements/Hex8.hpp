/// \file Hex8.hpp
/// \brief Eight-node trilinear isoparametric hexahedron (solid element).
///
/// Shape functions on the reference cube \f$(\xi,\eta,\zeta)\in[-1,1]^3\f$
/// with the VTK hexahedron node ordering (bottom face counter-clockwise seen
/// from +z, then the top face in the same order):
/// \f[
///   N_a = \tfrac{1}{8}(1+\xi\xi_a)(1+\eta\eta_a)(1+\zeta\zeta_a),
///   \qquad (\xi_a,\eta_a,\zeta_a) \in \{-1,+1\}^3 .
/// \f]
/// Isoparametric mapping \f$ x = \sum_a N_a x_a \f$, Jacobian
/// \f$ J_{ij} = \partial x_i / \partial \xi_j \f$ (inverted explicitly through
/// its adjugate), and \f$ \partial N_a/\partial x = J^{-T}\partial N_a/\partial\xi \f$.
///
/// Strain is stored in the 6-component Voigt order
/// \f$\{\varepsilon_{xx},\varepsilon_{yy},\varepsilon_{zz},\gamma_{xy},
/// \gamma_{yz},\gamma_{zx}\}\f$ with engineering shear strains, and the
/// element DOF ordering is node-major \f$\{u_{x1},u_{y1},u_{z1},u_{x2},\dots\}\f$.
///
/// A solid element has no thickness; the `thickness` argument of the kernels
/// must be 1 and is rejected otherwise, so a 2-D deck's thickness can never
/// leak into a 3-D model unnoticed.
#pragma once

#include "sparlab/elements/Element.hpp"

namespace sparlab {

/// Trilinear shape functions and their natural gradients (free functions so
/// they can be unit-tested and reused by the mesh layer for cell volumes).
/// \{
Eigen::Matrix<Scalar, 8, 1> hex8_shape_functions(Scalar xi, Scalar eta, Scalar zeta);
Eigen::Matrix<Scalar, 8, 3> hex8_shape_gradients_natural(Scalar xi, Scalar eta,
                                                         Scalar zeta);
/// \}

/// Volume of a hexahedron from its 3 x 8 nodal coordinates, integrated with
/// the 2 x 2 x 2 rule (exact for the trilinear map). `min_det`, when given,
/// receives the smallest Jacobian determinant over the integration points, so
/// a folded cell is detectable even when its volume is positive.
Scalar hex8_volume(const Matrix& coords, Scalar* min_det = nullptr);

/// Area of a bilinear quadrilateral face from its 3 x 4 nodal coordinates
/// (2 x 2 rule on \f$|x_{,s}\times x_{,t}|\f$).
Scalar hex8_face_area(const Matrix& face_coords);

class Hex8Element final : public Element {
 public:
  ElementType type() const override { return ElementType::Hex8; }
  int dim() const override { return 3; }
  int num_nodes() const override { return 8; }
  int num_faces() const override { return 6; }

  Matrix stiffness(const Matrix& coords, const Matrix& d, Scalar thickness,
                   const IntegrationOptions& opts) const override;

  Matrix consistent_mass(const Matrix& coords, Scalar density, Scalar thickness,
                         const IntegrationOptions& opts) const override;

  StrainOperator strain_operator(const Matrix& coords,
                                 const NaturalPoint& point) const override;

  Vector shape_functions(const NaturalPoint& point) const override;

  std::vector<IntegrationPoint> integration_rule(
      const IntegrationOptions& opts) const override;
};

}  // namespace sparlab
