/// \file Element.hpp
/// \brief Abstract element interface.
///
/// Every element implementation supplies the kernels the rest of the library
/// needs: a stiffness matrix, a consistent mass matrix, a strain-displacement
/// operator at a requested parametric location and its stiffness quadrature
/// rule. The interface is dimension-generic: coordinates, constitutive
/// matrices and the strain operator are dynamic Eigen matrices sized by
/// `dim()` and `num_voigt()`. The library's element is the Hex8, in its
/// standard and its incompatible-mode formulation (`make_element`).
///
/// Where this fits: Simulations stage - the element interface; make_element picks the
///   standard or incompatible-mode Hex8 from the deck's model.element_formulation.
/// Called from: fem/FemModel.cpp:FemModel::FemModel (make_element),
///   fem/Elastoplastic.cpp:kinematics_of,
///   fem/ExplicitKernel.cpp:ExplicitInternalForce::build_dedicated
/// Calls into: elements/Hex8, elements/Hex8Incompatible, mesh/Mesh (ElementType)
/// Start reading at: Element, make_element
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/mesh/Mesh.hpp"

#include <memory>
#include <string>
#include <vector>

namespace sparlab {

/// Element-local coordinates on the reference cube [-1, 1]^3.
struct NaturalPoint {
  Scalar xi = 0.0;
  Scalar eta = 0.0;
  Scalar zeta = 0.0;
};

/// Element formulation of the displacement elements.
enum class ElementFormulation {
  /// The isoparametric element with its compatible displacement field alone.
  Standard,
  /// Hex8 only: Wilson-Taylor incompatible modes, three condensed bubble
  /// fields per displacement component (Hex8Incompatible.hpp).
  IncompatibleModes
};

std::string to_string(ElementFormulation formulation);
/// "standard" or "incompatible_modes".
/// \throws ConfigError for any other text.
ElementFormulation parse_element_formulation(const std::string& text);

/// Quadrature orders used by the Hex8 kernels.
///
/// The element formulation travels with the orders because every kernel
/// that integrates an element receives them: `make_element` builds the
/// element of the formulation, and the kernels of the non-linear analysis
/// find it through `Element::num_internal_nodes`.
struct IntegrationOptions {
  int stiffness_points = 2;  ///< points per direction for K_e
  int mass_points = 3;       ///< points per direction for M_e
  /// Hex8: points of the stiffness rule through the thickness of a sheet,
  /// along the natural axis `thickness_axis` (1 to kMaxThicknessPoints of
  /// Quadrature.hpp); 0 keeps `stiffness_points`. The rule is then
  /// `stiffness_points` x `stiffness_points` x `thickness_points`
  /// (`gauss_legendre_box`), in the point order of every other rule, and
  /// the per-point history of the non-linear analysis follows it.
  int thickness_points = 0;
  /// Natural axis of the element through the sheet thickness: 0 (xi), 1 (eta)
  /// or 2 (zeta), which the structured hexahedral generator aligns with x, y
  /// and z.
  int thickness_axis = 2;
  ElementFormulation formulation = ElementFormulation::Standard;
};

/// Strain-displacement data evaluated at one parametric point.
struct StrainOperator {
  Matrix b;           ///< num_voigt x num_dofs operator [1/m]
  Scalar detJ = 0.0;  ///< Jacobian determinant [m^dim]
};

/// One point of an element's stiffness quadrature rule.
struct IntegrationPoint {
  NaturalPoint point;
  Scalar weight = 0.0;  ///< weight on the reference domain, without det J
};

/// Static condensation of an element's internal (incompatible) modes for a
/// linear-elastic constitutive matrix D. With the internal parameters
/// \f$\alpha\f$ and their strain operator \f$\tilde B\f$
/// (`Element::internal_strain_operator`),
/// \f[
///   K_{\alpha\alpha} = \int \tilde B^T D \tilde B\,dV ,\qquad
///   K_{\alpha u} = \int \tilde B^T D B\,dV ,\qquad
///   \alpha = K_{\alpha\alpha}^{-1}\Big(\int \tilde B^T D\,\varepsilon_0\,dV
///             - K_{\alpha u} u_e\Big)
/// \f]
/// for an element with the initial (thermal) strain \f$\varepsilon_0\f$,
/// so that the strain \f$Bu_e + \tilde B\alpha\f$ is
/// \f$\hat B u_e + \tilde B K_{\alpha\alpha}^{-1}\int\tilde B^T
/// D\varepsilon_0\f$ with \f$\hat B = B - \tilde B\,K_{\alpha\alpha}^{-1}
/// K_{\alpha u}\f$, and \f$\int\hat B^T D\hat B = K^*\f$, the condensed
/// stiffness. Empty (0 rows) for an element without internal modes.
struct InternalCondensation {
  Matrix coupling;  ///< \f$K_{\alpha\alpha}^{-1}K_{\alpha u}\f$, num_internal_dofs x num_dofs [-]
  Matrix inverse;   ///< \f$K_{\alpha\alpha}^{-1}\f$ [m/N]
};

/// Abstract continuum element.
class Element {
 public:
  virtual ~Element() = default;

  virtual ElementType type() const = 0;
  /// Spatial dimension (3: sparlab_form meshes are Hex8): the translational
  /// DOFs per node.
  virtual int dim() const = 0;
  virtual int num_nodes() const = 0;

  /// Degrees of freedom per node, in node-major order: the `dim` translations.
  int dofs_per_node() const { return dim(); }

  /// Degrees of freedom carried by the element (num_nodes * dofs_per_node).
  int num_dofs() const { return num_nodes() * dofs_per_node(); }

  /// Voigt components of the element's strain operator.
  int num_voigt() const { return voigt_components(dim()); }

  /// Element stiffness matrix
  /// \f$ K_e = \int_{\Omega_e} t\, B^T D B \, d\Omega \f$ [N/m].
  /// \param coords dim x num_nodes nodal coordinates [m].
  /// \param d constitutive matrix, num_voigt x num_voigt [Pa].
  /// \param thickness out-of-plane thickness [m] (must be 1 for 3-D elements).
  virtual Matrix stiffness(const Matrix& coords, const Matrix& d, Scalar thickness,
                           const IntegrationOptions& opts) const = 0;

  /// Consistent element mass matrix
  /// \f$ M_e = \int_{\Omega_e} \rho\, t\, N^T N \, d\Omega \f$ [kg].
  virtual Matrix consistent_mass(const Matrix& coords, Scalar density, Scalar thickness,
                                 const IntegrationOptions& opts) const = 0;

  /// Strain-displacement operator and Jacobian determinant at a natural point.
  /// \throws MeshError when detJ <= 0 (inverted or degenerate element).
  virtual StrainOperator strain_operator(const Matrix& coords,
                                         const NaturalPoint& point) const = 0;

  /// Shape function values at a natural point (size num_nodes).
  virtual Vector shape_functions(const NaturalPoint& point) const = 0;

  /// The stiffness quadrature rule: its points with their reference
  /// weights, so that
  /// \f$\sum_g w_g \det J_g\, f(\xi_g)\f$ integrates \f$f\f$ exactly as
  /// `stiffness` does.
  virtual std::vector<IntegrationPoint> integration_rule(
      const IntegrationOptions& opts) const = 0;

  /// Internal "pseudo-nodes" of the element: fields that are condensed
  /// inside the element (static condensation) and carry `dim()` parameters
  /// each, like a node - 3 for the incompatible-mode Hex8, 0 for every
  /// other element.
  virtual int num_internal_nodes() const { return 0; }

  /// Internal parameters condensed in the element (num_internal_nodes * dim).
  int num_internal_dofs() const { return num_internal_nodes() * dim(); }

  /// Gradients of the internal fields with respect to the reference
  /// coordinates at a natural point, dim x num_internal_nodes [1/m] (a
  /// dim x 0 matrix for an element without internal modes): the columns
  /// that extend the shape-function gradients \f$G\f$ of the element's
  /// nodes, so the displacement gradient is \f$H = U G^T + A\tilde G^T\f$
  /// with the internal parameters \f$A\f$ (dim x num_internal_nodes).
  /// \throws MeshError when det J <= 0.
  virtual Matrix internal_mode_gradients(const Matrix& coords, const NaturalPoint& point) const;

  /// The small-strain operator of the internal parameters at a natural
  /// point, num_voigt x num_internal_dofs [1/m], built from
  /// `internal_mode_gradients` as `strain_operator` is from the shape
  /// gradients (parameter k of pseudo-node m is column dim * m + k).
  Matrix internal_strain_operator(const Matrix& coords, const NaturalPoint& point) const;

  /// Condensation of the internal parameters of a linear-elastic element
  /// with constitutive matrix `d` over the stiffness rule (see
  /// InternalCondensation); empty for an element without internal modes.
  /// \throws SolverError when \f$K_{\alpha\alpha}\f$ is singular.
  InternalCondensation condense_internal(const Matrix& coords, const Matrix& d,
                                         Scalar thickness, const IntegrationOptions& opts) const;

  /// The strain operator with the internal parameters condensed,
  /// \f$\hat B = B - \tilde B\,K_{\alpha\alpha}^{-1}K_{\alpha u}\f$, at a
  /// natural point (the plain B of an element without internal modes).
  /// Never pass it where the shape-function gradients are read off B.
  Matrix condensed_strain_operator(const Matrix& coords, const NaturalPoint& point,
                                   const InternalCondensation& condensation) const;
};

/// Factory for the supported element topologies.
/// \throws ConfigError for an unsupported type.
std::unique_ptr<Element> make_element(ElementType type);

/// The element of the formulation `opts.formulation` for a topology.
/// \throws ConfigError for an unsupported type, or a formulation the
///         topology does not have (incompatible modes: Hex8 only).
std::unique_ptr<Element> make_element(ElementType type, const IntegrationOptions& opts);

}  // namespace sparlab
