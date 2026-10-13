#include "sparlab/elements/Element.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Hex8.hpp"
#include "sparlab/elements/Hex8Incompatible.hpp"

#include <Eigen/Cholesky>


namespace sparlab {

std::string to_string(ElementFormulation formulation) {
  switch (formulation) {
    case ElementFormulation::Standard: return "standard";
    case ElementFormulation::IncompatibleModes: return "incompatible_modes";
  }
  return "standard";
}

ElementFormulation parse_element_formulation(const std::string& text) {
  if (text == "standard") return ElementFormulation::Standard;
  if (text == "incompatible_modes") return ElementFormulation::IncompatibleModes;
  throw ConfigError("unknown element formulation '" + text +
                    "'; expected \"standard\" or \"incompatible_modes\"");
}

Matrix Element::internal_mode_gradients(const Matrix& /*coords*/,
                                        const NaturalPoint& /*point*/) const {
  return Matrix(dim(), 0);
}

Matrix Element::internal_strain_operator(const Matrix& coords, const NaturalPoint& point) const {
  const Matrix g = internal_mode_gradients(coords, point);
  const int nd = dim();
  const auto m = static_cast<int>(g.cols());
  Matrix b = Matrix::Zero(num_voigt(), nd * m);
  for (int a = 0; a < m; ++a) {
    const int c = nd * a;
    if (nd == 2) {
      b(0, c + 0) = g(0, a);
      b(1, c + 1) = g(1, a);
      b(2, c + 0) = g(1, a);
      b(2, c + 1) = g(0, a);
    } else {
      b(0, c + 0) = g(0, a);
      b(1, c + 1) = g(1, a);
      b(2, c + 2) = g(2, a);
      b(3, c + 0) = g(1, a);
      b(3, c + 1) = g(0, a);
      b(4, c + 1) = g(2, a);
      b(4, c + 2) = g(1, a);
      b(5, c + 0) = g(2, a);
      b(5, c + 2) = g(0, a);
    }
  }
  return b;
}

InternalCondensation Element::condense_internal(const Matrix& coords, const Matrix& d,
                                                Scalar thickness,
                                                const IntegrationOptions& opts) const {
  InternalCondensation out;
  const int ni = num_internal_dofs();
  if (ni == 0) {
    out.coupling = Matrix(0, num_dofs());
    out.inverse = Matrix(0, 0);
    return out;
  }
  const Scalar t = dim() == 2 ? thickness : 1.0;
  Matrix kaa = Matrix::Zero(ni, ni);
  Matrix kau = Matrix::Zero(ni, num_dofs());
  for (const IntegrationPoint& ip : integration_rule(opts)) {
    const StrainOperator op = strain_operator(coords, ip.point);
    const Matrix bt = internal_strain_operator(coords, ip.point);
    const Scalar w = t * ip.weight * op.detJ;
    const Matrix dbt = d * bt;
    kaa.noalias() += w * (bt.transpose() * dbt);
    kau.noalias() += w * (dbt.transpose() * op.b);
  }
  kaa = 0.5 * (kaa + kaa.transpose());
  const Eigen::LLT<Matrix> llt(kaa);
  if (llt.info() != Eigen::Success) {
    throw SolverError(to_string(type()) +
                      ": the stiffness of the internal modes is not positive definite; the "
                      "constitutive matrix is not positive definite or the element is "
                      "degenerate");
  }
  out.coupling = llt.solve(kau);
  out.inverse = llt.solve(Matrix::Identity(ni, ni));
  return out;
}

Matrix Element::condensed_strain_operator(const Matrix& coords, const NaturalPoint& point,
                                          const InternalCondensation& condensation) const {
  const StrainOperator op = strain_operator(coords, point);
  if (condensation.coupling.rows() == 0) return op.b;
  return op.b - internal_strain_operator(coords, point) * condensation.coupling;
}

std::unique_ptr<Element> make_element(ElementType type) {
  if (type == ElementType::Hex8) return std::make_unique<Hex8Element>();
  throw ConfigError("SparLab has no " + to_string(type) + " element: its forming analysis "
                    "takes Hex8 meshes");
}

std::unique_ptr<Element> make_element(ElementType type, const IntegrationOptions& opts) {
  if (opts.formulation == ElementFormulation::Standard) return make_element(type);
  if (type != ElementType::Hex8) {
    throw ConfigError("the \"" + to_string(opts.formulation) +
                      "\" element formulation is available for Hex8 meshes only; the mesh "
                      "is " + to_string(type));
  }
  return std::make_unique<Hex8IncompatibleElement>();
}

}  // namespace sparlab
