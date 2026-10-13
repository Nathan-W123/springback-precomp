#include "sparlab/fem/TotalLagrangian.hpp"

#include "IncompatibleModes.hpp"
#include "sparlab/core/Exceptions.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

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

Matrix displacement_matrix(const Vector& ue, int dim, int nodes) {
  Matrix u(dim, nodes);
  for (int a = 0; a < nodes; ++a) {
    for (int k = 0; k < dim; ++k) u(k, a) = ue(dim * a + k);
  }
  return u;
}

Matrix stress_tensor(const Vector& s, int dim) {
  Matrix t(dim, dim);
  const int nv = voigt_components(dim);
  for (int c = 0; c < nv; ++c) {
    int i = 0;
    int j = 0;
    voigt_pair(dim, c, i, j);
    t(i, j) = s(c);
    t(j, i) = s(c);
  }
  return t;
}

/// Temperature change per unit scale at a point: N . T_e - T_ref.
Scalar point_temperature_change(const Vector& n, const Vector& te, Scalar t_ref) {
  return n.dot(te) - t_ref;
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

TotalLagrangianElement total_lagrangian_element(const FemModel& model, Index e,
                                                const Vector& ue, HyperelasticModel law,
                                                const Vector* temperature,
                                                Scalar temperature_scale, bool want_tangent,
                                                const Vector* internal) {
  const Mesh& mesh = model.mesh();
  const Element& element = model.element();
  if (element.num_internal_dofs() > 0) {
    return detail::incompatible_total_lagrangian_element(model, e, ue, law, temperature,
                                                         temperature_scale, want_tangent,
                                                         internal);
  }
  const int dim = mesh.dim();
  const int n = mesh.nodes_per_elem();
  const int nd = dim * n;
  const Scalar t = dim == 2 ? model.thickness() : 1.0;
  const IsotropicMaterial& mat = model.material_of(e);
  const StressState state = model.stress_state();
  const Matrix x0 = mesh.element_coordinates(e);
  const Matrix u = displacement_matrix(ue, dim, n);
  const bool thermal = temperature != nullptr && temperature->size() > 0 &&
                       mat.thermal_expansion() != 0.0;
  Vector te;
  Matrix d_lin;
  if (thermal) {
    te.resize(n);
    const Index* nodes = mesh.element_nodes(e);
    for (int a = 0; a < n; ++a) te(a) = (*temperature)(nodes[a]);
    d_lin = mat.constitutive(state);
  }

  TotalLagrangianElement out;
  out.internal_force = Vector::Zero(nd);
  if (want_tangent) out.tangent = Matrix::Zero(nd, nd);
  if (thermal) out.thermal_force_rate = Vector::Zero(nd);
  for (const IntegrationPoint& ip : element.integration_rule(model.integration())) {
    const StrainOperator op = element.strain_operator(x0, ip.point);
    const Matrix g = reference_gradients(op, dim, n);
    const Matrix h = u * g.transpose();  // displacement gradient
    const Matrix f = Matrix::Identity(dim, dim) + h;
    Scalar dt0 = 0.0;
    if (thermal) {
      dt0 = point_temperature_change(element.shape_functions(ip.point), te,
                                     mat.reference_temperature());
    }
    const HyperelasticResponse resp =
        evaluate_hyperelastic(law, mat, state, h, temperature_scale * dt0);
    const Matrix bnl = green_lagrange_operator(f, g);
    const Scalar w = ip.weight * op.detJ * t;
    out.internal_force.noalias() += w * (bnl.transpose() * resp.stress);
    out.energy += w * resp.energy;
    if (thermal) {
      // dS/dlambda at the temperature change lambda dT0, for
      // S = D (E - E_theta) / theta with theta = 1 + alpha lambda dT0 and
      // E_theta = alpha lambda dT0 (1 + alpha lambda dT0 / 2), the Green
      // strain of the free thermal stretch, whose rate is
      // dE_theta/dlambda = theta alpha dT0:
      // dS/dlambda = -D [eps_th(dT0) + alpha dT0 (E - E_theta) / theta^2].
      const Scalar lambda_dt = temperature_scale * dt0;
      const Scalar theta = thermal_stretch(mat, lambda_dt);
      const Vector elastic = green_lagrange_voigt(h) -
                             mat.thermal_strain(state, thermal_green_lagrange_change(mat, lambda_dt));
      const Vector rate = mat.thermal_strain(state, dt0) +
                          (mat.thermal_expansion() * dt0 / (theta * theta)) * elastic;
      out.thermal_force_rate.noalias() -= w * (bnl.transpose() * (d_lin * rate));
    }
    if (want_tangent) {
      out.tangent.noalias() += w * (bnl.transpose() * resp.tangent * bnl);
      const Matrix geo = g.transpose() * stress_tensor(resp.stress, dim) * g;
      for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
          const Scalar v = w * geo(a, b);
          for (int k = 0; k < dim; ++k) out.tangent(dim * a + k, dim * b + k) += v;
        }
      }
    }
  }
  if (want_tangent) out.tangent = 0.5 * (out.tangent + out.tangent.transpose());
  return out;
}

TotalLagrangianStress total_lagrangian_stress(const FemModel& model, Index e, const Vector& ue,
                                              HyperelasticModel law, const Vector* temperature,
                                              Scalar temperature_scale, const Vector* internal) {
  const Mesh& mesh = model.mesh();
  const Element& element = model.element();
  if (element.num_internal_dofs() > 0) {
    return detail::incompatible_total_lagrangian_stress(model, e, ue, law, temperature,
                                                        temperature_scale, internal);
  }
  const int dim = mesh.dim();
  const int n = mesh.nodes_per_elem();
  const int nv = voigt_components(dim);
  const IsotropicMaterial& mat = model.material_of(e);
  const StressState state = model.stress_state();
  const Matrix x0 = mesh.element_coordinates(e);
  const Matrix u = displacement_matrix(ue, dim, n);
  const bool thermal = temperature != nullptr && temperature->size() > 0 &&
                       mat.thermal_expansion() != 0.0;
  Vector te;
  if (thermal) {
    te.resize(n);
    const Index* nodes = mesh.element_nodes(e);
    for (int a = 0; a < n; ++a) te(a) = (*temperature)(nodes[a]);
  }
  TotalLagrangianStress out;
  out.cauchy = Vector::Zero(nv);
  int count = 0;
  for (const IntegrationPoint& ip : element.integration_rule(model.integration())) {
    const StrainOperator op = element.strain_operator(x0, ip.point);
    const Matrix g = reference_gradients(op, dim, n);
    const Matrix h = u * g.transpose();
    Scalar dt = 0.0;
    if (thermal) {
      dt = temperature_scale * point_temperature_change(element.shape_functions(ip.point), te,
                                                        mat.reference_temperature());
    }
    const HyperelasticResponse resp = evaluate_hyperelastic(law, mat, state, h, dt);
    Scalar szz = 0.0;
    out.cauchy += cauchy_from_piola_kirchhoff(h, resp, state, &szz);
    out.cauchy_zz += szz;
    const Vector strain = green_lagrange_voigt(h);
    for (int c = 0; c < nv; ++c) {
      const Scalar tensor_component = c < dim ? strain(c) : 0.5 * strain(c);
      out.max_green_strain = std::max(out.max_green_strain, std::abs(tensor_component));
    }
    // In plane stress the thickness strains too: E_33 and J = det F l_3.
    const Scalar l3 = resp.thickness_stretch;
    out.max_green_strain = std::max(out.max_green_strain, std::abs(0.5 * (l3 * l3 - 1.0)));
    ++count;
  }
  out.cauchy /= static_cast<Scalar>(count);
  out.cauchy_zz /= static_cast<Scalar>(count);
  return out;
}

}  // namespace sparlab
