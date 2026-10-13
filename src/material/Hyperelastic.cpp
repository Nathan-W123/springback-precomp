#include "sparlab/material/Hyperelastic.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <sstream>

namespace sparlab {
namespace {

/// Tensor index pair of Voigt component I: {11, 22, 12} in 2-D,
/// {11, 22, 33, 12, 23, 31} in 3-D.
void voigt_pair(int dim, int component, int& i, int& j) {
  static constexpr int pairs3[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {2, 0}};
  static constexpr int pairs2[3][2] = {{0, 0}, {1, 1}, {0, 1}};
  if (dim == 3) {
    i = pairs3[component][0];
    j = pairs3[component][1];
  } else {
    i = pairs2[component][0];
    j = pairs2[component][1];
  }
}

Matrix3 tensor_from_voigt_stress(const Vector& s, int dim, Scalar s33) {
  Matrix3 t = Matrix3::Zero();
  const int nv = voigt_components(dim);
  for (int c = 0; c < nv; ++c) {
    int i = 0;
    int j = 0;
    voigt_pair(dim, c, i, j);
    t(i, j) = s(c);
    t(j, i) = s(c);
  }
  if (dim == 2) t(2, 2) = s33;
  return t;
}

/// The displacement gradient in 3 x 3 (zero out-of-plane rows and columns
/// for a plane model).
Matrix3 embed_gradient(const Matrix& h) {
  Matrix3 h3 = Matrix3::Zero();
  h3.topLeftCorner(h.rows(), h.cols()) = h;
  return h3;
}

/// det(I + H) - 1 = I1 + I2 + I3 from the invariants of H, without the
/// cancellation of det(F) - 1 at small strain.
Scalar jacobian_minus_one(const Matrix3& h) {
  const Scalar i1 = h.trace();
  const Scalar i2 = 0.5 * (i1 * i1 - (h * h).trace());
  const Scalar i3 = h.determinant();
  return i1 + i2 + i3;
}

}  // namespace

HyperelasticModel parse_hyperelastic_model(const std::string& text) {
  if (text == "saint_venant_kirchhoff") return HyperelasticModel::SaintVenantKirchhoff;
  if (text == "neo_hookean") return HyperelasticModel::NeoHookean;
  throw ConfigError("unknown material model '" + text +
                    "'; expected \"saint_venant_kirchhoff\" or \"neo_hookean\"");
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

Scalar thermal_green_lagrange_change(const IsotropicMaterial& m, Scalar delta_t) {
  return delta_t * (1.0 + 0.5 * m.thermal_expansion() * delta_t);
}

Scalar thermal_stretch(const IsotropicMaterial& m, Scalar delta_t) {
  return 1.0 + m.thermal_expansion() * delta_t;
}

HyperelasticResponse evaluate_hyperelastic(HyperelasticModel model, const IsotropicMaterial& m,
                                           StressState state, const Matrix& grad_u,
                                           Scalar delta_t) {
  const int dim = static_cast<int>(grad_u.rows());
  const int nv = voigt_components(dim);
  HyperelasticResponse out;
  if (model == HyperelasticModel::SaintVenantKirchhoff) {
    const Matrix d = m.constitutive(state);
    Vector e = green_lagrange_voigt(grad_u);
    // Thermal expansion splits off multiplicatively, F = F_e (theta I) with
    // theta = 1 + alpha dT: E - E_theta = theta^2 E_e, where the thermal
    // strain E_theta = alpha dT (1 + alpha dT / 2) is the Green-Lagrange
    // strain of the free stretch (IsotropicMaterial's linear thermal strain
    // at the effective change below). The elastic energy per unit volume of
    // the expanded, stress-free state is 1/2 E_e : D : E_e, so per unit
    // reference volume W = theta^3 / 2 E_e : D : E_e and
    // S = D (E - E_theta) / theta.
    const Scalar effective_dt = thermal_green_lagrange_change(m, delta_t);
    const Scalar theta = thermal_stretch(m, delta_t);
    if (effective_dt != 0.0 && m.thermal_expansion() != 0.0) {
      e -= m.thermal_strain(state, effective_dt);
    }
    out.tangent = d / theta;
    out.stress = out.tangent * e;
    out.energy = 0.5 * e.dot(out.stress);
    if (state == StressState::PlaneStrain) {
      // S_33 = [lambda tr(E - E_theta) - 2 mu E_theta] / theta
      //      = nu (S_11 + S_22) - Young E_theta / theta.
      out.s33 = m.plane_strain_sigma_zz(out.stress(0), out.stress(1), effective_dt / theta);
    } else if (state == StressState::PlaneStress) {
      const Vector strain = green_lagrange_voigt(grad_u);
      const Scalar nu = m.poisson_ratio();
      const Scalar e_th = m.thermal_expansion() * effective_dt;
      const Scalar e33 = ((1.0 + nu) * e_th - nu * (strain(0) + strain(1))) / (1.0 - nu);
      out.thickness_stretch = std::sqrt(1.0 + 2.0 * e33);
    }
    return out;
  }

  // Compressible neo-Hookean.
  if (state == StressState::PlaneStress) {
    throw ConfigError("the neo-Hookean law is formulated for plane strain and 3-D; in plane "
                      "stress the thickness stretch would be an extra unknown. Use "
                      "\"saint_venant_kirchhoff\", plane strain or a solid mesh");
  }
  if (delta_t != 0.0 && m.thermal_expansion() != 0.0) {
    throw ConfigError("a temperature change is supported with the Saint Venant-Kirchhoff law "
                      "only; the neo-Hookean law has no thermal split");
  }
  const Scalar lambda = m.lame_lambda();
  const Scalar mu = m.shear_modulus();
  const Matrix3 h3 = embed_gradient(grad_u);
  const Scalar jm1 = jacobian_minus_one(h3);
  if (!(jm1 > -1.0)) {
    std::ostringstream os;
    os << "an element is inverted (det F = " << 1.0 + jm1
       << "); the load step is too large or the mesh too coarse for this deformation";
    throw SolverError(os.str());
  }
  // E = (H + H^T + H^T H) / 2, C = I + 2E, and I - C^-1 = 2 C^-1 E.
  const Matrix3 e = 0.5 * (h3 + h3.transpose() + h3.transpose() * h3);
  const Matrix3 c = Matrix3::Identity() + 2.0 * e;
  const Matrix3 ci = c.inverse();
  const Scalar lnj = std::log1p(jm1);
  const Matrix3 s = 2.0 * mu * (ci * e) + lambda * lnj * ci;
  out.energy = mu * e.trace() - mu * lnj + 0.5 * lambda * lnj * lnj;
  out.stress.resize(nv);
  out.tangent.resize(nv, nv);
  const Scalar g = mu - lambda * lnj;
  for (int a = 0; a < nv; ++a) {
    int i = 0;
    int jj = 0;
    voigt_pair(dim, a, i, jj);
    out.stress(a) = s(i, jj);
    for (int b = 0; b < nv; ++b) {
      int k = 0;
      int l = 0;
      voigt_pair(dim, b, k, l);
      out.tangent(a, b) = lambda * ci(i, jj) * ci(k, l) +
                          g * (ci(i, k) * ci(jj, l) + ci(i, l) * ci(jj, k));
    }
  }
  if (dim == 2) out.s33 = s(2, 2);
  return out;
}

Vector cauchy_from_piola_kirchhoff(const Matrix& grad_u, const HyperelasticResponse& response,
                                   StressState state, Scalar* sigma_zz) {
  const int dim = static_cast<int>(grad_u.rows());
  const int nv = voigt_components(dim);
  Matrix3 f3 = Matrix3::Identity() + embed_gradient(grad_u);
  // In plane stress the thickness stretches by l_3 (S_33 = 0 leaves the
  // in-plane terms of F S F^T unchanged; only J carries it).
  if (dim == 2 && state == StressState::PlaneStress) f3(2, 2) = response.thickness_stretch;
  const Matrix3 st = tensor_from_voigt_stress(
      response.stress, dim, state == StressState::PlaneStrain ? response.s33 : 0.0);
  const Scalar j = f3.determinant();
  const Matrix3 sigma = f3 * st * f3.transpose() / j;
  Vector out(nv);
  for (int c = 0; c < nv; ++c) {
    int i = 0;
    int k = 0;
    voigt_pair(dim, c, i, k);
    out(c) = sigma(i, k);
  }
  if (sigma_zz != nullptr) *sigma_zz = sigma(2, 2);
  return out;
}

}  // namespace sparlab
