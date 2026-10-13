#include "sparlab/fem/Elastoplastic.hpp"

#include "IncompatibleModes.hpp"
#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/TotalLagrangian.hpp"
#include "sparlab/material/Hyperelastic.hpp"
#include "sparlab/material/LogarithmicStrain.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace sparlab {
namespace {

/// An operator or strain in the 3-D Voigt order {11, 22, 33, 12, 23, 31}: a
/// plane element's rows {11, 22, 12} go to rows 0, 1 and 3.
Matrix voigt6(const Matrix& b, int dim) {
  if (dim == 3) return b;
  Matrix out = Matrix::Zero(6, b.cols());
  out.row(0) = b.row(0);
  out.row(1) = b.row(1);
  out.row(3) = b.row(2);
  return out;
}

/// The dilatation row of a strain operator: the sum of its normal rows.
Vector dilatation_row(const Matrix& b6) {
  return (b6.row(0) + b6.row(1) + b6.row(2)).transpose();
}

/// Everything a point needs from the kinematics.
struct Point {
  Matrix b;              ///< 6 x nd: B, B_NL, P B_NL, or their mean-dilatation forms
  Vector6 strain;        ///< epsilon, E or E_log, likewise
  Matrix g;              ///< reference gradients dN/dX, dim x n
  Matrix h;              ///< displacement gradient, dim x dim
  Scalar weight = 0.0;   ///< w detJ t
  Scalar delta_t = 0.0;  ///< temperature change per unit load factor
  Matrix bnl;            ///< logarithmic: B_NL (6 x nd), the variation of E
  LogarithmicStrain log; ///< logarithmic: E_log and its derivatives
  /// logarithmic + averaged: s = b-bar - b, the mean less the point's
  /// variation of ln J (so that b = P B_NL + m s^T / 3).
  Vector shift;
  /// logarithmic, with the tangent: T and C_alg of the point's return,
  /// kept for the tangent assembled after all of the element's returns.
  Vector6 stress = Vector6::Zero();
  Matrix6 tangent = Matrix6::Zero();
};

struct Kinematic {
  std::vector<Point> points;
  bool averaged = false;  ///< mean dilatation applied
  Matrix gbar;            ///< finite + averaged: the volume average of G^T G, n x n
};

/// Adds w (G^T s G) (x) I - the geometric stiffness of the in-plane (2-D) or
/// full (3-D) part of the tensorial Voigt stress s - to k.
void add_geometric(Matrix& k, const Matrix& g, const Vector6& s, Scalar w, int dim) {
  Matrix sd(dim, dim);
  if (dim == 3) {
    sd << s(0), s(3), s(5), s(3), s(1), s(4), s(5), s(4), s(2);
  } else {
    sd << s(0), s(3), s(3), s(1);
  }
  const Matrix geo = g.transpose() * sd * g;
  const Eigen::Index n = geo.rows();
  for (Eigen::Index a = 0; a < n; ++a) {
    for (Eigen::Index b = 0; b < n; ++b) {
      const Scalar v = w * geo(a, b);
      for (int c = 0; c < dim; ++c) k(dim * a + c, dim * b + c) += v;
    }
  }
}

/// Logarithmic kinematics: throws SolverError at an inverted point. C = F^T F
/// is positive definite at det F < 0 too, and E_log, the energy and
/// tr E_log = ln|det F| are those of a valid state (F with an axis
/// reflected), which the solver would accept as an equilibrium. A plane
/// model's F_33 is 1 (plane strain) or exp(E_log,33) > 0 (plane stress), so
/// the sign is that of the in-plane determinant.
void refuse_inverted(const Matrix& f, int dim) {
  Matrix3 f3 = Matrix3::Identity();
  f3.topLeftCorner(dim, dim) = f;
  const Scalar j = f3.determinant();
  if (!(j > 0.0)) {
    std::ostringstream os;
    os << "an element is inverted (det F = " << j
       << "); the load step is too large or the mesh too coarse for this deformation";
    throw SolverError(os.str());
  }
}

Kinematic kinematics_of(const FemModel& model, Index e, const Vector& ue, Kinematics kinematics,
                        bool mean_dilatation, const Vector* temperature) {
  const Mesh& mesh = model.mesh();
  const Element& element = model.element();
  const int dim = mesh.dim();
  const int n = mesh.nodes_per_elem();
  const Scalar t = dim == 2 ? model.thickness() : 1.0;
  const Matrix x0 = mesh.element_coordinates(e);
  const IsotropicMaterial& mat = model.material_of(e);
  const bool thermal = temperature != nullptr && temperature->size() > 0 &&
                       mat.thermal_expansion() != 0.0;
  Vector te;
  if (thermal) {
    te.resize(n);
    const Index* nodes = mesh.element_nodes(e);
    for (int a = 0; a < n; ++a) te(a) = (*temperature)(nodes[a]);
  }
  Matrix u(dim, n);
  for (int a = 0; a < n; ++a) {
    for (int k = 0; k < dim; ++k) u(k, a) = ue(dim * a + k);
  }
  const bool finite = kinematics == Kinematics::Finite;
  const bool logarithmic = kinematics == Kinematics::FiniteLogarithmic;

  Kinematic out;
  for (const IntegrationPoint& ip : element.integration_rule(model.integration())) {
    const StrainOperator op = element.strain_operator(x0, ip.point);
    Point p;
    p.g = reference_gradients(op, dim, n);
    p.h = u * p.g.transpose();
    if (finite) {
      const Matrix f = Matrix::Identity(dim, dim) + p.h;
      p.b = voigt6(green_lagrange_operator(f, p.g), dim);
      p.strain = voigt6(green_lagrange_voigt(p.h), dim);
    } else if (logarithmic) {
      // E_log of E (a plane model's C_33 = 1: its normal is principal with
      // E_log,33 = 0), dE_log = P B_NL du.
      const Matrix f = Matrix::Identity(dim, dim) + p.h;
      refuse_inverted(f, dim);
      p.bnl = voigt6(green_lagrange_operator(f, p.g), dim);
      p.log = logarithmic_strain(voigt6(green_lagrange_voigt(p.h), dim));
      p.strain = p.log.strain;
      p.b = p.log.projection * p.bnl;
    } else {
      p.b = voigt6(op.b, dim);
      p.strain = p.b * ue;
    }
    p.weight = ip.weight * op.detJ * t;
    if (thermal) {
      p.delta_t = element.shape_functions(ip.point).dot(te) - mat.reference_temperature();
    }
    out.points.push_back(std::move(p));
  }
  // Mean dilatation: every point's dilatation (with logarithmic kinematics
  // tr E_log = ln J) replaced by the element's volume average, in the strain
  // and in its variation.
  if (mean_dilatation && model.stress_state() != StressState::PlaneStress &&
      out.points.size() > 1) {
    out.averaged = true;
    Vector mean_row = Vector::Zero(dim * n);
    Scalar mean_trace = 0.0;
    Scalar volume = 0.0;
    if (finite) out.gbar = Matrix::Zero(n, n);
    for (const Point& p : out.points) {
      mean_row += p.weight * dilatation_row(p.b);
      mean_trace += p.weight * (p.strain(0) + p.strain(1) + p.strain(2));
      volume += p.weight;
      if (finite) out.gbar += p.weight * (p.g.transpose() * p.g);
    }
    mean_row /= volume;
    mean_trace /= volume;
    if (finite) out.gbar /= volume;
    for (Point& p : out.points) {
      if (logarithmic) p.shift = mean_row - dilatation_row(p.b);
      const Vector correction = (mean_row - dilatation_row(p.b)) / 3.0;
      const Scalar shift = (mean_trace - (p.strain(0) + p.strain(1) + p.strain(2))) / 3.0;
      for (int r = 0; r < 3; ++r) {
        p.b.row(r) += correction.transpose();
        p.strain(r) += shift;
      }
    }
  }
  return out;
}

/// The temperature change the return sees at load factor lambda, and its
/// derivative with respect to lambda: lambda dT0 with small strain; with
/// finite kinematics the change whose linear thermal strain is the Green
/// strain of the free thermal stretch, dT (1 + alpha dT / 2); with
/// logarithmic kinematics the one whose thermal strain is its log strain,
/// ln(1 + alpha dT) / alpha.
void thermal_change(const IsotropicMaterial& mat, Kinematics kinematics, Scalar dt0,
                    Scalar lambda, Scalar& change, Scalar& rate) {
  const Scalar dt = lambda * dt0;
  const Scalar alpha = mat.thermal_expansion();
  if (kinematics == Kinematics::Finite) {
    change = thermal_green_lagrange_change(mat, dt);
    rate = dt0 * (1.0 + mat.thermal_expansion() * dt);
  } else if (kinematics == Kinematics::FiniteLogarithmic && alpha != 0.0) {
    const Scalar stretch = 1.0 + alpha * dt;
    if (!(stretch > 0.0)) {
      std::ostringstream os;
      os << "material '" << mat.name() << "': the free thermal stretch 1 + alpha dT = "
         << stretch << " is not positive";
      throw SolverError(os.str());
    }
    change = std::log1p(alpha * dt) / alpha;
    rate = dt0 / stretch;
  } else {
    change = dt;
    rate = dt0;
  }
}

Matrix3 tensor3(const Vector6& v) {
  Matrix3 t;
  t << v(0), v(3), v(5), v(3), v(1), v(4), v(5), v(4), v(2);
  return t;
}

Vector6 voigt_of(const Matrix3& t) {
  Vector6 v;
  v << t(0, 0), t(1, 1), t(2, 2), t(0, 1), t(1, 2), t(2, 0);
  return v;
}

}  // namespace

std::string to_string(Kinematics kinematics) {
  switch (kinematics) {
    case Kinematics::SmallStrain: return "small_strain";
    case Kinematics::FiniteLogarithmic: return "finite_logarithmic";
    case Kinematics::Finite: break;
  }
  return "finite";
}

Kinematics parse_kinematics(const std::string& text) {
  if (text == "finite") return Kinematics::Finite;
  if (text == "small_strain") return Kinematics::SmallStrain;
  if (text == "finite_logarithmic") return Kinematics::FiniteLogarithmic;
  throw ConfigError("unknown kinematics '" + text +
                    "'; expected \"finite\", \"finite_logarithmic\" or \"small_strain\"");
}

MeanDilatation parse_mean_dilatation(const std::string& text) {
  if (text == "auto") return MeanDilatation::Auto;
  if (text == "all") return MeanDilatation::All;
  if (text == "none") return MeanDilatation::None;
  throw ConfigError("unknown mean_dilatation '" + text +
                    "'; expected \"auto\", \"all\" or \"none\"");
}

int elastoplastic_points(const FemModel& model) {
  return static_cast<int>(model.element().integration_rule(model.integration()).size());
}

ElastoplasticElement elastoplastic_element(const FemModel& model, Index e, const Vector& ue,
                                           const std::vector<PlasticState>& committed,
                                           bool mean_dilatation, const Vector* temperature,
                                           Scalar temperature_scale, bool want_tangent,
                                           Kinematics kinematics, const Vector* internal) {
  if (model.element().num_internal_dofs() > 0) {
    detail::refuse_mean_dilatation(mean_dilatation);
    return detail::incompatible_elastoplastic_element(model, e, ue, committed, internal,
                                                      temperature, temperature_scale,
                                                      want_tangent, kinematics);
  }
  Kinematic kin = kinematics_of(model, e, ue, kinematics, mean_dilatation, temperature);
  if (committed.size() != kin.points.size()) {
    throw SolverError("elastoplastic element: the stored internal variables do not match "
                      "the integration rule");
  }
  const IsotropicMaterial& mat = model.material_of(e);
  const StressState state = model.stress_state();
  const int dim = model.dim();
  const int n = model.mesh().nodes_per_elem();
  const int nd = static_cast<int>(ue.size());
  const bool finite = kinematics == Kinematics::Finite;
  const bool logarithmic = kinematics == Kinematics::FiniteLogarithmic;
  const bool thermal = std::any_of(kin.points.begin(), kin.points.end(),
                                   [](const Point& p) { return p.delta_t != 0.0; });

  ElastoplasticElement out;
  out.internal_force = Vector::Zero(nd);
  if (want_tangent) out.tangent = Matrix::Zero(nd, nd);
  if (thermal) out.thermal_force_rate = Vector::Zero(nd);
  out.states.reserve(kin.points.size());
  for (std::size_t q = 0; q < kin.points.size(); ++q) {
    Point& p = kin.points[q];
    Scalar change = 0.0;
    Scalar rate = 0.0;
    thermal_change(mat, kinematics, p.delta_t, temperature_scale, change, rate);
    // The thermal load rate needs the tangent too.
    const PlasticResponse r =
        plastic_return(mat, state, p.strain, committed[q], change, want_tangent || thermal);
    const Scalar w = p.weight;
    out.internal_force.noalias() += w * (p.b.transpose() * r.stress);
    out.energy += w * r.energy;
    if (logarithmic && want_tangent) {
      // Logarithmic kinematics assembles its tangent after every return
      // (below).
      p.stress = r.stress;
      p.tangent = r.tangent;
    }
    if (want_tangent && !logarithmic) {
      out.tangent.noalias() += w * (p.b.transpose() * r.tangent * p.b);
      if (finite) {
        // Geometric stiffness: S : Delta delta E, and with the averaged
        // dilatation (tr S / 3) (Delta delta tr-bar E - Delta delta tr E).
        Matrix s(dim, dim);
        if (dim == 3) {
          s = tensor3(r.stress);
        } else {
          s << r.stress(0), r.stress(3), r.stress(3), r.stress(1);
        }
        Matrix geo = p.g.transpose() * s * p.g;
        if (kin.averaged) {
          geo += (r.stress(0) + r.stress(1) + r.stress(2)) / 3.0 *
                 (kin.gbar - p.g.transpose() * p.g);
        }
        for (int a = 0; a < n; ++a) {
          for (int b = 0; b < n; ++b) {
            const Scalar v = w * geo(a, b);
            for (int k = 0; k < dim; ++k) out.tangent(dim * a + k, dim * b + k) += v;
          }
        }
      }
    }
    if (thermal && rate != 0.0) {
      // At fixed displacement the thermal strain alpha dT_e(lambda) m moves
      // the stress by -C alpha (d dT_e/d lambda) m, with the consistent (and,
      // in plane stress, condensed) tangent.
      Vector6 m = Vector6::Zero();
      m.head(3).setConstant(mat.thermal_expansion() * rate);
      out.thermal_force_rate.noalias() -= w * (p.b.transpose() * (r.tangent * m));
    }
    if (r.yielding) ++out.yielding_points;
    out.states.push_back(r.state);
  }
  if (logarithmic && want_tangent) {
    // B_NL^T (P^T C P + T' : L) B_NL + (G^T S' G) (x) I with S' = P^T T', and
    // with mean dilatation the cross terms of b = P B_NL + m s^T / 3 in
    // b^T C b and, in T', the point's deviatoric T with the element's mean
    // Kirchhoff pressure - the second variation of the averaged ln J
    // (Elastoplastic.hpp).
    Scalar mean_pressure = 0.0;
    if (kin.averaged) {
      Scalar volume = 0.0;
      for (const Point& p : kin.points) {
        const Vector6& t = p.stress;
        mean_pressure += p.weight * (t(0) + t(1) + t(2)) / 3.0;
        volume += p.weight;
      }
      mean_pressure /= volume;
    }
    Vector6 m = Vector6::Zero();
    m.head(3).setOnes();
    for (const Point& p : kin.points) {
      const Scalar w = p.weight;
      const Matrix6& projection = p.log.projection;
      Vector6 t = p.stress;
      if (kin.averaged) t.head(3).array() += mean_pressure - (t(0) + t(1) + t(2)) / 3.0;
      const Matrix6 inner =
          projection.transpose() * p.tangent * projection + logarithmic_curvature(p.log, t);
      out.tangent.noalias() += w * (p.bnl.transpose() * (inner * p.bnl));
      if (kin.averaged) {
        // (A + m s^T/3)^T C (A + m s^T/3) - A^T C A with A = P B_NL (C need
        // not be symmetric).
        const Vector6 cm = p.tangent * m;
        const Vector6 mc = p.tangent.transpose() * m;
        const Vector a_cm = p.bnl.transpose() * (projection.transpose() * cm);
        const Vector a_mc = p.bnl.transpose() * (projection.transpose() * mc);
        const Vector left = a_cm + (m.dot(cm) / 3.0) * p.shift;
        out.tangent.noalias() += (w / 3.0) * (left * p.shift.transpose());
        out.tangent.noalias() += (w / 3.0) * (p.shift * a_mc.transpose());
      }
      add_geometric(out.tangent, p.g, logarithmic_stress(p.log, t), w, dim);
    }
  }
  // Symmetric to round-off unless a backstress recovers (Armstrong-Frederick),
  // whose consistent tangent is not symmetric and must stay so for Newton's
  // quadratic convergence.
  out.symmetric = mat.plasticity().symmetric_tangent();
  if (want_tangent && out.symmetric) out.tangent = 0.5 * (out.tangent + out.tangent.transpose());
  return out;
}

ElastoplasticStress elastoplastic_stress(const FemModel& model, Index e, const Vector& ue,
                                         const std::vector<PlasticState>& committed,
                                         bool mean_dilatation, const Vector* temperature,
                                         Scalar temperature_scale, Kinematics kinematics,
                                         const Vector* internal) {
  if (model.element().num_internal_dofs() > 0) {
    detail::refuse_mean_dilatation(mean_dilatation);
    return detail::incompatible_elastoplastic_stress(model, e, ue, committed, internal,
                                                     temperature, temperature_scale, kinematics);
  }
  const Kinematic kin = kinematics_of(model, e, ue, kinematics, mean_dilatation, temperature);
  if (committed.size() != kin.points.size()) {
    throw SolverError("elastoplastic element: the stored internal variables do not match "
                      "the integration rule");
  }
  const IsotropicMaterial& mat = model.material_of(e);
  const StressState state = model.stress_state();
  const int dim = model.dim();
  const bool finite = kinematics == Kinematics::Finite;
  const bool logarithmic = kinematics == Kinematics::FiniteLogarithmic;
  ElastoplasticStress out;
  out.min_jacobian = std::numeric_limits<Scalar>::infinity();
  for (std::size_t q = 0; q < kin.points.size(); ++q) {
    const Point& p = kin.points[q];
    Scalar change = 0.0;
    Scalar rate = 0.0;
    thermal_change(mat, kinematics, p.delta_t, temperature_scale, change, rate);
    // The committed state is the converged one at this displacement, so the
    // return reproduces it (an elastic check from it).
    const PlasticResponse r = plastic_return(mat, state, p.strain, committed[q], change, false);
    Vector6 cauchy = r.stress;
    Vector6 piola_kirchhoff = r.stress;
    // The deformation measures use the element's own strain, not the
    // averaged one.
    const Matrix sym = 0.5 * (p.h + p.h.transpose());
    if (logarithmic) {
      // tau = F S F^T with S = P^T T, sigma = tau / J-bar; the log strain the
      // return took, with the thickness strain it found in plane stress.
      Matrix3 f = Matrix3::Identity();
      f.topLeftCorner(dim, dim) += p.h;
      if (state == StressState::PlaneStress) f(2, 2) = std::exp(r.strain_33);
      piola_kirchhoff = logarithmic_stress(p.log, r.stress);
      const Vector6 tau = voigt_of(f * tensor3(piola_kirchhoff) * f.transpose());
      Vector6 strain = p.strain;
      strain(2) = r.strain_33;
      strain.tail(3) *= 0.5;  // tensor components
      cauchy = tau / std::exp(strain(0) + strain(1) + strain(2));
      out.kirchhoff += tau;
      out.logarithmic_strain += strain;
      Vector6 own = p.log.strain;  // not averaged
      if (state == StressState::PlaneStress) own(2) = r.strain_33;
      own.tail(3) *= 0.5;
      out.max_strain = std::max(out.max_strain, own.cwiseAbs().maxCoeff());
      out.min_jacobian = std::min(out.min_jacobian, f.determinant());
    } else if (finite) {
      Matrix3 f = Matrix3::Identity();
      f.topLeftCorner(dim, dim) += p.h;
      // Plane stress: the thickness stretch of the Green strain the return
      // found, sqrt(1 + 2 E_33).
      if (state == StressState::PlaneStress) f(2, 2) = std::sqrt(1.0 + 2.0 * r.strain_33);
      const Scalar j = f.determinant();
      cauchy = voigt_of(f * tensor3(r.stress) * f.transpose() / j);
      const Matrix green = sym + 0.5 * p.h.transpose() * p.h;
      out.max_strain = std::max(out.max_strain, green.cwiseAbs().maxCoeff());
      if (state == StressState::PlaneStress) {
        out.max_strain = std::max(out.max_strain, std::abs(r.strain_33));
      }
      out.min_jacobian = std::min(out.min_jacobian, j);
    } else {
      out.max_strain = std::max(out.max_strain, sym.cwiseAbs().maxCoeff());
      if (state == StressState::PlaneStress) {
        out.max_strain = std::max(out.max_strain, std::abs(r.strain_33));
      }
    }
    out.cauchy += cauchy;
    out.piola_kirchhoff += piola_kirchhoff;
    out.max_equivalent_plastic_strain =
        std::max(out.max_equivalent_plastic_strain, r.state.equivalent_plastic_strain);
  }
  const Scalar count = static_cast<Scalar>(kin.points.size());
  out.cauchy /= count;
  out.piola_kirchhoff /= count;
  out.kirchhoff /= count;
  out.logarithmic_strain /= count;
  out.von_mises = von_mises_stress(out.cauchy);
  if (!finite && !logarithmic) out.min_jacobian = 1.0;
  return out;
}

// The helpers the incompatible-mode kernels share (IncompatibleModes.hpp).
namespace detail {

void refuse_inverted_point(const Matrix& f, int dim) { refuse_inverted(f, dim); }

void elastoplastic_temperature_change(const IsotropicMaterial& mat, Kinematics kinematics,
                                      Scalar dt0, Scalar lambda, Scalar& change, Scalar& rate) {
  thermal_change(mat, kinematics, dt0, lambda, change, rate);
}

}  // namespace detail
}  // namespace sparlab
