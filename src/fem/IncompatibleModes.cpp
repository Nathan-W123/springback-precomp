// Where this fits: see src/fem/IncompatibleModes.hpp

#include "IncompatibleModes.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Hex8.hpp"
#include "sparlab/elements/Hex8Incompatible.hpp"
#include "sparlab/material/LogarithmicStrain.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace sparlab {
namespace detail {
namespace {

// ---------------------------------------------------------------------------
// Sizes and fixed-size types of the augmented Hex8: 8 nodes and 3 pseudo-nodes
// ---------------------------------------------------------------------------
constexpr int kNodes = 8;
constexpr int kModes = 3;
constexpr int kAug = kNodes + kModes;  // 11 "nodes"
constexpr int kU = 3 * kNodes;         // 24 displacements
constexpr int kA = 3 * kModes;         // 9 mode parameters
constexpr int kAll = kU + kA;          // 33

using OperatorA = Eigen::Matrix<Scalar, 6, kAll>;
using GradientsA = Eigen::Matrix<Scalar, 3, kAug>;
using VectorA = Eigen::Matrix<Scalar, kAll, 1>;
using VectorI = Eigen::Matrix<Scalar, kA, 1>;
using MatrixUU = Eigen::Matrix<Scalar, kU, kU>;
using MatrixUI = Eigen::Matrix<Scalar, kU, kA>;
using MatrixIU = Eigen::Matrix<Scalar, kA, kU>;
using MatrixII = Eigen::Matrix<Scalar, kA, kA>;
using MatrixGeo = Eigen::Matrix<Scalar, kAug, kAug>;

/// Everything one integration point keeps between the passes of the local
/// iteration: the alpha-independent geometry, and the kinematics and the
/// constitutive response at the current parameters.
struct PointWork {
  GradientsA g = GradientsA::Zero();  ///< [G | G~], reference gradients [1/m]
  Matrix3 hu = Matrix3::Zero();       ///< U G^T, the compatible displacement gradient
  Scalar weight = 0.0;                ///< w det J [m^3]
  Scalar delta_t = 0.0;               ///< temperature change per unit load factor [K]
  Scalar change = 0.0;                ///< the return's temperature change at lambda [K]
  Scalar rate = 0.0;                  ///< its derivative with respect to lambda [K]
  // At the current parameters.
  Matrix3 h = Matrix3::Zero();        ///< H = U G^T + A G~^T
  OperatorA mb = OperatorA::Zero();   ///< the strain variation: B (small) or B_NL
  Vector6 strain = Vector6::Zero();   ///< epsilon, E or E_log
  LogarithmicStrain log;              ///< logarithmic kinematics
  Vector6 s = Vector6::Zero();        ///< the stress conjugate to mb: sigma or S
  Matrix6 material = Matrix6::Zero(); ///< the tangent between mb's: C, or P^T C P + T:L
  Vector6 ts = Vector6::Zero();       ///< d s / d lambda at fixed strain (thermal)
  Vector6 own = Vector6::Zero();      ///< the law's own stress: sigma, S or T
  Scalar energy = 0.0;                ///< stored energy density [J/m^3]
  PlasticState state;
  bool yielding = false;
  Scalar strain_33 = 0.0;
};

using Workspace = std::vector<PointWork, Eigen::aligned_allocator<PointWork>>;

/// The per-thread workspace, reused from call to call so that neither the
/// local iteration nor, after the first element, the setup allocates.
Workspace& workspace() {
  static thread_local Workspace points;
  return points;
}

/// Sums of one pass over the points.
struct Sums {
  VectorA r = VectorA::Zero();       ///< (f_u, r_alpha) [N]
  VectorA t = VectorA::Zero();       ///< thermal load rate (q_u, q_alpha) [N]
  VectorI scale = VectorI::Zero();   ///< sum_q w |B~|^T |s|: the gross size of r_alpha [N]
  /// sum_q w |B~|^T 1 |C| |H| (1 + |H|): the force a relative strain error of
  /// 1 would cause - the round-off of the strain of H, times 1e-13, is the
  /// floor below which r_alpha is not resolved (a rigid rotation) [N].
  VectorI floor = VectorI::Zero();
  MatrixII kaa = MatrixII::Zero();
  MatrixUI kua = MatrixUI::Zero();
  MatrixIU kau = MatrixIU::Zero();   ///< non-symmetric tangents only
  MatrixUU kuu = MatrixUU::Zero();   ///< the final pass with a tangent only
  Scalar energy = 0.0;
};

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

/// The Green-Lagrange strain of H, engineering-shear Voigt.
Vector6 green_lagrange_6(const Matrix3& h) {
  const Matrix3 e = 0.5 * (h + h.transpose() + h.transpose() * h);
  Vector6 v;
  v << e(0, 0), e(1, 1), e(2, 2), 2.0 * e(0, 1), 2.0 * e(1, 2), 2.0 * e(2, 0);
  return v;
}

/// B_NL(F, [G | G~]) of TotalLagrangian.hpp on the eleven "nodes"; F = I
/// gives the linear B.
void green_lagrange_operator_11(const Matrix3& f, const GradientsA& g, OperatorA& b) {
  static constexpr int kPair[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {2, 0}};
  for (int c = 0; c < 6; ++c) {
    const int i = kPair[c][0];
    const int j = kPair[c][1];
    for (int a = 0; a < kAug; ++a) {
      for (int k = 0; k < 3; ++k) {
        b(c, 3 * a + k) = i == j ? f(k, i) * g(i, a) : f(k, i) * g(j, a) + f(k, j) * g(i, a);
      }
    }
  }
}

/// The element's setting: what is fixed for one call.
struct Setting {
  const FemModel* model = nullptr;
  Index e = 0;
  Kinematics kinematics = Kinematics::SmallStrain;
  bool thermal = false;       ///< some point has a temperature change
  bool symmetric = true;      ///< the material tangent is symmetric
  Eigen::Matrix<Scalar, 3, 8> u = Eigen::Matrix<Scalar, 3, 8>::Zero();
  Eigen::Matrix<Scalar, kU, 1> ue = Eigen::Matrix<Scalar, kU, 1>::Zero();
};

/// Fills the workspace with the alpha-independent data of every point.
Workspace& set_up(Setting& set, const FemModel& model, Index e, const Vector& ue,
                  Kinematics kinematics, const Vector* temperature, Scalar temperature_scale) {
  const Element& element = model.element();
  if (element.type() != ElementType::Hex8 || element.num_internal_nodes() != kModes) {
    throw ModelError("the incompatible-mode kernel is written for the Hex8 with three modes");
  }
  if (ue.size() != kU) {
    std::ostringstream os;
    os << "Hex8 incompatible modes: element " << e << " received " << ue.size()
       << " displacements, expected " << kU;
    throw ModelError(os.str());
  }
  set.model = &model;
  set.e = e;
  set.kinematics = kinematics;
  const Mesh& mesh = model.mesh();
  const Material& mat = model.material_of(e);
  set.symmetric = mat.plasticity().symmetric_tangent();
  for (int a = 0; a < kNodes; ++a) {
    for (int k = 0; k < 3; ++k) set.u(k, a) = ue(3 * a + k);
  }
  set.ue = ue;
  const Eigen::Matrix<Scalar, 3, 8> x0 = mesh.element_coordinates(e);
  const Matrix3 jac0 = x0 * hex8_shape_gradients_natural(0.0, 0.0, 0.0);
  const Scalar det0 = jac0.determinant();
  if (!(det0 > 0.0)) {
    std::ostringstream os;
    os << "Hex8 element " << e << " has a Jacobian determinant of " << det0
       << " m^3 at its centre; it is inverted or degenerate";
    throw MeshError(os.str());
  }
  const Matrix3 jinv0 = jac0.inverse();
  const bool heated = temperature != nullptr && temperature->size() > 0 &&
                      mat.thermal_expansion() != 0.0;
  Eigen::Matrix<Scalar, 8, 1> te = Eigen::Matrix<Scalar, 8, 1>::Zero();
  if (heated) {
    const Index* nodes = mesh.element_nodes(e);
    for (int a = 0; a < kNodes; ++a) te(a) = (*temperature)(nodes[a]);
  }
  const std::vector<IntegrationPoint> rule = element.integration_rule(model.integration());
  Workspace& points = workspace();
  points.resize(rule.size());
  set.thermal = false;
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const NaturalPoint& xi = rule[q].point;
    PointWork& p = points[q];
    const Eigen::Matrix<Scalar, 8, 3> dn_dxi =
        hex8_shape_gradients_natural(xi.xi, xi.eta, xi.zeta);
    const Matrix3 jac = x0 * dn_dxi;
    const Scalar det = jac.determinant();
    if (!(det > 0.0)) {
      std::ostringstream os;
      os << "Hex8 element " << e << " has a Jacobian determinant of " << det
         << " m^3 at an integration point; it is inverted or folded";
      throw MeshError(os.str());
    }
    p.g.leftCols<kNodes>() = (dn_dxi * jac.inverse()).transpose();
    p.g.rightCols<kModes>() =
        hex8_incompatible_gradients(jinv0, det0, det, xi.xi, xi.eta, xi.zeta);
    p.hu = set.u * p.g.leftCols<kNodes>().transpose();
    p.weight = rule[q].weight * det;
    p.delta_t = 0.0;
    if (heated) {
      p.delta_t = hex8_shape_functions(xi.xi, xi.eta, xi.zeta).dot(te) -
                  mat.reference_temperature();
    }
    if (p.delta_t != 0.0) set.thermal = true;
    elastoplastic_temperature_change(mat, kinematics, p.delta_t, temperature_scale, p.change,
                                     p.rate);
    if (kinematics == Kinematics::SmallStrain) {
      green_lagrange_operator_11(Matrix3::Identity(), p.g, p.mb);
    }
  }
  return points;
}

/// The kinematics of every point at the parameters `alpha`.
void kinematics_at(const Setting& set, Workspace& points, const VectorI& alpha) {
  Matrix3 a;  // A(k, m): component k of pseudo-node m
  for (int m = 0; m < kModes; ++m) {
    for (int k = 0; k < 3; ++k) a(k, m) = alpha(3 * m + k);
  }
  for (PointWork& p : points) {
    p.h.noalias() = p.hu + a * p.g.rightCols<kModes>().transpose();
    switch (set.kinematics) {
      case Kinematics::SmallStrain:
        p.strain.noalias() = p.mb.leftCols<kU>() * set.ue + p.mb.rightCols<kA>() * alpha;
        break;
      case Kinematics::Finite: {
        const Matrix3 f = Matrix3::Identity() + p.h;
        green_lagrange_operator_11(f, p.g, p.mb);
        p.strain = green_lagrange_6(p.h);
        break;
      }
      case Kinematics::FiniteLogarithmic: {
        const Matrix3 f = Matrix3::Identity() + p.h;
        if (!(f.determinant() > 0.0)) refuse_inverted_point(Matrix(f), 3);
        green_lagrange_operator_11(f, p.g, p.mb);
        p.log = logarithmic_strain(green_lagrange_6(p.h));
        p.strain = p.log.strain;
        break;
      }
    }
  }
}

/// The constitutive response of every point (with the tangent): the return
/// of Plasticity.hpp (J2, Hill48, Chaboche; elastic too) from the point's
/// committed state.
void respond(const Setting& set, Workspace& points, const std::vector<PlasticState>& committed) {
  const Material& mat = set.model->material_of(set.e);
  const Scalar expansion = mat.thermal_expansion();
  for (std::size_t q = 0; q < points.size(); ++q) {
    PointWork& p = points[q];
    const PlasticResponse r = plastic_return(mat, StressState::ThreeDimensional, p.strain,
                                             committed[q], p.change, true);
    p.own = r.stress;
    p.energy = r.energy;
    p.state = r.state;
    p.yielding = r.yielding;
    p.strain_33 = r.strain_33;
    Vector6 cm = Vector6::Zero();
    if (set.thermal && p.rate != 0.0) {
      Vector6 m = Vector6::Zero();
      m.head<3>().setConstant(expansion * p.rate);
      cm = r.tangent * m;
    }
    if (set.kinematics == Kinematics::FiniteLogarithmic) {
      p.s = logarithmic_stress(p.log, r.stress);
      p.material = p.log.projection.transpose() * r.tangent * p.log.projection +
                   logarithmic_curvature(p.log, r.stress);
      p.ts = -(p.log.projection.transpose() * cm);
    } else {
      p.s = r.stress;
      p.material = r.tangent;
      p.ts = -cm;
    }
  }
}

/// Adds the sums of every point the local iteration needs: the force
/// (f_u, r_alpha), its gross scale and round-off floor, the energy, the
/// thermal rate and K_aa.
void accumulate_local(const Setting& set, const Workspace& points, Sums& sums) {
  const bool geometric = set.kinematics != Kinematics::SmallStrain;
  for (const PointWork& p : points) {
    const Scalar w = p.weight;
    const auto ba = p.mb.rightCols<kA>();
    sums.r.noalias() += w * (p.mb.transpose() * p.s);
    sums.scale.noalias() += w * (ba.cwiseAbs().transpose() * p.s.cwiseAbs());
    const Scalar h = p.h.cwiseAbs().maxCoeff();
    sums.floor.noalias() += (w * p.material.cwiseAbs().maxCoeff() * h * (1.0 + h)) *
                            ba.cwiseAbs().colwise().sum().transpose();
    sums.energy += w * p.energy;
    if (set.thermal) sums.t.noalias() += w * (p.mb.transpose() * p.ts);
    sums.kaa.noalias() += w * (ba.transpose() * (p.material * ba));
    if (geometric) {
      // The alpha-alpha block of (G_aug^T S G_aug) (x) I.
      const Matrix3 gt = p.g.rightCols<kModes>();
      const Matrix3 geo = gt.transpose() * tensor3(p.s) * gt;
      for (int a = 0; a < kModes; ++a) {
        for (int b = 0; b < kModes; ++b) {
          for (int k = 0; k < 3; ++k) sums.kaa(3 * a + k, 3 * b + k) += w * geo(a, b);
        }
      }
    }
  }
}

/// Adds the coupling blocks at the converged parameters: K_ua, K_au for a
/// non-symmetric tangent, and K_uu when `full`.
void accumulate_coupling(const Setting& set, const Workspace& points, bool full, Sums& sums) {
  const bool geometric = set.kinematics != Kinematics::SmallStrain;
  for (const PointWork& p : points) {
    const Scalar w = p.weight;
    const auto bu = p.mb.leftCols<kU>();
    const auto ba = p.mb.rightCols<kA>();
    sums.kua.noalias() += w * (bu.transpose() * (p.material * ba));
    if (!set.symmetric) sums.kau.noalias() += w * (ba.transpose() * (p.material * bu));
    if (full) sums.kuu.noalias() += w * (bu.transpose() * (p.material * bu));
    if (geometric) {
      // (G_aug^T S G_aug) (x) I outside its alpha-alpha block.
      const MatrixGeo geo = p.g.transpose() * tensor3(p.s) * p.g;
      for (int a = 0; a < kNodes; ++a) {
        for (int b = 0; b < kAug; ++b) {
          const Scalar v = w * geo(a, b);
          for (int k = 0; k < 3; ++k) {
            if (b < kNodes) {
              if (full) sums.kuu(3 * a + k, 3 * b + k) += v;
            } else {
              sums.kua(3 * a + k, 3 * (b - kNodes) + k) += v;
              if (!set.symmetric) sums.kau(3 * (b - kNodes) + k, 3 * a + k) += v;
            }
          }
        }
      }
    }
  }
}

/// The result of the local iteration and the condensation.
struct Condensed {
  Vector force;
  Matrix tangent;
  Vector thermal_rate;
  Scalar energy = 0.0;
  Vector alpha;
  int iterations = 0;  ///< Newton directions taken
  int cuts = 0;        ///< line-search evaluations
};

/// Solves r_alpha(u, alpha) = 0 from the committed parameters, then
/// condenses (IncompatibleModes.hpp).
Condensed solve_and_condense(const Setting& set, Workspace& points,
                             const std::vector<PlasticState>& committed, const Vector* start,
                             bool want_tangent) {
  VectorI alpha = VectorI::Zero();
  if (start != nullptr && start->size() > 0) {
    if (start->size() != kA) {
      std::ostringstream os;
      os << "Hex8 element " << set.e << ": the committed incompatible-mode parameters have "
         << start->size() << " entries, expected " << kA;
      throw SolverError(os.str());
    }
    alpha = *start;
  }
  Condensed out;
  Sums sums;
  Eigen::FullPivLU<MatrixII> lu;
  // One pass: the kinematics, the responses and the sums at alpha.
  const auto pass = [&](const VectorI& at) {
    kinematics_at(set, points, at);
    respond(set, points, committed);
    sums = Sums();
    accumulate_local(set, points, sums);
  };
  // A pass at a trial of the iteration: false where the trial inverts a
  // point (SolverError of the logarithmic kinematics or of the return).
  const auto try_pass = [&](const VectorI& at) {
    try {
      pass(at);
      return true;
    } catch (const SolverError&) {
      return false;
    }
  };
  pass(alpha);
  Scalar residual = sums.r.tail<kA>().norm();
  // A search on |r_alpha| along d: the step `reach` d, halved until a trial
  // lowers |r_alpha| by kFullStepReduction, is taken at its best trial if
  // that lowers |r_alpha| at all; false (alpha and |r_alpha| unchanged, the
  // sums at a trial) if none does.
  const auto search_norm = [&](const VectorI& d, Scalar reach) {
    Scalar best_step = 0.0;
    Scalar best = residual;
    bool last_is_best = false;
    Scalar trial_step = reach;
    for (int search = 0; search < kMaxLineSearch; ++search, trial_step *= 0.5) {
      ++out.cuts;
      if (!try_pass(alpha + trial_step * d)) {
        last_is_best = false;
        continue;
      }
      const Scalar n = sums.r.tail<kA>().norm();
      last_is_best = n < best;
      if (last_is_best) {
        best = n;
        best_step = trial_step;
      }
      if (n <= kFullStepReduction * residual) break;
    }
    if (best_step == 0.0) return false;
    if (!last_is_best) pass(alpha + best_step * d);
    alpha += best_step * d;
    residual = best;
    return true;
  };
  // The line search along a descent direction d of the incremental
  // potential (d^T r_alpha < 0) on its directional residual phi(s) =
  // d^T r_alpha(alpha + s d) (Crisfield 1991, sec. 9.3; Matthies and Strang
  // 1979), the derivative of the potential along d wherever one exists (an
  // associative return): its root is the minimum of the potential on the
  // line, so every such step lowers it. The full step is taken when it
  // lowers |r_alpha|. Otherwise, with phi(0) < 0 < phi(reach),
  // regula falsi with the Illinois modification, each trial kept a tenth of
  // the bracket from its ends (a kinked phi would otherwise creep along one
  // end), to |phi(s)| <= kLineSearchRatio |phi(0)|; a trial that inverts a
  // point counts as beyond the root. Without such a bracket (d no descent
  // direction: a non-associative law; or the potential still falls at the
  // end of a step that raised |r_alpha|: not convex there), the search on
  // |r_alpha|. False where nothing moved alpha.
  const auto line_search = [&](const VectorI& d, const VectorI& r0) {
    const Scalar phi0 = d.dot(r0);
    // The full step, shortened while it inverts a point (finite kinematics).
    Scalar reach = 1.0;
    bool evaluated = try_pass(alpha + d);
    for (int halving = 0; !evaluated && halving < kMaxLineSearch; ++halving) {
      reach *= 0.5;
      ++out.cuts;
      evaluated = try_pass(alpha + reach * d);
    }
    if (!evaluated) return false;  // every step along d inverts a point
    const Scalar full = sums.r.tail<kA>().norm();
    if (full <= kFullStepReduction * residual ||
        sums.r.tail<kA>().cwiseAbs().maxCoeff() <=
            kLocalTolerance * sums.scale.maxCoeff() + kLocalRoundOff * sums.floor.maxCoeff()) {
      alpha += reach * d;
      residual = full;
      return true;
    }
    const Scalar phi1 = d.dot(sums.r.tail<kA>());
    if (!(phi0 < 0.0) || !(phi1 > 0.0) || !std::isfinite(phi1)) {
      return search_norm(d, 0.5 * reach);
    }
    Scalar s0 = 0.0;
    Scalar f0 = phi0;
    Scalar s1 = reach;
    Scalar f1 = phi1;
    Scalar step = reach;
    int side = 0;
    for (int search = 0; search < kMaxLineSearch; ++search) {
      const Scalar width = s1 - s0;
      step = std::clamp((s0 * f1 - s1 * f0) / (f1 - f0), s0 + 0.1 * width, s1 - 0.1 * width);
      evaluated = try_pass(alpha + step * d);
      ++out.cuts;
      const Scalar f =
          evaluated ? d.dot(sums.r.tail<kA>()) : std::numeric_limits<Scalar>::infinity();
      if (evaluated && std::abs(f) <= kLineSearchRatio * std::abs(phi0)) break;
      if (!evaluated || f > 0.0) {
        s1 = step;
        f1 = evaluated ? f : 0.5 * f1;
        if (side == 1) f0 *= 0.5;
        side = 1;
      } else {
        s0 = step;
        f0 = f;
        if (side == -1) f1 *= 0.5;
        side = -1;
      }
    }
    if (!evaluated) {
      // The search ended on an inverted trial: its last admissible bracket.
      if (s0 == 0.0) return false;
      step = s0;
      pass(alpha + step * d);
    }
    alpha += step * d;
    residual = sums.r.tail<kA>().norm();
    return true;
  };
  for (;;) {
    const Scalar largest = sums.r.tail<kA>().cwiseAbs().maxCoeff();
    const Scalar scale = sums.scale.maxCoeff();
    const Scalar floor = kLocalRoundOff * sums.floor.maxCoeff();
    lu.compute(sums.kaa);
    if (!lu.isInvertible()) {
      std::ostringstream os;
      os << "Hex8 element " << set.e
         << ": the stiffness of the incompatible modes is singular (an unstable element "
            "state, e.g. under large compression); the step is cut";
      throw SolverError(os.str());
    }
    if (largest <= kLocalTolerance * scale + floor || largest == 0.0) break;
    if (out.iterations == kMaxLocalIterations || !std::isfinite(largest)) {
      std::ostringstream os;
      os << "Hex8 element " << set.e << ": the incompatible modes did not converge in "
         << kMaxLocalIterations << " local iterations (residual " << largest << " N of "
         << scale << " N); the step is cut";
      throw SolverError(os.str());
    }
    ++out.iterations;
    // The residual and K_aa at alpha: the trials below overwrite the sums.
    const VectorI r0 = sums.r.tail<kA>();
    const MatrixII k0 = sums.kaa;
    // Newton's direction. Near the solution the full step is taken and
    // converges quadratically, also to a solution where K_aa is indefinite
    // (the element is then unstable in its modes, but in equilibrium). Far
    // from it (a return far from its committed state: a large increment, a
    // reversal) the points switch between elastic and plastic along the
    // step, r_alpha is only piecewise smooth and |r_alpha| is no merit
    // function (Newton's iterates cycle): then the line search on the
    // potential.
    const VectorI newton = -lu.solve(r0);
    const MatrixII sym = 0.5 * (k0 + k0.transpose());
    if (Eigen::LLT<MatrixII>(sym).info() == Eigen::Success) {
      if (line_search(newton, r0)) continue;
    } else {
      // The full Newton step if it lowers |r_alpha|; else, K_aa's symmetric
      // part being indefinite (a softening plastic tangent with the
      // geometric stress term of finite kinematics), Newton's direction may
      // climb the potential, and the search is along (K_aa + mu I) d =
      // -r_alpha instead, mu shifting the smallest eigenvalue of that
      // symmetric part to kLocalShift times its largest magnitude: a descent
      // direction of the potential (a modified Newton method, Nocedal and
      // Wright 2006, sec. 3.4).
      if (try_pass(alpha + newton)) {
        const Scalar full = sums.r.tail<kA>().norm();
        if (full <= kFullStepReduction * residual) {
          alpha += newton;
          residual = full;
          continue;
        }
      }
      const Eigen::SelfAdjointEigenSolver<MatrixII> spectrum(sym, Eigen::EigenvaluesOnly);
      const Scalar top = spectrum.eigenvalues().cwiseAbs().maxCoeff();
      const Scalar mu = kLocalShift * top - spectrum.eigenvalues()(0);
      const VectorI shifted = -(k0 + mu * MatrixII::Identity()).partialPivLu().solve(r0);
      if (line_search(shifted, r0) || search_norm(newton, 0.5)) continue;
    }
    // Nothing along those directions lowers |r_alpha| or the potential (or
    // every step inverts a point): Levenberg-Marquardt steps on
    // |r_alpha|^2 / 2, (K^T K + mu I) d = -K^T r_alpha with mu growing
    // tenfold, which turn towards its steepest descent -K^T r_alpha (More
    // 1978); the first that lowers |r_alpha| is taken, and none moves to a
    // worse point. Else the step is cut.
    const MatrixII normal = k0.transpose() * k0;
    const VectorI gradient = k0.transpose() * r0;
    Scalar mu = kLocalMarquardt * normal.diagonal().maxCoeff();
    bool moved = false;
    for (int search = 0; search < kMaxLineSearch && !moved; ++search, mu *= 10.0) {
      const VectorI d = -(normal + mu * MatrixII::Identity()).llt().solve(gradient);
      ++out.cuts;
      if (!try_pass(alpha + d)) continue;
      const Scalar n = sums.r.tail<kA>().norm();
      if (n < residual) {
        alpha += d;
        residual = n;
        moved = true;
      }
    }
    if (!moved) {
      std::ostringstream os;
      os << "Hex8 element " << set.e
         << ": no step of the incompatible modes lowers their residual (" << largest
         << " N of " << scale
         << " N) without inverting a point (an unstable or non-smooth element state); the "
            "step is cut";
      throw SolverError(os.str());
    }
  }
  // The coupling blocks (and K_uu) at the converged parameters, from the
  // points of the last pass.
  accumulate_coupling(set, points, want_tangent, sums);
  // K_aa^-1 applied to r_alpha (the last Newton correction), to K_au and to
  // the thermal rate q_alpha.
  const VectorI correction = lu.solve(sums.r.tail<kA>());
  out.force = sums.r.head<kU>() - sums.kua * correction;
  out.energy = sums.energy;
  if (set.thermal) {
    out.thermal_rate = sums.t.head<kU>() - sums.kua * lu.solve(sums.t.tail<kA>());
  }
  if (want_tangent) {
    const MatrixIU coupling = lu.solve(set.symmetric ? MatrixIU(sums.kua.transpose()) : sums.kau);
    MatrixUU k = sums.kuu - sums.kua * coupling;
    if (set.symmetric) k = (0.5 * (k + k.transpose())).eval();
    out.tangent = k;
  }
  out.alpha = alpha;
  return out;
}

}  // namespace

void refuse_mean_dilatation(bool mean_dilatation) {
  if (mean_dilatation) {
    throw ConfigError(
        "mean dilatation is not combined with the incompatible-mode Hex8: averaging the "
        "dilatation of its modes leaves a zero-energy mode, and averaging the compatible "
        "dilatation alone softens the bending of a sheet one element thick; the modes "
        "relax the isochoric constraint themselves. Use \"mean_dilatation\": \"auto\" or "
        "\"none\"");
  }
}

ElastoplasticElement incompatible_elastoplastic_element(
    const FemModel& model, Index e, const Vector& ue, const std::vector<PlasticState>& committed,
    const Vector* internal, const Vector* temperature, Scalar temperature_scale,
    bool want_tangent, Kinematics kinematics) {
  Setting set;
  Workspace& points = set_up(set, model, e, ue, kinematics, temperature, temperature_scale);
  if (committed.size() != points.size()) {
    throw SolverError("elastoplastic element: the stored internal variables do not match "
                      "the integration rule");
  }
  Condensed c = solve_and_condense(set, points, committed, internal, want_tangent);
  ElastoplasticElement out;
  out.internal_force = std::move(c.force);
  out.tangent = std::move(c.tangent);
  out.thermal_force_rate = std::move(c.thermal_rate);
  out.energy = c.energy;
  out.internal = std::move(c.alpha);
  out.internal_iterations = c.iterations;
  out.symmetric = set.symmetric;
  out.states.reserve(points.size());
  for (const PointWork& p : points) {
    out.states.push_back(p.state);
    if (p.yielding) ++out.yielding_points;
  }
  return out;
}

ElastoplasticStress incompatible_elastoplastic_stress(const FemModel& model, Index e,
                                                      const Vector& ue,
                                                      const std::vector<PlasticState>& committed,
                                                      const Vector* internal,
                                                      const Vector* temperature,
                                                      Scalar temperature_scale,
                                                      Kinematics kinematics) {
  Setting set;
  Workspace& points = set_up(set, model, e, ue, kinematics, temperature, temperature_scale);
  if (committed.size() != points.size()) {
    throw SolverError("elastoplastic element: the stored internal variables do not match "
                      "the integration rule");
  }
  VectorI alpha = VectorI::Zero();
  if (internal != nullptr && internal->size() == kA) alpha = *internal;
  kinematics_at(set, points, alpha);
  const Material& mat = model.material_of(e);
  const bool finite = kinematics == Kinematics::Finite;
  const bool logarithmic = kinematics == Kinematics::FiniteLogarithmic;
  ElastoplasticStress out;
  out.min_jacobian = std::numeric_limits<Scalar>::infinity();
  for (std::size_t q = 0; q < points.size(); ++q) {
    const PointWork& p = points[q];
    // The committed state is the converged one at this displacement, so the
    // return reproduces it (an elastic check from it).
    const PlasticResponse r = plastic_return(mat, StressState::ThreeDimensional, p.strain,
                                             committed[q], p.change, false);
    Vector6 cauchy = r.stress;
    Vector6 piola_kirchhoff = r.stress;
    // The measures are those of the enhanced deformation, F = I + H.
    const Matrix3 sym = 0.5 * (p.h + p.h.transpose());
    const Matrix3 f = Matrix3::Identity() + p.h;
    if (logarithmic) {
      piola_kirchhoff = logarithmic_stress(p.log, r.stress);
      const Vector6 tau = voigt_of(f * tensor3(piola_kirchhoff) * f.transpose());
      Vector6 strain = p.strain;
      strain.tail<3>() *= 0.5;  // tensor components
      cauchy = tau / std::exp(strain(0) + strain(1) + strain(2));
      out.kirchhoff += tau;
      out.logarithmic_strain += strain;
      out.max_strain = std::max(out.max_strain, strain.cwiseAbs().maxCoeff());
      out.min_jacobian = std::min(out.min_jacobian, f.determinant());
    } else if (finite) {
      const Scalar j = f.determinant();
      cauchy = voigt_of(f * tensor3(r.stress) * f.transpose() / j);
      const Matrix3 green = sym + 0.5 * p.h.transpose() * p.h;
      out.max_strain = std::max(out.max_strain, green.cwiseAbs().maxCoeff());
      out.min_jacobian = std::min(out.min_jacobian, j);
    } else {
      out.max_strain = std::max(out.max_strain, sym.cwiseAbs().maxCoeff());
    }
    out.cauchy += cauchy;
    out.piola_kirchhoff += piola_kirchhoff;
    out.max_equivalent_plastic_strain =
        std::max(out.max_equivalent_plastic_strain, r.state.equivalent_plastic_strain);
  }
  const Scalar count = static_cast<Scalar>(points.size());
  out.cauchy /= count;
  out.piola_kirchhoff /= count;
  out.kirchhoff /= count;
  out.logarithmic_strain /= count;
  out.von_mises = von_mises_stress(out.cauchy);
  if (!finite && !logarithmic) out.min_jacobian = 1.0;
  return out;
}

}  // namespace detail
}  // namespace sparlab
