// Where this fits: see src/fem/ExplicitKernel.hpp

#include "ExplicitKernel.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/TotalLagrangian.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <utility>

// The element kernel is compiled twice on x86-64 GCC builds - for the
// baseline instruction set and for x86-64-v3 (AVX2 with fused multiply-add,
// about 10 % faster) - and the processor picks at load time. The two clones
// round differently in the last bits (fused products); on one machine the
// same clone always runs, so a run is still reproducible bit for bit.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__linux__)
#define SPARLAB_KERNEL_CLONES __attribute__((target_clones("arch=x86-64-v3", "default")))
#else
#define SPARLAB_KERNEL_CLONES
#endif

namespace sparlab {
namespace {

constexpr int kP = ExplicitInternalForce::kPoints;
using PointBlock = ExplicitInternalForce::PointBlock;
using Predictor = ExplicitInternalForce::Predictor;

/// The elastic/plastic decision of the kernel's own predictor is left to the
/// return itself within this band of the yield function around
/// return_3d's threshold (1e-12 of the radius), relative to the radius.
constexpr Scalar kPredictorBand = 1.0e-9;
const Scalar kSqrt23 = std::sqrt(2.0 / 3.0);

/// A deterministic pseudo-random number in [-1, 1] (splitmix64 of `i`).
Scalar unit_noise(std::uint64_t i) {
  std::uint64_t z = i + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return 2.0 * (static_cast<Scalar>(z >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
}

/// A point's logarithmic strain, what the kernel needs of
/// LogarithmicStrain.hpp's: E_log and, for S = P^T T, the principal axes and
/// the first divided differences of ln - the same operations as
/// logarithmic_strain() and logarithmic_stress(), without the 6 x 6
/// projection (a quarter of their cost).
struct PointLog {
  Matrix3 axes = Matrix3::Identity();
  Matrix3 first = Matrix3::Identity();
};

/// ln[x, y] of x = 1 + 2 mx, y = 1 + 2 my (LogarithmicStrain.cpp).
Scalar log_divided(Scalar mx, Scalar my) {
  if (mx > my) std::swap(mx, my);
  const Scalar y = 1.0 + 2.0 * my;
  const Scalar d = 2.0 * (mx - my);
  if (d == 0.0) return 1.0 / y;
  return std::log1p(d / y) / d;
}

Matrix3 tensor_of(const Scalar* v, Scalar shear_factor) {
  Matrix3 t;
  t(0, 0) = v[0];
  t(1, 1) = v[1];
  t(2, 2) = v[2];
  t(0, 1) = t(1, 0) = shear_factor * v[3];
  t(1, 2) = t(2, 1) = shear_factor * v[4];
  t(2, 0) = t(0, 2) = shear_factor * v[5];
  return t;
}

/// Where the closed-form eigensolver's result is accepted: its residual
/// |A Q - Q diag(mu)| and the loss of orthogonality |Q^T Q - I| (largest
/// entries) within this many units of round-off of |A| (largest entry) and
/// of 1 - the backward error of the iterative solver's order, so that
/// ln(A), a Lipschitz function of A, keeps the iterative solver's accuracy.
constexpr Scalar kDirectTolerance = 64.0 * std::numeric_limits<Scalar>::epsilon();

bool direct_accurate(const Matrix3& a, const Vector3& mu, const Matrix3& q) {
  const Scalar scale = a.cwiseAbs().maxCoeff();
  const Scalar residual = (a * q - q * mu.asDiagonal()).cwiseAbs().maxCoeff();
  const Scalar orthogonality = (q.transpose() * q - Matrix3::Identity()).cwiseAbs().maxCoeff();
  return residual <= kDirectTolerance * scale && orthogonality <= kDirectTolerance &&
         std::isfinite(residual);
}

/// E_log (engineering Voigt, into `strain`) of the Green-Lagrange strain
/// `green`, and the point's axes and divided differences.
/// \throws SolverError as logarithmic_strain() does.
void point_log(const Scalar* green, Scalar* strain, PointLog& out) {
  for (int r = 0; r < 6; ++r) {
    if (!std::isfinite(green[r])) {
      throw SolverError("the logarithmic strain met a non-finite Green-Lagrange strain");
    }
  }
  // The closed-form solver (a third of the iterative one's cost), checked:
  // where it loses accuracy - nearly equal eigenvalues - its residual or
  // its axes' orthogonality shows it, and the iterative solver decides.
  const Matrix3 tensor = tensor_of(green, 0.5);
  Eigen::SelfAdjointEigenSolver<Matrix3> eigen;
  eigen.computeDirect(tensor);
  if (!direct_accurate(tensor, eigen.eigenvalues(), eigen.eigenvectors())) {
    eigen.compute(tensor);
  }
  const Vector3 mu = eigen.eigenvalues();
  out.axes = eigen.eigenvectors();
  Vector3 principal;
  for (int a = 0; a < 3; ++a) {
    const Scalar stretch2 = 1.0 + 2.0 * mu(a);
    if (!(stretch2 > 0.0) || !std::isfinite(stretch2)) {
      std::ostringstream os;
      os << "a point is degenerate: a principal stretch squared of C is " << stretch2
         << "; the load step is too large or the mesh too coarse";
      throw SolverError(os.str());
    }
    principal(a) = 0.5 * std::log1p(2.0 * mu(a));
  }
  const Matrix3& q = out.axes;
  const Matrix3 e = q * principal.asDiagonal() * q.transpose();
  strain[0] = e(0, 0);
  strain[1] = e(1, 1);
  strain[2] = e(2, 2);
  strain[3] = 2.0 * e(0, 1);
  strain[4] = 2.0 * e(1, 2);
  strain[5] = 2.0 * e(2, 0);
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) out.first(a, b) = out.first(b, a) = log_divided(mu(a), mu(b));
  }
}

/// S = P^T T of a point (tensorial Voigt in and out).
void point_log_stress(const PointLog& log, const Scalar* t, Scalar* s) {
  const Matrix3& q = log.axes;
  const Matrix3 principal = (q.transpose() * tensor_of(t, 1.0) * q).cwiseProduct(log.first);
  const Matrix3 r = q * principal * q.transpose();
  s[0] = r(0, 0);
  s[1] = r(1, 1);
  s[2] = r(2, 2);
  s[3] = r(0, 1);
  s[4] = r(1, 2);
  s[5] = r(2, 0);
}

/// Everything one element's evaluation reads and writes.
struct ElementJob {
  Index element = 0;
  const Scalar* u = nullptr;           ///< the full displacement
  const Index* nodes = nullptr;        ///< the element's eight nodes
  const Scalar* point_major = nullptr; ///< G [j][a][q]
  const Scalar* node_major = nullptr;  ///< G [q][j][a]
  const Scalar* weights = nullptr;     ///< w det J [q]
  const Scalar* dilatation = nullptr;  ///< A, B, V
  const Predictor* predictor = nullptr;
  const IsotropicMaterial* material = nullptr;
  const Scalar* elasticity = nullptr;  ///< D, row-major
  PointBlock* block = nullptr;         ///< the points' history (compact)
  PlasticState* full = nullptr;        ///< the points' history (general), 8 of them
  StressState state = StressState::ThreeDimensional;
  bool finite = true;
  bool logarithmic = false;  ///< finite, with the logarithmic strain (plastic only)
  bool plastic = false;
  bool averaged = false;
  bool commit = false;
  bool want_energy = false;
  PointLog* logs = nullptr;  ///< logarithmic: kP of them (scratch)
  Scalar* elastic = nullptr; ///< with want_energy: the element's elastic energy
};

/// The element's nodal forces, node-major (out[3 a + i]), and its stored
/// energy (0 unless `want_energy`, or from the points that took the full
/// return). Plain loops over the eight points or the eight nodes: the
/// compiler vectorises them, and every sum runs in a fixed order.
/// \throws SolverError at an inverted point or a failed return.
SPARLAB_KERNEL_CLONES
Scalar element_kernel(const ElementJob& job, Scalar* out) {
  alignas(64) Scalar uu[3][kP];  // uu[i][a]
  for (int a = 0; a < kP; ++a) {
    const Scalar* src = job.u + 3 * job.nodes[a];
    uu[0][a] = src[0];
    uu[1][a] = src[1];
    uu[2][a] = src[2];
  }
  // Pass 1, over the points: H_ij = sum_a u_ia G_ja, F = I + H, the strain.
  alignas(64) Scalar h[9][kP];
  for (int k = 0; k < 9; ++k) {
    for (int q = 0; q < kP; ++q) h[k][q] = 0.0;
  }
  for (int a = 0; a < kP; ++a) {
    for (int j = 0; j < 3; ++j) {
      const Scalar* g = job.point_major + (j * kP + a) * kP;
      for (int i = 0; i < 3; ++i) {
        const Scalar uia = uu[i][a];
        Scalar* hij = h[3 * i + j];
        for (int q = 0; q < kP; ++q) hij[q] += uia * g[q];
      }
    }
  }
  alignas(64) Scalar e[6][kP];  // strain, Voigt with engineering shears
  if (job.finite) {
    alignas(64) Scalar det[kP];
    for (int q = 0; q < kP; ++q) {
      const Scalar f0 = 1.0 + h[0][q], f1 = h[1][q], f2 = h[2][q];
      const Scalar f3 = h[3][q], f4 = 1.0 + h[4][q], f5 = h[5][q];
      const Scalar f6 = h[6][q], f7 = h[7][q], f8 = 1.0 + h[8][q];
      det[q] = f0 * (f4 * f8 - f5 * f7) - f1 * (f3 * f8 - f5 * f6) + f2 * (f3 * f7 - f4 * f6);
      // E = (H + H^T + H^T H) / 2.
      const Scalar c00 = h[0][q] * h[0][q] + h[3][q] * h[3][q] + h[6][q] * h[6][q];
      const Scalar c11 = h[1][q] * h[1][q] + h[4][q] * h[4][q] + h[7][q] * h[7][q];
      const Scalar c22 = h[2][q] * h[2][q] + h[5][q] * h[5][q] + h[8][q] * h[8][q];
      const Scalar c01 = h[0][q] * h[1][q] + h[3][q] * h[4][q] + h[6][q] * h[7][q];
      const Scalar c12 = h[1][q] * h[2][q] + h[4][q] * h[5][q] + h[7][q] * h[8][q];
      const Scalar c02 = h[0][q] * h[2][q] + h[3][q] * h[5][q] + h[6][q] * h[8][q];
      e[0][q] = h[0][q] + 0.5 * c00;
      e[1][q] = h[4][q] + 0.5 * c11;
      e[2][q] = h[8][q] + 0.5 * c22;
      e[3][q] = h[1][q] + h[3][q] + c01;
      e[4][q] = h[5][q] + h[7][q] + c12;
      e[5][q] = h[6][q] + h[2][q] + c02;
    }
    for (int q = 0; q < kP; ++q) {
      if (!(det[q] > 0.0)) {
        std::ostringstream os;
        os << "element " << job.element << " is inverted (det F = " << det[q]
           << " at an integration point)";
        throw SolverError(os.str());
      }
    }
    if (job.logarithmic) {
      // The logarithmic strain of every point's E (LogarithmicStrain.hpp),
      // which the return and the mean dilatation then take as the strain.
      for (int q = 0; q < kP; ++q) {
        Scalar green[6];
        Scalar strain[6];
        for (int r = 0; r < 6; ++r) green[r] = e[r][q];
        point_log(green, strain, job.logs[q]);
        for (int r = 0; r < 6; ++r) e[r][q] = strain[r];
      }
    }
  } else {
    for (int q = 0; q < kP; ++q) {
      e[0][q] = h[0][q];
      e[1][q] = h[4][q];
      e[2][q] = h[8][q];
      e[3][q] = h[1][q] + h[3][q];
      e[4][q] = h[5][q] + h[7][q];
      e[5][q] = h[6][q] + h[2][q];
    }
  }
  const Scalar* w = job.weights;
  const Scalar volume = job.dilatation[88];
  if (job.averaged) {
    // Every point's dilatation replaced by the element's volume average.
    Scalar mean = 0.0;
    for (int q = 0; q < kP; ++q) mean += w[q] * (e[0][q] + e[1][q] + e[2][q]);
    mean /= volume;
    for (int q = 0; q < kP; ++q) {
      const Scalar shift = (mean - (e[0][q] + e[1][q] + e[2][q])) / 3.0;
      e[0][q] += shift;
      e[1][q] += shift;
      e[2][q] += shift;
    }
  }

  // The stresses: S = D E for an elastic element with finite kinematics;
  // otherwise return_3d's elastic predictor for the points that stay
  // elastic, the return itself for the others (and for all of them when the
  // energy is wanted or the material needs the general return).
  alignas(64) Scalar s[6][kP];
  Scalar energy = 0.0;
  const Predictor& pr = *job.predictor;
  const bool svk = job.finite && !job.plastic;
  bool need_return[kP];
  if (svk) {
    const Scalar* d = job.elasticity;
    for (int r = 0; r < 6; ++r) {
      for (int q = 0; q < kP; ++q) {
        s[r][q] = d[6 * r] * e[0][q] + d[6 * r + 1] * e[1][q] + d[6 * r + 2] * e[2][q] +
                  d[6 * r + 3] * e[3][q] + d[6 * r + 4] * e[4][q] + d[6 * r + 5] * e[5][q];
      }
    }
    if (job.want_energy) {
      for (int q = 0; q < kP; ++q) {
        Scalar product = 0.0;
        for (int r = 0; r < 6; ++r) product += e[r][q] * s[r][q];
        energy += w[q] * (0.5 * product);
      }
      *job.elastic = energy;
    }
    for (int q = 0; q < kP; ++q) need_return[q] = false;
  } else if (pr.radial && !job.want_energy && job.full == nullptr) {
    PointBlock* b = job.block;
    alignas(64) Scalar es[6][kP];
    for (int r = 0; r < 6; ++r) {
      for (int q = 0; q < kP; ++q) es[r][q] = b != nullptr ? e[r][q] - b->plastic[r][q] : e[r][q];
    }
    alignas(64) Scalar volumetric[kP];
    for (int q = 0; q < kP; ++q) {
      volumetric[q] = es[0][q] + es[1][q] + es[2][q];
      const Scalar mean = volumetric[q] / 3.0;
      for (int r = 0; r < 3; ++r) s[r][q] = 2.0 * pr.shear * (es[r][q] - mean);
      for (int r = 3; r < 6; ++r) s[r][q] = pr.shear * es[r][q];
    }
    if (!pr.enabled) {
      for (int q = 0; q < kP; ++q) need_return[q] = false;
    } else {
      alignas(64) Scalar radius[kP];
      alignas(64) Scalar norm[kP];
      for (int q = 0; q < kP; ++q) {
        const Scalar alpha = b != nullptr ? b->alpha[q] : 0.0;
        radius[q] = pr.yield0 + pr.hardening * alpha;
      }
      if (pr.saturation != 0.0) {
        for (int q = 0; q < kP; ++q) {
          const Scalar alpha = b != nullptr ? b->alpha[q] : 0.0;
          radius[q] += pr.saturation * -std::expm1(-pr.rate * alpha);
        }
      }
      for (int q = 0; q < kP; ++q) {
        radius[q] *= kSqrt23;
        Scalar x[6];
        for (int r = 0; r < 6; ++r) x[r] = b != nullptr ? s[r][q] - b->back[r][q] : s[r][q];
        norm[q] = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] +
                            2.0 * (x[3] * x[3] + x[4] * x[4] + x[5] * x[5]));
      }
      for (int q = 0; q < kP; ++q) {
        const Scalar f_trial = norm[q] - radius[q];
        need_return[q] = !(f_trial <= (1.0e-12 - kPredictorBand) * radius[q]);
        if (!need_return[q] && b != nullptr && job.commit) {
          // return_3d's loading flag of an elastic return.
          b->loading[q] = static_cast<unsigned char>(
              b->loading[q] != 0 && f_trial > -1.0e-10 * radius[q] && norm[q] > 0.0);
        }
      }
    }
    for (int q = 0; q < kP; ++q) {
      s[0][q] += pr.bulk * volumetric[q];
      s[1][q] += pr.bulk * volumetric[q];
      s[2][q] += pr.bulk * volumetric[q];
    }
  } else {
    for (int q = 0; q < kP; ++q) need_return[q] = true;
  }
  if (!svk) {
    for (int q = 0; q < kP; ++q) {
      if (!need_return[q]) continue;
      Vector6 strain;
      for (int r = 0; r < 6; ++r) strain(r) = e[r][q];
      PlasticState committed;
      if (job.full != nullptr) {
        committed = job.full[q];
      } else if (job.block != nullptr) {
        const PointBlock& b = *job.block;
        for (int r = 0; r < 6; ++r) {
          committed.plastic_strain(r) = b.plastic[r][q];
          committed.back_stress(r) = b.back[r][q];
        }
        committed.equivalent_plastic_strain = b.alpha[q];
        committed.thickness_strain = b.thickness[q];
        committed.loading = b.loading[q] != 0;
      }
      const PlasticResponse ret =
          plastic_return(*job.material, job.state, strain, committed, 0.0, false);
      for (int r = 0; r < 6; ++r) s[r][q] = ret.stress(r);
      energy += w[q] * ret.energy;
      if (job.want_energy) {
        // The elastic part, 1/2 sigma : (eps - eps_p) (isotropic linear
        // elasticity in the strain measure; engineering shears).
        Scalar product = 0.0;
        for (int r = 0; r < 6; ++r) {
          product += ret.stress(r) * (e[r][q] - ret.state.plastic_strain(r));
        }
        *job.elastic += w[q] * (0.5 * product);
      }
      if (!job.commit || !job.plastic) continue;
      if (job.full != nullptr) {
        job.full[q] = ret.state;
      } else if (job.block != nullptr) {
        PointBlock& b = *job.block;
        for (int r = 0; r < 6; ++r) {
          b.plastic[r][q] = ret.state.plastic_strain(r);
          b.back[r][q] = ret.state.back_stress(r);
        }
        b.alpha[q] = ret.state.equivalent_plastic_strain;
        b.thickness[q] = ret.state.thickness_strain;
        b.loading[q] = static_cast<unsigned char>(ret.state.loading);
      }
    }
  }

  // Pass 2, over the points: P = w F T with T the stress tensor (its
  // deviator with mean dilatation); then over the nodes,
  // f_a = sum_q P G_a (+ the mean pressure times the averaged dilatation's
  // variation, dbar = (A + U B) / V; A / V with small strain).
  alignas(64) Scalar t[6][kP];
  Scalar pressure = 0.0;
  for (int r = 0; r < 6; ++r) {
    for (int q = 0; q < kP; ++q) t[r][q] = s[r][q];
  }
  if (job.averaged) {
    alignas(64) Scalar third[kP];
    for (int q = 0; q < kP; ++q) {
      third[q] = (s[0][q] + s[1][q] + s[2][q]) / 3.0;
      t[0][q] -= third[q];
      t[1][q] -= third[q];
      t[2][q] -= third[q];
    }
    for (int q = 0; q < kP; ++q) pressure += w[q] * third[q];
  }
  if (job.logarithmic) {
    // The second Piola-Kirchhoff stress S = P^T T of each point (T its
    // deviator with mean dilatation), and with mean dilatation the mean
    // pressure's term: the averaged variation of ln J is
    // sum_q w_q C_q^-1 : dE_q / V, and C^-1 = P^T I, so it enters as
    // (pressure / V) C_q^-1 at every point (Elastoplastic.cpp).
    const Scalar mean = job.averaged ? pressure / volume : 0.0;
    static constexpr Scalar kUnit[6] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    for (int q = 0; q < kP; ++q) {
      Scalar tq[6];
      Scalar xq[6];
      for (int r = 0; r < 6; ++r) tq[r] = t[r][q];
      point_log_stress(job.logs[q], tq, xq);
      if (job.averaged) {
        Scalar cinv[6];
        point_log_stress(job.logs[q], kUnit, cinv);
        for (int r = 0; r < 6; ++r) xq[r] += mean * cinv[r];
      }
      for (int r = 0; r < 6; ++r) t[r][q] = xq[r];
    }
  }
  // The tensor's rows: row i holds (T_i0, T_i1, T_i2).
  static constexpr int kVoigt[3][3] = {{0, 3, 5}, {3, 1, 4}, {5, 4, 2}};
  alignas(64) Scalar p[9][kP];
  if (job.finite) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const Scalar* t0 = t[kVoigt[0][j]];
        const Scalar* t1 = t[kVoigt[1][j]];
        const Scalar* t2 = t[kVoigt[2][j]];
        const Scalar* fi0 = h[3 * i];
        const Scalar* fi1 = h[3 * i + 1];
        const Scalar* fi2 = h[3 * i + 2];
        const Scalar d0 = i == 0 ? 1.0 : 0.0;
        const Scalar d1 = i == 1 ? 1.0 : 0.0;
        const Scalar d2 = i == 2 ? 1.0 : 0.0;
        for (int q = 0; q < kP; ++q) {
          p[3 * i + j][q] = w[q] * ((d0 + fi0[q]) * t0[q] + (d1 + fi1[q]) * t1[q] +
                                    (d2 + fi2[q]) * t2[q]);
        }
      }
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const Scalar* tij = t[kVoigt[i][j]];
        for (int q = 0; q < kP; ++q) p[3 * i + j][q] = w[q] * tij[q];
      }
    }
  }
  alignas(64) Scalar fe[3][kP];  // fe[i][a]
  for (int i = 0; i < 3; ++i) {
    for (int a = 0; a < kP; ++a) fe[i][a] = 0.0;
  }
  for (int q = 0; q < kP; ++q) {
    for (int j = 0; j < 3; ++j) {
      const Scalar* g = job.node_major + (q * 3 + j) * kP;
      for (int i = 0; i < 3; ++i) {
        const Scalar pij = p[3 * i + j][q];
        for (int a = 0; a < kP; ++a) fe[i][a] += pij * g[a];
      }
    }
  }
  if (job.averaged && !job.logarithmic) {
    const Scalar* amat = job.dilatation;       // A[i][a]
    const Scalar* bmat = job.dilatation + 24;  // B[a][b]
    const Scalar factor = pressure / volume;
    for (int i = 0; i < 3; ++i) {
      alignas(64) Scalar d[kP];
      for (int b = 0; b < kP; ++b) d[b] = amat[kP * i + b];
      if (job.finite) {
        for (int a = 0; a < kP; ++a) {
          const Scalar uia = uu[i][a];
          const Scalar* row = bmat + kP * a;
          for (int b = 0; b < kP; ++b) d[b] += uia * row[b];
        }
      }
      for (int b = 0; b < kP; ++b) fe[i][b] += factor * d[b];
    }
  }
  for (int a = 0; a < kP; ++a) {
    out[3 * a] = fe[0][a];
    out[3 * a + 1] = fe[1][a];
    out[3 * a + 2] = fe[2][a];
  }
  return energy;
}

}  // namespace

ExplicitInternalForce::ExplicitInternalForce(const FemModel& model, const Assembler& assembler,
                                             const NonlinearOptions& options,
                                             bool allow_dedicated)
    : model_(model), assembler_(assembler), options_(options) {
  system_ = std::make_unique<detail::NonlinearSystem>(model, assembler, options_);
  // The dedicated kernel covers what it reproduces exactly; everything else
  // takes the element dispatch.
  const Mesh& mesh = model.mesh();
  if (!allow_dedicated) {
    reason_ = "the dedicated kernel is switched off";
  } else if (options_.kinematics != Kinematics::Finite &&
             options_.kinematics != Kinematics::SmallStrain &&
             options_.kinematics != Kinematics::FiniteLogarithmic) {
    reason_ = "the dedicated kernel covers finite, logarithmic and small-strain kinematics";
  } else if (elastoplastic_points(model) != kPoints) {
    reason_ = "the dedicated kernel takes the 2 x 2 x 2 rule";
  } else if (model.stress_state() != StressState::ThreeDimensional) {
    reason_ = "the dedicated kernel is three-dimensional";
  } else {
    logarithmic_ = options_.kinematics == Kinematics::FiniteLogarithmic;
    finite_ = options_.kinematics == Kinematics::Finite || logarithmic_;
    bool elastic_element = false;
    for (Index e = 0; e < mesh.num_elements(); ++e) {
      if (!system_->elastoplastic(e)) elastic_element = true;
    }
    if (logarithmic_ && elastic_element) {
      reason_ = "the dedicated kernel's logarithmic kinematics are written for elastoplastic "
                "elements";
    }
    if (reason_.empty() && finite_ && elastic_element &&
        options_.law != HyperelasticModel::SaintVenantKirchhoff) {
      reason_ = "the dedicated kernel's elastic law with finite kinematics is Saint "
                "Venant-Kirchhoff";
    }
  }
  if (reason_.empty()) build_dedicated();
}

void ExplicitInternalForce::build_dedicated() {
  const Mesh& mesh = model_.mesh();
  const Element& element = model_.element();
  ne_ = mesh.num_elements();
  nn_ = mesh.num_nodes();
  const std::vector<IntegrationPoint> rule = element.integration_rule(model_.integration());
  const auto& info = mesh.structured_info();
  uniform_ = info.has_value() && info->uniform;
  const Index geometric = uniform_ ? 1 : ne_;
  const auto g = static_cast<std::size_t>(geometric);
  gradients_point_.assign(g * 192, 0.0);
  gradients_node_.assign(g * 192, 0.0);
  weights_.assign(g * kPoints, 0.0);
  dilatation_.assign(g * 89, 0.0);
  for (Index e = 0; e < geometric; ++e) {
    const auto ue = static_cast<std::size_t>(e);
    const Matrix x0 = mesh.element_coordinates(e);
    Scalar* point_major = gradients_point_.data() + ue * 192;
    Scalar* node_major = gradients_node_.data() + ue * 192;
    Scalar* dil = dilatation_.data() + ue * 89;
    for (int q = 0; q < kPoints; ++q) {
      const IntegrationPoint& ip = rule[static_cast<std::size_t>(q)];
      const StrainOperator op = element.strain_operator(x0, ip.point);
      const Matrix grad = reference_gradients(op, 3, 8);
      const Scalar w = ip.weight * op.detJ;
      weights_[ue * kPoints + static_cast<std::size_t>(q)] = w;
      for (int j = 0; j < 3; ++j) {
        for (int a = 0; a < 8; ++a) {
          point_major[(j * 8 + a) * 8 + q] = grad(j, a);
          node_major[(q * 3 + j) * 8 + a] = grad(j, a);
          dil[8 * j + a] += w * grad(j, a);
        }
      }
      for (int a = 0; a < 8; ++a) {
        for (int b = 0; b < 8; ++b) {
          dil[24 + 8 * a + b] += w * (grad(0, a) * grad(0, b) + grad(1, a) * grad(1, b) +
                                      grad(2, a) * grad(2, b));
        }
      }
      dil[88] += w;
    }
  }
  material_.assign(static_cast<std::size_t>(ne_), 0);
  plastic_.assign(static_cast<std::size_t>(ne_), 0);
  averaged_.assign(static_cast<std::size_t>(ne_), 0);
  slot_.assign(static_cast<std::size_t>(ne_), -1);
  compact_ = true;
  Index slots = 0;
  for (Index e = 0; e < ne_; ++e) {
    const IsotropicMaterial* mat = &model_.material_of(e);
    auto it = std::find(materials_.begin(), materials_.end(), mat);
    if (it == materials_.end()) {
      materials_.push_back(mat);
      const Matrix d = mat->constitutive(StressState::ThreeDimensional);
      for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) elasticity_.push_back(d(r, c));
      }
      Predictor pr;
      const PlasticityParameters& pp = mat->plasticity();
      pr.shear = mat->shear_modulus();
      pr.bulk = mat->lame_lambda() + 2.0 * pr.shear / 3.0;
      pr.enabled = pp.enabled();
      pr.radial = !pp.general();
      pr.yield0 = pp.yield_stress;
      pr.hardening = pp.hardening_modulus;
      pr.saturation = pp.saturation_stress;
      pr.rate = pp.saturation_rate;
      predictor_.push_back(pr);
      it = materials_.end() - 1;
    }
    const auto ue = static_cast<std::size_t>(e);
    material_[ue] = static_cast<int>(it - materials_.begin());
    plastic_[ue] = system_->elastoplastic(e) ? 1 : 0;
    averaged_[ue] = system_->averaged(e) ? 1 : 0;
    if (plastic_[ue]) {
      slot_[ue] = slots++;
      if (mat->plasticity().general()) compact_ = false;
    }
  }
  if (compact_) {
    PointBlock virgin;
    for (int r = 0; r < 6; ++r) {
      for (int q = 0; q < kPoints; ++q) virgin.plastic[r][q] = virgin.back[r][q] = 0.0;
    }
    for (int q = 0; q < kPoints; ++q) {
      virgin.alpha[q] = virgin.thickness[q] = 0.0;
      virgin.loading[q] = 0;
    }
    blocks_.assign(static_cast<std::size_t>(slots), virgin);
  } else {
    full_state_.assign(static_cast<std::size_t>(slots) * kPoints, PlasticState());
  }
  element_force_.assign(static_cast<std::size_t>(ne_) * 24, 0.0);
  element_energy_.assign(static_cast<std::size_t>(ne_), 0.0);
  element_elastic_.assign(static_cast<std::size_t>(ne_), 0.0);
  // Node -> (element, local node), ascending element (the order in which
  // NonlinearSystem's serial assembly adds the element forces).
  incidence_ptr_.assign(static_cast<std::size_t>(nn_) + 1, 0);
  for (Index e = 0; e < ne_; ++e) {
    const Index* nodes = mesh.element_nodes(e);
    for (int a = 0; a < 8; ++a) ++incidence_ptr_[static_cast<std::size_t>(nodes[a]) + 1];
  }
  for (Index n = 0; n < nn_; ++n) {
    incidence_ptr_[static_cast<std::size_t>(n) + 1] += incidence_ptr_[static_cast<std::size_t>(n)];
  }
  incidence_.assign(static_cast<std::size_t>(incidence_ptr_.back()), 0);
  std::vector<Index> fill(incidence_ptr_.begin(), incidence_ptr_.end() - 1);
  for (Index e = 0; e < ne_; ++e) {
    const Index* nodes = mesh.element_nodes(e);
    for (int a = 0; a < 8; ++a) {
      incidence_[static_cast<std::size_t>(fill[static_cast<std::size_t>(nodes[a])]++)] =
          e * 24 + 3 * a;
    }
  }
  dead_ = system_->dead();
  dedicated_ = true;
}

void ExplicitInternalForce::use_generic(const std::string& reason) {
  if (!dedicated_) return;
  system_->set_committed(history());
  dedicated_ = false;
  reason_ = reason;
}

PlasticState ExplicitInternalForce::point_state(Index e, int q) const {
  const auto slot = static_cast<std::size_t>(slot_[static_cast<std::size_t>(e)]);
  if (!compact_) return full_state_[slot * kPoints + static_cast<std::size_t>(q)];
  const PointBlock& b = blocks_[slot];
  PlasticState p;
  for (int r = 0; r < 6; ++r) {
    p.plastic_strain(r) = b.plastic[r][q];
    p.back_stress(r) = b.back[r][q];
  }
  p.equivalent_plastic_strain = b.alpha[q];
  p.thickness_strain = b.thickness[q];
  p.loading = b.loading[q] != 0;
  return p;
}

void ExplicitInternalForce::set_point_state(Index e, int q, const PlasticState& p) {
  const auto slot = static_cast<std::size_t>(slot_[static_cast<std::size_t>(e)]);
  if (!compact_) {
    full_state_[slot * kPoints + static_cast<std::size_t>(q)] = p;
    return;
  }
  PointBlock& b = blocks_[slot];
  for (int r = 0; r < 6; ++r) {
    b.plastic[r][q] = p.plastic_strain(r);
    b.back[r][q] = p.back_stress(r);
  }
  b.alpha[q] = p.equivalent_plastic_strain;
  b.thickness[q] = p.thickness_strain;
  b.loading[q] = static_cast<unsigned char>(p.loading);
}

void ExplicitInternalForce::set_history(const std::vector<std::vector<PlasticState>>& history) {
  if (!dedicated_) {
    if (history.empty()) {
      std::vector<std::vector<PlasticState>> virgin = system_->committed_all();
      for (std::vector<PlasticState>& e : virgin) {
        for (PlasticState& p : e) p = PlasticState();
      }
      system_->set_committed(std::move(virgin));
    } else {
      system_->set_committed(history);
    }
    return;
  }
  if (history.size() != static_cast<std::size_t>(ne_) && !history.empty()) {
    throw ConfigError("the saved plastic history has " + std::to_string(history.size()) +
                      " element(s); the model has " + std::to_string(ne_));
  }
  for (Index e = 0; e < ne_; ++e) {
    const auto ue = static_cast<std::size_t>(e);
    const std::size_t expected = plastic_[ue] ? static_cast<std::size_t>(kPoints) : 0;
    if (!history.empty() && history[ue].size() != expected) {
      throw ConfigError("the saved plastic history of element " + std::to_string(e) + " has " +
                        std::to_string(history[ue].size()) + " point(s); the model keeps " +
                        std::to_string(expected) + " for it (none for an elastic element)");
    }
    if (!plastic_[ue]) continue;
    for (int q = 0; q < kPoints; ++q) {
      set_point_state(e, q, history.empty() ? PlasticState()
                                            : history[ue][static_cast<std::size_t>(q)]);
    }
  }
}

std::vector<std::vector<PlasticState>> ExplicitInternalForce::history() const {
  if (!dedicated_) return system_->committed_all();
  std::vector<std::vector<PlasticState>> out(static_cast<std::size_t>(ne_));
  for (Index e = 0; e < ne_; ++e) {
    const auto ue = static_cast<std::size_t>(e);
    if (!plastic_[ue]) continue;
    out[ue].reserve(static_cast<std::size_t>(kPoints));
    for (int q = 0; q < kPoints; ++q) out[ue].push_back(point_state(e, q));
  }
  return out;
}

Scalar ExplicitInternalForce::max_plastic_strain() const {
  if (!dedicated_) return system_->max_plastic_strain();
  Scalar top = 0.0;
  for (const PointBlock& b : blocks_) {
    for (int q = 0; q < kPoints; ++q) top = std::max(top, b.alpha[q]);
  }
  for (const PlasticState& p : full_state_) top = std::max(top, p.equivalent_plastic_strain);
  return top;
}

void ExplicitInternalForce::checkpoint() {
  if (dedicated_) {
    blocks_saved_ = blocks_;
    full_saved_ = full_state_;
  } else {
    generic_saved_ = system_->committed_all();
  }
}

void ExplicitInternalForce::restore() {
  if (dedicated_) {
    if (blocks_saved_.size() == blocks_.size()) blocks_ = blocks_saved_;
    if (full_saved_.size() == full_state_.size()) full_state_ = full_saved_;
  } else if (generic_saved_.size() == system_->committed_all().size()) {
    system_->set_committed(generic_saved_);
  }
}

void ExplicitInternalForce::evaluate(const Vector& u, Scalar lambda, bool commit,
                                     Vector& internal, Vector& external, Scalar& energy,
                                     bool want_energy) {
  if (dedicated_) {
    dedicated_forces(u, lambda, commit, internal, external, energy, want_energy);
    return;
  }
  detail::Evaluation ev = system_->evaluate(u, lambda, false);
  internal = std::move(ev.internal);
  external = std::move(ev.external);
  energy = ev.energy;
  elastic_ = std::numeric_limits<Scalar>::quiet_NaN();
  if (commit) system_->commit(ev);
}

Vector ExplicitInternalForce::generic_internal(
    const Vector& u, const std::vector<std::vector<PlasticState>>& history) const {
  std::unique_ptr<detail::NonlinearSystem> system =
      std::make_unique<detail::NonlinearSystem>(model_, assembler_, options_);
  if (!history.empty()) system->set_committed(history);
  return system->evaluate(u, 0.0, false).internal;
}

Scalar ExplicitInternalForce::self_check(const Vector& u) {
  if (!dedicated_) return 0.0;
  // A perturbation of 1e-5 of the smallest element size: a strain of that
  // order, elastic from the history's state or not - the same returns
  // either way.
  Scalar h = std::numeric_limits<Scalar>::infinity();
  for (std::size_t e = 0; e < weights_.size() / kPoints; ++e) {
    Scalar v = 0.0;
    for (int q = 0; q < kPoints; ++q) v += weights_[e * kPoints + static_cast<std::size_t>(q)];
    h = std::min(h, std::cbrt(v));
  }
  Vector probe = u;
  for (Eigen::Index i = 0; i < probe.size(); ++i) {
    probe(i) += 1.0e-5 * h * unit_noise(static_cast<std::uint64_t>(i));
  }
  Vector fd(u.size());
  Vector external(u.size());
  Scalar energy = 0.0;
  // Both through the predictor and through the full returns.
  dedicated_forces(probe, 0.0, false, fd, external, energy, false);
  Vector fr(u.size());
  dedicated_forces(probe, 0.0, false, fr, external, energy, true);
  system_->set_committed(history());
  const Vector fg = system_->evaluate(probe, 0.0, false).internal;
  const Scalar scale = std::max(fg.norm(), std::numeric_limits<Scalar>::min());
  return std::max((fd - fg).norm(), (fr - fg).norm()) / scale;
}

Index ExplicitInternalForce::first_inverted(const Vector& u) const {
  const Mesh& mesh = model_.mesh();
  const int dim = mesh.dim();
  const int npe = mesh.nodes_per_elem();
  if (dedicated_ && !finite_) return -1;
  const Element& element = model_.element();
  const std::vector<IntegrationPoint> rule = element.integration_rule(model_.integration());
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    Matrix x = mesh.element_coordinates(e);
    const Index* nodes = mesh.element_nodes(e);
    for (int a = 0; a < npe; ++a) {
      for (int k = 0; k < dim; ++k) x(k, a) += u(nodes[a] * dim + k);
    }
    try {
      for (const IntegrationPoint& ip : rule) (void)element.strain_operator(x, ip.point);
    } catch (const SparLabError&) {
      return e;
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------
// The dedicated Hex8 kernel: the elements, then the gather
// ---------------------------------------------------------------------------
void ExplicitInternalForce::dedicated_forces(const Vector& u, Scalar lambda, bool commit,
                                             Vector& internal, Vector& external,
                                             Scalar& energy, bool want_energy) {
  const Mesh& mesh = model_.mesh();
  const Index* connectivity = mesh.element_nodes(0);
  const StressState state = model_.stress_state();
  // The first failure, rethrown as it was (its type and message).
  std::exception_ptr failure;
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp parallel
#endif
  {
  // Logarithmic kinematics' per-point scratch, once per thread.
  PointLog logs[kPoints];
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp for schedule(static)
#endif
  for (Index e = 0; e < ne_; ++e) {
    try {
      const auto ue = static_cast<std::size_t>(e);
      const std::size_t geometric = uniform_ ? 0 : ue;
      const int mi = material_[ue];
      ElementJob job;
      job.element = e;
      job.u = u.data();
      job.nodes = connectivity + 8 * ue;
      job.point_major = gradients_point_.data() + geometric * 192;
      job.node_major = gradients_node_.data() + geometric * 192;
      job.weights = weights_.data() + geometric * kPoints;
      job.dilatation = dilatation_.data() + geometric * 89;
      job.predictor = &predictor_[static_cast<std::size_t>(mi)];
      job.material = materials_[static_cast<std::size_t>(mi)];
      job.elasticity = elasticity_.data() + 36 * static_cast<std::size_t>(mi);
      job.state = state;
      job.finite = finite_;
      job.logarithmic = logarithmic_;
      job.logs = logs;
      job.plastic = plastic_[ue] != 0;
      job.averaged = averaged_[ue] != 0;
      job.commit = commit;
      job.want_energy = want_energy;
      element_elastic_[ue] = 0.0;
      job.elastic = &element_elastic_[ue];
      if (job.plastic) {
        const auto slot = static_cast<std::size_t>(slot_[ue]);
        if (compact_) {
          job.block = &blocks_[slot];
        } else {
          job.full = &full_state_[slot * kPoints];
        }
      }
      element_energy_[ue] = element_kernel(job, element_force_.data() + ue * 24);
    } catch (const SolverError&) {
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp critical(sparlab_explicit_failure)
#endif
      if (!failure) failure = std::current_exception();
    } catch (const std::exception& ex) {
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp critical(sparlab_explicit_failure)
#endif
      if (!failure) failure = std::make_exception_ptr(SolverError(ex.what()));
    }
  }
  }
  if (failure) std::rethrow_exception(failure);

  // Gather: every node sums its elements' forces in ascending element order.
  if (internal.size() != 3 * nn_) internal.resize(3 * nn_);
  const Scalar* forces = element_force_.data();
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (Index node = 0; node < nn_; ++node) {
    Scalar s0 = 0.0;
    Scalar s1 = 0.0;
    Scalar s2 = 0.0;
    const Index end = incidence_ptr_[static_cast<std::size_t>(node) + 1];
    for (Index k = incidence_ptr_[static_cast<std::size_t>(node)]; k < end; ++k) {
      const Scalar* fk = forces + incidence_[static_cast<std::size_t>(k)];
      s0 += fk[0];
      s1 += fk[1];
      s2 += fk[2];
    }
    internal(3 * node) = s0;
    internal(3 * node + 1) = s1;
    internal(3 * node + 2) = s2;
  }
  energy = 0.0;
  if (want_energy) {
    for (Scalar v : element_energy_) energy += v;
    elastic_ = 0.0;
    for (Scalar v : element_elastic_) elastic_ += v;
  }
  if (external.size() != dead_.size()) external.resize(dead_.size());
  external.noalias() = lambda * dead_;
}

}  // namespace sparlab
