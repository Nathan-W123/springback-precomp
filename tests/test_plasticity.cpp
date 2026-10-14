/// \file test_plasticity.cpp
/// \brief J2 plasticity: the return map at a material point and the
///        elastoplastic element.
///
/// Every check is exact or a derivative check. Uniaxial stress is a
/// proportional path, on which the backward-Euler radial return with linear
/// hardening is exact for any step, so stresses are compared with the
/// closed-form curves to round-off - monotonic, reversed (Bauschinger) and
/// with Voce saturation. Consistent tangents are compared with central
/// differences of the stress (3-D, plane strain, plane stress) and of the
/// element's internal force (with and without mean dilatation).
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/Elastoplastic.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;

namespace {

Material plastic_steel(Scalar h = 0.0, Scalar hk = 0.0, Scalar q = 0.0,
                                Scalar delta = 0.0) {
  Material m(200.0e9, 0.3, 7800.0, "steel");
  PlasticityParameters p;
  p.yield_stress = 250.0e6;
  p.hardening_modulus = h;
  p.kinematic_hardening_modulus = hk;
  p.saturation_stress = q;
  p.saturation_rate = delta;
  m.set_plasticity(p);
  return m;
}

/// Uniaxial stress sigma_11 at the axial strain e11: the lateral strains
/// found by Newton on the return so that sigma_22 = sigma_33 = 0.
PlasticResponse uniaxial(const Material& m, Scalar e11, const PlasticState& committed,
                         Scalar delta_t = 0.0) {
  const Scalar th = m.thermal_expansion() * delta_t;
  Vector6 strain = Vector6::Zero();
  strain(0) = e11;
  for (int i = 1; i < 3; ++i) {
    strain(i) = committed.plastic_strain(i) + th -
                m.poisson_ratio() * (e11 - committed.plastic_strain(0) - th);
  }
  PlasticResponse r;
  for (int it = 0; it < 50; ++it) {
    r = plastic_return(m, StressState::ThreeDimensional, strain, committed, delta_t);
    const Eigen::Vector2d residual(r.stress(1), r.stress(2));
    if (residual.cwiseAbs().maxCoeff() <=
        1.0e-13 * std::max(std::abs(r.stress(0)), m.plasticity().yield_stress)) {
      return r;
    }
    Eigen::Matrix2d j;
    j << r.tangent(1, 1), r.tangent(1, 2), r.tangent(2, 1), r.tangent(2, 2);
    const Eigen::Vector2d step = j.lu().solve(residual);
    strain(1) -= step(0);
    strain(2) -= step(1);
  }
  FAIL("the uniaxial-stress iteration did not converge");
  return r;
}

/// The uniaxial stress of a Voce material at total strain e (monotonic):
/// sigma = sigma_y(e - sigma / E), by bisection.
Scalar voce_uniaxial(const Material& m, Scalar e) {
  const PlasticityParameters& p = m.plasticity();
  Scalar lo = p.yield_stress;
  Scalar hi = m.youngs_modulus() * e;
  for (int it = 0; it < 200; ++it) {
    const Scalar mid = 0.5 * (lo + hi);
    (mid - p.yield(e - mid / m.youngs_modulus()) > 0.0 ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

Vector6 random_strain(Scalar amplitude, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  Vector6 v;
  for (int i = 0; i < 6; ++i) v(i) = amplitude * dist(gen);
  return v;
}

/// The consistent tangent against central differences of the stress, over
/// the strain components the idealisation takes.
Scalar tangent_error(const Material& m, StressState state, const Vector6& strain,
                     const PlasticState& committed, Scalar delta_t = 0.0) {
  const PlasticResponse r = plastic_return(m, state, strain, committed, delta_t);
  // (The plane-stress iteration leaves sigma_33 below 1e-12 of the stress,
  // which a step of 1e-8 keeps well below the tolerance.)
  const Scalar h = 1.0e-8;
  Scalar worst = 0.0;
  for (int j = 0; j < 6; ++j) {
    if (state != StressState::ThreeDimensional && (j == 4 || j == 5)) continue;
    if (state == StressState::PlaneStress && j == 2) continue;
    Vector6 up = strain;
    Vector6 um = strain;
    up(j) += h;
    um(j) -= h;
    const Vector6 fd = (plastic_return(m, state, up, committed, delta_t).stress -
                        plastic_return(m, state, um, committed, delta_t).stress) /
                       (2.0 * h);
    Vector6 column = r.tangent.col(j);
    worst = std::max(worst, (fd - column).cwiseAbs().maxCoeff());
  }
  return worst / r.tangent.cwiseAbs().maxCoeff();
}

}  // namespace

TEST_CASE("plasticity parameters are validated and the hardening laws are consistent",
          "[plasticity][material]") {
  Material m(200.0e9, 0.3, 7800.0, "steel");
  PlasticityParameters p;
  p.yield_stress = -1.0;
  REQUIRE_THROWS_AS(m.set_plasticity(p), ConfigError);
  p.yield_stress = 250.0e6;
  p.saturation_stress = 100.0e6;  // no rate
  REQUIRE_THROWS_AS(m.set_plasticity(p), ConfigError);
  p = PlasticityParameters();
  p.hardening_modulus = 1.0e9;  // hardening of an elastic material
  REQUIRE_THROWS_AS(m.set_plasticity(p), ConfigError);
  p.yield_stress = std::numeric_limits<Scalar>::quiet_NaN();
  REQUIRE_THROWS_AS(m.set_plasticity(p), ConfigError);

  // sigma_y' is the derivative of sigma_y, and the stored isotropic
  // hardening energy the integral of sigma_y - sigma_y0.
  p = PlasticityParameters();
  p.yield_stress = 250.0e6;
  p.hardening_modulus = 1.5e9;
  p.saturation_stress = 120.0e6;
  p.saturation_rate = 25.0;
  REQUIRE_NOTHROW(m.set_plasticity(p));
  for (const Scalar a : {0.0, 0.003, 0.04, 0.2}) {
    const Scalar h = 1.0e-7;
    REQUIRE((p.yield(a + h) - p.yield(a - (a > 0 ? h : 0.0))) / (a > 0 ? 2.0 * h : h) ==
            Approx(p.yield_slope(a)).epsilon(1e-6));
    if (a > 0.0) {
      REQUIRE((p.isotropic_energy(a + h) - p.isotropic_energy(a - h)) / (2.0 * h) ==
              Approx(p.yield(a) - p.yield_stress).epsilon(1e-6));
    }
  }
  REQUIRE(p.isotropic_energy(0.0) == 0.0);
}

TEST_CASE("uniaxial stress with linear hardening is exact for any step",
          "[plasticity][material]") {
  // sigma = sigma_y0 + (H + H_kin) eps_p on a monotonic path, so
  // sigma = E (sigma_y0 + (H + H_kin) eps) / (E + H + H_kin).
  struct Case {
    Scalar h;
    Scalar hk;
  };
  for (const Case c : {Case{0.0, 0.0}, Case{2.0e9, 0.0}, Case{0.0, 5.0e9}, Case{1.0e9, 3.0e9}}) {
    const Material m = plastic_steel(c.h, c.hk);
    const Scalar e = m.youngs_modulus();
    const Scalar hh = c.h + c.hk;
    const Scalar target = 0.01;  // eight times the yield strain
    const Scalar exact = e * (250.0e6 + hh * target) / (e + hh);
    // One step.
    const PlasticResponse one = uniaxial(m, target, PlasticState());
    INFO("H = " << c.h << ", H_kin = " << c.hk);
    REQUIRE(one.yielding);
    REQUIRE(one.stress(0) == Approx(exact).epsilon(1e-12));
    // Forty steps, each from the state the last one committed.
    PlasticState state;
    PlasticResponse r;
    for (int k = 1; k <= 40; ++k) {
      r = uniaxial(m, target * k / 40.0, state);
      state = r.state;
    }
    REQUIRE(r.stress(0) == Approx(exact).epsilon(1e-12));
    const Scalar eps_p = target - exact / e;
    REQUIRE(state.plastic_strain(0) == Approx(eps_p).epsilon(1e-10));
    REQUIRE(state.plastic_strain(1) == Approx(-0.5 * eps_p).epsilon(1e-10));
    REQUIRE(state.equivalent_plastic_strain == Approx(eps_p).epsilon(1e-10));
    REQUIRE(von_mises_stress(r.stress) == Approx(exact).epsilon(1e-12));
    // Stored energy: elastic sigma^2 / 2E plus H alpha^2 / 2 plus the back
    // stress's H_kin eps_p^2 / 2.
    REQUIRE(r.energy == Approx(exact * exact / (2.0 * e) + 0.5 * hh * eps_p * eps_p)
                            .epsilon(1e-10));
    // The lateral contraction is elastic plus plastic, -nu sigma/E - eps_p/2.
    REQUIRE(uniaxial(m, target, PlasticState()).state.plastic_strain(2) ==
            Approx(-0.5 * eps_p).epsilon(1e-10));
  }
}

TEST_CASE("a reversed uniaxial path shows isotropic and kinematic hardening exactly",
          "[plasticity][material]") {
  const Scalar sy = 250.0e6;
  const Scalar e1 = 0.01;
  for (const bool kinematic : {false, true}) {
    const Scalar hmod = 10.0e9;
    const Material m = kinematic ? plastic_steel(0.0, hmod) : plastic_steel(hmod, 0.0);
    const Scalar e = m.youngs_modulus();
    const Scalar s1 = e * (sy + hmod * e1) / (e + hmod);
    const Scalar ep1 = e1 - s1 / e;
    // Reverse yield: isotropic at -(sigma_y0 + H eps_p1); kinematic at
    // H_kin eps_p1 - sigma_y0 (the Bauschinger effect).
    const Scalar reverse_yield = kinematic ? hmod * ep1 - sy : -(sy + hmod * ep1);
    PlasticState state = uniaxial(m, e1, PlasticState()).state;
    // Unload halfway to reverse yield: elastic.
    const Scalar e_mid = e1 - 0.5 * (s1 - reverse_yield) / e;
    PlasticResponse r = uniaxial(m, e_mid, state);
    INFO((kinematic ? "kinematic" : "isotropic"));
    REQUIRE_FALSE(r.yielding);
    REQUIRE(r.stress(0) == Approx(0.5 * (s1 + reverse_yield)).epsilon(1e-12));
    state = r.state;
    // Past reverse yield to -e1, in one step.
    r = uniaxial(m, -e1, state);
    REQUIRE(r.yielding);
    // Isotropic: sigma = -(sigma_y0 + H alpha), alpha = eps_p1 + (eps_p1 -
    // eps_p), so sigma = E (H e - sigma_y0 - 2 H eps_p1) / (E + H).
    // Kinematic: sigma = H_kin eps_p - sigma_y0, so sigma = E (H_kin e -
    // sigma_y0) / (E + H_kin).
    const Scalar exact = kinematic ? e * (hmod * -e1 - sy) / (e + hmod)
                                   : e * (hmod * -e1 - sy - 2.0 * hmod * ep1) / (e + hmod);
    REQUIRE(r.stress(0) == Approx(exact).epsilon(1e-12));
    // The kinematic material's surface kept its size; the isotropic one's grew.
    const Scalar radius = von_mises_stress(r.stress - r.state.back_stress);
    REQUIRE(radius == Approx(kinematic ? sy : sy + hmod * r.state.equivalent_plastic_strain)
                          .epsilon(1e-10));
  }
}

TEST_CASE("Voce saturation is followed exactly on a uniaxial path", "[plasticity][material]") {
  const Material m = plastic_steel(1.0e9, 0.0, 150.0e6, 30.0);
  for (const Scalar target : {0.002, 0.02, 0.2}) {
    const Scalar exact = voce_uniaxial(m, target);
    REQUIRE(uniaxial(m, target, PlasticState()).stress(0) == Approx(exact).epsilon(1e-11));
    PlasticState state;
    PlasticResponse r;
    for (int k = 1; k <= 25; ++k) {
      r = uniaxial(m, target * k / 25.0, state);
      state = r.state;
    }
    REQUIRE(r.stress(0) == Approx(exact).epsilon(1e-11));
  }
}

TEST_CASE("the Voce return converges at trial stresses far outside the yield surface",
          "[plasticity][material]") {
  // A deviatoric trial increment of order one - a large-strain Newton
  // iterate reaches it (logarithmic kinematics) - puts |xi_trial| a
  // thousand times beyond the radius sqrt(2/3) sigma_y, where one ulp of
  // |xi_trial| exceeds 1e-13 of the radius: Newton sits on the root to
  // round-off but cannot meet that tolerance, and the return used to throw.
  // It must converge, onto the yield surface, in 3-D and in plane stress.
  struct Law {
    Scalar h, hk, q, delta;
  };
  const std::vector<Law> laws = {{1.0e9, 0.0, 60.0e6, 30.0},
                                 {1.0e9, 2.0e9, 60.0e6, 30.0},
                                 {0.5e9, 0.0, 150.0e6, 12.0}};
  Scalar worst = 0.0;
  int returns = 0;
  for (const Law& law : laws) {
    const Material m = plastic_steel(law.h, law.hk, law.q, law.delta);
    const PlasticityParameters& p = m.plasticity();
    // | |dev(sigma) - beta| - sqrt(2/3) sigma_y(alpha) | relative to sigma_y.
    const auto off_surface = [&](const PlasticResponse& r) {
      Vector6 xi = r.stress - r.state.back_stress;
      xi.head(3).array() -= (r.stress(0) + r.stress(1) + r.stress(2)) / 3.0;
      const Scalar norm = std::sqrt(xi.head(3).squaredNorm() + 2.0 * xi.tail(3).squaredNorm());
      const Scalar radius = std::sqrt(2.0 / 3.0) * p.yield(r.state.equivalent_plastic_strain);
      return std::abs(norm - radius) / p.yield_stress;
    };
    std::vector<Vector6> strains;
    for (int i = 0; i <= 400; ++i) {
      Vector6 e = Vector6::Zero();
      e(0) = std::log(2.2);
      e(1) = std::log(1.3);
      e(2) = -1.0 + 2.0 * i / 400.0;
      strains.push_back(e);
    }
    for (int i = 0; i <= 40; ++i) {
      const Scalar a = 0.5 + 0.025 * i;  // isochoric, with a shear
      Vector6 e = Vector6::Zero();
      e << a, -0.5 * a, -0.5 * a, 0.3 * a, 0.0, 0.0;
      strains.push_back(e);
    }
    for (const Vector6& e : strains) {
      INFO("h " << law.h << ", hk " << law.hk << ", strain " << e.transpose());
      PlasticResponse r;
      REQUIRE_NOTHROW(r = plastic_return(m, StressState::ThreeDimensional, e, PlasticState()));
      REQUIRE(r.yielding);
      worst = std::max(worst, off_surface(r));
      ++returns;
    }
    Vector6 sheet = Vector6::Zero();
    sheet(0) = std::log(2.2);
    sheet(1) = std::log(1.3);
    PlasticResponse r;
    REQUIRE_NOTHROW(r = plastic_return(m, StressState::PlaneStress, sheet, PlasticState()));
    REQUIRE(r.yielding);
    // sigma_33 = 0 to 1e-12 of the stress of the first iterate, the 3-D
    // return at the elastic thickness strain (whose volumetric stress,
    // 1e11 Pa, sets the plane-stress iteration's scale).
    Vector6 first = sheet;
    first(2) = -m.lame_lambda() / (m.lame_lambda() + 2.0 * m.shear_modulus()) *
               (sheet(0) + sheet(1));
    const Scalar scale =
        plastic_return(m, StressState::ThreeDimensional, first, PlasticState()).stress.head(3)
            .cwiseAbs()
            .maxCoeff();
    REQUIRE(std::abs(r.stress(2)) <= 1.0e-12 * scale);
    worst = std::max(worst, off_surface(r));
  }
  INFO("worst |f| " << worst << " sigma_y over " << returns << " returns");
  REQUIRE(worst <= 1.0e-12);
}

TEST_CASE("the consistent tangent is the derivative of the return", "[plasticity][material]") {
  // Every hardening mechanism at once, from a committed state with plastic
  // strain, back stress and accumulated strain, on a non-proportional
  // increment that yields again.
  const Material m = plastic_steel(1.5e9, 4.0e9, 100.0e6, 20.0);
  for (const StressState state :
       {StressState::ThreeDimensional, StressState::PlaneStrain, StressState::PlaneStress}) {
    Vector6 first = random_strain(4.0e-3, 3u);
    Vector6 second = first + random_strain(3.0e-3, 5u);
    if (state != StressState::ThreeDimensional) {
      first(4) = first(5) = second(4) = second(5) = 0.0;
      if (state == StressState::PlaneStrain) first(2) = second(2) = 0.0;
    }
    const PlasticResponse r1 = plastic_return(m, state, first, PlasticState(), 20.0);
    REQUIRE(r1.yielding);
    const PlasticResponse r2 = plastic_return(m, state, second, r1.state, 35.0);
    INFO(to_string(state));
    REQUIRE(r2.yielding);
    REQUIRE(tangent_error(m, state, second, r1.state, 35.0) <= 1.0e-6);
    // The elastic branch too: a small step back inside the surface.
    const Vector6 back = second - 0.1 * (second - first);
    REQUIRE_FALSE(plastic_return(m, state, back, r2.state).yielding);
    REQUIRE(tangent_error(m, state, back, r2.state) <= 1.0e-6);
    // The tangent is symmetric (associative flow, J2).
    REQUIRE((r2.tangent - r2.tangent.transpose()).cwiseAbs().maxCoeff() <=
            1.0e-12 * r2.tangent.cwiseAbs().maxCoeff());
  }
}

TEST_CASE("plane stress returns with sigma_33 = 0 and the 3-D state at its eps_33",
          "[plasticity][material]") {
  const Material m = plastic_steel(2.0e9, 1.0e9);
  Vector6 strain = Vector6::Zero();
  strain(0) = 6.0e-3;
  strain(1) = -1.0e-3;
  strain(3) = 4.0e-3;
  const PlasticResponse r = plastic_return(m, StressState::PlaneStress, strain, PlasticState());
  REQUIRE(r.yielding);
  REQUIRE(std::abs(r.stress(2)) <= 1.0e-11 * std::abs(r.stress(0)));
  REQUIRE(r.state.thickness_strain == r.strain_33);
  Vector6 full = strain;
  full(2) = r.strain_33;
  const PlasticResponse check = plastic_return(m, StressState::ThreeDimensional, full, PlasticState());
  REQUIRE((check.stress - r.stress).cwiseAbs().maxCoeff() <= 1.0e-12 * check.stress.norm());
  // The condensed tangent has no thickness row or column.
  REQUIRE(r.tangent.row(2).cwiseAbs().maxCoeff() == 0.0);
  REQUIRE(r.tangent.col(2).cwiseAbs().maxCoeff() == 0.0);
  // Below yield it is the plane-stress elasticity matrix.
  const PlasticResponse elastic =
      plastic_return(m, StressState::PlaneStress, 0.05 * strain, PlasticState());
  REQUIRE_FALSE(elastic.yielding);
  const Matrix3 d = m.plane_stress_matrix();
  const int rows[3] = {0, 1, 3};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE(elastic.tangent(rows[i], rows[j]) == Approx(d(i, j)).margin(1e-6 * d(0, 0)));
    }
  }
}

TEST_CASE("the loading flag gives the continuum tangent at zero increment",
          "[plasticity][material]") {
  // After a plastic step the point sits on its surface. At the same strain
  // the return is elastic (Delta gamma = 0) and, the point having yielded,
  // its tangent is the continuum elastoplastic one: the limit of the
  // consistent tangent as the increment vanishes.
  const Scalar h = 2.0e9;
  const Scalar hk = 1.0e9;
  const Material m = plastic_steel(h, hk);
  Vector6 strain = Vector6::Zero();
  strain(0) = 5.0e-3;
  strain(3) = 2.0e-3;
  const PlasticResponse r1 = plastic_return(m, StressState::ThreeDimensional, strain, PlasticState());
  REQUIRE(r1.state.loading);
  const PlasticResponse again = plastic_return(m, StressState::ThreeDimensional, strain, r1.state);
  REQUIRE_FALSE(again.yielding);
  REQUIRE(again.state.loading);
  REQUIRE((again.stress - r1.stress).cwiseAbs().maxCoeff() <= 1.0e-6);
  const Scalar g = m.shear_modulus();
  Vector6 xi = r1.stress - r1.state.back_stress;
  const Scalar mean = (xi(0) + xi(1) + xi(2)) / 3.0;
  for (int i = 0; i < 3; ++i) xi(i) -= mean;
  const Scalar norm = std::sqrt(xi.head(3).squaredNorm() + 2.0 * xi.tail(3).squaredNorm());
  const Vector6 n = xi / norm;
  const Matrix6 continuum = m.three_dimensional_matrix() -
                            2.0 * g / (1.0 + (h + hk) / (3.0 * g)) * (n * n.transpose());
  REQUIRE((again.tangent - continuum).cwiseAbs().maxCoeff() <=
          1.0e-9 * continuum.cwiseAbs().maxCoeff());
  // A small further increment along the loading direction follows it.
  Vector6 d = n;
  d.tail(3) *= 2.0;  // engineering shear
  d *= 1.0e-9;
  const PlasticResponse further =
      plastic_return(m, StressState::ThreeDimensional, strain + d, r1.state);
  REQUIRE(further.yielding);
  REQUIRE((further.stress - r1.stress - continuum * d).norm() <=
          1.0e-4 * (continuum * d).norm());
  // A step elsewhere returns elastically with the elastic tangent and
  // clears the flag.
  const PlasticResponse unload =
      plastic_return(m, StressState::ThreeDimensional, 0.9 * strain, r1.state);
  REQUIRE_FALSE(unload.state.loading);
  REQUIRE((unload.tangent - m.three_dimensional_matrix()).cwiseAbs().maxCoeff() <= 1.0e-3);
}

TEST_CASE("a temperature change enters as a free strain", "[plasticity][material]") {
  Material m = plastic_steel();
  m.set_thermal(1.2e-5, 20.0, 50.0);
  // Free expansion is stress-free however large.
  Vector6 free = Vector6::Zero();
  free.head(3).setConstant(1.2e-5 * 400.0);
  const PlasticResponse r = plastic_return(m, StressState::ThreeDimensional, free, PlasticState(), 400.0);
  REQUIRE(r.stress.cwiseAbs().maxCoeff() <= 1.0e-6);
  REQUIRE_FALSE(r.yielding);
  // A bar held axially: sigma = -E alpha dT until it yields, then -sigma_y
  // (no hardening).
  const Scalar dt_elastic = 50.0;
  REQUIRE(uniaxial(m, 0.0, PlasticState(), dt_elastic).stress(0) ==
          Approx(-200.0e9 * 1.2e-5 * dt_elastic).epsilon(1e-12));
  const PlasticResponse hot = uniaxial(m, 0.0, PlasticState(), 150.0);
  REQUIRE(hot.yielding);
  REQUIRE(hot.stress(0) == Approx(-250.0e6).epsilon(1e-12));
  // Cooling back to the reference leaves a tensile residual stress,
  // E alpha dT - sigma_y, below yield (E alpha dT < 2 sigma_y).
  const PlasticResponse cold = uniaxial(m, 0.0, hot.state, 0.0);
  REQUIRE_FALSE(cold.yielding);
  REQUIRE(cold.stress(0) == Approx(200.0e9 * 1.2e-5 * 150.0 - 250.0e6).epsilon(1e-10));
  // Heated further instead (E alpha dT > 2 sigma_y), it yields back to +sigma_y.
  const PlasticResponse hotter = uniaxial(m, 0.0, PlasticState(), 300.0);
  const PlasticResponse cooled = uniaxial(m, 0.0, hotter.state, 0.0);
  REQUIRE(cooled.yielding);
  REQUIRE(cooled.stress(0) == Approx(250.0e6).epsilon(1e-12));
}

namespace {

/// A distorted Hex8 model (the cells of `type`) with a plastic material.
FemModel one_element(ElementType type, StressState state, const Material& m) {
  StructuredMeshSpec spec;
  spec.nx = spec.ny = spec.nz = 2;
  spec.lx = 1.0;
  spec.ly = 0.8;
  spec.lz = 0.6;
  Mesh mesh = make_perturbed_hex_mesh(spec, 0.2);
  REQUIRE(mesh.element_type() == type);
  return FemModel(std::move(mesh), m, 1.0, state, IntegrationOptions());
}

Vector element_displacement(const FemModel& model, Index e, Scalar amplitude, unsigned seed) {
  const int dim = model.dim();
  const Matrix x0 = model.mesh().element_coordinates(e);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  // A homogeneous stretch and shear plus a random part.
  Vector ue(dim * x0.cols());
  for (Eigen::Index a = 0; a < x0.cols(); ++a) {
    for (int k = 0; k < dim; ++k) {
      const Scalar homogeneous = k == 0 ? 0.8 * x0(0, a) + 0.3 * x0(1, a) : -0.2 * x0(k, a);
      ue(dim * a + k) = amplitude * (homogeneous + 0.5 * dist(gen));
    }
  }
  return ue;
}

}  // namespace

TEST_CASE("the elastoplastic element tangent is the derivative of its internal force",
          "[plasticity][element]") {
  const Material m = plastic_steel(1.0e9, 2.0e9, 50.0e6, 40.0);
  struct Case {
    ElementType type;
    StressState state;
  };
  const std::vector<Case> cases = {{ElementType::Hex8, StressState::ThreeDimensional}};
  for (const Case& c : cases) {
    FemModel model = one_element(c.type, c.state, m);
    for (const bool bbar : {false, true}) {
      const Index e = 0;
      const std::vector<PlasticState> virgin(
          static_cast<std::size_t>(elastoplastic_points(model)));
      // Load into the plastic range, commit, then load on differently.
      const Vector u1 = element_displacement(model, e, 6.0e-3, 7u);
      const ElastoplasticElement first =
          elastoplastic_element(model, e, u1, virgin, bbar, nullptr, 0.0, false);
      REQUIRE(first.yielding_points > 0);
      const Vector u2 = u1 + element_displacement(model, e, 3.0e-3, 9u);
      const ElastoplasticElement base =
          elastoplastic_element(model, e, u2, first.states, bbar, nullptr, 0.0, true);
      const Scalar h = 1.0e-8;
      Scalar worst = 0.0;
      for (Eigen::Index j = 0; j < u2.size(); ++j) {
        Vector up = u2;
        Vector um = u2;
        up(j) += h;
        um(j) -= h;
        const Vector fd =
            (elastoplastic_element(model, e, up, first.states, bbar, nullptr, 0.0, false)
                 .internal_force -
             elastoplastic_element(model, e, um, first.states, bbar, nullptr, 0.0, false)
                 .internal_force) /
            (2.0 * h);
        worst = std::max(worst, (fd - base.tangent.col(j)).cwiseAbs().maxCoeff());
      }
      INFO(to_string(c.type) << " " << to_string(c.state) << (bbar ? " B-bar" : ""));
      REQUIRE(base.yielding_points > 0);
      REQUIRE(worst <= 1.0e-6 * base.tangent.cwiseAbs().maxCoeff());
    }
  }
}

TEST_CASE("the element's thermal load rate is the derivative of its internal force",
          "[plasticity][element]") {
  Material m = plastic_steel(2.0e9);
  m.set_thermal(1.2e-5, 20.0, 50.0);
  for (const ElementType type : {ElementType::Hex8}) {
    for (const StressState state : {StressState::ThreeDimensional}) {
      FemModel model = one_element(type, state, m);
      Vector temperature(model.mesh().num_nodes());
      for (Index node = 0; node < model.mesh().num_nodes(); ++node) {
        temperature(node) = 20.0 + 300.0 * model.mesh().node(node).x() + 100.0;
      }
      const std::vector<PlasticState> virgin(
          static_cast<std::size_t>(elastoplastic_points(model)));
      const Vector ue = element_displacement(model, 0, 2.0e-3, 3u);
      const Scalar lambda = 0.8;
      // With finite kinematics the change of the Green thermal strain,
      // alpha dT (1 + alpha dT / 2), is the rate.
      for (const Kinematics kinematics : {Kinematics::SmallStrain, Kinematics::Finite}) {
        for (const bool bbar : {false, true}) {
          const ElastoplasticElement base = elastoplastic_element(
              model, 0, ue, virgin, bbar, &temperature, lambda, false, kinematics);
          REQUIRE(base.yielding_points > 0);
          const Scalar h = 1.0e-7;
          const Vector fd = (elastoplastic_element(model, 0, ue, virgin, bbar, &temperature,
                                                   lambda + h, false, kinematics)
                                 .internal_force -
                             elastoplastic_element(model, 0, ue, virgin, bbar, &temperature,
                                                   lambda - h, false, kinematics)
                                 .internal_force) /
                            (2.0 * h);
          INFO(to_string(type) << " " << to_string(state) << " " << to_string(kinematics)
                               << (bbar ? " mean dilatation" : ""));
          REQUIRE((fd - base.thermal_force_rate).cwiseAbs().maxCoeff() <=
                  1.0e-6 * base.thermal_force_rate.cwiseAbs().maxCoeff());
        }
      }
    }
  }
}

TEST_CASE("an elastic element is the linear element, and mean dilatation leaves a constant "
          "dilatation alone",
          "[plasticity][element]") {
  const Material elastic(200.0e9, 0.3, 7800.0, "elastic");
  for (const ElementType type : {ElementType::Hex8}) {
    const StressState state = StressState::ThreeDimensional;
    FemModel model = one_element(type, state, elastic);
    const std::vector<PlasticState> virgin(static_cast<std::size_t>(elastoplastic_points(model)));
    const Vector ue = element_displacement(model, 0, 1.0e-3, 5u);
    const ElastoplasticElement r =
        elastoplastic_element(model, 0, ue, virgin, false, nullptr, 0.0, true);
    const Matrix k = model.element().stiffness(model.mesh().element_coordinates(0),
                                               model.constitutive(),
                                               model.dim() == 2 ? model.thickness() : 1.0,
                                               model.integration());
    INFO(to_string(type));
    REQUIRE((r.tangent - k).cwiseAbs().maxCoeff() <= 1.0e-12 * k.cwiseAbs().maxCoeff());
    REQUIRE((r.internal_force - k * ue).cwiseAbs().maxCoeff() <=
            1.0e-12 * (k.cwiseAbs() * ue.cwiseAbs()).maxCoeff());
    REQUIRE(r.energy == Approx(0.5 * ue.dot(k * ue)).epsilon(1e-12));
    // A homogeneous strain has a constant dilatation, which B-bar keeps.
    const Matrix x0 = model.mesh().element_coordinates(0);
    Vector homogeneous(ue.size());
    for (Eigen::Index a = 0; a < x0.cols(); ++a) {
      for (int kk = 0; kk < model.dim(); ++kk) {
        homogeneous(model.dim() * a + kk) = 1.0e-3 * (kk == 0 ? x0(0, a) - 0.4 * x0(1, a)
                                                              : 0.7 * x0(kk, a));
      }
    }
    const Vector f_std =
        elastoplastic_element(model, 0, homogeneous, virgin, false, nullptr, 0.0, false)
            .internal_force;
    const Vector f_bar =
        elastoplastic_element(model, 0, homogeneous, virgin, true, nullptr, 0.0, false)
            .internal_force;
    REQUIRE((f_std - f_bar).cwiseAbs().maxCoeff() <= 1.0e-10 * f_std.cwiseAbs().maxCoeff());
  }
}

TEST_CASE("the finite-kinematics elastoplastic tangent is the derivative of its internal force",
          "[plasticity][element]") {
  // Green-Lagrange strain and second Piola-Kirchhoff stress, with the
  // geometric stiffness and (averaged) the Green-strain mean dilatation's
  // extra term, at a large displacement with a rigid turn.
  const Material m = plastic_steel(1.0e9, 2.0e9, 50.0e6, 40.0);
  struct Case {
    ElementType type;
    StressState state;
  };
  const std::vector<Case> cases = {{ElementType::Hex8, StressState::ThreeDimensional}};
  for (const Case& c : cases) {
    FemModel model = one_element(c.type, c.state, m);
    const int dim = model.dim();
    const Matrix x0 = model.mesh().element_coordinates(0);
    Matrix rot = Matrix::Identity(dim, dim);
    rot(0, 0) = rot(1, 1) = std::cos(0.5);
    rot(0, 1) = -std::sin(0.5);
    rot(1, 0) = std::sin(0.5);
    Vector turn(dim * x0.cols());
    const Matrix rigid = (rot - Matrix::Identity(dim, dim)) * x0;
    for (Eigen::Index a = 0; a < x0.cols(); ++a) {
      for (int k = 0; k < dim; ++k) turn(dim * a + k) = rigid(k, a);
    }
    for (const bool averaged : {false, true}) {
      const std::vector<PlasticState> virgin(
          static_cast<std::size_t>(elastoplastic_points(model)));
      const Vector u1 = turn + element_displacement(model, 0, 6.0e-3, 7u);
      const ElastoplasticElement first = elastoplastic_element(
          model, 0, u1, virgin, averaged, nullptr, 0.0, false, Kinematics::Finite);
      REQUIRE(first.yielding_points > 0);
      const Vector u2 = u1 + element_displacement(model, 0, 3.0e-3, 9u);
      const ElastoplasticElement base = elastoplastic_element(
          model, 0, u2, first.states, averaged, nullptr, 0.0, true, Kinematics::Finite);
      const Scalar h = 1.0e-8;
      Scalar worst = 0.0;
      for (Eigen::Index j = 0; j < u2.size(); ++j) {
        Vector up = u2;
        Vector um = u2;
        up(j) += h;
        um(j) -= h;
        const Vector fd = (elastoplastic_element(model, 0, up, first.states, averaged, nullptr,
                                                 0.0, false, Kinematics::Finite)
                               .internal_force -
                           elastoplastic_element(model, 0, um, first.states, averaged, nullptr,
                                                 0.0, false, Kinematics::Finite)
                               .internal_force) /
                          (2.0 * h);
        worst = std::max(worst, (fd - base.tangent.col(j)).cwiseAbs().maxCoeff());
      }
      INFO(to_string(c.type) << " " << to_string(c.state) << (averaged ? " averaged" : ""));
      REQUIRE(base.yielding_points > 0);
      REQUIRE(worst <= 1.0e-6 * base.tangent.cwiseAbs().maxCoeff());
    }
  }
}

TEST_CASE("finite-kinematics plasticity is objective and tends to the linear stiffness",
          "[plasticity][element]") {
  const Material m = plastic_steel(2.0e9, 1.0e9);
  for (const ElementType type : {ElementType::Hex8}) {
    const StressState state = StressState::ThreeDimensional;
    FemModel model = one_element(type, state, m);
    const int dim = model.dim();
    const Matrix x0 = model.mesh().element_coordinates(0);
    const std::vector<PlasticState> virgin(static_cast<std::size_t>(elastoplastic_points(model)));
    const Vector u = element_displacement(model, 0, 5.0e-3, 3u);
    // x -> R x: the same Green strain, so the same state and energy.
    Matrix rot = Matrix::Identity(dim, dim);
    rot(0, 0) = rot(1, 1) = 0.0;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    Vector rotated(u.size());
    for (Eigen::Index a = 0; a < x0.cols(); ++a) {
      const Vector x = x0.col(a) + u.segment(dim * a, dim);
      rotated.segment(dim * a, dim) = rot * x - x0.col(a);
    }
    for (const bool averaged : {false, true}) {
      const ElastoplasticElement r1 =
          elastoplastic_element(model, 0, u, virgin, averaged, nullptr, 0.0, false,
                                Kinematics::Finite);
      const ElastoplasticElement r2 =
          elastoplastic_element(model, 0, rotated, virgin, averaged, nullptr, 0.0, false,
                                Kinematics::Finite);
      INFO(to_string(type) << (averaged ? " averaged" : ""));
      REQUIRE(r1.yielding_points > 0);
      REQUIRE(r2.yielding_points == r1.yielding_points);
      REQUIRE(r2.energy == Approx(r1.energy).epsilon(1e-10));
      for (std::size_t q = 0; q < r1.states.size(); ++q) {
        REQUIRE(r2.states[q].equivalent_plastic_strain ==
                Approx(r1.states[q].equivalent_plastic_strain).margin(1e-14));
        REQUIRE((r2.states[q].back_stress - r1.states[q].back_stress).norm() <=
                1.0e-8 * std::max(1.0, r1.states[q].back_stress.norm()));
      }
      // The internal force turns with the body: f_rotated = R f per node.
      for (Eigen::Index a = 0; a < x0.cols(); ++a) {
        const Vector expected = rot * r1.internal_force.segment(dim * a, dim);
        REQUIRE((r2.internal_force.segment(dim * a, dim) - expected).norm() <=
                1.0e-9 * r1.internal_force.cwiseAbs().maxCoeff());
      }
    }
    // At zero displacement, with the element's own operator, the tangent is
    // the linear stiffness.
    const ElastoplasticElement zero =
        elastoplastic_element(model, 0, Vector::Zero(u.size()), virgin, false, nullptr, 0.0,
                              true, Kinematics::Finite);
    const Matrix k = model.element().stiffness(x0, model.constitutive(),
                                               dim == 2 ? model.thickness() : 1.0,
                                               model.integration());
    REQUIRE((zero.tangent - k).cwiseAbs().maxCoeff() <= 1.0e-12 * k.cwiseAbs().maxCoeff());
  }
}

TEST_CASE("the plasticity block of a deck parses and validates", "[plasticity][config]") {
  const auto deck = [](const std::string& plasticity) {
    return json::parse(R"({
      "mesh": { "type": "structured_hex", "nx": 20, "ny": 2, "nz": 1, "lx": 1.0, "ly": 0.1,
                "lz": 0.01 },
      "material": { "youngs_modulus": 200e9, "poisson_ratio": 0.3,
                    "plasticity": )" + plasticity + R"( },
      "boundary_conditions": [ { "fix": ["x", "y", "z"], "region": { "box": { "xmax": 0.0 } } } ]
    })");
  };
  const std::string steel = R"({ "yield_stress": 250e6, "hardening_modulus": 2e9,
      "kinematic_hardening_modulus": 1e9, "saturation_stress": 80e6, "saturation_rate": 20 })";
  const Configuration config = parse_configuration(deck(steel), "inline", true);
  const PlasticityParameters& p = config.material().plasticity();
  REQUIRE(p.yield_stress == 250.0e6);
  REQUIRE(p.hardening_modulus == 2.0e9);
  REQUIRE(p.kinematic_hardening_modulus == 1.0e9);
  REQUIRE(p.saturation_stress == 80.0e6);
  REQUIRE(p.saturation_rate == 20.0);
  REQUIRE(build_model(config).material().plasticity().enabled());

  const auto refuses = [&](const std::string& plasticity) {
    REQUIRE_THROWS_AS(parse_configuration(deck(plasticity), "inline", true), ConfigError);
  };
  refuses(R"({ "yield_stress": -1.0 })");
  refuses(R"({ "hardening_modulus": 1e9 })");  // no yield stress
  refuses(R"({ "yield_stress": 250e6, "saturation_stress": 50e6 })");  // no rate
  refuses(R"({ "yield_stress": 250e6, "hardening": 1e9 })");  // unknown key
}
