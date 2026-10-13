/// \file test_explicit.cpp
/// \brief Explicit dynamics (ExplicitDynamics.hpp) and the explicit forming
///        step (`form_explicit`, Forming.hpp).
///
/// The central-difference method on a linear model decouples in the
/// M-orthonormal modes of the lumped mass, so a run is checked step for step
/// against the scalar recursion of every mode, and its stability limit
/// against the largest eigenvalue of a dense eigensolve (the element
/// eigenvalue bound must lie above it, power iteration on it). Mass scaling
/// is checked against its definition, the dedicated Hex8 kernel against the
/// generic element dispatch it replaces (small and finite kinematics,
/// elastic under small strain, J2, Hill48 and Chaboche, mean dilatation,
/// uniform and distorted meshes), the energy balance by its order in the
/// step and by the sign of the plastic dissipation, determinism by comparing
/// thread counts bit for bit, and the step loop by counting its heap
/// allocations. The explicit forming step hands its state to an implicit
/// release exactly, and the `explicit` block of a deck is read strictly.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"
#include "sparlab/fem/ExplicitDynamics.hpp"
#include "sparlab/fem/Forming.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/FormingWriter.hpp"
#include "sparlab/io/Json.hpp"
#include "sparlab/io/ResultWriter.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <vector>

#ifdef SPARLAB_HAVE_OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Heap allocation counting (the step loop must allocate nothing). The
// replacement operators are those of the whole test executable; they count
// only while `g_counting` is set.
// ---------------------------------------------------------------------------
namespace {
std::atomic<bool> g_counting{false};
std::atomic<long> g_allocations{0};
void* counted_allocation(std::size_t size) {
  if (g_counting.load(std::memory_order_relaxed)) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  }
  void* p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* counted_aligned(std::size_t size, std::size_t alignment) {
  if (g_counting.load(std::memory_order_relaxed)) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  }
  const std::size_t rounded = (std::max<std::size_t>(size, 1) + alignment - 1) / alignment *
                              alignment;
  void* p = std::aligned_alloc(alignment, rounded);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
}  // namespace

void* operator new(std::size_t size) { return counted_allocation(size); }
void* operator new[](std::size_t size) { return counted_allocation(size); }
void* operator new(std::size_t size, std::align_val_t a) {
  return counted_aligned(size, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t size, std::align_val_t a) {
  return counted_aligned(size, static_cast<std::size_t>(a));
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Scalar kInf = std::numeric_limits<Scalar>::infinity();

SelectorGroup box(Scalar xmin, Scalar xmax, Scalar ymin = -kInf, Scalar ymax = kInf,
                  Scalar zmin = -kInf, Scalar zmax = kInf, const std::string& name = "box") {
  SelectorGroup g;
  g.name = name;
  Selector s;
  s.kind = SelectorKind::Box;
  s.xmin = xmin;
  s.xmax = xmax;
  s.ymin = ymin;
  s.ymax = ymax;
  s.zmin = zmin;
  s.zmax = zmax;
  g.members.push_back(s);
  return g;
}

DisplacementConstraint clamp(const SelectorGroup& region, std::vector<int> components = {0, 1, 2},
                             const Vector3& value = Vector3::Zero()) {
  DisplacementConstraint c;
  c.region = region;
  for (int k : components) c.set(k, true, value(k));
  return c;
}

Mesh hex_block(Index nx, Index ny, Index nz, Scalar lx, Scalar ly, Scalar lz, Scalar x0 = 0.0,
               Scalar y0 = 0.0, Scalar z0 = 0.0) {
  StructuredMeshSpec spec;
  spec.nx = nx;
  spec.ny = ny;
  spec.nz = nz;
  spec.lx = lx;
  spec.ly = ly;
  spec.lz = lz;
  spec.x0 = x0;
  spec.y0 = y0;
  spec.z0 = z0;
  return make_structured_hex_mesh(spec);
}

/// The same mesh with every node moved by up to `amount` times the smallest
/// cell size (deterministically): no two cells alike.
Mesh distorted(const Mesh& mesh, Scalar amount) {
  Matrix x = mesh.coordinates();
  Scalar measure = 0.0;
  for (Index e = 0; e < mesh.num_elements(); ++e) measure += mesh.element_measure(e);
  const Scalar h = std::cbrt(measure / static_cast<Scalar>(mesh.num_elements()));
  std::mt19937 rng(7);
  std::uniform_real_distribution<Scalar> d(-1.0, 1.0);
  for (Index j = 0; j < x.cols(); ++j) {
    for (Index k = 0; k < x.rows(); ++k) x(k, j) += amount * h * d(rng);
  }
  return Mesh(x, mesh.connectivity(), mesh.element_type());
}

/// A finalised model with one load case that applies nothing (or `load`).
FemModel finalised(Mesh mesh, const IsotropicMaterial& m,
                   const std::vector<DisplacementConstraint>& bcs, const LoadCaseSpec* load = nullptr) {
  FemModel model(std::move(mesh), m, 1.0, StressState::ThreeDimensional, IntegrationOptions());
  model.constraints() = bcs;
  if (load != nullptr) {
    model.load_case_specs().push_back(*load);
  } else {
    LoadCaseSpec lc;
    lc.name = "none";
    model.load_case_specs().push_back(lc);
  }
  model.finalize();
  return model;
}

IsotropicMaterial j2_material(Scalar kinematic = 0.0, Scalar saturation = 0.0) {
  IsotropicMaterial m(70.0e9, 0.33, 2700.0, "aluminium");
  PlasticityParameters p;
  p.yield_stress = 150.0e6;
  p.hardening_modulus = 500.0e6;
  p.kinematic_hardening_modulus = kinematic;
  p.saturation_stress = saturation;
  p.saturation_rate = saturation > 0.0 ? 20.0 : 0.0;
  m.set_plasticity(p);
  return m;
}

/// The drive of a run without tools whose model constraints hold their
/// values.
ExplicitDrive held_drive(const FemModel& model, Scalar duration) {
  ExplicitDrive drive;
  drive.duration = duration;
  drive.fixed = model.dofs().constrained_dofs();
  drive.fixed_start.resize(static_cast<Eigen::Index>(drive.fixed.size()));
  for (std::size_t i = 0; i < drive.fixed.size(); ++i) {
    drive.fixed_start(static_cast<Eigen::Index>(i)) =
        model.dofs().prescribed_value(drive.fixed[i]);
  }
  drive.fixed_end = drive.fixed_start;
  return drive;
}

NonlinearOptions small_strain() {
  NonlinearOptions nl;
  nl.kinematics = Kinematics::SmallStrain;
  return nl;
}

/// The explicit options of a test run: a fixed step, no energy stop.
ExplicitOptions fixed_step(Scalar dt) {
  ExplicitOptions o;
  o.time_step = dt;
  o.history_every = 1000000;
  o.energy_limit = 1.0e300;
  o.energy_tolerance = 1.0e300;
  return o;
}

/// A linear elastic cantilever block clamped at x = 0 (lumped mass).
FemModel cantilever_block(Index nx = 4) {
  return finalised(hex_block(nx, 1, 1, 0.01 * static_cast<Scalar>(nx), 0.01, 0.013),
                   default_material(), {clamp(box(-kInf, 0.0, -kInf, kInf, -kInf, kInf, "root"))});
}

struct Modes {
  Vector omega2;
  Matrix phi;      ///< free DOFs x modes, M-orthonormal
  Matrix m;        ///< free-free lumped mass (diagonal)
};

Modes lumped_modes(const Assembler& assembler) {
  const Matrix k(assembler.reduce_free_free(assembler.assemble_stiffness()));
  const Matrix m(assembler.reduce_free_free(assembler.assemble_mass(MassType::Lumped)));
  Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> es(k, m);
  REQUIRE(es.info() == Eigen::Success);
  return {es.eigenvalues(), es.eigenvectors(), m};
}

Vector scatter_free(const FemModel& model, const Vector& free) {
  Vector full = Vector::Zero(model.dofs().num_dofs());
  const std::vector<Index>& f = model.dofs().free_dofs();
  for (std::size_t i = 0; i < f.size(); ++i) full(f[i]) = free(static_cast<Eigen::Index>(i));
  return full;
}

Vector gather_free(const FemModel& model, const Vector& full) {
  const std::vector<Index>& f = model.dofs().free_dofs();
  Vector out(static_cast<Eigen::Index>(f.size()));
  for (std::size_t i = 0; i < f.size(); ++i) out(static_cast<Eigen::Index>(i)) = full(f[i]);
  return out;
}

Vector random_vector(Eigen::Index n, unsigned seed, Scalar scale) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Scalar> d(-1.0, 1.0);
  Vector v(n);
  for (Eigen::Index i = 0; i < n; ++i) v(i) = scale * d(rng);
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The central-difference method on a linear model
// ---------------------------------------------------------------------------
TEST_CASE("an explicit run is the central-difference recursion of every lumped-mass mode",
          "[explicit]") {
  const FemModel model = cantilever_block();
  const Assembler assembler(model);
  const Modes modes = lumped_modes(assembler);
  const Scalar omega_max = std::sqrt(modes.omega2.maxCoeff());
  const Scalar dt = 0.7 * 2.0 / omega_max;
  const int steps = 250;

  // A start state with every mode in it.
  const Eigen::Index nf = modes.phi.rows();
  const Vector u0 = random_vector(nf, 1, 1.0e-6);
  const Vector v0 = random_vector(nf, 2, 1.0e-2);
  ExplicitState start;
  start.displacement = scatter_free(model, u0);
  start.velocity = scatter_free(model, v0);

  ExplicitDynamics dyn(model, assembler, small_strain(), {}, fixed_step(dt));
  REQUIRE(dyn.dedicated());
  const ExplicitResult r = dyn.run(held_drive(model, steps * dt), start);
  REQUIRE(r.completed);
  REQUIRE(r.steps == steps);

  // Every mode by the scalar recursion with half-step velocities.
  const Vector q0 = modes.phi.transpose() * (modes.m * u0);
  const Vector p0 = modes.phi.transpose() * (modes.m * v0);
  Vector q(nf), p(nf);
  for (Eigen::Index i = 0; i < nf; ++i) {
    const Scalar w2 = modes.omega2(i);
    Scalar qi = q0(i), vi = p0(i), ai = -w2 * qi;
    for (int n = 0; n < steps; ++n) {
      const Scalar vh = vi + 0.5 * dt * ai;
      qi += dt * vh;
      ai = -w2 * qi;
      vi = vh + 0.5 * dt * ai;
    }
    q(i) = qi;
    p(i) = vi;
  }
  const Vector u_ref = modes.phi * q;
  const Vector v_ref = modes.phi * p;
  const Vector u = gather_free(model, r.final_state.displacement);
  const Vector v = gather_free(model, r.final_state.velocity);
  const Scalar du = (u - u_ref).cwiseAbs().maxCoeff() / u_ref.cwiseAbs().maxCoeff();
  const Scalar dv = (v - v_ref).cwiseAbs().maxCoeff() / v_ref.cwiseAbs().maxCoeff();
  INFO("displacement " << du << ", velocity " << dv);
  REQUIRE(du < 1.0e-10);
  REQUIRE(dv < 1.0e-10);
}

TEST_CASE("the central-difference step is stable below 2/omega_max and unstable above it, "
          "with the element bound above omega_max and power iteration on it",
          "[explicit]") {
  const FemModel model = cantilever_block(3);
  const Assembler assembler(model);
  const Modes modes = lumped_modes(assembler);
  const Scalar omega_max = std::sqrt(modes.omega2.maxCoeff());
  const Scalar limit = 2.0 / omega_max;

  // The top mode, with a little of every other.
  const Eigen::Index nf = modes.phi.rows();
  const Vector u0 = modes.phi.col(nf - 1) * 1.0e-7 / modes.phi.col(nf - 1).cwiseAbs().maxCoeff() +
                    random_vector(nf, 3, 1.0e-10);
  ExplicitState start;
  start.displacement = scatter_free(model, u0);
  const int steps = 3000;
  {
    ExplicitDynamics dyn(model, assembler, small_strain(), {}, fixed_step(0.99 * limit));
    const ExplicitResult r = dyn.run(held_drive(model, steps * 0.99 * limit), start);
    REQUIRE(r.completed);
    const Scalar growth = r.final_state.displacement.cwiseAbs().maxCoeff() / 1.0e-7;
    INFO("growth " << growth);
    REQUIRE(growth < 20.0);  // bounded: 1 / sqrt(1 - (omega dt / 2)^2) = 7.1
  }
  {
    ExplicitOptions o = fixed_step(1.01 * limit);
    o.energy_limit = 0.5;
    o.energy_tolerance = 0.05;
    o.history_every = 10;
    ExplicitDynamics dyn(model, assembler, small_strain(), {}, o);
    const ExplicitResult r = dyn.run(held_drive(model, steps * 1.01 * limit), start);
    REQUIRE_FALSE(r.completed);
    REQUIRE(r.warnings.size() >= 1);  // the fixed step beyond the limit
    REQUIRE_THAT(r.warnings.front(), ContainsSubstring("exceeds the stability limit"));
    // The state kept is the last valid one.
    REQUIRE(r.final_state.displacement.allFinite());
  }
  // The element eigenvalue bound (Irons) lies above omega_max; power
  // iteration (its Rayleigh quotient, without the 1.05 margin) takes the
  // model without its constraints, whose omega_max bounds the constrained
  // one's (interlacing) - the drive of a forming step prescribes other DOFs
  // than the model's.
  ExplicitOptions o;
  o.stable_step.power_iterations = 4000;
  ExplicitDynamics dyn(model, assembler, small_strain(), {}, o);
  const Vector zero = Vector::Zero(model.dofs().num_dofs());
  const Scalar element = dyn.element_time_steps(zero).minCoeff();
  const Scalar power = dyn.power_iteration_time_step(zero);
  const Scalar omega_power = 2.0 / power / std::sqrt(1.05);
  const Matrix k_all(assembler.assemble_stiffness());
  const Matrix m_all(assembler.assemble_mass(MassType::Lumped));
  const Scalar omega_all = std::sqrt(
      Eigen::GeneralizedSelfAdjointEigenSolver<Matrix>(k_all, m_all, Eigen::EigenvaluesOnly)
          .eigenvalues()
          .maxCoeff());
  INFO("element bound " << 2.0 / element / omega_max << " omega_max, power "
                        << omega_power / omega_all << " of the unconstrained omega_max, "
                        << omega_all / omega_max << " omega_max");
  REQUIRE(2.0 / element >= omega_max * (1.0 - 1.0e-12));
  REQUIRE(omega_all >= omega_max);
  REQUIRE(std::abs(omega_power / omega_all - 1.0) < 1.0e-3);
}

TEST_CASE("selective mass scaling reaches its target step with the added mass it reports, and "
          "uniform scaling divides the frequencies by sqrt(s)",
          "[explicit]") {
  // A graded, distorted mesh: every element its own stable step.
  Mesh graded = hex_block(6, 2, 2, 0.03, 0.01, 0.01);
  Matrix x = graded.coordinates();
  for (Index j = 0; j < x.cols(); ++j) x(0, j) = 0.03 * std::pow(x(0, j) / 0.03, 2.0);
  const FemModel model =
      finalised(distorted(Mesh(x, graded.connectivity(), ElementType::Hex8), 0.05),
                default_material(), {clamp(box(-kInf, 1.0e-9))});
  const Assembler assembler(model);
  const Vector zero = Vector::Zero(model.dofs().num_dofs());
  ExplicitOptions probe;
  const Vector crit = ExplicitDynamics(model, assembler, small_strain(), {}, probe)
                          .element_time_steps(zero);
  REQUIRE(crit.maxCoeff() > 2.0 * crit.minCoeff());

  const Scalar safety = 0.9;
  const Scalar target = 1.6 * safety * crit.minCoeff();
  ExplicitOptions o;
  o.stable_step.safety = safety;
  o.mass_scaling.mode = MassScalingOptions::Mode::Selective;
  o.mass_scaling.target_time_step = target;
  o.history_every = 1000000;
  ExplicitDynamics dyn(model, assembler, small_strain(), {}, o);
  const ExplicitResult r = dyn.run(held_drive(model, 20.0 * target), ExplicitState{
                                                                          zero, {}, {}, {}});
  REQUIRE(r.completed);
  REQUIRE(r.time_step <= target);
  REQUIRE(r.time_step >= target * (1.0 - 1.0e-12));
  // The added mass: sum (s_e - 1) m_e / sum m_e, s_e = max(1, (target / (safety dt_e))^2).
  Scalar added = 0.0, total = 0.0;
  int scaled = 0;
  Vector scale(model.mesh().num_elements());
  for (Index e = 0; e < model.mesh().num_elements(); ++e) {
    const Scalar s = std::max(1.0, std::pow(target / (safety * crit(e)), 2));
    const Scalar m = model.mesh().element_measure(e) * 2700.0;
    scale(e) = s;
    added += (s - 1.0) * m;
    total += m;
    if (s > 1.0) ++scaled;
  }
  REQUIRE(scaled > 0);
  REQUIRE(scaled < model.mesh().num_elements());
  REQUIRE(r.scaled_elements == scaled);
  REQUIRE(r.added_mass_fraction == Approx(added / total).epsilon(1.0e-10));
  REQUIRE(r.physical_mass == Approx(total).epsilon(1.0e-10));
  // Every scaled element's step is the target's (over the safety factor).
  const Vector scaled_steps = dyn.element_time_steps(zero, &scale);
  REQUIRE(scaled_steps.minCoeff() == Approx(target / safety).epsilon(1.0e-10));

  // Uniform: one factor, omega_max / sqrt(s) by the dense eigensolve.
  ExplicitOptions ou = o;
  ou.mass_scaling.mode = MassScalingOptions::Mode::Uniform;
  ExplicitDynamics uniform(model, assembler, small_strain(), {}, ou);
  const ExplicitResult ru = uniform.run(held_drive(model, 20.0 * target), ExplicitState{
                                                                             zero, {}, {}, {}});
  REQUIRE(ru.completed);
  const Scalar s = ru.max_mass_scale;
  REQUIRE(s > 1.0);
  REQUIRE(ru.added_mass_fraction == Approx(s - 1.0).epsilon(1.0e-10));
  REQUIRE(ru.scaled_stable_time_step == Approx(ru.stable_time_step * std::sqrt(s)).epsilon(1e-12));
  REQUIRE(ru.time_step == Approx(target).epsilon(1.0e-12));
  const Vector scale_u = Vector::Constant(model.mesh().num_elements(), s);
  const Matrix k(assembler.reduce_free_free(assembler.assemble_stiffness()));
  const Matrix m0(assembler.reduce_free_free(assembler.assemble_mass(MassType::Lumped)));
  const Matrix m1(assembler.reduce_free_free(assembler.assemble_mass(MassType::Lumped, &scale_u)));
  const Scalar w0 = Eigen::GeneralizedSelfAdjointEigenSolver<Matrix>(k, m0, Eigen::EigenvaluesOnly)
                        .eigenvalues()
                        .maxCoeff();
  const Scalar w1 = Eigen::GeneralizedSelfAdjointEigenSolver<Matrix>(k, m1, Eigen::EigenvaluesOnly)
                        .eigenvalues()
                        .maxCoeff();
  REQUIRE(std::sqrt(w1) == Approx(std::sqrt(w0 / s)).epsilon(1.0e-10));
}

TEST_CASE("dynamic selective mass scaling holds the target step as elements thin, and books "
          "the added mass's energy",
          "[explicit]") {
  // A block crushed to 70 % of its height (finite kinematics, elastic in
  // effect): its elements thin and stiffen, and their stable steps fall.
  const FemModel model = finalised(
      hex_block(2, 2, 2, 0.01, 0.01, 0.01), never_yielding_material(),
      {clamp(box(-kInf, kInf, -kInf, kInf, -kInf, 1.0e-9)),
       clamp(box(-kInf, kInf, -kInf, kInf, 0.01 - 1.0e-9, kInf), {2}, Vector3(0, 0, -0.003))});
  const Assembler assembler(model);
  const Vector zero = Vector::Zero(model.dofs().num_dofs());
  NonlinearOptions nl;  // finite
  nl.mean_dilatation = MeanDilatation::None;  // an elastic block's elements
  ExplicitOptions probe;
  const Scalar crit = ExplicitDynamics(model, assembler, nl, {}, probe)
                          .element_time_steps(zero)
                          .minCoeff();
  ExplicitOptions o;
  o.mass_scaling.mode = MassScalingOptions::Mode::Selective;
  o.mass_scaling.target_time_step = 1.2 * 0.9 * crit;
  o.stable_step.update_every = 20;
  o.history_every = 20;
  const Scalar target = o.mass_scaling.target_time_step;
  const auto run = [&](bool dynamic) {
    ExplicitOptions oo = o;
    oo.mass_scaling.dynamic = dynamic;
    ExplicitDynamics dyn(model, assembler, nl, {}, oo);
    ExplicitDrive drive = held_drive(model, 600.0 * target);
    drive.fixed_start.setZero();
    return dyn.run(drive, ExplicitState{zero, {}, {}, {}});
  };
  const ExplicitResult fixed = run(false);
  const ExplicitResult dynamic = run(true);
  REQUIRE(fixed.completed);
  REQUIRE(dynamic.completed);
  INFO("fixed: smallest step " << fixed.min_time_step / target << " of the target; dynamic: "
                               << dynamic.mass_updates << " mass updates, added mass "
                               << fixed.added_mass_fraction << " -> "
                               << dynamic.added_mass_fraction << ", energy error "
                               << dynamic.max_energy_error);
  // Without it the step falls below the target; with it the step stays at
  // the target and the added mass grows.
  REQUIRE(fixed.mass_updates == 0);
  REQUIRE(fixed.min_time_step < 0.95 * target);  // 0.933 of it
  REQUIRE(dynamic.mass_updates > 0);
  REQUIRE(dynamic.min_time_step >= target * (1.0 - 1.0e-12));
  REQUIRE(dynamic.steps < fixed.steps);
  REQUIRE(dynamic.added_mass_fraction > fixed.added_mass_fraction);
  REQUIRE(dynamic.max_mass_scale > fixed.max_mass_scale);
  // The kinetic energy of the added mass is in the balance, which closes.
  REQUIRE(dynamic.records.back().mass_scaling > 0.0);
  REQUIRE(dynamic.max_energy_error < 1.0e-2);
  // The dynamic option belongs to selective scaling.
  ExplicitOptions wrong = o;
  wrong.mass_scaling.mode = MassScalingOptions::Mode::Uniform;
  wrong.mass_scaling.dynamic = true;
  REQUIRE_THROWS_AS(wrong.validate(), ConfigError);
}

TEST_CASE("a stopped run reports the mass of the state it keeps, not the mass added after it",
          "[explicit]") {
  // The crush of the dynamic scaling case, with no record before the end
  // and a balance limit no run meets: the run stops at its last step and
  // keeps its start state - whose mass is the start's, whatever dynamic
  // scaling added on the way.
  const FemModel model = finalised(
      hex_block(2, 2, 2, 0.01, 0.01, 0.01), never_yielding_material(),
      {clamp(box(-kInf, kInf, -kInf, kInf, -kInf, 1.0e-9)),
       clamp(box(-kInf, kInf, -kInf, kInf, 0.01 - 1.0e-9, kInf), {2}, Vector3(0, 0, -0.003))});
  const Assembler assembler(model);
  const Index n = model.dofs().num_dofs();
  const Vector zero = Vector::Zero(n);
  NonlinearOptions nl;
  nl.mean_dilatation = MeanDilatation::None;  // an elastic block's elements
  ExplicitOptions probe;
  const Scalar crit =
      ExplicitDynamics(model, assembler, nl, {}, probe).element_time_steps(zero).minCoeff();
  ExplicitOptions o;
  o.mass_scaling.mode = MassScalingOptions::Mode::Selective;
  o.mass_scaling.target_time_step = 1.2 * 0.9 * crit;
  o.mass_scaling.dynamic = true;
  o.stable_step.update_every = 20;
  o.history_every = 1000000;
  ExplicitDrive drive = held_drive(model, 600.0 * o.mass_scaling.target_time_step);
  drive.fixed_start.setZero();
  const ExplicitResult full =
      ExplicitDynamics(model, assembler, nl, {}, o).run(drive, {zero, {}, {}, {}});
  REQUIRE(full.completed);
  REQUIRE(full.mass_updates > 0);
  o.energy_tolerance = 1.0e-300;
  o.energy_limit = 1.0e-300;
  ExplicitDynamics dyn(model, assembler, nl, {}, o);
  const ExplicitResult stopped = dyn.run(drive, {zero, {}, {}, {}});
  INFO(stopped.termination);
  REQUIRE_FALSE(stopped.completed);
  REQUIRE(stopped.steps == 0);
  REQUIRE(stopped.final_state.displacement == zero);
  // The start's mass: no mass update, the start's added mass and scales.
  o.mass_scaling.dynamic = false;
  o.stable_step.update_every = 0;
  drive.duration = o.mass_scaling.target_time_step;
  o.energy_tolerance = 0.05;
  o.energy_limit = 0.5;
  const ExplicitResult start =
      ExplicitDynamics(model, assembler, nl, {}, o).run(drive, {zero, {}, {}, {}});
  REQUIRE(start.completed);
  CHECK(stopped.mass_updates == 0);
  CHECK(stopped.step_updates == 0);
  CHECK(stopped.added_mass_fraction == start.added_mass_fraction);
  CHECK(stopped.max_mass_scale == start.max_mass_scale);
  CHECK(stopped.scaled_elements == start.scaled_elements);
  CHECK(stopped.min_time_step == start.time_step);
  CHECK(full.added_mass_fraction > stopped.added_mass_fraction);
}

// ---------------------------------------------------------------------------
// 2. The dedicated Hex8 kernel
// ---------------------------------------------------------------------------
TEST_CASE("the dedicated Hex8 kernel gives the internal forces of the element dispatch",
          "[explicit]") {
  IsotropicMaterial hill = j2_material();
  {
    PlasticityParameters p = hill.plasticity();
    p.criterion = YieldCriterion::Hill48;
    p.hill.r0 = 1.8;
    p.hill.r45 = 1.3;
    p.hill.r90 = 2.3;
    hill.set_plasticity(p);
  }
  IsotropicMaterial chaboche = j2_material();
  {
    PlasticityParameters p = chaboche.plasticity();
    p.num_backstresses = 2;
    p.backstresses[0] = {20.0e9, 200.0};
    p.backstresses[1] = {2.0e9, 10.0};
    chaboche.set_plasticity(p);
  }
  struct Material {
    std::string name;
    IsotropicMaterial m;
  };
  const std::vector<Material> materials = {{"elastic", default_material(0.33)},
                                           {"J2 isotropic", j2_material()},
                                           {"J2 Prager + Voce", j2_material(2.0e9, 50.0e6)},
                                           {"Hill48", hill},
                                           {"Chaboche", chaboche}};
  const Mesh uniform = hex_block(3, 2, 2, 0.003, 0.002, 0.001);
  const Mesh irregular = distorted(uniform, 0.12);
  Scalar worst = 0.0;
  for (const Material& mat : materials) {
    for (const Kinematics kin :
         {Kinematics::SmallStrain, Kinematics::Finite, Kinematics::FiniteLogarithmic}) {
      // Finite kinematics take elastoplastic materials only.
      if (kin != Kinematics::SmallStrain && !mat.m.plasticity().enabled()) continue;
      for (const MeanDilatation md : {MeanDilatation::Auto, MeanDilatation::None}) {
        for (const bool regular : {true, false}) {
          INFO(mat.name << (kin == Kinematics::Finite ? ", finite"
                            : kin == Kinematics::FiniteLogarithmic ? ", logarithmic"
                                                                    : ", small strain")
                        << (md == MeanDilatation::Auto ? ", mean dilatation" : "")
                        << (regular ? ", uniform mesh" : ", distorted mesh"));
          const FemModel model = finalised(regular ? uniform : irregular, mat.m,
                                           {clamp(box(-kInf, 1.0e-9))});
          const Assembler assembler(model);
          NonlinearOptions nl;
          nl.kinematics = kin;
          nl.mean_dilatation = md;
          ExplicitDynamics dyn(model, assembler, nl, {}, ExplicitOptions());
          REQUIRE(dyn.dedicated());
          const Index n = model.dofs().num_dofs();
          // A large rotation (finite kinematics) and strains of about 1e-2:
          // points on and inside the yield surface.
          Vector u = random_vector(n, 11, 2.0e-6);
          if (kin != Kinematics::SmallStrain) {
            // (With a 20 % stretch along x: a finite strain, where the
            // logarithmic and Green-Lagrange strains differ.)
            const Matrix3 rot = Eigen::AngleAxisd(0.4, Vector3(1, 2, 3).normalized()).matrix() *
                                Vector3(1.2, 1.0, 1.0).asDiagonal();
            for (Index node = 0; node < model.mesh().num_nodes(); ++node) {
              const Vector3 xn = model.mesh().node(node);
              u.segment<3>(3 * node) += (rot - Matrix3::Identity()) * xn;
            }
          }
          std::vector<std::vector<PlasticState>> history;
          if (mat.m.plasticity().enabled()) {
            std::mt19937 rng(5);
            std::uniform_real_distribution<Scalar> d(-1.0, 1.0);
            history.resize(static_cast<std::size_t>(model.mesh().num_elements()));
            for (auto& points : history) {
              points.resize(8);
              for (PlasticState& p : points) {
                for (int r = 0; r < 6; ++r) p.plastic_strain(r) = 2.0e-3 * d(rng);
                const Scalar tr = (p.plastic_strain(0) + p.plastic_strain(1) +
                                   p.plastic_strain(2)) / 3.0;
                for (int r = 0; r < 3; ++r) p.plastic_strain(r) -= tr;
                p.equivalent_plastic_strain = 0.02 * (1.0 + d(rng));
                p.loading = d(rng) > 0.0;
                const int terms = mat.m.plasticity().kinematic_terms();
                if (mat.m.plasticity().general()) {
                  for (int i = 0; i < terms; ++i) {
                    Vector6& a = p.back_stresses[static_cast<std::size_t>(i)];
                    for (int r = 0; r < 6; ++r) a(r) = 10.0e6 * d(rng);
                    const Scalar m = (a(0) + a(1) + a(2)) / 3.0;
                    for (int r = 0; r < 3; ++r) a(r) -= m;
                    p.back_stress += a;
                  }
                } else if (terms > 0) {
                  for (int r = 0; r < 6; ++r) p.back_stress(r) = 10.0e6 * d(rng);
                  const Scalar m = (p.back_stress(0) + p.back_stress(1) + p.back_stress(2)) / 3.0;
                  for (int r = 0; r < 3; ++r) p.back_stress(r) -= m;
                }
              }
            }
          }
          const Vector fd = dyn.internal_force(u, history, false);
          const Vector fg = dyn.internal_force(u, history, true);
          const Scalar diff = (fd - fg).cwiseAbs().maxCoeff() / fg.cwiseAbs().maxCoeff();
          INFO("relative difference " << diff);
          REQUIRE(diff < 1.0e-13);
          worst = std::max(worst, diff);
        }
      }
    }
  }
  WARN("largest relative difference of the dedicated kernel: " << worst);
}

TEST_CASE("the dedicated kernel's logarithmic strain holds at equal and nearly equal principal "
          "stretches",
          "[explicit]") {
  // The kernel takes the closed-form eigensolver and falls back to the
  // iterative one where that loses accuracy: equal stretches (a uniform
  // dilatation, an equibiaxial stretch), stretches equal to 1e-9, tiny and
  // large strains, unrotated and rotated - against the element dispatch.
  // (Rotated by 0.7 rad, E = (H + H^T + H^T H) / 2 sums terms of order 1
  // to a strain of order 1e-7: both paths then carry a round-off of 1e-9
  // of the strain in their Green-Lagrange strain, whatever the eigensolver,
  // so the rotated cases start at a strain of 1e-2.)
  const FemModel model =
      finalised(hex_block(2, 2, 1, 0.002, 0.002, 0.001), j2_material(), {clamp(box(-kInf, 1e-9))});
  const Assembler assembler(model);
  NonlinearOptions nl;
  nl.kinematics = Kinematics::FiniteLogarithmic;
  ExplicitDynamics dyn(model, assembler, nl, {}, ExplicitOptions());
  REQUIRE(dyn.dedicated());
  const Matrix3 rotation = Eigen::AngleAxisd(0.7, Vector3(3, -1, 2).normalized()).matrix();
  Scalar worst = 0.0;
  for (const Scalar magnitude : {1.0e-7, 1.0e-4, 1.0e-2, 0.3}) {
    for (const Vector3& shape : {Vector3(1, 1, 1), Vector3(1, 1, -0.5), Vector3(1, 1 + 1e-9, 0.5),
                                 Vector3(-0.5, 1, 1 + 1e-12)}) {
      for (const bool rotated : {false, true}) {
        if (rotated && magnitude < 1.0e-2) continue;
        INFO("stretch 1 + " << magnitude << " x (" << shape.transpose() << ")"
                            << (rotated ? ", rotated" : ""));
        const Matrix3 f = (rotated ? rotation : Matrix3::Identity()) *
                          (Matrix3::Identity() + magnitude * Matrix3(shape.asDiagonal()));
        Vector u(model.dofs().num_dofs());
        for (Index node = 0; node < model.mesh().num_nodes(); ++node) {
          const Vector3 x = model.mesh().node(node);
          u.segment<3>(3 * node) = (f - Matrix3::Identity()) * x;
        }
        const Vector fd = dyn.internal_force(u, {}, false);
        const Vector fg = dyn.internal_force(u, {}, true);
        const Scalar diff = (fd - fg).cwiseAbs().maxCoeff() / fg.cwiseAbs().maxCoeff();
        INFO("relative difference " << diff);
        REQUIRE(diff < 1.0e-13);
        worst = std::max(worst, diff);
      }
    }
  }
  WARN("largest relative difference at equal stretches: " << worst);
}

TEST_CASE("the kernel switched off takes the generic element dispatch; an elastic material "
          "under finite kinematics is refused",
          "[explicit]") {
  const FemModel model = finalised(hex_block(2, 1, 1, 0.002, 0.001, 0.001), j2_material(),
                                   {clamp(box(-kInf, 1.0e-9))});
  const Assembler assembler(model);
  NonlinearOptions nl;
  nl.kinematics = Kinematics::FiniteLogarithmic;
  // Logarithmic kinematics: in the kernel.
  ExplicitDynamics log_kin(model, assembler, nl, {}, ExplicitOptions());
  REQUIRE(log_kin.dedicated());
  // An elastic material: refused under finite and logarithmic kinematics
  // (sparlab_form models elastoplastic sheets), in the kernel under small
  // strain.
  const FemModel elastic = finalised(hex_block(2, 1, 1, 0.002, 0.001, 0.001), default_material(),
                                     {clamp(box(-kInf, 1.0e-9))});
  const Assembler elastic_assembler(elastic);
  for (const Kinematics kin : {Kinematics::Finite, Kinematics::FiniteLogarithmic}) {
    NonlinearOptions finite;
    finite.kinematics = kin;
    REQUIRE_THROWS_AS(ExplicitDynamics(elastic, elastic_assembler, finite, {}, ExplicitOptions()),
                      ConfigError);
  }
  REQUIRE(ExplicitDynamics(elastic, elastic_assembler, small_strain(), {}, ExplicitOptions())
              .dedicated());
  ExplicitOptions off;
  off.dedicated_kernel = false;
  ExplicitDynamics generic(model, assembler, small_strain(), {}, off);
  REQUIRE_FALSE(generic.dedicated());
  const Vector u = random_vector(model.dofs().num_dofs(), 3, 1.0e-6);
  REQUIRE((generic.internal_force(u, {}, false) - generic.internal_force(u, {}, true)).norm() ==
          0.0);
}

// ---------------------------------------------------------------------------
// 3. Energy balance
// ---------------------------------------------------------------------------
TEST_CASE("an undamped linear run balances its energy to second order in the step",
          "[explicit]") {
  const FemModel model = cantilever_block();
  const Assembler assembler(model);
  const Modes modes = lumped_modes(assembler);
  const Scalar limit = 2.0 / std::sqrt(modes.omega2.maxCoeff());
  ExplicitState start;
  start.displacement = scatter_free(model, random_vector(modes.phi.rows(), 4, 1.0e-6));
  const Scalar duration = 200.0 * limit;
  std::vector<Scalar> errors;
  for (const Scalar fraction : {0.4, 0.2, 0.1}) {
    ExplicitOptions o = fixed_step(fraction * limit);
    o.history_every = 1;
    ExplicitDynamics dyn(model, assembler, small_strain(), {}, o);
    const ExplicitResult r = dyn.run(held_drive(model, duration), start);
    REQUIRE(r.completed);
    errors.push_back(r.max_energy_error);
    // The internal work of a linear model is its stored energy's change.
    const ExplicitRecord& last = r.records.back();
    REQUIRE(last.internal == Approx(last.stored - r.records.front().stored).epsilon(1e-9));
    REQUIRE(std::abs(last.plastic) < 1.0e-9 * last.stored);
  }
  INFO("errors " << errors[0] << " " << errors[1] << " " << errors[2]);
  REQUIRE(errors[0] < 0.1);
  for (std::size_t i = 1; i < errors.size(); ++i) {
    const Scalar order = std::log2(errors[i - 1] / errors[i]);
    INFO("order " << order);
    REQUIRE(order > 1.9);
    REQUIRE(order < 2.1);
  }
}

TEST_CASE("a plastic run dissipates energy and closes its balance", "[explicit]") {
  // A bar pulled 2 % at its end (small strain), damped.
  const Scalar length = 0.01;
  const FemModel model =
      finalised(hex_block(8, 1, 1, length, 0.001, 0.001), j2_material(2.0e9),
                {clamp(box(-kInf, 1.0e-9), {0}), clamp(box(-kInf, 1.0e-9, -kInf, 1e-9), {1}),
                 clamp(box(-kInf, 1e-9, -kInf, kInf, -kInf, 1e-9), {2}),
                 clamp(box(length - 1e-9, kInf), {0}, Vector3(0.02 * length, 0, 0))});
  const Assembler assembler(model);
  ExplicitOptions o;
  o.history_every = 20;
  o.mass_damping = 2.0e4;
  ExplicitDynamics dyn(model, assembler, small_strain(), {}, o);
  ExplicitDrive drive = held_drive(model, 2.0e-4);
  drive.fixed_start.setZero();  // from the reference, to the model's values
  const ExplicitResult r =
      dyn.run(drive, ExplicitState{Vector::Zero(model.dofs().num_dofs()), {}, {}, {}});
  REQUIRE(r.completed);
  Scalar previous = 0.0;
  const Scalar scale = r.records.back().internal;
  for (const ExplicitRecord& rec : r.records) {
    INFO("step " << rec.step);
    REQUIRE(rec.plastic >= previous - 1.0e-9 * scale);
    previous = rec.plastic;
  }
  REQUIRE(r.records.back().plastic > 0.5 * scale);
  REQUIRE(r.records.back().damping > 0.0);
  REQUIRE(r.max_energy_error < 1.0e-2);
  // The bar is plastic along its length.
  REQUIRE(r.records.back().max_plastic_strain > 0.01);
}

TEST_CASE("the dedicated kernel's elastic energy is the stored energy less the hardening energy",
          "[explicit]") {
  // The bar pulled 2 % (J2, linear isotropic hardening): the hardening
  // energy is sum_q w_q int_0^alpha (sigma_y - sigma_y0), from the history;
  // the elastic energy the rest of the stored energy, for every kinematics
  // the kernel takes. The generic dispatch does not separate it (NaN).
  const Scalar length = 0.01;
  const IsotropicMaterial material = j2_material();
  const FemModel model =
      finalised(hex_block(8, 1, 1, length, 0.001, 0.001), material,
                {clamp(box(-kInf, 1.0e-9), {0}), clamp(box(-kInf, 1.0e-9, -kInf, 1e-9), {1}),
                 clamp(box(-kInf, 1e-9, -kInf, kInf, -kInf, 1e-9), {2}),
                 clamp(box(length - 1e-9, kInf), {0}, Vector3(0.02 * length, 0, 0))});
  const Assembler assembler(model);
  const Scalar weight = model.mesh().element_measure(0) / 8.0;  // every point of the uniform bar
  ExplicitOptions o;
  o.history_every = 50;
  o.mass_damping = 2.0e4;
  ExplicitDrive drive = held_drive(model, 2.0e-4);
  drive.fixed_start.setZero();
  for (const Kinematics kin :
       {Kinematics::SmallStrain, Kinematics::Finite, Kinematics::FiniteLogarithmic}) {
    INFO((kin == Kinematics::SmallStrain ? "small strain"
          : kin == Kinematics::Finite    ? "finite"
                                         : "logarithmic"));
    NonlinearOptions nl;
    nl.kinematics = kin;
    ExplicitDynamics dyn(model, assembler, nl, {}, o);
    REQUIRE(dyn.dedicated());
    const ExplicitResult r =
        dyn.run(drive, ExplicitState{Vector::Zero(model.dofs().num_dofs()), {}, {}, {}});
    REQUIRE(r.completed);
    Scalar hardening = 0.0;
    for (const std::vector<PlasticState>& points : r.final_state.history) {
      for (const PlasticState& p : points) {
        hardening += weight * material.plasticity().isotropic_energy(p.equivalent_plastic_strain);
      }
    }
    const ExplicitRecord& last = r.records.back();
    INFO("stored " << last.stored << " J, elastic " << last.elastic << " J, hardening "
                   << hardening << " J");
    REQUIRE(hardening > 0.1 * last.stored);
    REQUIRE(last.elastic > 0.0);
    REQUIRE(last.elastic + hardening == Approx(last.stored).epsilon(1.0e-12));
  }
  ExplicitOptions generic = o;
  generic.dedicated_kernel = false;
  const ExplicitResult g = ExplicitDynamics(model, assembler, small_strain(), {}, generic)
                               .run(drive, ExplicitState{Vector::Zero(model.dofs().num_dofs()),
                                                         {}, {}, {}});
  REQUIRE(g.completed);
  REQUIRE(std::isnan(g.records.back().elastic));
}

TEST_CASE("the time map moves the fastest tool at the tool speed and skips standstills",
          "[explicit]") {
  // Tool a: 3 mm along x by t = 1 s, still to t = 2 s, 4 mm along y by 3 s.
  // Tool b: 1 mm up by t = 0.5 s, then still.
  ToolTrajectory a;
  a.times = {0.0, 1.0, 2.0, 3.0};
  a.points = {Vector3::Zero(), Vector3(0.003, 0.0, 0.0), Vector3(0.003, 0.0, 0.0),
              Vector3(0.003, 0.004, 0.0)};
  ToolTrajectory b;
  b.times = {0.0, 0.5, 3.0};
  b.points = {Vector3(0.0, 0.0, 1.0), Vector3(0.0, 0.0, 1.001), Vector3(0.0, 0.0, 1.001)};
  const ExplicitTimeMap map = ExplicitTimeMap::tool_speed({&a, &b}, 0.0, 3.0, 2.0);
  // [0, 0.5]: a travels 1.5 mm, b 1 mm -> 0.75 ms; [0.5, 1]: 1.5 mm -> 0.75
  // ms; [1, 2]: nothing moves -> no time; [2, 3]: 4 mm -> 2 ms.
  REQUIRE(map.duration() == Approx(3.5e-3).epsilon(1e-14));
  REQUIRE(map.pseudo_time(0.0) == 0.0);
  REQUIRE(map.pseudo_time(0.375e-3) == Approx(0.25).epsilon(1e-14));
  REQUIRE(map.pseudo_time(0.75e-3) == Approx(0.5).epsilon(1e-14));
  REQUIRE(map.pseudo_time(1.5e-3 - 1e-12) == Approx(1.0).epsilon(1e-8));
  REQUIRE(map.pseudo_time(2.5e-3) == Approx(2.5).epsilon(1e-14));
  REQUIRE(map.pseudo_time(-1.0) == 0.0);
  REQUIRE(map.pseudo_time(1.0) == 3.0);
  // The fastest tool moves at the speed on every segment.
  for (const Scalar tau : {0.2e-3, 1.0e-3, 2.0e-3, 3.0e-3}) {
    INFO("tau " << tau);
    const Scalar h = 1.0e-7;
    const Scalar t0 = map.pseudo_time(tau);
    const Scalar t1 = map.pseudo_time(tau + h);
    const Scalar fastest = std::max((a.position(t1) - a.position(t0)).norm(),
                                    (b.position(t1) - b.position(t0)).norm()) / h;
    REQUIRE(fastest == Approx(2.0).epsilon(1e-9));
  }
  // A window of a duration is linear.
  const ExplicitTimeMap linear = ExplicitTimeMap::linear(1.0, 3.0, 0.01);
  REQUIRE(linear.duration() == 0.01);
  REQUIRE(linear.pseudo_time(0.005) == Approx(2.0).epsilon(1e-14));
  // Refused: no tool moving in the window, a speed or duration not positive.
  REQUIRE_THROWS_AS(ExplicitTimeMap::tool_speed({&a}, 1.0, 2.0, 1.0), ConfigError);
  REQUIRE_THROWS_WITH(ExplicitTimeMap::tool_speed({&a}, 1.0, 2.0, 1.0),
                      ContainsSubstring("no tool moves"));
  REQUIRE_THROWS_AS(ExplicitTimeMap::tool_speed({&a}, 0.0, 3.0, 0.0), ConfigError);
  REQUIRE_THROWS_AS(ExplicitTimeMap::linear(0.0, 1.0, 0.0), ConfigError);
}

// ---------------------------------------------------------------------------
// 3b. Contact: penalty, penetration, stability
// ---------------------------------------------------------------------------
namespace {

/// A 4 x 4 x 2 block (1 x 1 x 0.5 mm elements) clamped at its base and
/// pressed 20 um by a plane from 2 um above its top, at about 1 m/s.
struct PunchCase {
  FemModel model;
  std::vector<RigidTool> tools;
  PunchCase(Scalar friction = 0.0, Scalar tangential = 1.0)
      : model(finalised(hex_block(4, 4, 2, 0.004, 0.004, 0.001), j2_material(),
                        {clamp(box(-kInf, kInf, -kInf, kInf, -kInf, 1.0e-9))})) {
    RigidTool t;
    t.name = "punch";
    t.shape = RigidTool::Shape::Plane;
    t.normal = Vector3(0, 0, -1);
    t.surface = box(-kInf, kInf, -kInf, kInf, 0.001 - 1e-9, kInf);
    t.friction = friction;
    t.tangential_ratio = tangential;
    t.trajectory.times = {0.0, 1.0};
    t.trajectory.points = {Vector3(0.002, 0.002, 0.001002), Vector3(0.0021, 0.002, 0.00098)};
    tools.push_back(t);
  }
  ExplicitDrive drive(Scalar duration = 2.2e-5) const {
    ExplicitDrive d = held_drive(model, duration);
    d.active = {1};
    d.time_map = ExplicitTimeMap::linear(0.0, 1.0, duration);
    return d;
  }
  ExplicitResult run(const ExplicitOptions& o, Scalar duration = 2.2e-5) const {
    const Assembler assembler(model);
    ExplicitDynamics dyn(model, assembler, NonlinearOptions(), tools, o);
    return dyn.run(drive(duration),
                   ExplicitState{Vector::Zero(model.dofs().num_dofs()), {}, {}, {}});
  }
};

}  // namespace

TEST_CASE("the contact penetration is measured against the element thickness, falls as the "
          "penalty stiffens, and is warned about",
          "[explicit]") {
  const PunchCase c;
  ExplicitOptions o;
  o.history_every = 1;  // every step recorded: the largest penetration is in the records
  const ExplicitResult r = c.run(o);
  REQUIRE(r.completed);
  REQUIRE(r.contact);
  Scalar largest = 0.0;
  for (const ExplicitRecord& rec : r.records) {
    for (const ExplicitToolRecord& t : rec.tools) {
      INFO("step " << rec.step);
      // Volume over largest face of the 1 x 1 x 0.5 mm elements: 0.5 mm.
      REQUIRE(t.max_penetration_ratio == Approx(t.max_penetration / 0.5e-3).epsilon(1.0e-12));
      largest = std::max(largest, t.max_penetration_ratio);
    }
  }
  REQUIRE(largest > 0.0);
  REQUIRE(r.max_penetration_ratio == largest);
  const auto warned = [](const ExplicitResult& x) {
    for (const std::string& w : x.warnings) {
      if (w.find("penetration") != std::string::npos) return true;
    }
    return false;
  };
  REQUIRE_FALSE(warned(r));  // below the default 1 %
  ExplicitOptions strict = o;
  strict.penetration_warning = 0.5 * largest;
  REQUIRE(warned(c.run(strict)));
  // The penetration is about f dt^2 / (s_c m): four times the stiffness,
  // about a quarter of the penetration (dt^2 changes with s_c: 4.3 times).
  ExplicitOptions soft = o;
  soft.contact_stiffness = 0.1;
  ExplicitOptions stiff = o;
  stiff.contact_stiffness = 0.4;
  const Scalar ratio = c.run(soft).max_penetration_ratio / c.run(stiff).max_penetration_ratio;
  INFO("penetration ratio " << ratio);
  REQUIRE(ratio > 3.0);
  REQUIRE(ratio < 6.0);
  // The default penalty.
  REQUIRE(ExplicitOptions().contact_stiffness == 0.5);
  ExplicitOptions bad;
  bad.penetration_warning = 0.0;
  REQUIRE_THROWS_AS(bad.validate(), ConfigError);
}

TEST_CASE("the stable step reserves the stiffer of the normal and tangential contact penalties",
          "[explicit]") {
  // No mass scaling or damping: dt = safety sqrt(4 - s) / omega_max with s
  // the penalty's share, s_c or s_c times a frictional tool's tangential
  // penalty ratio where that is larger.
  ExplicitOptions o;
  o.history_every = 1000000;
  const auto first_step = [&](Scalar friction, Scalar tangential) {
    const PunchCase c(friction, tangential);
    const ExplicitResult r = c.run(o, 1.0e-6);
    REQUIRE(r.completed);
    return r.time_step;
  };
  const Scalar s_c = o.contact_stiffness;
  const Scalar base = first_step(0.1, 1.0);
  CHECK(first_step(0.1, 0.5) == base);                   // a softer tangential spring
  CHECK(first_step(0.0, 3.0) == base);                   // no friction, no tangential spring
  CHECK(first_step(0.1, 3.0) ==
        Approx(base * std::sqrt((4.0 - 3.0 * s_c) / (4.0 - s_c))).epsilon(1.0e-12));
  // A share of 4 or more leaves no stable step.
  const PunchCase too_stiff(0.1, 8.0);
  REQUIRE_THROWS_AS(too_stiff.run(o, 1.0e-6), ConfigError);
}

TEST_CASE("the energy balance is not judged in the first time steps after the first contact",
          "[explicit]") {
  // At the first touch the new contact forces' half-step kick is in the
  // kinetic energy a step before their work is in the balance: a relative
  // error of order 1 when that kick is all the energy there is. Recorded at
  // every step, with the stiffest penalty, the run must not stop there.
  const PunchCase c;
  ExplicitOptions o;
  o.history_every = 1;
  o.contact_stiffness = 1.0;
  const ExplicitResult r = c.run(o);
  INFO(r.termination);
  REQUIRE(r.completed);
  REQUIRE(r.contact);
  REQUIRE(r.max_energy_error < 0.05);
}

// ---------------------------------------------------------------------------
// 4. Determinism and allocations
// ---------------------------------------------------------------------------
namespace {

/// A clamped plastic sheet dented by a sphere moving over it with friction
/// (finite kinematics): every part of a forming run.
struct DentCase {
  FemModel model;
  std::vector<RigidTool> tools;
  DentCase()
      : model(finalised(hex_block(8, 8, 2, 0.008, 0.008, 0.001, 0.0, 0.0, -0.001), j2_material(),
                        {clamp(box(-kInf, 1e-9)), clamp(box(0.008 - 1e-9, kInf)),
                         clamp(box(-kInf, kInf, -kInf, 1e-9)),
                         clamp(box(-kInf, kInf, 0.008 - 1e-9, kInf))})) {
    RigidTool t;
    t.name = "ball";
    t.radius = 0.002;
    t.surface = box(-kInf, kInf, -kInf, kInf, -1e-9, kInf);
    t.friction = 0.1;
    t.trajectory.times = {0.0, 1.0, 2.0};
    t.trajectory.points = {Vector3(0.003, 0.004, 0.0021), Vector3(0.003, 0.004, 0.0017),
                           Vector3(0.005, 0.004, 0.0017)};
    tools.push_back(t);
  }
  ExplicitDrive drive(Scalar speed) const {
    ExplicitDrive d = held_drive(model, 1.0);
    d.active = {1};
    d.time_map = ExplicitTimeMap::tool_speed({&tools[0].trajectory}, 0.0, 2.0, speed);
    d.duration = d.time_map.duration();
    return d;
  }
};

ExplicitOptions dent_options() {
  ExplicitOptions o;
  o.mass_scaling.mode = MassScalingOptions::Mode::Selective;
  o.mass_scaling.target_time_step = 2.0e-7;
  o.history_every = 50;
  o.stable_step.update_every = 200;
  return o;
}

}  // namespace

TEST_CASE("an explicit run gives the same result, bit for bit, on any number of threads",
          "[explicit]") {
  const DentCase c;
  const Assembler assembler(c.model);
  NonlinearOptions nl;  // finite kinematics
  std::vector<ExplicitResult> runs;
#ifdef SPARLAB_HAVE_OPENMP
  const int before = omp_get_max_threads();
  for (const int threads : {1, 3}) {
    omp_set_num_threads(threads);
#else
  for (int threads = 1; threads <= 2; ++threads) {
#endif
    ExplicitDynamics dyn(c.model, assembler, nl, c.tools, dent_options());
    REQUIRE(dyn.dedicated());
    runs.push_back(dyn.run(c.drive(20.0), ExplicitState{
                                              Vector::Zero(c.model.dofs().num_dofs()), {}, {}, {}}));
  }
#ifdef SPARLAB_HAVE_OPENMP
  omp_set_num_threads(before);
#endif
  const ExplicitResult& a = runs[0];
  const ExplicitResult& b = runs[1];
  REQUIRE(a.completed);
  REQUIRE(a.contact);
  REQUIRE(a.steps == b.steps);
  REQUIRE(a.records.back().max_plastic_strain > 0.0);
  REQUIRE((a.final_state.displacement - b.final_state.displacement).cwiseAbs().maxCoeff() == 0.0);
  REQUIRE((a.final_state.velocity - b.final_state.velocity).cwiseAbs().maxCoeff() == 0.0);
  REQUIRE(a.records.size() == b.records.size());
  for (std::size_t i = 0; i < a.records.size(); ++i) {
    REQUIRE(a.records[i].kinetic == b.records[i].kinetic);
    REQUIRE(a.records[i].internal == b.records[i].internal);
    for (std::size_t k = 0; k < a.records[i].tools.size(); ++k) {
      REQUIRE((a.records[i].tools[k].force - b.records[i].tools[k].force).norm() == 0.0);
    }
  }
}

TEST_CASE("the explicit step loop allocates no memory", "[explicit]") {
  // Runs of N and 2N steps allocate the same: nothing per step. (Records
  // and stable-step updates allocate, so there are none in between; the
  // end state's friction history holds one map entry per node in contact,
  // which the runs of different length may differ in.)
  const DentCase c;
  const Assembler assembler(c.model);
  NonlinearOptions nl;
  ExplicitOptions o = dent_options();
  o.history_every = 1000000;
  o.stable_step.update_every = 0;
  o.time_step = 1.0e-7;
  ExplicitState start{Vector::Zero(c.model.dofs().num_dofs()), {}, {}, {}};
  const auto count = [&](long steps) {
    ExplicitDynamics dyn(c.model, assembler, nl, c.tools, o);
    ExplicitDrive d = c.drive(20.0);
    d.duration = static_cast<Scalar>(steps) * o.time_step;
    g_allocations = 0;
    g_counting = true;
    // (Silent: the log lines' lengths differ with the step count.)
    const log::Level level = log::level();
    log::set_level(log::Level::Silent);
    const ExplicitResult r = dyn.run(d, start);
    g_counting = false;
    log::set_level(level);
    REQUIRE(r.completed);
    REQUIRE(r.steps == steps);
    if (steps >= 400) REQUIRE(r.contact);
    long entries = 0;
    for (const auto& tool : r.final_state.friction) entries += static_cast<long>(tool.size());
    return g_allocations.load() - entries;
  };
  (void)count(50);  // warm-up (thread pools)
  const long a = count(400);
  const long b = count(800);
  INFO("allocations " << a << " and " << b << " (less the friction history entries)");
  REQUIRE(a == b);
}

// ---------------------------------------------------------------------------
// 5. The explicit forming step (form_explicit) and its handoff
// ---------------------------------------------------------------------------
namespace {

/// The dent case as a forming analysis: `dent` (explicit) presses the ball
/// in and moves it along, `lift` (explicit) takes it off the sheet,
/// `settle` (explicit, no tool, damped) lets the sheet come to rest, and an
/// implicit `release` with the same clamps and no tool follows.
FormingOptions dent_forming(const DentCase& c) {
  FormingOptions o;
  RigidTool ball = c.tools[0];
  ball.trajectory.times = {0.0, 1.0, 2.0, 3.0};
  ball.trajectory.points = {Vector3(0.003, 0.004, 0.0021), Vector3(0.003, 0.004, 0.0017),
                            Vector3(0.005, 0.004, 0.0017), Vector3(0.005, 0.004, 0.0023)};
  o.tools.push_back(ball);
  FormingStep dent;
  dent.name = "dent";
  dent.type = FormingStep::Type::FormExplicit;
  dent.tools = {"ball"};
  dent.t_begin = 0.0;
  dent.t_end = 2.0;
  dent.tool_speed = 4.0;
  dent.explicit_options = dent_options();
  dent.explicit_options.mass_damping = 5.0e4;
  FormingStep lift = dent;
  lift.name = "lift";
  lift.t_begin = 2.0;
  lift.t_end = 3.0;
  // The sheet brought to rest: no tool, heavy mass-proportional damping.
  FormingStep settle;
  settle.name = "settle";
  settle.type = FormingStep::Type::FormExplicit;
  settle.duration = 2.5e-4;
  settle.explicit_options = dent_options();
  settle.explicit_options.mass_damping = 2.0e5;
  FormingStep release;
  release.name = "release";
  release.type = FormingStep::Type::Release;
  o.steps = {dent, lift, settle, release};
  return o;
}

}  // namespace

TEST_CASE("an explicit forming step hands its state to an implicit release exactly",
          "[explicit][forming]") {
  const DentCase c;
  const Assembler assembler(c.model);
  const FormingOptions o = dent_forming(c);
  const Index n = c.model.dofs().num_dofs();
  const FormingResult all = FormingAnalysis(c.model, assembler, o).run();
  REQUIRE(all.completed);
  REQUIRE(all.steps.size() == 4);
  const FormingStepResult& dent = all.steps[0];
  REQUIRE(dent.explicit_step);
  CHECK(dent.explicit_result.completed);
  CHECK(dent.explicit_result.contact);
  CHECK(dent.explicit_result.kernel == "dedicated Hex8");
  CHECK(dent.max_plastic_strain > 1.0e-3);
  CHECK(all.total_explicit_steps == dent.explicit_result.steps +
                                        all.steps[1].explicit_result.steps +
                                        all.steps[2].explicit_result.steps);
  // The recorded steps are the step's increments, the tool force averaged
  // over each record's steps; the ball is pushed up by the sheet.
  REQUIRE(dent.increments.size() + 1 == dent.explicit_result.records.size());
  CHECK(dent.increments.back().time == Approx(2.0));
  int pushed = 0;
  for (const FormingIncrement& inc : dent.increments) {
    if (inc.tools.at(0).active_nodes > 0 && inc.tools[0].force.z() > 0.0) ++pushed;
  }
  CHECK(pushed > static_cast<int>(dent.increments.size()) / 2);
  // The lift ends with the ball off the sheet. Its kinetic energy ratio is
  // measured against the dented sheet's elastic energy - neither against
  // the little work of the lift itself nor against the hardening energy of
  // the dent, which no retract can stir.
  CHECK(all.steps[1].increments.back().tools.at(0).active_nodes == 0);
  const ExplicitResult& lift = all.steps[1].explicit_result;
  INFO("lift: kinetic ratio " << lift.max_kinetic_ratio << ", internal work "
                              << lift.records.back().internal << " J, elastic energy "
                              << lift.records.back().elastic << " J of a stored energy "
                              << lift.records.back().stored << " J");
  CHECK(lift.contact);
  CHECK(lift.kinetic_ratio_basis == "elastic energy");
  // (The dent's stored energy is 92 % hardening energy: against it the
  // lift's ratio was 13 times smaller.)
  CHECK(lift.records.back().elastic > 0.0);
  CHECK(lift.records.back().elastic < 0.1 * lift.records.back().stored);
  CHECK(lift.max_kinetic_ratio > 0.0);

  // A restart from the end of the explicit steps releases to the same state
  // bit for bit: the handoff (displacement, velocity, plastic and friction
  // history) is the state the analysis carries.
  FormingOptions first = o;
  first.steps = {o.steps[0], o.steps[1], o.steps[2]};
  const FormingResult formed = FormingAnalysis(c.model, assembler, first).run();
  REQUIRE(formed.completed);
  REQUIRE(formed.final_state.velocity.size() == n);  // an explicit step hands on its velocity
  CHECK((formed.final_state.displacement - all.steps[2].displacement).cwiseAbs().maxCoeff() ==
        0.0);
  FormingOptions second = o;
  second.steps = {o.steps[3]};
  const FormingResult released = FormingAnalysis(c.model, assembler, second).run(formed.final_state);
  REQUIRE(released.completed);
  CHECK((released.final_state.displacement - all.final_state.displacement)
            .cwiseAbs()
            .maxCoeff() == 0.0);
  CHECK(all.final_state.velocity.size() == 0);  // an implicit step ends at rest

  // With the partition unchanged and the tool off the sheet, nothing
  // changed at the handoff (report item 8, the import round trip): the
  // release ramps out only what the explicit end state leaves - the inertia
  // and damping forces of a run damped to rest, 6e-14 of the forming force
  // here - and it moves the part by round-off (7e-19 m), at one iteration an
  // increment.
  const FormingStepResult& release = all.steps[3];
  Scalar max_formed = 0.0;
  for (Index node = 0; node < c.model.mesh().num_nodes(); ++node) {
    max_formed = std::max(max_formed, all.steps[2].displacement.segment(node * 3, 3).norm());
  }
  INFO("start imbalance " << release.start_imbalance << " N, reference force "
                          << release.reference_force << " N; displacement change "
                          << release.max_displacement_change << " m of " << max_formed
                          << " m; " << release.iterations << " iterations in "
                          << release.increments.size() << " increments");
  CHECK(release.start_imbalance < 1.0e-9 * release.reference_force);
  CHECK(release.max_displacement_change < 1.0e-12 * max_formed);
  CHECK(release.cuts == 0);
  CHECK(release.iterations <= static_cast<int>(release.increments.size()));
}

TEST_CASE("an implicit release straight after a moving explicit step ramps out the inertia "
          "forces it hands on",
          "[explicit][forming]") {
  // The dent and the lift, then the release at once - no settle: the
  // explicit end state still moves, and its inertia and damping forces are
  // part of the release's start imbalance, which it ramps out with the
  // rest. The lift continues a stressed part: its kinetic energy is
  // measured against the part's elastic energy.
  const DentCase c;
  const Assembler assembler(c.model);
  FormingOptions o = dent_forming(c);
  o.steps = {o.steps[0], o.steps[1], o.steps[3]};
  const FormingResult r = FormingAnalysis(c.model, assembler, o).run();
  REQUIRE(r.completed);
  const FormingStepResult& lift = r.steps[1];
  const FormingStepResult& release = r.steps[2];
  REQUIRE(lift.explicit_result.records.back().kinetic > 0.0);  // still moving
  INFO("start imbalance " << release.start_imbalance << " N, reference force "
                          << release.reference_force << " N; " << release.iterations
                          << " iterations in " << release.increments.size() << " increments");
  CHECK(release.start_imbalance > 1.0e-6 * release.reference_force);
  CHECK(release.start_imbalance < 0.1 * release.reference_force);
  CHECK(release.cuts == 0);
  CHECK(release.iterations <= 4 * static_cast<int>(release.increments.size()));
  // The kinetic energy ratios: against the internal energy (the dent, from
  // a virgin sheet) and the elastic energy (the lift), the unfiltered peak
  // at least the warned ratio.
  const ExplicitResult& dent = r.steps[0].explicit_result;
  const ExplicitResult& up = lift.explicit_result;
  CHECK(dent.kinetic_ratio_basis == "internal energy");
  CHECK(up.kinetic_ratio_basis == "elastic energy");
  CHECK(dent.peak_kinetic_ratio >= dent.max_kinetic_ratio);
  CHECK(up.peak_kinetic_ratio >= up.max_kinetic_ratio);
  Scalar peak = 0.0;
  for (const ExplicitRecord& rec : up.records) {
    REQUIRE(rec.elastic > 0.0);
    REQUIRE(rec.elastic < rec.stored);
    peak = std::max(peak, rec.kinetic / rec.elastic);
  }
  CHECK(up.peak_kinetic_ratio == peak);
}

namespace {

const char* kExplicitForming = R"({
      "kinematics": "finite",
      "tools": [
        {"name": "punch", "shape": "plane", "normal": [0, 0, -1],
         "surface": {"box": {"zmin": 0.002}}, "friction": 0.1,
         "trajectory": {"times": [0, 1, 1.5], "points": [[0, 0, 0.002], [0, 0, 0.00196],
                                                           [0, 0, 0.0021]]}}],
      "steps": [
        {"name": "press", "type": "form_explicit", "tools": ["punch"], "time": [0, 1],
         "explicit": {"tool_speed": 0.05,
                      "mass_scaling": {"mode": "selective", "target_time_step": 2e-7,
                                       "max_added_mass_fraction": 20, "dynamic": true},
                      "stable_step": {"method": "element_eigenvalue", "safety": 0.8,
                                      "update_every": 100, "power_iterations": 40},
                      "damping": 1000, "contact_stiffness": 0.2, "history_every": 50,
                      "snapshot_every": 400, "energy_tolerance": 0.1, "energy_limit": 0.8,
                      "kinetic_ratio_warning": 0.2}},
        {"name": "lift", "type": "form_explicit", "tools": ["punch"], "time": [1, 1.5],
         "explicit": {"duration": 0.002}},
        {"name": "release", "type": "release", "increments": 4,
         "boundary_conditions": [
           {"name": "A", "fix": ["x", "y", "z"], "region": {"nearest_node": [0, 0, 0]}},
           {"name": "B", "fix": ["y", "z"], "region": {"nearest_node": [0.004, 0, 0]}},
           {"name": "C", "fix": ["z"], "region": {"nearest_node": [0, 0.004, 0]}}]}],
      "output": {"vtk": false}
    })";

std::string explicit_deck(const std::string& forming) {
  return R"({
    "name": "tiny explicit forming",
    "mesh": {"type": "structured_hex", "nx": 2, "ny": 2, "nz": 2,
             "lx": 0.004, "ly": 0.004, "lz": 0.002},
    "material": {"youngs_modulus": 70e9, "poisson_ratio": 0.33, "density": 2700,
                 "plasticity": {"yield_stress": 100e6, "hardening_modulus": 300e6}},
    "boundary_conditions": [
      {"name": "base", "fix": ["z"], "region": {"box": {"zmax": 0.0}}},
      {"name": "corner", "fix": ["x", "y"], "region": {"nearest_node": [0, 0, 0]}},
      {"name": "edge", "fix": ["y"], "region": {"nearest_node": [0.004, 0, 0]}}],
    "forming": )" + forming + "}";
}

Configuration parse_explicit_deck(const std::string& text) {
  return parse_configuration(json::parse(text, "deck"), "deck", true, "");
}

}  // namespace

TEST_CASE("the explicit block of a form_explicit step is read in full, strictly",
          "[explicit][forming][io]") {
  const Configuration c = parse_explicit_deck(explicit_deck(kExplicitForming));
  const FormingOptions& o = c.forming.options;
  REQUIRE(o.steps.size() == 3);
  const FormingStep& press = o.steps[0];
  CHECK(press.type == FormingStep::Type::FormExplicit);
  CHECK(press.tool_speed == 0.05);
  CHECK(press.duration == 0.0);
  const ExplicitOptions& e = press.explicit_options;
  CHECK(e.mass_scaling.mode == MassScalingOptions::Mode::Selective);
  CHECK(e.mass_scaling.target_time_step == 2e-7);
  CHECK(e.mass_scaling.max_added_mass_fraction == 20.0);
  CHECK(e.mass_scaling.dynamic);
  CHECK(e.stable_step.method == StableStepOptions::Method::ElementEigenvalue);
  CHECK(e.stable_step.safety == 0.8);
  CHECK(e.stable_step.update_every == 100);
  CHECK(e.stable_step.power_iterations == 40);
  CHECK(e.mass_damping == 1000.0);
  CHECK(e.contact_stiffness == 0.2);
  CHECK(e.history_every == 50);
  CHECK(e.snapshot_every == 400);
  CHECK(e.energy_tolerance == 0.1);
  CHECK(e.energy_limit == 0.8);
  CHECK(e.kinetic_ratio_warning == 0.2);
  CHECK(o.steps[1].duration == 0.002);
  CHECK(o.steps[1].tool_speed == 0.0);
  CHECK(o.steps[1].explicit_options.mass_scaling.mode == MassScalingOptions::Mode::None);

  const auto refuse = [&](const std::string& from, const std::string& to,
                          const std::string& message) {
    std::string text = kExplicitForming;
    const std::size_t at = text.find(from);
    REQUIRE(at != std::string::npos);
    text.replace(at, from.size(), to);
    INFO(message);
    CHECK_THROWS_WITH(parse_explicit_deck(explicit_deck(text)), ContainsSubstring(message));
  };
  refuse(R"("duration": 0.002)", R"("duration": 0.002, "tool_speed": 1)", "not both");
  refuse(R"(,
         "explicit": {"duration": 0.002})", "", "needs an 'explicit' block");
  refuse(R"("duration": 0.002})", R"("duration": 0.002}, "max_tool_travel": 1e-4)",
         "'max_tool_travel' and 'increments' set the increments of an implicit step");
  refuse(R"("mode": "selective")", R"("mode": "sideways")", "unknown mass scaling mode");
  refuse(R"("method": "element_eigenvalue")", R"("method": "guess")",
         "unknown stable step method");
  refuse(R"("method": "element_eigenvalue")", R"("method": "power_iteration")",
         "selective mass scaling needs a time step per element");
  refuse(R"("safety": 0.8)", R"("safety": 1.5)", "'stable_step.safety' must lie in (0, 1]");
  refuse(R"("mode": "selective")", R"("mode": "uniform")",
         "'mass_scaling.dynamic' raises the scales of selective mass scaling");
  refuse(R"("contact_stiffness": 0.2)", R"("contact_stiffness": 2)", "'contact_stiffness'");
  refuse(R"("tool_speed": 0.05)", R"("tool_speed": -1)", "must be positive");
  refuse(R"("damping": 1000)", R"("damping": 1000, "dampnig": 1)", "dampnig");
  refuse(R"("type": "release", "increments": 4)",
         R"("type": "release", "increments": 4, "explicit": {"duration": 1})",
         "an 'explicit' block belongs to a step of type \"form_explicit\"");
}

TEST_CASE("a form_explicit run writes the step files, its energy history and its summary",
          "[explicit][forming][io]") {
  const Configuration c = parse_explicit_deck(explicit_deck(kExplicitForming));
  FemModel model = build_model(c);
  Assembler assembler(model);
  const FormingResult r = FormingAnalysis(model, assembler, c.forming.options).run();
  REQUIRE(r.completed);
  REQUIRE(r.steps.size() == 3);
  REQUIRE(r.steps[0].explicit_step);
  CHECK(r.steps[0].explicit_result.contact);
  CHECK_FALSE(r.steps[2].explicit_step);
  CHECK(r.timing.get("explicit") > 0.0);
  CHECK(r.timing.get("explicit_internal_force") > 0.0);
  // Snapshots every 400 steps of the press.
  CHECK(r.steps[0].snapshots.size() ==
        static_cast<std::size_t>((r.steps[0].explicit_result.steps - 1) / 400));

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "sparlab_test_explicit_out";
  std::filesystem::remove_all(dir);
  ResultWriter writer(dir.string(), c);
  const std::vector<std::string> files =
      write_forming_files(writer, model, c.forming.options, r, false, true);
  const auto first_line = [&](const std::string& name) {
    std::ifstream in((dir / name).string());
    std::string line;
    std::getline(in, line);
    return line;
  };
  const auto lines = [&](const std::string& name) {
    std::ifstream in((dir / name).string());
    std::string line;
    int count = 0;
    while (std::getline(in, line)) ++count;
    return count;
  };
  for (const std::string stem : {"step_1_press", "step_2_lift"}) {
    INFO(stem);
    CHECK(first_line(stem + "_nodes.csv") == "node,X,Y,Z,ux,uy,uz");
    CHECK(first_line(stem + "_energy.csv") ==
          "step,t_s,pseudo_t_s,time_step_s,kinetic_J,internal_work_J,stored_J,elastic_J,"
          "plastic_dissipation_J,contact_normal_work_J,contact_friction_work_J,damping_J,"
          "external_work_J,mass_scaling_work_J,energy_error_J,kinetic_ratio");
  }
  CHECK(lines("step_1_press_energy.csv") ==
        1 + static_cast<int>(r.steps[0].explicit_result.records.size()));
  CHECK_FALSE(std::filesystem::exists(dir / "step_3_release_energy.csv"));
  // The tool forces: a row per record of the explicit steps.
  CHECK(lines("tool_forces.csv") ==
        1 + static_cast<int>(r.steps[0].increments.size() + r.steps[1].increments.size()));

  const json::Value summary =
      forming_summary_json(c, model, c.forming.options, r, 1.0, "test", files);
  const json::Value& steps = *summary.find("steps");
  REQUIRE(steps.array_items().size() == 3);
  const json::Value* ex = steps.array_items()[0].find("explicit");
  REQUIRE(ex != nullptr);
  for (const char* key :
       {"steps", "physical_time_s", "tool_speed_m_s", "time_step_s", "stable_time_step_s",
        "scaled_stable_time_step_s", "mass_scaling", "mass_scale_max", "added_mass_fraction",
        "physical_mass_kg", "scaled_mass_kg", "max_kinetic_ratio", "max_energy_error",
        "kernel", "wall_s", "energy_file", "timing"}) {
    INFO(key);
    CHECK(ex->find(key) != nullptr);
  }
  CHECK(ex->find("steps")->number_value() ==
        static_cast<Scalar>(r.steps[0].explicit_result.steps));
  CHECK(ex->find("energy_file")->string_value() == "step_1_press_energy.csv");
  CHECK(steps.array_items()[2].find("explicit") == nullptr);
  std::filesystem::remove_all(dir);
}
