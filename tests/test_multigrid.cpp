/// \file test_multigrid.cpp
/// \brief Smoothed-aggregation multigrid and preconditioned CG.
///
/// The hierarchy is checked against its defining identities (the tentative
/// prolongator reproduces the rigid-body modes exactly, coarse operators are
/// Galerkin products, the V-cycle is a symmetric positive-definite operator),
/// the solver against the sparse Cholesky factorisation on every element
/// type, and the property that motivates it - an iteration count that barely
/// grows under refinement, where Jacobi-preconditioned CG's doubles - is
/// measured, not assumed.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/LinearSolver.hpp"
#include "sparlab/fem/ModalAnalysis.hpp"
#include "sparlab/fem/Multigrid.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

#if defined(SPARLAB_HAVE_OPENMP)
#include <omp.h>
#endif

#include <cmath>
#include <functional>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

StructuredMeshSpec box(Index nx, Index ny, Index nz, Scalar lx, Scalar ly, Scalar lz) {
  StructuredMeshSpec spec;
  spec.nx = nx;
  spec.ny = ny;
  spec.nz = nz;
  spec.lx = lx;
  spec.ly = ly;
  spec.lz = lz;
  return spec;
}

/// Clamped at x = 0 (optionally only some components elsewhere), loaded at
/// the far end; works for any mesh.
FemModel make_beam(Mesh mesh, bool roller_top = false, Scalar tip_displacement = 0.0) {
  const int dim = mesh.dim();
  const Scalar length = mesh.bounding_box().upper.x();
  const Scalar height = mesh.bounding_box().upper.y();
  FemModel model(std::move(mesh), default_material(0.3), dim == 2 ? 0.01 : 1.0,
                 dim == 2 ? StressState::PlaneStress : StressState::ThreeDimensional,
                 IntegrationOptions());
  DisplacementConstraint root;
  Selector left;
  left.kind = SelectorKind::Box;
  left.xmax = 0.0;
  root.region.members.push_back(left);
  root.fix_x = root.fix_y = true;
  root.fix_z = dim == 3;
  model.constraints().push_back(root);
  if (roller_top) {
    // A partial constraint: only y fixed along the top edge's first half,
    // which leaves nodes with a mix of free and fixed components.
    DisplacementConstraint roller;
    Selector top;
    top.kind = SelectorKind::Box;
    top.ymin = height;
    top.xmax = 0.5 * length;
    roller.region.members.push_back(top);
    roller.fix_y = true;
    model.constraints().push_back(roller);
  }
  if (tip_displacement != 0.0) {
    DisplacementConstraint pull;
    Selector tip;
    tip.kind = SelectorKind::Box;
    tip.xmin = length;
    pull.region.members.push_back(tip);
    pull.fix_x = true;
    pull.value_x = tip_displacement;
    model.constraints().push_back(pull);
  }
  LoadCaseSpec load;
  load.name = "tip";
  PointLoadSpec p;
  Selector corner;
  corner.kind = SelectorKind::NearestNode;
  corner.point = Vector3(length, 0.0, 0.0);
  p.region.members.push_back(corner);
  p.force = Vector3(0.0, -1000.0, dim == 3 ? 200.0 : 0.0);
  load.point_loads.push_back(p);
  model.load_case_specs().push_back(load);
  model.finalize();
  return model;
}

struct Solved {
  Vector u;
  int iterations = 0;
  Scalar compliance = 0.0;
};

Solved solve_with(const FemModel& model, LinearSolverOptions linear,
                  const Vector* scale = nullptr) {
  Assembler assembler(model);
  StaticAnalysisOptions options;
  options.linear = linear;
  options.check_model = scale == nullptr;
  StaticAnalysis analysis(model, assembler, options);
  const std::vector<StaticSolution> sol = analysis.solve_all(scale);
  return {sol.front().displacement, sol.front().solver_iterations, sol.front().compliance};
}

LinearSolverOptions amg_options(AmgSmoother smoother = AmgSmoother::Chebyshev,
                                Scalar tolerance = 1.0e-12) {
  LinearSolverOptions o;
  o.type = LinearSolverType::AmgCg;
  o.iterative_tolerance = tolerance;
  o.amg.smoother = smoother;
  o.amg.smoother_degree = smoother == AmgSmoother::Chebyshev ? 3 : 1;
  o.amg.coarse_size = 60;  // force several levels on test-sized models
  return o;
}

LinearSolverOptions ldlt_options() {
  LinearSolverOptions o;
  o.type = LinearSolverType::SimplicialLdlt;
  return o;
}

/// K_ff and its layout for direct use of the preconditioner.
struct System {
  SparseMatrix k;
  DofLayout layout;
};

System reduced_system(const FemModel& model, const Assembler& assembler,
                      const Vector* scale = nullptr) {
  System s;
  s.k = assembler.reduce_free_free(assembler.assemble_stiffness(scale));
  s.layout.dim = model.dim();
  s.layout.coordinates = &model.mesh().coordinates();
  s.layout.unknowns = &model.dofs().free_dofs();
  return s;
}

Vector pseudo_random(Index n, Scalar phase) {
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = std::sin(0.37 * static_cast<Scalar>(i) + phase) + 0.2;
  return v;
}

}  // namespace

TEST_CASE("deterministic kernels agree with the reference arithmetic",
          "[multigrid][solver]") {
  const Vector a = pseudo_random(100000, 0.1);
  const Vector b = pseudo_random(100000, 1.3);
  REQUIRE(deterministic_dot(a, b) == Approx(a.dot(b)).epsilon(1.0e-13));
  const FemModel model = make_beam(make_structured_hex_mesh(box(8, 3, 3, 0.8, 0.3, 0.3)));
  const Assembler assembler(model);
  const System s = reduced_system(model, assembler);
  const Vector x = pseudo_random(s.k.rows(), 0.7);
  Vector y;
  symmetric_spmv(s.k, x, y);
  const Vector reference = s.k * x;
  REQUIRE((y - reference).cwiseAbs().maxCoeff() <=
          1.0e-13 * reference.cwiseAbs().maxCoeff());
}

TEST_CASE("the multigrid hierarchy satisfies its defining identities",
          "[multigrid][solver][verification]") {
  for (int dim : {2, 3}) {
    INFO("dim " << dim);
    const FemModel model =
        dim == 2 ? make_beam(make_perturbed_quad_mesh(box(24, 8, 1, 1.2, 0.4, 1.0), 0.2, 3u),
                             true)
                 : make_beam(make_structured_tet_mesh(box(8, 3, 3, 0.8, 0.3, 0.3)), true);
    const Assembler assembler(model);
    const System s = reduced_system(model, assembler);
    AmgOptions options;
    options.coarse_size = 40;
    AmgPreconditioner amg(options);
    amg.setup(s.k, s.layout);
    REQUIRE(amg.num_levels() >= 3);
    const AmgStats& stats = amg.stats();
    REQUIRE(stats.near_null_space_dimension == (dim == 2 ? 3 : 6));
    REQUIRE(stats.operator_complexity > 1.0);
    REQUIRE(stats.operator_complexity < 3.0);
    for (int l = 0; l + 1 < amg.num_levels(); ++l) {
      INFO("level " << l);
      const SparseMatrix pt = amg.tentative_prolongator(l);
      const Matrix b = amg.near_null_space(l);
      const Matrix bc = amg.near_null_space(l + 1);
      // P-hat B_c = B: every rigid-body mode is represented exactly.
      REQUIRE((pt * bc - b).cwiseAbs().maxCoeff() <= 1.0e-12 * b.cwiseAbs().maxCoeff());
      // The columns of P-hat are orthonormal.
      const Matrix gram = Matrix(SparseMatrix(pt.transpose()) * pt);
      REQUIRE((gram - Matrix::Identity(gram.rows(), gram.cols())).cwiseAbs().maxCoeff() <
              1.0e-12);
      // A_{l+1} = P^T A_l P.
      const SparseMatrix p = amg.prolongator(l);
      const SparseMatrix al = amg.level_matrix(l);
      const Matrix galerkin = Matrix(SparseMatrix(p.transpose()) * al * p);
      const Matrix coarse = Matrix(amg.level_matrix(l + 1));
      REQUIRE((galerkin - coarse).cwiseAbs().maxCoeff() <=
              1.0e-12 * coarse.cwiseAbs().maxCoeff());
      REQUIRE(stats.levels[static_cast<std::size_t>(l) + 1].unknowns <
              stats.levels[static_cast<std::size_t>(l)].unknowns);
    }
    // Rigid-body motions of the fine level carry (almost) no energy: the
    // near-null space of the free unknowns is only resisted by the supports.
    const Matrix b0 = amg.near_null_space(0);
    REQUIRE(b0.cols() == (dim == 2 ? 3 : 6));
  }
}

TEST_CASE("the V-cycle is a symmetric positive-definite preconditioner",
          "[multigrid][solver][verification]") {
  const FemModel model =
      make_beam(make_perturbed_hex_mesh(box(8, 3, 3, 0.8, 0.3, 0.3), 0.2, 5u), true);
  const Assembler assembler(model);
  const System s = reduced_system(model, assembler);
  for (AmgSmoother smoother : {AmgSmoother::Chebyshev, AmgSmoother::SymmetricGaussSeidel}) {
    INFO("smoother " << to_string(smoother));
    AmgOptions options;
    options.smoother = smoother;
    options.coarse_size = 50;
    AmgPreconditioner amg(options);
    amg.setup(s.k, s.layout);
    const Vector x = pseudo_random(s.k.rows(), 0.2);
    const Vector y = pseudo_random(s.k.rows(), 2.1);
    Vector mx;
    Vector my;
    amg.apply(x, mx);
    amg.apply(y, my);
    REQUIRE(x.dot(my) == Approx(y.dot(mx)).epsilon(1.0e-11));
    REQUIRE(x.dot(mx) > 0.0);
    REQUIRE(y.dot(my) > 0.0);
    // A good preconditioner: one cycle already removes most of the error of
    // a smooth right-hand side's solution.
    Eigen::SimplicialLDLT<SparseMatrix> exact(s.k);
    const Vector truth = exact.solve(x);
    const Scalar reduction = (truth - mx).norm() / truth.norm();
    INFO("one-cycle relative error " << reduction);
    REQUIRE(reduction < 0.9);
  }
}

TEST_CASE("multigrid CG reproduces the direct solution on every element type",
          "[multigrid][solver][verification]") {
  struct Case {
    std::string name;
    std::function<Mesh()> mesh;
    bool roller;
    Scalar pull;
  };
  const std::vector<Case> cases = {
      {"quad4", [] { return make_perturbed_quad_mesh(box(40, 10, 1, 2.0, 0.5, 1.0), 0.2, 2u); },
       true, 0.0},
      {"tri3", [] { return make_perturbed_tri_mesh(box(30, 10, 1, 1.5, 0.5, 1.0), 0.2, 4u); },
       false, 1.0e-4},
      {"hex8", [] { return make_perturbed_hex_mesh(box(12, 4, 4, 1.2, 0.4, 0.4), 0.2, 6u); },
       true, 0.0},
      {"tet4", [] { return make_structured_tet_mesh(box(10, 4, 3, 1.0, 0.4, 0.3)); }, false,
       -2.0e-4}};
  for (const Case& c : cases) {
    for (AmgSmoother smoother : {AmgSmoother::Chebyshev, AmgSmoother::SymmetricGaussSeidel}) {
      INFO(c.name << ", " << to_string(smoother));
      const FemModel model = make_beam(c.mesh(), c.roller, c.pull);
      const Solved direct = solve_with(model, ldlt_options());
      const Solved amg = solve_with(model, amg_options(smoother));
      INFO("iterations " << amg.iterations);
      REQUIRE(amg.iterations > 0);
      REQUIRE(amg.iterations < 80);
      REQUIRE((amg.u - direct.u).norm() <= 1.0e-8 * direct.u.norm());
      REQUIRE(amg.compliance == Approx(direct.compliance).epsilon(1.0e-10));
    }
  }
}

TEST_CASE("multigrid iteration counts barely grow under refinement; Jacobi CG's do",
          "[multigrid][solver][verification]") {
  std::vector<int> amg_iterations;
  std::vector<int> jacobi_iterations;
  for (Index n : {4, 8, 16}) {
    const FemModel model = make_beam(make_structured_hex_mesh(box(2 * n, n, n, 1.0, 0.5, 0.5)));
    LinearSolverOptions amg = amg_options(AmgSmoother::Chebyshev, 1.0e-8);
    amg.amg.coarse_size = 200;
    amg_iterations.push_back(solve_with(model, amg).iterations);
    LinearSolverOptions jacobi;
    jacobi.type = LinearSolverType::ConjugateGradient;
    jacobi.iterative_tolerance = 1.0e-8;
    jacobi_iterations.push_back(solve_with(model, jacobi).iterations);
  }
  INFO("AMG " << amg_iterations[0] << ", " << amg_iterations[1] << ", " << amg_iterations[2]
              << "; Jacobi " << jacobi_iterations[0] << ", " << jacobi_iterations[1] << ", "
              << jacobi_iterations[2]);
  // Each refinement halves h: Jacobi-CG needs about twice the iterations,
  // multigrid CG a few more at most.
  REQUIRE(jacobi_iterations[2] > 3 * jacobi_iterations[0]);
  REQUIRE(amg_iterations[2] < 2 * amg_iterations[0]);
  REQUIRE(amg_iterations[2] < jacobi_iterations[2] / 4);
}

TEST_CASE("warm starts, thread counts and the automatic choice",
          "[multigrid][solver]") {
  const FemModel model = make_beam(make_structured_tet_mesh(box(12, 4, 4, 1.2, 0.4, 0.4)));
  const Assembler assembler(model);
  StaticAnalysisOptions options;
  options.linear = amg_options(AmgSmoother::Chebyshev, 1.0e-10);
  StaticAnalysis analysis(model, assembler, options);
  analysis.prepare();
  const Vector& f = model.load_vectors().front();
  const Vector cold = analysis.solve_load_vector(f);
  const int cold_iterations = analysis.last_iterations();
  // Starting from the answer costs nothing; from a nearby field, less.
  const Vector exact_start = analysis.solve_load_vector(f, &cold);
  REQUIRE(analysis.last_iterations() == 0);
  REQUIRE(exact_start == cold);
  const Vector nearby = cold * 1.001;
  (void)analysis.solve_load_vector(f, &nearby);
  REQUIRE(analysis.last_iterations() < cold_iterations);

#if defined(SPARLAB_HAVE_OPENMP)
  // Bitwise identical for any number of threads. The model is large enough
  // for the parallel kernels to engage.
  const FemModel big = make_beam(make_structured_hex_mesh(box(30, 10, 10, 1.5, 0.5, 0.5)));
  const int threads = omp_get_max_threads();
  omp_set_num_threads(1);
  const Solved serial = solve_with(big, amg_options());
  omp_set_num_threads(std::max(2, threads));
  const Solved parallel = solve_with(big, amg_options());
  omp_set_num_threads(threads);
  REQUIRE(serial.iterations == parallel.iterations);
  REQUIRE(serial.u == parallel.u);
#endif

  // Auto: direct below the limit, multigrid above it.
  LinearSolverOptions automatic;
  automatic.type = LinearSolverType::Auto;
  automatic.auto_direct_limit_3d = 1000000;
  const Solved small = solve_with(model, automatic);
  REQUIRE(small.iterations == 0);
  automatic.auto_direct_limit_3d = 100;
  automatic.iterative_tolerance = 1.0e-12;
  const Solved large = solve_with(model, automatic);
  REQUIRE(large.iterations > 0);
  REQUIRE((large.u - small.u).norm() <= 1.0e-8 * small.u.norm());
}

TEST_CASE("multigrid reports an under-constrained model and bad input",
          "[multigrid][solver][diagnostics]") {
  // Only x fixed at the root: the model can still slide in y and z and rotate.
  StructuredMeshSpec spec = box(8, 3, 3, 0.8, 0.3, 0.3);
  FemModel model(make_structured_hex_mesh(spec), default_material(), 1.0,
                 StressState::ThreeDimensional, IntegrationOptions());
  DisplacementConstraint weak;
  Selector left;
  left.kind = SelectorKind::Box;
  left.xmax = 0.0;
  weak.region.members.push_back(left);
  weak.fix_x = true;
  model.constraints().push_back(weak);
  LoadCaseSpec load;
  load.name = "pull";
  PointLoadSpec p;
  Selector tip;
  tip.kind = SelectorKind::Box;
  tip.xmin = 0.8;
  p.region.members.push_back(tip);
  p.force = Vector3(100.0, 0.0, 0.0);
  load.point_loads.push_back(p);
  model.load_case_specs().push_back(load);
  model.finalize();
  const Assembler assembler(model);
  const System s = reduced_system(model, assembler);
  AmgPreconditioner amg;
  REQUIRE_THROWS_WITH(amg.setup(s.k, s.layout), ContainsSubstring("under-constrained"));

  // A multigrid solver without a layout, and a non-symmetric matrix.
  auto solver = make_linear_solver(amg_options());
  REQUIRE_THROWS_WITH(solver->factorize(s.k), ContainsSubstring("layout"));
  const FemModel sound = make_beam(make_structured_hex_mesh(spec));
  const Assembler sound_assembler(sound);
  const System good = reduced_system(sound, sound_assembler);
  SparseMatrix skew = good.k;
  skew.coeffRef(0, 1) += 1.0e-6 * good.k.coeff(0, 0);
  AmgPreconditioner strict;
  REQUIRE_THROWS_WITH(strict.setup(skew, good.layout), ContainsSubstring("not symmetric"));
  REQUIRE_NOTHROW(strict.setup(good.k, good.layout));
  REQUIRE(strict.stats().coarse_pivot_ratio > 1.0e-10);
  DofLayout wrong = s.layout;
  std::vector<Index> short_list(3, 0);
  wrong.unknowns = &short_list;
  REQUIRE_THROWS_WITH(strict.setup(s.k, wrong), ContainsSubstring("layout"));
  AmgOptions bad;
  bad.strength_threshold = 1.5;
  REQUIRE_THROWS_AS(AmgPreconditioner(bad), ConfigError);
  REQUIRE(parse_linear_solver_type("amg") == LinearSolverType::AmgCg);
  REQUIRE(parse_linear_solver_type("auto") == LinearSolverType::Auto);
  REQUIRE(parse_amg_smoother("gauss_seidel") == AmgSmoother::SymmetricGaussSeidel);
  REQUIRE_THROWS_AS(parse_amg_smoother("jacobi"), ConfigError);
}

TEST_CASE("cached-pattern assembly is bitwise identical to triplet assembly",
          "[multigrid][assembly][verification]") {
  const std::vector<std::function<Mesh()>> meshes = {
      [] { return make_perturbed_quad_mesh(box(9, 5, 1, 0.9, 0.5, 1.0), 0.2, 1u); },
      [] { return make_perturbed_tri_mesh(box(9, 5, 1, 0.9, 0.5, 1.0), 0.2, 1u); },
      [] { return make_perturbed_hex_mesh(box(5, 3, 3, 0.5, 0.3, 0.3), 0.2, 1u); },
      [] { return make_perturbed_tet_mesh(box(5, 3, 3, 0.5, 0.3, 0.3), 0.2, 1u); }};
  for (const auto& mesh : meshes) {
    const FemModel model = make_beam(mesh(), true, 1.0e-4);
    INFO(to_string(model.mesh().element_type()));
    Assembler cached(model);
    Assembler triplets(model);
    triplets.set_pattern_assembly(false);
    Vector scale(model.mesh().num_elements());
    for (Index e = 0; e < scale.size(); ++e) {
      scale(e) = 1.0e-9 + std::pow(0.5 + 0.4 * std::sin(1.3 * e), 3.0);
    }
    for (const Vector* s : {static_cast<const Vector*>(nullptr), static_cast<const Vector*>(&scale)}) {
      const SparseMatrix a = cached.assemble_stiffness(s);
      const SparseMatrix b = triplets.assemble_stiffness(s);
      REQUIRE(a.nonZeros() == b.nonZeros());
      REQUIRE(std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.outerSize() + 1,
                         b.outerIndexPtr()));
      REQUIRE(std::equal(a.innerIndexPtr(), a.innerIndexPtr() + a.nonZeros(),
                         b.innerIndexPtr()));
      REQUIRE(std::equal(a.valuePtr(), a.valuePtr() + a.nonZeros(), b.valuePtr()));
      for (const auto& reduce : std::vector<std::function<SparseMatrix(const Assembler&,
                                                                      const SparseMatrix&)>>{
               [](const Assembler& x, const SparseMatrix& m) { return x.reduce_free_free(m); },
               [](const Assembler& x, const SparseMatrix& m) {
                 return x.reduce_free_prescribed(m);
               }}) {
        const SparseMatrix ra = reduce(cached, a);
        const SparseMatrix rb = reduce(triplets, b);
        REQUIRE(ra.nonZeros() == rb.nonZeros());
        REQUIRE(std::equal(ra.innerIndexPtr(), ra.innerIndexPtr() + ra.nonZeros(),
                           rb.innerIndexPtr()));
        REQUIRE(std::equal(ra.valuePtr(), ra.valuePtr() + ra.nonZeros(), rb.valuePtr()));
      }
    }
    const SparseMatrix ma = cached.assemble_mass(MassType::Consistent, &scale);
    const SparseMatrix mb = triplets.assemble_mass(MassType::Consistent, &scale);
    REQUIRE(std::equal(ma.valuePtr(), ma.valuePtr() + ma.nonZeros(), mb.valuePtr()));
    // A zero factor drops the element from a triplet assembly; the cached
    // path then defers to it rather than keep explicit zeros.
    Vector holes = scale;
    holes(0) = 0.0;
    const SparseMatrix ha = cached.assemble_stiffness(&holes);
    const SparseMatrix hb = triplets.assemble_stiffness(&holes);
    REQUIRE(ha.nonZeros() == hb.nonZeros());
  }
}

TEST_CASE("a reused hierarchy equals a fresh one bit for bit",
          "[multigrid][solver]") {
  const FemModel model =
      make_beam(make_perturbed_hex_mesh(box(10, 4, 4, 1.0, 0.4, 0.4), 0.2, 8u), true);
  const Assembler assembler(model);
  const System first = reduced_system(model, assembler);
  // A uniformly scaled matrix aggregates exactly like the first one, so the
  // reused hierarchy and a fresh one must coincide.
  const Vector factor = Vector::Constant(model.mesh().num_elements(), 2.5);
  const System second = reduced_system(model, assembler, &factor);
  for (AmgSmoother smoother : {AmgSmoother::Chebyshev, AmgSmoother::SymmetricGaussSeidel}) {
    AmgOptions options;
    options.coarse_size = 50;
    options.smoother = smoother;
    AmgPreconditioner reused(options);
    reused.setup(first.k, first.layout);
    reused.setup(second.k, first.layout);
    REQUIRE(reused.stats().reused_aggregates);
    AmgPreconditioner fresh(options);
    fresh.setup(second.k, second.layout);
    REQUIRE_FALSE(fresh.stats().reused_aggregates);
    REQUIRE(reused.num_levels() == fresh.num_levels());
    const Vector x = pseudo_random(second.k.rows(), 0.4);
    Vector zr;
    Vector zf;
    reused.apply(x, zr);
    fresh.apply(x, zf);
    REQUIRE(zr == zf);
  }
}
