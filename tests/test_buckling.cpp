/// \file test_buckling.cpp
/// \brief Geometric stiffness and linear buckling: element identities, the
///        Euler column in 2-D and 3-D, the subspace iteration against a dense
///        eigensolve, and the treatment of stiffening (tensile) load cases.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Element.hpp"
#include "sparlab/fem/Buckling.hpp"
#include "sparlab/fem/StaticAnalysis.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Eigenvalues>

#include <random>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

StructuredMeshSpec box_spec(Index nx, Index ny, Index nz, Scalar lx, Scalar ly, Scalar lz) {
  StructuredMeshSpec spec;
  spec.nx = nx;
  spec.ny = ny;
  spec.nz = nz;
  spec.lx = lx;
  spec.ly = ly;
  spec.lz = lz;
  return spec;
}

/// A column along x, clamped at x = 0, under an axial traction on the face
/// x = L whose resultant is `force` (negative = compression).
FemModel make_column(Mesh mesh, Scalar length, Scalar area, Scalar thickness, Scalar force,
                     Scalar poisson = 0.3) {
  const int dim = mesh.dim();
  FemModel model(std::move(mesh), IsotropicMaterial(200.0e9, poisson, 7850.0, "steel"),
                 thickness, dim == 2 ? StressState::PlaneStress : StressState::ThreeDimensional,
                 IntegrationOptions());
  DisplacementConstraint root;
  Selector box;
  box.kind = SelectorKind::Box;
  box.xmax = 0.0;
  root.region.members.push_back(box);
  root.fix_x = root.fix_y = true;
  root.fix_z = dim == 3;
  model.constraints().push_back(root);
  LoadCaseSpec load;
  load.name = "axial";
  TractionLoadSpec top;
  Selector end;
  end.kind = SelectorKind::Box;
  end.xmin = length;
  top.region.members.push_back(end);
  top.traction = Vector3(force / area, 0.0, 0.0);
  load.tractions.push_back(top);
  model.load_case_specs().push_back(load);
  model.finalize();
  return model;
}

Vector random_vector(Eigen::Index n, unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  Vector v(n);
  for (Eigen::Index i = 0; i < n; ++i) v(i) = dist(rng);
  return v;
}

}  // namespace

TEST_CASE("geometric stiffness: rigid translations, uniaxial stress and the adjoint identity",
          "[buckling][element]") {
  const IsotropicMaterial material = default_material(0.3);
  struct Sample {
    ElementType type;
    Mesh mesh;
  };
  std::vector<Sample> samples;
  samples.push_back({ElementType::Quad4, make_perturbed_quad_mesh(box_spec(3, 2, 1, 1.0, 0.6, 1.0), 0.25, 3u)});
  samples.push_back({ElementType::Tri3, make_perturbed_tri_mesh(box_spec(3, 2, 1, 1.0, 0.6, 1.0), 0.25, 3u)});
  samples.push_back({ElementType::Hex8, make_perturbed_hex_mesh(box_spec(3, 3, 3, 1.0, 0.6, 0.5), 0.2, 3u)});
  samples.push_back({ElementType::Tet4, make_perturbed_tet_mesh(box_spec(3, 3, 3, 1.0, 0.6, 0.5), 0.2, 3u)});
  samples.push_back({ElementType::Tet10, make_perturbed_tet10_mesh(box_spec(3, 3, 3, 1.0, 0.6, 0.5), 0.2, 3u)});
  for (const Sample& sample : samples) {
    INFO(to_string(sample.type));
    const Mesh& mesh = sample.mesh;
    const int dim = mesh.dim();
    const std::unique_ptr<Element> element = make_element(sample.type);
    const Matrix d = dim == 2 ? Matrix(material.plane_stress_matrix())
                              : Matrix(material.three_dimensional_matrix());
    const Scalar t = dim == 2 ? 0.01 : 1.0;
    // An interior-ish element with a distorted shape.
    const Index e = mesh.num_elements() / 2;
    const Matrix x = mesh.element_coordinates(e);
    const int edofs = element->num_dofs();
    const Vector ue = 1.0e-4 * random_vector(edofs, 11u);
    const Matrix kg = element->geometric_stiffness(x, d, ue, 0.7, t, IntegrationOptions());
    REQUIRE(kg.rows() == edofs);
    REQUIRE((kg - kg.transpose()).cwiseAbs().maxCoeff() <= 1.0e-12 * kg.cwiseAbs().maxCoeff());
    // A rigid translation has no displacement gradient.
    for (int k = 0; k < dim; ++k) {
      Vector shift = Vector::Zero(edofs);
      for (int a = 0; a < element->num_nodes(); ++a) shift(dim * a + k) = 1.0;
      REQUIRE((kg * shift).cwiseAbs().maxCoeff() <= 1.0e-10 * kg.cwiseAbs().maxCoeff());
    }
    // phi^T K_G(u) phi = g(phi)^T u, with g the derivative the adjoint uses.
    const Vector phi = random_vector(edofs, 29u);
    const Vector g = element->geometric_stiffness_derivative(x, d, phi, 0.7, t, IntegrationOptions());
    REQUIRE(phi.dot(kg * phi) == Approx(g.dot(ue)).epsilon(1.0e-11));

    // A uniform uniaxial stress sigma_xx = s0 (from u_x = eps x) contracted
    // with the mode phi_y = x gives s0 times the element measure (times t).
    const Scalar eps = 1.0e-3;
    Vector axial(edofs);
    Vector mode = Vector::Zero(edofs);
    for (int a = 0; a < element->num_nodes(); ++a) {
      axial.segment(dim * a, dim).setZero();
      axial(dim * a) = eps * x(0, a);
      // Lateral contraction so the stress is exactly uniaxial.
      const Scalar nu_eff = dim == 2 ? 0.3 : 0.3;
      for (int k = 1; k < dim; ++k) axial(dim * a + k) = -nu_eff * eps * x(k, a);
      mode(dim * a + 1) = x(0, a);
    }
    const Matrix kg_axial = element->geometric_stiffness(x, d, axial, 1.0, t, IntegrationOptions());
    const Scalar measure = mesh.element_measure(e) * t;
    REQUIRE(mode.dot(kg_axial * mode) ==
            Approx(material.youngs_modulus() * eps * measure).epsilon(1.0e-9));
  }
}

TEST_CASE("Euler columns: 2-D and 3-D buckling loads converge on the Engesser value",
          "[buckling][verification]") {
  const Scalar length = 1.0;
  SECTION("plane-stress Q4 column") {
    const Scalar h = 0.05;
    const Scalar t = 0.02;
    const Scalar inertia = t * h * h * h / 12.0;
    const Scalar theory = engesser_cantilever_load(200.0e9, 0.3, inertia, h * t, length);
    Scalar previous_error = 1.0;
    for (Index nx : {20, 40, 80}) {
      FemModel model = make_column(make_structured_quad_mesh(box_spec(nx, 4, 1, length, h, 1.0)),
                                   length, h * t, t, -1.0);
      Assembler assembler(model);
      BucklingOptions options;
      options.num_modes = 3;
      const BucklingResult result = analyse_buckling(model, assembler, 0, options);
      REQUIRE(result.converged);
      REQUIRE(result.load_factors.size() == 3);
      const Scalar error = result.load_factors(0) / theory - 1.0;
      INFO("nx " << nx << ": lambda_1 = " << result.load_factors(0) << " N, Engesser "
                 << theory << " N, error " << error);
      // Displacement elements are too stiff: the error is positive and falls
      // with refinement.
      REQUIRE(error > 0.0);
      REQUIRE(error < previous_error);
      previous_error = error;
      // The second cantilever mode sits near (3 pi / 2)^2 / (pi / 2)^2 = 9 times the first.
      REQUIRE(result.load_factors(1) / result.load_factors(0) == Approx(9.0).epsilon(0.05));
      REQUIRE(result.residuals.maxCoeff() < 1.0e-6);
    }
    // Measured: 39 %, 10 % and 3.1 % above Engesser for nx = 20, 40, 80
    // (four layers of full-integration Q4 lock in bending).
    REQUIRE(previous_error < 0.04);
  }
  SECTION("Hex8, Tet4 and Tet10 columns") {
    const Scalar h = 0.05;
    const Scalar inertia = h * h * h * h / 12.0;
    const Scalar theory = engesser_cantilever_load(200.0e9, 0.3, inertia, h * h, length);
    for (ElementType type : {ElementType::Hex8, ElementType::Tet10, ElementType::Tet4}) {
      INFO(to_string(type));
      const StructuredMeshSpec spec = box_spec(20, 2, 2, length, h, h);
      Mesh mesh = type == ElementType::Hex8   ? make_structured_hex_mesh(spec)
                  : type == ElementType::Tet4 ? make_structured_tet_mesh(spec)
                                              : make_structured_tet10_mesh(spec);
      FemModel model = make_column(std::move(mesh), length, h * h, 1.0, -1.0);
      Assembler assembler(model);
      BucklingOptions options;
      options.num_modes = 2;
      const BucklingResult result = analyse_buckling(model, assembler, 0, options);
      REQUIRE(result.converged);
      const Scalar error = result.load_factors(0) / theory - 1.0;
      INFO("lambda_1 = " << result.load_factors(0) << " N, Engesser " << theory
                         << " N, error " << error);
      REQUIRE(error > 0.0);
      // Measured on this 20 x 2 x 2 mesh: Tet10 0.5 % above Engesser, while
      // the fully integrated Hex8 (43 %) and the constant-strain Tet4 (155 %)
      // lock in bending.
      if (type == ElementType::Tet10) REQUIRE(error < 0.01);
      if (type == ElementType::Hex8) REQUIRE(error < 0.5);
      if (type == ElementType::Tet4) REQUIRE(error > 1.0);
      // The square section buckles about y and z at the same load; the Kuhn
      // split of the Tet4 mesh is not symmetric enough to show it.
      if (type != ElementType::Tet4) {
        REQUIRE(result.load_factors(1) == Approx(result.load_factors(0)).epsilon(0.002));
      }
    }
  }
}

TEST_CASE("buckling modes: normalisation, supports, and agreement with a dense eigensolve",
          "[buckling]") {
  // 2-D column with more than 400 free DOFs, so the subspace iteration runs.
  const Scalar length = 1.0;
  const Scalar h = 0.08;
  const Scalar t = 0.01;
  FemModel model = make_column(make_structured_quad_mesh(box_spec(60, 4, 1, length, h, 1.0)),
                               length, h * t, t, -1.0e3);
  Assembler assembler(model);
  REQUIRE(model.dofs().num_free() > 400);
  BucklingOptions options;
  options.num_modes = 4;
  const BucklingResult result = analyse_buckling(model, assembler, 0, options);
  REQUIRE(result.converged);
  REQUIRE(result.iterations > 0);
  REQUIRE(result.load_factors.size() == 4);

  // Dense reference on the same matrices.
  StaticAnalysis analysis(model, assembler);
  const Vector u = analysis.solve_all().front().displacement;
  const SparseMatrix k_full = assembler.assemble_stiffness();
  const SparseMatrix kg_full = assemble_geometric_stiffness(model, assembler, u);
  const Matrix k = Matrix(assembler.reduce_free_free(k_full));
  const Matrix g = -Matrix(assembler.reduce_free_free(kg_full));
  Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> ges(g, k);
  REQUIRE(ges.info() == Eigen::Success);
  const Vector mu = ges.eigenvalues().reverse();
  for (int i = 0; i < 4; ++i) {
    REQUIRE(result.load_factors(i) == Approx(1.0 / mu(i)).epsilon(1.0e-8));
  }
  // phi^T K phi = 1, (K + lambda K_G) phi = 0, zero at the supports.
  for (int i = 0; i < 4; ++i) {
    const Vector phi = result.mode_shapes.col(i);
    REQUIRE(phi.dot(k_full * phi) == Approx(1.0).epsilon(1.0e-10));
    const Vector r = model.dofs().restrict_to_free(k_full * phi + result.load_factors(i) * (kg_full * phi));
    REQUIRE(r.norm() <= 1.0e-6 * model.dofs().restrict_to_free(k_full * phi).norm());
    for (Index dof : model.dofs().constrained_dofs()) REQUIRE(phi(dof) == 0.0);
  }
  // Load factors scale inversely with the load.
  FemModel doubled = make_column(make_structured_quad_mesh(box_spec(60, 4, 1, length, h, 1.0)),
                                 length, h * t, t, -2.0e3);
  Assembler doubled_assembler(doubled);
  const BucklingResult half = analyse_buckling(doubled, doubled_assembler, 0, options);
  REQUIRE(half.load_factors(0) == Approx(0.5 * result.load_factors(0)).epsilon(1.0e-8));

  // A warm start from the converged subspace needs far fewer iterations.
  const DofManager& dofs = model.dofs();
  StaticAnalysis again(model, assembler);
  again.prepare();
  const FreeSolve solve = [&](const Vector& b) {
    return dofs.restrict_to_free(again.solve_homogeneous(dofs.expand(b)));
  };
  const BucklingResult warm =
      solve_buckling(model, assembler, k_full, kg_full, options, solve, &result.subspace);
  REQUIRE(warm.iterations < result.iterations);
  REQUIRE(warm.load_factors(0) == Approx(result.load_factors(0)).epsilon(1.0e-9));
}

TEST_CASE("a tensile load case has no positive load factor; crowded spectra are shifted",
          "[buckling]") {
  const Scalar length = 1.0;
  const Scalar h = 0.05;
  const Scalar t = 0.02;
  // Tension only stiffens the column.
  FemModel tension = make_column(make_structured_quad_mesh(box_spec(40, 4, 1, length, h, 1.0)),
                                 length, h * t, t, +1.0e3);
  Assembler assembler(tension);
  BucklingOptions options;
  options.num_modes = 2;
  const BucklingResult result = analyse_buckling(tension, assembler, 0, options);
  REQUIRE(result.no_positive_load_factor);
  REQUIRE(result.load_factors.size() == 0);

  // A transversely loaded cantilever carries tension and compression in
  // mirror image, so the pencil has load factors of both signs and equal
  // size. Under a large axial tension plus a small transverse load, the
  // reversed-load (negative) factors are many and far larger in magnitude
  // than the few positive ones, which is the case the spectral shift is for.
  // Both must reproduce the dense eigensolve.
  for (const Scalar tension : {0.0, 2.0e8}) {
    INFO("axial tension traction " << tension << " Pa");
    FemModel beam(make_structured_quad_mesh(box_spec(60, 6, 1, length, 0.1, 1.0)),
                  IsotropicMaterial(200.0e9, 0.3, 7850.0, "steel"), 0.002,
                  StressState::PlaneStress, IntegrationOptions());
    DisplacementConstraint root;
    Selector box;
    box.kind = SelectorKind::Box;
    box.xmax = 0.0;
    root.region.members.push_back(box);
    root.fix_x = root.fix_y = true;
    beam.constraints().push_back(root);
    LoadCaseSpec load;
    load.name = "tip";
    TractionLoadSpec tip;
    Selector end;
    end.kind = SelectorKind::Box;
    end.xmin = length;
    tip.region.members.push_back(end);
    tip.traction = Vector3(tension, -1.0e6, 0.0);
    load.tractions.push_back(tip);
    beam.load_case_specs().push_back(load);
    beam.finalize();
    Assembler beam_assembler(beam);
    options.num_modes = 3;
    const BucklingResult bending = analyse_buckling(beam, beam_assembler, 0, options);
    REQUIRE(bending.converged);
    StaticAnalysis analysis(beam, beam_assembler);
    const Vector u = analysis.solve_all().front().displacement;
    const Matrix k = Matrix(beam_assembler.reduce_free_free(beam_assembler.assemble_stiffness()));
    const Matrix g = -Matrix(beam_assembler.reduce_free_free(
        assemble_geometric_stiffness(beam, beam_assembler, u)));
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> ges(g, k);
    const Vector mu = ges.eigenvalues().reverse();
    REQUIRE(mu(0) > 0.0);
    REQUIRE(mu(mu.size() - 1) < 0.0);
    int crowding = 0;
    for (Eigen::Index i = 0; i < mu.size(); ++i) crowding += mu(i) < -mu(2) ? 1 : 0;
    INFO("negative load factors larger than the third positive one: " << crowding);
    if (tension > 0.0) {
      REQUIRE(crowding > bending.subspace_size);
      REQUIRE(bending.transformed);
      REQUIRE(bending.sigma > 0.0);
      REQUIRE(bending.sigma < bending.load_factors(0));
    } else {
      REQUIRE_FALSE(bending.transformed);
    }
    // Under the large tension only two positive load factors exist, both
    // above 1e7: the compressed zone of the small transverse load is tiny.
    int positive = 0;
    while (positive < 3 && mu(positive) > 1.0e-10 * mu.cwiseAbs().maxCoeff()) ++positive;
    INFO("dense mu: " << mu.head(4).transpose() << ", reported lambda: "
                       << bending.load_factors.transpose());
    REQUIRE(positive >= 1);
    REQUIRE(bending.load_factors.size() == positive);
    for (int i = 0; i < positive; ++i) {
      REQUIRE(bending.load_factors(i) == Approx(1.0 / mu(i)).epsilon(1.0e-7));
    }
  }
}

TEST_CASE("buckling input errors are reported", "[buckling]") {
  FemModel model = make_column(make_structured_quad_mesh(box_spec(4, 2, 1, 1.0, 0.1, 1.0)), 1.0,
                               0.1 * 0.01, 0.01, -1.0);
  Assembler assembler(model);
  BucklingOptions options;
  options.num_modes = 0;
  REQUIRE_THROWS_WITH(analyse_buckling(model, assembler, 0, options),
                      ContainsSubstring("num_modes"));
  options.num_modes = 1;
  REQUIRE_THROWS_WITH(analyse_buckling(model, assembler, 3, options),
                      ContainsSubstring("load case"));
  REQUIRE_THROWS_AS(assemble_geometric_stiffness(model, assembler, Vector::Zero(3)), ModelError);
  REQUIRE(euler_cantilever_load(200.0e9, 1.0e-8, 1.0) ==
          Approx(3.14159265358979 * 3.14159265358979 * 200.0e9 * 1.0e-8 / 4.0));
  REQUIRE(engesser_cantilever_load(200.0e9, 0.3, 1.0e-8, 1.0e-4, 1.0) <
          euler_cantilever_load(200.0e9, 1.0e-8, 1.0));
}
