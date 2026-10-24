/// \file test_assembly_solver.cpp
/// \brief Global assembly and Dirichlet partitioning.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include <limits>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;

TEST_CASE("global stiffness is symmetric and annihilates rigid-body modes",
          "[assembly][verification]") {
  StructuredMeshSpec spec;
  spec.nx = 5;
  spec.ny = 4;
  spec.nz = 2;
  spec.lx = 1.2;
  spec.ly = 0.8;
  spec.lz = 0.1;

  FemModel model(make_structured_hex_mesh(spec), default_material(), 1.0,
                 StressState::ThreeDimensional, IntegrationOptions());
  LoadCaseSpec load;
  load.name = "none";
  model.load_case_specs().push_back(load);
  // Deliberately unconstrained so the free-free block is the full matrix.
  model.finalize();

  Assembler assembler(model);
  const SparseMatrix k = assembler.assemble_stiffness();
  REQUIRE(k.rows() == model.dofs().num_dofs());

  const Matrix dense(k);
  const Scalar scale = dense.cwiseAbs().maxCoeff();
  REQUIRE((dense - dense.transpose()).cwiseAbs().maxCoeff() / scale ==
          Approx(0.0).margin(1.0e-14));

  const Matrix modes = rigid_body_modes_3d(model.mesh());
  for (int m = 0; m < 6; ++m) {
    const Vector residual = k * modes.col(m);
    REQUIRE(residual.cwiseAbs().maxCoeff() / (scale * modes.col(m).norm()) ==
            Approx(0.0).margin(1.0e-13));
  }
}

TEST_CASE("the element-matrix cache agrees with per-element integration",
          "[assembly][verification]") {
  // A uniform structured mesh triggers the cache; a perturbed one does not.
  // Assembling the *same* uniform mesh both ways must give identical matrices,
  // which is checked by comparing against a mesh with the uniform flag cleared.
  StructuredMeshSpec spec;
  spec.nx = 6;
  spec.ny = 5;
  spec.nz = 2;
  spec.lx = 1.0;
  spec.ly = 0.6;
  spec.lz = 0.04;

  Mesh uniform = make_structured_hex_mesh(spec);
  Mesh generic = uniform;
  StructuredGridInfo info = *uniform.structured_info();
  info.uniform = false;  // force the generic path
  generic.set_structured_info(info);

  const auto build = [&](Mesh mesh) {
    FemModel model(std::move(mesh), default_material(), 1.0, StressState::ThreeDimensional,
                   IntegrationOptions());
    DisplacementConstraint bc;
    Selector box;
    box.kind = SelectorKind::Box;
    box.xmax = 0.0;
    bc.region.members.push_back(box);
    bc.fix_x = true;
    bc.fix_y = true;
    bc.fix_z = true;
    model.constraints().push_back(bc);
    LoadCaseSpec load;
    load.name = "none";
    model.load_case_specs().push_back(load);
    model.finalize();
    return model;
  };

  FemModel cached_model = build(uniform);
  FemModel generic_model = build(generic);
  Assembler cached(cached_model);
  Assembler plain(generic_model);
  REQUIRE(cached.uses_element_cache());
  REQUIRE_FALSE(plain.uses_element_cache());

  const Matrix a(cached.assemble_stiffness());
  const Matrix b(plain.assemble_stiffness());
  REQUIRE((a - b).cwiseAbs().maxCoeff() / a.cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1.0e-15));

  const Matrix ma(cached.assemble_mass());
  const Matrix mb(plain.assemble_mass());
  REQUIRE((ma - mb).cwiseAbs().maxCoeff() / ma.cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1.0e-15));
}

TEST_CASE("per-element scaling enters the assembly linearly", "[assembly]") {
  FemModel model = make_small_block();
  Assembler assembler(model);
  const SparseMatrix k1 = assembler.assemble_stiffness();
  const Vector scale = Vector::Constant(model.mesh().num_elements(), 0.25);
  const SparseMatrix k2 = assembler.assemble_stiffness(&scale);
  const Matrix diff = Matrix(k2) - 0.25 * Matrix(k1);
  REQUIRE(diff.cwiseAbs().maxCoeff() / Matrix(k1).cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1.0e-15));
}

TEST_CASE("assembly rejects malformed scale vectors", "[assembly][diagnostics]") {
  FemModel model = make_small_block();
  Assembler assembler(model);

  const Vector wrong_length = Vector::Ones(3);
  REQUIRE_THROWS_AS(assembler.assemble_stiffness(&wrong_length), ModelError);

  Vector negative = Vector::Ones(model.mesh().num_elements());
  negative(0) = -1.0;
  REQUIRE_THROWS_AS(assembler.assemble_stiffness(&negative), ModelError);

  Vector nan_scale = Vector::Ones(model.mesh().num_elements());
  nan_scale(1) = std::numeric_limits<Scalar>::quiet_NaN();
  REQUIRE_THROWS_AS(assembler.assemble_stiffness(&nan_scale), ModelError);
}

TEST_CASE("the reduced stiffness matrix is positive definite once constrained",
          "[assembly][verification]") {
  FemModel model = make_small_block();
  Assembler assembler(model);
  const SparseMatrix k = assembler.assemble_stiffness();
  const SparseMatrix kff = assembler.reduce_free_free(k);

  REQUIRE(kff.rows() == model.dofs().num_free());
  // Every eigenvalue of the dense equivalent is strictly positive.
  const Matrix dense_kff(kff);
  Eigen::SelfAdjointEigenSolver<Matrix> es(dense_kff);
  REQUIRE(es.eigenvalues().minCoeff() > 0.0);
}

TEST_CASE("DOF partitioning numbers the free DOFs in ascending order", "[dofs]") {
  FemModel model = make_small_block();
  const DofManager& dofs = model.dofs();
  REQUIRE(dofs.num_free() + dofs.num_constrained() == dofs.num_dofs());

  // The free DOFs number the reduced system in ascending order; a
  // constrained DOF has no position in it.
  Index position = 0;
  for (Index d : dofs.free_dofs()) REQUIRE(dofs.reduced_index(d) == position++);
  for (Index d : dofs.constrained_dofs()) {
    REQUIRE(dofs.is_constrained(d));
    REQUIRE(dofs.reduced_index(d) == -1);
  }

  REQUIRE_THROWS_AS(dofs.reduced_index(dofs.num_dofs()), ModelError);
  REQUIRE_THROWS_AS(dofs.dof(-1, 0), ModelError);
  REQUIRE_THROWS_AS(dofs.dof(0, 3), ModelError);
}
