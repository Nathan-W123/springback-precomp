/// \file test_dofs.cpp
/// \brief Degrees of freedom per node: the DOF manager's numbering, gather and
///        scatter with rotational DOFs.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/BoundaryConditions.hpp"
#include "sparlab/fem/DofManager.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace sparlab;
using namespace sparlab::testing;

TEST_CASE("the DOF manager numbers six DOFs per node node-major", "[dofs]") {
  DofManager dofs(4, kMaxDofsPerNode);
  REQUIRE(dofs.num_dofs() == 24);
  REQUIRE(dofs.dof(2, 0) == 12);
  REQUIRE(dofs.dof(2, 5) == 17);
  REQUIRE_THROWS_AS(dofs.dof(2, 6), ModelError);
  REQUIRE_THROWS_AS(DofManager(4, 4), ModelError);

  // Gather and scatter are inverse copies in the element's node order.
  Vector full(24);
  for (Index d = 0; d < 24; ++d) full(d) = 0.5 * static_cast<Scalar>(d);
  const Index nodes[2] = {3, 1};
  const Vector local = dofs.gather(nodes, 2, full);
  REQUIRE(local.size() == 12);
  for (int k = 0; k < 6; ++k) {
    REQUIRE(local(k) == full(18 + k));
    REQUIRE(local(6 + k) == full(6 + k));
  }
  Vector sum = Vector::Zero(24);
  dofs.scatter_add(nodes, 2, local, sum, 2.0);
  for (int k = 0; k < 6; ++k) {
    REQUIRE(sum(18 + k) == 2.0 * full(18 + k));
    REQUIRE(sum(6 + k) == 2.0 * full(6 + k));
    REQUIRE(sum(k) == 0.0);
  }

  // Prescribing a rotation partitions it like any other DOF.
  dofs.prescribe(1, 4, 0.01);
  REQUIRE(dofs.is_constrained(dofs.dof(1, 4)));
  REQUIRE(dofs.prescribed_value(dofs.dof(1, 4)) == 0.01);
  REQUIRE(dofs.num_free() == 23);
}

TEST_CASE("displacement constraints carry six components", "[dofs]") {
  DisplacementConstraint bc;
  bc.set(3, true, 0.25);
  bc.set(5, true);
  REQUIRE(bc.fixes(3));
  REQUIRE(bc.value(3) == 0.25);
  REQUIRE(bc.fixes(5));
  REQUIRE_FALSE(bc.fixes(4));
  REQUIRE(bc.fixes_rotation());
  REQUIRE_THROWS_AS(bc.set(6, true), ModelError);
}
