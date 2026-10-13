/// \file TestSupport.hpp
/// \brief Shared fixtures for the SparLab test suite.
#pragma once

#include "sparlab/fem/Assembler.hpp"
#include "sparlab/fem/FemModel.hpp"
#include "sparlab/mesh/StructuredMesh.hpp"

#include <Eigen/Geometry>

#include <cmath>

namespace sparlab {
namespace testing {

/// Reference aluminium-like material used by most tests.
inline IsotropicMaterial default_material(Scalar poisson = 0.3) {
  return IsotropicMaterial(70.0e9, poisson, 2700.0, "test_aluminium");
}

/// A small Hex8 block (0.4 x 0.3 x 0.05 m, nx x ny x 1 cells) clamped on
/// x = 0, with its one empty load case, finalised.
inline FemModel make_small_block(Index nx = 4, Index ny = 3, Scalar poisson = 0.3) {
  StructuredMeshSpec spec;
  spec.nx = nx;
  spec.ny = ny;
  spec.nz = 1;
  spec.lx = 0.4;
  spec.ly = 0.3;
  spec.lz = 0.05;

  FemModel model(make_structured_hex_mesh(spec), default_material(poisson), 1.0,
                 StressState::ThreeDimensional, IntegrationOptions());

  DisplacementConstraint bc;
  bc.region.name = "left";
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
}

/// The six infinitesimal rigid-body modes of a 3-D node set about its
/// centroid. Columns: three translations, then rotations about x, y, z.
inline Matrix rigid_body_modes_3d(const Mesh& mesh) {
  const Index nn = mesh.num_nodes();
  Vector3 centroid = Vector3::Zero();
  for (Index n = 0; n < nn; ++n) centroid += mesh.node(n);
  centroid /= static_cast<Scalar>(nn);

  Matrix modes = Matrix::Zero(nn * 3, 6);
  for (Index n = 0; n < nn; ++n) {
    const Vector3 r = mesh.node(n) - centroid;
    for (int k = 0; k < 3; ++k) modes(n * 3 + k, k) = 1.0;
    for (int j = 0; j < 3; ++j) {
      Vector3 omega = Vector3::Zero();
      omega(j) = 1.0;
      const Vector3 u = omega.cross(r);
      for (int k = 0; k < 3; ++k) modes(n * 3 + k, 3 + j) = u(k);
    }
  }
  return modes;
}

/// Nodal coordinates of a single a x b x c box Hex8 in VTK order.
inline Matrix unit_box_coords(Scalar a = 1.0, Scalar b = 1.0, Scalar c = 1.0) {
  Matrix coords(3, 8);
  coords << 0.0, a, a, 0.0, 0.0, a, a, 0.0,
            0.0, 0.0, b, b, 0.0, 0.0, b, b,
            0.0, 0.0, 0.0, 0.0, c, c, c, c;
  return coords;
}

/// A distorted but valid hexahedron for isoparametric-mapping tests.
inline Matrix distorted_hex_coords() {
  Matrix coords(3, 8);
  coords << 0.05, 1.10, 1.20, -0.05, 0.10, 0.95, 1.05, 0.00,
            0.00, 0.10, 1.15, 0.90, 0.05, -0.05, 1.05, 1.10,
            0.00, -0.05, 0.10, 0.05, 0.90, 1.05, 1.20, 1.00;
  return coords;
}

}  // namespace testing
}  // namespace sparlab
