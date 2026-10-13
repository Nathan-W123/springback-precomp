/// \file StructuredMesh.hpp
/// \brief Generators for structured Hex8 grids.
///
/// Node and element numbering (nx, ny, nz = elements along x, y, z):
/// \code
///   node(i, j, k) = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
///   elem(i, j, k) = k * nx * ny + j * nx + i
/// \endcode
/// and each hexahedron lists its bottom face (z = z_k) counter-clockwise seen
/// from +z followed by the top face in the same order, which is the VTK
/// hexahedron convention.
#pragma once

#include "sparlab/mesh/Mesh.hpp"

namespace sparlab {

/// Parameters of a structured box domain.
struct StructuredMeshSpec {
  Index nx = 1;         ///< number of elements along x (>= 1)
  Index ny = 1;         ///< number of elements along y (>= 1)
  Index nz = 1;         ///< number of elements along z (>= 1)
  Scalar lx = 1.0;      ///< domain length along x [m] (> 0)
  Scalar ly = 1.0;      ///< domain length along y [m] (> 0)
  Scalar lz = 1.0;      ///< domain length along z [m] (> 0)
  Scalar x0 = 0.0;      ///< x coordinate of the lower corner [m]
  Scalar y0 = 0.0;      ///< y coordinate of the lower corner [m]
  Scalar z0 = 0.0;      ///< z coordinate of the lower corner [m]
};

/// Build a uniform structured Hex8 mesh. The resulting mesh is validated
/// before being returned and carries StructuredGridInfo so downstream
/// components can exploit the identical-cell property.
/// \throws ConfigError for non-positive counts or lengths.
Mesh make_structured_hex_mesh(const StructuredMeshSpec& spec);

/// Build a structured Hex8 mesh whose *interior* nodes are randomly displaced
/// by up to `perturbation` times the cell size in every direction; boundary
/// nodes stay in place, so the domain shape is preserved.
///
/// Distorted meshes are what make a patch test meaningful: on a uniform grid
/// an element reproduces linear fields for trivial reasons, whereas on a
/// distorted grid only a correctly formed isoparametric mapping does. The
/// generator is deterministic for a given seed.
/// \param perturbation fraction of the cell size, in [0, 0.35).
/// \throws ConfigError for an out-of-range perturbation.
Mesh make_perturbed_hex_mesh(const StructuredMeshSpec& spec, Scalar perturbation,
                             unsigned int seed = 12345u);

}  // namespace sparlab
