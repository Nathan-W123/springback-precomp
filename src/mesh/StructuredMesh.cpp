// Where this fits: see include/sparlab/mesh/StructuredMesh.hpp

#include "sparlab/mesh/StructuredMesh.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"

#include <Eigen/Geometry>

#include <random>
#include <sstream>
#include <utility>

namespace sparlab {
namespace {

void check_counts_2d(const StructuredMeshSpec& spec) {
  if (spec.nx < 1 || spec.ny < 1) {
    std::ostringstream os;
    os << "structured mesh needs at least one element per direction (got nx=" << spec.nx
       << ", ny=" << spec.ny << ")";
    throw ConfigError(os.str());
  }
  if (!(spec.lx > 0.0) || !(spec.ly > 0.0)) {
    std::ostringstream os;
    os << "structured mesh needs positive extents (got lx=" << spec.lx
       << " m, ly=" << spec.ly << " m)";
    throw ConfigError(os.str());
  }
}

void check_counts_3d(const StructuredMeshSpec& spec) {
  check_counts_2d(spec);
  if (spec.nz < 1) {
    std::ostringstream os;
    os << "structured hex mesh needs at least one element along z (got nz=" << spec.nz
       << ")";
    throw ConfigError(os.str());
  }
  if (!(spec.lz > 0.0)) {
    std::ostringstream os;
    os << "structured hex mesh needs a positive extent along z (got lz=" << spec.lz
       << " m)";
    throw ConfigError(os.str());
  }
}

StructuredGridInfo grid_info(const StructuredMeshSpec& spec, bool three_d, bool uniform) {
  StructuredGridInfo info;
  info.nx = spec.nx;
  info.ny = spec.ny;
  info.nz = three_d ? spec.nz : 0;
  info.lx = spec.lx;
  info.ly = spec.ly;
  info.lz = three_d ? spec.lz : 0.0;
  info.uniform = uniform;
  return info;
}

}  // namespace

Mesh make_structured_hex_mesh(const StructuredMeshSpec& spec) {
  check_counts_3d(spec);

  const Index nnx = spec.nx + 1;
  const Index nny = spec.ny + 1;
  const Index nnz = spec.nz + 1;
  const Scalar dx = spec.lx / static_cast<Scalar>(spec.nx);
  const Scalar dy = spec.ly / static_cast<Scalar>(spec.ny);
  const Scalar dz = spec.lz / static_cast<Scalar>(spec.nz);

  Matrix coords(3, nnx * nny * nnz);
  for (Index k = 0; k < nnz; ++k) {
    for (Index j = 0; j < nny; ++j) {
      for (Index i = 0; i < nnx; ++i) {
        const Index n = k * nnx * nny + j * nnx + i;
        coords(0, n) = spec.x0 + static_cast<Scalar>(i) * dx;
        coords(1, n) = spec.y0 + static_cast<Scalar>(j) * dy;
        coords(2, n) = spec.z0 + static_cast<Scalar>(k) * dz;
      }
    }
  }

  std::vector<Index> connectivity;
  connectivity.reserve(static_cast<std::size_t>(spec.nx) * spec.ny * spec.nz * 8);
  const Index layer = nnx * nny;
  for (Index k = 0; k < spec.nz; ++k) {
    for (Index j = 0; j < spec.ny; ++j) {
      for (Index i = 0; i < spec.nx; ++i) {
        const Index n0 = k * layer + j * nnx + i;
        // bottom face, counter-clockwise seen from +z
        connectivity.push_back(n0);
        connectivity.push_back(n0 + 1);
        connectivity.push_back(n0 + nnx + 1);
        connectivity.push_back(n0 + nnx);
        // top face, same order
        connectivity.push_back(n0 + layer);
        connectivity.push_back(n0 + layer + 1);
        connectivity.push_back(n0 + layer + nnx + 1);
        connectivity.push_back(n0 + layer + nnx);
      }
    }
  }

  Mesh mesh(std::move(coords), std::move(connectivity), ElementType::Hex8);
  mesh.set_structured_info(grid_info(spec, true, true));
  mesh.validate();

  log::debug("generated structured Hex8 mesh: ", spec.nx, " x ", spec.ny, " x ", spec.nz,
             " = ", mesh.num_elements(), " elements, ", mesh.num_nodes(),
             " nodes, cell ", dx, " x ", dy, " x ", dz, " m");
  return mesh;
}

Mesh make_perturbed_hex_mesh(const StructuredMeshSpec& spec, Scalar perturbation,
                             unsigned int seed) {
  if (!(perturbation >= 0.0 && perturbation < 0.35)) {
    std::ostringstream os;
    os << "hex mesh perturbation must lie in [0, 0.35) to keep cells valid (got "
       << perturbation << ")";
    throw ConfigError(os.str());
  }
  Mesh mesh = make_structured_hex_mesh(spec);
  if (perturbation == 0.0) return mesh;

  const Index nnx = spec.nx + 1;
  const Index nny = spec.ny + 1;
  const Index nnz = spec.nz + 1;
  const Scalar dx = spec.lx / static_cast<Scalar>(spec.nx);
  const Scalar dy = spec.ly / static_cast<Scalar>(spec.ny);
  const Scalar dz = spec.lz / static_cast<Scalar>(spec.nz);

  Matrix coords = mesh.coordinates();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  for (Index k = 1; k < nnz - 1; ++k) {
    for (Index j = 1; j < nny - 1; ++j) {
      for (Index i = 1; i < nnx - 1; ++i) {
        const Index n = k * nnx * nny + j * nnx + i;
        coords(0, n) += perturbation * dx * dist(rng);
        coords(1, n) += perturbation * dy * dist(rng);
        coords(2, n) += perturbation * dz * dist(rng);
      }
    }
  }

  Mesh perturbed(std::move(coords), mesh.connectivity(), ElementType::Hex8);
  perturbed.set_structured_info(grid_info(spec, true, false));
  perturbed.validate();
  log::debug("perturbed structured hex mesh: ", spec.nx, " x ", spec.ny, " x ", spec.nz,
             " with interior nodes displaced by up to ", perturbation,
             " of the cell size");
  return perturbed;
}

}  // namespace sparlab
