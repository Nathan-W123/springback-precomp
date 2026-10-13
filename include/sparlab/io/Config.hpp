/// \file Config.hpp
/// \brief Declarative JSON input deck: schema, parsing and model construction.
///
/// An input deck fully determines a run. The schema is documented in
/// docs/forming.md; the section names below mirror it one to one:
/// \code
/// {
///   "name": "spif_smoke",
///   "mesh":     { "type": "structured_hex", "nx","ny","nz", "lx","ly","lz" },
///               // or { "type": "file", "path": "part.inp", "scale": 0.001 }
///   "material": { "youngs_modulus":.., "poisson_ratio":.., "density":..,
///                 "plasticity": {..} },
///   "material_regions": [ { "region": {..}, "material": {..} } ],
///   "model":    { "stress_state": "three_dimensional",
///                 "element_formulation": "incompatible_modes",
///                 "integration": { "thickness_points": 5 } },
///   "boundary_conditions": [ { "fix": ["x","y","z"], "region": {..} } ],
///   "forming":  { "tools": [..], "steps": [..] }   // sparlab_form (docs/forming.md)
/// }
/// \endcode
///
/// The mesh is solid (Hex8 cells): vectors take three entries. A mesh file
/// (Abaqus/CalculiX .inp) is read while the deck is parsed; its `path` is
/// relative to the deck's own directory, and its *NSET / *ELSET cards become
/// regions through `{"group": "<name>"}`.
///
/// Every numeric field is in SI units. Unknown keys are reported (not ignored),
/// because a misspelled key that silently takes its default is a direct route
/// to a wrong published number.
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/fem/FemModel.hpp"
#include "sparlab/fem/Forming.hpp"
#include "sparlab/io/Json.hpp"
#include "sparlab/io/MeshReader.hpp"
#include "sparlab/mesh/StructuredMesh.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sparlab {

/// Mesh sources a deck can name.
enum class MeshKind {
  StructuredHex,   ///< "structured_hex": 3-D Hex8 grid
  File             ///< "file": an Abaqus/CalculiX (.inp) mesh of Hex8 cells
};

std::string to_string(MeshKind kind);

/// True for the generated box mesh, whose resolution a deck sets; false for a
/// mesh read from a file, whose resolution the mesher set.
bool is_structured(MeshKind kind);

/// A mesh read from a file.
struct MeshFileConfig {
  std::string path;           ///< as written in the deck
  std::string resolved_path;  ///< resolved against the deck's directory
  MeshReadOptions read;
};

/// The incremental-forming analysis of sparlab_form (Forming.hpp,
/// docs/forming.md): its tools, steps and Newton settings, and what it
/// writes.
struct FormingConfig {
  bool enabled = false;  ///< the deck has a `forming` block
  FormingOptions options;
  /// Per-step VTK files.
  bool write_vtk = true;
  /// Per-step node and element files ("steps", the default), none of them
  /// ("none"), or those plus a snapshot every `snapshot_stride` increments
  /// (an integer in the deck).
  enum class Snapshots { Steps, None, Stride };
  Snapshots snapshots = Snapshots::Steps;
  int snapshot_stride = 0;
};

/// A material assigned to an element region, overriding the deck's primary
/// `material` there (later regions win).
struct MaterialRegion {
  std::string name;
  SelectorGroup region;
  IsotropicMaterial material;
};

/// A complete, validated run specification.
class Configuration {
 public:
  std::string name = "case";
  std::string description;
  std::string source_path;

  MeshKind mesh_kind = MeshKind::StructuredHex;
  StructuredMeshSpec mesh_spec;   ///< MeshKind::StructuredHex
  MeshFileConfig mesh_file;       ///< MeshKind::File
  /// The mesh read from `mesh_file` while parsing (shared by copies of the
  /// configuration), and what the reader found.
  std::shared_ptr<const Mesh> file_mesh;
  MeshReadReport mesh_report;
  Scalar thickness = 1.0;
  StressState stress_state = StressState::PlaneStress;
  IntegrationOptions integration;

  std::vector<DisplacementConstraint> constraints;
  std::vector<LoadCaseSpec> load_cases;
  std::vector<MaterialRegion> material_regions;

  FormingConfig forming;

  /// The parsed document, retained so run summaries can record exactly what
  /// was requested.
  json::Value document;

  /// Spatial dimension implied by the mesh (3).
  int dim() const;

  /// One-line description of the mesh source for messages and summaries,
  /// e.g. "structured_hex 40 x 40 x 2" or "file 'mesh.inp' (Hex8, 9216 elements)".
  std::string describe_mesh() const;

  const IsotropicMaterial& material() const;
  void set_material(IsotropicMaterial material);

 private:
  std::optional<IsotropicMaterial> material_;
};

/// Parse a configuration from an already-parsed document.
/// \param source name used in error messages.
/// \param strict when true, unknown keys raise ConfigError instead of a warning.
/// \param base_directory directory a relative mesh-file path is resolved
///        against; empty means the current working directory.
Configuration parse_configuration(const json::Value& document, const std::string& source,
                                  bool strict = false,
                                  const std::string& base_directory = "");

/// Read and parse a configuration file; a relative mesh-file path is resolved
/// against the file's own directory.
Configuration load_configuration(const std::string& path, bool strict = false);

/// Build the mesh described by a configuration (without the model).
Mesh build_mesh(const Configuration& config);

/// Build the mesh, material, boundary conditions and load cases described by a
/// configuration and finalise the model.
FemModel build_model(const Configuration& config);

/// Parse a region specification (used by tests and by the config parser).
/// \param dim mesh dimension (2 or 3), which decides how many entries the
///        vector-valued fields must carry; 0 accepts either form.
SelectorGroup parse_region(const ConfigNode& node, const std::string& default_name,
                           int dim = 0);

}  // namespace sparlab
