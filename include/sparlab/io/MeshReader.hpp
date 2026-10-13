/// \file MeshReader.hpp
/// \brief Import of unstructured meshes from Abaqus / CalculiX (.inp) files.
///
/// Geometry that a structured box cannot represent comes from a mesh
/// generator. The reader turns such a file into a `Mesh` plus its named sets,
/// and checks it the way a file from someone else's tool needs checking:
///
///   * **element types.** The cells of the highest dimension present become
///     the mesh; they must be linear hexahedra (C3D8 and its variants).
///     Lower-dimensional elements (boundary lines and facets) are used only to
///     build named node sets. A mix of cell types and other cells are refused
///     with a message that says how to re-export;
///   * **orientation.** Mirrored hexahedra are re-ordered and counted; a cell
///     that is folded rather than mirrored is an error naming the element;
///   * **units.** Coordinates are multiplied by `scale` (0.001 for a CAD file in
///     millimetres), and a domain larger than 20 m or smaller than 0.1 mm draws
///     a warning that names the likely fix;
///   * **unused and duplicate nodes.** Nodes no cell references (geometry
///     points, dropped entities) are removed and counted. Coincident nodes,
///     which leave the cells on either side unconnected, are counted and
///     optionally merged.
///
/// Named sets. An `*NSET` becomes a node set, an `*ELSET` (or the `ELSET=` of
/// an `*ELEMENT` card) an element set when it holds cells and a node set when
/// it holds boundary elements.
///
/// Boundary conditions, loads and materials in an .inp file are not imported
/// - the SparLab deck defines those - and the report lists the ignored
/// keywords so nothing is dropped silently.
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/mesh/Mesh.hpp"

#include <istream>
#include <map>
#include <string>
#include <vector>

namespace sparlab {

struct MeshReadOptions {
  /// "auto" (from the extension: .inp -> abaqus) or "abaqus".
  std::string format = "auto";
  /// Factor applied to every coordinate, e.g. 0.001 for millimetres -> metres.
  Scalar scale = 1.0;
  /// Merge nodes closer than `duplicate_tolerance` instead of only reporting
  /// them.
  bool merge_duplicate_nodes = false;
  /// Absolute distance [m, after scaling] below which two nodes coincide;
  /// <= 0 selects 1e-9 times the bounding-box diagonal.
  Scalar duplicate_tolerance = 0.0;
};

/// What the reader found and what it did about it.
struct MeshReadReport {
  std::string path;
  std::string format;               ///< "abaqus"
  int dimension = 0;
  ElementType element_type = ElementType::Hex8;
  Scalar scale = 1.0;
  Index nodes_in_file = 0;
  Index nodes_used = 0;             ///< after dropping unreferenced nodes
  Index elements_in_file = 0;       ///< every element record, of any dimension
  Index cells = 0;                  ///< elements of the mesh dimension kept as cells
  Index boundary_elements = 0;      ///< lower-dimensional elements used for sets
  Index reoriented = 0;             ///< cells whose node order was reversed
  Index unreferenced_nodes = 0;     ///< nodes dropped because no cell uses them
  Index duplicate_nodes = 0;        ///< nodes coinciding with an earlier node
  bool duplicates_merged = false;
  std::map<std::string, Index> ignored;   ///< e.g. "point elements", "*BOUNDARY cards"
  std::map<std::string, Index> node_sets;     ///< name -> size
  std::map<std::string, Index> element_sets;  ///< name -> size
  MeshQuality quality;
  std::vector<std::string> warnings;
};

/// Read a mesh file. Relative `*INCLUDE` paths in an .inp file resolve against
/// the including file's directory.
/// \throws IoError for an unreadable or malformed file, MeshError for a mesh
///         SparLab cannot use, each with the reason and the likely fix.
Mesh read_mesh_file(const std::string& path, const MeshReadOptions& options = {},
                    MeshReadReport* report = nullptr);

/// Stream version used by the file reader and the tests. `source` names the
/// stream in messages; `base_directory` resolves .inp includes.
Mesh read_abaqus_inp(std::istream& in, const std::string& source,
                     const std::string& base_directory = ".",
                     const MeshReadOptions& options = {},
                     MeshReadReport* report = nullptr);

}  // namespace sparlab
