/// \file test_unstructured.cpp
/// \brief Unstructured meshes read from Abaqus / CalculiX (.inp) files, with
///        their named groups.
///
/// The reader test feeds a file written by hand and compares the result node
/// by node, and it checks that each malformed input fails with a message
/// that says what to change.
#include "TestSupport.hpp"

#include "sparlab/io/CsvWriter.hpp"
#include "sparlab/io/MeshReader.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <fstream>
#include <sstream>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

void write_text(const std::string& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
}

const char* kTmp = "results/_test_tmp/unstructured";

}  // namespace

TEST_CASE("the Abaqus reader handles includes, sets, GENERATE and continuation lines",
          "[unstructured][io][abaqus]") {
  const std::string dir = std::string(kTmp) + "/inp";
  ensure_directory(dir);
  // A 2 x 1 x 1 strip of C3D8 hexahedra, the nodes in an included file, the
  // second element continued on a new line, and a boundary S4R "element" that
  // names the loaded end face.
  write_text(dir + "/nodes.inp",
             "** node file\n"
             "*NODE, NSET=ALLN\n"
             "101, 0.0, 0.0, 0.0\n102, 1.0, 0.0, 0.0\n103, 2.0, 0.0, 0.0\n"
             "104, 0.0, 1.0, 0.0\n105, 1.0, 1.0, 0.0\n106, 2.0, 1.0, 0.0\n"
             "107, 0.0, 0.0, 1.0\n108, 1.0, 0.0, 1.0\n109, 2.0, 0.0, 1.0\n"
             "110, 0.0, 1.0, 1.0\n111, 1.0, 1.0, 1.0\n112, 2.0, 1.0, 1.0\n");
  write_text(dir + "/main.inp",
             "*HEADING\nstrip\n"
             "*INCLUDE, INPUT=nodes.inp\n"
             "*ELEMENT, TYPE=C3D8, ELSET=PLATE\n"
             "1, 101, 102, 105, 104, 107, 108, 111, 110\n"
             "2, 102, 103, 106, 105,\n"
             "   108, 109, 112, 111\n"
             "*ELEMENT, TYPE=S4R, ELSET=TIPFACE\n"
             "10, 103, 106, 112, 109\n"
             "*NSET, NSET=ROOT, GENERATE\n"
             "101, 110, 3\n"
             "*ELSET, ELSET=FIRST\n1\n"
             "*ELSET, ELSET=BOTH\nFIRST, 2\n"
             "*MATERIAL, NAME=STEEL\n*ELASTIC\n210000., 0.3\n"
             "*BOUNDARY\nROOT, 1, 3\n");
  MeshReadReport report;
  const Mesh mesh = read_mesh_file(dir + "/main.inp", MeshReadOptions(), &report);
  REQUIRE(mesh.element_type() == ElementType::Hex8);
  REQUIRE(mesh.num_nodes() == 12);
  REQUIRE(mesh.num_elements() == 2);
  REQUIRE(mesh.element_measure(1) == Approx(1.0));
  REQUIRE(mesh.node_sets().at("ROOT") == std::vector<Index>({0, 3, 6, 9}));
  REQUIRE(mesh.node_sets().at("TIPFACE") == std::vector<Index>({2, 5, 8, 11}));
  REQUIRE(mesh.element_sets().at("PLATE") == std::vector<Index>({0, 1}));
  REQUIRE(mesh.element_sets().at("FIRST") == std::vector<Index>({0}));
  REQUIRE(mesh.element_sets().at("BOTH") == std::vector<Index>({0, 1}));
  REQUIRE(mesh.element_sets().count("TIPFACE") == 0);
  REQUIRE(report.boundary_elements == 1);
  REQUIRE(report.ignored.at("*HEADING cards") == 1);
  REQUIRE(report.ignored.at("*BOUNDARY cards") == 1);

  // The groups are regions of a deck.
  SelectorGroup root;
  Selector group;
  group.kind = SelectorKind::Group;
  group.group = "ROOT";
  root.members.push_back(group);
  REQUIRE(root.select_nodes(mesh) == mesh.node_sets().at("ROOT"));
  REQUIRE_THROWS_WITH(root.select_elements(mesh), ContainsSubstring("node set"));
  SelectorGroup plate;
  group.group = "PLATE";
  plate.members.push_back(group);
  REQUIRE(plate.select_elements(mesh).size() == 2);

  // Keyword and data problems name the line.
  const auto read = [](const std::string& text) {
    std::istringstream in(text);
    return read_abaqus_inp(in, "bad.inp");
  };
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n*ELEMENT\n1, 1\n"), ContainsSubstring("TYPE="));
  REQUIRE_THROWS_WITH(read("1, 0, 0\n"), ContainsSubstring("bad.inp:1"));
  // Cells other than the linear hexahedron are refused with the fix.
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0, 0\n2, 1, 0, 0\n3, 0, 1, 0\n4, 0, 0, 1\n"
                           "*ELEMENT, TYPE=C3D4\n1, 1, 2, 3, 4\n"),
                      ContainsSubstring("8-node hexahedron"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n2, 1, 0\n3, 0, 1\n*ELEMENT, TYPE=C3D20\n"
                           "1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2\n"),
                      ContainsSubstring("8-node hexahedron"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n2, 1, 0\n3, 0, 1\n*ELEMENT, TYPE=C3D6\n"
                           "1, 1, 2, 3, 1, 2, 3\n"),
                      ContainsSubstring("hexahedra"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n*NSET, NSET=A\nB\n"),
                      ContainsSubstring("not defined before it"));
  REQUIRE_THROWS_WITH(read("*PART, NAME=A\n*NODE\n1, 0, 0\n*END PART\n*PART, NAME=B\n"
                           "*NODE\n1, 0, 0\n*END PART\n"),
                      ContainsSubstring("flat"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n*INSTANCE, NAME=I, PART=A\n1.0, 0.0, 0.0\n"),
                      ContainsSubstring("translation"));
  REQUIRE_THROWS_WITH(read("*NODE\n1, 0, 0\n*INCLUDE, INPUT=missing.inp\n"),
                      ContainsSubstring("cannot open included file"));
  REQUIRE_THROWS_WITH(read_mesh_file(dir + "/part.msh"), ContainsSubstring(".inp"));
}
