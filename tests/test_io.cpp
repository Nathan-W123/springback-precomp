/// \file test_io.cpp
/// \brief JSON parsing/serialisation, configuration validation and writers.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Timer.hpp"
#include "sparlab/io/CalculixWriter.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/CsvWriter.hpp"
#include "sparlab/io/Json.hpp"
#include "sparlab/io/ResultWriter.hpp"
#include "sparlab/io/VtkWriter.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;

namespace {

/// A minimal but complete deck used by the configuration tests.
std::string minimal_deck() {
  return R"({
    "name": "unit_case",
    "description": "a deck used by the test suite",
    "mesh": { "type": "structured_quad", "nx": 8, "ny": 4, "lx": 0.8, "ly": 0.4 },
    "material": { "name": "al", "youngs_modulus": 70e9, "poisson_ratio": 0.33,
                  "density": 2700 },
    "model": { "thickness": 0.01, "stress_state": "plane_stress" },
    "boundary_conditions": [
      { "name": "root", "fix": ["x", "y"], "region": { "box": { "xmax": 0.0 } } }
    ],
    "load_cases": [
      { "name": "tip", "weight": 1.0,
        "point_loads": [
          { "name": "tip_edge", "force": [0.0, -1000.0], "distribution": "total",
            "region": { "box": { "xmin": 0.8 } } }
        ]
      }
    ]
  })";
}

std::string temp_path(const std::string& name) {
  return path_join("results/_test_tmp", name);
}

}  // namespace

TEST_CASE("JSON parser handles the full value grammar", "[io][json]") {
  const std::string text = R"({
    // a line comment
    "string": "with \"escapes\" and \n newline and é",
    /* a block comment */
    "number": -1.25e-3,
    "integer": 42,
    "yes": true,
    "no": false,
    "nothing": null,
    "array": [1, 2, [3, 4], {"nested": "value"}],
    "object": { "a": 1, "b": 2 },
    "trailing": [1, 2, 3,],
  })";
  const json::Value doc = json::parse(text, "test");
  REQUIRE(doc.is_object());
  REQUIRE(doc.find("string")->string_value() ==
          std::string("with \"escapes\" and \n newline and \xc3\xa9"));
  REQUIRE(doc.find("number")->number_value() == Approx(-1.25e-3));
  REQUIRE(doc.find("integer")->number_value() == Approx(42.0));
  REQUIRE(doc.find("yes")->bool_value());
  REQUIRE_FALSE(doc.find("no")->bool_value());
  REQUIRE(doc.find("nothing")->is_null());
  REQUIRE(doc.find("array")->array_items().size() == 4);
  REQUIRE(doc.find("array")->array_items()[2].array_items().size() == 2);
  REQUIRE(doc.find("object")->find("b")->number_value() == Approx(2.0));
  REQUIRE(doc.find("trailing")->array_items().size() == 3);
  REQUIRE(doc.find("absent") == nullptr);
}

TEST_CASE("JSON syntax errors report the location", "[io][json][diagnostics]") {
  const auto message = [](const std::string& text) {
    try {
      json::parse(text, "deck.json");
    } catch (const ConfigError& e) {
      return std::string(e.what());
    }
    return std::string("no error");
  };

  REQUIRE(message("{\"a\": }").find("deck.json:1:") != std::string::npos);
  REQUIRE_THROWS_AS(json::parse("{\"a\": 1", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse("{a: 1}", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse("[1, 2", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse("{\"a\": tru}", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse("{\"a\": \"unterminated}", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse("{\"a\": 1} extra", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse("{\"a\": 1, \"a\": 2}", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse("{\"a\": /* unterminated", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse("", "x"), ConfigError);
  REQUIRE_THROWS_AS(json::parse_file("/definitely/not/here.json"), IoError);

  // A multi-line document reports the right line number.
  const std::string multi = "{\n  \"a\": 1,\n  \"b\": @\n}";
  REQUIRE(message(multi).find(":3:") != std::string::npos);
}

TEST_CASE("JSON serialisation round-trips", "[io][json]") {
  json::Value doc = json::Value::make_object();
  doc.set("name", json::Value::make_string("round\"trip\n"));
  doc.set("value", json::Value::make_number(1.0 / 3.0));
  doc.set("flag", json::Value::make_bool(true));
  doc.set("empty_object", json::Value::make_object());
  doc.set("empty_array", json::Value::make_array());
  Vector v(3);
  v << 1.5, -2.5, 1.0e-17;
  doc.set("vector", json::array_of(v));
  doc.set("names", json::array_of(std::vector<std::string>{"a", "b"}));

  const std::string dumped = json::dump(doc, 2);
  const json::Value again = json::parse(dumped, "roundtrip");
  REQUIRE(again.find("name")->string_value() == doc.find("name")->string_value());
  REQUIRE(again.find("value")->number_value() == doc.find("value")->number_value());
  REQUIRE(again.find("vector")->array_items()[2].number_value() == Approx(1.0e-17));
  REQUIRE(again.find("empty_array")->array_items().empty());

  // Compact form parses too.
  REQUIRE_NOTHROW(json::parse(json::dump(doc, 0), "compact"));

  // Non-finite numbers become null rather than invalid JSON.
  json::Value bad = json::Value::make_object();
  bad.set("inf", json::Value::make_number(std::numeric_limits<double>::infinity()));
  REQUIRE(json::dump(bad, 0).find("null") != std::string::npos);
  REQUIRE_NOTHROW(json::parse(json::dump(bad, 0), "inf"));

  // set() overwrites in place and preserves ordering.
  doc.set("flag", json::Value::make_bool(false));
  REQUIRE_FALSE(doc.find("flag")->bool_value());
}

TEST_CASE("ConfigNode reports paths, types and unread keys", "[io][config]") {
  const json::Value doc = json::parse(R"({
    "a": { "b": 3.5, "c": "text", "d": true, "v": [1.0, 2.0] },
    "list": [ {"x": 1}, {"x": 2} ],
    "typo_key": 1
  })",
                                      "t");
  const ConfigNode root(doc);

  REQUIRE(root.child("a").child("b").number() == Approx(3.5));
  REQUIRE(root.child("a").string_or("c", "") == "text");
  REQUIRE(root.child("a").boolean_or("d", false));
  REQUIRE(root.child("a").vector2_or("v", Vector2::Zero()).isApprox(Vector2(1.0, 2.0)));
  REQUIRE(root.child("a").number_or("missing", 7.0) == Approx(7.0));
  REQUIRE(root.array("list").size() == 2);
  REQUIRE(root.array("list")[1].child("x").integer() == 2);
  REQUIRE(root.array("missing").empty());

  // Type errors name the path.
  try {
    root.child("a").child("c").number();
    FAIL("expected a ConfigError");
  } catch (const ConfigError& e) {
    REQUIRE(std::string(e.what()).find("a.c") != std::string::npos);
    REQUIRE(std::string(e.what()).find("number") != std::string::npos);
  }
  REQUIRE_THROWS_AS(root.require("nope"), ConfigError);
  REQUIRE_THROWS_AS(root.child("a").child("b").string(), ConfigError);
  REQUIRE_THROWS_AS(root.child("a").child("b").boolean(), ConfigError);
  REQUIRE_THROWS_AS(root.child("a").child("c").vector2(), ConfigError);
  REQUIRE_THROWS_AS(root.child("a").child("b").integer(), ConfigError);
  REQUIRE_THROWS_AS(root.child("list").child("x"), ConfigError);

  // positive_number and bounded_number validate ranges.
  REQUIRE(root.child("a").positive_number("b") == Approx(3.5));
  REQUIRE_THROWS_AS(root.child("a").bounded_number("b", 0.0, 1.0), ConfigError);

  // "typo_key" was never read.
  const std::vector<std::string> unused = root.unused_keys();
  REQUIRE(std::find(unused.begin(), unused.end(), "typo_key") != unused.end());
  REQUIRE(std::find(unused.begin(), unused.end(), "a.b") == unused.end());
}

TEST_CASE("a minimal deck parses into a solvable model", "[io][config]") {
  const json::Value doc = json::parse(minimal_deck(), "minimal");
  const Configuration config = parse_configuration(doc, "minimal", /*strict=*/true);

  REQUIRE(config.name == "unit_case");
  REQUIRE(config.mesh_spec.nx == 8);
  REQUIRE(config.mesh_spec.ly == Approx(0.4));
  REQUIRE(config.material().youngs_modulus() == Approx(70.0e9));
  REQUIRE(config.material().poisson_ratio() == Approx(0.33));
  REQUIRE(config.thickness == Approx(0.01));
  REQUIRE(config.stress_state == StressState::PlaneStress);
  REQUIRE(config.constraints.size() == 1);
  REQUIRE(config.constraints.front().fix_x);
  REQUIRE(config.constraints.front().fix_y);
  REQUIRE(config.load_cases.size() == 1);
  REQUIRE(config.load_cases.front().point_loads.size() == 1);
  REQUIRE(config.load_cases.front().point_loads.front().force.y() == Approx(-1000.0));
  REQUIRE(config.load_cases.front().point_loads.front().distribute_total);
  // Defaults.
  REQUIRE(config.integration.stiffness_points == 2);
  REQUIRE(config.integration.mass_points == 3);
  // "auto": exact Cholesky below the size limits, multigrid CG above.
  REQUIRE(config.analysis.linear.type == LinearSolverType::Auto);
  REQUIRE(config.modal.options.linear.type == LinearSolverType::Auto);
  REQUIRE_FALSE(config.modal.enabled);

  FemModel model = build_model(config);
  REQUIRE(model.mesh().num_elements() == 32);
  REQUIRE(model.finalized());
  Assembler assembler(model);
  StaticAnalysis analysis(model, assembler, config.analysis);
  const StaticSolution sol = analysis.solve_all().front();
  REQUIRE(sol.compliance > 0.0);
  REQUIRE(sol.equilibrium.relative_force_error == Approx(0.0).margin(1.0e-10));
}

TEST_CASE("configuration errors are specific and actionable",
          "[io][config][diagnostics]") {
  const auto parse_modified = [](const std::string& body) {
    const json::Value doc = json::parse(body, "bad");
    return parse_configuration(doc, "bad");
  };

  REQUIRE_THROWS_AS(parse_modified("[1, 2]"), ConfigError);
  REQUIRE_THROWS_AS(parse_modified("{}"), ConfigError);

  // Missing mesh entries.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"nx": 2, "ny": 2, "ly": 1.0},
                         "material": {"youngs_modulus": 1, "poisson_ratio": 0.3},
                         "boundary_conditions": [], "load_cases": []})"),
      ConfigError);

  // Unknown mesh type.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"type": "tri3", "nx": 2, "ny": 2, "lx": 1, "ly": 1},
                         "material": {"youngs_modulus": 1, "poisson_ratio": 0.3},
                         "boundary_conditions": [], "load_cases": []})"),
      ConfigError);

  // Empty boundary conditions.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"nx": 2, "ny": 2, "lx": 1, "ly": 1},
                         "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},
                         "boundary_conditions": [],
                         "load_cases": [{"point_loads": [
                            {"force": [1,0], "region": {"all": true}}]}]})"),
      ConfigError);

  // A load case with no loads.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"nx": 2, "ny": 2, "lx": 1, "ly": 1},
                         "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},
                         "boundary_conditions": [
                            {"fix": ["x","y"], "region": {"box": {"xmax": 0}}}],
                         "load_cases": [{"name": "empty"}]})"),
      ConfigError);

  // An invalid fix component.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"nx": 2, "ny": 2, "lx": 1, "ly": 1},
                         "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},
                         "boundary_conditions": [
                            {"fix": ["z"], "region": {"box": {"xmax": 0}}}],
                         "load_cases": [{"point_loads": [
                            {"force": [1,0], "region": {"all": true}}]}]})"),
      ConfigError);

  // A region naming two primitives at once.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"nx": 2, "ny": 2, "lx": 1, "ly": 1},
                         "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},
                         "boundary_conditions": [
                            {"fix": ["x"], "region": {"box": {"xmax": 0},
                                                      "circle": {"center": [0,0],
                                                                 "radius": 1}}}],
                         "load_cases": [{"point_loads": [
                            {"force": [1,0], "region": {"all": true}}]}]})"),
      ConfigError);

  // A region naming no primitive.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"nx": 2, "ny": 2, "lx": 1, "ly": 1},
                         "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},
                         "boundary_conditions": [
                            {"fix": ["x"], "region": {"name": "empty"}}],
                         "load_cases": [{"point_loads": [
                            {"force": [1,0], "region": {"all": true}}]}]})"),
      ConfigError);

  // Modal analysis without a density.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"nx": 2, "ny": 2, "lx": 1, "ly": 1},
                         "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},
                         "boundary_conditions": [
                            {"fix": ["x","y"], "region": {"box": {"xmax": 0}}}],
                         "load_cases": [{"point_loads": [
                            {"force": [1,0], "region": {"all": true}}]}],
                         "modal": {"enabled": true}})"),
      ConfigError);

  // An empty BC region selects no nodes: caught at finalize().
  {
    const json::Value doc = json::parse(R"({
      "mesh": {"nx": 2, "ny": 2, "lx": 1, "ly": 1},
      "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3, "density": 1},
      "boundary_conditions": [
        {"name": "nowhere", "fix": ["x","y"],
         "region": {"box": {"xmin": 100.0}}}],
      "load_cases": [{"point_loads": [
        {"force": [1,0], "region": {"all": true}}]}]
    })",
                                        "empty_bc");
    const Configuration config = parse_configuration(doc, "empty_bc");
    REQUIRE_THROWS_AS(build_model(config), ConfigError);
  }
}

TEST_CASE("unknown configuration keys are reported", "[io][config][diagnostics]") {
  const std::string deck = R"({
    "name": "typo_case",
    "mesh": { "nx": 4, "ny": 2, "lx": 1.0, "ly": 0.5 },
    "material": { "youngs_modulus": 1e9, "poisson_ratio": 0.3, "density": 1000 },
    "model": { "thicknes": 0.01 },
    "boundary_conditions": [
      { "fix": ["x","y"], "region": { "box": { "xmax": 0.0 } } }
    ],
    "load_cases": [
      { "point_loads": [ { "force": [0,-1], "region": { "all": true } } ] }
    ]
  })";
  const json::Value doc = json::parse(deck, "typo");
  // In strict mode the typo is an error; the misspelled key means `thickness`
  // silently takes its default otherwise.
  REQUIRE_THROWS_AS(parse_configuration(doc, "typo", /*strict=*/true), ConfigError);
  // Non-strict mode warns and continues with the default thickness.
  const Configuration config = parse_configuration(doc, "typo", /*strict=*/false);
  REQUIRE(config.thickness == Approx(1.0));
}

TEST_CASE("CSV and VTK writers produce well-formed files", "[io][writers]") {
  ensure_directory("results/_test_tmp");

  SECTION("CSV enforces the declared column count") {
    const std::string path = temp_path("rows.csv");
    {
      CsvWriter csv(path, {"i", "a[m]", "b[N]"}, 9);
      csv.row(0, {1.5, -2.5});
      csv.row({3.0, 4.0, 5.0});
      csv.raw_row({"x", "y", "z"});
      REQUIRE_THROWS_AS(csv.row({1.0}), IoError);
      REQUIRE_THROWS_AS(csv.row(1, {1.0}), IoError);
      REQUIRE_THROWS_AS(csv.raw_row({"only_one"}), IoError);
      csv.close();
    }
    std::ifstream in(path);
    REQUIRE(in.good());
    std::string line;
    std::getline(in, line);
    REQUIRE(line == "i,a[m],b[N]");
    std::getline(in, line);
    REQUIRE(line == "0,1.5,-2.5");
    std::getline(in, line);
    REQUIRE(line == "3,4,5");
    std::getline(in, line);
    REQUIRE(line == "x,y,z");
    std::remove(path.c_str());
  }

  SECTION("CSV reports an unwritable path") {
    REQUIRE_THROWS_AS(CsvWriter("/definitely/not/a/dir/x.csv", {"a"}), IoError);
  }

  SECTION("VTK writes the expected header and data blocks") {
    FemModel model = make_small_plate(3, 2);
    Assembler assembler(model);
    StaticAnalysis analysis(model, assembler, StaticAnalysisOptions());
    const Vector u = analysis.solve_all().front().displacement;

    const std::string path = temp_path("fields.vtk");
    VtkWriter writer(model.mesh(), "unit test");
    writer.add_point_vectors("displacement", u);
    writer.add_point_scalars("dummy_point", Vector::Ones(model.mesh().num_nodes()));
    writer.add_cell_scalars("density", Vector::Ones(model.mesh().num_elements()));
    writer.write(path);

    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string content = buffer.str();
    REQUIRE(content.find("# vtk DataFile Version 3.0") == 0);
    REQUIRE(content.find("DATASET UNSTRUCTURED_GRID") != std::string::npos);
    REQUIRE(content.find("POINTS 12 double") != std::string::npos);
    REQUIRE(content.find("CELLS 6 30") != std::string::npos);
    REQUIRE(content.find("CELL_TYPES 6") != std::string::npos);
    REQUIRE(content.find("VECTORS displacement double") != std::string::npos);
    REQUIRE(content.find("SCALARS density double 1") != std::string::npos);
    std::remove(path.c_str());

    // Length mismatches are rejected.
    VtkWriter bad(model.mesh());
    REQUIRE_THROWS_AS(bad.add_point_scalars("x", Vector::Ones(3)), IoError);
    REQUIRE_THROWS_AS(bad.add_cell_scalars("y", Vector::Ones(3)), IoError);
    REQUIRE_THROWS_AS(bad.add_point_vectors("z", Vector::Ones(3)), IoError);
    REQUIRE_THROWS_AS(bad.write("/definitely/not/a/dir/x.vtk"), IoError);
  }

  SECTION("directory creation is recursive and idempotent") {
    const std::string nested = "results/_test_tmp/a/b/c";
    REQUIRE_NOTHROW(ensure_directory(nested));
    REQUIRE_NOTHROW(ensure_directory(nested));
    // A path that exists as a file cannot become a directory.
    const std::string file = temp_path("not_a_dir");
    std::ofstream(file) << "x";
    REQUIRE_THROWS_AS(ensure_directory(file), IoError);
    std::remove(file.c_str());
  }
}

TEST_CASE("CalculiX decks are written per load case with the matching element",
          "[io][writers][cross-validation]") {
  ensure_directory("results/_test_tmp");
  FemModel model = make_small_plate(3, 2);
  const std::vector<std::string> decks =
      write_calculix_decks(model, "results/_test_tmp/ccx", "unit");
  REQUIRE(decks.size() == 1);
  REQUIRE(calculix_element_type(model) == "CPS4");
  std::ifstream in(decks.front());
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  REQUIRE(text.find("*ELEMENT, TYPE=CPS4, ELSET=EALL") != std::string::npos);
  REQUIRE(text.find("*NODE, NSET=NALL\n1, 0, 0, 0\n") != std::string::npos);
  // The thickness follows the section header; it is written with full
  // precision, so parse it back rather than matching its text.
  const std::string section_header = "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1\n";
  const std::size_t section_at = text.find(section_header);
  REQUIRE(section_at != std::string::npos);
  const std::size_t thickness_at = section_at + section_header.size();
  const std::size_t thickness_end = text.find('\n', thickness_at);
  REQUIRE(thickness_end != std::string::npos);
  const double thickness = std::stod(text.substr(thickness_at, thickness_end - thickness_at));
  REQUIRE(thickness == Catch::Approx(0.005).epsilon(1e-15));
  REQUIRE(text.find("*BOUNDARY") != std::string::npos);
  REQUIRE(text.find("*CLOAD") != std::string::npos);
  REQUIRE(text.find("*END STEP") != std::string::npos);
  // One *BOUNDARY line per prescribed DOF, one *CLOAD line per non-zero force.
  std::size_t boundary_lines = 0;
  std::size_t cload_lines = 0;
  std::istringstream lines(text);
  std::string line;
  int section = 0;
  while (std::getline(lines, line)) {
    if (line.rfind("*BOUNDARY", 0) == 0) { section = 1; continue; }
    if (line.rfind("*CLOAD", 0) == 0) { section = 2; continue; }
    if (line.rfind("*", 0) == 0) { section = 0; continue; }
    if (section == 1) ++boundary_lines;
    if (section == 2) ++cload_lines;
  }
  REQUIRE(boundary_lines == static_cast<std::size_t>(model.dofs().num_constrained()));
  REQUIRE(cload_lines == 2);  // fx and fy at the loaded corner node
  std::remove(decks.front().c_str());

  StructuredMeshSpec spec;
  spec.nx = spec.ny = spec.nz = 2;
  FemModel solid(make_structured_hex_mesh(spec), default_material(), 1.0,
                 StressState::ThreeDimensional, IntegrationOptions());
  REQUIRE(calculix_element_type(solid) == "C3D8");
  REQUIRE_THROWS_AS(write_calculix_decks(solid, "results/_test_tmp/ccx3", "x"), IoError);
}

namespace {

std::string read_text(const std::string& path) {
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

/// Lines of `text` after the card `card` up to the next card.
std::vector<std::string> card_lines(const std::string& text, const std::string& card) {
  std::vector<std::string> out;
  std::istringstream lines(text);
  std::string line;
  bool inside = false;
  while (std::getline(lines, line)) {
    if (line.rfind("*", 0) == 0) {
      inside = line == card;
      continue;
    }
    if (inside) out.push_back(line);
  }
  return out;
}

}  // namespace

TEST_CASE("CalculiX decks carry pressure, body, thermal and conduction loads natively",
          "[io][writers][cross-validation]") {
  ensure_directory("results/_test_tmp");
  StructuredMeshSpec spec;
  spec.nx = 2;
  spec.ny = 2;
  spec.nz = 2;
  IsotropicMaterial steel(200.0e9, 0.3, 7850.0, "steel");
  steel.set_thermal(1.2e-5, 293.15, 45.0);
  FemModel model(make_structured_hex_mesh(spec), steel, 1.0, StressState::ThreeDimensional,
                 IntegrationOptions());
  IsotropicMaterial aluminium(70.0e9, 0.33, 2700.0, "aluminium");
  aluminium.set_thermal(2.3e-5, 293.15, 167.0);
  model.assign_material(aluminium, {0, 1});
  DisplacementConstraint root;
  Selector x0;
  x0.kind = SelectorKind::Box;
  x0.xmax = 0.0;
  root.region.members.push_back(x0);
  root.fix_x = root.fix_y = root.fix_z = true;
  model.constraints().push_back(root);

  LoadCaseSpec loads;
  loads.name = "mixed";
  PressureLoadSpec top;
  Selector z1;
  z1.kind = SelectorKind::Box;
  z1.zmin = 1.0;
  top.region.members.push_back(z1);
  top.pressure = 1.0e5;
  loads.pressures.push_back(top);
  PressureLoadSpec bottom = top;
  Selector z0;
  z0.kind = SelectorKind::Box;
  z0.zmax = 0.0;
  bottom.region.members = {z0};
  loads.pressures.push_back(bottom);
  loads.gravity = Vector3(0.0, 0.0, -9.81);
  loads.centrifugal.enabled = true;
  loads.centrifugal.angular_velocity = 10.0;
  loads.centrifugal.axis = Vector3(0.0, 0.0, 2.0);
  loads.temperature.source = TemperatureSpec::Source::Conduction;
  RegionValue hot;
  hot.region.members.push_back(x0);
  hot.value = 373.15;
  loads.temperature.conduction.prescribed.push_back(hot);
  ConvectionSpec film;
  film.region.members.push_back(z1);
  film.film_coefficient = 25.0;
  film.ambient = 293.15;
  loads.temperature.conduction.convection.push_back(film);
  model.load_case_specs().push_back(loads);
  model.finalize();

  const std::vector<std::string> decks =
      write_calculix_decks(model, "results/_test_tmp/native", "unit");
  REQUIRE(decks.size() == 2);
  REQUIRE(decks[1].find("_conduction.inp") != std::string::npos);

  const std::string text = read_text(decks[0]);
  // One material and section per material, on element sets.
  REQUIRE(text.find("*ELSET, ELSET=M1") != std::string::npos);
  REQUIRE(text.find("*ELSET, ELSET=M2\n1, 2\n") != std::string::npos);
  REQUIRE(text.find("*SOLID SECTION, ELSET=M2, MATERIAL=MAT2") != std::string::npos);
  REQUIRE(text.find("*DENSITY\n7850\n") != std::string::npos);
  REQUIRE(text.find("*EXPANSION, ZERO=293.14999999999998\n2.3e-05\n") != std::string::npos);
  REQUIRE(text.find("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 293.14999999999998") !=
          std::string::npos);
  // The pressure goes out as face loads: the top faces (z = 1) are CalculiX's
  // face 2 of a C3D8, the bottom faces its face 1, four of each; no nodal
  // force remains.
  const std::vector<std::string> dload = card_lines(text, "*DLOAD");
  int p1 = 0;
  int p2 = 0;
  for (const std::string& line : dload) {
    if (line.find(", P1, ") != std::string::npos) ++p1;
    if (line.find(", P2, ") != std::string::npos) ++p2;
  }
  REQUIRE(p1 == 4);
  REQUIRE(p2 == 4);
  REQUIRE(text.find("*CLOAD") == std::string::npos);
  REQUIRE(text.find("EALL, GRAV, 9.8100000000000005, 0, 0, -1\n") != std::string::npos);
  REQUIRE(text.find("EALL, CENTRIF, 100, 0, 0, 0, 0, 0, 1\n") != std::string::npos);
  REQUIRE(card_lines(text, "*TEMPERATURE").size() ==
          static_cast<std::size_t>(model.mesh().num_nodes()));

  const std::string heat = read_text(decks[1]);
  REQUIRE(heat.find("*ELEMENT, TYPE=DC3D8, ELSET=EALL") != std::string::npos);
  REQUIRE(heat.find("*HEAT TRANSFER, STEADY STATE") != std::string::npos);
  REQUIRE(heat.find("*CONDUCTIVITY\n167\n") != std::string::npos);
  // Nine nodes on x = 0 held on DOF 11; four convecting faces, CalculiX's F2.
  REQUIRE(card_lines(heat, "*BOUNDARY").size() == 9);
  for (const std::string& line : card_lines(heat, "*BOUNDARY")) {
    REQUIRE(line.find(", 11, 11, 373.") != std::string::npos);
  }
  const std::vector<std::string> films = card_lines(heat, "*FILM");
  REQUIRE(films.size() == 4);
  for (const std::string& line : films) {
    REQUIRE(line.find(", F2, 293.14999999999998, 25") != std::string::npos);
  }
  for (const std::string& path : decks) std::remove(path.c_str());
}

namespace {

/// The member `key` of a JSON object, which must exist.
const json::Value& member(const json::Value& object, const std::string& key) {
  const json::Value* found = object.find(key);
  REQUIRE(found != nullptr);
  return *found;
}

/// The minimal deck with dynamics blocks spliced in before its closing brace.
std::string dynamic_deck(const std::string& blocks) {
  std::string deck = minimal_deck();
  const std::size_t end = deck.rfind('}');
  return deck.substr(0, end) + ", " + blocks + "}";
}

}  // namespace

TEST_CASE("the transient and frequency-response blocks parse and validate",
          "[io][config][dynamics]") {
  const Configuration config = parse_configuration(
      json::parse(dynamic_deck(R"(
        "transient": {
          "enabled": true, "time_step": 1e-4, "end_time": 2e-3, "alpha": -0.05,
          "mass": "lumped", "damping": { "mass": 2.0, "stiffness": 1e-5 },
          "amplitude": { "type": "table", "times": [0, 1e-3], "values": [0, 1], "scale": 2 },
          "snapshot_every": 5, "load_cases": ["tip"],
          "monitors": [ { "name": "tip_vy", "component": "y", "quantity": "velocity",
                          "region": { "box": { "xmin": 0.8 } } } ]
        },
        "frequency_response": {
          "enabled": true,
          "frequencies": { "start": 10, "end": 1000, "count": 3, "spacing": "log" },
          "damping": { "structural": 0.02 }, "snapshot_frequencies": [100],
          "monitors": [ { "name": "tip_uy", "component": "y",
                          "region": { "box": { "xmin": 0.8 } } } ]
        })"),
                  "dynamic"),
      "dynamic", /*strict=*/true);
  const TransientOptions& t = config.transient.options;
  REQUIRE(config.transient.enabled);
  REQUIRE(t.time_step == 1e-4);
  REQUIRE(t.alpha == -0.05);
  REQUIRE(t.mass_type == MassType::Lumped);
  REQUIRE(t.mass_damping == 2.0);
  REQUIRE(t.stiffness_damping == 1e-5);
  REQUIRE(t.amplitude.kind == Amplitude::Kind::Table);
  REQUIRE(t.amplitude.value(0.5e-3) == Approx(1.0).epsilon(1e-15));
  REQUIRE(t.snapshot_every == 5);
  REQUIRE_FALSE(t.nonlinear);
  REQUIRE(t.monitors.size() == 1);
  REQUIRE(t.monitors[0].quantity == DynamicMonitor::Quantity::Velocity);
  REQUIRE(config.transient_load_cases() == std::vector<std::size_t>{0});
  const FrequencyResponseOptions& f = config.frequency_response.options;
  REQUIRE(f.frequencies.size() == 3);
  REQUIRE(f.frequencies[1] == Approx(100.0).epsilon(1e-14));
  REQUIRE(f.structural_damping == 0.02);
  REQUIRE(f.snapshot_frequencies == std::vector<Scalar>{100.0});

  const auto rejects = [](const std::string& blocks) {
    return parse_configuration(json::parse(dynamic_deck(blocks), "bad"), "bad", true);
  };
  // A duration that is not a whole number of steps.
  REQUIRE_THROWS_AS(rejects(R"("transient": {"enabled": true, "time_step": 1e-4,
                                             "end_time": 1.05e-4})"),
                    ConfigError);
  // HHT-alpha outside [-1/3, 0].
  REQUIRE_THROWS_AS(rejects(R"("transient": {"enabled": true, "time_step": 1e-4,
                                             "end_time": 1e-3, "alpha": 0.1})"),
                    ConfigError);
  // A non-linear key on a linear transient.
  REQUIRE_THROWS_AS(rejects(R"("transient": {"enabled": true, "time_step": 1e-4,
                                             "end_time": 1e-3, "kinematics": "finite"})"),
                    ConfigError);
  // A preloaded start of a non-linear transient.
  REQUIRE_THROWS_AS(rejects(R"("transient": {"enabled": true, "time_step": 1e-4,
                                             "end_time": 1e-3, "nonlinear": true,
                                             "start": "static"})"),
                    ConfigError);
  // An unknown load case.
  REQUIRE_THROWS_AS(rejects(R"("transient": {"enabled": true, "time_step": 1e-4,
                                             "end_time": 1e-3, "load_cases": ["nope"]})"),
                    ConfigError);
  // A frequency response without frequencies, or with a log sweep from zero.
  REQUIRE_THROWS_AS(rejects(R"("frequency_response": {"enabled": true})"), ConfigError);
  REQUIRE_THROWS_AS(rejects(R"("frequency_response": {"enabled": true, "frequencies":
                                  {"start": 0, "end": 10, "count": 4, "spacing": "log"}})"),
                    ConfigError);
  REQUIRE_THROWS_AS(rejects(R"("frequency_response": {"enabled": true, "frequencies": [5],
                                  "damping": {"structural": -0.1}})"),
                    ConfigError);
}

TEST_CASE("transient and frequency-response results are written as CSV, VTK and JSON",
          "[io][writers][dynamics]") {
  const std::string dir = "results/_test_tmp/dynamics";
  const Configuration config = parse_configuration(
      json::parse(dynamic_deck(R"("output": {"csv": true, "vtk": true})"), "writers"),
      "writers", true);
  const FemModel model = build_model(config);
  const Assembler assembler(model);
  DynamicMonitor tip;
  tip.name = "tip_uy";
  Selector right;
  right.kind = SelectorKind::Box;
  right.xmin = 0.8;
  tip.region.members.push_back(right);
  tip.component = 1;

  TransientOptions transient;
  transient.time_step = 1.0e-4;
  transient.end_time = 1.0e-3;
  transient.snapshot_every = 4;  // steps 0, 4, 8 and the last, 10
  transient.monitors.push_back(tip);
  const TransientResult t = solve_transient(model, assembler, 0, transient);
  REQUIRE(t.snapshots.size() == 4);

  FrequencyResponseOptions harmonic;
  harmonic.frequencies = {10.0, 20.0, 30.0};
  harmonic.structural_damping = 0.02;
  harmonic.snapshot_frequencies = {21.0};
  harmonic.monitors.push_back(tip);
  const FrequencyResponseResult h = solve_frequency_response(model, assembler, 0, harmonic);

  const ResultWriter writer(dir, config);
  writer.write_transient(model, t);
  writer.write_frequency_response(model, h);

  // The history: a header with the energies and the monitor, one row per step.
  std::ifstream history(dir + "/transient_tip.csv");
  std::string line;
  std::getline(history, line);
  REQUIRE(line ==
          "step,time[s],max_displacement[m],kinetic_energy[J],strain_energy[J],"
          "damping_energy[J],external_work[J],energy_balance[J],tip_uy[m]");
  int rows = 0;
  while (std::getline(history, line)) ++rows;
  REQUIRE(rows == 11);
  REQUIRE(read_text(dir + "/transient_state_tip.csv").find("vx[m/s],vy[m/s],ax[m/s^2]") !=
          std::string::npos);
  REQUIRE(read_text(dir + "/transient_reactions_tip.csv").find("ry[N]") != std::string::npos);
  // The snapshot series and its index of times.
  const json::Value series = json::parse(read_text(dir + "/transient_tip.vtk.series"), "series");
  const std::vector<json::Value> files = member(series, "files").array_items();
  REQUIRE(files.size() == 4);
  REQUIRE(member(files[3], "name").string_value() == "transient_tip_0003.vtk");
  REQUIRE(member(files[3], "time").number_value() == Approx(1.0e-3).epsilon(1e-12));
  const std::string snapshot = read_text(dir + "/transient_tip_0002.vtk");
  REQUIRE(snapshot.find("VECTORS velocity double") != std::string::npos);

  // The harmonic response: re, im, modulus and phase of each monitor.
  std::ifstream frf(dir + "/frequency_response_tip.csv");
  std::getline(frf, line);
  REQUIRE(line == "point,frequency[Hz],max_displacement[m],tip_uy_re[m],tip_uy_im[m],"
                  "tip_uy_abs[m],tip_uy_phase[deg]");
  const json::Value fseries =
      json::parse(read_text(dir + "/frequency_response_tip.vtk.series"), "series");
  const std::vector<json::Value> ffiles = member(fseries, "files").array_items();
  REQUIRE(ffiles.size() == 1);
  REQUIRE(member(ffiles[0], "time").number_value() == 20.0);  // the nearest solved
  const std::string field = read_text(dir + "/frequency_response_tip_0000.vtk");
  REQUIRE(field.find("VECTORS displacement_imag double") != std::string::npos);
  REQUIRE(field.find("SCALARS displacement_peak double 1") != std::string::npos);

  // The summary blocks.
  const json::Value tj = transient_json({t}, transient);
  REQUIRE(member(tj, "gamma").number_value() == 0.5);
  const json::Value tcase = member(tj, "load_cases").array_items().at(0);
  REQUIRE(member(tcase, "steps").number_value() == 10.0);
  REQUIRE(member(member(tcase, "energy"), "largest_relative_balance").number_value() < 1.0e-10);
  const json::Value fj = frequency_response_json({h}, harmonic);
  const json::Value fcase = member(fj, "load_cases").array_items().at(0);
  REQUIRE(member(member(fcase, "monitors").array_items().at(0), "name").string_value() ==
          "tip_uy");
}

TEST_CASE("CalculiX transient decks carry *DYNAMIC, the amplitude and the damping",
          "[io][writers][cross-validation][dynamics]") {
  ensure_directory("results/_test_tmp");
  SolidCantileverCase solid;
  FemModel model = make_cantilever_3d(solid, 4, 1, 1);
  TransientOptions options;
  options.time_step = 1.0e-4;
  options.end_time = 2.0e-3;
  options.alpha = -0.05;
  options.mass_damping = 20.0;
  options.stiffness_damping = 1.0e-5;
  options.snapshot_every = 5;
  options.amplitude.kind = Amplitude::Kind::Table;
  options.amplitude.times = {0.0, 1.0e-3};
  options.amplitude.values = {0.0, 1.0};
  REQUIRE(calculix_transient_obstacle(model, 0, options).empty());

  CalculixTransientExport dynamic;
  dynamic.load_cases = {0};
  dynamic.options = options;
  const std::vector<std::string> decks =
      write_calculix_decks(model, "results/_test_tmp/dyn", "unit", nullptr, &dynamic);
  REQUIRE(decks.size() == 2);
  REQUIRE(decks[1].find("_dynamic.inp") != std::string::npos);
  const std::string text = read_text(decks[1]);
  REQUIRE(text.find("*DAMPING, ALPHA=20, BETA=1e-05\n") != std::string::npos);
  REQUIRE(text.find("*DYNAMIC, DIRECT, ALPHA=-0.05\n") != std::string::npos);
  REQUIRE(text.find("*NODE FILE, FREQUENCY=5\nU\n") != std::string::npos);
  REQUIRE(text.find("*CLOAD, AMPLITUDE=A1\n") != std::string::npos);
  // The amplitude at every step time, 0 to 20 steps.
  const std::vector<std::string> table = card_lines(text, "*AMPLITUDE, NAME=A1");
  REQUIRE(table.size() == 21);
  for (std::size_t k = 0; k < table.size(); ++k) {
    const std::size_t comma = table[k].find(',');
    REQUIRE(comma != std::string::npos);
    const double t = std::stod(table[k].substr(0, comma));
    const double a = std::stod(table[k].substr(comma + 1));
    REQUIRE(t == Approx(1.0e-4 * static_cast<double>(k)).margin(1e-18));
    REQUIRE(a == Approx(options.amplitude.value(t)).margin(1e-15));
  }
  // Held DOFs on a plain *BOUNDARY card.
  REQUIRE(card_lines(text, "*BOUNDARY").size() ==
          static_cast<std::size_t>(model.dofs().num_constrained()));
  for (const std::string& path : decks) std::remove(path.c_str());

  // What CalculiX cannot integrate as the same problem is refused.
  TransientOptions lumped = options;
  lumped.mass_type = MassType::Lumped;
  REQUIRE_FALSE(calculix_transient_obstacle(model, 0, lumped).empty());
  TransientOptions preloaded = options;
  preloaded.start = TransientOptions::Start::Static;
  REQUIRE_FALSE(calculix_transient_obstacle(model, 0, preloaded).empty());
  TransientOptions sudden = options;
  sudden.amplitude = Amplitude();  // a step: the load acts at t = 0
  REQUIRE_FALSE(calculix_transient_obstacle(model, 0, sudden).empty());
  CalculixTransientExport refused = dynamic;
  refused.options = sudden;
  REQUIRE_THROWS_AS(write_calculix_decks(model, "results/_test_tmp/dyn2", "unit", nullptr,
                                         &refused),
                    IoError);
}

TEST_CASE("path_join handles separators", "[io]") {
  REQUIRE(path_join("a", "b") == "a/b");
  REQUIRE(path_join("a/", "b") == "a/b");
  REQUIRE(path_join("", "b") == "b");
}

TEST_CASE("the contact block parses and validates", "[io][config][contact]") {
  const auto deck = [](const std::string& blocks) {
    return json::parse(dynamic_deck(blocks), "contact");
  };
  const std::string nonlinear =
      R"("nonlinear": { "enabled": true, "kinematics": "small_strain" }, )";
  const Configuration config = parse_configuration(
      deck(nonlinear + R"(
        "contact": {
          "enabled": true, "complementarity": 2.0, "search_factor": 3.0,
          "pairs": [
            { "name": "floor", "slave": { "box": { "ymax": 0.0 } }, "friction": 0.2,
              "obstacle": { "type": "plane", "point": [0, -1e-3], "normal": [0, 1],
                            "motion": [0, 1e-4] } },
            { "name": "pin", "slave": { "box": { "xmin": 0.9 } },
              "obstacle": { "type": "cylinder", "point": [1.2, 0.05], "radius": 0.1,
                            "inside": true } },
            { "name": "interface", "slave": { "box": { "ymin": 0.1 } },
              "master": { "box": { "xmax": 0.1 } } }
          ]
        })"),
      "contact", /*strict=*/true);
  const ContactOptions& c = config.nonlinear.options.contact;
  REQUIRE(c.enabled);
  REQUIRE(c.complementarity == 2.0);
  REQUIRE(c.search_factor == 3.0);
  REQUIRE(c.pairs.size() == 3);
  REQUIRE(c.pairs[0].rigid);
  REQUIRE(c.pairs[0].friction == 0.2);
  REQUIRE(c.pairs[0].obstacle.kind == RigidObstacle::Kind::Plane);
  REQUIRE(c.pairs[0].obstacle.point.y() == -1.0e-3);
  REQUIRE(c.pairs[0].obstacle.motion.y() == 1.0e-4);
  REQUIRE(c.pairs[1].obstacle.kind == RigidObstacle::Kind::Cylinder);
  REQUIRE(c.pairs[1].obstacle.radius == 0.1);
  REQUIRE(c.pairs[1].obstacle.inside);
  REQUIRE_FALSE(c.pairs[2].rigid);
  REQUIRE(c.pairs[2].friction == 0.0);

  const std::string pair =
      R"({ "name": "p", "slave": { "box": { "ymax": 0.0 } },
           "obstacle": { "type": "plane", "normal": [0, 1] } })";
  const auto refused = [&](const std::string& blocks) {
    INFO(blocks);
    REQUIRE_THROWS_AS(parse_configuration(deck(blocks), "contact", /*strict=*/true),
                      ConfigError);
  };
  // Without the non-linear analysis, or with finite kinematics or arc length.
  refused(R"("contact": { "enabled": true, "pairs": [)" + pair + "] }");
  refused(R"("nonlinear": { "enabled": true }, "contact": { "enabled": true, "pairs": [)" +
          pair + "] }");
  refused(R"("nonlinear": { "enabled": true, "kinematics": "small_strain", "method": )"
          R"("arc_length" }, "contact": { "enabled": true, "pairs": [)" + pair + "] }");
  // No pair, two pairs of one name, a negative friction coefficient.
  refused(nonlinear + R"("contact": { "enabled": true })");
  refused(nonlinear + R"("contact": { "enabled": true, "pairs": [)" + pair + ", " + pair + "] }");
  refused(nonlinear + R"("contact": { "enabled": true, "pairs": [
      { "slave": { "box": { "ymax": 0.0 } }, "friction": -0.1,
        "obstacle": { "type": "plane", "normal": [0, 1] } } ] })");
  // An obstacle and a master surface, neither, an unknown obstacle.
  refused(nonlinear + R"("contact": { "enabled": true, "pairs": [
      { "slave": { "box": { "ymax": 0.0 } }, "master": { "box": { "xmin": 0.5 } },
        "obstacle": { "type": "plane", "normal": [0, 1] } } ] })");
  refused(nonlinear + R"("contact": { "enabled": true, "pairs": [
      { "slave": { "box": { "ymax": 0.0 } } } ] })");
  refused(nonlinear + R"("contact": { "enabled": true, "pairs": [
      { "slave": { "box": { "ymax": 0.0 } }, "obstacle": { "type": "cone" } } ] })");
  refused(nonlinear + R"("contact": { "enabled": true, "pairs": [
      { "slave": { "box": { "ymax": 0.0 } },
        "obstacle": { "type": "sphere", "center": [0, 1], "radius": -1 } } ] })");
}
