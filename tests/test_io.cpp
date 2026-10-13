/// \file test_io.cpp
/// \brief JSON parsing/serialisation, configuration validation and writers.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/CsvWriter.hpp"
#include "sparlab/io/Json.hpp"
#include "sparlab/io/VtkWriter.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

/// A minimal but complete deck used by the configuration tests.
std::string minimal_deck() {
  return R"({
    "name": "unit_case",
    "description": "a deck used by the test suite",
    "mesh": { "type": "structured_hex", "nx": 8, "ny": 4, "nz": 1, "lx": 0.8, "ly": 0.4,
              "lz": 0.01 },
    "material": { "name": "al", "youngs_modulus": 70e9, "poisson_ratio": 0.33,
                  "density": 2700 },
    "model": { "stress_state": "three_dimensional" },
    "boundary_conditions": [
      { "name": "root", "fix": ["x", "y", "z"], "region": { "box": { "xmax": 0.0 } } }
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
  json::Value v = json::Value::make_array();
  for (const Scalar x : {1.5, -2.5, 1.0e-17}) v.push_back(json::Value::make_number(x));
  doc.set("vector", v);
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
  REQUIRE(root.child("a").vector3_or("v", Vector3::Ones()).isApprox(Vector3(1.0, 2.0, 0.0)));
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
  REQUIRE_THROWS_AS(root.child("a").child("c").vector3(), ConfigError);
  REQUIRE_THROWS_AS(root.child("a").child("b").integer(), ConfigError);
  REQUIRE_THROWS_AS(root.child("list").child("x"), ConfigError);

  // positive_number validates the sign.
  REQUIRE(root.child("a").positive_number("b") == Approx(3.5));
  REQUIRE_THROWS_AS(root.child("a").child("list").positive_number("x"), ConfigError);

  // "typo_key" was never read.
  const std::vector<std::string> unused = root.unused_keys();
  REQUIRE(std::find(unused.begin(), unused.end(), "typo_key") != unused.end());
  REQUIRE(std::find(unused.begin(), unused.end(), "a.b") == unused.end());
}

TEST_CASE("a minimal deck parses into a model", "[io][config]") {
  const json::Value doc = json::parse(minimal_deck(), "minimal");
  const Configuration config = parse_configuration(doc, "minimal", /*strict=*/true);

  REQUIRE(config.name == "unit_case");
  REQUIRE(config.mesh_spec.nx == 8);
  REQUIRE(config.mesh_spec.ly == Approx(0.4));
  REQUIRE(config.material().youngs_modulus() == Approx(70.0e9));
  REQUIRE(config.material().poisson_ratio() == Approx(0.33));
  REQUIRE(config.thickness == Approx(1.0));
  REQUIRE(config.stress_state == StressState::ThreeDimensional);
  REQUIRE(config.constraints.size() == 1);
  REQUIRE(config.constraints.front().fix_x);
  REQUIRE(config.constraints.front().fix_z);
  // The one empty load case of a model the forming analysis drives.
  REQUIRE(config.load_cases.size() == 1);
  REQUIRE(config.load_cases.front().name == "forming");
  // Defaults.
  REQUIRE(config.integration.stiffness_points == 2);
  REQUIRE(config.integration.mass_points == 3);
  REQUIRE_FALSE(config.forming.enabled);

  FemModel model = build_model(config);
  REQUIRE(model.mesh().num_elements() == 32);
  REQUIRE(model.finalized());
  REQUIRE(model.dofs().num_constrained() == 3 * 5 * 2);
}

TEST_CASE("configuration errors are specific and actionable",
          "[io][config][diagnostics]") {
  const auto parse_modified = [](const std::string& body) {
    const json::Value doc = json::parse(body, "bad");
    return parse_configuration(doc, "bad");
  };
  const std::string mesh = R"("mesh": {"type": "structured_hex", "nx": 2, "ny": 2, "nz": 1,
                                       "lx": 1, "ly": 1, "lz": 0.1},)";
  const std::string material = R"("material": {"youngs_modulus": 1e9, "poisson_ratio": 0.3},)";

  REQUIRE_THROWS_AS(parse_modified("[1, 2]"), ConfigError);
  REQUIRE_THROWS_AS(parse_modified("{}"), ConfigError);

  // Missing mesh entries.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"type": "structured_hex", "nx": 2, "ny": 2, "nz": 1,
                                  "ly": 1.0, "lz": 0.1},)" +
                     material + R"("boundary_conditions": [
                         {"fix": ["x"], "region": {"all": true}}]})"),
      ConfigError);

  // Unknown mesh type.
  REQUIRE_THROWS_AS(
      parse_modified(R"({"mesh": {"type": "tri3", "nx": 2, "ny": 2, "lx": 1, "ly": 1},)" +
                     material + R"("boundary_conditions": [
                         {"fix": ["x"], "region": {"all": true}}]})"),
      ConfigError);

  // Empty boundary conditions without a forming block.
  REQUIRE_THROWS_AS(parse_modified("{" + mesh + material + R"("boundary_conditions": []})"),
                    ConfigError);

  // An invalid fix component.
  REQUIRE_THROWS_AS(parse_modified("{" + mesh + material + R"("boundary_conditions": [
                            {"fix": ["w"], "region": {"box": {"xmax": 0}}}]})"),
                    ConfigError);

  // A region naming two primitives at once.
  REQUIRE_THROWS_AS(parse_modified("{" + mesh + material + R"("boundary_conditions": [
                            {"fix": ["x"], "region": {"box": {"xmax": 0},
                                                      "circle": {"center": [0,0,0],
                                                                 "radius": 1}}}]})"),
                    ConfigError);

  // A region naming no primitive.
  REQUIRE_THROWS_AS(parse_modified("{" + mesh + material + R"("boundary_conditions": [
                            {"fix": ["x"], "region": {"name": "empty"}}]})"),
                    ConfigError);

  // The decks the checks above modify are valid.
  REQUIRE_NOTHROW(parse_modified("{" + mesh + material + R"("boundary_conditions": [
                            {"fix": ["x"], "region": {"box": {"xmax": 0}}}]})"));

  // An empty BC region selects no nodes: caught at finalize().
  {
    const Configuration config =
        parse_modified("{" + mesh + material + R"("boundary_conditions": [
            {"name": "nowhere", "fix": ["x","y","z"], "region": {"box": {"xmin": 100.0}}}]})");
    REQUIRE_THROWS_AS(build_model(config), ConfigError);
  }
}

TEST_CASE("a deck without mesh.type is refused", "[io][config][diagnostics]") {
  // There is no default mesh type: before the 2-D meshes were removed a
  // missing type meant structured_quad, so a structured_hex default would
  // silently run a deck that used to be refused.
  std::string deck = minimal_deck();
  const std::string type = R"("type": "structured_hex", )";
  REQUIRE(deck.find(type) != std::string::npos);
  REQUIRE_NOTHROW(parse_configuration(json::parse(deck, "deck"), "deck"));
  deck.erase(deck.find(type), type.size());
  REQUIRE_THROWS_AS(parse_configuration(json::parse(deck, "deck"), "deck"), ConfigError);
  REQUIRE_THROWS_WITH(parse_configuration(json::parse(deck, "deck"), "deck"),
                      ContainsSubstring("mesh.type"));
}

TEST_CASE("unknown configuration keys are reported", "[io][config][diagnostics]") {
  const std::string deck = R"({
    "name": "typo_case",
    "mesh": { "type": "structured_hex", "nx": 4, "ny": 2, "nz": 1, "lx": 1.0, "ly": 0.5,
              "lz": 0.1 },
    "material": { "youngs_modulus": 1e9, "poisson_ratio": 0.3, "density": 1000 },
    "model": { "thicknes": 0.01 },
    "boundary_conditions": [
      { "fix": ["x","y","z"], "region": { "box": { "xmax": 0.0 } } }
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
      csv.row(3, {4.0, 5.0});
      csv.raw_row({"x", "y", "z"});
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
    FemModel model = make_small_block(3, 2);
    const Vector u = Vector::LinSpaced(model.dofs().num_dofs(), 0.0, 1.0e-3);

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
    REQUIRE(content.find("POINTS 24 double") != std::string::npos);
    REQUIRE(content.find("CELLS 6 54") != std::string::npos);
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

TEST_CASE("path_join handles separators", "[io]") {
  REQUIRE(path_join("a", "b") == "a/b");
  REQUIRE(path_join("a/", "b") == "a/b");
  REQUIRE(path_join("", "b") == "b");
}
