/// \file test_forming.cpp
/// \brief Incremental forming: tool trajectories, the penalty contact of
///        rigid tools in the current configuration with Coulomb friction,
///        and the forming analysis (form and release steps).
///
/// The contact kernel is checked against central finite differences of its
/// own residual (sphere, plane and cylinder; frictionless, sticking and
/// slipping), and the analysis against exact solutions: a flat punch on an
/// elastic block (uniaxial stress, the penalty compliance in series), tools
/// ironing a block with friction (in steady sliding every contact node
/// slips, so the friction load is mu times the normal load - for a flat
/// punch the resultants too), rigid-body invariance of a release, and the
/// springback of an elastic-perfectly plastic beam bent past yield and
/// released onto statically determinate supports, against the exact elastic
/// unloading of its moment-curvature relation. A small single-point
/// incremental forming case runs the whole sequence. The `forming` block of
/// a deck is read strictly and refused where wrong, and a run's result files
/// follow the output contract of FormingWriter.hpp.
#include "TestSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"
#include "sparlab/core/Timer.hpp"
#include "sparlab/fem/Forming.hpp"
#include "sparlab/fem/RigidTool.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/FormingWriter.hpp"
#include "sparlab/io/Json.hpp"
#include "sparlab/io/ResultWriter.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>

using namespace sparlab;
using namespace sparlab::testing;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Scalar kInf = std::numeric_limits<Scalar>::infinity();

SelectorGroup box(Scalar xmin, Scalar xmax, Scalar ymin = -kInf, Scalar ymax = kInf,
                  Scalar zmin = -kInf, Scalar zmax = kInf, const std::string& name = "box") {
  SelectorGroup g;
  g.name = name;
  Selector s;
  s.kind = SelectorKind::Box;
  s.xmin = xmin;
  s.xmax = xmax;
  s.ymin = ymin;
  s.ymax = ymax;
  s.zmin = zmin;
  s.zmax = zmax;
  g.members.push_back(s);
  return g;
}

SelectorGroup node_set(const std::vector<Index>& nodes, const std::string& name = "nodes") {
  SelectorGroup g;
  g.name = name;
  Selector s;
  s.kind = SelectorKind::NodeIds;
  s.ids = nodes;
  g.members.push_back(s);
  return g;
}

ToolTrajectory path(std::vector<Scalar> times, std::vector<Vector3> points) {
  ToolTrajectory t;
  t.times = std::move(times);
  t.points = std::move(points);
  return t;
}

StepConstraint constraint(const SelectorGroup& region, std::vector<int> components,
                          StepConstraint::Mode mode = StepConstraint::Mode::Hold,
                          const Vector3& value = Vector3::Zero()) {
  StepConstraint c;
  c.name = region.name;
  c.constraint.region = region;
  for (int k : components) c.constraint.set(k, true, value(k));
  c.mode = mode;
  return c;
}

/// A finalised model with one load case that applies nothing.
FemModel finalised(Mesh mesh, const IsotropicMaterial& m, Scalar thickness = 1.0,
                   StressState state = StressState::ThreeDimensional,
                   const std::vector<DisplacementConstraint>& bcs = {}) {
  FemModel model(std::move(mesh), m, thickness, state, IntegrationOptions());
  model.constraints() = bcs;
  LoadCaseSpec lc;
  lc.name = "none";
  model.load_case_specs().push_back(lc);
  model.finalize();
  return model;
}

Mesh hex_block(Index nx, Index ny, Index nz, Scalar lx, Scalar ly, Scalar lz, Scalar x0 = 0.0,
               Scalar y0 = 0.0, Scalar z0 = 0.0) {
  StructuredMeshSpec spec;
  spec.nx = nx;
  spec.ny = ny;
  spec.nz = nz;
  spec.lx = lx;
  spec.ly = ly;
  spec.lz = lz;
  spec.x0 = x0;
  spec.y0 = y0;
  spec.z0 = z0;
  return make_structured_hex_mesh(spec);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Trajectories
// ---------------------------------------------------------------------------
TEST_CASE("a tool trajectory interpolates linearly, holds its ends and reads a CSV file",
          "[forming]") {
  const ToolTrajectory t = path({0.0, 1.0, 3.0}, {Vector3(0, 0, 1), Vector3(2, 0, 1),
                                                  Vector3(2, 4, 0)});
  t.validate();
  CHECK((t.position(0.5) - Vector3(1, 0, 1)).norm() < 1e-15);
  CHECK((t.position(2.0) - Vector3(2, 2, 0.5)).norm() < 1e-15);
  CHECK((t.position(-1.0) - Vector3(0, 0, 1)).norm() == 0.0);  // clamped before
  CHECK((t.position(7.0) - Vector3(2, 4, 0)).norm() == 0.0);   // and after
  CHECK((t.position(1.0) - Vector3(2, 0, 1)).norm() < 1e-15);  // a knot exactly
  CHECK((t.velocity(0.25) - Vector3(2, 0, 0)).norm() < 1e-15);
  CHECK((t.velocity(1.0) - Vector3(0, 2, -0.5)).norm() < 1e-15);  // right derivative
  CHECK(t.velocity(3.0).norm() == 0.0);
  CHECK(t.velocity(-0.1).norm() == 0.0);
  // Monotone and random-access lookups agree (the cached segment).
  for (Scalar s : {2.5, 0.1, 2.9, 1.5, 0.0, 2.0}) {
    const Vector3 expected = s <= 1.0 ? Vector3(2 * s, 0, 1)
                                      : Vector3(2, 2 * (s - 1), 1 - 0.5 * (s - 1));
    CHECK((t.position(s) - expected).norm() < 1e-14);
  }
  CHECK(t.knots_in(0.0, 3.0) == std::vector<Scalar>{1.0});
  CHECK(t.knots_in(1.0, 3.0).empty());

  // Refusals.
  CHECK_THROWS_WITH(path({0.0}, {Vector3::Zero()}).validate(),
                    ContainsSubstring("at least two knots"));
  CHECK_THROWS_WITH(path({0.0, 0.0}, {Vector3::Zero(), Vector3::Ones()}).validate(),
                    ContainsSubstring("increase strictly"));
  CHECK_THROWS_WITH(path({0.0, 1.0}, {Vector3::Zero(), Vector3(0, std::nan(""), 0)}).validate(),
                    ContainsSubstring("not finite"));
  CHECK_THROWS_WITH(path({0.0, 1.0}, {Vector3::Zero()}).validate(),
                    ContainsSubstring("point(s)"));

  // CSV: spaces, blank and comment lines tolerated.
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "sparlab_test_forming_csv";
  std::filesystem::create_directories(dir);
  const auto write = [&](const std::string& name, const std::string& text) {
    const std::string p = (dir / name).string();
    std::ofstream(p) << text;
    return p;
  };
  const ToolTrajectory csv = ToolTrajectory::from_csv(
      write("ok.csv", "# a path\n t , x, y ,z\n0, 0,0,0.01\n\n 1.5 ,0.02, 0, 0.01 \n"));
  REQUIRE(csv.times.size() == 2);
  CHECK(csv.times[1] == 1.5);
  CHECK((csv.points[1] - Vector3(0.02, 0, 0.01)).norm() == 0.0);
  CHECK_THROWS_WITH(ToolTrajectory::from_csv(write("h.csv", "time,x,y,z\n0,0,0,0\n1,1,1,1\n")),
                    ContainsSubstring("line 1") && ContainsSubstring("header"));
  CHECK_THROWS_WITH(ToolTrajectory::from_csv(write("f.csv", "t,x,y,z\n0,0,0,0\n1,1,abc,1\n")),
                    ContainsSubstring("line 3") && ContainsSubstring("abc"));
  CHECK_THROWS_WITH(ToolTrajectory::from_csv(write("n.csv", "t,x,y,z\n0,0,0,0\n1,1,1\n")),
                    ContainsSubstring("line 3") && ContainsSubstring("4 values"));
  CHECK_THROWS_WITH(ToolTrajectory::from_csv(write("d.csv", "t,x,y,z\n0,0,0,0\n0,1,1,1\n")),
                    ContainsSubstring("increase strictly"));
  CHECK_THROWS_AS(ToolTrajectory::from_csv((dir / "missing.csv").string()), IoError);
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// 2. The contact tangent against finite differences
// ---------------------------------------------------------------------------
namespace {

/// The largest error of the contact tangent against central differences of
/// the contact residual, over the DOFs of the nodes in contact, relative to
/// the largest tangent entry.
Scalar tangent_error(const ToolContact& contact, const FemModel& model, const Vector& u,
                     Scalar t, int& active, int& slipping, bool& symmetric) {
  const int dim = model.dim();
  const ToolContactEvaluation ev = contact.evaluate(u, t, true);
  active = 0;
  slipping = 0;
  for (const ToolResultant& r : ev.tools) {
    active += r.active_nodes;
    slipping += r.slipping_nodes;
  }
  symmetric = ev.symmetric;
  Scalar top = 0.0;
  for (const auto& b : ev.tangent) top = std::max(top, b.k.cwiseAbs().maxCoeff());
  const Scalar h = 1.0e-9;
  Scalar worst = 0.0;
  for (const auto& b : ev.tangent) {
    for (int c = 0; c < dim; ++c) {
      const Index d = b.node * dim + c;
      Vector up = u;
      Vector dn = u;
      up(d) += h;
      dn(d) -= h;
      Vector fd = (contact.evaluate(up, t, false).residual -
                   contact.evaluate(dn, t, false).residual) / (2.0 * h);
      // The tangent is node-diagonal: the node's own block, nothing else.
      for (int a = 0; a < dim; ++a) fd(b.node * dim + a) -= b.k(a, c);
      worst = std::max(worst, fd.cwiseAbs().maxCoeff() / top);
    }
  }
  return worst;
}

}  // namespace

TEST_CASE("the penalty contact tangent is the exact derivative of its forces, with friction",
          "[forming]") {
  const IsotropicMaterial m = default_material();
  // A block 10 mm x 10 mm x 2 mm, its top at z = 0, with its nodes moved
  // off the grid so no symmetry helps.
  std::mt19937 rng(7u);
  std::uniform_real_distribution<Scalar> jitter(-1.0e-4, 1.0e-4);
  for (const RigidTool::Shape shape :
       {RigidTool::Shape::Sphere, RigidTool::Shape::Plane, RigidTool::Shape::Cylinder}) {
    for (const std::string regime : {"frictionless", "stick", "slip", "new"}) {
      INFO(to_string(shape) << ", " << regime);
      FemModel model = finalised(hex_block(20, 20, 1, 0.01, 0.01, 0.001, 0.0, 0.0, -0.001), m);
      RigidTool tool;
      tool.name = "tool";
      tool.shape = shape;
      tool.radius = 0.005;
      tool.normal = Vector3(0.1, -0.2, 1.0);          // plane: tilted, facing +z
      tool.axis = Vector3(1.0, 0.3, 0.0);             // cylinder
      tool.surface = box(-kInf, kInf, -kInf, kInf, 0.0, kInf);
      tool.friction = regime == "frictionless" ? 0.0 : regime == "slip" ? 0.1 : 0.8;
      // Centre (or plane point) placed so that several top nodes penetrate,
      // by up to 0.1 mm (the plane, tilted, more).
      const Vector3 centre = shape == RigidTool::Shape::Plane
                                 ? Vector3(0.005, 0.005, -2.0e-5)
                                 : Vector3(0.0052, 0.0047, 0.005 - 1.0e-4);
      tool.trajectory = path({0.0, 1.0}, {centre, centre + Vector3(1e-3, 0, 0)});
      ToolContact contact(model, {tool});
      const Index n = model.dofs().num_dofs();
      Vector u = Vector::Zero(n);
      for (Index d = 0; d < n; ++d) u(d) = 0.02 * jitter(rng);  // up to 2 um
      const Scalar t = 0.0;
      contact.begin_increment(u, t, t, {1}, 1.0e-3);
      // A committed history for every node in contact at (u, t): a force
      // and an anchor so the trial sticks or slips clearly.
      ToolContactEvaluation ev0 = contact.evaluate(u, t, false);
      REQUIRE(ev0.tools[0].active_nodes >= 3);
      ToolHistory h(1);
      if (regime != "frictionless" && regime != "new") {
        const Mesh& mesh = model.mesh();
        for (Index node : contact.slave_nodes(0)) {
          Vector3 x = mesh.node(node) + u.segment(node * 3, 3);
          Vector3 nrm;
          Matrix3 dn;
          Scalar d = 0.0;
          const Vector3 c = tool.trajectory.position(t);
          if (tool.gap(x, c, 3, nrm, dn, d) >= 0.0) continue;
          const Vector3 r = x - c;
          const Vector3 along(1, 0.4, 0);
          const Vector3 tangent = (along - along.dot(nrm) * nrm).normalized();
          ToolNodeHistory entry;
          // Stick: a small force and a short slip; slip: the anchor far away.
          // The committed force leans off the tangent plane (as one of an
          // earlier normal does), which the transport rotates back.
          entry.force = (regime == "stick" ? 5.0 : 50.0) * (tangent + 0.2 * nrm);
          entry.relative = r - (regime == "stick" ? 1.0e-8 : 1.0e-3) * tangent;
          h[0][node] = entry;
        }
        contact.set_history(h);
      }
      int active = 0;
      int slipping = 0;
      bool symmetric = true;
      const Scalar err = tangent_error(contact, model, u, t, active, slipping, symmetric);
      log::info("contact tangent vs central differences, ", to_string(shape), ", ", regime, ": ",
                err, " (", active, " node(s) in contact, ", slipping, " slipping)");
      CHECK(err < 1.0e-6);
      if (regime == "slip") {
        CHECK(slipping == active);
        CHECK_FALSE(symmetric);
      }
      if (regime == "stick" || regime == "new") CHECK(slipping == 0);
      if (regime == "frictionless") CHECK(symmetric);

      // The symmetric friction tangent: the same forces, symmetric node
      // blocks whose friction part is positive semi-definite - for the
      // plane, whose normal part is too, the whole block. (The symmetric
      // part of the exact tangent is indefinite where a node slips.)
      ToolContact sym(model, {tool}, FrictionTangent::Symmetric);
      sym.begin_increment(u, t, t, {1}, 1.0e-3);
      if (!h[0].empty()) sym.set_history(h);
      const ToolContactEvaluation es = sym.evaluate(u, t, true);
      CHECK(es.symmetric);
      CHECK((es.residual - contact.evaluate(u, t, false).residual).cwiseAbs().maxCoeff() == 0.0);
      Scalar lowest = 0.0;
      Scalar top = 0.0;
      for (const auto& b : es.tangent) {
        CHECK((b.k - b.k.transpose()).cwiseAbs().maxCoeff() == 0.0);
        const Vector3 values = Eigen::SelfAdjointEigenSolver<Matrix3>(b.k).eigenvalues();
        lowest = std::min(lowest, values.minCoeff());
        top = std::max(top, values.maxCoeff());
      }
      if (shape == RigidTool::Shape::Plane) CHECK(lowest >= -1.0e-12 * top);
    }
  }
}

TEST_CASE("tool contact refuses what it cannot model", "[forming]") {
  const IsotropicMaterial m = default_material();
  FemModel model = finalised(hex_block(2, 2, 1, 0.01, 0.01, 0.002), m);
  RigidTool tool;
  tool.radius = 0.005;
  tool.surface = box(-kInf, kInf, -kInf, kInf, 0.002, kInf);
  tool.trajectory = path({0.0, 1.0}, {Vector3(0, 0, 0.01), Vector3(0, 0, 0.0)});
  RigidTool bad = tool;
  bad.radius = 0.0;
  CHECK_THROWS_WITH(ToolContact(model, {bad}), ContainsSubstring("positive radius"));
  bad = tool;
  bad.friction = -0.1;
  CHECK_THROWS_WITH(ToolContact(model, {bad}), ContainsSubstring("friction"));
  bad = tool;
  bad.surface = box(-kInf, kInf, -kInf, kInf, 0.5, kInf);
  CHECK_THROWS_WITH(ToolContact(model, {bad}), ContainsSubstring("selects no boundary face"));
  CHECK_THROWS_WITH(ToolContact(model, {tool, tool}), ContainsSubstring("used twice"));
  // The name is a field of tool_forces.csv: no separator to escape.
  for (const std::string name : {"roll,er", "a b", "", "\"q\""}) {
    bad = tool;
    bad.name = name;
    CHECK_THROWS_WITH(ToolContact(model, {bad}), ContainsSubstring("must be non-empty and use only"));
  }
  bad = tool;
  bad.name = "Tool_2-b.v1";
  CHECK_NOTHROW(ToolContact(model, {bad}));
  // A node through the sphere's centre: the increment is cut.
  ToolContact contact(model, {tool});
  Vector u = Vector::Zero(model.dofs().num_dofs());
  CHECK_THROWS_AS(contact.evaluate(u, 1.0, false), SolverError);
  // The active-set prediction skips such a node (it is not outside the tool)
  // rather than use the normal of a gap it could not evaluate; the others
  // stay clear of the tool after the correction, so none is added.
  Vector extra = Vector::Zero(u.size());
  std::vector<ToolContactEvaluation::NodeBlock> blocks;
  CHECK(contact.anticipate(u, Vector::Constant(u.size(), 1.0e-4), 1.0, extra, blocks) == 0);
  CHECK(blocks.empty());
  CHECK(extra.cwiseAbs().maxCoeff() == 0.0);
}

// ---------------------------------------------------------------------------
// 3. A flat punch on an elastic block
// ---------------------------------------------------------------------------
TEST_CASE("a flat punch compresses an elastic block by the exact force, penalty compliance in "
          "series",
          "[forming]") {
  // A column H tall of one Hex8 in section (every top node with the same
  // area, so the same penalty stiffness), on rollers at its base and free
  // to expand sideways, pressed by a rigid plane lowered by delta:
  // uniaxial stress. The top nodes penetrate the punch by p / kappa, so
  // F = E A (delta - F / (A kappa)) / H exactly, F = E A delta / H / (1 +
  // E / (kappa H)) - which tends to the rigid-contact force as the penalty
  // grows.
  const IsotropicMaterial m = default_material();
  const Scalar e = m.youngs_modulus();
  const Scalar a_side = 0.004;
  const Scalar height = 0.01;
  const Scalar delta = 1.0e-5;
  const Scalar area = a_side * a_side;
  const Scalar rigid = e * area * delta / height;
  std::vector<Scalar> errors_rigid;
  for (const Scalar s : {10.0, 100.0, 1000.0}) {
    for (const Index nxy : {Index{1}, Index{3}}) {
      INFO("penalty scale " << s << ", " << nxy << " x " << nxy << " in section");
      std::vector<DisplacementConstraint> bcs;
      DisplacementConstraint base;
      base.region = box(-kInf, kInf, -kInf, kInf, -kInf, 0.0, "base");
      base.set(2, true, 0.0);
      bcs.push_back(base);
      DisplacementConstraint corner;
      corner.region = box(-kInf, 0.0, -kInf, 0.0, -kInf, 0.0, "corner");
      corner.set(0, true, 0.0);
      corner.set(1, true, 0.0);
      bcs.push_back(corner);
      DisplacementConstraint edge;
      edge.region = box(a_side, kInf, -kInf, 0.0, -kInf, 0.0, "edge");
      edge.set(1, true, 0.0);
      bcs.push_back(edge);
      FemModel model = finalised(hex_block(nxy, nxy, 4, a_side, a_side, height), m, 1.0,
                                 StressState::ThreeDimensional, bcs);
      Assembler assembler(model);
      RigidTool punch;
      punch.name = "punch";
      punch.shape = RigidTool::Shape::Plane;
      punch.normal = Vector3(0, 0, -1);  // out of the punch, down into the block
      punch.penalty = s;
      punch.surface = box(-kInf, kInf, -kInf, kInf, height, kInf, "top");
      punch.trajectory = path({0.0, 1.0}, {Vector3(0, 0, height), Vector3(0, 0, height - delta)});
      FormingOptions o;
      o.kinematics = Kinematics::SmallStrain;
      o.residual_tolerance = 1.0e-12;
      o.displacement_tolerance = 1.0e-12;
      o.tools.push_back(punch);
      FormingStep step;
      step.name = "press";
      step.tools = {"punch"};
      o.steps.push_back(step);
      const FormingResult r = FormingAnalysis(model, assembler, o).run();
      REQUIRE(r.completed);
      const FormingIncrement& last = r.steps.back().increments.back();
      REQUIRE(last.tools.size() == 1);
      const Scalar force = last.tools[0].force.z();  // on the punch, upwards
      CHECK(std::abs(last.tools[0].force.x()) < 1.0e-9 * force);
      CHECK(last.tools[0].active_nodes == (nxy + 1) * (nxy + 1));
      if (nxy == 1) {
        const Scalar kappa = s * e / std::sqrt(area / 4.0);
        const Scalar exact = rigid / (1.0 + e / (kappa * height));
        log::info("flat punch, s = ", s, ": force ", force, " N against ", exact,
                  " N with the penalty in series (relative error ",
                  std::abs(force - exact) / exact, "), rigid contact ", rigid, " N");
        CHECK(std::abs(force - exact) / exact < 1.0e-9);
        CHECK(last.tools[0].max_penetration == Approx(exact / (area * kappa)).epsilon(1e-9));
        errors_rigid.push_back(std::abs(force - rigid) / rigid);
      } else {
        // Edge and corner nodes have smaller areas, so stiffer springs: no
        // longer uniaxial, but within the compliance bound and converging.
        const Scalar kappa_min = s * e / std::sqrt(area / 9.0);
        CHECK(force < rigid);
        CHECK(force > rigid / (1.0 + e / (kappa_min * height)) * (1.0 - 1.0e-3));
      }
    }
  }
  // The force converges to the rigid-contact one as 1/s.
  REQUIRE(errors_rigid.size() == 3);
  log::info("flat punch: error against rigid contact ", errors_rigid[0], ", ", errors_rigid[1],
            ", ", errors_rigid[2], " at s = 10, 100, 1000");
  CHECK(errors_rigid[1] / errors_rigid[0] == Approx(0.1).epsilon(0.02));
  CHECK(errors_rigid[2] / errors_rigid[1] == Approx(0.1).epsilon(0.02));
}

TEST_CASE("a tool that grazes the surface from the reference state converges without a cut",
          "[forming]") {
  // The sphere's path reaches the block's top exactly at the end of the
  // first increment (0.5 mm of 1.5 mm travel), where a node's gap is zero to
  // round-off: its contact force, and so the reference force, are round-off
  // too, and the state must converge on the round-off of the positions.
  std::vector<DisplacementConstraint> bcs;
  DisplacementConstraint base;
  base.region = box(-kInf, kInf, -kInf, kInf, -kInf, -0.002, "base");
  for (int k = 0; k < 3; ++k) base.set(k, true);
  bcs.push_back(base);
  FemModel model = finalised(hex_block(10, 10, 2, 0.01, 0.01, 0.002, 0.0, 0.0, -0.002),
                             default_material(), 1.0, StressState::ThreeDimensional, bcs);
  Assembler assembler(model);
  RigidTool tool;
  tool.radius = 0.005;
  tool.friction = 0.1;
  tool.surface = box(-kInf, kInf, -kInf, kInf, 0.0, kInf, "top");
  tool.trajectory = path({0.0, 1.0}, {Vector3(0.005, 0.005, 0.005 + 5.0e-4),
                                      Vector3(0.005, 0.005, 0.005 - 1.0e-3)});
  FormingOptions o;
  o.tools.push_back(tool);
  FormingStep step;
  step.name = "press";
  step.tools = {"tool"};
  step.max_tool_travel = 5.0e-4;
  o.steps.push_back(step);
  const FormingResult r = FormingAnalysis(model, assembler, o).run();
  REQUIRE(r.completed);
  CHECK(r.total_cuts == 0);
  REQUIRE(r.steps[0].increments.size() >= 3);
  CHECK(r.steps[0].increments[0].iterations <= 3);
  CHECK(r.steps[0].increments.back().tools.at(0).force.z() > 0.0);
}

// ---------------------------------------------------------------------------
// 4. Ironing with friction
// ---------------------------------------------------------------------------
TEST_CASE("a tool dragged along a block with friction slides at mu times its normal force",
          "[forming]") {
  // An elastic block, its base clamped, pressed by a tool that is then
  // dragged. Once every node in contact slips, each carries the Coulomb
  // force mu p_N along its slip (that the tool's friction load is then mu
  // times its normal load is an identity of the return map, not a check).
  // A flat punch is dragged obliquely, at 30 degrees to x: every node's
  // normal is the same and, in steady sliding, its slip is the drag, so
  // each node's friction force and the resultant must point along the drag
  // - x and y in its proportion, which a friction force that did not follow
  // the slip would miss - with |F_T| = mu F_z. A sphere is dragged along x:
  // its resultants differ from F_x = mu F_z by its contact normals' tilt
  // (ploughing), 7e-4 on this mesh.
  const IsotropicMaterial m = default_material();
  const Scalar mu = 0.2;
  const Scalar pi = std::acos(-1.0);
  for (const RigidTool::Shape shape : {RigidTool::Shape::Sphere, RigidTool::Shape::Plane}) {
    INFO(to_string(shape));
    const bool plane = shape == RigidTool::Shape::Plane;
    std::vector<DisplacementConstraint> bcs;
    DisplacementConstraint base;
    base.region = box(-kInf, kInf, -kInf, kInf, -kInf, -0.004, "base");
    base.set(0, true);
    base.set(1, true);
    base.set(2, true);
    bcs.push_back(base);
    FemModel model = finalised(hex_block(20, 10, 4, 0.02, 0.01, 0.004, 0.0, 0.0, -0.004), m,
                               1.0, StressState::ThreeDimensional, bcs);
    Assembler assembler(model);
    RigidTool tool;
    tool.name = "tool";
    tool.shape = shape;
    tool.radius = 0.02;
    tool.normal = Vector3(0, 0, -1);
    tool.friction = mu;
    tool.surface = box(-kInf, kInf, -kInf, kInf, 0.0, kInf, "top");
    const Scalar depth = plane ? 2.0e-6 : 1.0e-4;
    const Scalar lift = plane ? 0.0 : tool.radius;
    const Vector3 drag = plane ? Vector3(std::cos(pi / 6.0), std::sin(pi / 6.0), 0.0)
                               : Vector3(1.0, 0.0, 0.0);
    const Vector3 pressed(0.005, 0.005, lift - depth);
    tool.trajectory = path({0.0, 1.0, 2.0}, {Vector3(0.005, 0.005, lift + 1.0e-4), pressed,
                                             pressed + 0.01 * drag});
    FormingOptions o;
    o.residual_tolerance = 1.0e-10;
    o.displacement_tolerance = 1.0e-10;
    o.tools.push_back(tool);
    FormingStep step;
    step.name = "iron";
    step.tools = {"tool"};
    step.max_tool_travel = 5.0e-4;
    o.steps.push_back(step);
    const FormingResult r = FormingAnalysis(model, assembler, o).run();
    REQUIRE(r.completed);
    const FormingIncrement& last = r.steps.back().increments.back();
    const ToolRecord& t = last.tools.at(0);
    const Vector3 ft(t.force.x(), t.force.y(), 0.0);  // the friction resultant on the tool
    // Each node's friction force (on the node) against the drag direction.
    Scalar direction = 0.0;
    const std::map<Index, ToolNodeHistory>& history = r.final_state.friction.at(0);
    for (const auto& [node, h] : history) {
      direction = std::max(direction, (h.force.normalized() - drag).norm());
    }
    INFO("F = " << t.force.transpose() << " N, " << t.active_nodes << " node(s), "
                << t.slipping_nodes << " slipping");
    log::info("ironing, ", to_string(shape), ": ", t.active_nodes, " node(s) in contact, ",
              t.slipping_nodes, " slipping; |F_T| / (mu F_z) - 1 = ",
              ft.norm() / (mu * t.force.z()) - 1.0, ", F_T off the drag by ",
              (ft.normalized() + drag).norm(), ", the nodes' friction forces by up to ",
              direction);
    CHECK(t.active_nodes >= 4);
    CHECK(t.slipping_nodes == t.active_nodes);
    CHECK(static_cast<int>(history.size()) == t.active_nodes);
    CHECK(t.force.z() > 0.0);    // the block pushes the tool up
    CHECK(ft.dot(drag) < 0.0);   // and resists the drag
    if (plane) {
      CHECK(ft.norm() == Approx(mu * t.force.z()).epsilon(1.0e-6));
      CHECK((ft.normalized() + drag).norm() < 1.0e-6);
      CHECK(direction < 1.0e-6);
    } else {
      CHECK(-t.force.x() / t.force.z() == Approx(mu).epsilon(3.0e-3));
      CHECK(std::abs(t.force.y()) < 1.0e-6 * t.force.z());
    }
    // The symmetric friction tangent reaches the same state (Newton's
    // method converges more slowly with it, from contact on).
    if (shape == RigidTool::Shape::Plane) {
      o.friction_tangent = FrictionTangent::Symmetric;
      const FormingResult rs = FormingAnalysis(model, assembler, o).run();
      REQUIRE(rs.completed);
      log::info("ironing, plane, symmetric friction tangent: ", rs.total_iterations,
                " iterations (exact tangent ", r.total_iterations, "), ", rs.total_cuts,
                " cuts; ", rs.linear_solver);
      CHECK(rs.total_cuts == 0);
      const ToolRecord& ts = rs.steps.back().increments.back().tools.at(0);
      CHECK((ts.force - t.force).norm() < 1.0e-8 * t.force.norm());
    }
  }
}

TEST_CASE("a tool active in consecutive steps cannot jump between them", "[forming]") {
  // A sphere pressed into a clamped block and moved along it; a second step
  // whose window starts later than the first ended would take the tool from
  // where it was to further along its path without travelling between -
  // with the penetration there ramped in as if it were a load.
  std::vector<DisplacementConstraint> bcs;
  DisplacementConstraint base;
  base.region = box(-kInf, kInf, -kInf, kInf, -kInf, -0.002, "base");
  for (int k = 0; k < 3; ++k) base.set(k, true);
  bcs.push_back(base);
  FemModel model = finalised(hex_block(10, 10, 1, 0.01, 0.01, 0.002, 0.0, 0.0, -0.002),
                             default_material(), 1.0, StressState::ThreeDimensional, bcs);
  Assembler assembler(model);
  RigidTool tool;
  tool.name = "ball";
  tool.radius = 0.004;
  tool.friction = 0.2;
  tool.surface = box(-kInf, kInf, -kInf, kInf, 0.0, kInf, "top");
  const Scalar low = tool.radius - 5.0e-5;  // a contact radius of 0.63 mm, 1 mm elements
  tool.trajectory = path({0.0, 1.0, 2.0}, {Vector3(0.003, 0.005, tool.radius + 1.0e-4),
                                           Vector3(0.003, 0.005, low),
                                           Vector3(0.007, 0.005, low)});
  FormingOptions o;
  o.tools.push_back(tool);
  FormingStep a;
  a.name = "a";
  a.tools = {"ball"};
  a.t_begin = 0.0;
  a.t_end = 1.2;
  a.max_tool_travel = 5.0e-4;
  FormingStep b = a;
  b.name = "b";
  b.t_begin = 1.8;
  b.t_end = 2.0;
  o.steps = {a, b};
  CHECK_THROWS_WITH(FormingAnalysis(model, assembler, o),
                    ContainsSubstring("step 'b' starts at t = 1.8 s") &&
                        ContainsSubstring("step 'a' ended at t = 1.2 s") &&
                        ContainsSubstring("tool 'ball', active in both, would jump 0.0024 m"));
  // Held where it was over the gap, the tool does not jump.
  {
    FormingOptions held = o;
    held.tools[0].trajectory =
        path({0.0, 1.0, 1.2, 1.8, 2.0},
             {Vector3(0.003, 0.005, tool.radius + 1.0e-4), Vector3(0.003, 0.005, low),
              Vector3(0.0038, 0.005, low), Vector3(0.0038, 0.005, low),
              Vector3(0.0046, 0.005, low)});
    CHECK_NOTHROW(FormingAnalysis(model, assembler, held));
  }
  // Nor does one taken away in between (a release), which comes back.
  {
    FormingOptions away = o;
    FormingStep release;
    release.name = "release";
    release.type = FormingStep::Type::Release;
    away.steps = {a, release, b};
    away.steps[2].t_begin = 2.5;
    away.steps[2].t_end = 3.0;
    CHECK_NOTHROW(FormingAnalysis(model, assembler, away));
  }

  // A restart knows which tools its state had active.
  FormingOptions first = o;
  first.steps = {a};
  const FormingResult ra = FormingAnalysis(model, assembler, first).run();
  REQUIRE(ra.completed);
  CHECK(ra.final_state.tools_active == std::vector<char>{1});
  FormingOptions second = o;
  second.steps = {b};
  FormingAnalysis restart(model, assembler, second);
  CHECK_THROWS_WITH(restart.run(ra.final_state),
                    ContainsSubstring("the start state is at t = 1.2 s") &&
                        ContainsSubstring("would jump"));
  // Without that record the tool becomes active in contact at t = 1.8 s:
  // the run warns that the force of its penetration is ramped in.
  AnalysisState unknown = ra.final_state;
  unknown.tools_active.clear();
  const FormingResult rb = restart.run(unknown);
  REQUIRE(rb.completed);
  REQUIRE(rb.steps[0].warnings.size() == 1);
  CHECK_THAT(rb.steps[0].warnings[0],
             ContainsSubstring("tool 'ball' becomes active in contact") &&
                 ContainsSubstring("ramped in over the step"));
  // A tool whose path starts clear of the surface is not warned about.
  CHECK(ra.steps[0].warnings.empty());
}

TEST_CASE("two opposed tools pinch a sheet with equal and opposite forces", "[forming]") {
  // Double-sided forming: a sphere on each face of a clamped elastic sheet,
  // their paths mirror images about its mid-plane, pressed in and moved
  // together. By that symmetry the two tools take equal and opposite normal
  // forces and equal friction forces, and each face touches only its own
  // tool; the clamp carries the friction - the resultant of the reactions is
  // the total force on the tools (the body's equilibrium).
  const Scalar t = 0.002;
  std::vector<DisplacementConstraint> bcs;
  DisplacementConstraint frame;
  frame.region = box(-kInf, 0.0, -kInf, kInf, -kInf, kInf, "left");
  frame.region.members.push_back(box(0.012, kInf).members[0]);
  for (int k = 0; k < 3; ++k) frame.set(k, true);
  bcs.push_back(frame);
  FemModel model = finalised(hex_block(12, 6, 4, 0.012, 0.006, t, 0.0, 0.0, -t),
                             default_material(), 1.0, StressState::ThreeDimensional, bcs);
  Assembler assembler(model);
  const Scalar r = 0.004;
  const Scalar depth = 5.0e-5;
  RigidTool top;
  top.name = "top";
  top.radius = r;
  top.friction = 0.1;
  top.surface = box(-kInf, kInf, -kInf, kInf, 0.0, kInf, "top face");
  top.trajectory = path({0.0, 1.0, 2.0}, {Vector3(0.005, 0.003, r + 1.0e-4),
                                          Vector3(0.005, 0.003, r - depth),
                                          Vector3(0.007, 0.003, r - depth)});
  RigidTool bottom = top;
  bottom.name = "bottom";
  bottom.surface = box(-kInf, kInf, -kInf, kInf, -kInf, -t, "bottom face");
  for (Vector3& p : bottom.trajectory.points) p.z() = -t - p.z();
  FormingOptions o;
  o.residual_tolerance = 1.0e-10;
  o.displacement_tolerance = 1.0e-10;
  o.tools = {top, bottom};
  FormingStep step;
  step.name = "pinch";
  step.tools = {"top", "bottom"};
  step.max_tool_travel = 5.0e-4;
  o.steps.push_back(step);
  const FormingResult res = FormingAnalysis(model, assembler, o).run();
  REQUIRE(res.completed);
  const FormingIncrement& last = res.steps[0].increments.back();
  REQUIRE(last.tools.size() == 2);
  const ToolRecord& a = last.tools[0];
  const ToolRecord& b = last.tools[1];
  INFO("top " << a.force.transpose() << " N, bottom " << b.force.transpose() << " N");
  CHECK(a.active_nodes > 0);
  CHECK(a.active_nodes == b.active_nodes);
  CHECK(a.force.z() > 0.0);
  CHECK(b.force.z() < 0.0);
  CHECK(std::abs(a.force.z() + b.force.z()) < 1.0e-6 * a.force.z());
  CHECK(std::abs(a.force.x() - b.force.x()) < 1.0e-6 * a.force.z());
  Vector3 reactions = Vector3::Zero();
  const Vector& rf = res.steps[0].reactions;
  for (Index node = 0; node < model.mesh().num_nodes(); ++node) {
    reactions += rf.segment(node * 3, 3);
  }
  CHECK((reactions - (a.force + b.force)).norm() < 1.0e-6 * a.force.z());
}

// ---------------------------------------------------------------------------
// 5. Rigid-body invariance of a release
// ---------------------------------------------------------------------------
namespace {

/// Nodes nearest to a point (one node each).
Index nearest(const Mesh& mesh, const Vector3& p) {
  Index best = 0;
  for (Index n = 1; n < mesh.num_nodes(); ++n) {
    if ((mesh.node(n) - p).norm() < (mesh.node(best) - p).norm()) best = n;
  }
  return best;
}

/// 3-2-1 supports: A fixed in x, y, z; B in y, z; C in z, holding where they
/// are, or moved to the absolute values `target` (per node, full vectors).
std::vector<StepConstraint> three_two_one(Index a, Index b, Index c, const Vector* target = nullptr,
                                          int dim = 3) {
  const auto mode = target ? StepConstraint::Mode::Absolute : StepConstraint::Mode::Hold;
  const auto value = [&](Index node) {
    Vector3 v = Vector3::Zero();
    if (target) v.head(dim) = target->segment(node * dim, dim);
    return v;
  };
  std::vector<StepConstraint> out;
  if (dim == 3) {
    out.push_back(constraint(node_set({a}, "A"), {0, 1, 2}, mode, value(a)));
    out.push_back(constraint(node_set({b}, "B"), {1, 2}, mode, value(b)));
    out.push_back(constraint(node_set({c}, "C"), {2}, mode, value(c)));
  } else {
    out.push_back(constraint(node_set({a}, "A"), {0, 1}, mode, value(a)));
    out.push_back(constraint(node_set({b}, "B"), {1}, mode, value(b)));
  }
  return out;
}

IsotropicMaterial aluminium_plastic() {
  IsotropicMaterial m(70.0e9, 0.33, 2700.0, "aluminium");
  PlasticityParameters p;
  p.yield_stress = 150.0e6;
  p.hardening_modulus = 500.0e6;
  m.set_plasticity(p);
  return m;
}

}  // namespace

TEST_CASE("a release onto 3-2-1 supports is invariant under a rigid motion of the supports",
          "[forming]") {
  const Scalar length = 0.04;
  const Scalar width = 0.006;
  const Scalar thick = 0.002;
  const Mesh mesh0 = hex_block(20, 3, 2, length, width, thick);
  const Index a = nearest(mesh0, Vector3(0, 0, 0));
  const Index b = nearest(mesh0, Vector3(length, 0, 0));
  const Index c = nearest(mesh0, Vector3(0, width, 0));

  // A stress-free body released stays where it is, and follows its supports
  // rigidly.
  {
    FemModel model = finalised(mesh0, default_material());
    Assembler assembler(model);
    FormingOptions o;
    FormingStep release;
    release.name = "release";
    release.type = FormingStep::Type::Release;
    release.constraints = three_two_one(a, b, c);
    o.steps.push_back(release);
    const FormingResult r = FormingAnalysis(model, assembler, o).run();
    REQUIRE(r.completed);
    CHECK(r.final_state.displacement.cwiseAbs().maxCoeff() == 0.0);
    const Vector3 shift(1.0e-3, -2.0e-3, 5.0e-4);
    Vector target = Vector::Zero(model.dofs().num_dofs());
    for (Index n = 0; n < mesh0.num_nodes(); ++n) target.segment(n * 3, 3) = shift;
    o.steps[0].constraints = three_two_one(a, b, c, &target);
    const FormingResult moved = FormingAnalysis(model, assembler, o).run();
    REQUIRE(moved.completed);
    CHECK((moved.final_state.displacement - target).cwiseAbs().maxCoeff() < 1.0e-12);
    // The body follows its determinate supports from the first iterate
    // (the rigid predictor): one iteration per increment.
    CHECK(moved.total_iterations == moved.total_increments);
  }

  // A plastically bent strip (finite kinematics), released from its clamp
  // twice from the same state - restarted - once holding the 3-2-1 nodes,
  // once moving them by a finite rotation and a translation: the second
  // result is the first moved rigidly.
  FemModel model = finalised(mesh0, aluminium_plastic());
  Assembler assembler(model);
  FormingOptions o;
  o.residual_tolerance = 1.0e-10;
  o.displacement_tolerance = 1.0e-10;
  FormingStep bend;
  bend.name = "bend";
  bend.increments = 5;
  bend.constraints.push_back(constraint(box(-kInf, 0.0, -kInf, kInf, -kInf, kInf, "root"),
                                        {0, 1, 2}, StepConstraint::Mode::Absolute));
  bend.constraints.push_back(constraint(box(length, kInf, -kInf, kInf, -kInf, kInf, "tip"), {2},
                                        StepConstraint::Mode::Absolute, Vector3(0, 0, 0.004)));
  o.steps.push_back(bend);
  const FormingResult formed = FormingAnalysis(model, assembler, o).run();
  REQUIRE(formed.completed);
  REQUIRE(formed.steps[0].max_plastic_strain > 1.0e-3);

  FormingOptions ro = o;
  ro.steps.clear();
  FormingStep release;
  release.name = "release";
  release.type = FormingStep::Type::Release;
  release.constraints = three_two_one(a, b, c);
  ro.steps.push_back(release);
  const FormingResult held = FormingAnalysis(model, assembler, ro).run(formed.final_state);
  REQUIRE(held.completed);
  const Vector& ua = held.final_state.displacement;
  CHECK(held.steps[0].max_displacement_change > 1.0e-4);  // it springs back
  CHECK(held.steps[0].warnings.empty());
  CHECK(held.steps[0].reaction_norm < 1.0e-8 * held.steps[0].reference_force);

  // A translation, then a finite rotation with it. (The imbalance a release
  // ramps out is a dead load, fixed in space: supports that turned the body
  // through a large angle while it springs back would change the path of
  // its unloading, and with it any reverse yielding - the end state is
  // invariant, to round-off, for rotations that do not.)
  for (const Scalar angle : {0.0, 0.03}) {
    const Matrix3 rotation = (Eigen::AngleAxisd(angle, Vector3::UnitZ()) *
                              Eigen::AngleAxisd(-0.5 * angle, Vector3(1, 1, 0).normalized()))
                                 .toRotationMatrix();
    const Vector3 shift(2.0e-3, -1.0e-3, 3.0e-3);
    Vector rigid(ua.size());
    for (Index n = 0; n < mesh0.num_nodes(); ++n) {
      const Vector3 x = mesh0.node(n) + ua.segment(n * 3, 3);
      rigid.segment(n * 3, 3) = rotation * x + shift - mesh0.node(n);
    }
    ro.steps[0].constraints = three_two_one(a, b, c, &rigid);
    const FormingResult moved = FormingAnalysis(model, assembler, ro).run(formed.final_state);
    REQUIRE(moved.completed);
    const Scalar err = (moved.final_state.displacement - rigid).cwiseAbs().maxCoeff();
    INFO("rotation " << angle << " rad: largest deviation from the rigid motion " << err
                     << " m");
    log::info("release with the supports moved rigidly (rotation ", angle,
              " rad): largest deviation from the rigid motion ", err, " m");
    CHECK(err < 1.0e-14);
    CHECK(moved.steps[0].max_plastic_strain ==
          Approx(held.steps[0].max_plastic_strain).epsilon(1e-12));
    // Moved rigidly with its supports at every increment, the body springs
    // back as the held one does, without a cut (moving the three support
    // nodes alone distorted the elements around them: 40 increments, 765
    // iterations and 2 cuts for the translation, 175 iterations with the
    // rotation, against 41 held).
    log::info("release with the supports moved rigidly (rotation ", angle, " rad): ",
              moved.steps[0].iterations, " iterations, ", moved.steps[0].cuts,
              " cuts (held: ", held.steps[0].iterations, ")");
    CHECK(moved.steps[0].cuts == 0);
    CHECK(moved.steps[0].iterations <= 2 * held.steps[0].iterations);
  }
}

// ---------------------------------------------------------------------------
// 6. Springback of an elastoplastic beam against the exact unloading
// ---------------------------------------------------------------------------
namespace {

/// The exact moment of a rectangle t x h at curvature k, elastic-perfectly
/// plastic in uniaxial stress.
Scalar bending_moment(Scalar k, Scalar youngs, Scalar sy, Scalar t, Scalar h) {
  const Scalar ky = 2.0 * sy / (youngs * h);
  const Scalar inertia = t * h * h * h / 12.0;
  if (std::abs(k) <= ky) return youngs * inertia * k;
  const Scalar mp = sy * t * h * h / 4.0;
  const Scalar ratio = ky / std::abs(k);
  return std::copysign(mp * (1.0 - ratio * ratio / 3.0), k);
}

/// The curvature of the beam's mid-depth line over its central part: twice
/// the leading coefficient of the least-squares parabola through u_y.
Scalar curvature(const Mesh& mesh, const Vector& u, Scalar length, Scalar h, Scalar margin) {
  std::vector<std::array<Scalar, 3>> rows;
  std::vector<Scalar> values;
  for (Index n = 0; n < mesh.num_nodes(); ++n) {
    const Vector3 x = mesh.node(n);
    if (std::abs(x.y() - 0.5 * h) > 1.0e-12 || x.x() < margin || x.x() > length - margin) continue;
    const Scalar s = x.x() - 0.5 * length;
    rows.push_back({1.0, s, s * s});
    values.push_back(u(3 * n + 1));
  }
  Matrix a(static_cast<Eigen::Index>(rows.size()), 3);
  Vector y(static_cast<Eigen::Index>(rows.size()));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (int k = 0; k < 3; ++k) {
      a(static_cast<Eigen::Index>(i), k) = rows[i][static_cast<std::size_t>(k)];
    }
    y(static_cast<Eigen::Index>(i)) = values[i];
  }
  const Vector coef = a.colPivHouseholderQr().solve(y);
  return 2.0 * coef(2);
}

}  // namespace

TEST_CASE("an elastoplastic beam bent past yield springs back by the exact elastic unloading",
          "[forming]") {
  // A beam L = 8 h of one Hex8 layer through its thickness t, free faces
  // (E = 200 GPa, sigma_y = 250 MPa, no hardening), small strain, bent by
  // end displacements u_x = -k1 (x - L/2)(y - h/2) to k1 = 3 k_y in step 1 -
  // the section in uniaxial stress, M(k1) = M_p (1 - (k_y/k1)^2/3) - then
  // released onto 3-2-1 supports (the left end's mid-depth node on z = 0
  // fixed in x, y and z, the right end's in y and z, the left end's on z = t
  // in y) in step 2, which ramps out the end forces. The
  // unloading is elastic (the outer fibre reverses by 1.44 sigma_y, short of
  // reverse yield at 2 sigma_y), so the curvature springs back by
  // M(k1) / (E I). Measured in the central half (the end forces' self-
  // equilibrated part decays within a depth of the ends, Saint-Venant).
  const Scalar h = 0.05;
  const Scalar length = 8.0 * h;
  const Scalar t = 0.01;
  const Scalar youngs = 200.0e9;
  const Scalar sy = 250.0e6;
  const Scalar ky = 2.0 * sy / (youngs * h);
  const Scalar k1 = 3.0 * ky;
  const Scalar inertia = t * h * h * h / 12.0;
  const Scalar springback = bending_moment(k1, youngs, sy, t, h) / (youngs * inertia);
  IsotropicMaterial material(youngs, 0.3, 7800.0, "steel");
  PlasticityParameters params;
  params.yield_stress = sy;
  material.set_plasticity(params);
  std::vector<Scalar> errors;
  for (const Index ny : {Index{4}, Index{8}, Index{16}}) {
    StructuredMeshSpec spec;
    spec.nx = 8 * ny;
    spec.ny = ny;
    spec.nz = 1;
    spec.lx = length;
    spec.ly = h;
    spec.lz = t;
    const Mesh mesh = make_structured_hex_mesh(spec);
    FemModel model = finalised(mesh, material);
    Assembler assembler(model);
    FormingOptions o;
    o.kinematics = Kinematics::SmallStrain;
    o.residual_tolerance = 1.0e-10;
    o.displacement_tolerance = 1.0e-10;
    FormingStep bend;
    bend.name = "bend";
    bend.increments = 10;
    Index left_mid = -1;
    Index left_top = -1;
    Index right_mid = -1;
    for (Index n = 0; n < mesh.num_nodes(); ++n) {
      const Vector3 x = mesh.node(n);
      const bool left = x.x() <= 1.0e-12;
      const bool right = x.x() >= length - 1.0e-12;
      if (!left && !right) continue;
      const bool mid = std::abs(x.y() - 0.5 * h) < 1.0e-12;
      const bool bottom = x.z() <= 1.0e-12;
      if (mid && bottom) (left ? left_mid : right_mid) = n;
      if (mid && left && !bottom) left_top = n;
      bend.constraints.push_back(constraint(
          node_set({n}),
          left && mid ? (bottom ? std::vector<int>{0, 1, 2} : std::vector<int>{0, 1})
                      : std::vector<int>{0},
          StepConstraint::Mode::Absolute,
          Vector3(-k1 * (x.x() - 0.5 * length) * (x.y() - 0.5 * h), 0.0, 0.0)));
    }
    o.steps.push_back(bend);
    FormingStep release;
    release.name = "release";
    release.type = FormingStep::Type::Release;
    release.constraints = {constraint(node_set({left_mid}, "A"), {0, 1, 2}),
                           constraint(node_set({right_mid}, "B"), {1, 2}),
                           constraint(node_set({left_top}, "C"), {1})};
    o.steps.push_back(release);
    const FormingResult r = FormingAnalysis(model, assembler, o).run();
    REQUIRE(r.completed);
    const Scalar margin = 2.0 * h;
    const Scalar bent = curvature(mesh, r.steps[0].displacement, length, h, margin);
    const Scalar released = curvature(mesh, r.steps[1].displacement, length, h, margin);
    const Scalar error = std::abs((bent - released) - springback) / springback;
    INFO("ny = " << ny << ": bent to " << bent / ky << " k_y, released to " << released / ky
                 << " k_y; springback " << (bent - released) / ky << " k_y, exact "
                 << springback / ky << " k_y, error " << error);
    log::info("springback, ny = ", ny, ": bent to ", bent / ky, " k_y, released to ",
              released / ky, " k_y, springback ", (bent - released) / ky, " k_y (exact ",
              springback / ky, " k_y), error ", error);
    CHECK(std::abs(bent - k1) / k1 < 1.0e-6);
    CHECK(r.steps[1].reaction_norm < 1.0e-6 * r.steps[1].reference_force);
    CHECK(r.steps[1].warnings.empty());
    errors.push_back(error);
  }
  // Converges under refinement, to within half a percent on the finest mesh.
  CHECK(errors[1] < errors[0]);
  CHECK(errors[2] < errors[1]);
  CHECK(errors[2] < 5.0e-3);
}

// ---------------------------------------------------------------------------
// 7. Single-point incremental forming, smoke test
// ---------------------------------------------------------------------------
namespace {

/// A two-level contour path of a sphere of radius r over a sheet whose top
/// is at z = 0: plunge at (r1, 0), a circle of radius r1 at depth d1, step
/// down and in, a circle of radius r2 at depth d2, retract.
ToolTrajectory contour_path(Scalar r, Scalar r1, Scalar d1, Scalar r2, Scalar d2, int segments) {
  ToolTrajectory p;
  const auto add = [&](Scalar t, const Vector3& x) {
    p.times.push_back(t);
    p.points.push_back(x);
  };
  add(0.0, Vector3(r1, 0.0, r + 5.0e-4));
  add(1.0, Vector3(r1, 0.0, r - d1));
  const Scalar pi = std::acos(-1.0);
  for (int i = 1; i <= segments; ++i) {
    const Scalar a = 2.0 * pi * i / segments;
    add(1.0 + static_cast<Scalar>(i) / segments,
        Vector3(r1 * std::cos(a), r1 * std::sin(a), r - d1));
  }
  add(2.1, Vector3(r2, 0.0, r - d2));
  for (int i = 1; i <= segments; ++i) {
    const Scalar a = 2.0 * pi * i / segments;
    add(2.1 + static_cast<Scalar>(i) / segments,
        Vector3(r2 * std::cos(a), r2 * std::sin(a), r - d2));
  }
  add(3.2, Vector3(r2, 0.0, r + 1.0e-3));
  return p;
}

}  // namespace

TEST_CASE("single-point incremental forming of a clamped blank completes and springs back",
          "[forming]") {
  // A 40 mm x 40 mm x 1 mm aluminium blank (Hex8, 2 layers, mean dilatation),
  // clamped on a 5 mm frame, formed by a 5 mm radius sphere along two
  // contour levels to 2 mm depth, the tool retracted, the blank unclamped
  // onto 3-2-1 supports at three corners. A coarse mesh (12 x 12 x 2, 3.3 mm
  // elements) and path (12 chords a contour) keep it at about 17 s on one
  // thread; configs/forming/spif_smoke.json is the same case on a 20 x 20 x 2
  // mesh (docs/forming.md, sections 4 and 5).
  IsotropicMaterial m(70.0e9, 0.33, 2700.0, "aa1050");
  PlasticityParameters p;
  p.yield_stress = 100.0e6;
  p.hardening_modulus = 300.0e6;
  m.set_plasticity(p);
  const Scalar half = 0.02;
  const Scalar t = 0.001;
  const Mesh mesh = hex_block(12, 12, 2, 2 * half, 2 * half, t, -half, -half, -t);
  FemModel model = finalised(mesh, m);
  Assembler assembler(model);
  const Scalar frame = 0.015;
  SelectorGroup clamp;
  clamp.name = "frame";
  for (const Selector& s : {box(-kInf, -frame).members[0], box(frame, kInf).members[0],
                            box(-kInf, kInf, -kInf, -frame).members[0],
                            box(-kInf, kInf, frame, kInf).members[0]}) {
    clamp.members.push_back(s);
  }
  RigidTool tool;
  tool.name = "tool";
  tool.radius = 0.005;
  tool.friction = 0.05;
  tool.surface = box(-kInf, kInf, -kInf, kInf, 0.0, kInf, "top");
  tool.trajectory = contour_path(tool.radius, 0.008, 0.001, 0.007, 0.002, 12);
  FormingOptions o;
  o.tools.push_back(tool);
  FormingStep form;
  form.name = "form";
  form.tools = {"tool"};
  form.t_begin = 0.0;
  form.t_end = 3.1;
  form.max_tool_travel = 1.6e-3;  // half the element size
  form.constraints.push_back(constraint(clamp, {0, 1, 2}));
  o.steps.push_back(form);
  FormingStep retract = form;
  retract.name = "retract";
  retract.t_begin = 3.1;
  retract.t_end = 3.2;
  o.steps.push_back(retract);
  FormingStep release;
  release.name = "unclamp";
  release.type = FormingStep::Type::Release;
  release.constraints = three_two_one(nearest(mesh, Vector3(-half, -half, 0)),
                                      nearest(mesh, Vector3(half, -half, 0)),
                                      nearest(mesh, Vector3(-half, half, 0)));
  o.steps.push_back(release);
  Timer timer;
  const FormingResult r = FormingAnalysis(model, assembler, o).run();
  const Scalar seconds = timer.elapsed_seconds();
  {
    std::ostringstream os;
    os << "SPIF smoke case: " << seconds << " s, " << r.total_increments << " increments, "
       << r.total_iterations << " iterations, " << r.total_cuts << " cuts; timing [s]:";
    for (const auto& [k, v] : r.timing.totals()) os << " " << k << " " << v;
    os << "; " << r.linear_solver;
    log::info(os.str());
  }
  REQUIRE(r.completed);
  REQUIRE(r.steps.size() == 3);
  // The centre of the formed blank lies about the contour depth down.
  const Index centre = nearest(mesh, Vector3(0, 0, 0));
  const Scalar formed = r.steps[1].displacement(centre * 3 + 2);
  const Scalar released = r.steps[2].displacement(centre * 3 + 2);
  log::info("SPIF smoke case: centre depth ", formed, " m formed, ", released,
            " m released; largest plastic strain ", r.steps[2].max_plastic_strain,
            "; springback (largest displacement change of the release) ",
            r.steps[2].max_displacement_change, " m");
  CHECK(formed < -0.5e-3);
  CHECK(formed > -3.0e-3);
  CHECK(r.steps[2].max_displacement_change > 1.0e-6);  // it springs back
  CHECK(r.steps[0].max_plastic_strain > 0.01);
  // The tool is pushed up by the sheet at every increment in contact.
  int in_contact = 0;
  for (const FormingIncrement& inc : r.steps[0].increments) {
    if (inc.tools.at(0).active_nodes == 0) continue;
    ++in_contact;
    CHECK(inc.tools[0].force.z() > 0.0);
  }
  CHECK(in_contact > 50);
}

// ---------------------------------------------------------------------------
// The `forming` block of a deck, and the result files
// ---------------------------------------------------------------------------
namespace {

/// A 4 mm x 4 mm x 2 mm aluminium block pressed by a flat punch, then
/// released onto 3-2-1 supports: the `forming` block with every key, and
/// the rest of the deck without load cases (a forming deck needs none).
std::string forming_deck(const std::string& forming) {
  return R"({
    "name": "tiny forming",
    "mesh": {"type": "structured_hex", "nx": 2, "ny": 2, "nz": 2,
             "lx": 0.004, "ly": 0.004, "lz": 0.002},
    "material": {"youngs_modulus": 70e9, "poisson_ratio": 0.33, "density": 2700,
                 "plasticity": {"yield_stress": 100e6, "hardening_modulus": 300e6}},
    "boundary_conditions": [
      {"name": "base", "fix": ["z"], "region": {"box": {"zmax": 0.0}}},
      {"name": "corner", "fix": ["x", "y"], "region": {"nearest_node": [0, 0, 0]}},
      {"name": "edge", "fix": ["y"], "region": {"nearest_node": [0.004, 0, 0]}}],
    "forming": )" + forming + "}";
}

const char* kFullForming = R"({
      "kinematics": "finite",
      "mean_dilatation": "auto",
      "friction_tangent": "exact",
      "solver": "eigen",
      "tools": [
        {"name": "punch", "shape": "plane", "normal": [0, 0, -1],
         "surface": {"box": {"zmin": 0.002}}, "friction": 0.1, "penalty": 20,
         "tangential_penalty": 0.5,
         "trajectory": {"times": [0, 1, 1.5], "points": [[0, 0, 0.002], [0, 0, 0.00196],
                                                           [0, 0, 0.0021]]}},
        {"name": "ball", "shape": "sphere", "radius": 0.003,
         "surface": {"box": {"zmin": 0.002}},
         "trajectory": {"times": [0, 1], "points": [[0, 0, 0.01], [0, 0, 0.01]]}}],
      "steps": [
        {"name": "press", "type": "form", "tools": ["punch"], "time": [0, 1],
         "max_tool_travel": 1e-5, "increments": 2},
        {"name": "lift", "type": "form", "tools": ["punch"], "time": [1, 1.5]},
        {"name": "release", "type": "release", "increments": 4,
         "boundary_conditions": [
           {"name": "A", "fix": ["x", "y", "z"], "mode": "hold",
            "region": {"nearest_node": [0, 0, 0]}},
           {"name": "B", "fix": ["y", "z"], "region": {"nearest_node": [0.004, 0, 0]}},
           {"name": "C", "fix": ["z"], "mode": "hold",
            "region": {"nearest_node": [0, 0.004, 0]}}]}],
      "newton": {"max_iterations": 20, "residual_tolerance": 1e-8,
                 "displacement_tolerance": 1e-8, "line_search": true, "max_cuts": 6,
                 "max_increments": 1000},
      "output": {"vtk": true, "snapshots": 2}
    })";

Configuration parse_deck(const std::string& text, bool strict = true,
                         const std::string& base = "") {
  return parse_configuration(json::parse(text, "deck"), "deck", strict, base);
}

}  // namespace

TEST_CASE("the forming block of a deck is read in full, strictly, and refused when wrong",
          "[forming][io]") {
  const Configuration c = parse_deck(forming_deck(kFullForming));
  REQUIRE(c.forming.enabled);
  const FormingOptions& o = c.forming.options;
  CHECK(o.kinematics == Kinematics::Finite);
  CHECK(o.friction_tangent == FrictionTangent::Exact);
  CHECK_FALSE(o.suitesparse);
  REQUIRE(o.tools.size() == 2);
  CHECK(o.tools[0].shape == RigidTool::Shape::Plane);
  CHECK(o.tools[0].friction == 0.1);
  CHECK(o.tools[0].penalty == 20.0);
  CHECK(o.tools[0].tangential_ratio == 0.5);
  CHECK(o.tools[0].trajectory.times.size() == 3);
  CHECK(o.tools[1].radius == 0.003);
  REQUIRE(o.steps.size() == 3);
  CHECK(o.steps[0].t_begin == 0.0);
  CHECK(o.steps[0].t_end == 1.0);
  CHECK(o.steps[0].max_tool_travel == 1e-5);
  CHECK(o.steps[0].increments == 2);
  CHECK(o.steps[2].type == FormingStep::Type::Release);
  REQUIRE(o.steps[2].constraints.size() == 3);
  CHECK(o.steps[2].constraints[1].mode == StepConstraint::Mode::Hold);  // the default
  CHECK(o.max_iterations == 20);
  CHECK(o.max_cuts == 6);
  CHECK(o.snapshot_stride == 2);
  CHECK(c.forming.snapshots == FormingConfig::Snapshots::Stride);
  // No load case in the deck: one empty case is supplied.
  REQUIRE(c.load_cases.size() == 1);
  CHECK(c.load_cases[0].name == "forming");

  // A trajectory file resolves against the deck's directory.
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "sparlab_test_forming_deck";
  std::filesystem::create_directories(dir);
  std::ofstream((dir / "path.csv").string()) << "t,x,y,z\n0,0,0,0.01\n1,0,0,0.0049\n";
  std::string with_file = kFullForming;
  const std::string inline_path =
      R"("trajectory": {"times": [0, 1], "points": [[0, 0, 0.01], [0, 0, 0.01]]})";
  REQUIRE(with_file.find(inline_path) != std::string::npos);
  with_file.replace(with_file.find(inline_path), inline_path.size(),
                    R"("trajectory": {"file": "path.csv"})");
  const Configuration f = parse_deck(forming_deck(with_file), true, dir.string());
  CHECK(f.forming.options.tools[1].trajectory.points[1].z() == 0.0049);
  std::filesystem::remove_all(dir);

  // Refusals, each naming what is wrong.
  const auto refuse = [&](const std::string& from, const std::string& to,
                          const std::string& message) {
    std::string text = kFullForming;
    const std::size_t at = text.find(from);
    REQUIRE(at != std::string::npos);
    text.replace(at, from.size(), to);
    INFO(message);
    CHECK_THROWS_WITH(parse_deck(forming_deck(text)), ContainsSubstring(message));
  };
  refuse(R"("tools": ["punch"], "time": [0, 1])", R"("tools": ["hammer"], "time": [0, 1])",
         "'hammer', which 'forming.tools' does not define");
  refuse(R"("type": "release", "increments": 4,)",
         R"("type": "release", "increments": 4, "tools": ["ball"],)",
         "a release that keeps tool 'ball'");
  refuse(R"("radius": 0.003)", R"("radius": 0.0)", "'radius' must be positive");
  refuse(R"("friction": 0.1)", R"("friction": -0.1)", "'friction' must be >= 0");
  refuse(R"("time": [1, 1.5])", R"("time": [0.5, 1.5])", "cannot run backwards");
  refuse(R"("penalty": 20,)", R"("penalty": 20, "penlaty": 3,)", "penlaty");  // strict
  refuse(R"("trajectory": {"times": [0, 1, 1.5])",
         R"("trajectory": {"file": "x.csv", "times": [0, 1, 1.5])", "not both");
  refuse(R"("mode": "hold",
            "region": {"nearest_node": [0, 0, 0]})", R"("mode": "clamp",
            "region": {"nearest_node": [0, 0, 0]})", "expected \"hold\" or \"absolute\"");
  refuse(R"("snapshots": 2)", R"("snapshots": "all")", "'forming.output.snapshots'");
  // A value on a held constraint would be ignored (the top level prescribes it).
  refuse(R"({"name": "B", "fix": ["y", "z"],)",
         R"({"name": "B", "fix": ["y", "z"], "value": [0, 0, 0.001],)",
         "'forming.steps[2].boundary_conditions[1].value' is not zero, but the constraint's "
         "mode is \"hold\"");
  // Zero - "fixed" - is accepted with hold (decks write it for supports), and
  // so is a value of a component the constraint does not fix.
  for (const std::string value : {"[0, 0, 0]", "[0.001, 0, 0]"}) {
    std::string text = kFullForming;
    const std::string from = R"({"name": "B", "fix": ["y", "z"],)";
    text.replace(text.find(from), from.size(),
                 R"({"name": "B", "fix": ["y", "z"], "value": )" + value + ",");
    CHECK_NOTHROW(parse_deck(forming_deck(text)));
  }
  refuse(R"({"name": "punch", "shape")", R"({"name": "punch,1", "shape")",
         "tool name 'punch,1' must be non-empty");
  {
    std::string text = kFullForming;
    const std::string from = R"({"name": "B", "fix": ["y", "z"],)";
    text.replace(text.find(from), from.size(),
                 R"({"name": "B", "fix": ["y", "z"], "mode": "absolute", "value": [0, 0, 0.001],)");
    const Configuration a = parse_deck(forming_deck(text));
    CHECK(a.forming.options.steps[2].constraints[1].constraint.value(2) == 0.001);
    // The API refuses a held constraint with a value too.
    FormingOptions held = a.forming.options;
    held.steps[2].constraints[1].mode = StepConstraint::Mode::Hold;
    FemModel held_model = build_model(a);
    Assembler held_assembler(held_model);
    CHECK_THROWS_WITH(FormingAnalysis(held_model, held_assembler, held),
                      ContainsSubstring("step 'release', constraint 'B' holds its DOFs") &&
                          ContainsSubstring("use mode absolute"));
  }
  refuse(R"("friction_tangent": "exact")", R"("friction_tangent": "approximate")",
         "expected \"exact\" or \"symmetric\"");
  {
    std::string text = kFullForming;
    text.replace(text.find(R"("exact")"), 7, R"("symmetric")");
    CHECK(parse_deck(forming_deck(text)).forming.options.friction_tangent ==
          FrictionTangent::Symmetric);
  }

  // A surface that selects no face is refused when the analysis is built.
  std::string empty = kFullForming;
  empty.replace(empty.find(R"("surface": {"box": {"zmin": 0.002}}, "friction")"),
                std::string(R"("surface": {"box": {"zmin": 0.002}}, "friction")").size(),
                R"("surface": {"box": {"zmin": 0.5}}, "friction")");
  const Configuration e = parse_deck(forming_deck(empty));
  FemModel model = build_model(e);
  Assembler assembler(model);
  CHECK_THROWS_WITH(FormingAnalysis(model, assembler, e.forming.options),
                    ContainsSubstring("selects no boundary face"));
  // So is a step whose constraints leave the body free to move.
  std::string loose = kFullForming;
  loose.replace(loose.find(R"({"name": "C", "fix": ["z"])"),
                std::string(R"({"name": "C", "fix": ["z"])").size(),
                R"({"name": "C", "fix": ["x"])");
  const Configuration l = parse_deck(forming_deck(loose));
  FemModel loose_model = build_model(l);
  Assembler loose_assembler(loose_model);
  CHECK_THROWS_WITH(FormingAnalysis(loose_model, loose_assembler, l.forming.options),
                    ContainsSubstring("step 'release'") &&
                        ContainsSubstring("rotation about x"));
}

TEST_CASE("a forming run writes the result files of its output contract", "[forming][io]") {
  const Configuration c = parse_deck(forming_deck(kFullForming));
  FemModel model = build_model(c);
  Assembler assembler(model);
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "sparlab_test_forming_out";
  std::filesystem::remove_all(dir);
  ResultWriter writer(dir.string(), c);
  // As sparlab_form does: each step's files, and the tool forces so far,
  // written as the step ends.
  FormingAnalysis analysis(model, assembler, c.forming.options);
  std::vector<std::string> files;
  std::vector<std::size_t> seen;
  std::vector<int> tool_rows;
  analysis.set_step_observer([&](const FormingResult& so_far) {
    const std::size_t k = so_far.steps.size() - 1;
    seen.push_back(so_far.steps.size());
    for (std::string& f :
         write_forming_step(writer, model, k, so_far.steps[k], true, true)) {
      files.push_back(std::move(f));
    }
    write_tool_forces(writer, model, c.forming.options, so_far);
    std::ifstream in((dir / "tool_forces.csv").string());
    std::string line;
    int count = 0;
    while (std::getline(in, line)) ++count;
    tool_rows.push_back(count);
  });
  const FormingResult r = analysis.run();
  REQUIRE(r.completed);
  REQUIRE(r.steps.size() == 3);
  CHECK(seen == std::vector<std::size_t>{1, 2, 3});
  CHECK(r.timing.totals().count("output") == 1);
  // The tool forces grow step by step: 1 + the press's increments, then
  // the lift's too (the release has no tool).
  REQUIRE(tool_rows.size() == 3);
  CHECK(tool_rows[0] == 1 + static_cast<int>(r.steps[0].increments.size()));
  CHECK(tool_rows[1] ==
        1 + static_cast<int>(r.steps[0].increments.size() + r.steps[1].increments.size()));
  CHECK(tool_rows[2] == tool_rows[1]);
  CHECK(r.steps[2].reaction_norm < 1.0e-6 * r.steps[2].reference_force);
  // The same files as writing them all after the run.
  const std::vector<std::string> after =
      write_forming_files(writer, model, c.forming.options, r, true, true);
  std::vector<std::string> expected = files;
  expected.push_back("tool_forces.csv");
  CHECK(after == expected);
  files = after;
  const auto first_line = [&](const std::string& name) {
    std::ifstream in((dir / name).string());
    std::string line;
    std::getline(in, line);
    return line;
  };
  const auto lines = [&](const std::string& name) {
    std::ifstream in((dir / name).string());
    std::string line;
    int count = 0;
    while (std::getline(in, line)) ++count;
    return count;
  };
  for (const std::string stem : {"step_1_press", "step_2_lift", "step_3_release"}) {
    INFO(stem);
    CHECK(first_line(stem + "_nodes.csv") == "node,X,Y,Z,ux,uy,uz");
    CHECK(lines(stem + "_nodes.csv") == 1 + model.mesh().num_nodes());
    CHECK(first_line(stem + "_elements.csv") == "element,eq_plastic_strain,von_mises_Pa");
    CHECK(lines(stem + "_elements.csv") == 1 + model.mesh().num_elements());
    CHECK(std::filesystem::exists(dir / (stem + ".vtk")));
  }
  CHECK(first_line("tool_forces.csv") ==
        "step,increment,t,tool,cx,cy,cz,fx,fy,fz,active_nodes,max_penetration_m");
  int records = 0;
  for (const FormingStepResult& s : r.steps) {
    for (const FormingIncrement& inc : s.increments) records += static_cast<int>(inc.tools.size());
  }
  CHECK(lines("tool_forces.csv") == 1 + records);
  // A snapshot every 2 increments (the stride), step ends excluded.
  CHECK(std::filesystem::exists(dir / "step_3_release_inc_2_nodes.csv"));
  CHECK(std::find(files.begin(), files.end(), "step_3_release_inc_2_nodes.csv") != files.end());

  const json::Value summary =
      forming_summary_json(c, model, c.forming.options, r, 1.5, "test", files);
  for (const char* key : {"case", "sparlab_version", "completed", "termination", "runtime_s",
                          "timing", "steps", "tools", "mesh"}) {
    CHECK(summary.find(key) != nullptr);
  }
  const json::Value& steps = *summary.find("steps");
  REQUIRE(steps.array_items().size() == 3);
  for (const char* key : {"name", "type", "completed", "increments", "iterations", "cuts",
                          "max_plastic_strain", "reaction_norm_N", "warnings"}) {
    CHECK(steps.array_items()[0].find(key) != nullptr);
  }
  CHECK(steps.array_items()[2].find("type")->string_value() == "release");
  // The factorisations used, and apart from them those that failed.
  const json::Value& timing = *summary.find("timing");
  REQUIRE(timing.find("linear_solver") != nullptr);
  REQUIRE(timing.find("failed_factorisations") != nullptr);
  CHECK(timing.find("linear_solver")->string_value() == r.linear_solver);
  CHECK(timing.find("failed_factorisations")->string_value() == r.failed_factorisations);
  CHECK(r.linear_solver.find('(') == std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST_CASE("a step whose start state cannot be evaluated stops the run with its state recorded",
          "[forming][io]") {
  // A sphere whose path starts with its centre on the top face: its first
  // evaluation throws (a node closer than half the radius). The step is
  // recorded, not completed, with the state it started from, and the result
  // files of a stopped run are written (summary.json among them).
  const Configuration c = parse_deck(forming_deck(R"({
      "tools": [
        {"name": "ball", "shape": "sphere", "radius": 0.003,
         "surface": {"box": {"zmin": 0.002}},
         "trajectory": {"times": [0, 1], "points": [[0.002, 0.002, 0.002],
                                                      [0.002, 0.002, 0.003]]}}],
      "steps": [
        {"name": "press", "type": "form", "tools": ["ball"]},
        {"name": "release", "type": "release"}]
    })"));
  FemModel model = build_model(c);
  Assembler assembler(model);
  FormingAnalysis analysis(model, assembler, c.forming.options);
  int observed = 0;
  analysis.set_step_observer([&](const FormingResult& so_far) {
    ++observed;
    CHECK_FALSE(so_far.steps.back().completed);  // told of the stopped step too
  });
  const FormingResult r = analysis.run();
  CHECK(observed == 1);
  CHECK_FALSE(r.completed);
  REQUIRE(r.steps.size() == 1);
  const FormingStepResult& s = r.steps[0];
  CHECK_FALSE(s.completed);
  CHECK_THAT(s.termination, ContainsSubstring("start state (t = 0 s) cannot be evaluated") &&
                                ContainsSubstring("tool 'ball'"));
  CHECK_THAT(r.termination, ContainsSubstring("step 'press' (1 of 2)"));
  CHECK(s.increments.empty());
  REQUIRE(s.displacement.size() == model.dofs().num_dofs());
  CHECK(s.displacement.cwiseAbs().maxCoeff() == 0.0);
  CHECK(s.reactions.size() == model.dofs().num_dofs());
  CHECK(s.element_von_mises.size() == model.mesh().num_elements());
  CHECK(r.final_state.displacement.size() == model.dofs().num_dofs());

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "sparlab_test_forming_stopped";
  std::filesystem::remove_all(dir);
  ResultWriter writer(dir.string(), c);
  const std::vector<std::string> files =
      write_forming_files(writer, model, c.forming.options, r, true, true);
  CHECK(files == std::vector<std::string>{"tool_forces.csv"});  // no completed step
  const json::Value summary =
      forming_summary_json(c, model, c.forming.options, r, 0.1, "test", files);
  CHECK_FALSE(summary.find("completed")->bool_value());
  const json::Value& steps = *summary.find("steps");
  REQUIRE(steps.array_items().size() == 1);
  CHECK(steps.array_items()[0].find("max_displacement_m")->number_value() == 0.0);
  std::filesystem::remove_all(dir);
}
