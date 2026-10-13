#include "sparlab/fem/RigidTool.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/FaceGeometry.hpp"
#include "sparlab/fem/BoundaryConditions.hpp"
#include "sparlab/fem/FemModel.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace sparlab {
namespace {

/// A candidate tangent that vanishes (a committed friction force turned
/// normal to the tool) is taken as zero below this fraction of its length.
constexpr Scalar kVanishing = 1.0e-12;

std::string trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r");
  if (first == std::string::npos) return "";
  const auto last = text.find_last_not_of(" \t\r");
  return text.substr(first, last - first + 1);
}

std::vector<std::string> split_fields(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream in(line);
  while (std::getline(in, field, ',')) fields.push_back(trim(field));
  if (!line.empty() && line.back() == ',') fields.emplace_back();
  return fields;
}

/// Distance from p to the segment [a, b].
Scalar segment_distance(const Vector3& p, const Vector3& a, const Vector3& b) {
  const Vector3 s = b - a;
  const Scalar len2 = s.squaredNorm();
  Scalar t = 0.0;
  if (len2 > 0.0) t = std::clamp((p - a).dot(s) / len2, 0.0, 1.0);
  return (p - a - t * s).norm();
}

Vector3 planar(Vector3 v, int dim) {
  if (dim == 2) v.z() = 0.0;
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// ToolTrajectory
// ---------------------------------------------------------------------------
void ToolTrajectory::validate(const std::string& what) const {
  if (times.size() != points.size()) {
    throw ConfigError(what + ": " + std::to_string(times.size()) + " time(s) but " +
                      std::to_string(points.size()) + " point(s)");
  }
  if (times.size() < 2) {
    throw ConfigError(what + " needs at least two knots (time and point)");
  }
  for (std::size_t i = 0; i < times.size(); ++i) {
    if (!std::isfinite(times[i]) || !points[i].allFinite()) {
      throw ConfigError(what + ": knot " + std::to_string(i + 1) + " is not finite");
    }
    if (i > 0 && !(times[i] > times[i - 1])) {
      std::ostringstream os;
      os << what << ": the times must increase strictly, but knot " << i + 1 << " (t = "
         << times[i] << " s) does not follow knot " << i << " (t = " << times[i - 1] << " s)";
      throw ConfigError(os.str());
    }
  }
}

std::size_t ToolTrajectory::segment(Scalar t) const {
  const std::size_t last = times.size() - 2;  // index of the last segment
  std::size_t i = std::min(hint_, last);
  if (times[i] <= t && t < times[i + 1]) return i;
  // Monotone use: the next segment.
  if (i < last && times[i + 1] <= t && t < times[i + 2]) {
    hint_ = i + 1;
    return i + 1;
  }
  const auto it = std::upper_bound(times.begin(), times.end(), t);
  i = static_cast<std::size_t>(std::max<std::ptrdiff_t>(it - times.begin() - 1, 0));
  i = std::min(i, last);
  hint_ = i;
  return i;
}

Vector3 ToolTrajectory::position(Scalar t) const {
  if (t <= times.front()) return points.front();
  if (t >= times.back()) return points.back();
  const std::size_t i = segment(t);
  const Scalar w = (t - times[i]) / (times[i + 1] - times[i]);
  return (1.0 - w) * points[i] + w * points[i + 1];
}

Vector3 ToolTrajectory::velocity(Scalar t) const {
  if (t < times.front() || t >= times.back()) return Vector3::Zero();
  const std::size_t i = segment(t);
  return (points[i + 1] - points[i]) / (times[i + 1] - times[i]);
}

std::vector<Scalar> ToolTrajectory::knots_in(Scalar ta, Scalar tb) const {
  std::vector<Scalar> out;
  for (Scalar t : times) {
    if (t > ta && t < tb) out.push_back(t);
  }
  return out;
}

ToolTrajectory ToolTrajectory::from_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw IoError("cannot read the trajectory file '" + path + "'");
  ToolTrajectory out;
  std::string line;
  int number = 0;
  bool header = false;
  while (std::getline(in, line)) {
    ++number;
    const std::string text = trim(line);
    if (text.empty() || text[0] == '#') continue;
    const std::vector<std::string> fields = split_fields(text);
    const std::string where = "trajectory file '" + path + "', line " + std::to_string(number);
    if (!header) {
      if (fields.size() != 4 || fields[0] != "t" || fields[1] != "x" || fields[2] != "y" ||
          fields[3] != "z") {
        throw ConfigError(where + ": expected the header 't,x,y,z' (pseudo-time in s, the "
                                  "tool's reference point in m), got '" + text + "'");
      }
      header = true;
      continue;
    }
    if (fields.size() != 4) {
      throw ConfigError(where + ": expected 4 values (t, x, y, z), got " +
                        std::to_string(fields.size()));
    }
    Scalar values[4];
    for (int k = 0; k < 4; ++k) {
      const std::string& f = fields[static_cast<std::size_t>(k)];
      std::size_t used = 0;
      bool ok = !f.empty();
      if (ok) {
        try {
          values[k] = std::stod(f, &used);
        } catch (const std::exception&) {
          ok = false;
        }
      }
      if (!ok || used != f.size() || !std::isfinite(values[k])) {
        throw ConfigError(where + ": field " + std::to_string(k + 1) + " ('" + f +
                          "') is not a finite number");
      }
    }
    out.times.push_back(values[0]);
    out.points.emplace_back(values[1], values[2], values[3]);
  }
  if (!header) throw ConfigError("trajectory file '" + path + "' is empty (no 't,x,y,z' header)");
  out.validate("trajectory file '" + path + "'");
  return out;
}

// ---------------------------------------------------------------------------
// RigidTool
// ---------------------------------------------------------------------------
std::string to_string(RigidTool::Shape shape) {
  switch (shape) {
    case RigidTool::Shape::Sphere: return "sphere";
    case RigidTool::Shape::Plane: return "plane";
    case RigidTool::Shape::Cylinder: return "cylinder";
  }
  return "sphere";
}

RigidTool::Shape parse_tool_shape(const std::string& text) {
  if (text == "sphere") return RigidTool::Shape::Sphere;
  if (text == "plane") return RigidTool::Shape::Plane;
  if (text == "cylinder") return RigidTool::Shape::Cylinder;
  throw ConfigError("unknown tool shape '" + text +
                    "'; expected \"sphere\", \"plane\" or \"cylinder\"");
}

std::string to_string(FrictionTangent mode) {
  return mode == FrictionTangent::Symmetric ? "symmetric" : "exact";
}

FrictionTangent parse_friction_tangent(const std::string& text) {
  if (text == "exact") return FrictionTangent::Exact;
  if (text == "symmetric") return FrictionTangent::Symmetric;
  throw ConfigError("unknown friction_tangent '" + text +
                    "'; expected \"exact\" or \"symmetric\"");
}

void RigidTool::validate(int dim) const {
  const std::string label = "tool '" + name + "'";
  // The name is a field of tool_forces.csv (and names the tool in the
  // summary): no separator, quote or space to escape.
  const bool plain = !name.empty() && std::all_of(name.begin(), name.end(), [](char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_' || ch == '-' ||
           ch == '.';
  });
  if (!plain) {
    throw ConfigError("tool name '" + name + "' must be non-empty and use only letters, digits, "
                      "'_', '-' and '.' (it is a field of tool_forces.csv)");
  }
  if (shape != Shape::Plane && !(radius > 0.0 && std::isfinite(radius))) {
    throw ConfigError(label + ": the " + to_string(shape) + " needs a positive radius");
  }
  if (shape == Shape::Plane && !(planar(normal, dim).norm() > 0.0)) {
    throw ConfigError(label + ": the plane needs a non-zero normal" +
                      std::string(dim == 2 ? " in the model plane" : ""));
  }
  if (shape == Shape::Cylinder && dim == 3 && !(axis.norm() > 0.0)) {
    throw ConfigError(label + ": the cylinder needs a non-zero axis");
  }
  if (!(friction >= 0.0) || !std::isfinite(friction)) {
    throw ConfigError(label + ": the friction coefficient must be >= 0");
  }
  if (!(penalty > 0.0) || !std::isfinite(penalty)) {
    throw ConfigError(label + ": the penalty scale must be positive");
  }
  if (!(tangential_ratio > 0.0) || !std::isfinite(tangential_ratio)) {
    throw ConfigError(label + ": the tangential penalty ratio must be positive");
  }
  trajectory.validate(label + "'s trajectory");
}

Scalar RigidTool::gap(const Vector3& x, const Vector3& c, int dim, Vector3& n, Matrix3& dn,
                      Scalar& d) const {
  switch (shape) {
    case Shape::Plane: {
      n = planar(normal, dim).normalized();
      dn.setZero();
      d = 0.0;
      return n.dot(planar(x - c, dim));
    }
    case Shape::Sphere:
    case Shape::Cylinder: {
      Vector3 r = planar(x - c, dim);
      Matrix3 base = Matrix3::Identity();
      if (shape == Shape::Cylinder && dim == 3) {
        const Vector3 a = axis.normalized();
        r -= a.dot(r) * a;
        base -= a * a.transpose();
      }
      d = r.norm();
      if (!(d > 0.5 * radius)) {
        std::ostringstream os;
        os << "a node lies " << d << " m from the " << to_string(shape) << "'s "
           << (shape == Shape::Sphere ? "centre" : "axis") << " (radius " << radius
           << " m) of tool '" << name
           << "', less than half the radius: the tool is through the surface";
        throw SolverError(os.str());
      }
      n = r / d;
      dn = (base - n * n.transpose()) / d;
      return d - radius;
    }
  }
  return 0.0;
}

// ---------------------------------------------------------------------------
// ToolContact: construction
// ---------------------------------------------------------------------------
ToolContact::ToolContact(const FemModel& model, std::vector<RigidTool> tools,
                         FrictionTangent friction_tangent)
    : model_(model), tools_(std::move(tools)), friction_tangent_(friction_tangent) {
  const Mesh& mesh = model.mesh();
  const int dim = mesh.dim();
  const ElementType type = mesh.element_type();
  std::set<std::string> names;
  for (const RigidTool& t : tools_) {
    t.validate(dim);
    if (!names.insert(t.name).second) {
      throw ConfigError("tool name '" + t.name + "' is used twice");
    }
  }
  const FaceShape shape = face_shape_of(type);
  const Scalar thickness = dim == 2 ? model.thickness() : 1.0;
  const std::vector<Mesh::BoundaryFace> boundary = mesh.boundary_faces();
  min_size_ = std::numeric_limits<Scalar>::infinity();
  for (const RigidTool& t : tools_) {
    const std::vector<Mesh::BoundaryFace> faces = faces_in_region(mesh, boundary, t.surface);
    if (faces.empty()) {
      throw ConfigError("tool '" + t.name + "': its surface region selects no boundary face "
                        "(every node of a face must lie in it)");
    }
    std::map<Index, Scalar> area;
    std::map<Index, Scalar> modulus;
    const std::vector<std::vector<int>>& table = element_local_faces(type);
    for (const Mesh::BoundaryFace& f : faces) {
      const Matrix coords = element_face_coordinates(mesh, f.element, f.local_face);
      const Vector w = face_shape_integrals(shape, coords, 3);
      const Index* en = mesh.element_nodes(f.element);
      const std::vector<int>& local = table[static_cast<std::size_t>(f.local_face)];
      const Scalar e = model.material_of(f.element).youngs_modulus();
      for (std::size_t a = 0; a < local.size(); ++a) {
        const Index node = en[local[a]];
        area[node] += w(static_cast<Eigen::Index>(a));
        modulus[node] = std::max(modulus[node], e);
      }
    }
    Slaves s;
    for (const auto& [node, a] : area) {
      if (!(a > 0.0)) continue;
      const Scalar h = dim == 2 ? a / thickness : std::sqrt(a);
      s.nodes.push_back(node);
      s.area.push_back(a);
      s.kappa.push_back(t.penalty * modulus[node] / h);
      min_size_ = std::min(min_size_, h);
    }
    s.candidates.resize(s.nodes.size());
    for (std::size_t i = 0; i < s.nodes.size(); ++i) s.candidates[i] = static_cast<Index>(i);
    slaves_.push_back(std::move(s));
  }
  history_.assign(tools_.size(), {});
  // Until an increment is prepared: from the reference state at the start
  // of each trajectory, every node a candidate.
  start_ = Vector::Zero(model.dofs().num_dofs());
  t0_ = tools_.empty() ? 0.0 : tools_.front().trajectory.start();
  margin_ = std::numeric_limits<Scalar>::infinity();
  active_.assign(tools_.size(), 1);
}

int ToolContact::find(const std::string& name) const {
  for (std::size_t k = 0; k < tools_.size(); ++k) {
    if (tools_[k].name == name) return static_cast<int>(k);
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Increment set-up: the candidate nodes
// ---------------------------------------------------------------------------
void ToolContact::begin_increment(const Vector& u0, Scalar t0, Scalar t1,
                                  const std::vector<char>& active, Scalar margin) {
  const Mesh& mesh = model_.mesh();
  const int dim = mesh.dim();
  if (active.size() != tools_.size()) {
    throw ModelError("tool contact: one activity flag per tool is needed");
  }
  start_ = u0;
  t0_ = t0;
  margin_ = margin;
  active_ = active;
  for (std::size_t k = 0; k < tools_.size(); ++k) {
    Slaves& s = slaves_[k];
    s.candidates.clear();
    if (!active_[k]) continue;
    const RigidTool& tool = tools_[k];
    const Vector3 c0 = planar(tool.trajectory.position(t0), dim);
    const Vector3 c1 = planar(tool.trajectory.position(t1), dim);
    Vector3 a = Vector3::UnitZ();
    if (tool.shape == RigidTool::Shape::Cylinder && dim == 3) a = tool.axis.normalized();
    const Vector3 plane_normal = planar(tool.normal, dim).normalized();
    for (std::size_t i = 0; i < s.nodes.size(); ++i) {
      const Index node = s.nodes[i];
      Vector3 x = mesh.node(node);
      for (int c = 0; c < dim; ++c) x(c) += u0(node * dim + c);
      x = planar(x, dim);
      bool candidate = false;
      switch (tool.shape) {
        case RigidTool::Shape::Sphere:
          candidate = segment_distance(x, c0, c1) <= tool.radius + margin;
          break;
        case RigidTool::Shape::Cylinder: {
          if (dim == 3) {
            const Matrix3 p = Matrix3::Identity() - a * a.transpose();
            candidate = segment_distance(p * (x - c0), Vector3::Zero(), p * (c1 - c0)) <=
                        tool.radius + margin;
          } else {
            candidate = segment_distance(x, c0, c1) <= tool.radius + margin;
          }
          break;
        }
        case RigidTool::Shape::Plane:
          candidate = std::min(plane_normal.dot(x - c0), plane_normal.dot(x - c1)) <= margin;
          break;
      }
      if (candidate) s.candidates.push_back(static_cast<Index>(i));
    }
  }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
ToolContactEvaluation ToolContact::evaluate(const Vector& u, Scalar t, bool want_tangent) const {
  const Mesh& mesh = model_.mesh();
  const int dim = mesh.dim();
  const Index n = model_.dofs().num_dofs();
  ToolContactEvaluation out;
  out.residual = Vector::Zero(n);
  out.tools.assign(tools_.size(), ToolResultant());
  out.trial.assign(tools_.size(), {});
  Vector gross = Vector::Zero(n);
  Vector positions = Vector::Zero(n);
  const Matrix3 identity = Matrix3::Identity();
  std::vector<Index> all;
  for (std::size_t k = 0; k < tools_.size(); ++k) {
    const RigidTool& tool = tools_[k];
    const Slaves& s = slaves_[k];
    ToolResultant& res = out.tools[k];
    const Vector3 c = planar(tool.trajectory.position(t), dim);
    res.centre = tool.trajectory.position(t);
    if (!active_[k]) continue;
    // The candidates hold only while no node has moved farther than the
    // margin (see the file comment); beyond it, every slave node.
    const std::vector<Index>* nodes = &s.candidates;
    Scalar moved = 0.0;
    for (Index node : s.nodes) {
      Scalar m2 = 0.0;
      for (int comp = 0; comp < dim; ++comp) {
        const Scalar du = u(node * dim + comp) - start_(node * dim + comp);
        m2 += du * du;
      }
      moved = std::max(moved, m2);
    }
    if (std::sqrt(moved) > margin_) {
      all.resize(s.nodes.size());
      for (std::size_t i = 0; i < s.nodes.size(); ++i) all[i] = static_cast<Index>(i);
      nodes = &all;
    }
    const Scalar mu = tool.friction;
    const std::map<Index, ToolNodeHistory>& committed = history_[k];
    for (Index i : *nodes) {
      const auto idx = static_cast<std::size_t>(i);
      const Index node = s.nodes[idx];
      Vector3 x = mesh.node(node);
      for (int comp = 0; comp < dim; ++comp) x(comp) += u(node * dim + comp);
      x = planar(x, dim);
      Vector3 nrm;
      Matrix3 dn;
      Scalar d = 0.0;
      const Scalar g = tool.gap(x, c, dim, nrm, dn, d);
      if (!(g < 0.0)) continue;
      const Scalar ka = s.kappa[idx] * s.area[idx];  // kappa A [N/m]
      const Scalar pn = -ka * g;                      // normal force magnitude [N]
      Vector3 force = pn * nrm;                       // on the node
      Matrix3 k_node = ka * (nrm * nrm.transpose()) - pn * dn;  // d(-f_N)/dx
      Vector3 ft = Vector3::Zero();
      bool slipping = false;
      const Vector3 r = planar(x - c, dim);
      if (mu > 0.0) {
        const Scalar kt = tool.tangential_ratio * ka;  // kappa_T A [N/m]
        // The committed history, or none: a node new to contact slips from
        // its position relative to the tool at the start of the increment.
        Vector3 f0 = Vector3::Zero();
        Vector3 q;
        const auto it = committed.find(node);
        if (it != committed.end()) {
          f0 = it->second.force;
          q = it->second.relative;
        } else {
          Vector3 x0 = mesh.node(node);
          for (int comp = 0; comp < dim; ++comp) x0(comp) += start_(node * dim + comp);
          q = planar(x0 - planar(tool.trajectory.position(t0_), dim), dim);
        }
        // Transport the committed force onto the current tangent plane,
        // keeping its magnitude.
        const Scalar s0 = f0.norm();
        const Vector3 f_tilde = f0 - f0.dot(nrm) * nrm;
        const Scalar f_tilde_norm = f_tilde.norm();
        Vector3 f_hat = Vector3::Zero();
        Matrix3 d_hat = Matrix3::Zero();
        if (s0 > 0.0 && f_tilde_norm > kVanishing * s0) {
          const Vector3 e = f_tilde / f_tilde_norm;
          f_hat = s0 * e;
          const Matrix3 d_tilde =
              -(nrm * (f0.transpose() * dn) + f0.dot(nrm) * dn);
          d_hat = (s0 / f_tilde_norm) * (identity - e * e.transpose()) * d_tilde;
        }
        // Slip increment: the tangential motion relative to the tool.
        const Vector3 w = r - q;
        const Matrix3 p = identity - nrm * nrm.transpose();
        const Vector3 delta = p * w;
        const Matrix3 d_delta = p - nrm.dot(w) * dn - nrm * (w.transpose() * dn);
        const Vector3 trial = f_hat - kt * delta;
        const Matrix3 d_trial = d_hat - kt * d_delta;
        const Scalar trial_norm = trial.norm();
        const Scalar limit = mu * pn;
        Matrix3 d_ft;
        // The symmetric tangent's friction part (see the header): the
        // stiffness of the friction force in the tangent plane alone.
        Matrix3 k_sym;
        if (trial_norm <= limit) {
          ft = trial;
          d_ft = d_trial;
          k_sym = kt * p;
        } else {
          slipping = true;
          const Vector3 e = trial / trial_norm;
          ft = limit * e;
          // d(mu p_N)/dx = -mu kappa A n^T.
          d_ft = -(mu * ka) * (e * nrm.transpose()) +
                 (limit / trial_norm) * (identity - e * e.transpose()) * d_trial;
          k_sym = (limit / trial_norm) * kt * (p - e * e.transpose());
        }
        force += ft;
        if (friction_tangent_ == FrictionTangent::Symmetric) {
          k_node += k_sym;
        } else {
          k_node -= d_ft;
        }
        ToolNodeHistory h;
        h.force = ft;
        h.relative = r;
        out.trial[k].emplace(node, h);
      }
      // Round-off: the gap is a difference of positions, each exact to its
      // relative rounding, which the penalty turns into forces of
      // kappa A eps (|x| + |c|) - far above eps |f| when the penetration is
      // small against the coordinates.
      const Scalar position_scale = ka * (x.norm() + c.norm());
      for (int comp = 0; comp < dim; ++comp) {
        out.residual(node * dim + comp) -= force(comp);
        gross(node * dim + comp) += std::abs(force(comp));
        positions(node * dim + comp) += position_scale;
      }
      ++res.active_nodes;
      if (slipping) ++res.slipping_nodes;
      res.force -= force;
      res.max_penetration = std::max(res.max_penetration, -g);
      if (want_tangent) {
        if (friction_tangent_ == FrictionTangent::Symmetric) {
          k_node = 0.5 * (k_node + k_node.transpose()).eval();  // its round-off
        }
        const Matrix3 kd = k_node.topLeftCorner(3, 3);
        const Scalar top = kd.cwiseAbs().maxCoeff();
        if (dim == 3) {
          if ((kd - kd.transpose()).cwiseAbs().maxCoeff() > 1.0e-12 * top) out.symmetric = false;
        } else if (std::abs(kd(0, 1) - kd(1, 0)) > 1.0e-12 * top) {
          out.symmetric = false;
        }
        out.tangent.push_back({node, k_node});
      }
    }
  }
  out.gross = gross.norm();
  out.round_off = positions.norm();
  out.force_norm = out.residual.norm();
  return out;
}

int ToolContact::anticipate(const Vector& u, const Vector& du, Scalar t, Vector& residual,
                            std::vector<ToolContactEvaluation::NodeBlock>& blocks) const {
  const Mesh& mesh = model_.mesh();
  const int dim = mesh.dim();
  int added = 0;
  for (std::size_t k = 0; k < tools_.size(); ++k) {
    if (!active_[k]) continue;
    const RigidTool& tool = tools_[k];
    const Slaves& s = slaves_[k];
    const Vector3 c = planar(tool.trajectory.position(t), dim);
    for (Index i : s.candidates) {
      const auto idx = static_cast<std::size_t>(i);
      const Index node = s.nodes[idx];
      Vector3 x = mesh.node(node);
      Vector3 dx = Vector3::Zero();
      for (int comp = 0; comp < dim; ++comp) {
        x(comp) += u(node * dim + comp);
        dx(comp) = du(node * dim + comp);
      }
      Vector3 nrm = Vector3::Zero();
      Matrix3 dn;
      Scalar d = 0.0;
      Scalar g = 0.0;
      try {
        g = tool.gap(x, c, dim, nrm, dn, d);
      } catch (const SolverError&) {
        // Deep in the tool: not a node outside it to anticipate (and a state
        // that evaluate() refuses).
        continue;
      }
      if (!(g > 0.0)) continue;  // already active: in the tangent
      Scalar g_next = 0.0;
      try {
        Vector3 n_next;
        g_next = tool.gap(x + dx, c, dim, n_next, dn, d);
      } catch (const SolverError&) {
        g_next = -std::numeric_limits<Scalar>::infinity();  // through the tool
      }
      if (!(g_next < 0.0)) continue;
      const Scalar ka = s.kappa[idx] * s.area[idx];
      for (int comp = 0; comp < dim; ++comp) residual(node * dim + comp) += ka * g * nrm(comp);
      blocks.push_back({node, ka * (nrm * nrm.transpose())});
      ++added;
    }
  }
  return added;
}

void ToolContact::commit(ToolContactEvaluation& ev) {
  for (std::size_t k = 0; k < tools_.size(); ++k) {
    if (!active_[k]) continue;
    history_[k] = std::move(ev.trial[k]);
  }
  ev.trial.clear();
}

void ToolContact::set_history(ToolHistory history) {
  if (history.size() != tools_.size()) {
    throw ConfigError("the saved friction history has " + std::to_string(history.size()) +
                      " tool(s); the analysis has " + std::to_string(tools_.size()));
  }
  for (std::size_t k = 0; k < tools_.size(); ++k) {
    for (const auto& [node, h] : history[k]) {
      (void)h;
      if (!std::binary_search(slaves_[k].nodes.begin(), slaves_[k].nodes.end(), node)) {
        throw ConfigError("the saved friction history of tool '" + tools_[k].name +
                          "' names node " + std::to_string(node) +
                          ", which is not on its surface");
      }
    }
  }
  history_ = std::move(history);
}

void ToolContact::clear_inactive(const std::vector<char>& active) {
  for (std::size_t k = 0; k < tools_.size() && k < active.size(); ++k) {
    if (!active[k]) history_[k].clear();
  }
}

}  // namespace sparlab
