#include "sparlab/fem/ExplicitDynamics.hpp"

#include "ExplicitKernel.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"
#include "sparlab/elements/Hex8.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

namespace sparlab {
namespace {

/// A dedicated kernel that differs from the element dispatch by more than
/// this (relative) is not used.
constexpr Scalar kSelfCheckTolerance = 1.0e-10;
/// A stable-step re-estimate never lets the step grow by more than this.
constexpr Scalar kStepGrowth = 1.01;
/// Power iteration: the Rayleigh quotient times this bounds omega^2.
constexpr Scalar kPowerMargin = 1.05;
/// A run reports its progress at most this often [s].
constexpr double kProgressSeconds = 60.0;
/// The vector updates and energies run over fixed chunks of DOFs, summed in
/// chunk order: the same result for any thread count.
constexpr Index kChunk = 4096;
/// The kinetic-to-internal energy ratio counts over the records whose
/// internal work exceeds this fraction of its value at the end.
constexpr Scalar kRatioFloor = 0.01;
/// The energy balance is not judged over this many time steps after the
/// first contact: the new contact forces' half-step kick enters the kinetic
/// energy a step before their work enters the balance - the scheme's
/// second-order lag, of the order of all the energy there is at the first
/// touch (a relative error of 1 there, 1e-2 ten steps on).
constexpr long kOnsetSteps = 10;

/// A deterministic pseudo-random number in [-1, 1] (splitmix64 of `i`).
Scalar noise(std::uint64_t i) {
  std::uint64_t z = i + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return 2.0 * (static_cast<Scalar>(z >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
}

Vector3 planar(Vector3 v, int dim) {
  if (dim == 2) v.z() = 0.0;
  return v;
}

/// The step the central-difference method takes at the largest angular
/// frequency omega [rad/s]: omega^2 dt^2 + 2 alpha dt <= 4 - s_c (the lagged
/// mass-proportional damping alpha and the contact penalty s_c), times the
/// safety factor.
Scalar limit_step(Scalar omega, Scalar alpha, Scalar contact) {
  const Scalar room = 4.0 - contact;
  if (!(omega > 0.0)) return std::numeric_limits<Scalar>::infinity();
  const Scalar w2 = omega * omega;
  return (-alpha + std::sqrt(alpha * alpha + w2 * room)) / w2;
}

// ---------------------------------------------------------------------------
// Penalty contact with the rigid tools (the explicit law of the file comment)
// ---------------------------------------------------------------------------
struct ContactNodeForce {
  Index node = 0;
  Vector3 normal = Vector3::Zero();    ///< on the node [N]
  Vector3 friction = Vector3::Zero();  ///< on the node [N]
};

struct ContactToolState {
  Vector3 force = Vector3::Zero();  ///< on the tool [N]
  int active_nodes = 0;
  int slipping_nodes = 0;
  Scalar max_penetration = 0.0;
  Scalar max_penetration_ratio = 0.0;  ///< over the node's element thickness
};

class PenaltyContact {
 public:
  PenaltyContact(const FemModel& model, const std::vector<RigidTool>& tools)
      : model_(model), tools_(tools) {
    if (tools.empty()) return;
    // The slave nodes of every tool, exactly as the implicit contact takes
    // them (it validates the tools and their surfaces too).
    const ToolContact probe(model, tools);
    slaves_.resize(tools.size());
    for (std::size_t k = 0; k < tools.size(); ++k) {
      slaves_[k].nodes = probe.slave_nodes(k);
      slaves_[k].area = probe.areas(k);
      slaves_[k].friction.assign(slaves_[k].nodes.size(), Vector3::Zero());
      slaves_[k].touching.assign(slaves_[k].nodes.size(), 0);
      slaves_[k].mass.assign(slaves_[k].nodes.size(), 0.0);
      slaves_[k].thickness.assign(slaves_[k].nodes.size(), 1.0);
    }
  }

  /// The element thickness at every node (per node of the mesh), which the
  /// penetrations are measured against.
  void set_thickness(const std::vector<Scalar>& per_node) {
    for (Slaves& s : slaves_) {
      for (std::size_t i = 0; i < s.nodes.size(); ++i) {
        s.thickness[i] = per_node[static_cast<std::size_t>(s.nodes[i])];
      }
    }
  }

  const std::vector<Index>& slave_nodes(std::size_t k) const { return slaves_[k].nodes; }

  void set_masses(const Vector& m) {
    const int dim = model_.mesh().dim();
    for (Slaves& s : slaves_) {
      for (std::size_t i = 0; i < s.nodes.size(); ++i) s.mass[i] = m(s.nodes[i] * dim);
    }
  }

  void import_friction(const ToolHistory& history) {
    for (Slaves& s : slaves_) {
      std::fill(s.friction.begin(), s.friction.end(), Vector3::Zero());
      std::fill(s.touching.begin(), s.touching.end(), 0);
    }
    if (history.empty()) return;
    if (history.size() != tools_.size()) {
      throw ConfigError("the saved friction history has " + std::to_string(history.size()) +
                        " tool(s); the analysis has " + std::to_string(tools_.size()));
    }
    for (std::size_t k = 0; k < tools_.size(); ++k) {
      Slaves& s = slaves_[k];
      for (const auto& [node, h] : history[k]) {
        const auto it = std::lower_bound(s.nodes.begin(), s.nodes.end(), node);
        if (it == s.nodes.end() || *it != node) {
          throw ConfigError("the saved friction history of tool '" + tools_[k].name +
                            "' names node " + std::to_string(node) +
                            ", which is not on its surface");
        }
        const auto i = static_cast<std::size_t>(it - s.nodes.begin());
        s.friction[i] = h.force;
        s.touching[i] = 1;
      }
    }
  }

  /// The friction history of the nodes in contact, the implicit analysis's
  /// layout (only for frictional tools, as it keeps them).
  ToolHistory export_friction(const Vector& u, const std::vector<Vector3>& centres) const {
    const Mesh& mesh = model_.mesh();
    const int dim = mesh.dim();
    ToolHistory out(tools_.size());
    for (std::size_t k = 0; k < tools_.size(); ++k) {
      if (!(tools_[k].friction > 0.0)) continue;
      const Slaves& s = slaves_[k];
      for (std::size_t i = 0; i < s.nodes.size(); ++i) {
        if (!s.touching[i]) continue;
        Vector3 x = mesh.node(s.nodes[i]);
        for (int c = 0; c < dim; ++c) x(c) += u(s.nodes[i] * dim + c);
        ToolNodeHistory h;
        h.force = s.friction[i];
        h.relative = planar(x - centres[k], dim);
        out[k].emplace(s.nodes[i], h);
      }
    }
    return out;
  }

  /// The contact forces at u, which moved by du over the step while the
  /// tools moved from c0 to c1; the friction state advances (every
  /// explicit step is final). `dt` sets the penalty k = s_c m / dt^2.
  /// \throws SolverError when a node lies within half a sphere's radius of
  ///         its centre (or a cylinder's axis): the tool is through the body.
  void evaluate(const Vector& u, const Vector& du, const std::vector<char>& active,
                const std::vector<Vector3>& c0, const std::vector<Vector3>& c1, Scalar dt,
                Scalar stiffness, std::vector<ContactNodeForce>& forces,
                std::vector<ContactToolState>& state) {
    const Mesh& mesh = model_.mesh();
    const int dim = mesh.dim();
    forces.clear();
    state.assign(tools_.size(), ContactToolState());
    const Scalar per_mass = stiffness / (dt * dt);
    for (std::size_t k = 0; k < tools_.size(); ++k) {
      Slaves& s = slaves_[k];
      if (!active[k]) {
        std::fill(s.friction.begin(), s.friction.end(), Vector3::Zero());
        std::fill(s.touching.begin(), s.touching.end(), 0);
        continue;
      }
      const RigidTool& tool = tools_[k];
      const Vector3 c = planar(c1[k], dim);
      const Vector3 tool_motion = planar(c1[k] - c0[k], dim);
      const Scalar r2 = tool.radius * tool.radius;
      Vector3 axis = Vector3::UnitZ();
      if (tool.shape == RigidTool::Shape::Cylinder && dim == 3) axis = tool.axis.normalized();
      const Vector3 plane_normal = planar(tool.normal, dim).normalized();
      const Scalar mu = tool.friction;
      ContactToolState& res = state[k];
      for (std::size_t i = 0; i < s.nodes.size(); ++i) {
        const Index node = s.nodes[i];
        Vector3 x = mesh.node(node);
        for (int comp = 0; comp < dim; ++comp) x(comp) += u(node * dim + comp);
        x = planar(x, dim);
        // Cheap rejection before the exact gap.
        bool may = false;
        switch (tool.shape) {
          case RigidTool::Shape::Sphere:
            may = (x - c).squaredNorm() < r2;
            break;
          case RigidTool::Shape::Cylinder: {
            Vector3 r = x - c;
            if (dim == 3) r -= axis.dot(r) * axis;
            may = r.squaredNorm() < r2;
            break;
          }
          case RigidTool::Shape::Plane:
            may = plane_normal.dot(x - c) < 0.0;
            break;
        }
        if (!may) {
          s.touching[i] = 0;
          s.friction[i].setZero();
          continue;
        }
        Vector3 n;
        Matrix3 dn;
        Scalar d = 0.0;
        const Scalar g = tool.gap(x, c, dim, n, dn, d);
        if (!(g < 0.0)) {
          s.touching[i] = 0;
          s.friction[i].setZero();
          continue;
        }
        const Scalar k_n = per_mass * s.mass[i];
        const Scalar pn = -k_n * g;
        const Vector3 fn = pn * n;
        Vector3 ft = Vector3::Zero();
        bool slipping = false;
        if (mu > 0.0) {
          // The committed force transported onto the current tangent plane
          // (magnitude kept), incremented by the tangential relative motion.
          const Vector3 f0 = s.touching[i] ? s.friction[i] : Vector3::Zero();
          const Scalar s0 = f0.norm();
          Vector3 f_hat = Vector3::Zero();
          const Vector3 f_tilde = f0 - f0.dot(n) * n;
          const Scalar tilde = f_tilde.norm();
          if (s0 > 0.0 && tilde > 1.0e-12 * s0) f_hat = (s0 / tilde) * f_tilde;
          Vector3 motion = Vector3::Zero();
          for (int comp = 0; comp < dim; ++comp) motion(comp) = du(node * dim + comp);
          const Vector3 relative = motion - tool_motion;
          const Vector3 slip = relative - relative.dot(n) * n;
          const Scalar k_t = tool.tangential_ratio * k_n;
          const Vector3 trial = f_hat - k_t * slip;
          const Scalar trial_norm = trial.norm();
          const Scalar limit = mu * pn;
          if (trial_norm <= limit) {
            ft = trial;
          } else {
            ft = (limit / trial_norm) * trial;
            slipping = true;
          }
        }
        s.touching[i] = 1;
        s.friction[i] = ft;
        forces.push_back({node, fn, ft});
        res.force -= fn + ft;
        ++res.active_nodes;
        if (slipping) ++res.slipping_nodes;
        res.max_penetration = std::max(res.max_penetration, -g);
        res.max_penetration_ratio = std::max(res.max_penetration_ratio, -g / s.thickness[i]);
      }
    }
  }

  struct Saved {
    std::vector<std::vector<Vector3>> friction;
    std::vector<std::vector<char>> touching;
  };
  Saved save() const {
    Saved out;
    for (const Slaves& s : slaves_) {
      out.friction.push_back(s.friction);
      out.touching.push_back(s.touching);
    }
    return out;
  }
  void restore(const Saved& saved) {
    for (std::size_t k = 0; k < slaves_.size() && k < saved.friction.size(); ++k) {
      slaves_[k].friction = saved.friction[k];
      slaves_[k].touching = saved.touching[k];
    }
  }

 private:
  struct Slaves {
    std::vector<Index> nodes;
    std::vector<Scalar> area;  ///< tributary areas [m^2]
    std::vector<Scalar> mass;
    std::vector<Scalar> thickness;  ///< element thickness at the node [m]
    std::vector<Vector3> friction;  ///< committed tangential force on the node
    std::vector<char> touching;     ///< in contact at the last step
  };
  const FemModel& model_;
  const std::vector<RigidTool>& tools_;
  std::vector<Slaves> slaves_;
};

/// Lumped element masses: per element, the row sums of its consistent mass,
/// one value per element DOF.
std::vector<Vector> lumped_element_masses(const FemModel& model, const Assembler& assembler) {
  const Mesh& mesh = model.mesh();
  const int npe = mesh.nodes_per_elem();
  const int dim = model.dofs_per_node();
  std::vector<Vector> out(static_cast<std::size_t>(mesh.num_elements()));
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    const Matrix& me = assembler.element_mass(e);
    Vector m(npe * dim);
    for (int i = 0; i < npe * dim; ++i) m(i) = me.row(i).sum();
    out[static_cast<std::size_t>(e)] = std::move(m);
  }
  return out;
}

/// The characteristic length of an element at its current coordinates x
/// (dim x npe): volume over the largest face area [m].
Scalar characteristic_length(ElementType type, const Matrix& x) {
  switch (type) {
    case ElementType::Hex8: {
      const Scalar volume = hex8_volume(x);
      static constexpr int faces[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
                                          {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
      Scalar largest = 0.0;
      for (const auto& f : faces) {
        Matrix face(3, 4);
        for (int a = 0; a < 4; ++a) face.col(a) = x.col(f[a]);
        largest = std::max(largest, hex8_face_area(face));
      }
      return volume / largest;
    }
    default: break;
  }
  throw ConfigError("the element-length time step is written for the Hex8; use the element "
                    "eigenvalue estimate");
}

/// Per node, the thickness of its thinnest element in the reference
/// configuration: the characteristic length (volume over largest face) of
/// a linear element, the dim-th root of the measure of any other [m].
std::vector<Scalar> node_thickness(const FemModel& model) {
  const Mesh& mesh = model.mesh();
  const ElementType type = mesh.element_type();
  const bool linear = type == ElementType::Hex8;
  const int npe = mesh.nodes_per_elem();
  std::vector<Scalar> out(static_cast<std::size_t>(mesh.num_nodes()),
                          std::numeric_limits<Scalar>::infinity());
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    const Scalar length =
        linear ? characteristic_length(type, mesh.element_coordinates(e))
               : std::pow(mesh.element_measure(e), 1.0 / static_cast<Scalar>(mesh.dim()));
    const Index* nodes = mesh.element_nodes(e);
    for (int a = 0; a < npe; ++a) {
      Scalar& t = out[static_cast<std::size_t>(nodes[a])];
      t = std::min(t, length);
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
std::string to_string(StableStepOptions::Method method) {
  switch (method) {
    case StableStepOptions::Method::ElementEigenvalue: return "element_eigenvalue";
    case StableStepOptions::Method::ElementLength: return "element_length";
    case StableStepOptions::Method::PowerIteration: return "power_iteration";
  }
  return "element_eigenvalue";
}

StableStepOptions::Method parse_stable_step_method(const std::string& text) {
  if (text == "element_eigenvalue") return StableStepOptions::Method::ElementEigenvalue;
  if (text == "element_length") return StableStepOptions::Method::ElementLength;
  if (text == "power_iteration") return StableStepOptions::Method::PowerIteration;
  throw ConfigError("unknown stable step method '" + text +
                    "'; expected \"element_eigenvalue\", \"element_length\" or "
                    "\"power_iteration\"");
}

std::string to_string(MassScalingOptions::Mode mode) {
  switch (mode) {
    case MassScalingOptions::Mode::None: return "none";
    case MassScalingOptions::Mode::Uniform: return "uniform";
    case MassScalingOptions::Mode::Selective: return "selective";
  }
  return "none";
}

MassScalingOptions::Mode parse_mass_scaling_mode(const std::string& text) {
  if (text == "none") return MassScalingOptions::Mode::None;
  if (text == "uniform") return MassScalingOptions::Mode::Uniform;
  if (text == "selective") return MassScalingOptions::Mode::Selective;
  throw ConfigError("unknown mass scaling mode '" + text +
                    "'; expected \"none\", \"uniform\" or \"selective\"");
}

void ExplicitOptions::validate(const std::string& what) const {
  const auto finite = [](Scalar v) { return std::isfinite(v); };
  if (!(stable_step.safety > 0.0 && stable_step.safety <= 1.0)) {
    throw ConfigError(what + ": 'stable_step.safety' must lie in (0, 1]");
  }
  if (stable_step.update_every < 0 || stable_step.power_iterations < 1) {
    throw ConfigError(what + ": 'stable_step.update_every' must be >= 0 and the power "
                             "iterations >= 1");
  }
  if (mass_scaling.mode != MassScalingOptions::Mode::None &&
      !(mass_scaling.target_time_step > 0.0 && finite(mass_scaling.target_time_step))) {
    throw ConfigError(what + ": mass scaling '" + to_string(mass_scaling.mode) +
                      "' needs a positive 'target_time_step' [s]");
  }
  if (mass_scaling.mode == MassScalingOptions::Mode::Selective &&
      stable_step.method == StableStepOptions::Method::PowerIteration) {
    throw ConfigError(what + ": selective mass scaling needs a time step per element "
                             "(stable_step method \"element_eigenvalue\" or \"element_length\"); "
                             "power iteration estimates the whole model's only");
  }
  if (mass_scaling.dynamic && mass_scaling.mode != MassScalingOptions::Mode::Selective) {
    throw ConfigError(what + ": 'mass_scaling.dynamic' raises the scales of selective mass "
                             "scaling during the run; it needs mode \"selective\"");
  }
  if (!(mass_scaling.max_added_mass_fraction >= 0.0) ||
      !finite(mass_scaling.max_added_mass_fraction)) {
    throw ConfigError(what + ": 'mass_scaling.max_added_mass_fraction' must be >= 0");
  }
  if (!(time_step >= 0.0) || !finite(time_step)) {
    throw ConfigError(what + ": a fixed time step must be positive (0: the stable step)");
  }
  if (!(mass_damping >= 0.0) || !finite(mass_damping)) {
    throw ConfigError(what + ": the mass-proportional damping must be >= 0 [1/s]");
  }
  if (!(contact_stiffness > 0.0 && contact_stiffness <= 1.0)) {
    throw ConfigError(what + ": 'contact_stiffness' s_c must lie in (0, 1]: the penalty "
                             "k = s_c m / dt^2 adds s_c to (omega dt)^2, and the time step "
                             "shrinks to keep that below 4 (0.5 is the default)");
  }
  if (!(penetration_warning > 0.0) || !finite(penetration_warning)) {
    throw ConfigError(what + ": 'penetration_warning' (penetration over element thickness) "
                             "must be positive");
  }
  if (history_every < 1 || snapshot_every < 0) {
    throw ConfigError(what + ": 'history_every' must be >= 1 and 'snapshot_every' >= 0");
  }
  if (!(energy_tolerance > 0.0) || !(energy_limit >= energy_tolerance) ||
      !(kinetic_ratio_warning > 0.0) || !finite(energy_limit)) {
    throw ConfigError(what + ": 'energy_tolerance' and 'kinetic_ratio_warning' must be "
                             "positive and 'energy_limit' at least the tolerance");
  }
}

// ---------------------------------------------------------------------------
// Time map
// ---------------------------------------------------------------------------
Scalar ExplicitTimeMap::pseudo_time(Scalar tau) const {
  if (physical.empty()) return 0.0;
  if (tau <= physical.front()) {
    // The last knot at the start (a skipped standstill at the start).
    std::size_t i = 0;
    while (i + 1 < physical.size() && physical[i + 1] <= physical.front()) ++i;
    return tau < physical.front() ? pseudo.front() : pseudo[i];
  }
  if (tau >= physical.back()) return pseudo.back();
  const auto it = std::upper_bound(physical.begin(), physical.end(), tau);
  const auto i = static_cast<std::size_t>(it - physical.begin()) - 1;  // last <= tau
  const Scalar w = (tau - physical[i]) / (physical[i + 1] - physical[i]);
  return pseudo[i] + w * (pseudo[i + 1] - pseudo[i]);
}

ExplicitTimeMap ExplicitTimeMap::linear(Scalar t0, Scalar t1, Scalar duration) {
  if (!(duration > 0.0) || !std::isfinite(duration) || !(t1 >= t0)) {
    throw ConfigError("an explicit step needs a positive duration [s]");
  }
  ExplicitTimeMap map;
  map.physical = {0.0, duration};
  map.pseudo = {t0, t1};
  return map;
}

ExplicitTimeMap ExplicitTimeMap::tool_speed(const std::vector<const ToolTrajectory*>& paths,
                                            Scalar t0, Scalar t1, Scalar speed) {
  if (!(speed > 0.0) || !std::isfinite(speed)) {
    throw ConfigError("an explicit step's tool speed must be positive [m/s]");
  }
  std::vector<Scalar> knots = {t0, t1};
  for (const ToolTrajectory* p : paths) {
    for (Scalar t : p->knots_in(t0, t1)) knots.push_back(t);
  }
  std::sort(knots.begin(), knots.end());
  knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
  ExplicitTimeMap map;
  map.physical.push_back(0.0);
  map.pseudo.push_back(t0);
  Scalar tau = 0.0;
  for (std::size_t i = 1; i < knots.size(); ++i) {
    Scalar travel = 0.0;
    for (const ToolTrajectory* p : paths) {
      travel = std::max(travel, (p->position(knots[i]) - p->position(knots[i - 1])).norm());
    }
    tau += travel / speed;
    map.physical.push_back(tau);
    map.pseudo.push_back(knots[i]);
  }
  if (!(tau > 0.0)) {
    std::ostringstream os;
    os << "no tool moves between t = " << t0 << " and " << t1
       << " s: a tool speed cannot set the step's duration; give 'duration' [s] instead";
    throw ConfigError(os.str());
  }
  return map;
}

// ---------------------------------------------------------------------------
// ExplicitDynamics
// ---------------------------------------------------------------------------
ExplicitDynamics::ExplicitDynamics(const FemModel& model, const Assembler& assembler,
                                   const NonlinearOptions& nonlinear,
                                   std::vector<RigidTool> tools, ExplicitOptions options)
    : model_(model), assembler_(assembler), nonlinear_(nonlinear), tools_(std::move(tools)),
      options_(options) {
  if (!model.finalized()) throw ModelError("the model must be finalised before a solve");
  options_.validate();
  for (const IsotropicMaterial& m : model.materials()) {
    if (!(m.density() > 0.0)) {
      throw ConfigError("explicit dynamics needs a positive density of every material; "
                        "material '" + m.name() + "' has none - set its 'density' [kg/m^3]");
    }
  }
  if (options_.stable_step.method == StableStepOptions::Method::ElementLength) {
    (void)characteristic_length(model.mesh().element_type(),
                                model.mesh().element_coordinates(0));
  }
  force_ = std::make_unique<ExplicitInternalForce>(model, assembler, nonlinear_,
                                                   options_.dedicated_kernel);
}

ExplicitDynamics::~ExplicitDynamics() = default;

bool ExplicitDynamics::dedicated() const { return force_->dedicated(); }

Vector ExplicitDynamics::internal_force(const Vector& u,
                                        const std::vector<std::vector<PlasticState>>& history,
                                        bool generic) const {
  if (u.size() != model_.dofs().num_dofs()) {
    throw ConfigError("explicit dynamics: the displacement has the wrong size");
  }
  if (generic || !force_->dedicated()) return force_->generic_internal(u, history);
  const std::vector<std::vector<PlasticState>> kept = force_->history();
  force_->set_history(history);
  Vector internal(u.size());
  Vector external(u.size());
  Scalar energy = 0.0;
  // The path of an unrecorded step (the kernel's own elastic predictor).
  force_->evaluate(u, 0.0, false, internal, external, energy, false);
  force_->set_history(kept);
  return internal;
}

Vector ExplicitDynamics::element_time_steps(const Vector& u, const Vector* scale) const {
  const Mesh& mesh = model_.mesh();
  const int dim = mesh.dim();
  const int npe = mesh.nodes_per_elem();
  const Index ne = mesh.num_elements();
  const Element& element = model_.element();
  const Scalar thickness = dim == 2 ? model_.thickness() : 1.0;
  Vector out(ne);
  const bool eigenvalue =
      options_.stable_step.method != StableStepOptions::Method::ElementLength;
  std::vector<Vector> lumped;
  if (eigenvalue) lumped = lumped_element_masses(model_, assembler_);
  std::vector<Matrix> constitutive(static_cast<std::size_t>(ne));
  for (Index e = 0; e < ne; ++e) constitutive[static_cast<std::size_t>(e)] = model_.constitutive_of(e);
  std::string failure;
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (Index e = 0; e < ne; ++e) {
    try {
      Matrix x = mesh.element_coordinates(e);
      const Index* nodes = mesh.element_nodes(e);
      for (int a = 0; a < npe; ++a) {
        for (int k = 0; k < dim; ++k) x(k, a) += u(nodes[a] * dim + k);
      }
      const Scalar s = scale ? (*scale)(e) : 1.0;
      if (eigenvalue) {
        const Matrix k = element.stiffness(x, constitutive[static_cast<std::size_t>(e)],
                                           thickness, model_.integration());
        const Vector inv = lumped[static_cast<std::size_t>(e)].cwiseSqrt().cwiseInverse();
        const Matrix a = inv.asDiagonal() * k * inv.asDiagonal();
        const Eigen::SelfAdjointEigenSolver<Matrix> eig(a, Eigen::EigenvaluesOnly);
        const Scalar top = eig.eigenvalues().maxCoeff();
        out(e) = top > 0.0 ? 2.0 * std::sqrt(s / top) : std::numeric_limits<Scalar>::infinity();
      } else {
        const IsotropicMaterial& m = model_.material_of(e);
        const Scalar modulus =
            model_.stress_state() == StressState::PlaneStress
                ? m.youngs_modulus() / (1.0 - m.poisson_ratio() * m.poisson_ratio())
                : m.lame_lambda() + 2.0 * m.shear_modulus();
        const Scalar c = std::sqrt(modulus / (s * m.density()));
        out(e) = characteristic_length(mesh.element_type(), x) / c;
      }
    } catch (const std::exception& ex) {
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp critical(sparlab_explicit_step)
#endif
      if (failure.empty()) failure = "element " + std::to_string(e) + ": " + ex.what();
    }
  }
  if (!failure.empty()) {
    throw SolverError("the stable time step cannot be estimated: " + failure);
  }
  return out;
}

Scalar ExplicitDynamics::power_iteration_time_step(const Vector& u, const Vector* scale) const {
  const Mesh& mesh = model_.mesh();
  const int dim = mesh.dim();
  const int npe = mesh.nodes_per_elem();
  const Element& element = model_.element();
  const Scalar thickness = dim == 2 ? model_.thickness() : 1.0;
  const bool deformed = u.cwiseAbs().maxCoeff() > 0.0;
  const SparseMatrix k =
      deformed ? assembler_.assemble_elementwise([&](Index e) -> Matrix {
        Matrix x = mesh.element_coordinates(e);
        const Index* nodes = mesh.element_nodes(e);
        for (int a = 0; a < npe; ++a) {
          for (int c = 0; c < dim; ++c) x(c, a) += u(nodes[a] * dim + c);
        }
        return element.stiffness(x, model_.constitutive_of(e), thickness, model_.integration());
      })
               : assembler_.assemble_stiffness();
  const Vector m = assembler_.assemble_mass(MassType::Lumped, scale).diagonal();
  const Vector root = m.cwiseSqrt().cwiseInverse();
  Vector x(m.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) x(i) = noise(static_cast<std::uint64_t>(i));
  x.normalize();
  Scalar rq = 0.0;
  for (int it = 0; it < options_.stable_step.power_iterations; ++it) {
    Vector y = root.cwiseProduct(k * root.cwiseProduct(x));
    rq = x.dot(y);
    const Scalar norm = y.norm();
    if (!(norm > 0.0)) break;
    x = y / norm;
  }
  const Scalar omega2 = kPowerMargin * rq;
  return omega2 > 0.0 ? 2.0 / std::sqrt(omega2) : std::numeric_limits<Scalar>::infinity();
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------
ExplicitResult ExplicitDynamics::run(const ExplicitDrive& drive, const ExplicitState& start) {
  Timer wall;
  ExplicitResult res;
  TimingLedger& timing = res.timing;
  const Mesh& mesh = model_.mesh();
  const int dim = mesh.dim();
  const Index n = model_.dofs().num_dofs();
  const Index ne = mesh.num_elements();
  const std::size_t nt = tools_.size();
  const StableStepOptions& sso = options_.stable_step;
  const MassScalingOptions& mso = options_.mass_scaling;

  // --- the drive ---------------------------------------------------------
  const Scalar duration = drive.duration;
  if (!(duration > 0.0) || !std::isfinite(duration)) {
    throw ConfigError("an explicit run needs a positive, finite duration [s]");
  }
  if (start.displacement.size() != n) {
    throw ConfigError("the explicit start state has " +
                      std::to_string(start.displacement.size()) +
                      " displacement DOF(s); the model has " + std::to_string(n));
  }
  if (start.velocity.size() != 0 && start.velocity.size() != n) {
    throw ConfigError("the explicit start state's velocity has the wrong size");
  }
  const auto nfix = static_cast<Eigen::Index>(drive.fixed.size());
  if (drive.fixed_start.size() != nfix || drive.fixed_end.size() != nfix) {
    throw ConfigError("an explicit drive needs a start and an end value per prescribed DOF");
  }
  for (std::size_t i = 0; i < drive.fixed.size(); ++i) {
    if (drive.fixed[i] < 0 || drive.fixed[i] >= n ||
        (i > 0 && !(drive.fixed[i] > drive.fixed[i - 1]))) {
      throw ConfigError("an explicit drive's prescribed DOFs must be ascending and in range");
    }
  }
  if (drive.active.size() != nt) {
    throw ConfigError("an explicit drive needs one activity flag per tool");
  }
  const auto ramp = [&](Scalar tau) -> Scalar { return std::clamp(tau / duration, 0.0, 1.0); };
  bool any_tool = false;
  for (char a : drive.active) any_tool = any_tool || a;
  const auto pseudo = [&](Scalar tau) { return drive.time_map.pseudo_time(tau); };

  // --- the state ---------------------------------------------------------
  std::vector<char> fixed(static_cast<std::size_t>(n), 0);
  for (Index d : drive.fixed) fixed[static_cast<std::size_t>(d)] = 1;
  const Vector span = drive.fixed_end - drive.fixed_start;
  const auto prescribed = [&](Scalar tau, Vector& out) {
    const Scalar r = ramp(tau);
    out.noalias() = drive.fixed_start + r * span;
  };
  const bool moving_fixed = span.size() > 0 && span.cwiseAbs().maxCoeff() > 0.0;
  Vector u = start.displacement;
  Vector v = start.velocity.size() == n ? start.velocity : Vector::Zero(n);
  force_->set_history(start.history);
  PenaltyContact contact(model_, tools_);
  contact.import_friction(start.friction);
  if (nt > 0) contact.set_thickness(node_thickness(model_));
  res.kernel = force_->dedicated() ? "dedicated Hex8" : "generic element dispatch";

  // The dedicated kernel must reproduce the element dispatch.
  if (force_->dedicated()) {
    const Scalar difference = force_->self_check(u);
    if (!(difference <= kSelfCheckTolerance)) {
      std::ostringstream os;
      os << "the dedicated Hex8 kernel differs from the element dispatch by " << difference
         << " (relative) on this model; the generic element dispatch is used instead";
      force_->use_generic(os.str());
      res.warnings.push_back(os.str());
      log::warn("explicit: ", os.str());
      res.kernel = "generic element dispatch";
    } else {
      log::debug("explicit: dedicated Hex8 kernel agrees with the element dispatch to ",
                 difference);
    }
  } else {
    log::info("explicit: generic element dispatch (", force_->fallback_reason(), ")");
  }

  // --- mass and time step ------------------------------------------------
  Vector scale = Vector::Ones(ne);
  Scalar dt = 0.0;
  Scalar omega = 0.0;
  // The penalty's share of the stability limit: s_c, or s_c times the
  // tangential penalty ratio of a frictional tool where that is stiffer
  // (the normal and the tangential springs act in orthogonal directions).
  Scalar s_c = 0.0;
  if (any_tool) {
    Scalar reserve = 1.0;
    std::string stiffest;
    for (std::size_t k = 0; k < nt; ++k) {
      if (!drive.active[k] || !(tools_[k].friction > 0.0)) continue;
      if (tools_[k].tangential_ratio > reserve) {
        reserve = tools_[k].tangential_ratio;
        stiffest = tools_[k].name;
      }
    }
    s_c = options_.contact_stiffness * reserve;
    if (!(s_c < 4.0)) {
      std::ostringstream os;
      os << "the contact penalty's share of the stability limit, contact_stiffness "
         << options_.contact_stiffness << " times tool '" << stiffest
         << "''s tangential_penalty " << reserve << " = " << s_c
         << ", must stay below 4 (keep it below 1 or so: the time step shrinks as "
            "sqrt(4 - s)); lower either";
      throw ConfigError(os.str());
    }
  }
  const Scalar alpha = options_.mass_damping;
  const bool power = sso.method == StableStepOptions::Method::PowerIteration;
  {
    ScopedTimer st(timing, "stable_step");
    Vector steps;
    if (power) {
      res.stable_time_step = power_iteration_time_step(u, nullptr);
    } else {
      steps = element_time_steps(u, nullptr);
      res.stable_time_step = steps.minCoeff();
    }
    // The usable step of an element (or of the model) at mass scale s:
    // safety times the limit with damping and contact, which grows as
    // sqrt(s) when the damping is negligible.
    const auto usable = [&](Scalar crit) {
      return sso.safety * limit_step(2.0 / crit, alpha, s_c);
    };
    if (mso.mode == MassScalingOptions::Mode::Uniform) {
      const Scalar u0 = usable(res.stable_time_step);
      const Scalar f = std::max(1.0, std::pow(mso.target_time_step / u0, 2));
      scale.setConstant(f);
    } else if (mso.mode == MassScalingOptions::Mode::Selective) {
      for (Index e = 0; e < ne; ++e) {
        scale(e) = std::max(1.0, std::pow(mso.target_time_step / usable(steps(e)), 2));
      }
    }
    if (power) {
      res.scaled_stable_time_step =
          mso.mode == MassScalingOptions::Mode::None
              ? res.stable_time_step
              : power_iteration_time_step(u, &scale);
    } else {
      res.scaled_stable_time_step = (steps.array() * scale.array().sqrt()).minCoeff();
    }
    omega = 2.0 / res.scaled_stable_time_step;
    dt = sso.safety * limit_step(omega, alpha, s_c);
    if (mso.mode != MassScalingOptions::Mode::None) dt = std::min(dt, mso.target_time_step);
  }
  const Scalar limit = limit_step(omega, alpha, s_c);
  if (options_.time_step > 0.0) {
    if (options_.time_step > limit) {
      std::ostringstream os;
      os << "the fixed time step " << options_.time_step << " s exceeds the stability limit "
         << limit << " s: the run will not be stable";
      res.warnings.push_back(os.str());
      log::warn("explicit: ", os.str());
    }
    dt = options_.time_step;
  }
  Vector m = assembler_.assemble_mass(MassType::Lumped, &scale).diagonal();
  res.physical_mass = assembler_.total_mass();
  res.scaled_mass = assembler_.total_mass(&scale);
  res.added_mass_fraction = (res.scaled_mass - res.physical_mass) / res.physical_mass;
  res.max_mass_scale = scale.maxCoeff();
  res.scaled_elements = static_cast<int>((scale.array() > 1.0).count());
  if (mso.mode != MassScalingOptions::Mode::None &&
      res.added_mass_fraction > mso.max_added_mass_fraction) {
    std::ostringstream os;
    os << to_string(mso.mode) << " mass scaling to " << mso.target_time_step
       << " s adds " << 100.0 * res.added_mass_fraction << " % of the physical mass ("
       << res.scaled_elements << " of " << ne << " elements scaled, by up to "
       << res.max_mass_scale << "), above the " << 100.0 * mso.max_added_mass_fraction
       << " % threshold: the inertia forces are " << res.max_mass_scale
       << " times the physical ones - check the kinetic energy ratio";
    res.warnings.push_back(os.str());
    log::warn("explicit: ", os.str());
  }
  Vector m_inverse = m.cwiseInverse();
  contact.set_masses(m);
  const Scalar initial_added_mass = res.added_mass_fraction;
  res.time_step = dt;
  res.min_time_step = dt;
  const bool update =
      nonlinear_.kinematics != Kinematics::SmallStrain && sso.update_every > 0 &&
      options_.time_step == 0.0;
  // Dynamic selective scaling: at the updates, mass is added where an
  // element's step (thinned, distorted) has fallen below the target.
  const bool rescale = update && mso.mode == MassScalingOptions::Mode::Selective && mso.dynamic;
  log::info("explicit: ", mesh.num_elements(), " elements, stable step ", res.stable_time_step,
            " s (", to_string(sso.method), "), ", to_string(mso.mode), " mass scaling (added ",
            100.0 * res.added_mass_fraction, " %), time step ", dt, " s, duration ", duration,
            " s: about ", std::ceil(duration / dt), " steps; ", res.kernel);

  // --- the initial state -------------------------------------------------
  Vector values(nfix);
  prescribed(0.0, values);
  for (Eigen::Index i = 0; i < nfix; ++i) u(drive.fixed[static_cast<std::size_t>(i)]) = values(i);
  Vector f_int(n), f_ext(n), f_int_old(n), f_ext_old(n);
  Vector fc = Vector::Zero(n);   // contact forces on the nodes, dense
  Vector a = Vector::Zero(n);
  Vector du = Vector::Zero(n);
  Vector reactions = Vector::Zero(n);
  Vector reactions_old = Vector::Zero(n);
  Vector previous_values = values;
  Scalar stored = 0.0;
  Scalar stored_start = 0.0;
  Scalar elastic = 0.0;  // elastic strain energy at the last recorded step
  // The contact buffers hold every slave node: they never grow in the loop.
  std::vector<ContactNodeForce> contact_forces;
  std::vector<ContactNodeForce> contact_old;
  {
    std::size_t slaves = 0;
    for (std::size_t k = 0; k < nt; ++k) slaves += contact.slave_nodes(k).size();
    contact_forces.reserve(slaves);
    contact_old.reserve(slaves);
  }
  std::vector<ContactToolState> tool_state(nt);
  std::vector<Vector3> c_old(nt, Vector3::Zero());
  std::vector<Vector3> c_new(nt, Vector3::Zero());
  const auto centres = [&](Scalar tau, std::vector<Vector3>& out) {
    const Scalar t = pseudo(tau);
    for (std::size_t k = 0; k < nt; ++k) out[k] = tools_[k].trajectory.position(t);
  };
  // Velocity and acceleration of the prescribed DOFs, from their motion at
  // the neighbouring steps (central differences, consistent with the
  // integrator): v_p(t) = (v_{-} + v_{+}) / 2, a_p = (v_{+} - v_{-}) / dt.
  // (Buffers kept for the run: the step loop allocates nothing.)
  Vector ahead(nfix), behind(nfix), rate_here(nfix), rate_next(nfix), rate_last(nfix);
  const auto prescribed_rates = [&](Scalar tau, Scalar h_behind, Scalar h_ahead) {
    prescribed(tau, rate_here);
    prescribed(tau + h_ahead, rate_next);
    prescribed(std::max(tau - h_behind, 0.0), rate_last);
    const Scalar back = tau > 0.0 ? std::min(h_behind, tau) : 1.0;
    const Scalar mid = 0.5 * (h_ahead + (tau > 0.0 ? h_behind : h_ahead));
    for (Eigen::Index i = 0; i < nfix; ++i) {
      const Scalar v_plus = (rate_next(i) - rate_here(i)) / h_ahead;
      const Scalar v_minus = tau > 0.0 ? (rate_here(i) - rate_last(i)) / back : 0.0;
      ahead(i) = 0.5 * (v_plus + v_minus);
      behind(i) = (v_plus - v_minus) / mid;
    }
  };
  const auto contact_step = [&](Scalar tau_new, Scalar h_nominal) {
    ScopedTimer st(timing, "contact");
    for (const ContactNodeForce& c : contact_forces) {
      for (int k = 0; k < dim; ++k) fc(c.node * dim + k) = 0.0;
    }
    contact_old.swap(contact_forces);
    centres(tau_new, c_new);
    if (any_tool) {
      contact.evaluate(u, du, drive.active, c_old, c_new, h_nominal, options_.contact_stiffness,
                       contact_forces, tool_state);
    } else {
      contact_forces.clear();
    }
    for (const ContactNodeForce& c : contact_forces) {
      for (int k = 0; k < dim; ++k) fc(c.node * dim + k) += c.normal(k) + c.friction(k);
    }
  };

  std::string failure;
  try {
    ScopedTimer st(timing, "internal_force");
    force_->evaluate(u, ramp(0.0), true, f_int, f_ext, stored);
  } catch (const SolverError& ex) {
    throw ConfigError(std::string("the explicit start state cannot be evaluated: ") + ex.what());
  }
  stored_start = stored;
  elastic = force_->elastic_energy();
  // A step that continues a stressed part (a retract) measures its kinetic
  // energy against the elastic energy of the state: its internal energy
  // holds the hardening energy of the whole forming history.
  const bool stressed_start = stored_start > 0.0;
  const bool elastic_basis = stressed_start && std::isfinite(elastic);
  res.kinetic_ratio_basis = elastic_basis ? "elastic energy" : "internal energy";
  centres(0.0, c_old);
  try {
    contact_step(0.0, dt);
  } catch (const SolverError& ex) {
    throw ConfigError(std::string("the explicit start state cannot be evaluated: ") + ex.what());
  }
  prescribed_rates(0.0, dt, dt);
  for (Index d = 0; d < n; ++d) {
    if (!fixed[static_cast<std::size_t>(d)]) {
      a(d) = (f_ext(d) - f_int(d) + fc(d) - alpha * m(d) * v(d)) * m_inverse(d);
    }
  }
  for (Eigen::Index i = 0; i < nfix; ++i) {
    const Index d = drive.fixed[static_cast<std::size_t>(i)];
    v(d) = ahead(i);
    a(d) = behind(i);
    reactions(d) = m(d) * a(d) + f_int(d) - f_ext(d) - fc(d) + alpha * m(d) * v(d);
  }
  const auto kinetic = [&]() {
    Scalar t = 0.0;
    for (Index d = 0; d < n; ++d) t += m(d) * v(d) * v(d);
    return 0.5 * t;
  };
  const Scalar kinetic_start = kinetic();

  // --- energies, records, checkpoints -------------------------------------
  Scalar w_int = 0.0, w_ext = 0.0, w_cn = 0.0, w_ct = 0.0, w_damp = 0.0;
  Scalar w_mass = 0.0;  // kinetic energy of the mass added during the run
  Scalar kin = kinetic_start;
  Scalar energy_scale = std::max(kin, 0.0);
  long step = 0;
  Scalar tau = 0.0;
  Scalar tau_base = 0.0;   // time at which the current step size began
  long since_base = 0;     // steps of the current size
  std::vector<Vector3> mean_force(nt, Vector3::Zero());
  Scalar mean_time = 0.0;
  long first_contact = -1;
  const auto record = [&]() {
    ExplicitRecord r;
    r.step = step;
    r.time = tau;
    r.pseudo_time = pseudo(tau);
    r.time_step = dt;
    r.kinetic = kin;
    r.internal = w_int;
    r.internal_energy = stored_start + w_int;
    r.stored = stored;
    r.elastic = elastic;
    r.plastic = w_int - (stored - stored_start);
    r.contact_normal = w_cn;
    r.contact_friction = w_ct;
    r.damping = w_damp;
    r.external = w_ext;
    r.mass_scaling = w_mass;
    r.error = kin - kinetic_start + w_int + w_damp - w_ext - w_cn - w_ct - w_mass;
    r.max_plastic_strain = force_->max_plastic_strain();
    for (std::size_t k = 0; k < nt; ++k) {
      if (!drive.active[k]) continue;
      ExplicitToolRecord tr;
      tr.tool = k;
      tr.centre = tools_[k].trajectory.position(r.pseudo_time);
      tr.force = mean_time > 0.0 ? Vector3(mean_force[k] / mean_time) : tool_state[k].force;
      tr.active_nodes = tool_state[k].active_nodes;
      tr.slipping_nodes = tool_state[k].slipping_nodes;
      tr.max_penetration = tool_state[k].max_penetration;
      tr.max_penetration_ratio = tool_state[k].max_penetration_ratio;
      r.tools.push_back(tr);
      mean_force[k].setZero();
    }
    mean_time = 0.0;
    res.records.push_back(std::move(r));
  };
  struct Checkpoint {
    Vector u, v, a, f_int, f_ext, fc, reactions;
    Vector scale, m, m_inverse;  // dynamic mass scaling changes them
    Scalar tau = 0, tau_base = 0, dt = 0, stored = 0;
    Scalar elastic = 0;
    Scalar max_penetration_ratio = 0;
    Scalar min_time_step = 0;
    int mass_updates = 0, step_updates = 0;
    long step = 0, since_base = 0;
    Scalar w_int = 0, w_ext = 0, w_cn = 0, w_ct = 0, w_damp = 0, w_mass = 0, kin = 0;
    std::vector<ContactNodeForce> contact_forces;
    std::vector<ContactToolState> tool_state;
    PenaltyContact::Saved friction;
    std::size_t records = 0, snapshots = 0;
  } saved;
  saved.contact_forces.reserve(contact_forces.capacity());  // copies into it never allocate
  const auto checkpoint = [&]() {
    saved.u = u;
    saved.v = v;
    saved.a = a;
    saved.f_int = f_int;
    saved.f_ext = f_ext;
    saved.fc = fc;
    saved.reactions = reactions;
    saved.scale = scale;
    saved.m = m;
    saved.m_inverse = m_inverse;
    saved.tau = tau;
    saved.tau_base = tau_base;
    saved.dt = dt;
    saved.stored = stored;
    saved.elastic = elastic;
    saved.min_time_step = res.min_time_step;
    saved.max_penetration_ratio = res.max_penetration_ratio;
    saved.mass_updates = res.mass_updates;
    saved.step_updates = res.step_updates;
    saved.step = step;
    saved.since_base = since_base;
    saved.w_int = w_int;
    saved.w_ext = w_ext;
    saved.w_cn = w_cn;
    saved.w_ct = w_ct;
    saved.w_damp = w_damp;
    saved.w_mass = w_mass;
    saved.kin = kin;
    saved.contact_forces = contact_forces;
    saved.tool_state = tool_state;
    saved.friction = contact.save();
    saved.records = res.records.size();
    saved.snapshots = res.snapshots.size();
    force_->checkpoint();
  };
  const auto restore = [&]() {
    u = saved.u;
    v = saved.v;
    a = saved.a;
    f_int = saved.f_int;
    f_ext = saved.f_ext;
    fc = saved.fc;
    reactions = saved.reactions;
    if (res.mass_updates != saved.mass_updates) {
      scale = saved.scale;
      m = saved.m;
      m_inverse = saved.m_inverse;
      contact.set_masses(m);
    }
    tau = saved.tau;
    tau_base = saved.tau_base;
    dt = saved.dt;
    stored = saved.stored;
    elastic = saved.elastic;
    res.min_time_step = saved.min_time_step;
    res.max_penetration_ratio = saved.max_penetration_ratio;
    res.mass_updates = saved.mass_updates;
    res.step_updates = saved.step_updates;
    step = saved.step;
    since_base = saved.since_base;
    w_int = saved.w_int;
    w_ext = saved.w_ext;
    w_cn = saved.w_cn;
    w_ct = saved.w_ct;
    w_damp = saved.w_damp;
    w_mass = saved.w_mass;
    kin = saved.kin;
    contact_forces = saved.contact_forces;
    tool_state = saved.tool_state;
    contact.restore(saved.friction);
    res.records.resize(saved.records);
    res.snapshots.resize(saved.snapshots);
    force_->restore();
  };
  for (const ContactToolState& s : tool_state) {
    if (s.active_nodes > 0 && first_contact < 0) first_contact = 0;
  }
  record();
  checkpoint();

  // --- the steps ----------------------------------------------------------
  const Index chunks = (n + kChunk - 1) / kChunk;
  std::vector<Scalar> partial(static_cast<std::size_t>(chunks) * 4, 0.0);
  const Scalar end_tolerance = 1.0e-9;
  Timer progress;
  bool done = false;
  const auto stop = [&](const std::string& why) {
    std::ostringstream os;
    os << "step " << step + 1 << " (t = " << tau << " s of " << duration << " s): " << why
       << "; the state of step " << saved.step << " (t = " << saved.tau << " s) is kept";
    failure = os.str();
  };
  while (!done && failure.empty()) {
    // The step size: the current dt, the last step landing on the duration.
    Scalar h = dt;
    Scalar tau_new = tau_base + static_cast<Scalar>(since_base + 1) * dt;
    if (duration - tau_new <= end_tolerance * dt) {
      tau_new = duration;
      h = duration - tau;
      done = true;
    }
    const Scalar half = 0.5 * h;
    {
      ScopedTimer st(timing, "integration");
      // v_{n+1/2}, u_{n+1} on the free DOFs; the prescribed ones directly.
      prescribed(tau_new, values);
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp parallel for schedule(static) if (chunks > 1)
#endif
      for (Index c = 0; c < chunks; ++c) {
        const Index end = std::min(n, (c + 1) * kChunk);
        for (Index d = c * kChunk; d < end; ++d) {
          if (fixed[static_cast<std::size_t>(d)]) continue;
          const Scalar vh = v(d) + half * a(d);
          v(d) = vh;  // the half-step velocity until the acceleration is known
          du(d) = h * vh;
          u(d) += du(d);
        }
      }
      for (Eigen::Index i = 0; i < nfix; ++i) {
        const Index d = drive.fixed[static_cast<std::size_t>(i)];
        du(d) = values(i) - u(d);
        v(d) = du(d) / h;
        u(d) = values(i);
      }
    }
    f_int_old.swap(f_int);
    f_ext_old.swap(f_ext);
    try {
      ScopedTimer st(timing, "internal_force");
      const bool recorded = (step + 1) % options_.history_every == 0 || done;
      force_->evaluate(u, ramp(tau_new), true, f_int, f_ext, stored, recorded);
      if (recorded) elastic = force_->elastic_energy();
    } catch (const SolverError& ex) {
      stop(ex.what());
      break;
    }
    c_old.swap(c_new);
    try {
      contact_step(tau_new, dt);
    } catch (const SolverError& ex) {
      stop(ex.what());
      break;
    }
    {
      ScopedTimer st(timing, "integration");
      // Contact work over the step, from the forces at both ends.
      for (const ContactNodeForce& c : contact_old) {
        for (int k = 0; k < dim; ++k) {
          const Scalar dd = du(c.node * dim + k);
          w_cn += 0.5 * dd * c.normal(k);
          w_ct += 0.5 * dd * c.friction(k);
        }
      }
      for (const ContactNodeForce& c : contact_forces) {
        for (int k = 0; k < dim; ++k) {
          const Scalar dd = du(c.node * dim + k);
          w_cn += 0.5 * dd * c.normal(k);
          w_ct += 0.5 * dd * c.friction(k);
        }
      }
      // a_{n+1}, v_{n+1}, and the energies, chunk by chunk.
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp parallel for schedule(static) if (chunks > 1)
#endif
      for (Index c = 0; c < chunks; ++c) {
        const Index end = std::min(n, (c + 1) * kChunk);
        Scalar p_int = 0.0, p_ext = 0.0, p_damp = 0.0, p_kin = 0.0;
        for (Index d = c * kChunk; d < end; ++d) {
          const Scalar dd = du(d);
          p_int += dd * (f_int_old(d) + f_int(d));
          p_ext += dd * (f_ext_old(d) + f_ext(d));
          const Scalar damping = alpha * m(d) * v(d);  // v holds v_{n+1/2}
          p_damp += dd * damping;
          if (fixed[static_cast<std::size_t>(d)]) continue;
          a(d) = (f_ext(d) - f_int(d) + fc(d) - damping) * m_inverse(d);
          v(d) += half * a(d);
          p_kin += m(d) * v(d) * v(d);
        }
        partial[static_cast<std::size_t>(4 * c)] = p_int;
        partial[static_cast<std::size_t>(4 * c + 1)] = p_ext;
        partial[static_cast<std::size_t>(4 * c + 2)] = p_damp;
        partial[static_cast<std::size_t>(4 * c + 3)] = p_kin;
      }
      Scalar s_int = 0.0, s_ext = 0.0, s_damp = 0.0, kin_sum = 0.0;
      for (Index c = 0; c < chunks; ++c) {
        s_int += partial[static_cast<std::size_t>(4 * c)];
        s_ext += partial[static_cast<std::size_t>(4 * c + 1)];
        s_damp += partial[static_cast<std::size_t>(4 * c + 2)];
        kin_sum += partial[static_cast<std::size_t>(4 * c + 3)];
      }
      w_int += 0.5 * s_int;
      w_ext += 0.5 * s_ext;
      w_damp += s_damp;
      // The prescribed DOFs: velocity, acceleration and reactions.
      if (nfix > 0) {
        prescribed_rates(tau_new, h, dt);
        reactions_old.swap(reactions);
        Scalar s_react = 0.0;
        for (Eigen::Index i = 0; i < nfix; ++i) {
          const Index d = drive.fixed[static_cast<std::size_t>(i)];
          const Scalar half_velocity = v(d);
          v(d) = ahead(i);
          a(d) = behind(i);
          reactions(d) = m(d) * a(d) + f_int(d) - f_ext(d) - fc(d) + alpha * m(d) * half_velocity;
          if (moving_fixed) s_react += du(d) * (reactions_old(d) + reactions(d));
          kin_sum += m(d) * v(d) * v(d);
        }
        w_ext += 0.5 * s_react;
      }
      kin = 0.5 * kin_sum;
    }
    ++step;
    ++since_base;
    tau = tau_new;
    for (std::size_t k = 0; k < nt; ++k) {
      mean_force[k] += h * tool_state[k].force;
      res.max_penetration_ratio =
          std::max(res.max_penetration_ratio, tool_state[k].max_penetration_ratio);
    }
    mean_time += h;
    if (first_contact < 0) {
      for (const ContactToolState& s : tool_state) {
        if (s.active_nodes > 0) first_contact = step;
      }
    }
    if (!std::isfinite(kin) || !std::isfinite(w_int)) {
      stop("the state is no longer finite (the step is unstable)");
      break;
    }
    // Records, with the checks.
    if (step % options_.history_every == 0 || done) {
      if (!force_->dedicated() && nonlinear_.kinematics != Kinematics::SmallStrain) {
        const Index inverted = force_->first_inverted(u);
        if (inverted >= 0) {
          stop("element " + std::to_string(inverted) + " is inverted");
          break;
        }
      }
      if (!u.allFinite() || !v.allFinite()) {
        stop("the state is no longer finite (the step is unstable)");
        break;
      }
      const Scalar error =
          kin - kinetic_start + w_int + w_damp - w_ext - w_cn - w_ct - w_mass;
      energy_scale = std::max({energy_scale, kin, std::abs(w_int),
                               std::abs(w_ext) + std::abs(w_cn) + std::abs(w_ct)});
      const bool onset = first_contact >= 0 && step - first_contact < kOnsetSteps;
      const Scalar relative =
          energy_scale > 0.0 && !onset ? std::abs(error) / energy_scale : 0.0;
      if (relative > options_.energy_limit) {
        std::ostringstream os;
        os << "the energy balance is off by " << relative << " of the largest energy (limit "
           << options_.energy_limit << "): the integration is unstable or the contact "
           << "penalty too stiff";
        stop(os.str());
        break;
      }
      res.max_energy_error = std::max(res.max_energy_error, relative);
      record();
      checkpoint();
    }
    if (options_.snapshot_every > 0 && step % options_.snapshot_every == 0 && !done) {
      res.snapshots.push_back({step, tau, pseudo(tau), u});
    }
    // The stable step on the current configuration.
    if (update && !done && step % sso.update_every == 0) {
      ScopedTimer st(timing, "stable_step");
      Scalar crit = 0.0;
      try {
        if (power) {
          crit = power_iteration_time_step(u, &scale);
        } else {
          Vector steps = element_time_steps(u, &scale);
          if (rescale) {
            // Raise the scale of every element whose usable step fell below
            // the target (dt_e grows as sqrt(s_e)); the velocities are kept,
            // the kinetic energy of the added mass is booked as W_mass.
            bool changed = false;
            for (Index e = 0; e < ne; ++e) {
              const Scalar usable_e = sso.safety * limit_step(2.0 / steps(e), alpha, s_c);
              if (usable_e < mso.target_time_step) {
                const Scalar factor = std::pow(mso.target_time_step / usable_e, 2);
                scale(e) *= factor;
                steps(e) *= std::sqrt(factor);
                changed = true;
              }
            }
            if (changed) {
              const Vector m_new = assembler_.assemble_mass(MassType::Lumped, &scale).diagonal();
              Scalar added = 0.0;
              for (Index d = 0; d < n; ++d) {
                const Scalar dm = m_new(d) - m(d);
                if (dm == 0.0) continue;
                added += dm * v(d) * v(d);
                // The free DOFs' acceleration from the same forces.
                if (!fixed[static_cast<std::size_t>(d)]) a(d) *= m(d) / m_new(d);
              }
              w_mass += 0.5 * added;
              m = m_new;
              m_inverse = m.cwiseInverse();
              contact.set_masses(m);
              kin = kinetic();
              ++res.mass_updates;
            }
          }
          crit = steps.minCoeff();
        }
      } catch (const SolverError& ex) {
        stop(ex.what());
        break;
      }
      const Scalar fresh = sso.safety * limit_step(2.0 / crit, alpha, s_c);
      Scalar next = std::min(fresh, kStepGrowth * dt);
      if (mso.mode != MassScalingOptions::Mode::None) next = std::min(next, mso.target_time_step);
      ++res.step_updates;
      if (next != dt) {
        dt = next;
        tau_base = tau;
        since_base = 0;
        res.min_time_step = std::min(res.min_time_step, dt);
      }
    }
    if (log::level() <= log::Level::Info && progress.elapsed_seconds() >= kProgressSeconds) {
      progress.reset();
      log::info("explicit: step ", step, ", t = ", tau, " s of ", duration, " s (pseudo-time ",
                pseudo(tau), " s), time step ", dt, " s, kinetic ", kin, " J, internal work ",
                w_int, " J, ", wall.elapsed_seconds(), " s elapsed");
    }
  }
  if (!failure.empty()) {
    restore();
    res.completed = false;
    res.termination = failure;
    log::warn("explicit run stopped: ", failure);
  } else {
    res.completed = true;
    res.termination = "reached the end of its duration";
  }
  if (res.mass_updates > 0) {
    // The mass at the end of the run (dynamic selective scaling).
    res.scaled_mass = assembler_.total_mass(&scale);
    res.added_mass_fraction = (res.scaled_mass - res.physical_mass) / res.physical_mass;
    res.max_mass_scale = scale.maxCoeff();
    res.scaled_elements = static_cast<int>((scale.array() > 1.0).count());
    if (res.added_mass_fraction > mso.max_added_mass_fraction &&
        !(initial_added_mass > mso.max_added_mass_fraction)) {
      std::ostringstream os;
      os << "dynamic selective mass scaling raised the added mass to "
         << 100.0 * res.added_mass_fraction << " % of the physical mass, above the "
         << 100.0 * mso.max_added_mass_fraction << " % threshold";
      res.warnings.push_back(os.str());
      log::warn("explicit: ", os.str());
    }
  }
  res.steps = step;
  res.duration = tau;
  res.final_time_step = dt;
  res.contact = first_contact >= 0;

  // The kinetic energy ratio after the first contact: against the internal
  // energy of a virgin start, the elastic energy of a stressed one (a
  // retract). The warned ratio skips the records whose reference energy is
  // below 1 % of its final value (the first touch of a flat sheet); the
  // peak over all of them is reported beside it.
  if (first_contact >= 0 && !res.records.empty()) {
    const auto reference = [&](const ExplicitRecord& r) {
      return elastic_basis ? r.elastic : r.internal_energy;
    };
    const Scalar final_reference = reference(res.records.back());
    for (const ExplicitRecord& r : res.records) {
      const Scalar e = reference(r);
      if (r.step < first_contact || !(e > 0.0)) continue;
      const Scalar ratio = r.kinetic / e;
      res.peak_kinetic_ratio = std::max(res.peak_kinetic_ratio, ratio);
      if (e > kRatioFloor * final_reference) {
        res.max_kinetic_ratio = std::max(res.max_kinetic_ratio, ratio);
      }
    }
  }
  if (res.max_kinetic_ratio > options_.kinetic_ratio_warning) {
    std::ostringstream os;
    os << "the kinetic energy reached " << res.max_kinetic_ratio << " of the "
       << res.kinetic_ratio_basis << " after the first contact (warning above "
       << options_.kinetic_ratio_warning
       << "): the run is not quasi-static - slow the tool or reduce the mass scaling";
    res.warnings.push_back(os.str());
    log::warn("explicit: ", os.str());
  }
  if (res.max_penetration_ratio > options_.penetration_warning) {
    std::ostringstream os;
    os << "the contact penetration reached " << res.max_penetration_ratio
       << " of the element thickness at the node (warning above "
       << options_.penetration_warning << "): the formed shape is biased by about that "
       << "much, at any tool speed - raise 'contact_stiffness' (up to 1)";
    res.warnings.push_back(os.str());
    log::warn("explicit: ", os.str());
  }
  if (res.max_energy_error > options_.energy_tolerance) {
    std::ostringstream os;
    os << "the energy balance error reached " << res.max_energy_error
       << " of the largest energy (warning above " << options_.energy_tolerance << ")";
    res.warnings.push_back(os.str());
    log::warn("explicit: ", os.str());
  }

  // The end state.
  res.final_state.displacement = u;
  res.final_state.velocity = v;
  res.final_state.history = force_->history();
  {
    std::vector<Vector3> c_end(nt, Vector3::Zero());
    centres(tau, c_end);
    res.final_state.friction = contact.export_friction(u, c_end);
  }
  res.acceleration = a;
  res.residual = f_int - f_ext - fc;
  res.reactions = Vector::Zero(n);
  for (Index d : drive.fixed) res.reactions(d) = reactions(d);
  timing.add("total", wall.elapsed_seconds());
  log::info("explicit run ", res.completed ? "completed" : "STOPPED", ": ", step, " steps, ",
            wall.elapsed_seconds(), " s (", wall.elapsed_seconds() / std::max<long>(step, 1) * 1e3,
            " ms a step), kinetic/internal ", res.max_kinetic_ratio, " (peak ",
            res.peak_kinetic_ratio, "), penetration/thickness ", res.max_penetration_ratio,
            ", energy error ",
            res.max_energy_error, ", added mass ", 100.0 * res.added_mass_fraction, " %");
  return res;
}

}  // namespace sparlab
