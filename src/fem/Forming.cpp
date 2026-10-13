// Where this fits: see include/sparlab/fem/Forming.hpp

#include "sparlab/fem/Forming.hpp"

#include "NonlinearSystem.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"
#include "sparlab/fem/TotalLagrangian.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#ifdef SPARLAB_HAVE_CHOLMOD
#include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace sparlab {
namespace {

using namespace detail;

/// The von Mises stress of a plane stress state with the out-of-plane
/// normal stress sz [Pa].
Scalar von_mises_plane(Scalar sx, Scalar sy, Scalar sxy, Scalar sz) {
  const Scalar j = 0.5 * ((sx - sy) * (sx - sy) + (sy - sz) * (sy - sz) + (sz - sx) * (sz - sx)) +
                   3.0 * sxy * sxy;
  return std::sqrt(std::max(j, 0.0));
}

/// The von Mises stress of a Voigt stress (3 entries in 2-D, the plane
/// strain sz from the Poisson ratio; 6 in 3-D) [Pa].
Scalar von_mises(const Vector& s, StressState state, Scalar poisson) {
  if (s.size() == 3) {
    const Scalar sx = s(0);
    const Scalar sy = s(1);
    const Scalar sxy = s(2);
    const Scalar sz = (state == StressState::PlaneStrain) ? poisson * (sx + sy) : 0.0;
    const Scalar j = 0.5 * ((sx - sy) * (sx - sy) + (sy - sz) * (sy - sz) +
                            (sz - sx) * (sz - sx)) +
                     3.0 * sxy * sxy;
    return std::sqrt(std::max(j, 0.0));
  }
  if (s.size() == 6) {
    const Scalar sx = s(0);
    const Scalar sy = s(1);
    const Scalar sz = s(2);
    const Scalar j = 0.5 * ((sx - sy) * (sx - sy) + (sy - sz) * (sy - sz) +
                            (sz - sx) * (sz - sx)) +
                     3.0 * (s(3) * s(3) + s(4) * s(4) + s(5) * s(5));
    return std::sqrt(std::max(j, 0.0));
  }
  std::ostringstream os;
  os << "von_mises expects a Voigt vector of 3 or 6 components, received " << s.size();
  throw ModelError(os.str());
}

#ifdef SPARLAB_HAVE_CHOLMOD
constexpr bool kHaveSuiteSparse = true;
#else
constexpr bool kHaveSuiteSparse = false;
#endif

/// An increment that converges in at most this many iterations lets the
/// next one grow by kGrowth (up to the step's cap).
constexpr int kQuickIterations = 4;
constexpr Scalar kGrowth = 1.5;
/// At the end of a release, reactions of the supports above this fraction
/// of the reference force mean they are not statically determinate: they
/// hold the body in a deformed shape (a warning).
constexpr Scalar kReleaseReactionWarning = 1.0e-3;
/// A plastic strain model meant for small strains: the run warns beyond it.
constexpr Scalar kStrainWarning = 0.05;
/// A step reports its progress at most this often [s] (a forming path of
/// hundreds of millimetres over a fine sheet runs for hours).
constexpr double kProgressSeconds = 60.0;
/// A tool active in two consecutive steps may be this fraction of the
/// smallest slave-node size away from where it was (rounding of the
/// windows); farther, it would jump. A tool that becomes active is warned
/// about when it penetrates deeper than this fraction.
constexpr Scalar kJumpTolerance = 1.0e-6;

// ---------------------------------------------------------------------------
// The free-free tangent of one Dirichlet partition: pattern analysed once,
// values scattered through a cached map, refactorised numerically.
// ---------------------------------------------------------------------------
class PartitionedFactor {
 public:
  explicit PartitionedFactor(bool suitesparse) {
#ifdef SPARLAB_HAVE_CHOLMOD
    suitesparse_ = suitesparse;
    // A tangent that is not positive definite is expected now and then (it
    // falls back to LDL^T): no CHOLMOD message on stderr.
    llt_.cholmod().print = 0;
    llt_.cholmod().error_handler = nullptr;
#else
    (void)suitesparse;  // Eigen's factorisations only
#endif
  }

  /// A new partition: the free DOFs (ascending) of an n-DOF model. The same
  /// partition again (a step that keeps the constraints of the one before)
  /// keeps the pattern and its analyses.
  void reset(const std::vector<Index>& free, Index n) {
    if (built_ && free == free_ && static_cast<Index>(map_.size()) == n) return;
    free_ = free;
    map_.assign(static_cast<std::size_t>(n), -1);
    for (std::size_t i = 0; i < free.size(); ++i) {
      map_[static_cast<std::size_t>(free[i])] = static_cast<Index>(i);
    }
    built_ = false;
    forget_analyses();
  }

  /// Scatter the free-free block of `full` into the cached pattern (built
  /// on first use, and again if `full`'s pattern changed).
  void assemble(const SparseMatrix& full) {
    if (!built_ || !same_pattern(full)) build(full);
    const Scalar* values = full.valuePtr();
    Scalar* out = kff_.valuePtr();
    const auto nnz = static_cast<std::size_t>(full.nonZeros());
    for (std::size_t k = 0; k < nnz; ++k) {
      const Eigen::Index p = positions_[k];
      if (p >= 0) out[p] = values[k];
    }
  }

  /// Factorise the assembled block. Symmetric: supernodal Cholesky
  /// (CHOLMOD) when available, else or on failure LDL^T, and LU when LDL^T
  /// meets a vanishing pivot; non-symmetric: LU (UMFPACK, else SparseLU).
  /// False when the matrix is singular.
  /// Each attempt's time goes to `timing` under "factor_<kind>".
  bool factorize(bool symmetric, TimingLedger& timing) {
    if (symmetric) {
#ifdef SPARLAB_HAVE_CHOLMOD
      if (suitesparse_) {
        ScopedTimer st(timing, "factor_cholmod_llt");
        if (!llt_ready_) {
          llt_.analyzePattern(kff_);
          llt_ready_ = true;
        }
        llt_.factorize(kff_);
        if (llt_.info() == Eigen::Success) return use(Kind::Cholmod, "CHOLMOD supernodal LLT");
        ++failed_["CHOLMOD supernodal LLT (not positive definite)"];
      }
#endif
      ScopedTimer st(timing, "factor_ldlt");
      if (!ldlt_ready_) {
        ldlt_.analyzePattern(kff_);
        ldlt_ready_ = true;
      }
      ldlt_.factorize(kff_);
      if (ldlt_.info() == Eigen::Success) {
        const Vector d = ldlt_.vectorD();
        const Scalar top = d.size() > 0 ? d.cwiseAbs().maxCoeff() : 0.0;
        if (top > 0.0 && d.cwiseAbs().minCoeff() > 1.0e-13 * top && d.allFinite()) {
          return use(Kind::Ldlt, "SimplicialLDLT");
        }
      }
      ++failed_["SimplicialLDLT (vanishing pivot)"];
    }
#ifdef SPARLAB_HAVE_CHOLMOD
    if (suitesparse_) {
      ScopedTimer st(timing, "factor_umfpack_lu");
      if (!umf_ready_) {
        umf_.analyzePattern(kff_);
        umf_ready_ = true;
      }
      umf_.factorize(kff_);
      if (umf_.info() != Eigen::Success) {
        ++failed_["UMFPACK LU (singular)"];
        return false;
      }
      return use(Kind::Umfpack, "UMFPACK LU");
    }
#endif
    ScopedTimer st(timing, "factor_sparse_lu");
    if (!lu_ready_) {
      lu_.analyzePattern(kff_);
      lu_ready_ = true;
    }
    lu_.factorize(kff_);
    if (lu_.info() != Eigen::Success) {
      ++failed_["SparseLU (singular)"];
      return false;
    }
    return use(Kind::Lu, "SparseLU");
  }

  Vector solve(const Vector& b) {
    switch (kind_) {
#ifdef SPARLAB_HAVE_CHOLMOD
      case Kind::Cholmod: return llt_.solve(b);
      case Kind::Umfpack: return umf_.solve(b);
#endif
      case Kind::Ldlt: return ldlt_.solve(b);
      case Kind::Lu: return lu_.solve(b);
      default: break;
    }
    throw SolverError("the forming tangent was solved before it was factorised");
  }

  /// The factorisations used so far and how often, e.g.
  /// "CHOLMOD supernodal LLT x 812, UMFPACK LU x 40".
  std::string names() const { return listing(counts_); }
  /// The attempts that failed, each followed by the next factorisation in
  /// line (an LU that fails fails the iteration), and how often, e.g.
  /// "CHOLMOD supernodal LLT (not positive definite) x 2"; empty if none.
  std::string failures() const { return listing(failed_); }

 private:
  enum class Kind { None, Cholmod, Ldlt, Lu, Umfpack };

  static std::string listing(const std::map<std::string, int>& counts) {
    std::string out;
    for (const auto& [name, count] : counts) {
      out += (out.empty() ? "" : ", ") + name + " x " + std::to_string(count);
    }
    return out;
  }

  bool use(Kind kind, const char* name) {
    kind_ = kind;
    ++counts_[name];
    return true;
  }

  bool same_pattern(const SparseMatrix& full) const {
    if (full.nonZeros() != static_cast<Eigen::Index>(inner_.size()) ||
        full.outerSize() + 1 != static_cast<Eigen::Index>(outer_.size())) {
      return false;
    }
    return std::memcmp(full.outerIndexPtr(), outer_.data(),
                       outer_.size() * sizeof(StorageIndex)) == 0 &&
           std::memcmp(full.innerIndexPtr(), inner_.data(),
                       inner_.size() * sizeof(StorageIndex)) == 0;
  }

  // The pattern of the free-free block: the entries of `full` whose row and
  // column are free, in `full`'s (column-major, sorted) order - the free
  // DOFs are ascending, so the block's columns keep their rows sorted.
  void build(const SparseMatrix& full) {
    const auto nf = static_cast<Eigen::Index>(free_.size());
    const auto nnz = static_cast<std::size_t>(full.nonZeros());
    outer_.assign(full.outerIndexPtr(), full.outerIndexPtr() + full.outerSize() + 1);
    inner_.assign(full.innerIndexPtr(), full.innerIndexPtr() + nnz);
    positions_.assign(nnz, -1);
    std::vector<StorageIndex> outer(static_cast<std::size_t>(nf) + 1, 0);
    std::vector<StorageIndex> inner;
    inner.reserve(nnz);
    for (Eigen::Index col = 0; col < full.outerSize(); ++col) {
      const Index c = map_[static_cast<std::size_t>(col)];
      if (c < 0) continue;
      for (StorageIndex k = full.outerIndexPtr()[col]; k < full.outerIndexPtr()[col + 1]; ++k) {
        const Index r = map_[static_cast<std::size_t>(full.innerIndexPtr()[k])];
        if (r < 0) continue;
        positions_[static_cast<std::size_t>(k)] = static_cast<Eigen::Index>(inner.size());
        inner.push_back(static_cast<StorageIndex>(r));
      }
      outer[static_cast<std::size_t>(c) + 1] = static_cast<StorageIndex>(inner.size());
    }
    kff_ = SparseMatrix(nf, nf);
    kff_.resizeNonZeros(static_cast<Eigen::Index>(inner.size()));
    std::copy(outer.begin(), outer.end(), kff_.outerIndexPtr());
    std::copy(inner.begin(), inner.end(), kff_.innerIndexPtr());
    std::fill(kff_.valuePtr(), kff_.valuePtr() + inner.size(), 0.0);
    built_ = true;
    forget_analyses();
  }

  void forget_analyses() {
    ldlt_ready_ = false;
    lu_ready_ = false;
#ifdef SPARLAB_HAVE_CHOLMOD
    llt_ready_ = false;
    umf_ready_ = false;
#endif
    kind_ = Kind::None;
  }

#ifdef SPARLAB_HAVE_CHOLMOD
  bool suitesparse_ = false;  ///< CHOLMOD and UMFPACK, else Eigen's factorisations
#endif
  std::vector<Index> free_;
  std::vector<Index> map_;  ///< DOF -> free position, -1 if prescribed
  bool built_ = false;
  std::vector<StorageIndex> outer_;
  std::vector<StorageIndex> inner_;
  std::vector<Eigen::Index> positions_;  ///< nonzero of `full` -> nonzero of kff_, or -1
  SparseMatrix kff_;
  Kind kind_ = Kind::None;
  std::map<std::string, int> counts_;  ///< factorisations used
  std::map<std::string, int> failed_;  ///< attempts that failed
  Eigen::SimplicialLDLT<SparseMatrix> ldlt_;
  bool ldlt_ready_ = false;
  Eigen::SparseLU<SparseMatrix> lu_;
  bool lu_ready_ = false;
#ifdef SPARLAB_HAVE_CHOLMOD
  Eigen::CholmodSupernodalLLT<SparseMatrix> llt_;
  bool llt_ready_ = false;
  Eigen::UmfPackLU<SparseMatrix> umf_;
  bool umf_ready_ = false;
#endif
};

/// Add a node-diagonal block into a full matrix whose pattern already holds
/// it (every node's own block is in the element pattern).
void add_node_block(SparseMatrix& k, Index node, int dim, const Matrix3& block) {
  for (int b = 0; b < dim; ++b) {
    const Eigen::Index col = node * dim + b;
    StorageIndex* rows = k.innerIndexPtr();
    Scalar* values = k.valuePtr();
    const StorageIndex begin = k.outerIndexPtr()[col];
    const StorageIndex end = k.outerIndexPtr()[col + 1];
    for (int a = 0; a < dim; ++a) {
      const auto row = static_cast<StorageIndex>(node * dim + a);
      const StorageIndex* it = std::lower_bound(rows + begin, rows + end, row);
      if (it != rows + end && *it == row) {
        values[it - rows] += block(a, b);
      } else {
        k.coeffRef(row, col) += block(a, b);  // outside the pattern: rebuilt
      }
    }
  }
}

/// The DOFs a step prescribes, and their values at the step's start and end.
struct Partition {
  std::vector<Index> fixed;  ///< ascending
  std::vector<Index> free;   ///< ascending
  std::vector<StepConstraint::Mode> mode;  ///< per fixed DOF
  Vector target;             ///< absolute end values (absolute mode) [m]
};

Partition build_partition(const Mesh& mesh, const std::vector<StepConstraint>& constraints,
                          const std::string& step) {
  const int dim = mesh.dim();
  std::map<Index, std::pair<StepConstraint::Mode, Scalar>> dofs;
  for (const StepConstraint& c : constraints) {
    const std::string label = "step '" + step + "', constraint '" +
                              (c.name.empty() ? c.constraint.region.name : c.name) + "'";
    if (dim == 2 && c.constraint.fix_z) {
      throw ConfigError(label + " fixes z but the mesh is two-dimensional");
    }
    bool any = false;
    for (int k = 0; k < dim; ++k) any = any || c.constraint.fixes(k);
    if (!any) throw ConfigError(label + " fixes no component");
    for (int k = 0; k < dim; ++k) {
      if (c.mode == StepConstraint::Mode::Hold && c.constraint.fixes(k) &&
          c.constraint.value(k) != 0.0) {
        throw ConfigError(label + " holds its DOFs where the step finds them (mode hold) but "
                                  "gives them a value, which would be ignored; use mode "
                                  "absolute to move them to it");
      }
    }
    const std::vector<Index> nodes = c.constraint.region.select_nodes(mesh);
    if (nodes.empty()) throw ConfigError(label + " selects no node");
    for (Index node : nodes) {
      for (int k = 0; k < dim; ++k) {
        if (!c.constraint.fixes(k)) continue;
        dofs[node * dim + k] = {c.mode, c.constraint.value(k)};  // later wins
      }
    }
  }
  Partition p;
  p.target = Vector::Zero(static_cast<Eigen::Index>(dofs.size()));
  for (const auto& [dof, mv] : dofs) {
    p.target(static_cast<Eigen::Index>(p.fixed.size())) = mv.second;
    p.fixed.push_back(dof);
    p.mode.push_back(mv.first);
  }
  const Index n = mesh.num_nodes() * dim;
  std::size_t next = 0;
  for (Index d = 0; d < n; ++d) {
    if (next < p.fixed.size() && p.fixed[next] == d) {
      ++next;
      continue;
    }
    p.free.push_back(d);
  }
  return p;
}

/// The rigid-body motions of the whole mesh (at its reference positions)
/// that the prescribed DOFs leave free, described in words; empty when
/// they suppress every one.
std::vector<std::string> free_rigid_motions(const Mesh& mesh, const std::vector<Index>& fixed) {
  const int dim = mesh.dim();
  const int modes = dim == 3 ? 6 : 3;
  const BoundingBox box = mesh.bounding_box();
  const Vector3 centre = 0.5 * (box.lower + box.upper);
  const Scalar size = std::max(box.extent().norm(), 1.0e-300);
  Matrix m = Matrix::Zero(static_cast<Eigen::Index>(fixed.size()), modes);
  for (std::size_t i = 0; i < fixed.size(); ++i) {
    const Index node = fixed[i] / dim;
    const int comp = fixed[i] % dim;
    const Vector3 r = (mesh.node(node) - centre) / size;
    const auto row = static_cast<Eigen::Index>(i);
    m(row, comp) = 1.0;  // translations
    if (dim == 2) {
      m(row, 2) = comp == 0 ? -r.y() : r.x();
    } else {
      for (int j = 0; j < 3; ++j) {
        const Vector3 rot = Vector3::Unit(j).cross(r);  // e_j x r
        m(row, 3 + j) = rot(comp);
      }
    }
  }
  const Matrix gram = m.transpose() * m;
  Eigen::SelfAdjointEigenSolver<Matrix> eig(gram);
  const Vector values = eig.eigenvalues();
  const Scalar top = std::max(values.maxCoeff(), 0.0);
  std::vector<std::string> out;
  const char* axes = "xyz";
  for (int i = 0; i < modes; ++i) {
    if (top > 0.0 && values(i) > 1.0e-12 * top) continue;
    const Vector v = eig.eigenvectors().col(i);
    int dominant = 0;
    v.cwiseAbs().maxCoeff(&dominant);
    std::string what;
    if (dim == 2) {
      what = dominant < 2 ? std::string("translation along ") + axes[dominant]
                          : std::string("rotation about z");
    } else {
      what = dominant < 3 ? std::string("translation along ") + axes[dominant]
                          : std::string("rotation about ") + axes[dominant - 3];
    }
    if (v.cwiseAbs().maxCoeff() < 0.95) what += " (combined with others)";
    out.push_back(what);
  }
  return out;
}

/// "(x, y[, z])" [m].
std::string point_text(const Vector3& p, int dim) {
  std::ostringstream os;
  os << "(";
  for (int k = 0; k < dim; ++k) os << (k ? ", " : "") << p(k);
  os << ")";
  return os.str();
}

Scalar max_nodal(const Vector& u, int dim) {
  Scalar top = 0.0;
  for (Eigen::Index node = 0; node < u.size() / dim; ++node) {
    top = std::max(top, u.segment(node * dim, dim).norm());
  }
  return top;
}

/// The predictor of an increment whose prescribed DOFs move by `dp` from
/// the converged state `u`: when `dp` is a rigid motion of the body -
/// always so for statically determinate supports, which cannot deform it -
/// the free DOFs of `trial` follow that motion (linearised about the
/// current configuration) instead of staying where they were, which would
/// distort the elements around each moved support node. False (and
/// `trial` untouched) when `dp` vanishes or deforms the body.
bool predict_rigid_motion(const Mesh& mesh, const Vector& u, const Partition& p, const Vector& dp,
                          Vector& trial) {
  const Scalar size = dp.norm();
  if (!(size > 0.0)) return false;
  const int dim = mesh.dim();
  const int modes = dim == 3 ? 6 : 3;
  const BoundingBox box = mesh.bounding_box();
  const Vector3 centre = 0.5 * (box.lower + box.upper);
  const Scalar extent = std::max(box.extent().norm(), 1.0e-300);
  const auto position = [&](Index node) {
    Vector3 x = mesh.node(node);
    for (int k = 0; k < dim; ++k) x(k) += u(node * dim + k);
    return Vector3((x - centre) / extent);
  };
  // dp_d = (t + w x r_node)_comp, the rotation scaled by the extent.
  const auto row = [&](Index dof, Eigen::Ref<Vector> out) {
    const Index node = dof / dim;
    const auto comp = static_cast<int>(dof % dim);
    const Vector3 r = position(node);
    out.setZero();
    out(comp) = 1.0;
    if (dim == 2) {
      out(2) = comp == 0 ? -r.y() : r.x();
    } else {
      for (int j = 0; j < 3; ++j) out(3 + j) = Vector3::Unit(j).cross(r)(comp);
    }
  };
  Matrix m(static_cast<Eigen::Index>(p.fixed.size()), modes);
  for (std::size_t i = 0; i < p.fixed.size(); ++i) {
    Vector r(modes);
    row(p.fixed[i], r);
    m.row(static_cast<Eigen::Index>(i)) = r.transpose();
  }
  const Eigen::ColPivHouseholderQR<Matrix> qr(m);
  if (qr.rank() < modes) return false;
  const Vector motion = qr.solve(dp);
  if (!motion.allFinite() || (m * motion - dp).norm() > 1.0e-8 * size) return false;
  Vector r(modes);
  for (Index dof : p.free) {
    row(dof, r);
    trial(dof) += r.dot(motion);
  }
  return true;
}

}  // namespace

bool forming_suitesparse_available() { return kHaveSuiteSparse; }

StepConstraint::Mode parse_constraint_mode(const std::string& text) {
  if (text == "hold") return StepConstraint::Mode::Hold;
  if (text == "absolute") return StepConstraint::Mode::Absolute;
  throw ConfigError("unknown constraint mode '" + text + "'; expected \"hold\" or \"absolute\"");
}

std::string to_string(FormingStep::Type type) {
  switch (type) {
    case FormingStep::Type::Release: return "release";
    case FormingStep::Type::FormExplicit: return "form_explicit";
    case FormingStep::Type::Form: break;
  }
  return "form";
}

FormingStep::Type parse_step_type(const std::string& text) {
  if (text == "form") return FormingStep::Type::Form;
  if (text == "release") return FormingStep::Type::Release;
  if (text == "form_explicit") return FormingStep::Type::FormExplicit;
  throw ConfigError("unknown step type '" + text +
                    "'; expected \"form\", \"release\" or \"form_explicit\"");
}

// ---------------------------------------------------------------------------
// Construction: everything that can be refused before a step runs
// ---------------------------------------------------------------------------
FormingAnalysis::FormingAnalysis(const FemModel& model, const Assembler& assembler,
                                 FormingOptions options)
    : model_(model), assembler_(assembler), options_(std::move(options)) {
  if (!model.finalized()) throw ModelError("the model must be finalised before a solve");
  if (options_.steps.empty()) throw ConfigError("the forming analysis has no step");
  if (options_.max_iterations < 1 || options_.max_cuts < 0 || options_.max_increments < 1 ||
      options_.snapshot_stride < 0) {
    throw ConfigError("the forming analysis needs max_iterations >= 1, max_cuts >= 0, "
                      "max_increments >= 1 and a snapshot stride >= 0");
  }
  if (!(options_.residual_tolerance > 0.0) || !(options_.displacement_tolerance > 0.0)) {
    throw ConfigError("the forming tolerances must be positive");
  }
  if (options_.law == HyperelasticModel::NeoHookean) {
    if (options_.kinematics == Kinematics::SmallStrain) {
      throw ConfigError("the small-strain kinematics is linear elasticity; the neo-Hookean "
                        "law needs \"finite\" kinematics");
    }
    for (Index e = 0; e < model.mesh().num_elements(); ++e) {
      if (model.material_of(e).plasticity().enabled()) {
        throw ConfigError("material '" + model.material_of(e).name() +
                          "' is elastoplastic, which takes the Saint Venant-Kirchhoff form; "
                          "the neo-Hookean law cannot be combined with plasticity");
      }
    }
  }
  // The tools and their surfaces (refused here rather than at the first
  // step).
  const ToolContact probe(model, options_.tools, options_.friction_tangent);
  const std::size_t nt = options_.tools.size();
  const Mesh& mesh = model.mesh();
  std::vector<char> previous_active(nt, 0);
  std::set<std::string> names;
  for (std::size_t s = 0; s < options_.steps.size(); ++s) {
    const FormingStep& step = options_.steps[s];
    const std::string label = "forming step '" + step.name + "'";
    if (!names.insert(step.name).second) {
      throw ConfigError("forming step name '" + step.name + "' is used twice");
    }
    if (step.increments < 0 || !(step.max_tool_travel >= 0.0)) {
      throw ConfigError(label + ": 'increments' and 'max_tool_travel' must not be negative");
    }
    if (step.type == FormingStep::Type::FormExplicit) {
      const bool speed = step.tool_speed > 0.0 && std::isfinite(step.tool_speed);
      const bool duration = step.duration > 0.0 && std::isfinite(step.duration);
      if (speed == duration || !(step.tool_speed >= 0.0) || !(step.duration >= 0.0)) {
        throw ConfigError(label + " (form_explicit) needs its physical time: either a positive "
                                  "'tool_speed' [m/s] or a positive 'duration' [s], not both");
      }
      if (speed && step.tools.empty()) {
        throw ConfigError(label + " (form_explicit) has no tool to set its time by "
                                  "'tool_speed'; give a 'duration' [s]");
      }
      step.explicit_options.validate(label);
    }
    std::vector<char> active(nt, 0);
    for (const std::string& name : step.tools) {
      const int k = probe.find(name);
      if (k < 0) {
        throw ConfigError(label + " names tool '" + name + "', which is not defined");
      }
      if (active[static_cast<std::size_t>(k)]) {
        throw ConfigError(label + " lists tool '" + name + "' twice");
      }
      if (step.type == FormingStep::Type::Release &&
          !previous_active[static_cast<std::size_t>(k)]) {
        throw ConfigError(label + " is a release that keeps tool '" + name +
                          "', which the step before it does not have active; a release can "
                          "only keep tools already in place");
      }
      active[static_cast<std::size_t>(k)] = 1;
    }
    // The constraints must hold the body against every rigid motion.
    std::vector<StepConstraint> constraints = step.constraints;
    if (constraints.empty()) {
      for (const DisplacementConstraint& c : model.constraints()) {
        constraints.push_back({c.region.name, c, StepConstraint::Mode::Absolute});
      }
    }
    if (constraints.empty()) {
      throw ConfigError(label + " has no constraint and the model has no boundary condition");
    }
    const Partition p = build_partition(mesh, constraints, step.name);
    const std::vector<std::string> motions = free_rigid_motions(mesh, p.fixed);
    if (!motions.empty()) {
      std::ostringstream os;
      os << label << ": its constraints leave " << motions.size()
         << " rigid-body motion(s) of the model free (";
      for (std::size_t i = 0; i < motions.size(); ++i) os << (i ? ", " : "") << motions[i];
      os << "); the tangent would be singular - constrain them (a release needs "
            "statically determinate supports, e.g. 3-2-1: a node fixed in x, y, z, a second "
            "in two components, a third in one)";
      throw ConfigError(os.str());
    }
    active_.push_back(active);
    previous_active = active;
  }
  windows_ = resolve_windows(-std::numeric_limits<Scalar>::infinity());
  check_tool_continuity(windows_, std::vector<char>(nt, 0),
                        -std::numeric_limits<Scalar>::infinity(),
                        kJumpTolerance * probe.min_node_size());
}

void FormingAnalysis::check_tool_continuity(const std::vector<std::pair<Scalar, Scalar>>& windows,
                                            std::vector<char> previous, Scalar previous_end,
                                            Scalar tolerance) const {
  const std::size_t nt = options_.tools.size();
  for (std::size_t s = 0; s < options_.steps.size(); ++s) {
    const Scalar tb = windows[s].first;
    for (std::size_t k = 0; k < nt; ++k) {
      if (!active_[s][k] || !previous[k] || !std::isfinite(previous_end)) continue;
      const ToolTrajectory& path = options_.tools[k].trajectory;
      const Vector3 from = path.position(previous_end);
      const Vector3 to = path.position(tb);
      const Scalar jump = (to - from).norm();
      if (!(jump > tolerance)) continue;
      std::ostringstream os;
      os << "forming step '" << options_.steps[s].name << "' starts at t = " << tb << " s, but "
         << (s == 0 ? std::string("the start state is at")
                    : "step '" + options_.steps[s - 1].name + "' ended at")
         << " t = " << previous_end << " s: tool '" << options_.tools[k].name
         << "', active in both, would jump " << jump << " m, from "
         << point_text(from, model_.dim()) << " to " << point_text(to, model_.dim())
         << ", without travelling the path between - start the step where the one before "
         << "it ended, or hold the tool there over the gap";
      throw ConfigError(os.str());
    }
    previous = active_[s];
    previous_end = windows[s].second;
  }
}

std::vector<std::pair<Scalar, Scalar>> FormingAnalysis::resolve_windows(Scalar previous_end) const {
  std::vector<std::pair<Scalar, Scalar>> windows;
  const std::size_t nt = options_.tools.size();
  for (std::size_t s = 0; s < options_.steps.size(); ++s) {
    const FormingStep& step = options_.steps[s];
    const std::vector<char>& active = active_[s];
    const std::string label = "forming step '" + step.name + "'";
    Scalar tb = step.t_begin;
    Scalar te = step.t_end;
    if (std::isnan(tb) != std::isnan(te)) {
      throw ConfigError(label + ": give both ends of the time window or neither");
    }
    if (std::isnan(tb)) {
      const Scalar start = std::isfinite(previous_end) ? previous_end : 0.0;
      if (step.type != FormingStep::Type::Release && !step.tools.empty()) {
        Scalar lo = std::numeric_limits<Scalar>::infinity();
        Scalar hi = -std::numeric_limits<Scalar>::infinity();
        for (std::size_t k = 0; k < nt; ++k) {
          if (!active[k]) continue;
          lo = std::min(lo, options_.tools[k].trajectory.start());
          hi = std::max(hi, options_.tools[k].trajectory.end());
        }
        tb = std::isfinite(previous_end) ? std::max(previous_end, lo) : lo;
        te = hi;
        if (!(te > tb)) {
          std::ostringstream os;
          os << label << " has an empty time window: its tools' trajectories end at t = " << hi
             << " s, but the step before it ended at t = " << previous_end
             << " s; give the step a 'time' window";
          throw ConfigError(os.str());
        }
      } else {
        tb = start;
        te = start + 1.0;
      }
    }
    if (!std::isfinite(tb) || !std::isfinite(te) || !(te > tb)) {
      throw ConfigError(label + ": the time window must be finite with its end after its "
                                "beginning");
    }
    if (std::isfinite(previous_end) &&
        tb < previous_end - 1.0e-12 * std::max(1.0, std::abs(tb))) {
      std::ostringstream os;
      os << label << " starts at t = " << tb << " s, before "
         << (s == 0 ? "the start state (t = " : "the step before it ended (t = ") << previous_end
         << " s): the pseudo-time cannot run backwards";
      throw ConfigError(os.str());
    }
    windows.emplace_back(tb, te);
    previous_end = te;
  }
  return windows;
}

FormingResult FormingAnalysis::run() {
  AnalysisState start;
  start.displacement = Vector::Zero(model_.dofs().num_dofs());
  start.time = windows_.front().first;
  return run(start);
}

// ---------------------------------------------------------------------------
// The step sequence
// ---------------------------------------------------------------------------
FormingResult FormingAnalysis::run(const AnalysisState& start) {
  Timer wall;
  const Mesh& mesh = model_.mesh();
  const int dim = mesh.dim();
  const Index n = model_.dofs().num_dofs();
  const Scalar eps = std::numeric_limits<Scalar>::epsilon();
  if (start.displacement.size() != n) {
    throw ConfigError("the start state has " + std::to_string(start.displacement.size()) +
                      " displacement DOF(s); the model has " + std::to_string(n));
  }
  if (!std::isfinite(start.time)) throw ConfigError("the start state's time is not finite");
  // The steps' windows from the start state on (a restart continues the
  // pseudo-time where it left off).
  const std::vector<std::pair<Scalar, Scalar>> windows = resolve_windows(start.time);

  NonlinearOptions nl;
  nl.kinematics = options_.kinematics;
  nl.law = options_.law;
  nl.mean_dilatation = options_.mean_dilatation;
  NonlinearSystem system(model_, assembler_, nl);
  if (!start.plastic.empty()) system.set_committed(start.plastic);
  if (!start.internal.empty()) system.set_internal(start.internal);
  ToolContact contact(model_, options_.tools, options_.friction_tangent);
  if (!start.friction.empty()) contact.set_history(start.friction);
  // The tools in place in the state (active in the step that produced it).
  const std::size_t nt = options_.tools.size();
  std::vector<char> placed(nt, 0);
  if (!start.tools_active.empty()) {
    if (start.tools_active.size() != nt) {
      throw ConfigError("the start state names " + std::to_string(start.tools_active.size()) +
                        " tool(s) active; the analysis has " + std::to_string(nt));
    }
    placed = start.tools_active;
  }
  check_tool_continuity(windows, placed, start.time,
                        kJumpTolerance * contact.min_node_size());
  PartitionedFactor factor(options_.suitesparse);

  FormingResult result;
  result.plastic = system.plastic();
  for (Index e = 0; e < mesh.num_elements(); ++e) {
    if (system.averaged(e)) result.mean_dilatation = true;
  }
  result.kinematics = to_string(options_.kinematics);
  TimingLedger& timing = result.timing;

  Vector u = start.displacement;
  // The velocity an explicit step hands on (empty: at rest).
  Vector velocity = start.velocity.size() == n ? start.velocity : Vector();
  Scalar time = start.time;
  Scalar reference = start.reference_force;  // running largest F_ref
  Vector last_residual = start.residual.size() == n ? start.residual : Vector::Zero(n);
  Scalar last_du = 0.0;                      // largest nodal |du| of the last increment
  bool strain_warned = false;                // the small-strain warning is given once
  // |X| per DOF, and its norm [m].
  Vector coordinates_abs(n);
  for (Index node = 0; node < mesh.num_nodes(); ++node) {
    const Vector3 x = mesh.node(node);
    for (int k = 0; k < dim; ++k) coordinates_abs(node * dim + k) = std::abs(x(k));
  }
  const Scalar coordinates = coordinates_abs.norm();

  // Residual, reactions and scales of a state.
  struct Balance {
    Vector full;       // f_int - f_c
    Vector free;       // R_f - (1 - s) R_0f
    Scalar scale = 0;  // F_ref at this state
    Scalar floor = 0;  // round-off floor (0 if not resolved)
    Scalar raw_floor = 0;  // the same, not gated
    Scalar reactions = 0;
    Scalar parts = 0;  // the largest force measure of the state itself
  };
  const auto balance = [&](const Evaluation& ev, const ToolContactEvaluation& cev,
                           const Partition& p, const Vector& r0, Scalar ramp, Scalar k_gross,
                           Scalar k_position) {
    Balance b;
    b.full = ev.residual + cev.residual;
    b.free = restrict(b.full, p.free);
    if (r0.size() > 0 && ramp != 0.0) b.free -= ramp * r0;
    b.reactions = restrict(b.full, p.fixed).norm();
    b.parts = std::max({ramp * r0.norm(), b.reactions, cev.force_norm});
    b.scale = std::max({reference, b.parts, 1.0e-300});
    // The round-off floor (NonlinearSystem.hpp), with the contact's: its
    // gaps are differences of positions, each exact to its rounding.
    const Scalar gross = ev.gross + cev.gross + ramp * r0.norm();
    const Scalar floor =
        std::max({1024.0 * eps * gross, 64.0 * eps * k_gross, 8.0 * eps * cev.round_off});
    b.floor = floor <= kFloorLimit * b.scale ? floor : 0.0;
    // With contact a displacement is resolved only to the rounding of the
    // position X + u it defines, which the stiffness turns into forces of
    // eps |K| (|X| + |u|): the floor of a state whose correction is already
    // at round-off (never accepted alone).
    b.raw_floor = std::max(floor, 64.0 * eps * k_position);
    return b;
  };
  const auto evaluate_tangent = [&](const Vector& state, Scalar t, Evaluation& ev,
                                    ToolContactEvaluation& cev) {
    {
      ScopedTimer st(timing, "element_tangent");
      ev = system.evaluate(state, 0.0, true);
    }
    ScopedTimer st(timing, "contact");
    cev = contact.evaluate(state, t, true);
  };
  const auto evaluate_residual = [&](const Vector& state, Scalar t, Evaluation& ev,
                                     ToolContactEvaluation& cev) {
    {
      ScopedTimer st(timing, "element_residual");
      ev = system.evaluate(state, 0.0, false);
    }
    ScopedTimer st(timing, "contact");
    cev = contact.evaluate(state, t, false);
  };

  // Step loop: the deck's steps in order, each starting from the state the one before left.
  for (std::size_t si = 0; si < options_.steps.size(); ++si) {
    const FormingStep& step = options_.steps[si];
    const std::vector<char>& active = active_[si];
    const Scalar tb = windows[si].first;
    const Scalar te = windows[si].second;
    const Scalar duration = te - tb;
    const std::string label = "step '" + step.name + "'";
    FormingStepResult sr;
    sr.name = step.name;
    sr.type = step.type;
    sr.t_begin = tb;
    sr.t_end = te;
    for (std::size_t k = 0; k < active.size(); ++k) {
      if (active[k]) sr.tools.push_back(options_.tools[k].name);
    }
    const bool release = step.type == FormingStep::Type::Release;

    // The step's partition and its prescribed values.
    std::vector<StepConstraint> constraints = step.constraints;
    if (constraints.empty()) {
      for (const DisplacementConstraint& c : model_.constraints()) {
        constraints.push_back({c.region.name, c, StepConstraint::Mode::Absolute});
      }
    }
    const Partition part = build_partition(mesh, constraints, step.name);
    sr.constrained_dofs = static_cast<int>(part.fixed.size());
    const auto nfix = static_cast<Eigen::Index>(part.fixed.size());
    Vector p_start(nfix);
    Vector p_end(nfix);
    for (Eigen::Index i = 0; i < nfix; ++i) {
      const Index d = part.fixed[static_cast<std::size_t>(i)];
      p_start(i) = u(d);
      p_end(i) = part.mode[static_cast<std::size_t>(i)] == StepConstraint::Mode::Hold
                     ? u(d)
                     : part.target(i);
    }
    factor.reset(part.free, n);
    contact.clear_inactive(active);
    const Vector u_step = u;

    // The imbalance of the start state on the step's free DOFs, ramped out
    // over the step. A start state that cannot be evaluated (a tool placed
    // through the surface) stops the analysis there: the step is recorded,
    // not completed, with the state it started from.
    Vector r0;
    bool started = true;
    {
      contact.begin_increment(u, tb, tb, active, std::numeric_limits<Scalar>::infinity());
      Evaluation ev;
      ToolContactEvaluation cev;
      try {
        evaluate_residual(u, tb, ev, cev);
        r0 = restrict(ev.residual + cev.residual, part.free);
        sr.start_imbalance = r0.norm();
      } catch (const SolverError& ex) {
        std::ostringstream os;
        os << "its start state (t = " << tb << " s) cannot be evaluated: " << ex.what();
        sr.termination = os.str();
        started = false;
      }
      // A tool that becomes active already in contact: the force of its
      // penetration is part of R_0, ramped in over the step at the tools'
      // positions rather than reached by travel.
      for (std::size_t k = 0; started && k < nt; ++k) {
        const ToolResultant& tr = cev.tools[k];
        if (!active[k] || placed[k] || tr.active_nodes == 0 ||
            !(tr.max_penetration > kJumpTolerance * contact.min_node_size())) {
          continue;
        }
        std::ostringstream os;
        os << "tool '" << options_.tools[k].name << "' becomes active in contact: "
           << tr.active_nodes << " node(s) up to " << tr.max_penetration
           << " m inside it at t = " << tb << " s; the force of that penetration, "
           << tr.force.norm()
           << (step.type == FormingStep::Type::FormExplicit
                   ? " N (the implicit penalty's), acts at once instead of being reached by "
                     "travel - start its path clear of the surface"
                   : " N, is ramped in over the step at the tool's positions instead of being "
                     "reached by travel - start its path clear of the surface");
        sr.warnings.push_back(os.str());
        log::warn(label, ": ", os.str());
      }
    }
    if (started && step.type == FormingStep::Type::FormExplicit) {
      // An explicit step ramps nothing out: it integrates from the state as
      // it is (its imbalance above, with the implicit penalty, is only the
      // check that the state can be evaluated).
      sr.start_imbalance = 0.0;
      log::info("forming ", label, " (", to_string(step.type), ", t = ", tb, " .. ", te, " s, ",
                sr.tools.size(), " tool(s), ", part.fixed.size(), " prescribed DOF(s))");
    } else if (started) {
      log::info("forming ", label, " (", to_string(step.type), ", t = ", tb, " .. ", te, " s, ",
                sr.tools.size(), " tool(s), ", part.fixed.size(), " prescribed DOF(s)): start "
                "imbalance ", sr.start_imbalance, " N");
    }

    // Stations: the knots of the active tools inside the window, and its end.
    std::vector<Scalar> stations;
    for (std::size_t k = 0; k < active.size(); ++k) {
      if (!active[k]) continue;
      for (Scalar t : options_.tools[k].trajectory.knots_in(tb, te)) stations.push_back(t);
    }
    std::sort(stations.begin(), stations.end());
    stations.erase(std::unique(stations.begin(), stations.end()), stations.end());
    stations.push_back(te);
    const int parts = step.increments > 0 ? step.increments : (release ? 10 : 1);
    const Scalar dt_max = duration / static_cast<Scalar>(parts);
    Scalar travel = step.max_tool_travel;
    if (!(travel > 0.0)) travel = 0.5 * contact.min_node_size();
    bool any_tool = false;
    for (char a : active) any_tool = any_tool || a;

    // Newton's method for one increment to (t1, s1).
    std::string failure;  // why the last increment failed
    Scalar accepted_round_off = 0.0;  // round-off floor of the last accepted state
    const auto newton = [&](Scalar t0, Scalar t1, Scalar s1, FormingIncrement& inc,
                            Evaluation& accepted, ToolContactEvaluation& accepted_contact,
                            Vector& state) -> bool {
      // Predictor at (t1, s1): prescribed DOFs moved, start imbalance R_0f scaled by 1 - s1.
      const Scalar ramp = 1.0 - s1;
      Vector trial = u;
      Vector dp(nfix);
      for (Eigen::Index i = 0; i < nfix; ++i) {
        const Index d = part.fixed[static_cast<std::size_t>(i)];
        trial(d) = p_start(i) + s1 * (p_end(i) - p_start(i));
        dp(i) = trial(d) - u(d);
      }
      predict_rigid_motion(mesh, u, part, dp, trial);
      Scalar tool_travel = 0.0;
      for (std::size_t k = 0; k < active.size(); ++k) {
        if (!active[k]) continue;
        const ToolTrajectory& path = options_.tools[k].trajectory;
        tool_travel = std::max(tool_travel, (path.position(t1) - path.position(t0)).norm());
      }
      // Contact update: the candidate nodes of each active tool for this increment.
      contact.begin_increment(u, t0, t1, active,
                              2.0 * tool_travel + 2.0 * last_du + contact.min_node_size());
      const Vector& from = u;
      // A correction below this is converged: the displacement tolerance
      // relative to the increment, no finer than the round-off of the
      // positions X + u (a gap subtracts positions, so from the reference
      // state - u = 0 - a grazing contact resolves no finer than that).
      const auto correction_limit = [&](Scalar increment, const Vector& x) {
        return std::max(options_.displacement_tolerance * std::max(increment, 1.0e-300),
                        64.0 * eps * (x.norm() + coordinates));
      };
      // Line search on the energy along du, with
      // g(1) already known (NaN: the full step was invalid - an inverted
      // element or a node through a tool - and is halved).
      const auto line_search = [&](const Vector& x, const Vector& du, Scalar g0,
                                   Scalar g1) -> Scalar {
        const auto g = [&](Scalar a) -> Scalar {
          Vector probe = x;
          add_to(probe, part.free, du, a);
          try {
            Evaluation ev;
            ToolContactEvaluation cev;
            evaluate_residual(probe, t1, ev, cev);
            Vector rf = restrict(ev.residual + cev.residual, part.free);
            if (ramp != 0.0) rf -= ramp * r0;
            return du.dot(rf);
          } catch (const SolverError&) {
            return std::numeric_limits<Scalar>::quiet_NaN();
          }
        };
        constexpr Scalar slack = 0.8;
        constexpr Scalar shortest = 0.1;
        Scalar a_hi = 1.0;
        Scalar g_hi = g1;
        for (int k = 0; k < 6 && !std::isfinite(g_hi); ++k) {
          a_hi *= 0.5;
          g_hi = g(a_hi);
        }
        if (!std::isfinite(g_hi) || std::abs(g_hi) <= slack * std::abs(g0) || g_hi < 0.0) {
          return a_hi;
        }
        Scalar a_lo = 0.0;
        Scalar g_lo = g0;
        Scalar best = a_hi;
        for (int it = 0; it < 5; ++it) {
          const Scalar a =
              std::clamp(a_lo - g_lo * (a_hi - a_lo) / (g_hi - g_lo), shortest * a_hi, a_hi);
          const Scalar ga = g(a);
          if (!std::isfinite(ga)) {
            a_hi = a;
            g_hi = std::abs(g0);
            continue;
          }
          best = a;
          if (std::abs(ga) <= slack * std::abs(g0)) break;
          if (ga < 0.0) {
            a_lo = a;
            g_lo = ga;
          } else {
            a_hi = a;
            g_hi = ga;
          }
        }
        return best;
      };

      // The evaluation (with the tangent) at `trial`, when the last full
      // Newton step computed it for the line search.
      Evaluation ev;
      ToolContactEvaluation cev;
      bool have = false;
      int anticipated = 0;  // nodes the active-set prediction added
      // Newton iteration: assemble residual and tangent, solve, line search, convergence test.
      for (int it = 1; it <= options_.max_iterations; ++it) {
        if (!have) evaluate_tangent(trial, t1, ev, cev);
        have = false;
        for (const ToolContactEvaluation::NodeBlock& block : cev.tangent) {
          add_node_block(ev.tangent, block.node, dim, block.k);
        }
        if (!ev.tangent.isCompressed()) ev.tangent.makeCompressed();
        const Scalar k_gross = stiffness_gross(ev.tangent, trial, part.free);
        const Scalar k_position =
            stiffness_gross(ev.tangent, coordinates_abs + trial.cwiseAbs(), part.free);
        const Balance b = balance(ev, cev, part, r0, ramp, k_gross, k_position);
        const Vector& r = b.free;
        if (!r.allFinite()) {
          failure = "a non-finite residual";
          return false;
        }
        // Solve K_ff du = -r on the step's free DOFs (PartitionedFactor above).
        bool factorised = false;
        {
          ScopedTimer st(timing, "factorisation");
          factor.assemble(ev.tangent);
          factorised = factor.factorize(system.symmetric() && cev.symmetric, timing);
        }
        if (!factorised) {
          failure = "the tangent is singular - a rigid-body motion or a mechanism the "
                    "step's constraints and tools leave free";
          return false;
        }
        Vector du;
        {
          ScopedTimer st(timing, "solve");
          du = factor.solve(-r);
        }
        if (!du.allFinite()) {
          failure = "a non-finite Newton correction (a singular tangent)";
          return false;
        }
        // Semismooth Newton: a node the correction would drive into a tool
        // from outside has no penalty stiffness in the tangent, so the step
        // overshoots it deep into the tool and the line search then creeps
        // towards the contact, iteration after iteration. Such nodes join
        // the active set with the penalty law extended to their gap, and the
        // correction is solved again (once).
        if (any_tool) {
          Vector du_full = Vector::Zero(n);
          add_to(du_full, part.free, du, 1.0);
          Vector extra = Vector::Zero(n);
          std::vector<ToolContactEvaluation::NodeBlock> blocks;
          int added = 0;
          {
            ScopedTimer st(timing, "contact");
            added = contact.anticipate(trial, du_full, t1, extra, blocks);
          }
          if (added > 0) {
            for (const ToolContactEvaluation::NodeBlock& block : blocks) {
              add_node_block(ev.tangent, block.node, dim, block.k);
            }
            if (!ev.tangent.isCompressed()) ev.tangent.makeCompressed();
            bool again = false;
            {
              ScopedTimer st(timing, "factorisation");
              factor.assemble(ev.tangent);
              again = factor.factorize(system.symmetric() && cev.symmetric, timing);
            }
            if (again) {
              ScopedTimer st(timing, "solve");
              const Vector predicted = factor.solve(-(r + restrict(extra, part.free)));
              if (predicted.allFinite()) du = predicted;
            }
            anticipated += added;
          }
        }
        // The full step is evaluated with its tangent: kept, it is the next
        // iteration's evaluation (and the accepted state's, at convergence).
        Scalar alpha = 1.0;
        Evaluation next;
        ToolContactEvaluation next_contact;
        const Scalar g0 = du.dot(r);
        if (options_.line_search && g0 < 0.0) {
          Vector probe = trial;
          add_to(probe, part.free, du, 1.0);
          Scalar g1 = std::numeric_limits<Scalar>::quiet_NaN();
          try {
            evaluate_tangent(probe, t1, next, next_contact);
            Vector rf = restrict(next.residual + next_contact.residual, part.free);
            if (ramp != 0.0) rf -= ramp * r0;
            g1 = du.dot(rf);
          } catch (const SolverError&) {
          }
          alpha = line_search(trial, du, g0, g1);
          have = alpha == 1.0;
        }
        const bool debug = static_cast<int>(log::level()) <= static_cast<int>(log::Level::Debug);
        int in_contact = 0;
        int slipping = 0;
        const int yielding = ev.yielding_points;
        if (debug) {
          for (const ToolResultant& tr : cev.tools) {
            in_contact += tr.active_nodes;
            slipping += tr.slipping_nodes;
          }
        }
        const Scalar previous_norm = r.norm();
        const Scalar previous_scale = b.scale;
        const Scalar floor = b.floor;
        const Scalar raw_floor = b.raw_floor;
        add_to(trial, part.free, du, alpha);
        if (have) {
          ev = std::move(next);
          cev = std::move(next_contact);
        }
        const Scalar increment = restrict(trial - from, part.free).norm();
        const Scalar rel = previous_norm / previous_scale;
        if (debug) {
          log::debug("  forming newton ", it, " at t = ", t1, ": |R| = ", previous_norm,
                     " (scale ", previous_scale, ", floor ", floor, "), |du| = ",
                     alpha * du.norm(), " (limit ", correction_limit(increment, trial),
                     "), line search ", alpha, ", ", in_contact, " node(s) in contact (",
                     slipping, " slipping), ", yielding, " point(s) yielding, ",
                     anticipated, " node(s) anticipated so far");
        }
        // Converged: residual and correction within tolerance, or the
        // residual at its round-off floor. A floor far above the forces of
        // the state is not accepted alone (NonlinearSystem.hpp: a runaway
        // inflates it), but it is with a correction within tolerance - a
        // state with no force to resolve, such as a stress-free body moved
        // rigidly by its supports, whose reference force is round-off.
        const bool small_correction = alpha * du.norm() <= correction_limit(increment, trial);
        if ((previous_norm <= options_.residual_tolerance * previous_scale && small_correction) ||
            previous_norm <= floor || (small_correction && previous_norm <= raw_floor)) {
          // The residual of the accepted state, from its own evaluation.
          const bool residual_only = !have;
          if (residual_only) evaluate_residual(trial, t1, ev, cev);
          const Balance fb = balance(ev, cev, part, r0, ramp, k_gross, k_position);
          const Scalar final_norm = fb.free.norm();
          const Scalar accepted_floor = small_correction ? fb.raw_floor : fb.floor;
          if (final_norm <= std::max(options_.residual_tolerance * fb.scale, accepted_floor)) {
            inc.iterations = it;
            accepted_round_off = fb.raw_floor;
            inc.residual = final_norm / fb.scale;
            reference = std::max(reference, fb.parts);
            accepted = std::move(ev);
            accepted_contact = std::move(cev);
            state = std::move(trial);
            return true;
          }
          // Not accepted: the next iteration needs the tangent here.
          have = !residual_only;
        }
        if (rel > 1.0e8) {
          failure = "Newton's method diverged";
          return false;
        }
      }
      failure = "Newton's method did not converge in max_iterations";
      return false;
    };

    // An explicit step: central differences over the window's physical time.
    const bool explicit_step = step.type == FormingStep::Type::FormExplicit;
    bool explicit_done = false;
    if (explicit_step && started) {
      ExplicitDrive drive;
      drive.fixed = part.fixed;
      drive.fixed_start = p_start;
      drive.fixed_end = p_end;
      drive.active = active;
      std::vector<const ToolTrajectory*> paths;
      for (std::size_t k = 0; k < nt; ++k) {
        if (active[k]) paths.push_back(&options_.tools[k].trajectory);
      }
      drive.time_map = step.tool_speed > 0.0
                           ? ExplicitTimeMap::tool_speed(paths, tb, te, step.tool_speed)
                           : ExplicitTimeMap::linear(tb, te, step.duration);
      drive.duration = drive.time_map.duration();
      ExplicitState from;
      from.displacement = u;
      from.velocity = velocity;
      from.history = system.committed_all();
      from.friction = contact.history();
      ExplicitDynamics integrator(model_, assembler_, nl, options_.tools, step.explicit_options);
      log::info("forming ", label, ": explicit, ", drive.duration, " s of physical time",
                step.tool_speed > 0.0 ? " (tool speed " + std::to_string(step.tool_speed) +
                                            " m/s)"
                                      : std::string());
      ExplicitResult er = integrator.run(drive, from);
      for (const auto& [name, seconds] : er.timing.totals()) {
        timing.add(name == "total" ? "explicit" : "explicit_" + name, seconds);
      }
      // The state it hands on (the last valid one when it stopped).
      u = er.final_state.displacement;
      velocity = er.final_state.velocity;
      system.set_committed(er.final_state.history);
      contact.set_history(er.final_state.friction);
      last_residual = er.residual;
      time = drive.time_map.pseudo_time(er.duration);
      if (er.steps > 0) placed = active;
      // The recorded steps are the step's increments (tool_forces.csv).
      for (const ExplicitRecord& rec : er.records) {
        for (const ExplicitToolRecord& tr : rec.tools) {
          reference = std::max(reference, tr.force.norm());
        }
        if (rec.step == 0) continue;
        FormingIncrement inc;
        inc.index = static_cast<int>(std::min<long>(rec.step, std::numeric_limits<int>::max()));
        inc.time = rec.pseudo_time;
        inc.max_plastic_strain = rec.max_plastic_strain;
        for (const ExplicitToolRecord& tr : rec.tools) {
          ToolRecord tool;
          tool.tool = tr.tool;
          tool.centre = tr.centre;
          tool.force = tr.force;
          tool.active_nodes = tr.active_nodes;
          tool.slipping_nodes = tr.slipping_nodes;
          tool.max_penetration = tr.max_penetration;
          inc.tools.push_back(tool);
        }
        sr.increments.push_back(std::move(inc));
      }
      for (const ExplicitSnapshot& snap : er.snapshots) {
        sr.snapshots.push_back(
            {static_cast<int>(std::min<long>(snap.step, std::numeric_limits<int>::max())),
             snap.pseudo_time, snap.displacement});
      }
      reference = std::max(reference, restrict(er.reactions, part.fixed).norm());
      for (const std::string& w : er.warnings) sr.warnings.push_back(w);
      if (!er.completed) sr.termination = "the explicit integration stopped at " + er.termination;
      explicit_done = er.completed;
      // The report without the end state, which the step keeps itself.
      er.final_state = ExplicitState();
      er.acceleration.resize(0);
      er.residual.resize(0);
      er.reactions.resize(0);
      er.snapshots.clear();
      sr.explicit_step = true;
      sr.explicit_tool_speed = step.tool_speed;
      sr.explicit_result = std::move(er);
    }

    // The increments.
    Timer progress;  // since the last progress report
    Scalar t = tb;
    Scalar dt = dt_max;
    std::size_t next_station = 0;
    int cuts_in_a_row = 0;
    bool done = explicit_done;
    // Increment loop: size the increment (tool travel, stations), run Newton, cut or commit.
    while (started && !done && !explicit_step) {
      if (static_cast<int>(sr.increments.size()) >= options_.max_increments) {
        sr.termination = "the increment budget (max_increments) ran out";
        break;
      }
      while (next_station < stations.size() && stations[next_station] <= t) ++next_station;
      const Scalar station = stations[next_station];
      Scalar h = std::min(dt, dt_max);
      if (any_tool) {
        Scalar speed = 0.0;
        for (std::size_t k = 0; k < active.size(); ++k) {
          if (active[k]) speed = std::max(speed, options_.tools[k].trajectory.velocity(t).norm());
        }
        if (speed > 0.0) h = std::min(h, travel / speed);
      }
      const bool reach = station - t <= h * (1.0 + 1.0e-6);
      const Scalar t1 = reach ? station : t + h;
      const Scalar s1 = reach && station == te ? 1.0 : (t1 - tb) / duration;
      FormingIncrement inc;
      Evaluation accepted;
      ToolContactEvaluation accepted_contact;
      Vector state;
      bool ok = false;
      try {
        ok = newton(t, t1, s1, inc, accepted, accepted_contact, state);
      } catch (const SolverError& ex) {
        failure = ex.what();
        ok = false;
      }
      // Cut: halve the increment and retry from the last converged state.
      if (!ok) {
        ++cuts_in_a_row;
        ++sr.cuts;
        if (cuts_in_a_row > options_.max_cuts) {
          std::ostringstream os;
          os << "an increment from t = " << t << " s failed to converge after "
             << options_.max_cuts << " halvings (last increment " << t1 - t << " s): "
             << failure;
          sr.termination = os.str();
          break;
        }
        dt = 0.5 * (t1 - t);
        log::info("forming increment to t = ", t1, " failed (", failure, "); halving to ", dt);
        continue;
      }
      // Commit the converged increment.
      Scalar du_max = 0.0;
      for (Index node = 0; node < mesh.num_nodes(); ++node) {
        du_max = std::max(
            du_max, (state.segment(node * dim, dim) - u.segment(node * dim, dim)).norm());
      }
      last_du = du_max;
      system.commit(accepted);
      contact.commit(accepted_contact);
      placed = active;
      u = std::move(state);
      last_residual = accepted.residual + accepted_contact.residual;
      t = t1;
      time = t1;
      inc.index = static_cast<int>(sr.increments.size()) + 1;
      inc.time = t1;
      inc.max_plastic_strain = system.max_plastic_strain();
      for (std::size_t k = 0; k < active.size(); ++k) {
        if (!active[k]) continue;
        const ToolResultant& tr = accepted_contact.tools[k];
        ToolRecord rec;
        rec.tool = k;
        rec.centre = tr.centre;
        rec.force = tr.force;
        rec.active_nodes = tr.active_nodes;
        rec.slipping_nodes = tr.slipping_nodes;
        rec.max_penetration = tr.max_penetration;
        inc.tools.push_back(rec);
      }
      sr.iterations += inc.iterations;
      {
        std::ostringstream os;
        os << "forming " << label << " increment " << inc.index << ": t = " << t1 << " s, "
           << inc.iterations << " iteration(s), residual " << inc.residual;
        for (const ToolRecord& rec : inc.tools) {
          os << ", tool '" << options_.tools[rec.tool].name << "' " << rec.active_nodes
             << " node(s), force " << rec.force.head(dim).transpose() << " N";
        }
        if (system.plastic()) os << ", largest plastic strain " << inc.max_plastic_strain;
        log::debug(os.str());
      }
      if (options_.snapshot_stride > 0 && inc.index % options_.snapshot_stride == 0 && s1 < 1.0) {
        sr.snapshots.push_back({inc.index, t1, u});
      }
      if (log::level() <= log::Level::Info && progress.elapsed_seconds() >= kProgressSeconds) {
        progress.reset();
        log::info("forming ", label, ": t = ", t1, " s of ", te, " (", sr.increments.size() + 1,
                  " increment(s), ", sr.iterations, " iteration(s), ", sr.cuts, " cut(s), ",
                  wall.elapsed_seconds(), " s elapsed)");
      }
      sr.increments.push_back(std::move(inc));
      if (sr.increments.back().iterations <= kQuickIterations) dt = std::min(kGrowth * dt, dt_max);
      cuts_in_a_row = 0;
      done = s1 >= 1.0;
    }
    result.total_cuts += sr.cuts;
    result.total_iterations += sr.iterations;
    if (explicit_step) {
      result.total_explicit_steps += sr.explicit_result.steps;
    } else {
      result.total_increments += static_cast<int>(sr.increments.size());
      velocity.resize(0);  // an implicit step ends at rest
    }
    sr.completed = done;

    // The end state of the step (the last converged one if it failed).
    sr.displacement = u;
    sr.reactions = Vector::Zero(n);
    for (Index d : part.fixed) sr.reactions(d) = last_residual(d);
    sr.reaction_norm = sr.reactions.norm();
    sr.reference_force = std::max(reference, 1.0e-300);
    sr.max_plastic_strain = system.max_plastic_strain();
    sr.max_displacement_change = max_nodal(u - u_step, dim);
    sr.element_plastic_strain = Vector::Zero(mesh.num_elements());
    sr.element_von_mises = Vector::Zero(mesh.num_elements());
    Scalar max_strain = 0.0;
    for (Index e = 0; e < mesh.num_elements(); ++e) {
      const Index* nodes = mesh.element_nodes(e);
      const int npe = mesh.nodes_per_elem();
      Vector ue(dim * npe);
      for (int a = 0; a < npe; ++a) {
        for (int k = 0; k < dim; ++k) ue(dim * a + k) = u(nodes[a] * dim + k);
      }
      if (system.routed_through_return(e)) {
        const ElastoplasticStress st =
            elastoplastic_stress(model_, e, ue, system.committed(e), system.averaged(e), nullptr,
                                 0.0, options_.kinematics, system.internal(e));
        sr.element_von_mises(e) = st.von_mises;
        if (system.elastoplastic(e)) {
          sr.element_plastic_strain(e) = st.max_equivalent_plastic_strain;
        }
        max_strain = std::max(max_strain, st.max_strain);
      } else {
        const TotalLagrangianStress st = total_lagrangian_stress(model_, e, ue, options_.law,
                                                                 nullptr, 0.0, system.internal(e));
        sr.element_von_mises(e) =
            dim == 3 ? von_mises(st.cauchy, StressState::ThreeDimensional, 0.0)
                     : von_mises_plane(st.cauchy(0), st.cauchy(1), st.cauchy(2), st.cauchy_zz);
        max_strain = std::max(max_strain, st.max_green_strain);
      }
    }
    // Release check: supports that still carry load hold the part deformed (a warning).
    // (Reactions at their round-off, as those of a body that was never
    // loaded, are not over-constraint.)
    if (release && sr.completed &&
        sr.reaction_norm > std::max(kReleaseReactionWarning * sr.reference_force,
                                    100.0 * accepted_round_off)) {
      std::ostringstream os;
      os << "the supports of release " << label << " carry reactions of " << sr.reaction_norm
         << " N, " << sr.reaction_norm / sr.reference_force
         << " of the reference force: they are not statically determinate (over-constrained) "
            "and hold the part in a deformed shape - release onto 3-2-1 supports";
      sr.warnings.push_back(os.str());
      log::warn(os.str());
    }
    // Logarithmic kinematics is a large-strain formulation: no warning.
    if (system.plastic() && options_.kinematics != Kinematics::FiniteLogarithmic &&
        max_strain > kStrainWarning && !strain_warned) {
      strain_warned = true;
      std::ostringstream os;
      os << "the largest strain at the end of " << label << " is " << max_strain
         << ", beyond the small strains the additive elastoplastic law is meant for (its "
            "stresses are then approximate; given once)";
      sr.warnings.push_back(os.str());
    }
    if (!sr.completed && sr.termination.empty()) sr.termination = "the step did not complete";
    if (sr.completed) sr.termination = release ? "released" : "reached the end of its window";
    log::info("forming ", label, (sr.completed ? " completed" : " STOPPED"), ": ",
              sr.increments.size(), " increment(s), ", sr.iterations, " iteration(s), ", sr.cuts,
              " cut(s), reactions ", sr.reaction_norm, " N, largest plastic strain ",
              sr.max_plastic_strain, ", largest displacement change ", sr.max_displacement_change,
              " m");
    const bool completed = sr.completed;
    const std::string termination = sr.termination;
    result.steps.push_back(std::move(sr));
    // Hand the ended step to the observer: sparlab_form writes its files here.
    if (observer_) {
      ScopedTimer st(timing, "output");
      observer_(result);
    }
    if (!completed) {
      result.termination = label + " (" + std::to_string(si + 1) + " of " +
                           std::to_string(options_.steps.size()) + "): " + termination;
      break;
    }
  }
  result.completed = result.steps.size() == options_.steps.size() && result.steps.back().completed;
  if (result.completed) result.termination = "completed every step";
  if (!result.completed && result.termination.empty() && !result.steps.empty()) {
    result.termination =
        "step '" + result.steps.back().name + "': " + result.steps.back().termination;
  }
  for (const FormingStepResult& s : result.steps) {
    for (const std::string& w : s.warnings) {
      result.warnings.push_back("step '" + s.name + "': " + w);
    }
  }
  result.final_state.displacement = u;
  result.final_state.plastic = system.committed_all();
  result.final_state.internal = system.internal_all();
  result.final_state.friction = contact.history();
  result.final_state.time = time;
  result.final_state.residual = last_residual;
  result.final_state.reference_force = reference;
  result.final_state.tools_active = placed;
  result.final_state.velocity = velocity;
  result.linear_solver = factor.names();
  result.failed_factorisations = factor.failures();
  timing.add("total", wall.elapsed_seconds());
  if (!result.completed) log::warn("forming analysis stopped: ", result.termination);
  return result;
}

}  // namespace sparlab
