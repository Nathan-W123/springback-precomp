/// \file RigidTool.hpp
/// \brief Rigid forming tools that travel along tabulated paths, and their
///        penalty contact with a surface of the model in the current
///        configuration, with Coulomb friction.
///
/// **Tools.** A tool is a rigid body given analytically - a sphere (the
/// hemispherical tip of an incremental-forming stylus), a plane (a backing
/// plate, a flat punch) or an infinite cylinder (a roller) - whose reference
/// point follows a piecewise-linear trajectory c(t) in pseudo-time: the
/// sphere's centre, a point of the plane, a point of the cylinder's axis.
/// The tool translates; it does not rotate (no tool spin).
///
/// **Contact (penalty, current configuration, large sliding).** Each tool
/// acts on the nodes of a slave surface of the model, the boundary faces
/// whose nodes all lie in a region (as a pressure selects them). The gap of
/// slave node j at the current position \f$x = X_j + u_j\f$ is evaluated
/// exactly, with no linearisation of the geometry, so a tool may travel any
/// distance over the sheet: for the sphere of radius R
/// \f[
///   r = x - c(t),\quad d = |r|,\quad n = r/d,\quad g = d - R ;
/// \f]
/// for the plane with unit normal n (out of the tool, towards the body)
/// \f$g = n\cdot(x - c(t))\f$; for the cylinder with unit axis a, r is
/// taken normal to the axis, \f$r_\perp = (I - aa^T)(x - c)\f$. Where
/// \f$g < 0\f$ the node penetrates the tool and takes the normal force
/// \f[
///   f_N = -\kappa_j A_j\, g\, n = p_N\, n, \qquad p_N = -\kappa_j A_j\, g > 0 ,
/// \f]
/// with \f$A_j = \sum_f \int N_j\,dA\f$ the node's tributary area on the
/// selected reference faces (times the thickness in 2-D) and the penalty
/// stiffness \f$\kappa_j = s E_j / h_j\f$ [Pa/m]: s the penalty scale
/// (default 10), \f$E_j\f$ the largest Young's modulus of the adjacent faces'
/// elements and \f$h_j = \sqrt{A_j}\f$ (3-D) or \f$A_j/t\f$ (2-D) the node's
/// characteristic size, so that the node's spring \f$\kappa_j A_j \approx
/// s E_j h_j\f$ is s times the stiffness of an element of that size. The
/// penetration this leaves is about \f$p\,h/(sE)\f$ for a contact pressure
/// p (4 um for 300 MPa, 1 mm and aluminium at s = 10). The contact force is
/// an external force: it enters the residual \f$R = f_{int} - f_{ext}\f$ as
/// \f$-f_N\f$, and its exact tangent (a node-diagonal block) is
/// \f[
///   \partial(-f_N)/\partial x = \kappa_j A_j \left[ n n^T + g\,\partial n/\partial x \right],
///   \qquad \partial n/\partial x = (I - nn^T)/d
/// \f]
/// for the sphere (\f$(I - aa^T - nn^T)/d\f$ for the cylinder, 0 for the
/// plane): symmetric, the curvature term softening it tangentially.
///
/// **Friction** is regularised Coulomb friction by an elastic-slip return
/// map (backward Euler over an increment), with a history per slave node
/// and tool: the tangential force \f$F_T\f$ on the node and the node's
/// position relative to the tool, \f$q = x - c\f$, both at the last
/// converged increment. At a state (x, t):
/// * the committed force is transported onto the current tangent plane,
///   \f$\tilde F = F_T - (F_T\cdot n)n\f$, rescaled to keep its magnitude,
///   \f$\hat F = |F_T|\,\tilde F/|\tilde F|\f$;
/// * the slip increment is the tangential relative motion
///   \f$\delta = (I - nn^T)(x - c(t) - q)\f$ (for a node that was not in
///   contact at the last converged increment, \f$F_T = 0\f$ and q is its
///   relative position at the start of the increment);
/// * the trial force \f$F^{tr} = \hat F - \kappa_{T,j} A_j\,\delta\f$,
///   \f$\kappa_T = \rho_T \kappa\f$ with the tangential penalty ratio
///   \f$\rho_T\f$ (default 1);
/// * stick where \f$|F^{tr}| \le \mu p_N\f$, \f$F_T = F^{tr}\f$; slip
///   otherwise, \f$F_T = \mu p_N\,F^{tr}/|F^{tr}|\f$.
///
/// The force on the node is \f$f_N + F_T\f$. The tangent is the exact
/// derivative of that force at fixed history - transport, projection, the
/// normal force in the slip cone and the normalisation included - which is
/// not symmetric. `FrictionTangent::Symmetric` replaces its friction part by
/// a positive semi-definite approximation, the friction force's stiffness in
/// the tangent plane alone: \f$\kappa_T A P\f$ in stick and
/// \f$(\mu p_N/|F^{tr}|)\,\kappa_T A\,(P - ee^T)\f$ in slip
/// (\f$P = I - nn^T\f$, \f$e = F^{tr}/|F^{tr}|\f$), without the transport
/// and curvature terms and without the slip force's dependence on the gap,
/// \f$\mu\kappa A\,en^T\f$ - a symmetric factorisation, and Newton
/// converges linearly where nodes slip. (The symmetric part of the exact
/// tangent is no substitute: in the plane of n and e a slipping node's block
/// is \f$\kappa A\,[[1, \mu/2], [\mu/2, 0]]\f$, indefinite, and Newton's
/// method fails with it as contact begins.) A node that leaves contact
/// loses its history. The history is keyed by mesh node, so it does not
/// depend on the Dirichlet partition of an analysis step.
///
/// **Search.** Before each increment, from the converged state
/// \f$u_0\f$ at \f$t_0\f$ to \f$t_1\f$, a node is a candidate for tool k if
/// its position \f$x_j(u_0)\f$ lies within \f$R + m\f$ of the segment
/// \f$[c(t_0), c(t_1)]\f$ the tool's reference point sweeps (increments
/// never cross a trajectory knot, so c moves on that segment), where the
/// margin m bounds how far a node may move in the increment. Every
/// evaluation checks that bound: if some slave node of the tool has moved
/// more than m from \f$x_j(u_0)\f$, that evaluation takes all of the tool's
/// slave nodes. A node left out therefore provably has
/// \f$|x_j - c(t)| > R\f$ for every t of the increment, g > 0: the filter
/// never changes the result.
///
/// **Units.** Strict SI: positions [m], times [s] (pseudo-time),
/// forces [N], penalty stiffness [Pa/m].
#pragma once

#include "sparlab/core/Types.hpp"
#include "sparlab/fem/Selector.hpp"

#include <map>
#include <string>
#include <vector>

namespace sparlab {

class FemModel;

/// A piecewise-linear path of a tool's reference point: `points[i]` at
/// `times[i]`, linear between, constant before the first knot and after the
/// last.
struct ToolTrajectory {
  std::vector<Scalar> times;    ///< pseudo-time [s], strictly increasing
  std::vector<Vector3> points;  ///< [m], one per time

  /// \throws ConfigError for fewer than two knots, times that do not
  ///         increase strictly, a point count that differs from the time
  ///         count, or a non-finite value; `what` names the trajectory.
  void validate(const std::string& what = "trajectory") const;

  Scalar start() const { return times.front(); }
  Scalar end() const { return times.back(); }

  /// The position at t (clamped outside [start, end]).
  Vector3 position(Scalar t) const;
  /// The velocity on the segment that starts at or before t and ends after
  /// it - the right derivative - and 0 from the last knot on and before the
  /// first.
  Vector3 velocity(Scalar t) const;
  /// The knots strictly inside (ta, tb), ascending.
  std::vector<Scalar> knots_in(Scalar ta, Scalar tb) const;

  /// Read a trajectory from a CSV file with the header `t,x,y,z` (pseudo-time
  /// in s, the reference point in m; spaces around fields are ignored, as
  /// are blank lines and lines starting with '#'), and validate it.
  /// \throws IoError when the file cannot be read, ConfigError naming the
  ///         file and line of a malformed header or row.
  static ToolTrajectory from_csv(const std::string& path);

 private:
  /// Index i of the segment [times[i], times[i+1]) holding t (t within
  /// [start, end)). A cached hint makes monotone lookups O(1); the cache is
  /// not synchronised - a trajectory is queried from one thread at a time.
  std::size_t segment(Scalar t) const;
  mutable std::size_t hint_ = 0;
};

/// A rigid tool, its path and the surface of the model it acts on.
struct RigidTool {
  enum class Shape { Sphere, Plane, Cylinder };
  std::string name = "tool";
  Shape shape = Shape::Sphere;
  Scalar radius = 0.0;                ///< sphere, cylinder [m]
  /// Plane: the unit normal, out of the tool towards the body (normalised
  /// on use). Unused otherwise.
  Vector3 normal = Vector3::UnitZ();
  /// Cylinder: the axis direction (normalised on use); in 2-D the cylinder
  /// is a circle in the model plane and the axis is z.
  Vector3 axis = Vector3::UnitZ();
  /// The path of the sphere's centre, of a point of the plane or of a point
  /// of the cylinder's axis.
  ToolTrajectory trajectory;
  /// The slave surface: the boundary faces whose nodes all lie in it.
  SelectorGroup surface;
  Scalar friction = 0.0;          ///< Coulomb coefficient mu >= 0 [-]
  Scalar penalty = 10.0;          ///< penalty scale s > 0 [-]
  Scalar tangential_ratio = 1.0;  ///< kappa_T / kappa > 0 [-]

  /// \throws ConfigError for a non-positive radius (sphere, cylinder), a
  ///         zero normal or axis, negative friction, a non-positive penalty
  ///         or tangential ratio, or an invalid trajectory.
  void validate(int dim) const;

  /// The gap of a point x from the tool with its reference point at c
  /// (negative when x penetrates), the unit normal n of the tool towards x,
  /// the distance d the normal's derivative divides by (0 for the plane) and
  /// dn = dn/dx. In 2-D the out-of-plane components are dropped.
  Scalar gap(const Vector3& x, const Vector3& c, int dim, Vector3& n, Matrix3& dn,
             Scalar& d) const;
};

std::string to_string(RigidTool::Shape shape);
/// "sphere", "plane" or "cylinder".
RigidTool::Shape parse_tool_shape(const std::string& text);

/// How the frictional contact tangent enters Newton's method.
enum class FrictionTangent {
  Exact,     ///< the consistent, non-symmetric tangent (LU)
  Symmetric  ///< a positive semi-definite friction part (a symmetric factorisation)
};
std::string to_string(FrictionTangent mode);
/// "exact" or "symmetric".
FrictionTangent parse_friction_tangent(const std::string& text);

/// The friction history of one slave node against one tool.
struct ToolNodeHistory {
  Vector3 force = Vector3::Zero();     ///< committed tangential force on the node [N]
  Vector3 relative = Vector3::Zero();  ///< committed x - c [m]
};
/// Per tool, the history of its slave nodes in contact, keyed by mesh node.
using ToolHistory = std::vector<std::map<Index, ToolNodeHistory>>;

/// One tool's resultant at an evaluated state.
struct ToolResultant {
  Vector3 centre = Vector3::Zero();  ///< the tool's reference point c(t) [m]
  /// The force the body exerts on the tool, minus the sum of the contact
  /// forces on the nodes [N].
  Vector3 force = Vector3::Zero();
  int active_nodes = 0;          ///< slave nodes with g < 0
  int slipping_nodes = 0;        ///< of those, the ones on the slip cone
  Scalar max_penetration = 0.0;  ///< max -g over them [m]
};

/// The contact contribution at a state.
struct ToolContactEvaluation {
  /// Full length: the contact part of R = f_int - f_ext, i.e. minus the
  /// contact forces on the nodes [N].
  Vector residual;
  /// Per node in contact, d(residual)/du of the node's own components
  /// (dim x dim used): the contact tangent is node-diagonal.
  struct NodeBlock {
    Index node = 0;
    Matrix3 k = Matrix3::Zero();
  };
  std::vector<NodeBlock> tangent;
  std::vector<ToolResultant> tools;  ///< one per tool (zero for an inactive one)
  ToolHistory trial;                 ///< the friction history at this state
  Scalar gross = 0.0;      ///< norm over the DOFs of |contact force| [N]
  /// The round-off scale of the contact forces: the norm over the DOFs of
  /// kappa A (|x| + |c|), the penalty times the positions whose difference
  /// the gap is; a gap is exact only to eps times that [N].
  Scalar round_off = 0.0;
  Scalar force_norm = 0.0;           ///< norm of the contact force vector [N]
  /// The tangent blocks are symmetric (no friction in slip or stick, or the
  /// symmetric friction tangent).
  bool symmetric = true;
};

/// Penalty contact of a set of rigid tools with the model, in the current
/// configuration. Built once from the reference mesh.
class ToolContact {
 public:
  /// \throws ConfigError for an invalid tool, a surface that selects no
  ///         boundary face, or duplicate tool names.
  ToolContact(const FemModel& model, std::vector<RigidTool> tools,
              FrictionTangent friction_tangent = FrictionTangent::Exact);

  /// Index of the tool called `name`, or -1.
  int find(const std::string& name) const;
  /// The slave nodes of tool k (ascending), their tributary areas A_j [m^2]
  /// and penalty stiffnesses kappa_j [Pa/m].
  const std::vector<Index>& slave_nodes(std::size_t k) const { return slaves_[k].nodes; }
  const std::vector<Scalar>& areas(std::size_t k) const { return slaves_[k].area; }
  /// The smallest characteristic size h_j of any slave node [m].
  Scalar min_node_size() const { return min_size_; }

  /// Prepare an increment from the converged state u0 at t0 to t1 with the
  /// tools flagged in `active` (one flag per tool): the candidate nodes of
  /// every active tool (see the file comment) and the start positions the
  /// slip of nodes without history is measured from. `margin` bounds how
  /// far a node may move in the increment [m]; an evaluation beyond it
  /// falls back to all slave nodes.
  void begin_increment(const Vector& u0, Scalar t0, Scalar t1, const std::vector<char>& active,
                       Scalar margin);

  /// The contact forces and tangent at the state (u, t) of the current
  /// increment, from the committed history. Pure: nothing is stored.
  /// \throws SolverError when a node lies within half the radius of a
  ///         sphere's centre or a cylinder's axis - an increment too long,
  ///         which the driver cuts.
  ToolContactEvaluation evaluate(const Vector& u, Scalar t, bool want_tangent) const;

  /// The candidate nodes that are open at u (g > 0) but would penetrate a
  /// tool after the Newton correction du (full length): the active set a
  /// semismooth Newton step predicts. For each, the penalty law extended
  /// linearly to its current gap - the force kappa A g n (pulling, as g > 0)
  /// added to `residual` and the normal stiffness kappa A n n^T as a node
  /// block - so that a re-solved correction lands it on the tool rather than
  /// deep inside it. Returns the number of nodes added.
  int anticipate(const Vector& u, const Vector& du, Scalar t, Vector& residual,
                 std::vector<ToolContactEvaluation::NodeBlock>& blocks) const;

  /// Make the friction history of a converged evaluation the committed one
  /// (nodes out of contact lose theirs).
  void commit(ToolContactEvaluation& ev);

  const ToolHistory& history() const { return history_; }
  /// Replace the committed history (a restart).
  /// \throws ConfigError when the tool count differs or a node is not a
  ///         slave node of its tool.
  void set_history(ToolHistory history);
  /// Forget the history of every tool not flagged in `active`.
  void clear_inactive(const std::vector<char>& active);

 private:
  struct Slaves {
    std::vector<Index> nodes;
    std::vector<Scalar> area;
    std::vector<Scalar> kappa;
    std::vector<Index> candidates;  ///< positions in `nodes` for this increment
  };
  const FemModel& model_;
  std::vector<RigidTool> tools_;
  FrictionTangent friction_tangent_;
  std::vector<Slaves> slaves_;
  Scalar min_size_ = 0.0;
  ToolHistory history_;
  // The increment prepared by begin_increment.
  Vector start_;
  Scalar t0_ = 0.0;
  Scalar margin_ = 0.0;
  std::vector<char> active_;
};

}  // namespace sparlab
