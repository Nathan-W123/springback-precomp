/// \file Forming.hpp
/// \brief Incremental-forming analysis: a sequence of quasi-static steps in
///        which rigid tools travel over the model along their trajectories
///        (`form`) and the model is then released from its tools and clamps
///        onto statically determinate supports to spring back (`release`).
///
/// **State.** The analysis carries one state from step to step: the
/// displacement u from the reference configuration, the internal variables
/// of every elastoplastic integration point, the friction history of every
/// tool's contact nodes (RigidTool.hpp) and the pseudo-time t. A step starts
/// where the previous one ended (or from an `AnalysisState` given to `run`,
/// a restart) and commits its history only at converged increments, so a
/// cut increment leaves no trace. The model's elements, kinematics
/// (`finite`, the total Lagrangian formulation, `finite_logarithmic` or
/// `small_strain`, Elastoplastic.hpp) and plasticity are evaluated by the
/// library's `NonlinearSystem`: internal forces, tangent, the pure-evaluate /
/// commit-on-convergence history and the mean dilatation of Hex8. No load
/// acts: the tools and the prescribed displacements drive the model.
///
/// **Constraints per step.** Each step has its own Dirichlet partition, a
/// list of displacement constraints (region, fixed components) with a mode:
/// `hold` keeps a constrained DOF where the step found it; `absolute` moves
/// it linearly over the step from there to the given value. A step that
/// lists no constraint takes the model's own boundary conditions, in
/// `absolute` mode (so a clamp with value 0 holds the reference position).
/// Every step's constraints must suppress every rigid-body motion of the
/// model (checked up front, naming the step and the free motions).
///
/// **Equilibrium of a step.** With s in [0, 1] the fraction of the step,
/// the prescribed DOFs at their values of s and the free DOFs f,
/// \f[
///   R_f(u, t) - (1 - s)\,R_{0f} = 0 ,\qquad
///   R = f_{int}(u) - f_{c}(u, t) ,
/// \f]
/// \f$f_c\f$ the penalty contact forces of the step's active tools at
/// pseudo-time \f$t(s)\f$ and \f$R_0 = R(u_0, t_0)\f$ the imbalance of the
/// start state on the step's free DOFs. For a step that keeps the partition
/// and the tools of the previous one, \f$R_{0f}\f$ is the previous step's
/// converged residual, within tolerance of zero; after a change it holds
/// the reactions of released constraints and the forces of removed tools,
/// which the step thereby ramps out linearly - the springback of a release.
/// A tool active in two consecutive steps must start the second where it
/// ended the first (refused otherwise: a gap between their windows over
/// which its path moves would make it jump, and the ramp would hide the
/// penetration). A tool that becomes active in contact has the force of its
/// penetration ramped in over the step, which is warned about.
///
/// **Increments.** A `form` step advances the pseudo-time over its window
/// (by default the union of its active tools' trajectories, starting no
/// earlier than the previous step ended) in increments such that no active
/// tool travels more than `max_tool_travel` (default half the smallest
/// slave-node size) in one; trajectory knots are reached exactly, so no
/// corner of a path is cut. A `release` step runs s from 0 to 1 in
/// `increments` equal parts (default 10). An increment that fails is halved
/// (up to `max_cuts` times in a row); one that converges in at most 4
/// iterations lets the next grow by 1.5, up to the step's cap.
///
/// **Newton's method** in each increment starts from the converged state
/// with the prescribed DOFs at their new values and the tools at their new
/// positions - when the prescribed DOFs move as a rigid motion of the body
/// (always so for statically determinate supports), the free DOFs move by
/// it too, linearised about the current configuration, rather than leave
/// the elements around the moved support nodes distorted - and takes the
/// consistent tangent (elements, contact and, with friction, its
/// non-symmetric part) with a line search on the energy - the
/// non-linear static analysis's: full step when \f$|g(1)| \le 0.8|g(0)|\f$,
/// else regula falsi, an element inversion or a node through a tool halving
/// the step; the full step's evaluation, tangent included, is kept as the
/// next iteration's. A node the correction would drive into a tool from
/// outside - without penalty stiffness in the tangent, it would overshoot
/// deep into the tool - joins the active set with the penalty extended to
/// its gap and the correction is solved again (a semismooth Newton step on
/// the predicted active set). It converges when both
/// \f$\|R_f - (1-s)R_{0f}\| \le \epsilon_R F_{ref}\f$ and the correction is
/// below \f$\epsilon_u\f$ times the increment of u - or the residual is at
/// its round-off floor, which includes the contact's (a gap subtracts
/// positions: a displacement is resolved only to the rounding of X + u) -
/// and the residual of the accepted state is checked again before its
/// history is committed. The reference force
/// \f$F_{ref}\f$ is the largest of the ramped imbalance, the reactions,
/// the contact forces and every such value met at a converged state of the
/// analysis so far (never zero): at the end of a release the external
/// forces and the reactions of determinate supports vanish, and a scale
/// made of them alone would leave the tolerance at zero.
///
/// **Linear algebra.** The free-free tangent's sparsity pattern is analysed
/// once per Dirichlet partition; each iteration scatters the values into it
/// through a cached index map and refactorises numerically: supernodal
/// Cholesky (CHOLMOD) for a symmetric positive definite tangent when
/// SuiteSparse is built in, else (or if it fails) \f$LDL^T\f$; LU (UMFPACK,
/// else Eigen's SparseLU) for a non-symmetric one or when \f$LDL^T\f$ meets
/// a vanishing pivot.
///
/// **Assumptions and limits.** Quasi-static (no inertia), isothermal; the
/// penalty leaves a penetration of about \f$p\,h/(sE)\f$; the elastoplastic
/// law is the additive split of the Green-Lagrange strain, meant for small
/// elastic strains (Elastoplastic.hpp) - plastic strains of order one in
/// forming are beyond its accuracy for the stress, not for equilibrium; the
/// mesh is never refined, trimmed or remapped.
///
/// Where this fits: Simulations stage - the analysis sparlab_form runs on a forming deck:
///   form steps drive the tools over the sheet, release steps let it spring back.
/// Called from: apps/sparlab_form.cpp:main, io/Config.cpp:parse_configuration (parse_step_type),
///   io/FormingWriter.cpp:forming_summary_json (forming_suitesparse_available)
/// Calls into: fem/NonlinearSystem, fem/RigidTool, fem/Elastoplastic
/// Start reading at: FormingAnalysis::run(const AnalysisState&) (fem/Forming.cpp)
#pragma once

#include "sparlab/core/Timer.hpp"
#include "sparlab/core/Types.hpp"
#include "sparlab/fem/Assembler.hpp"
#include "sparlab/fem/BoundaryConditions.hpp"
#include "sparlab/fem/Elastoplastic.hpp"
#include "sparlab/fem/FemModel.hpp"
#include "sparlab/fem/RigidTool.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace sparlab {

/// A displacement constraint of a step, and how its values evolve.
struct StepConstraint {
  enum class Mode {
    Hold,     ///< the DOF stays where the step found it
    Absolute  ///< linearly from where the step found it to `constraint.value_*`
  };
  std::string name;
  DisplacementConstraint constraint;  ///< region, fixed components, absolute values [m]
  Mode mode = Mode::Hold;
};
/// "hold" or "absolute".
StepConstraint::Mode parse_constraint_mode(const std::string& text);

/// One step of a forming analysis.
struct FormingStep {
  enum class Type {
    Form,    ///< the listed tools follow their trajectories over the window
    Release  ///< the start imbalance is ramped out; tools usually removed
  };
  std::string name = "step";
  Type type = Type::Form;
  /// The tools active in the step, by name. A release step may keep a
  /// subset of the previous step's tools.
  std::vector<std::string> tools;
  /// The pseudo-time window [t_begin, t_end] [s]. Unset (both NaN): a form
  /// step takes the union of its tools' trajectory spans, starting no
  /// earlier than the previous step ended; a release step (or a form step
  /// without tools) the unit interval after the previous step. A window may
  /// start after the previous step ended, but a tool active in both steps
  /// must then be where it was (its trajectory constant over the gap):
  /// otherwise it would jump over the part of its path between.
  Scalar t_begin = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar t_end = std::numeric_limits<Scalar>::quiet_NaN();
  /// The longest travel of an active tool in one increment [m]; 0: half the
  /// smallest characteristic size of the slave nodes.
  Scalar max_tool_travel = 0.0;
  /// The step's increments at the start (and the most it grows back to):
  /// the window in equal parts. 0: 1 for a form step (its tools' travel caps
  /// it), 10 for a release step.
  int increments = 0;
  /// The step's own constraints; empty: the model's boundary conditions, in
  /// `absolute` mode.
  std::vector<StepConstraint> constraints;
};
std::string to_string(FormingStep::Type type);
/// "form" or "release".
FormingStep::Type parse_step_type(const std::string& text);

struct FormingOptions {
  std::vector<RigidTool> tools;
  std::vector<FormingStep> steps;
  Kinematics kinematics = Kinematics::Finite;
  MeanDilatation mean_dilatation = MeanDilatation::Auto;
  FrictionTangent friction_tangent = FrictionTangent::Exact;
  int max_iterations = 30;             ///< Newton iterations per increment
  Scalar residual_tolerance = 1.0e-6;  ///< relative to the reference force
  Scalar displacement_tolerance = 1.0e-6;
  bool line_search = true;
  int max_cuts = 10;                   ///< successive halvings of an increment
  int max_increments = 100000;         ///< per step, converged increments
  /// Use SuiteSparse (CHOLMOD, UMFPACK) when the library was built with it;
  /// false: Eigen's own factorisations.
  bool suitesparse = true;
  /// Record the displacement every `snapshot_stride` converged increments
  /// (0: at step ends only).
  int snapshot_stride = 0;
};

/// The complete state an analysis step starts from and ends at.
struct AnalysisState {
  Vector displacement;                                ///< full, from the reference [m]
  std::vector<std::vector<PlasticState>> plastic;     ///< [element][point]; empty: elastic
  /// [element]: the committed parameters of the incompatible modes [m]
  /// (IncompatibleModes.hpp); empty for the standard element formulation.
  std::vector<Vector> internal;
  ToolHistory friction;                               ///< per tool, keyed by node
  Scalar time = 0.0;                                  ///< pseudo-time [s]
  /// The last converged full residual f_int - f_c (the reactions at the
  /// prescribed DOFs, the ramped imbalance at the free ones) [N].
  Vector residual;
  /// The running largest reference force of the analysis [N] (0 at the
  /// start of an analysis).
  Scalar reference_force = 0.0;
  /// Per tool, 1 when it was active in the step that produced this state
  /// (empty: none was). A restart checks against it that a tool active in
  /// its first step does not jump (FormingStep::t_begin).
  std::vector<char> tools_active;
};

/// One active tool at a converged increment.
struct ToolRecord {
  std::size_t tool = 0;              ///< index into FormingOptions::tools
  Vector3 centre = Vector3::Zero();  ///< reference point c(t) [m]
  Vector3 force = Vector3::Zero();   ///< exerted by the body on the tool [N]
  int active_nodes = 0;
  int slipping_nodes = 0;
  Scalar max_penetration = 0.0;      ///< [m]
};

/// One converged increment.
struct FormingIncrement {
  int index = 0;              ///< 1-based within the step
  Scalar time = 0.0;          ///< pseudo-time at its end [s]
  int iterations = 0;
  Scalar residual = 0.0;      ///< relative residual at convergence
  std::vector<ToolRecord> tools;
  Scalar max_plastic_strain = 0.0;  ///< committed [-]
};

/// A displacement recorded at an increment.
struct FormingSnapshot {
  int increment = 0;
  Scalar time = 0.0;
  Vector displacement;
};

struct FormingStepResult {
  std::string name;
  FormingStep::Type type = FormingStep::Type::Form;
  bool completed = false;
  std::string termination;
  Scalar t_begin = 0.0;
  Scalar t_end = 0.0;
  std::vector<std::string> tools;  ///< active tool names
  std::vector<FormingIncrement> increments;
  int iterations = 0;
  int cuts = 0;
  /// The end state: displacement [m], full residual [N] (reactions at the
  /// prescribed DOFs), and their norms.
  Vector displacement;
  Vector reactions;
  Scalar reaction_norm = 0.0;       ///< [N]
  Scalar reference_force = 0.0;     ///< F_ref at the end [N]
  Scalar start_imbalance = 0.0;     ///< |R_0f|, the imbalance ramped out [N]
  Scalar max_plastic_strain = 0.0;
  /// Per element at the end: the largest equivalent plastic strain of its
  /// points and the von Mises stress of its point-averaged Cauchy stress.
  Vector element_plastic_strain;
  Vector element_von_mises;
  /// The largest |u_end - u_start| of a node over the step [m] (the
  /// springback of a release).
  Scalar max_displacement_change = 0.0;
  int constrained_dofs = 0;
  std::vector<std::string> warnings;
  std::vector<FormingSnapshot> snapshots;
};

struct FormingResult {
  bool completed = false;
  std::string termination;
  std::vector<FormingStepResult> steps;
  AnalysisState final_state;  ///< the last converged state
  int total_increments = 0;
  int total_iterations = 0;
  int total_cuts = 0;
  bool plastic = false;
  bool mean_dilatation = false;
  std::string kinematics;
  std::string linear_solver;  ///< the factorisations used, and how often
  /// The factorisation attempts that failed, and how often (a Cholesky of a
  /// tangent that is not positive definite, followed by LDL^T; an LDL^T with
  /// a vanishing pivot, followed by LU; a singular LU, which fails the
  /// iteration); empty if none.
  std::string failed_factorisations;
  /// Wall-clock seconds by phase: "element_tangent" (element loop and
  /// assembly with the tangent), "element_residual" (without it: line search
  /// and acceptance), "contact", "factorisation" (scatter into the
  /// partition's pattern, analysis, numeric factorisation), "solve",
  /// "output" (the step observer), "total".
  TimingLedger timing;
  std::vector<std::string> warnings;
};

/// True when the library was built with SuiteSparse (CHOLMOD and UMFPACK),
/// which the forming analysis then uses by default.
bool forming_suitesparse_available();

/// The forming analysis of a finalised model (continuum elements).
class FormingAnalysis {
 public:
  /// \throws ConfigError for invalid options: an elastic material under finite
  ///         or logarithmic kinematics (sparlab_form models elastoplastic
  ///         sheets: require_elastoplastic_materials), no step, an unknown tool
  ///         name, a release step keeping a tool the step before it did not
  ///         have active, a time window that is empty or starts before the
  ///         previous step ended, a tool active in two consecutive steps that
  ///         would jump between them, constraints that leave a rigid-body
  ///         motion free, or a tool whose surface selects no face.
  FormingAnalysis(const FemModel& model, const Assembler& assembler, FormingOptions options);

  /// Run every step from the reference state (u = 0, virgin history).
  FormingResult run();
  /// Run every step from `start` (a restart).
  /// \throws ConfigError when `start` does not match the model, or a tool
  ///         active in the step that produced it and in the first step
  ///         would jump between them.
  FormingResult run(const AnalysisState& start);

  /// Called after each step ends - completed or stopped - with the result
  /// so far, whose last step is the one that ended (its time is counted as
  /// "output" in the timing). sparlab_form writes that step's files there,
  /// so a long run that is killed keeps the steps it completed.
  using StepObserver = std::function<void(const FormingResult& so_far)>;
  void set_step_observer(StepObserver observer) { observer_ = std::move(observer); }

 private:
  /// The steps' windows when the step before the first ended at
  /// `previous_end` (-inf: nothing before it).
  std::vector<std::pair<Scalar, Scalar>> resolve_windows(Scalar previous_end) const;
  /// Refuse a tool active in two consecutive steps whose position at the
  /// start of the second differs from its position at the end of the first
  /// by more than `tolerance` [m]; `previous` are the tools active before
  /// the first step, which ended at `previous_end`.
  void check_tool_continuity(const std::vector<std::pair<Scalar, Scalar>>& windows,
                             std::vector<char> previous, Scalar previous_end,
                             Scalar tolerance) const;

  const FemModel& model_;
  const Assembler& assembler_;
  FormingOptions options_;
  std::vector<std::vector<char>> active_;  ///< per step, per tool
  std::vector<std::pair<Scalar, Scalar>> windows_;
  StepObserver observer_;
};

}  // namespace sparlab
