/// \file ExplicitDynamics.hpp
/// \brief Explicit dynamics: the central-difference method with a lumped
///        mass, the stable time step, uniform and selective mass scaling,
///        mass-proportional damping, penalty contact with moving rigid
///        tools and an energy balance - the integrator of the explicit
///        forming step (`form_explicit`, Forming.hpp, docs/forming.md).
///
/// **Equations of motion.** With the lumped (diagonal) mass M, the
/// mass-proportional damping \f$C = \alpha M\f$, the internal forces
/// \f$f_{int}(u)\f$ of the non-linear system (the elements, kinematics and
/// plasticity of Elastoplastic.hpp, the plastic history of every point) and
/// the contact forces \f$f_c\f$ of the tools (no other load acts,
/// \f$f_{ext} = 0\f$),
/// \f[
///   M\ddot u + C\dot u + f_{int}(u) = f_{ext}(t) + f_c(u, \dot u, t)
/// \f]
/// on the free DOFs; a prescribed DOF follows
/// \f$u_p(t) = u_{p,0} + r(t)\,(u_{p,1} - u_{p,0})\f$ (r a ramp, linear in
/// the physical time).
///
/// **Central differences** (Belytschko, Liu and Moran, *Nonlinear Finite
/// Elements for Continua and Structures*, Wiley 2000, box 6.1), with the
/// velocities at the half steps and a step that may change between steps:
/// \f[
///   v_{n+1/2} = v_n + (t_{n+1/2} - t_n)\,a_n ,\quad
///   u_{n+1} = u_n + \Delta t_{n+1/2}\,v_{n+1/2} ,\quad
///   a_{n+1} = M^{-1}\big(f_{ext} - f_{int}(u_{n+1}) + f_c - \alpha M v_{n+1/2}\big) ,\quad
///   v_{n+1} = v_{n+1/2} + (t_{n+1} - t_{n+1/2})\,a_{n+1} .
/// \f]
/// The damping takes the half-step velocity (lagged by half a step, so the
/// update stays explicit); a prescribed DOF is set to \f$u_p(t_{n+1})\f$
/// directly (no drift) with \f$v_{n+1/2} = (u_p(t_{n+1}) - u_p(t_n))/\Delta t\f$.
/// Every step is final: the plastic history is updated in place at every
/// step, never iterated.
///
/// **Lumped mass**: the row sums of the consistent element masses,
/// Assembler::assemble_mass. **Mass scaling** multiplies an
/// element's density by \f$s_e \ge 1\f$: uniform,
/// \f$s_e = (\Delta t_{target}/\Delta t_{crit})^2\f$ for every element;
/// selective (conventional, per element),
/// \f$s_e = \max\big(1, (\Delta t_{target}/\Delta t_e)^2\big)\f$, which adds
/// mass only to the elements that limit the step (the added mass is
/// reported, and warned about above a fraction); with `dynamic` selective
/// scaling the scales are raised at every stable-step update where an
/// element's step has fallen below the target (as a sheet thins under the
/// tool), keeping the velocities and booking the added kinetic energy
/// \f$W_m = \tfrac12\sum\Delta m\,v^2\f$ in the balance. Mass scaling and a faster
/// tool (time scaling) raise the inertia forces alike - by \f$s\f$ and by
/// \f$v^2\f$ - so the measure of a run's dynamics is the equivalent speed
/// \f$v\sqrt{s}\f$, and its cost (the number of steps) is inversely
/// proportional to it. (Selective mass scaling proper, a non-diagonal mass
/// that leaves the rigid-body inertia intact - Olovsson, Simonsson and
/// Unosson, IJNME 63 (2005) 1436-1445 - needs an iterative mass solve and is
/// not provided.) Body loads keep the physical density.
///
/// **Stable time step.** The central-difference method is stable for
/// \f$\omega_{max}\Delta t \le 2\f$. `ElementEigenvalue` (default) bounds
/// \f$\omega_{max}\f$ by the largest element eigenvalue,
/// \f$\omega_e^2 = \lambda_{max}(M_e^{-1/2}K_eM_e^{-1/2})\f$ with the linear
/// elastic element stiffness at the current configuration and the element's
/// lumped (scaled) mass - by Irons' theorem an upper bound of the assembled
/// system's; `PowerIteration` estimates \f$\omega_{max}\f$ of the assembled
/// \f$M^{-1}K\f$ without constraints (Rayleigh quotient, times 1.05; the
/// constraints can only lower it) - tighter, for the whole model only; `ElementLength` takes \f$L_e/c_e\f$ with
/// \f$L_e = V_e/A_{max}\f$ and the dilatational wave speed
/// \f$c_e = \sqrt{(\lambda + 2\mu)/(s_e\rho)}\f$ - the conventional estimate,
/// not a bound. With finite kinematics the estimate is repeated every
/// `update_every` steps on the current configuration (a thinning sheet
/// lowers it), the step never growing by more than 1 % at an update. The
/// damping and the contact penalty lower the limit to
/// \f$\omega^2\Delta t^2 + 2\alpha\Delta t \le 4 - s_c\f$ (below); the step
/// is `safety` times that limit.
///
/// **Contact** with rigid tools (sphere, plane, cylinder: the geometry and
/// trajectories of RigidTool.hpp) in the current configuration, per slave
/// node j and tool: where the gap \f$g < 0\f$, the normal force
/// \f$f_N = -k_j g\,n\f$ with the mass-based penalty
/// \f$k_j = s_c m_j/\Delta t^2\f$ (\f$m_j\f$ the node's lumped mass, \f$s_c\f$
/// `contact_stiffness`, default 0.5 - the "soft constraint" of explicit
/// codes, LS-DYNA theory manual): it adds at most \f$s_c/\Delta t^2\f$ to
/// \f$\omega^2\f$, so it never lowers the stable step by more than that,
/// unlike the implicit law \f$\kappa = sE/h\f$, which is stiffer still and
/// would. Its penetration, about \f$f\,\Delta t^2/(s_c m_j)\f$, does not
/// shrink with the tool speed (with mass scaling, \f$m_j/\Delta t^2\f$ is
/// fixed by the target step): it biases the formed shape by about the
/// penetration, so the largest penetration relative to the node's element
/// thickness is reported and warned about (`penetration_warning`).
/// Friction: regularised Coulomb friction with a committed
/// tangential force per node, transported onto the current tangent plane,
/// incremented by \f$-k_T P\,\Delta_{rel}\f$ (\f$k_T = \rho_T k_j\f$,
/// \f$\Delta_{rel}\f$ the node's motion relative to the tool over the step)
/// and returned onto the cone \f$|F_T| \le \mu|f_N|\f$; the stable step
/// reserves \f$s_c\max(1, \rho_T)\f$ (the stiffer of the two directions)
/// for a frictional tool. The tool's
/// reference point follows its trajectory at the pseudo-time the drive maps
/// the physical time to (ExplicitTimeMap).
///
/// **Energy balance.** Kinetic energy \f$T = \tfrac12 v^TMv\f$ (at the
/// integer steps), the work of the internal forces W_int, of the loads and
/// the reactions of the prescribed DOFs W_ext and of the contact forces
/// (normal and friction parts, on the body) W_c - trapezoidal sums
/// \f$\Delta u\cdot(f_n + f_{n+1})/2\f$ - and the damping dissipation
/// \f$D = \sum\Delta u\cdot\alpha M v_{n+1/2}\f$:
/// \f$E_{err} = T - T_0 + W_{int} + D - W_{ext} - W_c - W_m\f$, relative to the
/// largest energy of the run so far. The plastic dissipation is W_int less
/// the change of the stored (elastic and hardening) energy. Quasi-static
/// validity is judged by the ratio of T to an energy of the state after
/// the first contact and by the balance (both warned about): from a virgin
/// start, the internal energy \f$W_{int}\f$; for a step that continues a
/// stressed part (a retract), the elastic strain energy of the state (the
/// dedicated kernel; the generic dispatch gives only the stored energy,
/// elastic plus hardening, and falls back to it). The warned ratio skips the
/// records whose energy is below 1 % of its final value; the unfiltered
/// peak - the plunge into a flat sheet - is reported beside it. A balance
/// beyond `energy_limit` stops the run, as do non-finite values and an
/// inverted element.
///
/// **Internal forces.** A dedicated kernel for Hex8 (finite, logarithmic -
/// elastoplastic elements only - or small-strain kinematics; elastic, or
/// elastoplastic through the return of
/// Plasticity.hpp; mean dilatation) keeps the reference gradients and
/// weights of every point (one set on a uniform structured mesh), updates
/// the history in place and assembles by a gather over each node's
/// elements in element order - the same order the serial assembly of
/// NonlinearSystem sums them in, so the result is the same for any thread
/// count, bitwise. It is checked against the generic element dispatch at
/// the start of every run (a relative difference above 1e-10 falls back
/// to the generic path). Every other element, kinematics, material law or
/// load uses the generic dispatch, `NonlinearSystem::evaluate` - the single
/// entry point of the element technologies. Neither path allocates in the
/// dedicated kernel's loop.
///
/// **Units.** Strict SI: m, s (physical time; the tool trajectories are in
/// their own pseudo-time), kg, N, J.
///
/// Where this fits: Simulations stage - the integrator of a form_explicit step; the decks of
///   python/precomp/fea/deck.py use implicit form and release steps only.
/// Called from: fem/Forming.cpp:FormingAnalysis::run, io/Config.cpp:parse_explicit_step
/// Calls into: fem/ExplicitKernel (internal forces), fem/RigidTool, fem/Assembler (lumped mass)
/// Start reading at: ExplicitDynamics::run (fem/ExplicitDynamics.cpp)
#pragma once

#include "sparlab/core/Timer.hpp"
#include "sparlab/core/Types.hpp"
#include "sparlab/fem/Assembler.hpp"
#include "sparlab/fem/Elastoplastic.hpp"
#include "sparlab/fem/FemModel.hpp"
#include "sparlab/fem/RigidTool.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <memory>
#include <string>
#include <vector>

namespace sparlab {

/// How the stable time step is estimated.
struct StableStepOptions {
  enum class Method {
    ElementEigenvalue,  ///< max_e omega_e (Irons' bound), default
    ElementLength,      ///< min_e L_e / c_e (conventional, not a bound)
    PowerIteration      ///< omega_max of the assembled M^-1 K, times 1.05
  };
  Method method = Method::ElementEigenvalue;
  /// The step is this fraction of the stability limit, in (0, 1].
  Scalar safety = 0.9;
  /// Finite kinematics: re-estimate every this many steps on the current
  /// configuration (0: never). Ignored with small-strain kinematics.
  int update_every = 1000;
  /// PowerIteration: the iterations (from a fixed pseudo-random start).
  int power_iterations = 60;
};
std::string to_string(StableStepOptions::Method method);
/// "element_eigenvalue", "element_length" or "power_iteration".
StableStepOptions::Method parse_stable_step_method(const std::string& text);

/// Mass scaling to a target time step.
struct MassScalingOptions {
  enum class Mode {
    None,       ///< the physical mass
    Uniform,    ///< one factor for every element
    Selective   ///< per element, only where the element limits the step
  };
  Mode mode = Mode::None;
  Scalar target_time_step = 0.0;          ///< [s]; required with a mode other than None
  Scalar max_added_mass_fraction = 0.05;  ///< added / physical mass above which a run warns
  /// Selective, finite kinematics: at every stable-step update, raise the
  /// scale of the elements whose step (thinned, distorted) has fallen below
  /// the target, so the step stays at the target (the velocities are kept;
  /// the added mass's kinetic energy is booked in the balance).
  bool dynamic = false;
};
std::string to_string(MassScalingOptions::Mode mode);
/// "none", "uniform" or "selective".
MassScalingOptions::Mode parse_mass_scaling_mode(const std::string& text);

/// Settings of an explicit run.
struct ExplicitOptions {
  StableStepOptions stable_step;
  MassScalingOptions mass_scaling;
  /// A fixed time step [s] instead of the stable one (0: the stable one).
  /// Beyond the stability limit it is warned about; the run then stops when
  /// its energy balance fails.
  Scalar time_step = 0.0;
  Scalar mass_damping = 0.0;       ///< alpha of C = alpha M [1/s], >= 0
  /// s_c of the contact penalty k_j = s_c m_j / dt^2, in (0, 1].
  Scalar contact_stiffness = 0.5;
  int history_every = 100;         ///< record every this many steps (and the last)
  int snapshot_every = 0;          ///< keep the displacement every this many steps (0: none)
  Scalar energy_tolerance = 0.05;  ///< relative energy balance error warned about
  Scalar energy_limit = 0.5;       ///< relative energy balance error that stops the run
  /// T over the internal energy after the first contact warned about.
  Scalar kinetic_ratio_warning = 0.1;
  /// The largest contact penetration over the element thickness at the
  /// node (volume over largest face of its thinnest element) warned about.
  Scalar penetration_warning = 0.01;
  /// Use the dedicated Hex8 kernel where it applies (false: the generic
  /// element dispatch everywhere, for comparison).
  bool dedicated_kernel = true;

  /// \throws ConfigError for a value out of range; `what` names the owner.
  void validate(const std::string& what = "explicit") const;
};

/// The tools' pseudo-time as a piecewise-linear, non-decreasing function of
/// the physical time tau in [0, duration].
struct ExplicitTimeMap {
  std::vector<Scalar> physical;  ///< [s], strictly increasing from 0
  std::vector<Scalar> pseudo;    ///< [s], non-decreasing

  Scalar duration() const { return physical.empty() ? 0.0 : physical.back(); }
  /// The pseudo-time at tau (clamped to the ends).
  Scalar pseudo_time(Scalar tau) const;

  /// The window [t0, t1] of pseudo-time over `duration` seconds, linearly.
  static ExplicitTimeMap linear(Scalar t0, Scalar t1, Scalar duration);
  /// The window [t0, t1] with every tool of `paths` travelling no faster
  /// than `speed` [m/s] and the fastest at exactly `speed`: each pseudo-time
  /// segment between the paths' knots lasts (largest tool travel on it) /
  /// speed; segments on which no tool moves take no time.
  /// \throws ConfigError when no tool moves in the window.
  static ExplicitTimeMap tool_speed(const std::vector<const ToolTrajectory*>& paths, Scalar t0,
                                    Scalar t1, Scalar speed);
};

/// The state an explicit run starts from and ends at.
struct ExplicitState {
  Vector displacement;  ///< full [m]
  Vector velocity;      ///< full [m/s]; empty: at rest
  /// The plastic history in NonlinearSystem's layout, [element][point]
  /// (empty for an elastic element); empty: virgin.
  std::vector<std::vector<PlasticState>> history;
  /// Per tool, the friction history of its nodes in contact (the forming
  /// analysis's layout: the tangential force and x - c); empty: none.
  ToolHistory friction;
};

/// What an explicit run integrates.
struct ExplicitDrive {
  Scalar duration = 0.0;          ///< physical time [s]
  std::vector<Index> fixed;       ///< prescribed DOFs, ascending
  Vector fixed_start;             ///< their values at tau = 0 [m]
  Vector fixed_end;               ///< their values at r = 1 [m]
  std::vector<char> active;       ///< per tool: in contact this run
  ExplicitTimeMap time_map;       ///< the tools' pseudo-time; empty: constant 0
};

/// One tool at a recorded step.
struct ExplicitToolRecord {
  std::size_t tool = 0;
  Vector3 centre = Vector3::Zero();      ///< reference point c(t) [m]
  /// The force the body exerts on the tool averaged over the steps since
  /// the last record (the decimation's box filter) [N].
  Vector3 force = Vector3::Zero();
  int active_nodes = 0;
  int slipping_nodes = 0;
  Scalar max_penetration = 0.0;          ///< [m]
  /// The largest penetration over the node's element thickness [-].
  Scalar max_penetration_ratio = 0.0;
};

/// A recorded step (every `history_every` steps, and the last).
struct ExplicitRecord {
  long step = 0;             ///< steps since the start of the run
  Scalar time = 0.0;         ///< physical [s]
  Scalar pseudo_time = 0.0;  ///< of the tools [s]
  Scalar time_step = 0.0;    ///< [s]
  Scalar kinetic = 0.0;      ///< T [J]
  Scalar internal = 0.0;     ///< W_int since the start [J]
  /// The internal energy of the state: the stored energy at the start plus
  /// W_int (from a virgin start, W_int) [J].
  Scalar internal_energy = 0.0;
  Scalar stored = 0.0;       ///< elastic + hardening energy of the state [J]
  /// The elastic strain energy of the state [J] (dedicated kernel; NaN with
  /// the generic element dispatch, which does not separate it).
  Scalar elastic = 0.0;
  Scalar plastic = 0.0;      ///< W_int - (stored - stored at the start) [J]
  Scalar contact_normal = 0.0;    ///< work of the normal contact forces on the body [J]
  Scalar contact_friction = 0.0;  ///< work of the friction forces on the body [J]
  Scalar damping = 0.0;      ///< D [J]
  /// Kinetic energy of the mass added by dynamic mass scaling [J].
  Scalar mass_scaling = 0.0;
  Scalar external = 0.0;     ///< W_ext: loads and reactions [J]
  Scalar error = 0.0;        ///< E_err [J]
  Scalar max_plastic_strain = 0.0;  ///< [-]
  std::vector<ExplicitToolRecord> tools;  ///< the active tools
};

struct ExplicitSnapshot {
  long step = 0;
  Scalar time = 0.0;         ///< physical [s]
  Scalar pseudo_time = 0.0;  ///< [s]
  Vector displacement;       ///< [m]
};

struct ExplicitResult {
  bool completed = false;
  std::string termination;
  long steps = 0;
  Scalar duration = 0.0;             ///< physical time integrated [s]
  Scalar time_step = 0.0;            ///< the first step [s]
  Scalar min_time_step = 0.0;        ///< [s]
  Scalar final_time_step = 0.0;      ///< [s]
  /// 2 / omega_max of the unscaled model at the start (the stability limit
  /// without mass scaling, damping, contact or safety) [s].
  Scalar stable_time_step = 0.0;
  /// The same with the scaled mass [s].
  Scalar scaled_stable_time_step = 0.0;
  int step_updates = 0;              ///< stable-step re-estimates
  int mass_updates = 0;              ///< updates that added mass (dynamic scaling)
  Scalar physical_mass = 0.0;        ///< [kg]
  /// The scaled mass [kg] and what follows from it - at the end of the run
  /// (with dynamic scaling; the start's otherwise, the same).
  Scalar scaled_mass = 0.0;
  Scalar added_mass_fraction = 0.0;  ///< (scaled - physical) / physical
  Scalar max_mass_scale = 1.0;       ///< largest s_e
  int scaled_elements = 0;           ///< elements with s_e > 1
  /// The largest T / reference energy (`kinetic_ratio_basis`) over the
  /// records after the first contact whose reference energy exceeds 1 % of
  /// its value at the end (0 without contact): the ratio warned about.
  Scalar max_kinetic_ratio = 0.0;
  /// The same over every record after the first contact (the plunge of a
  /// tool into a flat sheet included) [-].
  Scalar peak_kinetic_ratio = 0.0;
  /// What the kinetic energy is measured against: "internal energy" (a
  /// virgin start), "elastic energy" (a stressed start, dedicated kernel)
  /// or "stored energy" (a stressed start, generic dispatch).
  std::string kinetic_ratio_basis = "internal energy";
  /// The largest contact penetration over the node's element thickness,
  /// over every time step [-].
  Scalar max_penetration_ratio = 0.0;
  Scalar max_energy_error = 0.0;     ///< largest relative |E_err|
  bool contact = false;              ///< some tool touched the body
  std::vector<ExplicitRecord> records;
  std::vector<ExplicitSnapshot> snapshots;
  std::vector<std::string> warnings;
  /// The last state (the last valid one when the run stopped).
  ExplicitState final_state;
  Vector acceleration;  ///< full [m/s^2]
  /// f_int - f_ext - f_c at the end (its prescribed entries are the static
  /// reactions; the free ones the inertia and damping forces) [N].
  Vector residual;
  /// M a + C v + f_int - f_ext - f_c at the prescribed DOFs, 0 elsewhere [N].
  Vector reactions;
  std::string kernel;   ///< "dedicated Hex8" or "generic element dispatch"
  /// Wall-clock seconds: "internal_force", "contact", "stable_step",
  /// "integration" (the vector updates and energies), "total".
  TimingLedger timing;
};

/// Everything the element loop needs; private to the library.
class ExplicitInternalForce;

/// The explicit integrator of a finalised continuum model.
class ExplicitDynamics {
 public:
  /// `nonlinear` gives the kinematics, elastic law and mean dilatation.
  /// \throws ConfigError for invalid options, a model with a material
  ///         without density or with shells, or an invalid tool.
  ExplicitDynamics(const FemModel& model, const Assembler& assembler,
                   const NonlinearOptions& nonlinear, std::vector<RigidTool> tools,
                   ExplicitOptions options);
  ~ExplicitDynamics();
  ExplicitDynamics(const ExplicitDynamics&) = delete;
  ExplicitDynamics& operator=(const ExplicitDynamics&) = delete;

  /// Integrate `drive` from `start`. Never throws for a run that goes
  /// wrong: it stops with `completed` false, the reason in `termination`
  /// and the last valid state.
  /// \throws ConfigError for a drive or start state that does not match the
  ///         model or the tools.
  ExplicitResult run(const ExplicitDrive& drive, const ExplicitState& start);

  /// The internal forces at u from the history (empty: virgin) by the
  /// kernel the runs use (as at a step that records nothing: the dedicated
  /// kernel's elastic predictor where it applies), or by the generic element
  /// dispatch (`generic` true) - for the tests that compare them. The
  /// history is not changed.
  Vector internal_force(const Vector& u, const std::vector<std::vector<PlasticState>>& history,
                        bool generic) const;

  /// Per element, the stable time step 2/omega_e (or L_e/c_e) at the
  /// displacement u with the mass scale `scale` (null: 1) - no safety,
  /// damping or contact [s].
  Vector element_time_steps(const Vector& u, const Vector* scale = nullptr) const;
  /// 2 / omega_max of the assembled, unconstrained system at u by power
  /// iteration (the Rayleigh quotient times 1.05) [s].
  Scalar power_iteration_time_step(const Vector& u, const Vector* scale = nullptr) const;

  /// True when the dedicated Hex8 kernel applies to the model and options.
  bool dedicated() const;

 private:
  const FemModel& model_;
  const Assembler& assembler_;
  NonlinearOptions nonlinear_;
  std::vector<RigidTool> tools_;
  ExplicitOptions options_;
  std::unique_ptr<ExplicitInternalForce> force_;
};

}  // namespace sparlab
