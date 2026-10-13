/// \file NonlinearSystem.hpp
/// \brief The non-linear system of a model - internal forces, tangent, the
///        committed history of the elastoplastic points - shared by the
///        forming analysis (Forming.cpp) and the explicit integrator
///        (ExplicitKernel.cpp). Internal to the library.
///
/// Where this fits: Simulations stage - internal forces, tangent and plastic history of the
///   whole model: what each Newton iteration of a forming step evaluates.
/// Called from: fem/Forming.cpp:FormingAnalysis::run (evaluate, commit),
///   fem/ExplicitKernel.cpp:ExplicitInternalForce::evaluate (the generic dispatch)
/// Calls into: fem/Elastoplastic, fem/Assembler (assemble_elementwise)
/// Start reading at: NonlinearSystem::evaluate, NonlinearSystem::commit
#pragma once

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/fem/Assembler.hpp"
#include "sparlab/fem/Elastoplastic.hpp"
#include "sparlab/fem/FemModel.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace sparlab {
namespace detail {

/// Everything the path-following needs from the model at a state (u, lambda).
struct Evaluation {
  Vector internal;   ///< f_int [N]
  Vector external;   ///< f_ext at (u, lambda) [N]
  Vector residual;   ///< f_int - f_ext
  Vector load_rate;  ///< q = -dR/dlambda
  SparseMatrix tangent;
  Scalar energy = 0.0;
  /// Norm of the gross assembly, sum_e |f_e| per DOF plus |f_ext| (and the
  /// thermal forces below): the scale of the round-off in the residual,
  /// below which no tolerance can be met.
  Scalar gross = 0.0;
  /// Norm of the gross thermal forces, sum_e |lambda df_e/dlambda| per DOF:
  /// the load a temperature change exerts, which a free expansion balances
  /// with no external force or reaction at all.
  Scalar thermal = 0.0;
  /// Elastoplastic elements: the internal variables of every point after the
  /// return from the committed ones (empty for an elastic element), and the
  /// number of points whose return was plastic.
  std::vector<std::vector<PlasticState>> states;
  int yielding_points = 0;
  /// Elements with internal modes (the incompatible-mode Hex8): their
  /// parameters at this state, found from the committed ones (empty when
  /// the model has none).
  std::vector<Vector> modes;
};

/// Norm over the free DOFs of |K| |u|, the tangent's absolute entries times
/// the absolute displacement: each stored displacement is exact only to its
/// relative rounding eps, which the stiffness turns into forces of
/// eps |K| |u|. In a slender structure under large rotation this dominates:
/// an elastica's residual stagnated at 0.3 eps times it (1.3e-5 N), 180
/// times above the gross element forces' share.
inline Scalar stiffness_gross(const SparseMatrix& k, const Vector& u, const std::vector<Index>& free) {
  Vector product = Vector::Zero(k.rows());
  for (Eigen::Index col = 0; col < k.outerSize(); ++col) {
    const Scalar uc = std::abs(u(col));
    if (uc == 0.0) continue;
    for (SparseMatrix::InnerIterator it(k, col); it; ++it) {
      product(it.row()) += std::abs(it.value()) * uc;
    }
  }
  Scalar sum = 0.0;
  for (Index d : free) sum += product(d) * product(d);
  return std::sqrt(sum);
}

/// Round-off floor of a residual: a multiple of the machine epsilon times the
/// gross size of the sums that form it, and of the forces the rounding of
/// the displacement itself causes. Newton's residual stagnates at about 150
/// epsilon times the gross element forces on the plane cantilever (the
/// element forces are themselves sums of stress times area), so 1024 epsilon
/// sits safely above that; and at 0.3 epsilon |K||u| on the elastica, so 64
/// epsilon sits above that - both far below any useful tolerance.
///
/// A floor belongs to a state that is resolved to working precision. One
/// above a millionth of the residual's scale (`scale`) means the state is
/// not: past a limit or a plastic collapse load the displacement runs away
/// and inflates |K||u| without bound, until the "floor" exceeds the load
/// itself and any residual would pass. Such a floor is not accepted in place
/// of the tolerance (the forming analysis's balance, Forming.cpp).
inline constexpr Scalar kFloorLimit = 1.0e-6;

class NonlinearSystem {
 public:
  /// A system under no applied load at all - no load case: prescribed
  /// displacements and forces added by the caller (the moving tools of the
  /// forming analysis) drive it. `evaluate` then returns f_ext = 0 and
  /// q = 0 whatever lambda. Every element is evaluated by the return of
  /// Elastoplastic.hpp; under finite kinematics every material must be
  /// elastoplastic.
  /// \throws ConfigError for an elastic material under finite kinematics
  ///         (require_elastoplastic_materials).
  NonlinearSystem(const FemModel& model, const Assembler& assembler,
                  const NonlinearOptions& options)
      : model_(model), assembler_(assembler), options_(options) {
    require_elastoplastic_materials(model, options.kinematics);
    dead_ = Vector::Zero(model.dofs().num_dofs());
    initialise_history();
  }

  bool symmetric() const { return symmetric_; }
  const Vector& dead() const { return dead_; }
  /// Some material is elastoplastic.
  bool plastic() const { return plastic_; }
  /// Element e averages its dilatation.
  bool averaged(Index e) const { return averaged_[static_cast<std::size_t>(e)] != 0; }
  /// Element e is elastoplastic.
  bool elastoplastic(Index e) const { return !committed_[static_cast<std::size_t>(e)].empty(); }
  /// The internal variables of element e's points at the last converged
  /// step (the virgin state for an elastic element).
  const std::vector<PlasticState>& committed(Index e) const {
    const std::vector<PlasticState>& c = committed_[static_cast<std::size_t>(e)];
    return c.empty() ? virgin_ : c;
  }
  /// Make the internal variables of a converged evaluation the committed
  /// ones: the points' states, and the parameters of the elements' internal
  /// modes (elastic elements have them too).
  void commit(Evaluation& ev) {
    if (!internal_.empty() && ev.modes.size() == internal_.size()) {
      internal_ = std::move(ev.modes);
      ev.modes.clear();
    }
    if (!plastic_) return;
    for (std::size_t e = 0; e < committed_.size(); ++e) {
      if (!committed_[e].empty()) committed_[e] = std::move(ev.states[e]);
    }
    ev.states.clear();
  }
  /// The committed parameters of element e's internal modes (null for an
  /// element without them).
  const Vector* internal(Index e) const {
    return internal_.empty() ? nullptr : &internal_[static_cast<std::size_t>(e)];
  }
  /// The committed internal-mode parameters of every element (empty for a
  /// model without internal modes): part of the history a restart carries.
  const std::vector<Vector>& internal_all() const { return internal_; }
  /// Replace them - a restart from `internal_all` of a system of the same
  /// model and element formulation.
  /// \throws ConfigError when the element count or an element's parameter
  ///         count differs.
  void set_internal(std::vector<Vector> parameters) {
    if (parameters.size() != internal_.size()) {
      throw ConfigError("the saved internal-mode history has " +
                        std::to_string(parameters.size()) + " element(s); the model keeps " +
                        std::to_string(internal_.size()) +
                        " (none unless its elements have incompatible modes)");
    }
    for (std::size_t e = 0; e < parameters.size(); ++e) {
      if (parameters[e].size() != internal_[e].size() || !parameters[e].allFinite()) {
        throw ConfigError("the saved internal-mode parameters of element " + std::to_string(e) +
                          " have " + std::to_string(parameters[e].size()) +
                          " entries or are not finite; the element has " +
                          std::to_string(internal_[e].size()));
      }
    }
    internal_ = std::move(parameters);
  }
  /// The committed internal variables of every element (empty for an
  /// elastic element): the history a restart carries over.
  const std::vector<std::vector<PlasticState>>& committed_all() const { return committed_; }
  /// Replace the committed internal variables - a restart from a state saved
  /// by `committed_all` of a system of the same model, kinematics and
  /// integration rule.
  /// \throws ConfigError when the element count differs, or when an element
  ///         has a different number of points than this system keeps for it
  ///         (none for an elastic element).
  void set_committed(std::vector<std::vector<PlasticState>> states) {
    if (states.size() != committed_.size()) {
      throw ConfigError("the saved plastic history has " + std::to_string(states.size()) +
                        " element(s); the model has " + std::to_string(committed_.size()));
    }
    for (std::size_t e = 0; e < states.size(); ++e) {
      if (states[e].size() != committed_[e].size()) {
        throw ConfigError("the saved plastic history of element " + std::to_string(e) +
                          " has " + std::to_string(states[e].size()) +
                          " point(s); the model keeps " + std::to_string(committed_[e].size()) +
                          " for it (none for an elastic element)");
      }
    }
    committed_ = std::move(states);
  }
  /// The largest accumulated plastic strain of the committed state.
  Scalar max_plastic_strain() const {
    Scalar top = 0.0;
    for (const std::vector<PlasticState>& element : committed_) {
      for (const PlasticState& p : element) top = std::max(top, p.equivalent_plastic_strain);
    }
    return top;
  }

  Evaluation evaluate(const Vector& u, Scalar lambda, bool want_tangent) const {
    const Mesh& mesh = model_.mesh();
    const int dim = mesh.dim();
    const int npe = mesh.nodes_per_elem();
    const int nd = dim * npe;
    const Index ne = mesh.num_elements();
    const Index n = model_.dofs().num_dofs();
    const Vector* temperature = nullptr;

    std::vector<Vector> f_el(static_cast<std::size_t>(ne));
    std::vector<Vector> rate_el(static_cast<std::size_t>(ne));
    std::vector<Matrix> k_el(want_tangent ? static_cast<std::size_t>(ne) : 0);
    std::vector<Scalar> energy_el(static_cast<std::size_t>(ne), 0.0);
    std::vector<std::vector<PlasticState>> states(plastic_ ? static_cast<std::size_t>(ne) : 0);
    std::vector<int> yielding(plastic_ ? static_cast<std::size_t>(ne) : 0, 0);
    std::vector<Vector> modes(internal_.size());
    std::string failure;
    // Element loop (OpenMP when built in): every element through the return.
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (Index e = 0; e < ne; ++e) {
      try {
        const Index* nodes = mesh.element_nodes(e);
        Vector ue(nd);
        for (int a = 0; a < npe; ++a) {
          for (int k = 0; k < dim; ++k) ue(dim * a + k) = u(nodes[a] * dim + k);
        }
        ElastoplasticElement ep = elastoplastic_element(
            model_, e, ue, committed(e), averaged(e), temperature, lambda, want_tangent,
            options_.kinematics, internal_of(e));
        if (elastoplastic(e)) {
          states[static_cast<std::size_t>(e)] = std::move(ep.states);
          yielding[static_cast<std::size_t>(e)] = ep.yielding_points;
        }
        if (!internal_.empty()) {
          modes[static_cast<std::size_t>(e)] = std::move(ep.internal);
        }
        f_el[static_cast<std::size_t>(e)] = std::move(ep.internal_force);
        rate_el[static_cast<std::size_t>(e)] = std::move(ep.thermal_force_rate);
        energy_el[static_cast<std::size_t>(e)] = ep.energy;
        if (want_tangent) k_el[static_cast<std::size_t>(e)] = std::move(ep.tangent);
      } catch (const std::exception& ex) {
#ifdef SPARLAB_HAVE_OPENMP
#pragma omp critical(sparlab_nonlinear_failure)
#endif
        if (failure.empty()) failure = ex.what();
      }
    }
    if (!failure.empty()) throw SolverError(failure);

    // Gather: element forces into global vectors, element tangents through the Assembler.
    Evaluation out;
    out.internal = Vector::Zero(n);
    out.load_rate = dead_;
    Vector gross = Vector::Zero(n);
    Vector thermal = Vector::Zero(n);  // sum_e |lambda thermal force rate|
    for (Index e = 0; e < ne; ++e) {
      const Index* nodes = mesh.element_nodes(e);
      const Vector& fe = f_el[static_cast<std::size_t>(e)];
      const Vector& re = rate_el[static_cast<std::size_t>(e)];
      for (int a = 0; a < npe; ++a) {
        for (int k = 0; k < dim; ++k) {
          const Index dof = nodes[a] * dim + k;
          out.internal(dof) += fe(dim * a + k);
          gross(dof) += std::abs(fe(dim * a + k));
          if (re.size() > 0) {
            out.load_rate(dof) -= re(dim * a + k);
            thermal(dof) += std::abs(lambda * re(dim * a + k));
          }
        }
      }
      out.energy += energy_el[static_cast<std::size_t>(e)];
      if (plastic_) out.yielding_points += yielding[static_cast<std::size_t>(e)];
    }
    out.states = std::move(states);
    out.modes = std::move(modes);
    out.external = lambda * dead_;
    out.residual = out.internal - out.external;
    out.gross = (gross + thermal + out.external.cwiseAbs()).norm();
    out.thermal = thermal.norm();
    if (want_tangent) {
      out.tangent = assembler_.assemble_elementwise(
          [&](Index e) -> Matrix { return k_el[static_cast<std::size_t>(e)]; });
    }
    return out;
  }

 private:
  /// Elastoplastic elements keep their points' internal variables, starting
  /// virgin; mean dilatation where it relaxes a constraint.
  void initialise_history() {
    const FemModel& model = model_;
    const Index ne = model.mesh().num_elements();
    const int points = elastoplastic_points(model);
    committed_.resize(static_cast<std::size_t>(ne));
    averaged_.assign(static_cast<std::size_t>(ne), 0);
    virgin_.assign(static_cast<std::size_t>(points), PlasticState());
    // Incompatible modes: every element keeps its parameters, starting at
    // zero; they relax the isochoric constraint themselves and take no mean
    // dilatation (IncompatibleModes.hpp).
    const int modes = model.element().num_internal_dofs();
    const bool incompatible = modes > 0;
    if (incompatible) {
      internal_.assign(static_cast<std::size_t>(ne), Vector::Zero(modes));
      if (options_.mean_dilatation == MeanDilatation::All) refuse_mean_dilatation();
    }
    for (Index e = 0; e < ne; ++e) {
      if (!model.material_of(e).plasticity().enabled()) continue;
      if (model.dim() == 2 && !model.material_of(e).plasticity().plane_compatible()) {
        throw ConfigError("material '" + model.material_of(e).name() +
                          "': in a plane model the out-of-plane axis z must be the rolling, "
                          "transverse or normal direction of its Hill48 frame");
      }
      plastic_ = true;
      // Armstrong-Frederick recovery makes the consistent tangent
      // non-symmetric (factorised by LU, without the inertia test).
      if (!model.material_of(e).plasticity().symmetric_tangent()) symmetric_ = false;
      committed_[static_cast<std::size_t>(e)].assign(static_cast<std::size_t>(points),
                                                     PlasticState());
      const ElementType type = model.mesh().element_type();
      const bool wanted =
          !incompatible &&
          (options_.mean_dilatation == MeanDilatation::All ||
           (options_.mean_dilatation == MeanDilatation::Auto &&
            type == ElementType::Hex8));
      averaged_[static_cast<std::size_t>(e)] =
          wanted && points > 1 && model.stress_state() != StressState::PlaneStress;
    }
  }

  /// The committed internal-mode parameters of element e, or null.
  const Vector* internal_of(Index e) const { return internal(e); }
  static void refuse_mean_dilatation() {
    throw ConfigError(
        "\"mean_dilatation\": \"all\" is not combined with the incompatible-mode Hex8: "
        "averaging the dilatation of its modes leaves a zero-energy mode, and averaging the "
        "compatible dilatation alone softens the bending of a sheet one element thick; the "
        "modes relax the isochoric constraint of plastic flow themselves. Use \"auto\" or "
        "\"none\"");
  }

  const FemModel& model_;
  const Assembler& assembler_;
  NonlinearOptions options_;  ///< by value: see the constructor
  bool plastic_ = false;
  std::vector<std::vector<PlasticState>> committed_;  ///< per element; empty if elastic
  std::vector<char> averaged_;                        ///< per element: mean dilatation
  std::vector<PlasticState> virgin_;                  ///< one per point, for elastic elements
  std::vector<Vector> internal_;  ///< per element: committed internal-mode parameters, or empty
  Vector dead_;
  bool symmetric_ = true;
};

inline Vector restrict(const Vector& full, const std::vector<Index>& dofs) {
  Vector out(static_cast<Eigen::Index>(dofs.size()));
  for (std::size_t i = 0; i < dofs.size(); ++i) out(static_cast<Eigen::Index>(i)) = full(dofs[i]);
  return out;
}

inline void add_to(Vector& full, const std::vector<Index>& dofs, const Vector& part, Scalar factor) {
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    full(dofs[i]) += factor * part(static_cast<Eigen::Index>(i));
  }
}

}  // namespace detail
}  // namespace sparlab
