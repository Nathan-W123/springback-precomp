/// \file ExplicitKernel.hpp
/// \brief The internal forces of the explicit integrator (ExplicitDynamics.hpp):
///        a dedicated Hex8 kernel and the generic element dispatch.
///        Internal to the library.
///
/// The dedicated kernel evaluates the same equations as the element
/// dispatch of NonlinearSystem (Elastoplastic.hpp for small-strain and for
/// elastoplastic elements, TotalLagrangian.hpp for elastic elements with
/// finite kinematics), in the form an explicit step needs:
/// * the reference gradients \f$G = \partial N/\partial X\f$ and weights
///   \f$w\det J\f$ of every point are computed once (one set on a uniform
///   structured mesh, whose cells are the same box);
/// * at a point, \f$F = I + UG^T\f$, the strain (Green-Lagrange, or the
///   linear one), the return of Plasticity.hpp from the point's history
///   (or \f$S = DE\f$ for an elastic element with finite kinematics), and
///   the nodal forces \f$f_a = w\,(FS)\,G_a\f$ (\f$w\,\sigma G_a\f$ with
///   small strain) - \f$B_{NL}^TS\f$ written as \f$FSG\f$;
/// * with mean dilatation the point's dilatation is replaced by the
///   element's average in the strain, and the force gains
///   \f$\tfrac w3\,\mathrm{tr}S\,(\bar d - d)\f$ with \f$d = FG\f$ the
///   variation of the dilatation (Elastoplastic.hpp);
/// * with logarithmic kinematics (elastoplastic elements) the strain is
///   \f$E_{\log}\f$ of every point's E (LogarithmicStrain.hpp), the return
///   gives T, and the force takes \f$S = P^TT\f$ (T's deviator with mean
///   dilatation, plus the mean pressure over the volume times
///   \f$C^{-1} = P^TI\f$, the variation of \f$\ln J\f$) in place of the
///   Green-Lagrange S - about 3.6 us an element and step on one thread (the
///   cone's first 1 518 steps; 4.0 us with the iterative eigensolver
///   throughout), against 30 us through the dispatch; the spectral
///   decomposition takes the closed-form 3 x 3 solver where its residual
///   shows it accurate;
/// * the history is updated in place (an explicit step is final), in a
///   compact form (plastic strain, back stress, equivalent plastic strain,
///   loading flag), one array per component over the element's eight
///   points, when no material needs the general return's backstresses;
/// * the eight points of an element are processed together, every array
///   running over them (and the force over the eight nodes), in plain loops
///   the compiler vectorises (with an x86-64-v3 clone - AVX2 and FMA - on
///   x86-64 GCC builds, picked by the processor at load time);
/// * the element forces are gathered per node over the node's elements in
///   ascending element order: NonlinearSystem's serial assembly sums them
///   in the same order, and the result does not depend on the thread count.
/// Nothing is allocated in the element loop.
#pragma once

#include "NonlinearSystem.hpp"

#include "sparlab/core/Types.hpp"
#include "sparlab/fem/Assembler.hpp"
#include "sparlab/fem/FemModel.hpp"
#include "sparlab/material/Plasticity.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace sparlab {

class ExplicitInternalForce {
 public:
  /// `allow_dedicated` false forces the generic dispatch.
  ExplicitInternalForce(const FemModel& model, const Assembler& assembler,
                        const NonlinearOptions& options, bool allow_dedicated);

  /// The dedicated Hex8 kernel is in use.
  bool dedicated() const { return dedicated_; }
  /// Why the generic dispatch is used (empty when the kernel is dedicated).
  const std::string& fallback_reason() const { return reason_; }

  /// Replace the history (NonlinearSystem's layout; empty: virgin).
  /// \throws ConfigError when it does not match the model.
  void set_history(const std::vector<std::vector<PlasticState>>& history);
  /// The current history in NonlinearSystem's layout.
  std::vector<std::vector<PlasticState>> history() const;

  /// f_int and f_ext at (u, lambda) and, with `want_energy`, the stored
  /// energy [J] (0 otherwise); with `commit` the history advances to the
  /// returns' states.
  /// \throws SolverError when a return fails or (dedicated, finite
  ///         kinematics) an element inverts.
  void evaluate(const Vector& u, Scalar lambda, bool commit, Vector& internal, Vector& external,
                Scalar& energy, bool want_energy = true);
  /// The elastic strain energy [J] of the last evaluation with
  /// `want_energy` (the dedicated kernel: the stored energy less the
  /// hardening energy); NaN with the generic dispatch, which does not
  /// separate it.
  Scalar elastic_energy() const { return elastic_; }

  /// The internal force by the generic dispatch at u from `history`
  /// (empty: virgin), with nothing stored.
  Vector generic_internal(const Vector& u,
                          const std::vector<std::vector<PlasticState>>& history) const;

  /// Compare the dedicated kernel with the generic dispatch at u plus a
  /// small deterministic perturbation, from the current history (not
  /// changed); returns the relative difference of the internal forces.
  Scalar self_check(const Vector& u);
  /// Give up the dedicated kernel (after a failed self-check).
  void use_generic(const std::string& reason);

  /// The largest equivalent plastic strain of the history.
  Scalar max_plastic_strain() const;
  /// The first element with a point at det F <= 0 at u (current Jacobian
  /// not positive), or -1.
  Index first_inverted(const Vector& u) const;

  /// Keep a copy of the history / return to it.
  void checkpoint();
  void restore();

 private:
  void build_dedicated();
  void dedicated_forces(const Vector& u, Scalar lambda, bool commit, Vector& internal,
                        Vector& external, Scalar& energy, bool want_energy);
  PlasticState point_state(Index e, int q) const;
  void set_point_state(Index e, int q, const PlasticState& p);

  const FemModel& model_;
  const Assembler& assembler_;
  NonlinearOptions options_;
  std::unique_ptr<detail::NonlinearSystem> system_;  ///< the generic dispatch (and its history)
  bool dedicated_ = false;
  std::string reason_;

 public:
  /// The dedicated kernel's integration points (the Hex8's 2 x 2 x 2 rule).
  static constexpr int kPoints = 8;
  /// The history of an elastoplastic element's eight points, one array per
  /// component with the point index running fastest, when no material
  /// needs the general return's backstresses.
  struct PointBlock {
    alignas(64) Scalar plastic[6][kPoints];  ///< plastic strain, engineering shears
    alignas(64) Scalar back[6][kPoints];     ///< back stress, tensorial [Pa]
    alignas(64) Scalar alpha[kPoints];       ///< equivalent plastic strain
    alignas(64) Scalar thickness[kPoints];   ///< plane-stress thickness strain (unused in 3-D)
    unsigned char loading[kPoints];          ///< PlasticState::loading
  };
  /// Per material, the constants of the elastic predictor of the J2
  /// return (Plasticity.cpp's return_3d), which the kernel evaluates itself
  /// for a point that stays elastic.
  struct Predictor {
    Scalar shear = 0.0;     ///< G
    Scalar bulk = 0.0;      ///< K
    Scalar yield0 = 0.0;    ///< sigma_y0
    Scalar hardening = 0.0; ///< H
    Scalar saturation = 0.0;
    Scalar rate = 0.0;
    bool enabled = false;   ///< plastic at all
    bool radial = false;    ///< the radial return applies (J2, Prager): the predictor is exact
  };

 private:
  // --- dedicated Hex8 kernel ---------------------------------------------
  bool finite_ = true;
  bool logarithmic_ = false;  ///< finite with the logarithmic strain
  bool uniform_ = false;    ///< one set of gradients for every element
  bool compact_ = true;     ///< point blocks (no backstresses)
  Index ne_ = 0;
  Index nn_ = 0;
  /// Per geometric element (one on a uniform mesh): the reference gradients
  /// G(j, a) = dN_a/dX_j of the eight points, laid out point-fastest,
  /// [j][a][q], and node-fastest, [q][j][a] (192 values each); the weights
  /// w det J, [q]; and the mean-dilatation constants A = sum_q w G ([i][a],
  /// 24), B = sum_q w G^T G (64) and V = sum_q w (89 values).
  std::vector<Scalar> gradients_point_;
  std::vector<Scalar> gradients_node_;
  std::vector<Scalar> weights_;
  std::vector<Scalar> dilatation_;
  std::vector<int> material_;      ///< per element: index into materials_
  std::vector<const IsotropicMaterial*> materials_;
  std::vector<Scalar> elasticity_;  ///< per material: D (3-D), 36 values row-major
  std::vector<Predictor> predictor_;
  std::vector<char> plastic_;        ///< per element
  std::vector<char> averaged_;       ///< per element
  std::vector<Index> slot_;          ///< per element: its block (or first full state), -1 if elastic
  std::vector<PointBlock> blocks_;
  std::vector<PlasticState> full_state_;
  std::vector<PointBlock> blocks_saved_;
  std::vector<PlasticState> full_saved_;
  std::vector<Scalar> element_force_;   ///< ne x 24
  std::vector<Scalar> element_energy_;  ///< ne
  std::vector<Scalar> element_elastic_; ///< ne: their elastic parts
  Scalar elastic_ = std::numeric_limits<Scalar>::quiet_NaN();
  std::vector<Index> incidence_ptr_;    ///< per node, into incidence_
  std::vector<Index> incidence_;        ///< offsets e * 24 + 3 a, ascending element
  Vector dead_;                         ///< the applied loads at lambda = 1 (none: zero)
  std::vector<std::vector<PlasticState>> generic_saved_;
};

}  // namespace sparlab
