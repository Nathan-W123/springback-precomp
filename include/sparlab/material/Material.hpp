/// \file Material.hpp
/// \brief The sheet material: isotropic linear elasticity and its constitutive
///        matrices, plus the plasticity parameters (von Mises or Hill48 yield,
///        isotropic and kinematic hardening) that Plasticity.hpp integrates.
///
/// Voigt ordering used throughout SparLab is
/// \f$ \{\sigma_{xx}, \sigma_{yy}, \sigma_{xy}\} \f$ in 2-D and
/// \f$ \{\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{yz},
/// \sigma_{zx}\} \f$ in 3-D, paired with the *engineering* strain vector
/// (\f$\gamma_{xy} = 2\varepsilon_{xy}\f$ and so on).
///
/// Plane stress (\f$\sigma_{zz}=0\f$):
/// \f[
///   \mathbf{D} = \frac{E}{1-\nu^2}
///   \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \tfrac{1-\nu}{2}
///   \end{bmatrix}
/// \f]
/// Plane strain (\f$\varepsilon_{zz}=0\f$):
/// \f[
///   \mathbf{D} = \frac{E}{(1+\nu)(1-2\nu)}
///   \begin{bmatrix} 1-\nu & \nu & 0 \\ \nu & 1-\nu & 0 \\ 0 & 0 &
///   \tfrac{1-2\nu}{2}\end{bmatrix}
/// \f]
/// In plane strain the out-of-plane stress is
/// \f$\sigma_{zz} = \nu(\sigma_{xx}+\sigma_{yy})\f$.
///
/// **Thermal strain.** A temperature change \f$\Delta T = T - T_{ref}\f$ gives
/// the isotropic free strain \f$\varepsilon_0 = \alpha\Delta T\{1,1,1,0,0,0\}\f$,
/// which the plastic returns (Plasticity.hpp) take off the 3-D strain, the
/// plane states included: \f$\sigma = D(\varepsilon - \varepsilon_0)\f$.
///
/// Three-dimensional elasticity, with Lame constants
/// \f$\lambda = E\nu/((1+\nu)(1-2\nu))\f$ and \f$G = E/(2(1+\nu))\f$:
/// \f[
///   \mathbf{D} = \begin{bmatrix}
///     \lambda+2G & \lambda & \lambda & 0 & 0 & 0
///     \\ \lambda & \lambda+2G & \lambda & 0 & 0 & 0
///     \\ \lambda & \lambda & \lambda+2G & 0 & 0 & 0
///     \\ 0 & 0 & 0 & G & 0 & 0 \\ 0 & 0 & 0 & 0 & G & 0 \\ 0 & 0 & 0 & 0 & 0 & G
///   \end{bmatrix}
/// \f]
///
/// Where this fits: Simulations stage - the sheet material of the deck's material block:
///   elastic constants plus PlasticityParameters (von Mises or Hill48 yield, hardening).
/// Called from: io/Config.cpp:parse_material, fem/FemModel.cpp:FemModel::FemModel,
///   fem/RigidTool.cpp:ToolContact::ToolContact (youngs_modulus)
/// Calls into: core/Exceptions (ConfigError)
/// Start reading at: PlasticityParameters, Material::set_plasticity
#pragma once

#include "sparlab/core/Types.hpp"

#include <array>
#include <string>

namespace sparlab {

/// The yield criterion of a plastic material.
enum class YieldCriterion {
  VonMises,  ///< J2: isotropic
  Hill48     ///< Hill's (1948) quadratic orthotropic criterion
};

/// How the general return integrates an Armstrong-Frederick backstress over
/// a step, with the flow direction of the end of the step.
enum class KinematicIntegration {
  Exponential,   ///< the exact solution for a fixed direction (default)
  BackwardEuler  ///< implicit Euler: first-order accurate
};

/// The most backstresses a material may have (fixed, so that the state of an
/// integration point needs no heap allocation).
constexpr int kMaxBackstresses = 4;

/// One Armstrong-Frederick backstress of a Chaboche sum,
/// \f$\dot\alpha_i = \tfrac{2}{3} C_i\,\dot\varepsilon^p - \gamma_i\,\alpha_i\,\dot{\bar\alpha}\f$.
/// \f$\gamma_i = 0\f$ is Prager's linear rule.
struct Backstress {
  Scalar modulus = 0.0;   ///< \f$C_i\f$ [Pa]
  Scalar recovery = 0.0;  ///< \f$\gamma_i\f$, the dynamic recovery [-]
};

/// How the Hill48 coefficients are given.
enum class HillCalibration {
  RValues,       ///< Lankford coefficients r0, r45, r90 (and L, M)
  StressRatios,  ///< yield stresses at 45 deg, 90 deg and equibiaxial over sigma_0 (and L, M)
  Coefficients   ///< F, G, H, L, M, N directly
};

/// Hill's 1948 criterion in the material frame (1 = rolling direction RD,
/// 2 = transverse direction TD, 3 = sheet normal ND):
/// \f[
///   \bar\sigma^2 = F(\sigma_{22}-\sigma_{33})^2 + G(\sigma_{33}-\sigma_{11})^2
///                + H(\sigma_{11}-\sigma_{22})^2 + 2L\sigma_{23}^2
///                + 2M\sigma_{31}^2 + 2N\sigma_{12}^2 ,
/// \f]
/// of the relative stress. From the r-values (normalised so that
/// \f$\bar\sigma\f$ is the uniaxial stress along RD, G + H = 1):
/// \f$F = r_0/(r_{90}(1+r_0))\f$, \f$G = 1/(1+r_0)\f$,
/// \f$H = r_0/(1+r_0)\f$, \f$N = (r_0+r_{90})(1+2r_{45})/(2r_{90}(1+r_0))\f$;
/// from the stress ratios \f$s_{90} = \sigma_{90}/\sigma_0\f$,
/// \f$s_b = \sigma_b/\sigma_0\f$, \f$s_{45}\f$: G + H = 1,
/// \f$F + H = s_{90}^{-2}\f$, \f$F + G = s_b^{-2}\f$,
/// \f$2N = 4s_{45}^{-2} - F - G\f$. r = 1 (or all ratios 1) is von Mises:
/// F = G = H = 1/2, L = M = N = 3/2. Given explicitly, the coefficients need
/// not satisfy G + H = 1; the uniaxial yield stress along RD is then
/// \f$\sigma_y/\sqrt{G+H}\f$.
struct Hill48Parameters {
  HillCalibration calibration = HillCalibration::RValues;
  Scalar r0 = 1.0;   ///< Lankford coefficient along RD [-]
  Scalar r45 = 1.0;  ///< at 45 deg to RD [-]
  Scalar r90 = 1.0;  ///< along TD [-]
  Scalar sigma45 = 1.0;        ///< StressRatios: \f$\sigma_{45}/\sigma_0\f$ [-]
  Scalar sigma90 = 1.0;        ///< StressRatios: \f$\sigma_{90}/\sigma_0\f$ [-]
  Scalar sigma_biaxial = 1.0;  ///< StressRatios: \f$\sigma_b/\sigma_0\f$ [-]
  /// The out-of-plane shear coefficients L and M of the r-value and stress
  /// ratio calibrations, which in-plane tests do not measure (3/2: isotropic).
  Scalar shear_l = 1.5;
  Scalar shear_m = 1.5;
  /// The coefficients: given with Coefficients, found by set_plasticity
  /// otherwise [-].
  Scalar F = 0.5, G = 0.5, H = 0.5, L = 1.5, M = 1.5, N = 1.5;
  /// The material frame in global axes: the rolling direction is projected
  /// onto the plane normal to the sheet normal, and TD = ND x RD.
  Vector3 rolling_direction = Vector3::UnitX();
  Vector3 sheet_normal = Vector3::UnitZ();
  /// Found by set_plasticity: the orthonormal axes RD, TD, ND as rows.
  Matrix3 axes = Matrix3::Identity();
  /// Found by set_plasticity: the global z is one of the axes, as a plane
  /// model needs (it has no out-of-plane shear strain, which any other frame
  /// couples to the in-plane flow).
  bool z_on_axis = true;
};

/// The yield matrix of von Mises, \f$\bar\sigma^2 = \xi^T P\,\xi =
/// \tfrac{3}{2}\|\mathrm{dev}\,\xi\|^2\f$ (tensorial Voigt components).
Matrix6 von_mises_yield_matrix();

/// Plasticity of an isotropic elastic material: the yield stress after an
/// accumulated plastic strain \f$\bar\alpha\f$ is
/// \f[
///   \sigma_y(\bar\alpha) = \sigma_{y0} + H\bar\alpha + Q\,(1 - e^{-\delta\bar\alpha})
/// \f]
/// (linear isotropic hardening plus Voce saturation). The yield criterion is
/// von Mises (J2) or Hill48 (anisotropic plastic flow, Hill48Parameters), of
/// the relative stress \f$\xi = \sigma - \sum_i\alpha_i\f$: the back stress
/// is Prager's linear kinematic hardening,
/// \f$\dot\beta = \tfrac{2}{3} H_{kin}\,\dot\varepsilon^p\f$, and/or a
/// Chaboche sum of Armstrong-Frederick backstresses (Backstress). A yield
/// stress of 0 means the material stays elastic. Material/Plasticity.hpp
/// integrates the law: the radial return for von Mises with Prager at most,
/// the general closest-point return otherwise (general()).
struct PlasticityParameters {
  Scalar yield_stress = 0.0;                 ///< \f$\sigma_{y0}\f$ [Pa]
  Scalar hardening_modulus = 0.0;            ///< H, linear isotropic [Pa]
  Scalar kinematic_hardening_modulus = 0.0;  ///< \f$H_{kin}\f$, Prager [Pa]
  Scalar saturation_stress = 0.0;            ///< Q, Voce saturation increment [Pa]
  Scalar saturation_rate = 0.0;              ///< \f$\delta\f$, Voce rate [-]

  YieldCriterion criterion = YieldCriterion::VonMises;
  Hill48Parameters hill;  ///< with YieldCriterion::Hill48
  /// The Chaboche backstresses, the first num_backstresses of them.
  std::array<Backstress, kMaxBackstresses> backstresses{};
  int num_backstresses = 0;
  KinematicIntegration kinematic_integration = KinematicIntegration::Exponential;
  /// Found by set_plasticity: the matrix of the yield criterion in global
  /// axes, \f$\bar\sigma^2 = \xi^T P\,\xi\f$ on tensorial Voigt components
  /// (von Mises unless Hill48).
  Matrix6 yield_matrix = von_mises_yield_matrix();

  bool enabled() const { return yield_stress > 0.0; }
  /// The law needs the general return: Hill48, or backstresses.
  bool general() const {
    return criterion == YieldCriterion::Hill48 || num_backstresses > 0;
  }
  /// The consistent tangent is symmetric: false as soon as a backstress
  /// has dynamic recovery.
  bool symmetric_tangent() const;
  /// The law may be used by a plane model (plane strain, plane stress): not
  /// Hill48 in a frame with z off its axes.
  bool plane_compatible() const {
    return criterion != YieldCriterion::Hill48 || hill.z_on_axis;
  }
  /// The backstresses the general return integrates: the Chaboche ones,
  /// then Prager's \f$H_{kin}\f$ as one without recovery.
  int kinematic_terms() const {
    return num_backstresses + (kinematic_hardening_modulus > 0.0 ? 1 : 0);
  }
  Backstress kinematic_term(int i) const {
    return i < num_backstresses ? backstresses[static_cast<std::size_t>(i)]
                                : Backstress{kinematic_hardening_modulus, 0.0};
  }
  /// \f$\sigma_y(\bar\alpha)\f$ [Pa].
  Scalar yield(Scalar alpha) const;
  /// \f$d\sigma_y/d\bar\alpha\f$ [Pa].
  Scalar yield_slope(Scalar alpha) const;
  /// Stored energy of the isotropic hardening per unit volume,
  /// \f$\int_0^{\bar\alpha} (\sigma_y - \sigma_{y0})\,d\bar\alpha\f$ [J/m^3].
  Scalar isotropic_energy(Scalar alpha) const;
};

/// The material of the deck's `material` block: isotropic linear elasticity
/// described by \f$(E, \nu, \rho)\f$, its PlasticityParameters (elastic
/// unless set_plasticity is called) and its thermal properties.
class Material {
 public:
  /// \param youngs_modulus E [Pa], must be > 0.
  /// \param poisson_ratio nu [-], must lie in (-1, 0.5) for plane strain and
  ///        3-D elasticity and (-1, 1) excluding +-1 for plane stress. SparLab
  ///        restricts the input to (-1, 0.5) so every idealisation remains
  ///        well posed.
  /// \param density rho [kg/m^3], must be >= 0 (only the mass matrices use it).
  /// \param name optional label carried into result files.
  Material(Scalar youngs_modulus, Scalar poisson_ratio, Scalar density,
                    std::string name = "material");

  Scalar youngs_modulus() const { return e_; }
  Scalar poisson_ratio() const { return nu_; }
  Scalar density() const { return rho_; }
  const std::string& name() const { return name_; }

  /// Shear modulus \f$G = E / (2(1+\nu))\f$ [Pa].
  Scalar shear_modulus() const { return e_ / (2.0 * (1.0 + nu_)); }

  /// First Lame constant \f$\lambda = E\nu/((1+\nu)(1-2\nu))\f$ [Pa].
  Scalar lame_lambda() const { return e_ * nu_ / ((1.0 + nu_) * (1.0 - 2.0 * nu_)); }

  /// Constitutive matrix for the requested idealisation [Pa]: 3 x 3 for the
  /// plane states, 6 x 6 for three-dimensional elasticity.
  Matrix constitutive(StressState state) const;

  /// Plane-stress constitutive matrix [Pa].
  Matrix3 plane_stress_matrix() const;

  /// Plane-strain constitutive matrix [Pa].
  Matrix3 plane_strain_matrix() const;

  /// Three-dimensional constitutive matrix [Pa].
  Matrix6 three_dimensional_matrix() const;

  /// Plasticity; `plasticity().enabled()` is false for an elastic
  /// material (the default).
  const PlasticityParameters& plasticity() const { return plasticity_; }
  /// Set the plasticity parameters, and find the Hill48 coefficients, the
  /// material frame and the yield matrix from them.
  /// \throws ConfigError for a negative or non-finite parameter, a
  ///         saturation stress without a positive saturation rate, hardening
  ///         or backstresses without a yield stress, more backstresses than
  ///         kMaxBackstresses (Prager's modulus counting as one in the
  ///         general return), a non-positive r-value, stress ratio or
  ///         backstress modulus, Hill coefficients whose quadratic form is
  ///         not positive definite on deviators, or a rolling direction
  ///         parallel to the sheet normal.
  void set_plasticity(const PlasticityParameters& parameters);

  /// Linear thermal expansion coefficient alpha [1/K]; 0 means no thermal
  /// strain.
  Scalar thermal_expansion() const { return alpha_; }
  /// Temperature at which the material carries no thermal strain [K, or
  /// deg C when every temperature of the deck is in deg C].
  Scalar reference_temperature() const { return t_ref_; }
  /// Thermal conductivity k [W/(m K)]: read and validated, but no analysis
  /// of sparlab_form uses it.
  Scalar conductivity() const { return k_; }
  /// Set the thermal properties.
  /// \throws ConfigError for a negative conductivity or non-finite input.
  void set_thermal(Scalar expansion, Scalar reference_temperature, Scalar conductivity);

 private:
  Scalar e_;
  Scalar nu_;
  Scalar rho_;
  std::string name_;
  Scalar alpha_ = 0.0;
  Scalar t_ref_ = 0.0;
  Scalar k_ = 0.0;
  PlasticityParameters plasticity_;
};

}  // namespace sparlab
