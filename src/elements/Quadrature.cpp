// Where this fits: see include/sparlab/elements/Quadrature.hpp

#include "sparlab/elements/Quadrature.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <array>
#include <cmath>
#include <map>
#include <sstream>

namespace sparlab {
namespace {

/// Abscissae / weights of Gauss-Legendre rules on [-1, 1].
const std::vector<QuadraturePoint1D>& line_rule(int points) {
  static const std::map<int, std::vector<QuadraturePoint1D>> rules = [] {
    std::map<int, std::vector<QuadraturePoint1D>> table;
    table[1] = {{0.0, 2.0}};

    const Scalar a2 = 1.0 / std::sqrt(3.0);
    table[2] = {{-a2, 1.0}, {a2, 1.0}};

    const Scalar a3 = std::sqrt(3.0 / 5.0);
    const Scalar w3a = 5.0 / 9.0;
    const Scalar w3b = 8.0 / 9.0;
    table[3] = {{-a3, w3a}, {0.0, w3b}, {a3, w3a}};

    // 4-point rule: x = sqrt((3 -+ 2 sqrt(6/5))/7)
    const Scalar r = std::sqrt(6.0 / 5.0);
    const Scalar x1 = std::sqrt((3.0 - 2.0 * r) / 7.0);
    const Scalar x2 = std::sqrt((3.0 + 2.0 * r) / 7.0);
    const Scalar s = std::sqrt(30.0);
    const Scalar w1 = (18.0 + s) / 36.0;
    const Scalar w2 = (18.0 - s) / 36.0;
    table[4] = {{-x2, w2}, {-x1, w1}, {x1, w1}, {x2, w2}};
    return table;
  }();

  const auto it = rules.find(points);
  if (it == rules.end()) {
    std::ostringstream os;
    os << "Gauss-Legendre rule with " << points
       << " points is not available (supported: 1, 2, 3, 4)";
    throw ConfigError(os.str());
  }
  return it->second;
}

}  // namespace

const std::vector<QuadraturePoint2D>& gauss_legendre_square(int points_per_direction) {
  static std::map<int, std::vector<QuadraturePoint2D>> cache;
  const auto cached = cache.find(points_per_direction);
  if (cached != cache.end()) return cached->second;

  const auto& rule = line_rule(points_per_direction);  // validates the order
  std::vector<QuadraturePoint2D> tensor;
  tensor.reserve(rule.size() * rule.size());
  for (const auto& gj : rule) {
    for (const auto& gi : rule) {
      QuadraturePoint2D p;
      p.xi = gi.xi;
      p.eta = gj.xi;
      p.weight = gi.weight * gj.weight;
      tensor.push_back(p);
    }
  }
  return cache.emplace(points_per_direction, std::move(tensor)).first->second;
}

const std::vector<QuadraturePoint3D>& gauss_legendre_cube(int points_per_direction) {
  static std::map<int, std::vector<QuadraturePoint3D>> cache;
  const auto cached = cache.find(points_per_direction);
  if (cached != cache.end()) return cached->second;

  const auto& rule = line_rule(points_per_direction);
  std::vector<QuadraturePoint3D> tensor;
  tensor.reserve(rule.size() * rule.size() * rule.size());
  for (const auto& gk : rule) {
    for (const auto& gj : rule) {
      for (const auto& gi : rule) {
        QuadraturePoint3D p;
        p.xi = gi.xi;
        p.eta = gj.xi;
        p.zeta = gk.xi;
        p.weight = gi.weight * gj.weight * gk.weight;
        tensor.push_back(p);
      }
    }
  }
  return cache.emplace(points_per_direction, std::move(tensor)).first->second;
}

namespace {

/// The n-point Gauss-Legendre rule by Newton's iteration on P_n (n >= 1): the
/// roots are symmetric, so the negative ones are found and mirrored, the
/// middle one of an odd rule is 0 exactly.
std::vector<QuadraturePoint1D> newton_legendre_rule(int n) {
  const Scalar pi = std::acos(-1.0);
  std::vector<QuadraturePoint1D> rule(static_cast<std::size_t>(n));
  for (int i = 0; i < (n + 1) / 2; ++i) {
    // Tricomi's initial guess for the i-th largest root, then Newton on P_n.
    Scalar x = std::cos(pi * (static_cast<Scalar>(i) + 0.75) / (static_cast<Scalar>(n) + 0.5));
    Scalar derivative = 0.0;
    for (int it = 0; it < 100; ++it) {
      Scalar p0 = 1.0;
      Scalar p1 = x;
      for (int k = 2; k <= n; ++k) {
        const Scalar p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      const Scalar pn = n == 1 ? x : p1;
      const Scalar pm = n == 1 ? 1.0 : p0;
      derivative = static_cast<Scalar>(n) * (x * pn - pm) / (x * x - 1.0);
      const Scalar step = pn / derivative;
      x -= step;
      if (std::abs(step) <= 1.0e-16) break;
    }
    if (2 * i + 1 == n) x = 0.0;
    // Re-evaluate the derivative at the final root for the weight.
    Scalar p0 = 1.0;
    Scalar p1 = x;
    for (int k = 2; k <= n; ++k) {
      const Scalar p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
      p0 = p1;
      p1 = p2;
    }
    const Scalar pn = n == 1 ? x : p1;
    const Scalar pm = n == 1 ? 1.0 : p0;
    derivative = static_cast<Scalar>(n) * (x * pn - pm) / (x * x - 1.0);
    const Scalar weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
    rule[static_cast<std::size_t>(i)] = {-x, weight};
    rule[static_cast<std::size_t>(n - 1 - i)] = {x, weight};
  }
  return rule;
}

void check_thickness_order(int points) {
  if (points < 1 || points > kMaxThicknessPoints) {
    std::ostringstream os;
    os << "Gauss-Legendre rule with " << points
       << " points through the thickness is not available (supported: 1 to "
       << kMaxThicknessPoints << ")";
    throw ConfigError(os.str());
  }
}

}  // namespace

const std::vector<QuadraturePoint1D>& gauss_legendre_line_extended(int points) {
  check_thickness_order(points);
  static const std::array<std::vector<QuadraturePoint1D>, kMaxThicknessPoints + 1> rules = [] {
    std::array<std::vector<QuadraturePoint1D>, kMaxThicknessPoints + 1> table;
    for (int n = 1; n <= kMaxThicknessPoints; ++n) {
      table[static_cast<std::size_t>(n)] = n <= 4 ? line_rule(n) : newton_legendre_rule(n);
    }
    return table;
  }();
  return rules[static_cast<std::size_t>(points)];
}

const std::vector<QuadraturePoint3D>& gauss_legendre_box(int in_plane, int thickness,
                                                          int thickness_axis) {
  if (in_plane < 1 || in_plane > 4) {
    std::ostringstream os;
    os << "Gauss-Legendre rule with " << in_plane
       << " in-plane points per direction is not available (supported: 1, 2, 3, 4)";
    throw ConfigError(os.str());
  }
  check_thickness_order(thickness);
  if (thickness_axis < 0 || thickness_axis > 2) {
    std::ostringstream os;
    os << "the thickness axis of a box rule is " << thickness_axis
       << "; expected 0 (xi), 1 (eta) or 2 (zeta)";
    throw ConfigError(os.str());
  }
  // Every (in-plane, thickness, axis) combination, built once: 4 x 7 x 3
  // rules of at most 112 points.
  constexpr int kAxes = 3;
  constexpr int kCombinations = 4 * kMaxThicknessPoints * kAxes;
  static const std::array<std::vector<QuadraturePoint3D>, kCombinations> rules = [] {
    std::array<std::vector<QuadraturePoint3D>, kCombinations> table;
    for (int np = 1; np <= 4; ++np) {
      for (int nt = 1; nt <= kMaxThicknessPoints; ++nt) {
        for (int axis = 0; axis < kAxes; ++axis) {
          const std::vector<QuadraturePoint1D>& plane = gauss_legendre_line_extended(np);
          const std::vector<QuadraturePoint1D>& through = gauss_legendre_line_extended(nt);
          const std::vector<QuadraturePoint1D>& rx = axis == 0 ? through : plane;
          const std::vector<QuadraturePoint1D>& ry = axis == 1 ? through : plane;
          const std::vector<QuadraturePoint1D>& rz = axis == 2 ? through : plane;
          std::vector<QuadraturePoint3D> tensor;
          tensor.reserve(rx.size() * ry.size() * rz.size());
          // The loop order and the product order of the weights are those of
          // gauss_legendre_cube, so equal orders reproduce it bit for bit.
          for (const auto& gk : rz) {
            for (const auto& gj : ry) {
              for (const auto& gi : rx) {
                QuadraturePoint3D p;
                p.xi = gi.xi;
                p.eta = gj.xi;
                p.zeta = gk.xi;
                p.weight = gi.weight * gj.weight * gk.weight;
                tensor.push_back(p);
              }
            }
          }
          table[static_cast<std::size_t>(((np - 1) * kMaxThicknessPoints + (nt - 1)) * kAxes +
                                         axis)] = std::move(tensor);
        }
      }
    }
    return table;
  }();
  return rules[static_cast<std::size_t>(((in_plane - 1) * kMaxThicknessPoints +
                                         (thickness - 1)) * kAxes + thickness_axis)];
}

}  // namespace sparlab
