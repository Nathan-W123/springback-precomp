/// \file test_quadrature_shape.cpp
/// \brief Quadrature exactness and shape-function identities.
#include "sparlab/core/Exceptions.hpp"
#include "sparlab/elements/Quadrature.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace sparlab;
using Catch::Approx;

TEST_CASE("Gauss-Legendre square rules are tensor products with total weight 4",
          "[quadrature]") {
  for (int n = 1; n <= 4; ++n) {
    const auto& rule = gauss_legendre_square(n);
    REQUIRE(static_cast<int>(rule.size()) == n * n);
    Scalar total = 0.0;
    for (const auto& p : rule) total += p.weight;
    REQUIRE(total == Approx(4.0).margin(1.0e-14));
  }

  // 2x2 integrates the bilinear monomial xi^3 eta^3 exactly (degree 3 each).
  const auto& rule = gauss_legendre_square(2);
  Scalar value = 0.0;
  for (const auto& p : rule) {
    value += p.weight * std::pow(p.xi, 3) * std::pow(p.eta, 3);
  }
  REQUIRE(value == Approx(0.0).margin(1.0e-14));

  Scalar quadratic = 0.0;
  for (const auto& p : rule) quadratic += p.weight * p.xi * p.xi * p.eta * p.eta;
  REQUIRE(quadratic == Approx(4.0 / 9.0).margin(1.0e-14));
}

TEST_CASE("unsupported quadrature orders are rejected", "[quadrature][diagnostics]") {
  REQUIRE_THROWS_AS(gauss_legendre_cube(0), ConfigError);
  REQUIRE_THROWS_AS(gauss_legendre_cube(5), ConfigError);
  REQUIRE_THROWS_AS(gauss_legendre_square(7), ConfigError);
}
