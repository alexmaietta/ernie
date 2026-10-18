// SPDX-License-Identifier: BSD-3-Clause
/// @file linear_algebra_test.cpp
/// @brief Tests of the small dense solves and least-squares fits.
#include "ernie/core/linear_algebra.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace ernie {
namespace {

TEST(LinearAlgebra, SolvesSystemsThatNeedPivoting) {
    // The first pivot is zero, so rows must swap.
    const std::vector<double> x = SolveLinearSystem<double>({0.0, 2.0, 1.0,  //
                                                             1.0, 1.0, 0.0,  //
                                                             3.0, 0.0, 1.0},
                                                            {5.0, 3.0, 6.0});
    EXPECT_NEAR(x[0], 1.4, 1e-12);
    EXPECT_NEAR(x[1], 1.6, 1e-12);
    EXPECT_NEAR(x[2], 1.8, 1e-12);
}

TEST(LinearAlgebra, SolvesVectorRightHandSidesComponentWise) {
    const std::vector<glm::dvec3> x = SolveLinearSystem<glm::dvec3>(
        {2.0, 0.0, 0.0, 4.0}, {glm::dvec3(2.0, 4.0, 6.0), glm::dvec3(4.0)});
    EXPECT_EQ(x[0], glm::dvec3(1.0, 2.0, 3.0));
    EXPECT_EQ(x[1], glm::dvec3(1.0));
}

TEST(LinearAlgebra, SingularSystemsGiveZeros) {
    EXPECT_EQ(SolveLinearSystem<double>({1.0, 2.0, 2.0, 4.0}, {1.0, 2.0}),
              (std::vector<double>{0.0, 0.0}));
    EXPECT_EQ(Solve(glm::dmat3(0.0), glm::dvec3(1.0)), glm::dvec3(0.0));
    EXPECT_EQ(Solve(glm::dmat3(2.0), glm::dvec3(4.0)), glm::dvec3(2.0));
}

TEST(LinearAlgebra, NormalEquationsFitALine) {
    // y = 2 + 3 x, exactly, at five points.
    NormalEquations<2> fit;
    for (int i = 0; i < 5; ++i) {
        const double x = i;
        fit.Add({1.0, x}, 2.0 + 3.0 * x);
    }
    const std::array<double, 2> coefficients = fit.Solve();
    EXPECT_NEAR(coefficients[0], 2.0, 1e-12);
    EXPECT_NEAR(coefficients[1], 3.0, 1e-12);
}

}  // namespace
}  // namespace ernie
