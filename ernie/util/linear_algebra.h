// SPDX-License-Identifier: BSD-3-Clause
/// @file linear_algebra.h
/// @brief Small dense linear solves and least-squares fits.
#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace ernie {

/// @brief Solves a x = b by Gaussian elimination with partial pivoting.
/// @param a The n x n matrix, row-major.
/// @param b The n right-hand sides; `T` is a number or a vector solved component-wise.
/// @return x, or all zeros if the matrix is singular.
template <typename T> std::vector<T> SolveLinearSystem(std::vector<double> a, std::vector<T> b) {
    const std::size_t n = b.size();
    const auto at = [&](std::size_t row, std::size_t column) -> double& {
        return a[row * n + column];
    };
    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < n; ++row) {
            if (std::abs(at(row, column)) > std::abs(at(pivot, column))) {
                pivot = row;
            }
        }
        if (std::abs(at(pivot, column)) < 1e-300) {
            return std::vector<T>(n, T{});
        }
        for (std::size_t k = 0; k < n; ++k) {
            std::swap(at(column, k), at(pivot, k));
        }
        std::swap(b[column], b[pivot]);
        for (std::size_t row = column + 1; row < n; ++row) {
            const double factor = at(row, column) / at(column, column);
            for (std::size_t k = column; k < n; ++k) {
                at(row, k) -= factor * at(column, k);
            }
            b[row] -= factor * b[column];
        }
    }
    std::vector<T> x(n, T{});
    for (std::size_t row = n; row-- > 0;) {
        T sum = b[row];
        for (std::size_t k = row + 1; k < n; ++k) {
            sum -= at(row, k) * x[k];
        }
        x[row] = sum / at(row, row);
    }
    return x;
}

/// @brief Solves a 3 x 3 system.
/// @param a The matrix.
/// @param b The right-hand side.
/// @return x with a x = b, or zero if the matrix is singular.
inline glm::dvec3 Solve(const glm::dmat3& a, const glm::dvec3& b) {
    if (std::abs(glm::determinant(a)) < 1e-300) {
        return glm::dvec3(0.0);
    }
    return glm::inverse(a) * b;
}

/// @brief Solves the normal equations `normal` x = `right`, Jacobi-scaled for conditioning.
/// @param normal The normal matrix.
/// @param right  The right-hand side.
/// @param n      How many leading unknowns to solve for.
/// @return The solution, or all zeros if the system is singular.
template <std::size_t N>
std::array<double, N> SolveNormal(const std::array<std::array<double, N>, N>& normal,
                                  const std::array<double, N>& right, std::size_t n) {
    std::array<double, N> scale{};
    for (std::size_t i = 0; i < n; ++i) {
        scale[i] = normal[i][i] > 0.0 ? 1.0 / std::sqrt(normal[i][i]) : 0.0;
    }
    std::vector<double> a(n * n);
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            a[i * n + j] = normal[i][j] * scale[i] * scale[j];
        }
        b[i] = right[i] * scale[i];
    }
    const std::vector<double> scaled = SolveLinearSystem(std::move(a), std::move(b));
    std::array<double, N> x{};
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = scaled[i] * scale[i];
    }
    return x;
}

/// @brief Accumulates a linear least-squares fit's normal equations.
template <std::size_t N> struct NormalEquations {
    /// The sum of row row^T.
    std::array<std::array<double, N>, N> normal{};
    /// The sum of row value.
    std::array<double, N> right{};

    /// @brief Adds the equation `row` . x = `value`.
    /// @param row   The coefficients.
    /// @param value The right-hand side.
    /// @param n     How many leading unknowns to use.
    void Add(const std::array<double, N>& row, double value, std::size_t n = N) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                normal[i][j] += row[i] * row[j];
            }
            right[i] += row[i] * value;
        }
    }

    /// @brief Solves the accumulated system (`SolveNormal`).
    /// @param n How many leading unknowns to solve for.
    /// @return The solution, or zeros if singular.
    std::array<double, N> Solve(std::size_t n = N) const {
        return SolveNormal(normal, right, n);
    }
};

}  // namespace ernie
