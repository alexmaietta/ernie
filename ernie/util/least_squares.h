// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace ernie {

/// The least-squares solution of the first `n` equations of normal system `normal` x = `right`
/// (the rest ignored), scaled by its diagonal (Jacobi) for conditioning and solved by Gaussian
/// elimination with partial pivoting; all zeros where the system is singular.
template <std::size_t N>
std::array<double, N> SolveNormal(std::array<std::array<double, N>, N> normal,
                                  std::array<double, N> right, std::size_t n) {
    std::array<double, N> scale{};
    for (std::size_t i = 0; i < n; ++i) {
        scale[i] = normal[i][i] > 0.0 ? 1.0 / std::sqrt(normal[i][i]) : 0.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            normal[i][j] *= scale[i] * scale[j];
        }
        right[i] *= scale[i];
    }
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t pivot = c;
        for (std::size_t r = c + 1; r < n; ++r) {
            if (std::abs(normal[r][c]) > std::abs(normal[pivot][c])) {
                pivot = r;
            }
        }
        if (std::abs(normal[pivot][c]) < 1e-300) {
            return {};
        }
        std::swap(normal[c], normal[pivot]);
        std::swap(right[c], right[pivot]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const double factor = normal[r][c] / normal[c][c];
            for (std::size_t j = c; j < n; ++j) {
                normal[r][j] -= factor * normal[c][j];
            }
            right[r] -= factor * right[c];
        }
    }
    std::array<double, N> x{};
    for (std::size_t c = n; c-- > 0;) {
        double sum = right[c];
        for (std::size_t j = c + 1; j < n; ++j) {
            sum -= normal[c][j] * x[j];
        }
        x[c] = sum / normal[c][c];
    }
    for (std::size_t i = 0; i < n; ++i) {
        x[i] *= scale[i];
    }
    return x;
}

/// Accumulates the normal equations of a linear least-squares fit, row by row.
template <std::size_t N> struct NormalEquations {
    std::array<std::array<double, N>, N> normal{};
    std::array<double, N> right{};

    /// Adds equation `row` . x = `value` over the first `n` unknowns.
    void Add(const std::array<double, N>& row, double value, std::size_t n = N) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                normal[i][j] += row[i] * row[j];
            }
            right[i] += row[i] * value;
        }
    }

    std::array<double, N> Solve(std::size_t n = N) const {
        return SolveNormal(normal, right, n);
    }
};

}  // namespace ernie
