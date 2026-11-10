// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/uplift.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <thread>

#include "ernie/util/log.h"

namespace ernie {

namespace {

constexpr uint32_t kSamples = 95;
constexpr float kStep = (kSigmoidLastNm - kSigmoidFirstNm) / float(kSamples - 1);
constexpr float kWhiteKelvin = 6504;

float Lobe(float nanometers, float mean, float below, float above) {
    const float t = (nanometers - mean) / (nanometers < mean ? below : above);
    return std::exp(-0.5f * t * t);
}

std::array<float, 3> Observer(float nanometers) {
    const float x = 1.056f * Lobe(nanometers, 599.8f, 37.9f, 31.0f) +
                    0.362f * Lobe(nanometers, 442.0f, 16.0f, 26.7f) -
                    0.065f * Lobe(nanometers, 501.1f, 20.4f, 26.2f);
    const float y = 0.821f * Lobe(nanometers, 568.8f, 46.9f, 40.5f) +
                    0.286f * Lobe(nanometers, 530.9f, 16.3f, 31.1f);
    const float z = 1.217f * Lobe(nanometers, 437.0f, 11.8f, 36.0f) +
                    0.681f * Lobe(nanometers, 459.0f, 26.0f, 13.8f);
    return {x, y, z};
}

float Planck(float nanometers, float kelvin) {
    const double meters = double(nanometers) * 1e-9;
    return float(1 / (std::pow(meters, 5) * (std::exp(1.438776877e-2 / (meters * kelvin)) - 1)));
}

struct Weights {
    std::array<std::array<float, 3>, kSamples> rgb{};
    std::array<float, kSamples> nanometers{};
};

const Weights& RgbWeights() {
    static const Weights weights = [] {
        Weights made;
        std::array<std::array<float, 3>, kSamples> xyz{};
        std::array<double, 3> white{};
        for (uint32_t i = 0; i < kSamples; ++i) {
            const float nanometers = kSigmoidFirstNm + float(i) * kStep;
            made.nanometers[i] = nanometers;
            const std::array<float, 3> observer = Observer(nanometers);
            const float light = Planck(nanometers, kWhiteKelvin);
            for (int c = 0; c < 3; ++c) {
                xyz[i][c] = observer[c] * light;
            }
        }
        constexpr float kToRgb[3][3] = {{3.2404542f, -1.5371385f, -0.4985314f},
                                        {-0.9692660f, 1.8760108f, 0.0415560f},
                                        {0.0556434f, -0.2040259f, 1.0572252f}};
        for (uint32_t i = 0; i < kSamples; ++i) {
            for (int c = 0; c < 3; ++c) {
                made.rgb[i][c] =
                    kToRgb[c][0] * xyz[i][0] + kToRgb[c][1] * xyz[i][1] + kToRgb[c][2] * xyz[i][2];
                white[c] += made.rgb[i][c];
            }
        }
        for (uint32_t i = 0; i < kSamples; ++i) {
            for (int c = 0; c < 3; ++c) {
                made.rgb[i][c] = float(made.rgb[i][c] / white[c]);
            }
        }
        return made;
    }();
    return weights;
}

float Sigmoid(float x) {
    return std::isinf(x) ? (x > 0 ? 1.0f : 0.0f) : 0.5f + x / (2 * std::sqrt(1 + x * x));
}

float Normalized(float nanometers) {
    return (nanometers - kSigmoidFirstNm) / (kSigmoidLastNm - kSigmoidFirstNm);
}

std::array<float, 3> Residual(const std::array<float, 3>& c, const std::array<float, 3>& rgb) {
    const Weights& weights = RgbWeights();
    std::array<float, 3> sum{};
    for (uint32_t i = 0; i < kSamples; ++i) {
        const float value = SigmoidValue(c, weights.nanometers[i]);
        for (int k = 0; k < 3; ++k) {
            sum[k] += value * weights.rgb[i][k];
        }
    }
    return {sum[0] - rgb[0], sum[1] - rgb[1], sum[2] - rgb[2]};
}

template <size_t N>
std::optional<std::array<double, N>> Solve(std::array<std::array<double, N + 1>, N> m) {
    for (size_t pivot = 0; pivot < N; ++pivot) {
        size_t best = pivot;
        for (size_t r = pivot + 1; r < N; ++r) {
            best = std::abs(m[r][pivot]) > std::abs(m[best][pivot]) ? r : best;
        }
        std::swap(m[pivot], m[best]);
        if (std::abs(m[pivot][pivot]) < 1e-30) {
            return std::nullopt;
        }
        for (size_t r = 0; r < N; ++r) {
            if (r == pivot) {
                continue;
            }
            const double factor = m[r][pivot] / m[pivot][pivot];
            for (size_t c = pivot; c <= N; ++c) {
                m[r][c] -= factor * m[pivot][c];
            }
        }
    }
    std::array<double, N> x{};
    for (size_t r = 0; r < N; ++r) {
        x[r] = m[r][N] / m[r][r];
    }
    return x;
}

size_t TableIndex(uint32_t l, uint32_t zi, uint32_t y, uint32_t x) {
    constexpr size_t n = kSigmoidResolution;
    return (((l * n + zi) * n + y) * n + x) * 3;
}

std::array<float, 3> TableColor(uint32_t l, float z, uint32_t y, uint32_t x) {
    constexpr float last = float(kSigmoidResolution - 1);
    std::array<float, 3> rgb{};
    rgb[l] = z;
    rgb[(l + 1) % 3] = float(x) / last * z;
    rgb[(l + 2) % 3] = float(y) / last * z;
    return rgb;
}

}

std::array<float, 3> FitSigmoid(std::array<float, 3> rgb, std::array<float, 3> c) {
    if (rgb[0] <= 0 && rgb[1] <= 0 && rgb[2] <= 0) {
        return {0, 0, -1e4f};
    }
    const auto norm = [](const std::array<float, 3>& r) {
        return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    };
    std::array<float, 3> r = Residual(c, rgb);
    for (int iteration = 0; iteration < 50 && norm(r) > 1e-14f; ++iteration) {
        std::array<std::array<double, 4>, 3> system{};
        for (int k = 0; k < 3; ++k) {
            std::array<float, 3> moved = c;
            const float delta = 1e-3f * std::max(1.0f, std::abs(c[k]));
            moved[k] += delta;
            const std::array<float, 3> shifted = Residual(moved, rgb);
            for (int row = 0; row < 3; ++row) {
                system[row][k] = (shifted[row] - r[row]) / delta;
                system[row][3] = r[row];
            }
        }
        const std::optional<std::array<double, 3>> solved = Solve<3>(system);
        if (!solved) {
            break;
        }
        const std::array<float, 3> step = {float((*solved)[0]), float((*solved)[1]),
                                           float((*solved)[2])};
        bool improved = false;
        for (float scale = 1; scale > 1e-4f; scale *= 0.5f) {
            const std::array<float, 3> tried = {c[0] - scale * step[0], c[1] - scale * step[1],
                                                c[2] - scale * step[2]};
            const std::array<float, 3> residual = Residual(tried, rgb);
            if (norm(residual) < norm(r)) {
                c = tried;
                r = residual;
                improved = true;
                break;
            }
        }
        if (!improved) {
            break;
        }
    }
    return c;
}

float SigmoidValue(const std::array<float, 3>& c, float nanometers) {
    const float t = Normalized(nanometers);
    return Sigmoid((c[0] * t + c[1]) * t + c[2]);
}

std::array<float, 3> SigmoidRgb(const std::array<float, 3>& c) {
    return Residual(c, {0, 0, 0});
}

std::vector<std::array<float, 3>> LinearUplift() {
    const SigmoidTable& table = Sigmoids();
    constexpr uint32_t n = kSigmoidResolution;
    std::vector<std::array<float, 3>> colours;
    std::vector<std::array<float, 3>> coefficients;
    for (uint32_t l = 0; l < 3; ++l) {
        for (uint32_t zi = 0; zi < n; ++zi) {
            for (uint32_t y = 0; y < n; ++y) {
                for (uint32_t x = 0; x < n; ++x) {
                    const size_t at = TableIndex(l, zi, y, x);
                    colours.push_back(TableColor(l, table.scales[zi], y, x));
                    coefficients.push_back({table.coefficients[at], table.coefficients[at + 1],
                                            table.coefficients[at + 2]});
                }
            }
        }
    }
    std::vector<std::array<float, 3>> weights;
    for (uint32_t k = 0; k < kLinearUpliftSamples; ++k) {
        const float nanometers = kLinearUpliftFirstNm + float(k) * kLinearUpliftStepNm;
        std::array<std::array<double, 5>, 4> m{};
        for (size_t i = 0; i < colours.size(); ++i) {
            const double value = SigmoidValue(coefficients[i], nanometers);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    m[r][c] += 2.0 * colours[i][r] * colours[i][c];
                }
                m[r][4] += 2.0 * colours[i][r] * value;
            }
        }
        for (int r = 0; r < 3; ++r) {
            m[r][3] = 1;
            m[3][r] = 1;
        }
        m[3][4] = 1;
        const std::array<double, 4> solved = Solve<4>(m).value_or(std::array<double, 4>{});
        weights.push_back({float(solved[0]), float(solved[1]), float(solved[2])});
    }
    return weights;
}

const SigmoidTable& Sigmoids() {
    static const SigmoidTable table = [] {
        const auto start = std::chrono::steady_clock::now();
        constexpr uint32_t n = kSigmoidResolution;
        SigmoidTable made;
        made.scales.resize(n);
        const auto smoothstep = [](float x) { return x * x * (3 - 2 * x); };
        for (uint32_t k = 0; k < n; ++k) {
            made.scales[k] = smoothstep(smoothstep(float(k) / float(n - 1)));
        }
        made.coefficients.assign(size_t(3) * n * n * n * 3, 0);
        std::vector<std::thread> workers;
        for (uint32_t l = 0; l < 3; ++l) {
            workers.emplace_back([&made, l, n] {
                for (uint32_t y = 0; y < n; ++y) {
                    for (uint32_t x = 0; x < n; ++x) {
                        const auto fit = [&](uint32_t zi, std::array<float, 3> c) {
                            c = FitSigmoid(TableColor(l, made.scales[zi], y, x), c);
                            std::copy(c.begin(), c.end(),
                                      made.coefficients.begin() +
                                          ptrdiff_t(TableIndex(l, zi, y, x)));
                            return c;
                        };
                        const uint32_t middle = n / 5;
                        std::array<float, 3> c{};
                        for (uint32_t zi = middle; zi < n; ++zi) {
                            c = fit(zi, c);
                        }
                        c = {};
                        for (uint32_t zi = middle; zi-- > 0;) {
                            c = fit(zi, c);
                        }
                    }
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
        Log(LogLevel::Debug, "fitted the {}-cubed RGB sigmoid uplift table in {:.1f} s", n,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        return made;
    }();
    return table;
}
}
