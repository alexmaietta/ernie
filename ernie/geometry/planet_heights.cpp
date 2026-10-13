// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/planet_heights.h"

#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/util/profile.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace ernie {

namespace {

template <typename T> double Load(const std::byte* sample) {
    T value;
    std::memcpy(&value, sample, sizeof(T));
    return static_cast<double>(value);
}

double SampleOf(const std::byte* sample, RasterSampleType type) {
    switch (type) {
    case RasterSampleType::UInt8:
        return Load<std::uint8_t>(sample);
    case RasterSampleType::UInt16:
        return Load<std::uint16_t>(sample);
    case RasterSampleType::Int16:
        return Load<std::int16_t>(sample);
    case RasterSampleType::UInt32:
        return Load<std::uint32_t>(sample);
    case RasterSampleType::Int32:
        return Load<std::int32_t>(sample);
    case RasterSampleType::Float32:
        break;
    }
    return Load<float>(sample);
}

/// The raster box-filtered by `factor` in both directions: heights
/// (NaN without data) of cells `factor` samples square, the last row
/// and column of cells over what remains.
std::vector<float> Filtered(const RasterSource& source,
                            std::uint32_t factor, std::uint32_t width,
                            std::uint32_t height,
                            const PlanetHeightsOptions& options) {
    const std::uint32_t pixel =
        source.Channels() * source.SampleBytes();
    const RasterSampleType type = source.Type();
    const double scale = source.Scale() * options.unit;
    const double offset =
        source.Offset() * options.unit - options.reference;
    const std::optional<double> noData = source.NoData();

    std::vector<float> cells(std::size_t{width} * height);
    std::vector<double> sums(width);
    std::vector<std::uint32_t> counts(width);
    std::vector<std::byte> band;
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::uint32_t y = row * factor;
        const std::uint32_t rows =
            std::min(factor, source.Height() - y);
        band.resize(std::size_t{source.Width()} * rows * pixel);
        source.Read(0, y, source.Width(), rows, band);
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0u);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const std::byte* line =
                band.data() + std::size_t{r} * source.Width() * pixel;
            for (std::uint32_t x = 0; x < source.Width(); ++x) {
                const double sample =
                    SampleOf(line + x * pixel, type);
                // PDS marks missing reals with values below -1e30.
                if (!std::isfinite(sample) || sample < -1e30 ||
                    (noData && sample == *noData)) {
                    continue;
                }
                sums[x / factor] += sample;
                ++counts[x / factor];
            }
        }
        for (std::uint32_t col = 0; col < width; ++col) {
            cells[std::size_t{row} * width + col] =
                counts[col] == 0
                    ? std::numeric_limits<float>::quiet_NaN()
                    : static_cast<float>(
                          sums[col] / counts[col] * scale + offset);
        }
    }
    return cells;
}

}  // namespace

std::vector<float>
PlanetHeightsOf(const RasterSource& source, std::uint32_t faceCells,
                const PlanetHeightsOptions& options) {
    ERNIE_ZONE("Resample planet heights");
    if (faceCells == 0) {
        throw std::invalid_argument("Planet faces need cells");
    }
    const LatLongGrid grid =
        options.grid.value_or(source.Grid().value_or(
            LatLongGrid{-180.0, 90.0, 360.0 / source.Width(),
                        180.0 / source.Height()}));
    if (!(grid.degreesAcross > 0.0) || !(grid.degreesDown > 0.0)) {
        throw std::invalid_argument(
            "A latitude-longitude grid needs positive spacing");
    }
    const double spacing = 90.0 / faceCells;
    const auto factor = static_cast<std::uint32_t>(std::max(
        1.0, std::floor(spacing / std::max(grid.degreesAcross,
                                           grid.degreesDown))));
    const std::uint32_t width =
        (source.Width() + factor - 1) / factor;
    const std::uint32_t height =
        (source.Height() + factor - 1) / factor;
    const std::vector<float> cells =
        Filtered(source, factor, width, height, options);

    const double across = grid.degreesAcross * factor;
    const double down = grid.degreesDown * factor;
    const double spanned = grid.degreesAcross * source.Width();
    const bool wraps = spanned >= 360.0 * (1.0 - 1e-9);
    const double south =
        grid.north - grid.degreesDown * source.Height();
    const auto at = [&](double latitude, double longitude) {
        double east = std::fmod(longitude - grid.west, 360.0);
        east += east < 0.0 ? 360.0 : 0.0;
        if ((!wraps && east > spanned) || latitude > grid.north ||
            latitude < south) {
            return 0.0f;
        }
        const double x = east / across - 0.5;
        const double y =
            std::clamp((grid.north - latitude) / down - 0.5, 0.0,
                       static_cast<double>(height - 1));
        const double x0 = std::floor(x);
        const double y0 = std::floor(y);
        const double fx = x - x0;
        const double fy = y - y0;
        const auto column = [&](double c) {
            const auto i = static_cast<std::int64_t>(c);
            if (wraps) {
                return static_cast<std::uint32_t>(
                    ((i % width) + width) % width);
            }
            return static_cast<std::uint32_t>(
                std::clamp<std::int64_t>(
                    i, 0, static_cast<std::int64_t>(width) - 1));
        };
        const std::uint32_t c0 = column(x0);
        const std::uint32_t c1 = column(x0 + 1.0);
        const auto r0 = static_cast<std::uint32_t>(y0);
        const std::uint32_t r1 = std::min(r0 + 1, height - 1);
        double sum = 0.0;
        double weights = 0.0;
        for (const auto& [r, c, w] :
             {std::tuple{r0, c0, (1.0 - fx) * (1.0 - fy)},
              std::tuple{r0, c1, fx * (1.0 - fy)},
              std::tuple{r1, c0, (1.0 - fx) * fy},
              std::tuple{r1, c1, fx * fy}}) {
            const float h = cells[std::size_t{r} * width + c];
            if (!std::isnan(h) && w > 0.0) {
                sum += w * static_cast<double>(h);
                weights += w;
            }
        }
        return weights > 0.0 ? static_cast<float>(sum / weights)
                             : 0.0f;
    };

    const std::size_t side = std::size_t{faceCells} + 1;
    std::vector<float> heights(CubeSphereLod::kFaces * side * side);
    const double degrees = 180.0 / std::numbers::pi;
    // As CubeSphereLod::VertexDirection, so shared edges agree
    // exactly.
    const double scale = 2.0 / static_cast<double>(faceCells);
    std::vector<std::jthread> faces;
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces;
         ++face) {
        faces.emplace_back([&, face] {
            float* out = heights.data() + face * side * side;
            for (std::uint32_t row = 0; row <= faceCells; ++row) {
                const double t = row * scale - 1.0;
                for (std::uint32_t col = 0; col <= faceCells; ++col) {
                    const double s = col * scale - 1.0;
                    const glm::dvec3 d =
                        CubeSphereLod::FaceDirection(face, s, t);
                    out[row * side + col] =
                        at(std::asin(std::clamp(d.z, -1.0, 1.0)) *
                               degrees,
                           std::atan2(d.y, d.x) * degrees);
                }
            }
        });
    }
    return heights;
}

}  // namespace ernie
