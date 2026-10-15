// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/planet_heights.h"

#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/util/profile.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

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
    std::vector<std::thread> faces;
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
    for (std::thread& face : faces) {
        face.join();
    }
    return heights;
}

namespace {

/// A window of a raster read as heights (NaN without data): columns
/// from `x0` (below 0 or past the width across a wrapping raster's
/// seam), rows from `y0`; with `full`, every column, indexed modulo
/// the width.
struct HeightWindow {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t width = 0;
    std::int64_t height = 0;
    bool full = false;
    std::vector<float> values;

    /// The height at window column `x` and row `y`, clamped to the
    /// window (wrapped across a full one).
    float At(std::int64_t x, std::int64_t y) const {
        x = full ? ((x % width) + width) % width
                 : std::clamp<std::int64_t>(x, 0, width - 1);
        y = std::clamp<std::int64_t>(y, 0, height - 1);
        return values[static_cast<std::size_t>(y * width + x)];
    }
};

/// Bilinear between the samples of `at(x, y)` (heights, NaN without
/// data) around fractional (`x`, `y`), leaving out those without
/// data; NaN with none.
template <typename At> float Bilinear(double x, double y, At&& at) {
    const double x0 = std::floor(x);
    const double y0 = std::floor(y);
    const double fx = x - x0;
    const double fy = y - y0;
    const auto ix = static_cast<std::int64_t>(x0);
    const auto iy = static_cast<std::int64_t>(y0);
    double sum = 0.0;
    double weights = 0.0;
    for (const auto& [dx, dy, w] :
         {std::tuple{0, 0, (1.0 - fx) * (1.0 - fy)},
          std::tuple{1, 0, fx * (1.0 - fy)},
          std::tuple{0, 1, (1.0 - fx) * fy},
          std::tuple{1, 1, fx * fy}}) {
        const float h = at(ix + dx, iy + dy);
        if (!std::isnan(h) && w > 0.0) {
            sum += w * static_cast<double>(h);
            weights += w;
        }
    }
    return weights > 0.0 ? static_cast<float>(sum / weights)
                         : std::numeric_limits<float>::quiet_NaN();
}

}  // namespace

PlanetTile PlanetTileOf(const RasterSource& source,
                        const PlanetHeightsOptions& options,
                        const glm::dvec3& radii,
                        std::uint32_t faceCells, std::uint32_t face,
                        std::uint32_t column, std::uint32_t row,
                        std::uint32_t cells, bool ranges,
                        std::mutex* reading) {
    ERNIE_ZONE("Planet height tile");
    if (cells == 0 || faceCells == 0 ||
        face >= CubeSphereLod::kFaces || column > faceCells ||
        cells > faceCells - column || row > faceCells ||
        cells > faceCells - row) {
        throw std::invalid_argument(
            "A planet tile needs cells on its face");
    }
    const LatLongGrid grid =
        options.grid.value_or(source.Grid().value_or(
            LatLongGrid{-180.0, 90.0, 360.0 / source.Width(),
                        180.0 / source.Height()}));
    if (!(grid.degreesAcross > 0.0) || !(grid.degreesDown > 0.0)) {
        throw std::invalid_argument(
            "A latitude-longitude grid needs positive spacing");
    }
    const auto rasterWidth =
        static_cast<std::int64_t>(source.Width());
    const auto rasterHeight =
        static_cast<std::int64_t>(source.Height());
    const double spanned = grid.degreesAcross * source.Width();
    const bool wraps = spanned >= 360.0 * (1.0 - 1e-9);
    const double degrees = 180.0 / std::numbers::pi;

    // The tile's points and a border of one around them (for slopes'
    // central differences), as raster pixel coordinates (NaN off the
    // raster) and their base points on the ellipsoid.
    const std::int64_t side = std::int64_t{cells} + 3;
    const std::size_t points = static_cast<std::size_t>(side * side);
    std::vector<glm::dvec2> pixels(points);
    std::vector<glm::dvec3> bases(points);
    // As CubeSphereLod::VertexDirection, so shared points agree
    // exactly between tiles and depths.
    const double scale = 2.0 / static_cast<double>(faceCells);
    double minX = std::numeric_limits<double>::infinity();
    double maxX = -minX;
    double minY = minX;
    double maxY = -minX;
    for (std::int64_t j = 0; j < side; ++j) {
        for (std::int64_t i = 0; i < side; ++i) {
            const glm::dvec3 d = CubeSphereLod::FaceDirection(
                face,
                static_cast<double>(std::int64_t{column} + i - 1) *
                        scale -
                    1.0,
                static_cast<double>(std::int64_t{row} + j - 1) *
                        scale -
                    1.0);
            const std::size_t at =
                static_cast<std::size_t>(j * side + i);
            bases[at] =
                d / std::sqrt(glm::dot(d * d, 1.0 / (radii * radii)));
            const double latitude =
                std::asin(std::clamp(d.z, -1.0, 1.0)) * degrees;
            double east = std::fmod(
                std::atan2(d.y, d.x) * degrees - grid.west, 360.0);
            east += east < 0.0 ? 360.0 : 0.0;
            const double south =
                grid.north - grid.degreesDown * source.Height();
            if ((!wraps && east > spanned) || latitude > grid.north ||
                latitude < south) {
                pixels[at] = glm::dvec2(
                    std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            const glm::dvec2 p(
                east / grid.degreesAcross - 0.5,
                (grid.north - latitude) / grid.degreesDown - 0.5);
            pixels[at] = p;
            minX = std::min(minX, p.x);
            maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y);
            maxY = std::max(maxY, p.y);
        }
    }

    // The window under the tile, widened to whole cells of the
    // slopes' box filter (about the grid's spacing at the face's
    // centre) and a sample around: whole rows across the seam or
    // around a pole.
    const double spacing = 90.0 / faceCells;
    const auto factor = static_cast<std::int64_t>(std::max(
        1.0, std::floor(spacing / std::max(grid.degreesAcross,
                                           grid.degreesDown))));
    const auto down = [](std::int64_t value, std::int64_t step) {
        return (value >= 0 ? value : value - step + 1) / step * step;
    };
    HeightWindow window;
    if (minX > maxX) {
        window.width = 1;
        window.height = 1;
        window.values.assign(1,
                             std::numeric_limits<float>::quiet_NaN());
    } else {
        window.y0 = std::max<std::int64_t>(
            down(static_cast<std::int64_t>(std::floor(minY)) - 1 -
                     factor,
                 factor),
            0);
        const std::int64_t y1 = std::min<std::int64_t>(
            static_cast<std::int64_t>(std::ceil(maxY)) + 2 + factor,
            rasterHeight);
        window.height = y1 - window.y0;
        window.full =
            wraps &&
            maxX - minX > 0.5 * static_cast<double>(rasterWidth);
        if (window.full) {
            window.x0 = 0;
            window.width = rasterWidth;
        } else {
            window.x0 =
                down(static_cast<std::int64_t>(std::floor(minX)) - 1 -
                         factor,
                     factor);
            std::int64_t x1 =
                static_cast<std::int64_t>(std::ceil(maxX)) + 2 +
                factor;
            if (!wraps) {
                window.x0 = std::max<std::int64_t>(window.x0, 0);
                x1 = std::min(x1, rasterWidth);
            }
            window.width = x1 - window.x0;
        }
        const std::uint32_t pixel =
            source.Channels() * source.SampleBytes();
        const RasterSampleType type = source.Type();
        const double sampleScale = source.Scale() * options.unit;
        const double sampleOffset =
            source.Offset() * options.unit - options.reference;
        const std::optional<double> noData = source.NoData();
        window.values.resize(
            static_cast<std::size_t>(window.width * window.height));
        // Columns in the raster: one span, or two across the seam.
        std::vector<std::pair<std::int64_t, std::int64_t>> spans;
        const std::int64_t start =
            ((window.x0 % rasterWidth) + rasterWidth) % rasterWidth;
        if (start + window.width <= rasterWidth) {
            spans.emplace_back(start, window.width);
        } else {
            spans.emplace_back(start, rasterWidth - start);
            spans.emplace_back(0,
                               window.width - (rasterWidth - start));
        }
        std::vector<std::byte> samples;
        std::int64_t into = 0;
        for (const auto& [first, count] : spans) {
            samples.resize(
                static_cast<std::size_t>(count * window.height) *
                pixel);
            {
                std::unique_lock<std::mutex> lock;
                if (reading != nullptr) {
                    lock = std::unique_lock(*reading);
                }
                source.Read(static_cast<std::uint32_t>(first),
                            static_cast<std::uint32_t>(window.y0),
                            static_cast<std::uint32_t>(count),
                            static_cast<std::uint32_t>(window.height),
                            samples);
            }
            for (std::int64_t y = 0; y < window.height; ++y) {
                for (std::int64_t x = 0; x < count; ++x) {
                    const double sample = SampleOf(
                        samples.data() +
                            static_cast<std::size_t>(y * count + x) *
                                pixel,
                        type);
                    window.values[static_cast<std::size_t>(
                        y * window.width + into + x)] =
                        !std::isfinite(sample) || sample < -1e30 ||
                                (noData && sample == *noData)
                            ? std::numeric_limits<float>::quiet_NaN()
                            : static_cast<float>(sample *
                                                     sampleScale +
                                                 sampleOffset);
                }
            }
            into += count;
        }
    }

    // The window box-filtered by `factor`, cells aligned with the
    // window's start.
    const std::int64_t filteredWidth =
        (window.width + factor - 1) / factor;
    const std::int64_t filteredHeight =
        (window.height + factor - 1) / factor;
    std::vector<float> filtered(
        static_cast<std::size_t>(filteredWidth * filteredHeight));
    for (std::int64_t fy = 0; fy < filteredHeight; ++fy) {
        for (std::int64_t fx = 0; fx < filteredWidth; ++fx) {
            double sum = 0.0;
            std::int64_t count = 0;
            for (std::int64_t y = fy * factor;
                 y < std::min((fy + 1) * factor, window.height);
                 ++y) {
                for (std::int64_t x = fx * factor;
                     x < std::min((fx + 1) * factor, window.width);
                     ++x) {
                    const float h = window.At(x, y);
                    if (!std::isnan(h)) {
                        sum += static_cast<double>(h);
                        ++count;
                    }
                }
            }
            filtered[static_cast<std::size_t>(fy * filteredWidth +
                                              fx)] =
                count > 0 ? static_cast<float>(
                                sum / static_cast<double>(count))
                          : std::numeric_limits<float>::quiet_NaN();
        }
    }
    const auto raw = [&](const glm::dvec2& p) {
        if (std::isnan(p.x)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return Bilinear(p.x - static_cast<double>(window.x0),
                        p.y - static_cast<double>(window.y0),
                        [&](std::int64_t x, std::int64_t y) {
                            return window.At(x, y);
                        });
    };
    const auto smooth = [&](const glm::dvec2& p) {
        if (std::isnan(p.x)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        const double half = 0.5 * static_cast<double>(factor - 1);
        return Bilinear(
            (p.x - static_cast<double>(window.x0) - half) /
                static_cast<double>(factor),
            (p.y - static_cast<double>(window.y0) - half) /
                static_cast<double>(factor),
            [&](std::int64_t x, std::int64_t y) {
                x = window.full
                        ? ((x % filteredWidth) + filteredWidth) %
                              filteredWidth
                        : std::clamp<std::int64_t>(x, 0,
                                                   filteredWidth - 1);
                y = std::clamp<std::int64_t>(y, 0,
                                             filteredHeight - 1);
                return filtered[static_cast<std::size_t>(
                    y * filteredWidth + x)];
            });
    };

    PlanetTile tile;
    const std::size_t corners = std::size_t{cells} + 1;
    tile.heights.resize(corners * corners);
    tile.slopes.resize(corners * corners);
    std::vector<float> smoothed(points);
    for (std::size_t i = 0; i < points; ++i) {
        const float h = smooth(pixels[i]);
        smoothed[i] = std::isnan(h) ? 0.0f : h;
    }
    const auto index = [side](std::int64_t i, std::int64_t j) {
        return static_cast<std::size_t>((j + 1) * side + i + 1);
    };
    for (std::int64_t j = 0; j <= cells; ++j) {
        for (std::int64_t i = 0; i <= cells; ++i) {
            const float h = raw(pixels[index(i, j)]);
            const std::size_t at =
                static_cast<std::size_t>(j) * corners +
                static_cast<std::size_t>(i);
            tile.heights[at] = std::isnan(h) ? 0.0f : h;
            const auto slope = [&](std::size_t a, std::size_t b) {
                return static_cast<float>(
                    static_cast<double>(smoothed[b] - smoothed[a]) /
                    glm::distance(bases[a], bases[b]));
            };
            tile.slopes[at] =
                glm::vec2(slope(index(i - 1, j), index(i + 1, j)),
                          slope(index(i, j - 1), index(i, j + 1)));
        }
    }
    if (!ranges) {
        return tile;
    }
    // Each quad's extremes: its corners' and those of the raster's
    // samples around the pixels its corners span, which bound the
    // bilinear surface between them (whole rows where they straddle
    // the seam).
    tile.ranges.resize(std::size_t{cells} * cells);
    for (std::int64_t j = 0; j < cells; ++j) {
        for (std::int64_t i = 0; i < cells; ++i) {
            glm::vec2 range(std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity());
            const auto take = [&range](float h) {
                if (!std::isnan(h)) {
                    range = glm::vec2(std::min(range.x, h),
                                      std::max(range.y, h));
                }
            };
            glm::dvec2 low(std::numeric_limits<double>::infinity());
            glm::dvec2 high(-std::numeric_limits<double>::infinity());
            for (const auto& [di, dj] :
                 {std::pair{0, 0}, std::pair{1, 0}, std::pair{0, 1},
                  std::pair{1, 1}}) {
                const glm::dvec2& p = pixels[index(i + di, j + dj)];
                take(tile.heights[static_cast<std::size_t>(j + dj) *
                                      corners +
                                  static_cast<std::size_t>(i + di)]);
                if (!std::isnan(p.x)) {
                    low = glm::min(low, p);
                    high = glm::max(high, p);
                }
            }
            if (low.x <= high.x) {
                const bool across =
                    high.x - low.x >
                    0.5 * static_cast<double>(rasterWidth);
                const std::int64_t x0 =
                    across ? 0
                           : static_cast<std::int64_t>(
                                 std::floor(low.x)) -
                                 window.x0;
                const std::int64_t x1 =
                    across ? window.width - 1
                           : static_cast<std::int64_t>(
                                 std::ceil(high.x)) -
                                 window.x0;
                const std::int64_t y0 =
                    static_cast<std::int64_t>(std::floor(low.y)) -
                    window.y0;
                const std::int64_t y1 =
                    static_cast<std::int64_t>(std::ceil(high.y)) -
                    window.y0;
                for (std::int64_t y = y0; y <= y1; ++y) {
                    for (std::int64_t x = x0; x <= x1; ++x) {
                        take(window.At(x, y));
                    }
                }
            }
            tile.ranges[static_cast<std::size_t>(j) * cells +
                        static_cast<std::size_t>(i)] =
                range.x <= range.y ? range : glm::vec2(0.0f);
        }
    }
    return tile;
}

PlanetTile PlanetFacesOf(const RasterSource& source,
                         const PlanetHeightsOptions& options,
                         const glm::dvec3& radii,
                         std::uint32_t faceCells) {
    ERNIE_ZONE("Planet faces");
    const std::uint32_t tileCells = std::min(faceCells, 256u);
    if (tileCells == 0 || faceCells % tileCells != 0) {
        throw std::invalid_argument(
            "Planet faces need a whole number of 256-quad tiles");
    }
    const std::uint32_t across = faceCells / tileCells;
    const std::size_t corners = std::size_t{faceCells} + 1;
    PlanetTile faces;
    faces.heights.resize(CubeSphereLod::kFaces * corners * corners);
    faces.slopes.resize(faces.heights.size());
    faces.ranges.resize(CubeSphereLod::kFaces *
                        std::size_t{faceCells} * faceCells);
    const std::uint32_t tiles =
        CubeSphereLod::kFaces * across * across;
    std::atomic<std::uint32_t> next = 0;
    std::mutex reading;
    std::mutex failing;
    std::exception_ptr failure;
    const auto work = [&] {
        for (std::uint32_t t = next++; t < tiles; t = next++) {
            const std::uint32_t face = t / (across * across);
            const std::uint32_t tx = t % across * tileCells;
            const std::uint32_t ty = t / across % across * tileCells;
            try {
                const PlanetTile tile = PlanetTileOf(
                    source, options, radii, faceCells, face, tx, ty,
                    tileCells, true, &reading);
                for (std::uint32_t j = 0; j <= tileCells; ++j) {
                    std::copy_n(
                        tile.heights.begin() + j * (tileCells + 1),
                        tileCells + 1,
                        faces.heights.begin() +
                            static_cast<std::ptrdiff_t>(
                                (face * corners + ty + j) * corners +
                                tx));
                    std::copy_n(
                        tile.slopes.begin() + j * (tileCells + 1),
                        tileCells + 1,
                        faces.slopes.begin() +
                            static_cast<std::ptrdiff_t>(
                                (face * corners + ty + j) * corners +
                                tx));
                }
                for (std::uint32_t j = 0; j < tileCells; ++j) {
                    std::copy_n(
                        tile.ranges.begin() + j * tileCells,
                        tileCells,
                        faces.ranges.begin() +
                            static_cast<std::ptrdiff_t>(
                                (std::size_t{face} * faceCells + ty +
                                 j) *
                                    faceCells +
                                tx));
                }
            } catch (...) {
                const std::scoped_lock lock(failing);
                failure = std::current_exception();
            }
        }
    };
    {
        std::vector<std::thread> threads;
        for (std::uint32_t i = 1;
             i <
             std::clamp(std::thread::hardware_concurrency(), 1u, 16u);
             ++i) {
            threads.emplace_back(work);
        }
        work();
        for (std::thread& thread : threads) {
            thread.join();
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    return faces;
}

}  // namespace ernie
