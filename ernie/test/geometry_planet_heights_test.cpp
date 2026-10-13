// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/geometry/planet_heights.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

using ernie::CubeSphereLod;
using ernie::LatLongGrid;
using ernie::PlanetHeightsOf;
using ernie::PlanetHeightsOptions;
using ernie::RasterSampleType;

namespace {

constexpr double kDegrees = 180.0 / std::numbers::pi;

/// A one-channel raster in memory, of floats or 16-bit integers.
class MemoryRaster : public ernie::RasterSource {
  public:
    MemoryRaster(std::uint32_t width, std::uint32_t height,
                 RasterSampleType type)
        : width_(width), height_(height), type_(type),
          bytes_(std::size_t{width} * height * SampleBytesOf(type)) {}

    std::uint32_t Width() const override {
        return width_;
    }
    std::uint32_t Height() const override {
        return height_;
    }
    std::uint32_t Channels() const override {
        return 1;
    }
    RasterSampleType Type() const override {
        return type_;
    }
    double Scale() const override {
        return scale;
    }
    double Offset() const override {
        return offset;
    }
    std::optional<double> NoData() const override {
        return noData;
    }
    std::optional<LatLongGrid> Grid() const override {
        return grid;
    }

    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w,
              std::uint32_t h,
              std::span<std::byte> out) const override {
        const std::size_t bytes = SampleBytes();
        for (std::uint32_t row = 0; row < h; ++row) {
            std::memcpy(out.data() + std::size_t{row} * w * bytes,
                        bytes_.data() +
                            (std::size_t{y + row} * width_ + x) *
                                bytes,
                        w * bytes);
        }
    }

    /// Sets every sample to `value(latitude, longitude)` of its
    /// centre, on the whole body from 180 degrees west.
    void Fill(const std::function<double(double, double)>& value) {
        for (std::uint32_t row = 0; row < height_; ++row) {
            for (std::uint32_t col = 0; col < width_; ++col) {
                const double latitude =
                    90.0 - (row + 0.5) * 180.0 / height_;
                const double longitude =
                    -180.0 + (col + 0.5) * 360.0 / width_;
                Set(col, row, value(latitude, longitude));
            }
        }
    }

    void Set(std::uint32_t col, std::uint32_t row, double value) {
        std::byte* at =
            bytes_.data() +
            (std::size_t{row} * width_ + col) * SampleBytes();
        if (type_ == RasterSampleType::Float32) {
            const auto f = static_cast<float>(value);
            std::memcpy(at, &f, sizeof(f));
        } else {
            const auto i = static_cast<std::int16_t>(value);
            std::memcpy(at, &i, sizeof(i));
        }
    }

    double scale = 1.0;
    double offset = 0.0;
    std::optional<double> noData;
    std::optional<LatLongGrid> grid;

  private:
    std::uint32_t width_;
    std::uint32_t height_;
    RasterSampleType type_;
    std::vector<std::byte> bytes_;
};

/// Each vertex's latitude and longitude (degrees), face by face, row
/// by row.
template <typename F>
void ForEachVertex(std::uint32_t faceCells, F&& f) {
    const double scale = 2.0 / faceCells;
    std::size_t i = 0;
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces;
         ++face) {
        for (std::uint32_t row = 0; row <= faceCells; ++row) {
            for (std::uint32_t col = 0; col <= faceCells; ++col) {
                const glm::dvec3 d = CubeSphereLod::FaceDirection(
                    face, col * scale - 1.0, row * scale - 1.0);
                f(i++, std::asin(d.z) * kDegrees,
                  std::atan2(d.y, d.x) * kDegrees);
            }
        }
    }
}

}  // namespace

// A smooth field of kilometres, four samples a degree, filtered to
// the faces' 5.6 degrees and sampled at every vertex: within a few
// metres of the field there, across longitude's seam and at the
// poles.
TEST(PlanetHeights, ResampleASmoothFieldOntoTheFaces) {
    MemoryRaster raster(1440, 720, RasterSampleType::Float32);
    const auto field = [](double latitude, double longitude) {
        return std::sin(latitude / kDegrees) +
               0.5 * std::cos(longitude / kDegrees);
    };
    raster.Fill(field);
    PlanetHeightsOptions options;
    options.unit = 1000.0;
    const std::vector<float> heights =
        PlanetHeightsOf(raster, 16, options);
    ASSERT_EQ(heights.size(), 6u * 17u * 17u);
    ForEachVertex(
        16, [&](std::size_t i, double latitude, double longitude) {
            EXPECT_NEAR(heights[i],
                        1000.0 * field(latitude, longitude), 8.0)
                << "vertex " << i << " at " << latitude << ", "
                << longitude;
        });
}

// LOLA's PDS LDEMs: 16-bit samples of half metres, offset by the
// Moon's radius, the grid from 0 degrees east. Their reference
// subtracted they are heights, the samples without data left out.
TEST(PlanetHeights, RadiiLessTheReferenceAreHeights) {
    MemoryRaster raster(360, 180, RasterSampleType::Int16);
    raster.scale = 0.5;
    raster.offset = 1737400.0;
    raster.noData = -32768.0;
    raster.grid = LatLongGrid{0.0, 90.0, 1.0, 1.0};
    // The eastern hemisphere 100 m high, the western without data;
    // vertices over a filtered cell (11 degrees) from their edges.
    for (std::uint32_t row = 0; row < 180; ++row) {
        for (std::uint32_t col = 0; col < 360; ++col) {
            raster.Set(col, row, col < 180 ? 200.0 : -32768.0);
        }
    }
    PlanetHeightsOptions options;
    options.reference = 1737400.0;
    const std::vector<float> heights =
        PlanetHeightsOf(raster, 8, options);
    std::size_t east = 0;
    std::size_t west = 0;
    ForEachVertex(
        8, [&](std::size_t i, double latitude, double longitude) {
            if (std::abs(latitude) < 80.0 && longitude > 25.0 &&
                longitude < 155.0) {
                EXPECT_FLOAT_EQ(heights[i], 100.0f) << "vertex " << i;
                ++east;
            } else if (std::abs(latitude) < 80.0 &&
                       longitude < -25.0 && longitude > -155.0) {
                EXPECT_FLOAT_EQ(heights[i], 0.0f) << "vertex " << i;
                ++west;
            }
        });
    EXPECT_GT(east, 20u);
    EXPECT_GT(west, 20u);
}

// A raster of part of the body: its heights where it lies, 0 off it.
TEST(PlanetHeights, VerticesOffAPartialRasterAreAtZero) {
    MemoryRaster raster(90, 45, RasterSampleType::Float32);
    raster.Fill([](double, double) { return 5.0; });
    PlanetHeightsOptions options;
    options.grid = LatLongGrid{0.0, 45.0, 1.0, 1.0};
    const std::vector<float> heights =
        PlanetHeightsOf(raster, 16, options);
    ForEachVertex(
        16, [&](std::size_t i, double latitude, double longitude) {
            const bool on = latitude > 0.0 && latitude < 45.0 &&
                            longitude > 0.0 && longitude < 90.0;
            const bool off = latitude < -1.0 || latitude > 46.0 ||
                             longitude < -1.0 || longitude > 91.0;
            if (on) {
                EXPECT_FLOAT_EQ(heights[i], 5.0f) << "vertex " << i;
            } else if (off) {
                EXPECT_FLOAT_EQ(heights[i], 0.0f) << "vertex " << i;
            }
        });
}

// Vertices on edges and corners shared by faces carry one height, as
// CubeSphereLod requires.
TEST(PlanetHeights, FacesAgreeWhereTheyMeet) {
    MemoryRaster raster(720, 360, RasterSampleType::Float32);
    raster.Fill([](double latitude, double longitude) {
        return std::sin(3.0 * latitude / kDegrees) *
               std::cos(5.0 * longitude / kDegrees);
    });
    const std::vector<float> heights = PlanetHeightsOf(raster, 8);
    std::vector<glm::dvec3> directions;
    const double scale = 2.0 / 8.0;
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces;
         ++face) {
        for (std::uint32_t row = 0; row <= 8; ++row) {
            for (std::uint32_t col = 0; col <= 8; ++col) {
                directions.push_back(CubeSphereLod::FaceDirection(
                    face, col * scale - 1.0, row * scale - 1.0));
            }
        }
    }
    std::size_t shared = 0;
    for (std::size_t i = 0; i < directions.size(); ++i) {
        for (std::size_t j = i + 1; j < directions.size(); ++j) {
            if (directions[i] == directions[j]) {
                EXPECT_EQ(heights[i], heights[j]);
                ++shared;
            }
        }
    }
    // A pair for each of the 12 edges' 7 inner vertices, three for
    // each of the 8 corners.
    EXPECT_EQ(shared, 12u * 7u + 8u * 3u);
}

TEST(PlanetHeights, NeedFaceCellsAndAGrid) {
    MemoryRaster raster(4, 2, RasterSampleType::Float32);
    EXPECT_THROW(PlanetHeightsOf(raster, 0), std::invalid_argument);
    raster.grid = LatLongGrid{0.0, 90.0, 0.0, 1.0};
    EXPECT_THROW(PlanetHeightsOf(raster, 4), std::invalid_argument);
}
