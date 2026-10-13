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

// A tile's heights are unfiltered, so a grid twice as fine has the
// same heights at the coarse grid's points, and neighbouring tiles
// the same along their shared edge: patches drawn from different
// levels' tiles meet without cracks.
TEST(PlanetHeights, TilesAgreeAcrossDepthsAndEdges) {
    MemoryRaster raster(720, 360, RasterSampleType::Float32);
    raster.Fill([](double latitude, double longitude) {
        return std::sin(3.0 * latitude / kDegrees) *
               std::cos(5.0 * longitude / kDegrees);
    });
    const glm::dvec3 radii(1e6);
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces;
         ++face) {
        const ernie::PlanetTile coarse = ernie::PlanetTileOf(
            raster, {}, radii, 32, face, 0, 0, 32);
        const ernie::PlanetTile fine = ernie::PlanetTileOf(
            raster, {}, radii, 64, face, 0, 0, 64);
        for (std::uint32_t j = 0; j <= 32; ++j) {
            for (std::uint32_t i = 0; i <= 32; ++i) {
                ASSERT_EQ(coarse.heights[j * 33 + i],
                          fine.heights[2 * j * 65 + 2 * i])
                    << "face " << face << " at " << i << ", " << j;
            }
        }
        const ernie::PlanetTile left = ernie::PlanetTileOf(
            raster, {}, radii, 64, face, 0, 0, 32);
        const ernie::PlanetTile right = ernie::PlanetTileOf(
            raster, {}, radii, 64, face, 32, 0, 32);
        for (std::uint32_t j = 0; j <= 32; ++j) {
            ASSERT_EQ(left.heights[j * 33 + 32],
                      right.heights[j * 33])
                << "face " << face << " row " << j;
        }
    }
    EXPECT_THROW(
        ernie::PlanetTileOf(raster, {}, radii, 64, 0, 40, 0, 32),
        std::invalid_argument);
}

// A field linear in space, h = H x, rises along the surface at H / R
// times the x axis's tangential part: the tiles' slopes along their
// grid lines are its component along each, here as on the faces whose
// lines run across the seam and around the poles.
TEST(PlanetHeights, TileSlopesFollowTheSurfaceGradient) {
    constexpr double kRadius = 1e6;
    constexpr double kRise = 1e4;
    MemoryRaster raster(1440, 720, RasterSampleType::Float32);
    raster.Fill([](double latitude, double longitude) {
        return kRise * std::cos(latitude / kDegrees) *
               std::cos(longitude / kDegrees);
    });
    const std::uint32_t cells = 32;
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces;
         ++face) {
        const ernie::PlanetTile tile =
            ernie::PlanetTileOf(raster, {}, glm::dvec3(kRadius),
                                cells, face, 0, 0, cells);
        const double scale = 2.0 / cells;
        const auto direction = [&](double i, double j) {
            return CubeSphereLod::FaceDirection(face, i * scale - 1.0,
                                                j * scale - 1.0);
        };
        for (std::uint32_t j = 1; j < cells; j += 5) {
            for (std::uint32_t i = 1; i < cells; i += 5) {
                const glm::dvec3 d = direction(i, j);
                const glm::dvec3 gradient =
                    kRise / kRadius * (glm::dvec3(1, 0, 0) - d.x * d);
                const glm::dvec3 u = glm::normalize(
                    direction(i + 1, j) - direction(i - 1, j));
                const glm::dvec3 v = glm::normalize(
                    direction(i, j + 1) - direction(i, j - 1));
                const glm::vec2 slope =
                    tile.slopes[j * (cells + 1) + i];
                EXPECT_NEAR(slope.x, glm::dot(gradient, u),
                            0.02 * kRise / kRadius)
                    << "face " << face << " at " << i << ", " << j;
                EXPECT_NEAR(slope.y, glm::dot(gradient, v),
                            0.02 * kRise / kRadius)
                    << "face " << face << " at " << i << ", " << j;
            }
        }
    }
}

// A quad's range takes in a peak between its corners, which bilinear
// heights miss; the whole planet's faces are its tiles stitched.
TEST(PlanetHeights, QuadRangesHoldWhatTheirCornersMiss) {
    MemoryRaster raster(720, 360, RasterSampleType::Float32);
    raster.Fill([](double, double) { return 0.0; });
    // A one-sample peak at about latitude 0.25, longitude 0.25:
    // inside the quad of face 0 (centred on longitude 0) right of its
    // centre.
    raster.Set(360, 179, 500.0);
    const glm::dvec3 radii(1e6);
    const ernie::PlanetTile faces =
        ernie::PlanetFacesOf(raster, {}, radii, 256);
    ASSERT_EQ(faces.heights.size(), 6u * 257u * 257u);
    ASSERT_EQ(faces.ranges.size(), 6u * 256u * 256u);
    const ernie::PlanetTile face0 = ernie::PlanetTileOf(
        raster, {}, radii, 256, 0, 0, 0, 256, true);
    for (std::size_t i = 0; i < face0.heights.size(); ++i) {
        ASSERT_EQ(faces.heights[i], face0.heights[i]) << i;
    }
    float highest = 0.0f;
    std::size_t peak = 0;
    for (std::size_t i = 0; i < 256u * 256u; ++i) {
        if (faces.ranges[i].y > highest) {
            highest = faces.ranges[i].y;
            peak = i;
        }
        EXPECT_LE(faces.ranges[i].x, 0.0f);
    }
    EXPECT_FLOAT_EQ(highest, 500.0f);
    // Its corners, bilinear between samples a quarter degree apart
    // over a quad of a third of one, stay well below.
    const std::size_t col = peak % 256;
    const std::size_t row = peak / 256;
    for (const std::size_t corner :
         {row * 257 + col, row * 257 + col + 1, (row + 1) * 257 + col,
          (row + 1) * 257 + col + 1}) {
        EXPECT_LT(faces.heights[corner], 500.0f);
    }
}
