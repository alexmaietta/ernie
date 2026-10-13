// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/subdivision.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

using ernie::MeshData;
using ernie::Subdivide;
using ernie::SubdivisionScheme;
using ernie::SubdivisionSurface;

namespace {

// A cube from -1 to 1, its faces wound counterclockwise from outside.
const std::vector<float> kCubePoints = {-1, -1, -1, 1,  -1, -1, 1, 1,
                                        -1, -1, 1,  -1, -1, -1, 1, 1,
                                        -1, 1,  1,  1,  1,  -1, 1, 1};
const std::vector<std::int64_t> kCubeCounts = {4, 4, 4, 4, 4, 4};
const std::vector<std::int64_t> kCubeIndices = {
    0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4,
    2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3};

SubdivisionSurface Cube() {
    SubdivisionSurface surface;
    surface.points = kCubePoints;
    surface.faceVertexCounts = kCubeCounts;
    surface.faceVertexIndices = kCubeIndices;
    return surface;
}

/// How many triangles use each edge.
std::map<std::pair<std::uint32_t, std::uint32_t>, int>
EdgeUses(const MeshData& data) {
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> uses;
    for (std::size_t t = 0; t < data.indices.size(); t += 3) {
        for (std::size_t k = 0; k < 3; ++k) {
            const std::uint32_t a = data.indices[t + k];
            const std::uint32_t b = data.indices[t + (k + 1) % 3];
            ++uses[{std::min(a, b), std::max(a, b)}];
        }
    }
    return uses;
}

TEST(SubdivisionTest, ACatmullClarkCubeIsWatertightAndSmooth) {
    for (const int level : {1, 2}) {
        const MeshData data = Subdivide(Cube(), level);
        const std::size_t rate = std::size_t{1} << level;
        // Shared corners and edge points, and each face's own
        // interior.
        EXPECT_EQ(data.points.size() / 3,
                  8 + 12 * (rate - 1) + 6 * (rate - 1) * (rate - 1));
        EXPECT_EQ(data.indices.size() / 3, 6 * rate * rate * 2);
        // Closed: every edge between exactly two triangles.
        for (const auto& [edge, uses] : EdgeUses(data)) {
            EXPECT_EQ(uses, 2);
        }
        std::array<double, 3> centroid{};
        for (std::size_t p = 0; p < data.points.size(); p += 3) {
            const float* x = &data.points[p];
            const float* n = &data.normals[p];
            // Inside the cage, and normals facing out.
            for (int k = 0; k < 3; ++k) {
                EXPECT_LT(std::abs(x[k]), 1.0f);
                centroid[static_cast<std::size_t>(k)] +=
                    static_cast<double>(x[k]);
            }
            EXPECT_NEAR(n[0] * n[0] + n[1] * n[1] + n[2] * n[2], 1.0f,
                        1e-5f);
            EXPECT_GT(x[0] * n[0] + x[1] * n[1] + x[2] * n[2], 0.0f);
        }
        for (const double c : centroid) {
            EXPECT_NEAR(c, 0.0, 1e-4);
        }
        // Triangles wound counterclockwise from outside, as the cage.
        for (std::size_t t = 0; t < data.indices.size(); t += 3) {
            const float* a = &data.points[3 * data.indices[t]];
            const float* b = &data.points[3 * data.indices[t + 1]];
            const float* c = &data.points[3 * data.indices[t + 2]];
            const float u[3] = {b[0] - a[0], b[1] - a[1],
                                b[2] - a[2]};
            const float v[3] = {c[0] - a[0], c[1] - a[1],
                                c[2] - a[2]};
            const float n[3] = {u[1] * v[2] - u[2] * v[1],
                                u[2] * v[0] - u[0] * v[2],
                                u[0] * v[1] - u[1] * v[0]};
            EXPECT_GT(n[0] * (a[0] + b[0] + c[0]) +
                          n[1] * (a[1] + b[1] + c[1]) +
                          n[2] * (a[2] + b[2] + c[2]),
                      0.0f);
        }
    }
}

TEST(SubdivisionTest, InfinitelySharpCreasesKeepTheCube) {
    // Each of the cube's 12 edges a crease of its own.
    std::vector<std::int64_t> creases;
    for (std::int64_t a = 0; a < 8; ++a) {
        for (std::int64_t b = a + 1; b < 8; ++b) {
            int same = 0;
            for (std::size_t k = 0; k < 3; ++k) {
                same +=
                    kCubePoints[3 * static_cast<std::size_t>(a) +
                                k] ==
                    kCubePoints[3 * static_cast<std::size_t>(b) + k];
            }
            if (same == 2) {
                creases.insert(creases.end(), {a, b});
            }
        }
    }
    ASSERT_EQ(creases.size(), 24u);
    const std::vector<std::int64_t> lengths(12, 2);
    const std::vector<float> sharpnesses(12, 10.0f);
    SubdivisionSurface surface = Cube();
    surface.creaseIndices = creases;
    surface.creaseLengths = lengths;
    surface.creaseSharpnesses = sharpnesses;
    const MeshData data = Subdivide(surface, 2);
    // Points on the sharp edges are each face's own, so the faces
    // keep flat normals.
    EXPECT_EQ(data.points.size() / 3, 6 * 25u);
    for (std::size_t p = 0; p < data.points.size(); p += 3) {
        const float* x = &data.points[p];
        const float* n = &data.normals[p];
        EXPECT_NEAR(std::max({std::abs(x[0]), std::abs(x[1]),
                              std::abs(x[2])}),
                    1.0f, 1e-5f);
        EXPECT_NEAR(std::max({std::abs(n[0]), std::abs(n[1]),
                              std::abs(n[2])}),
                    1.0f, 1e-5f);
    }
}

TEST(SubdivisionTest, BilinearSurfacesInterpolateTextureCoordinates) {
    const std::vector<float> points = {0, 0, 0, 2, 0, 0,
                                       2, 1, 0, 0, 1, 0};
    const std::vector<std::int64_t> counts = {4};
    const std::vector<std::int64_t> indices = {0, 1, 2, 3};
    const std::vector<float> uvs = {0, 0, 1, 0, 1, 1, 0, 1};
    SubdivisionSurface surface;
    surface.scheme = SubdivisionScheme::Bilinear;
    surface.points = points;
    surface.faceVertexCounts = counts;
    surface.faceVertexIndices = indices;
    surface.cornerUvs = uvs;
    const MeshData data = Subdivide(surface, 2);
    EXPECT_EQ(data.points.size() / 3, 25u);
    ASSERT_EQ(data.uvs.size(), 2 * data.indices.size());
    for (std::size_t c = 0; c < data.indices.size(); ++c) {
        const float* x = &data.points[3 * data.indices[c]];
        EXPECT_NEAR(x[2], 0.0f, 1e-6f);
        EXPECT_NEAR(data.uvs[2 * c], x[0] / 2.0f, 1e-5f);
        EXPECT_NEAR(data.uvs[2 * c + 1], x[1], 1e-5f);
    }
}

// The coarser levels' triangles index the finest points, their errors
// growing as they coarsen.
TEST(SubdivisionTest, CoarserLevelsComeAsLevelsOfDetail) {
    EXPECT_TRUE(Subdivide(Cube(), 0).levels.empty());
    const MeshData data = Subdivide(Cube(), 3);
    ASSERT_EQ(data.levels.size(), 3u);
    std::size_t rate = 4;
    for (std::size_t l = 0; l < data.levels.size(); ++l, rate /= 2) {
        EXPECT_EQ(data.levels[l].count, 6 * rate * rate * 2 * 3);
        EXPECT_GT(data.levels[l].error, 0.0f);
        if (l > 0) {
            EXPECT_GT(data.levels[l].error, data.levels[l - 1].error);
        }
    }
    EXPECT_EQ(data.levels.back().first + data.levels.back().count,
              data.lodIndices.size());
    for (const std::uint32_t index : data.lodIndices) {
        EXPECT_LT(index, data.points.size() / 3);
    }
    // The coarsest level's points are the limit surface's corners.
    for (std::size_t c = data.levels.back().first;
         c < data.lodIndices.size(); ++c) {
        const float* x = &data.points[3 * data.lodIndices[c]];
        EXPECT_NEAR(std::abs(x[0]), std::abs(x[1]), 1e-5f);
        EXPECT_NEAR(std::abs(x[1]), std::abs(x[2]), 1e-5f);
    }

    // A flat pentagon's coarser levels lie on it, their texture
    // coordinates evaluated with the points.
    const std::vector<float> points = {0, 0, 0, 2, 0, 0, 2, 1,
                                       0, 1, 2, 0, 0, 1, 0};
    const std::vector<std::int64_t> counts = {5};
    const std::vector<std::int64_t> indices = {0, 1, 2, 3, 4};
    const std::vector<float> uvs = {0,    0,    1, 0, 1,
                                    0.5f, 0.5f, 1, 0, 0.5f};
    SubdivisionSurface surface;
    surface.scheme = SubdivisionScheme::Bilinear;
    surface.points = points;
    surface.faceVertexCounts = counts;
    surface.faceVertexIndices = indices;
    surface.cornerUvs = uvs;
    const MeshData flat = Subdivide(surface, 2);
    ASSERT_EQ(flat.levels.size(), 2u);
    ASSERT_EQ(flat.lodUvs.size(), 2 * flat.lodIndices.size());
    for (const MeshData::Level& level : flat.levels) {
        EXPECT_GT(level.count, 0u);
        EXPECT_LT(level.error, 1e-5f);
    }
    for (std::size_t c = 0; c < flat.lodIndices.size(); ++c) {
        const float* x = &flat.points[3 * flat.lodIndices[c]];
        EXPECT_NEAR(flat.lodUvs[2 * c], x[0] / 2.0f, 1e-5f);
        EXPECT_NEAR(flat.lodUvs[2 * c + 1], x[1] / 2.0f, 1e-5f);
    }
}

// The stencils give the refined points from the control points: from
// the cube's as it is, and from it moved, as Subdivide would.
TEST(SubdivisionTest, StencilsRefineMovedControlPoints) {
    ernie::RefinementStencils stencils;
    const MeshData data = Subdivide(Cube(), 2, stencils);
    ASSERT_EQ(stencils.starts.size(), data.points.size() / 3 + 1);
    std::vector<float> moved = kCubePoints;
    for (std::size_t i = 0; i < moved.size(); i += 3) {
        moved[i] = 2.0f * moved[i] + 0.5f;
        moved[i + 2] += 0.25f * moved[i + 1] * moved[i + 1];
    }
    SubdivisionSurface surface = Cube();
    surface.points = moved;
    const MeshData expected = Subdivide(surface, 2);
    using Case = std::pair<const std::vector<float>*,
                           const std::vector<float>*>;
    for (const auto& [points, control] :
         {Case(&data.points, &kCubePoints),
          Case(&expected.points, &moved)}) {
        for (std::size_t p = 0; p + 1 < stencils.starts.size(); ++p) {
            float sum[3] = {0.0f, 0.0f, 0.0f};
            for (std::uint32_t k = stencils.starts[p];
                 k < stencils.starts[p + 1]; ++k) {
                for (std::size_t c = 0; c < 3; ++c) {
                    sum[c] += stencils.weights[k] *
                              (*control)[3 * stencils.indices[k] + c];
                }
            }
            for (std::size_t c = 0; c < 3; ++c) {
                EXPECT_NEAR(sum[c], (*points)[3 * p + c], 1e-4f) << p;
            }
        }
    }
}

TEST(SubdivisionTest, HolesLeaveTheirFacesOut) {
    const std::vector<std::int64_t> holes = {0};
    SubdivisionSurface surface = Cube();
    surface.holeIndices = holes;
    EXPECT_EQ(Subdivide(surface, 1).indices.size() / 3, 5 * 4 * 2u);
}

TEST(SubdivisionTest, LoopSurfacesTakeTriangles) {
    const std::vector<float> points = {1,  1, 1,  -1, -1, 1,
                                       -1, 1, -1, 1,  -1, -1};
    const std::vector<std::int64_t> counts = {3, 3, 3, 3};
    const std::vector<std::int64_t> indices = {0, 1, 2, 0, 3, 1,
                                               0, 2, 3, 1, 3, 2};
    SubdivisionSurface surface;
    surface.scheme = SubdivisionScheme::Loop;
    surface.points = points;
    surface.faceVertexCounts = counts;
    surface.faceVertexIndices = indices;
    const MeshData data = Subdivide(surface, 2);
    EXPECT_EQ(data.indices.size() / 3, 4 * 16u);
    for (const auto& [edge, uses] : EdgeUses(data)) {
        EXPECT_EQ(uses, 2);
    }
    SubdivisionSurface quads = Cube();
    quads.scheme = SubdivisionScheme::Loop;
    EXPECT_THROW(Subdivide(quads, 1), std::invalid_argument);
}

TEST(SubdivisionTest, ColorsAreEvaluatedOnTheSurface) {
    std::vector<float> colors;
    for (int p = 0; p < 8; ++p) {
        colors.insert(colors.end(), {0.25f, 0.5f, 0.75f});
    }
    SubdivisionSurface surface = Cube();
    surface.colors = colors;
    const MeshData data = Subdivide(surface, 1);
    ASSERT_EQ(data.colors.size(), data.points.size());
    for (std::size_t p = 0; p < data.colors.size(); p += 3) {
        EXPECT_NEAR(data.colors[p], 0.25f, 1e-5f);
        EXPECT_NEAR(data.colors[p + 1], 0.5f, 1e-5f);
        EXPECT_NEAR(data.colors[p + 2], 0.75f, 1e-5f);
    }
}

TEST(SubdivisionTest, BadInputsThrow) {
    SubdivisionSurface none = Cube();
    none.scheme = SubdivisionScheme::None;
    EXPECT_THROW(Subdivide(none, 1), std::invalid_argument);
    EXPECT_THROW(Subdivide(Cube(), 9), std::invalid_argument);
    const std::vector<std::int64_t> outOfRange = {
        0, 3, 2, 8, 4, 5, 6, 7, 0, 1, 5, 4,
        2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3};
    SubdivisionSurface bad = Cube();
    bad.faceVertexIndices = outOfRange;
    EXPECT_THROW(Subdivide(bad, 1), std::invalid_argument);
    const std::vector<float> uvs = {0, 0};
    SubdivisionSurface shortUvs = Cube();
    shortUvs.cornerUvs = uvs;
    EXPECT_THROW(Subdivide(shortUvs, 1), std::invalid_argument);
}

}  // namespace
