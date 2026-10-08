#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/geometry/frustum.h"
#include "support/test_raster.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

using ernie::CubeSphereLod;
using ernie::CubeSphereRegion;
using ernie::Frustum;
using ernie::LodPatch;
using ernie::LodRange;
using ernie::RasterView;
using ernie::testing::TestRaster;

namespace {

constexpr float kMorphStart = 0.67f;
constexpr double kRadius = 100.0;

using Faces = std::array<TestRaster, CubeSphereLod::kFaces>;

/// Six face rasters sampling a smooth relief of a few units by direction, so vertices shared by
/// two faces get the same height.
Faces Relief(std::uint32_t faceCells, double amplitude = 3.0) {
    const CubeSphereLod shape(kRadius, faceCells, 4);
    Faces faces;
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces; ++face) {
        faces[face] = TestRaster(faceCells + 1, faceCells + 1);
        for (std::uint32_t row = 0; row <= faceCells; ++row) {
            for (std::uint32_t col = 0; col <= faceCells; ++col) {
                const glm::dvec3 d = shape.VertexDirection(face, col, row);
                faces[face](col, row) = static_cast<float>(
                    amplitude * (std::sin(7.0 * d.x + 1.0) * std::cos(5.0 * d.y) +
                                 0.5 * std::sin(11.0 * d.z - 2.0 * d.x)));
            }
        }
    }
    return faces;
}

CubeSphereLod Build(const Faces& faces, std::uint32_t leafCells) {
    std::array<RasterView, CubeSphereLod::kFaces> views;
    for (std::size_t i = 0; i < views.size(); ++i) {
        views[i] = faces[i].View();
    }
    return CubeSphereLod(kRadius, views, leafCells);
}

glm::dvec3 Vertex(const CubeSphereLod& lod, const Faces& faces, std::uint32_t face, glm::ivec2 c,
                  float exaggeration) {
    const double h = faces[face](static_cast<std::size_t>(c.x), static_cast<std::size_t>(c.y));
    return lod.VertexDirection(face, static_cast<std::uint32_t>(c.x),
                               static_cast<std::uint32_t>(c.y)) *
           (lod.Radius() + h * static_cast<double>(exaggeration));
}

glm::ivec2 GridIndex(const LodPatch& patch, glm::uvec2 local) {
    return glm::ivec2(patch.originX, patch.originY) + glm::ivec2(local) * (1 << patch.level);
}

/// Reference for the sphere's vertex shader: vertex `local` of `patch` after morphing its odd
/// vertices towards the midpoint of their coarser-level neighbours.
glm::dvec3 MorphedVertex(const CubeSphereLod& lod, const Faces& faces, const LodPatch& patch,
                         glm::uvec2 local, const glm::dvec3& eye, std::span<const LodRange> ranges,
                         float exaggeration) {
    const int spacing = 1 << patch.level;
    const glm::ivec2 g = GridIndex(patch, local);
    const glm::dvec3 p = Vertex(lod, faces, patch.face, g, exaggeration);
    const glm::ivec2 odd(static_cast<int>(local.x & 1u), static_cast<int>(local.y & 1u));
    if (odd == glm::ivec2(0)) {
        return p;
    }
    const glm::ivec2 step =
        odd.x != 0 && odd.y != 0 ? glm::ivec2(spacing, -spacing) : odd * spacing;
    const glm::dvec3 target = 0.5 * (Vertex(lod, faces, patch.face, g + step, exaggeration) +
                                     Vertex(lod, faces, patch.face, g - step, exaggeration));
    double k = 0.0;
    const LodRange& range = ranges[patch.level];
    if (std::isfinite(range.morphEnd)) {
        const auto start = static_cast<double>(range.morphStart);
        const auto end = static_cast<double>(range.morphEnd);
        k = std::clamp((glm::distance(eye, p) - start) / (end - start), 0.0, 1.0);
    }
    return glm::mix(p, target, k);
}

using Key = std::tuple<long long, long long, long long>;

Key DirectionKey(const glm::dvec3& d) {
    return {std::llround(d.x * 1e9), std::llround(d.y * 1e9), std::llround(d.z * 1e9)};
}

/// One side of a patch: its unmorphed vertices' keys and morphed positions, in order.
struct PatchEdge {
    std::size_t patch = 0;
    std::vector<Key> keys;
    std::vector<glm::dvec3> points;
};

std::vector<PatchEdge> Edges(const CubeSphereLod& lod, const Faces& faces,
                             std::span<const LodPatch> patches, const glm::dvec3& eye,
                             std::span<const LodRange> ranges, float exaggeration) {
    const std::uint32_t n = lod.PatchCells();
    std::vector<PatchEdge> edges;
    for (std::size_t i = 0; i < patches.size(); ++i) {
        for (int side = 0; side < 4; ++side) {
            PatchEdge edge{i, {}, {}};
            for (std::uint32_t along = 0; along <= n; ++along) {
                const glm::uvec2 local = side == 0   ? glm::uvec2(along, 0)
                                         : side == 1 ? glm::uvec2(along, n)
                                         : side == 2 ? glm::uvec2(0, along)
                                                     : glm::uvec2(n, along);
                const glm::ivec2 g = GridIndex(patches[i], local);
                edge.keys.push_back(DirectionKey(
                    lod.VertexDirection(patches[i].face, static_cast<std::uint32_t>(g.x),
                                        static_cast<std::uint32_t>(g.y))));
                edge.points.push_back(
                    MorphedVertex(lod, faces, patches[i], local, eye, ranges, exaggeration));
            }
            edges.push_back(std::move(edge));
        }
    }
    return edges;
}

double SegmentDistance(const glm::dvec3& p, const glm::dvec3& a, const glm::dvec3& b) {
    const glm::dvec3 ab = b - a;
    const double len2 = glm::dot(ab, ab);
    const double t = len2 > 0.0 ? std::clamp(glm::dot(p - a, ab) / len2, 0.0, 1.0) : 0.0;
    return glm::distance(p, a + t * ab);
}

/// Whether every point of `points[first..last]` lies on the polyline `line[from..to]`.
bool OnPolyline(const std::vector<glm::dvec3>& points, std::size_t first, std::size_t last,
                const std::vector<glm::dvec3>& line, std::size_t from, std::size_t to,
                double tolerance) {
    for (std::size_t i = first; i <= last; ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t j = from; j < to; ++j) {
            best = std::min(best, SegmentDistance(points[i], line[j], line[j + 1]));
        }
        if (from == to) {
            best = glm::distance(points[i], line[from]);
        }
        if (best > tolerance) {
            return false;
        }
    }
    return true;
}

struct SeamReport {
    int sharedEdges = 0;
    int faceEdges = 0;
    int levelChanges = 0;
    int levelJumps = 0;
    int cracks = 0;
};

/// Compares the morphed geometry of every stretch of edge shared by two patches, within a face
/// or across faces, from both sides.
SeamReport CheckSeams(const CubeSphereLod& lod, const Faces& faces,
                      std::span<const LodPatch> patches, const glm::dvec3& eye,
                      std::span<const LodRange> ranges, float exaggeration) {
    const std::vector<PatchEdge> edges = Edges(lod, faces, patches, eye, ranges, exaggeration);
    std::map<Key, std::vector<std::size_t>> byKey;
    for (std::size_t e = 0; e < edges.size(); ++e) {
        for (const Key& key : edges[e].keys) {
            byKey[key].push_back(e);
        }
    }
    std::map<std::pair<std::size_t, std::size_t>, int> shared;
    for (const auto& [key, list] : byKey) {
        for (const std::size_t a : list) {
            for (const std::size_t b : list) {
                if (a < b && edges[a].patch != edges[b].patch) {
                    ++shared[{a, b}];
                }
            }
        }
    }
    auto span = [](const PatchEdge& edge, const std::set<Key>& common) {
        std::size_t first = edge.keys.size();
        std::size_t last = 0;
        for (std::size_t i = 0; i < edge.keys.size(); ++i) {
            if (common.contains(edge.keys[i])) {
                first = std::min(first, i);
                last = std::max(last, i);
            }
        }
        return std::pair{first, last};
    };
    SeamReport report;
    const double tolerance = 1e-9 * lod.Radius();
    for (const auto& [pair, count] : shared) {
        if (count < 2) {
            continue;
        }
        const PatchEdge& a = edges[pair.first];
        const PatchEdge& b = edges[pair.second];
        const LodPatch& pa = patches[a.patch];
        const LodPatch& pb = patches[b.patch];
        ++report.sharedEdges;
        report.faceEdges += pa.face != pb.face ? 1 : 0;
        const int step = std::abs(static_cast<int>(pa.level) - static_cast<int>(pb.level));
        report.levelChanges += step == 1 ? 1 : 0;
        report.levelJumps += step > 1 ? 1 : 0;
        std::set<Key> common;
        for (const Key& key : a.keys) {
            if (std::find(b.keys.begin(), b.keys.end(), key) != b.keys.end()) {
                common.insert(key);
            }
        }
        const auto [a0, a1] = span(a, common);
        const auto [b0, b1] = span(b, common);
        const bool same = OnPolyline(a.points, a0, a1, b.points, b0, b1, tolerance) &&
                          OnPolyline(b.points, b0, b1, a.points, a0, a1, tolerance);
        report.cracks += same ? 0 : 1;
    }
    return report;
}

/// Times each quad of each face is covered by `patches`, face by face and row-major.
std::vector<int> Coverage(const CubeSphereLod& lod, std::span<const LodPatch> patches) {
    const std::size_t n = lod.FaceCells();
    std::vector<int> covered(CubeSphereLod::kFaces * n * n, 0);
    for (const LodPatch& patch : patches) {
        const std::size_t extent = static_cast<std::size_t>(lod.PatchCells()) << patch.level;
        for (std::size_t row = patch.originY; row < patch.originY + extent; ++row) {
            for (std::size_t col = patch.originX; col < patch.originX + extent; ++col) {
                ++covered[(patch.face * n + row) * n + col];
            }
        }
    }
    return covered;
}

std::vector<glm::dvec3> TestEyes() {
    const glm::dvec3 corner = glm::normalize(glm::dvec3(1.0, 1.0, 1.0));
    const glm::dvec3 edge = glm::normalize(glm::dvec3(-1.0, 0.0, 1.0));
    return {glm::dvec3(kRadius + 4.0, 0.0, 0.0),
            corner * (kRadius + 4.0),
            edge * (kRadius + 1.0),
            glm::normalize(glm::dvec3(0.3, -0.8, -0.5)) * (kRadius + 10.0),
            glm::dvec3(0.0, 0.0, -kRadius - 40.0),
            glm::dvec3(0.0, 0.0, 0.0),
            corner * (8.0 * kRadius)};
}

/// Whether the segment from `eye` to `point` passes through the sphere of radius `radius`.
bool BehindSphere(const glm::dvec3& eye, const glm::dvec3& point, double radius) {
    const glm::dvec3 d = point - eye;
    const double t = std::clamp(-glm::dot(eye, d) / glm::dot(d, d), 0.0, 1.0);
    return glm::length(eye + t * d) < radius;
}

}  // namespace

TEST(CubeSphereLod, FacesSitOnTheAxesAndWindTheSameWay) {
    const std::array<glm::dvec3, 6> centres = {
        {{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces; ++face) {
        const glm::dvec3 c = CubeSphereLod::FaceDirection(face, 0.0, 0.0);
        EXPECT_EQ(c, centres[face]) << face;
        const glm::dvec3 ds = CubeSphereLod::FaceDirection(face, 1e-3, 0.0) - c;
        const glm::dvec3 dt = CubeSphereLod::FaceDirection(face, 0.0, 1e-3) - c;
        EXPECT_LT(glm::dot(glm::cross(ds, dt), c), 0.0) << face;
    }
    for (std::uint32_t face = 0; face < 4; ++face) {
        const glm::dvec3 c = CubeSphereLod::FaceDirection(face, 0.0, 0.0);
        const glm::dvec3 east = glm::cross(glm::dvec3(0.0, 0.0, 1.0), c);
        EXPECT_GT(glm::dot(CubeSphereLod::FaceDirection(face, 0.5, 0.0) - c, east), 0.0);
        EXPECT_LT(CubeSphereLod::FaceDirection(face, 0.0, 0.5).z, 0.0) << "rows run south";
    }
}

TEST(CubeSphereLod, FacePointInvertsFaceDirection) {
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces; ++face) {
        for (const double s : {-0.99, -0.5, 0.0, 0.31, 0.9}) {
            for (const double t : {-0.8, 0.0, 0.45, 0.97}) {
                const auto [found, st] =
                    CubeSphereLod::FacePoint(3.0 * CubeSphereLod::FaceDirection(face, s, t));
                EXPECT_EQ(found, face);
                EXPECT_NEAR(st.x, s, 1e-12);
                EXPECT_NEAR(st.y, t, 1e-12);
            }
        }
    }
    EXPECT_EQ(CubeSphereLod::FacePoint(glm::dvec3(1.0, 1.0, 0.0)).first, 0u) << "ties go low";
}

TEST(CubeSphereLod, SharedEdgeAndCornerVerticesAreBitIdentical) {
    const std::uint32_t n = 16;
    const CubeSphereLod lod(kRadius, n, 4);
    std::set<std::tuple<double, double, double>> unique;
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces; ++face) {
        for (std::uint32_t row = 0; row <= n; ++row) {
            for (std::uint32_t col = 0; col <= n; ++col) {
                const glm::dvec3 d = lod.VertexDirection(face, col, row);
                EXPECT_NEAR(glm::length(d), 1.0, 1e-15);
                unique.insert({d.x, d.y, d.z});
            }
        }
    }
    EXPECT_EQ(unique.size(), 6u * n * n + 2u);
}

TEST(CubeSphereLod, CellsSpanEqualAnglesAlongTheCentreLines) {
    const std::uint32_t n = 32;
    const CubeSphereLod lod(kRadius, n, 4);
    const double expected = 0.5 * std::numbers::pi / n;
    for (std::uint32_t col = 0; col < n; ++col) {
        const double angle = std::acos(
            glm::dot(lod.VertexDirection(2, col, n / 2), lod.VertexDirection(2, col + 1, n / 2)));
        EXPECT_NEAR(angle, expected, 1e-12);
    }
}

TEST(CubeSphereLod, RejectsInvalidShapes) {
    EXPECT_THROW(CubeSphereLod(0.0, 16, 4), std::invalid_argument);
    EXPECT_THROW(CubeSphereLod(-1.0, 16, 4), std::invalid_argument);
    EXPECT_THROW(CubeSphereLod(std::numeric_limits<double>::infinity(), 16, 4),
                 std::invalid_argument);
    EXPECT_THROW(CubeSphereLod(kRadius, 24, 4), std::invalid_argument);
    EXPECT_THROW(CubeSphereLod(kRadius, 4, 8), std::invalid_argument);
    EXPECT_THROW(CubeSphereLod(kRadius, 16, 3), std::invalid_argument);
    EXPECT_THROW(CubeSphereLod(kRadius, 1u << 24, 4), std::invalid_argument);

    Faces faces = Relief(16);
    EXPECT_NO_THROW(Build(faces, 4));
    faces[3] = TestRaster(17, 16);
    EXPECT_THROW(Build(faces, 4), std::invalid_argument);
    faces = Relief(16);
    faces[5] = TestRaster(9, 9);
    EXPECT_THROW(Build(faces, 4), std::invalid_argument);
    faces.fill(TestRaster(13, 13));
    EXPECT_THROW(Build(faces, 4), std::invalid_argument);
    EXPECT_TRUE(CubeSphereLod().Empty());
}

TEST(CubeSphereLod, CountsLevelsFromTheFaceSize) {
    const CubeSphereLod lod(kRadius, 256, 16);
    EXPECT_EQ(lod.Levels(), 5u);
    EXPECT_EQ(lod.LeavesPerSide(), 16u);
    EXPECT_EQ(lod.PatchCells(), 8u);
    EXPECT_EQ(lod.FaceCells(), 256u);
    EXPECT_EQ(CubeSphereLod(kRadius, 16, 16).Levels(), 1u);
}

TEST(CubeSphereLod, RegionsContainTheirVerticesTrianglesAndMorphTargets) {
    const std::uint32_t n = 32;
    const Faces faces = Relief(n);
    const CubeSphereLod lod = Build(faces, 4);
    for (const float exaggeration : {1.0f, 6.0f}) {
        for (std::uint32_t face = 0; face < CubeSphereLod::kFaces; ++face) {
            for (std::uint32_t level = 0; level < lod.Levels(); ++level) {
                const std::uint32_t cells = 4u << level;
                const std::uint32_t side = n / cells;
                for (std::uint32_t y = 0; y < side; ++y) {
                    for (std::uint32_t x = 0; x < side; ++x) {
                        const CubeSphereRegion region =
                            lod.NodeRegion(face, level, x, y, exaggeration);
                        const auto [low, high] = region.Box();
                        for (std::uint32_t row = y * cells; row < (y + 1) * cells; ++row) {
                            for (std::uint32_t col = x * cells; col < (x + 1) * cells; ++col) {
                                const glm::ivec2 c(static_cast<int>(col), static_cast<int>(row));
                                const glm::dvec3 a = Vertex(lod, faces, face, c, exaggeration);
                                const glm::dvec3 b =
                                    Vertex(lod, faces, face, c + glm::ivec2(1, 1), exaggeration);
                                const glm::dvec3 m = 0.5 * (a + b);
                                for (const glm::dvec3& p : {a, b, m}) {
                                    EXPECT_LE(region.Distance(p), 1e-9 * kRadius);
                                    EXPECT_TRUE(glm::all(glm::lessThanEqual(low, p + 1e-9)));
                                    EXPECT_TRUE(glm::all(glm::lessThanEqual(p - 1e-9, high)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(CubeSphereLod, RegionDistanceIsExactForSimpleCases) {
    CubeSphereRegion region;
    region.axis = glm::dvec3(0.0, 0.0, 1.0);
    region.angle = 0.25 * std::numbers::pi;
    region.inner = 10.0;
    region.outer = 12.0;
    EXPECT_DOUBLE_EQ(region.Distance(glm::dvec3(0.0, 0.0, 20.0)), 8.0);
    EXPECT_DOUBLE_EQ(region.Distance(glm::dvec3(0.0, 0.0, 11.0)), 0.0);
    EXPECT_DOUBLE_EQ(region.Distance(glm::dvec3(0.0, 0.0, 4.0)), 6.0);
    EXPECT_DOUBLE_EQ(region.Distance(glm::dvec3(0.0)), 10.0);
    const double rim = std::cos(0.25 * std::numbers::pi);
    EXPECT_NEAR(region.Distance(glm::dvec3(11.0, 0.0, 0.0)), std::sqrt(121.0 + 100.0 - 220.0 * rim),
                1e-12);
    EXPECT_NEAR(region.Distance(glm::dvec3(0.0, 0.0, -5.0)), std::sqrt(25.0 + 100.0 + 100.0 * rim),
                1e-12);
    const auto [low, high] = region.Box();
    EXPECT_NEAR(high.z, 12.0, 1e-12);
    EXPECT_NEAR(low.z, 10.0 * std::cos(0.25 * std::numbers::pi), 1e-12);
    EXPECT_NEAR(high.x, 12.0 * std::sin(0.25 * std::numbers::pi), 1e-12);
    EXPECT_NEAR(low.y, -high.y, 1e-12);
}

TEST(CubeSphereLod, LeavesFromTheGpuReplaceEmptyNodes) {
    CubeSphereLod lod(kRadius, 64, 8);
    EXPECT_GT(lod.HeightRange().x, lod.HeightRange().y);
    std::vector<LodPatch> patches;
    lod.Select(glm::dvec3(2.0 * kRadius, 0.0, 0.0), lod.Ranges(10.0, kMorphStart), nullptr, 1.0f,
               true, patches);
    EXPECT_TRUE(patches.empty());
    EXPECT_EQ(lod.OccluderRadius(1.0f), 0.0);

    const std::vector<glm::vec2> leaves(64, glm::vec2(-2.0f, 3.0f));
    EXPECT_THROW(lod.SetLeaves(0, std::span(leaves).first(63)), std::invalid_argument);
    EXPECT_THROW(lod.SetLeaves(6, leaves), std::invalid_argument);
    lod.SetLeaves(0, leaves);
    std::vector<glm::vec2> raised(leaves);
    raised[5] = glm::vec2(-7.0f, 1.0f);
    lod.SetLeaves(3, raised);
    EXPECT_EQ(lod.HeightRange(), glm::vec2(-7.0f, 3.0f));

    lod.SelectAll(patches);
    EXPECT_EQ(patches.size(), 2u * 64u * 4u);
    EXPECT_TRUE(std::all_of(patches.begin(), patches.end(),
                            [](const LodPatch& p) { return p.face == 0 || p.face == 3; }));
}

TEST(CubeSphereLod, SelectAllCoversEveryFaceOnceAtFullResolution) {
    const Faces faces = Relief(32);
    const CubeSphereLod lod = Build(faces, 8);
    std::vector<LodPatch> patches;
    lod.SelectAll(patches);
    for (const LodPatch& patch : patches) {
        EXPECT_EQ(patch.level, 0u);
    }
    for (const int count : Coverage(lod, patches)) {
        EXPECT_EQ(count, 1);
    }
    EXPECT_EQ(lod.TriangleCount(patches), 2u * 6u * 32u * 32u);
    std::vector<LodPatch> none;
    CubeSphereLod().SelectAll(none);
    CubeSphereLod().Select(glm::dvec3(0.0), {}, nullptr, 1.0f, true, none);
    EXPECT_TRUE(none.empty());
}

TEST(CubeSphereLod, SelectionTilesTheSphereWithoutGapsOrOverlaps) {
    const Faces faces = Relief(128);
    const CubeSphereLod lod = Build(faces, 8);
    const auto ranges = lod.Ranges(lod.MinFirstRange(1.0f, kMorphStart), kMorphStart);
    for (const glm::dvec3& eye : TestEyes()) {
        std::vector<LodPatch> patches;
        lod.Select(eye, ranges, nullptr, 1.0f, false, patches);
        for (const int count : Coverage(lod, patches)) {
            ASSERT_EQ(count, 1);
        }
    }
}

TEST(CubeSphereLod, DetailFallsWithDistance) {
    const Faces faces = Relief(128);
    const CubeSphereLod lod = Build(faces, 8);
    const auto ranges = lod.Ranges(lod.MinFirstRange(1.0f, kMorphStart), kMorphStart);

    std::vector<LodPatch> far;
    lod.Select(glm::dvec3(1e6, 0.0, 0.0), ranges, nullptr, 1.0f, false, far);
    ASSERT_EQ(far.size(), 24u);
    for (const LodPatch& patch : far) {
        EXPECT_EQ(patch.level, lod.Levels() - 1);
    }

    const glm::dvec3 eye = CubeSphereLod::FaceDirection(1, 0.3, -0.6) * (kRadius + 1.0);
    std::vector<LodPatch> near;
    lod.Select(eye, ranges, nullptr, 1.0f, false, near);
    EXPECT_GT(near.size(), far.size());
    const glm::dvec2 st = CubeSphereLod::FacePoint(eye).second;
    const glm::uvec2 cell((st + 1.0) * 64.0);
    const auto under = std::find_if(near.begin(), near.end(), [&](const LodPatch& p) {
        return p.face == 1 && p.originX <= cell.x && cell.x < p.originX + 4 &&
               p.originY <= cell.y && cell.y < p.originY + 4;
    });
    ASSERT_NE(under, near.end());
    EXPECT_EQ(under->level, 0u);
    EXPECT_LT(lod.TriangleCount(near), 2u * 6u * 128u * 128u);
}

TEST(CubeSphereLod, NeighboursDifferByOneLevelAndSeamsMatchAcrossFaces) {
    const Faces faces = Relief(64);
    const CubeSphereLod lod = Build(faces, 8);
    for (const float exaggeration : {1.0f, 4.0f}) {
        const double first = lod.MinFirstRange(exaggeration, kMorphStart);
        int levelChanges = 0;
        int faceEdges = 0;
        for (const double scale : {1.0, 1.7}) {
            const auto ranges = lod.Ranges(scale * first, kMorphStart);
            for (const glm::dvec3& eye : TestEyes()) {
                std::vector<LodPatch> patches;
                lod.Select(eye, ranges, nullptr, exaggeration, false, patches);
                const SeamReport report =
                    CheckSeams(lod, faces, patches, eye, ranges, exaggeration);
                EXPECT_GT(report.sharedEdges, 0);
                EXPECT_EQ(report.levelJumps, 0);
                EXPECT_EQ(report.cracks, 0);
                levelChanges += report.levelChanges;
                faceEdges += report.faceEdges;
            }
        }
        EXPECT_GT(levelChanges, 0) << "no edges between levels were checked";
        EXPECT_GT(faceEdges, 0) << "no edges between faces were checked";
    }
}

TEST(CubeSphereLod, RangesBelowTheMinimumCanCrack) {
    const Faces faces = Relief(64);
    const CubeSphereLod lod = Build(faces, 8);
    const auto ranges = lod.Ranges(0.3 * lod.MinFirstRange(1.0f, kMorphStart), kMorphStart);
    int cracks = 0;
    for (const glm::dvec3& eye : TestEyes()) {
        std::vector<LodPatch> patches;
        lod.Select(eye, ranges, nullptr, 1.0f, false, patches);
        cracks += CheckSeams(lod, faces, patches, eye, ranges, 1.0f).cracks;
    }
    EXPECT_GT(cracks, 0) << "the seam check would not notice a bad range";
}

TEST(CubeSphereLod, FullyMorphedPatchMatchesTheCoarserLevel) {
    const Faces faces = Relief(32);
    const CubeSphereLod lod = Build(faces, 8);
    const std::vector<LodRange> fine(lod.Levels(), LodRange{-2.0f, -1.0f});
    const std::vector<LodRange> still(lod.Levels(), LodRange{1e9f, 2e9f});
    const LodPatch patch{8, 16, 1, 4};
    const LodPatch coarser{8, 16, 2, 4};
    auto barycentric = [&](glm::uvec2 cell, glm::dvec2 f) {
        const auto at = [&](glm::uvec2 local) {
            return MorphedVertex(lod, faces, coarser, local, glm::dvec3(0.0), still, 1.0f);
        };
        const glm::dvec3 a = at(cell + glm::uvec2(1, 0));
        const glm::dvec3 b = at(cell + glm::uvec2(0, 1));
        if (f.x + f.y <= 1.0) {
            const glm::dvec3 o = at(cell);
            return o + f.x * (a - o) + f.y * (b - o);
        }
        const glm::dvec3 o = at(cell + glm::uvec2(1, 1));
        return o + (1.0 - f.x) * (b - o) + (1.0 - f.y) * (a - o);
    };
    for (std::uint32_t j = 0; j <= 4; ++j) {
        for (std::uint32_t i = 0; i <= 4; ++i) {
            const glm::dvec3 morphed =
                MorphedVertex(lod, faces, patch, {i, j}, glm::dvec3(0.0), fine, 1.0f);
            const glm::uvec2 cell(std::min(i / 2, 1u), std::min(j / 2, 1u));
            const glm::dvec2 f(0.5 * i - cell.x, 0.5 * j - cell.y);
            EXPECT_NEAR(glm::distance(morphed, barycentric(cell, f)), 0.0, 1e-9 * kRadius)
                << i << ", " << j;
        }
    }
}

TEST(CubeSphereLod, HorizonCullingDropsOnlyHiddenPatches) {
    const Faces faces = Relief(128);
    const CubeSphereLod lod = Build(faces, 8);
    const auto ranges = lod.Ranges(lod.MinFirstRange(1.0f, kMorphStart), kMorphStart);
    const double occluder = lod.OccluderRadius(1.0f);
    EXPECT_LT(occluder, kRadius + static_cast<double>(lod.HeightRange().x));
    EXPECT_GT(occluder, 0.9 * kRadius);
    for (const glm::dvec3& eye : TestEyes()) {
        std::vector<LodPatch> all;
        lod.Select(eye, ranges, nullptr, 1.0f, false, all);
        std::vector<LodPatch> kept;
        lod.Select(eye, ranges, nullptr, 1.0f, true, kept);
        EXPECT_LE(kept.size(), all.size());
        for (const LodPatch& patch : all) {
            if (std::find(kept.begin(), kept.end(), patch) != kept.end()) {
                continue;
            }
            for (std::uint32_t j = 0; j <= lod.PatchCells(); ++j) {
                for (std::uint32_t i = 0; i <= lod.PatchCells(); ++i) {
                    const glm::dvec3 p =
                        Vertex(lod, faces, patch.face, GridIndex(patch, {i, j}), 1.0f);
                    ASSERT_TRUE(BehindSphere(eye, p, occluder));
                }
            }
        }
        for (const LodPatch& patch : kept) {
            EXPECT_NE(std::find(all.begin(), all.end(), patch), all.end());
        }
    }

    const glm::dvec3 low(kRadius + 2.0, 0.0, 0.0);
    std::vector<LodPatch> all;
    lod.Select(low, ranges, nullptr, 1.0f, false, all);
    std::vector<LodPatch> kept;
    lod.Select(low, ranges, nullptr, 1.0f, true, kept);
    EXPECT_LT(kept.size(), all.size());
    EXPECT_TRUE(std::none_of(kept.begin(), kept.end(), [](const LodPatch& p) {
        return p.face == 2;
    })) << "the far side is culled";
}

TEST(CubeSphereLod, FrustumCullingWorksRelativeToTheEye) {
    const Faces faces = Relief(128);
    const CubeSphereLod lod = Build(faces, 8);
    const auto ranges = lod.Ranges(lod.MinFirstRange(1.0f, kMorphStart), kMorphStart);
    const glm::dvec3 eye(kRadius + 5.0, 0.0, 0.0);
    const glm::mat4 projection = glm::perspectiveRH_ZO(1.0f, 1.0f, 0.1f, 1e4f);

    const glm::mat4 level =
        glm::lookAtRH(glm::vec3(0.0f), glm::vec3(-0.2f, 1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    const Frustum frustum = Frustum::FromMatrix(projection * level);
    std::vector<LodPatch> all;
    lod.Select(eye, ranges, nullptr, 1.0f, true, all);
    std::vector<LodPatch> visible;
    lod.Select(eye, ranges, &frustum, 1.0f, true, visible);
    EXPECT_FALSE(visible.empty());
    EXPECT_LT(visible.size(), all.size());
    for (const LodPatch& patch : visible) {
        EXPECT_NE(std::find(all.begin(), all.end(), patch), all.end());
    }

    const glm::mat4 up =
        glm::lookAtRH(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    const Frustum sky = Frustum::FromMatrix(projection * up);
    std::vector<LodPatch> none;
    lod.Select(eye, ranges, &sky, 1.0f, true, none);
    EXPECT_TRUE(none.empty());
}

TEST(CubeSphereLod, SelectsFullDetailUnderTheEyeOnAMoonSizedBody) {
    const double radius = 1737400.0;
    CubeSphereLod lod(radius, 1u << 15, 64);
    const std::uint32_t leaves = lod.LeavesPerSide();
    const std::vector<glm::vec2> range(static_cast<std::size_t>(leaves) * leaves,
                                       glm::vec2(-9000.0f, 10000.0f));
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces; ++face) {
        lod.SetLeaves(face, range);
    }
    const auto ranges = lod.Ranges(lod.MinFirstRange(1.0f, kMorphStart), kMorphStart);
    const double lon = 0.5;
    const double lat = 0.3;
    const glm::dvec3 up(std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon),
                        std::sin(lat));
    const glm::dvec3 eye = up * (radius + 10002.0);
    std::vector<LodPatch> patches;
    lod.Select(eye, ranges, nullptr, 1.0f, true, patches);
    const auto [face, st] = CubeSphereLod::FacePoint(eye);
    const glm::uvec2 cell((st + 1.0) * (0.5 * lod.FaceCells()));
    const auto under = std::find_if(patches.begin(), patches.end(), [&](const LodPatch& p) {
        const std::uint32_t extent = lod.PatchCells() << p.level;
        return p.face == face && p.originX <= cell.x && cell.x < p.originX + extent &&
               p.originY <= cell.y && cell.y < p.originY + extent;
    });
    ASSERT_NE(under, patches.end());
    EXPECT_EQ(under->level, 0u);
    EXPECT_LT(patches.size(), 20000u);
    EXPECT_TRUE(std::none_of(patches.begin(), patches.end(),
                             [face](const LodPatch& p) { return p.face == (face + 2) % 4; }));
}
