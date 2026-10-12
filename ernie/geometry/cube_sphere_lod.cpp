// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/cube_sphere_lod.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

constexpr double kPi = std::numbers::pi;

/// Normal, column axis and row axis of each face, with column x row =
/// -normal.
struct FaceFrame {
    glm::dvec3 normal;
    glm::dvec3 u;
    glm::dvec3 v;
};

constexpr std::array<FaceFrame, CubeSphereLod::kFaces> kFrames = {{
    {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, -1.0}},
    {{0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, -1.0}},
    {{-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}},
    {{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, -1.0}},
    {{0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}},
    {{0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}},
}};

/// Equiangular offset on the face plane, exact at the face's centre
/// and edges.
double PlaneOffset(double s) {
    if (s == 1.0 || s == -1.0 || s == 0.0) {
        return s;
    }
    return std::tan(0.25 * kPi * s);
}

double Angle(const glm::dvec3& a, const glm::dvec3& b) {
    return std::acos(std::clamp(glm::dot(a, b), -1.0, 1.0));
}

}  // namespace

double CubeSphereRegion::Distance(const glm::dvec3& point) const {
    const double e = glm::length(point);
    if (e == 0.0) {
        return inner;
    }
    const double beyond = Angle(point / e, axis) - angle;
    if (beyond <= 0.0) {
        return std::max({0.0, inner - e, e - outer});
    }
    const double c = std::cos(beyond);
    const double r = std::clamp(e * c, inner, outer);
    return std::sqrt(std::max(0.0, e * e + r * r - 2.0 * e * r * c));
}

std::pair<glm::dvec3, glm::dvec3> CubeSphereRegion::Box() const {
    glm::dvec3 low(0.0);
    glm::dvec3 high(0.0);
    for (int i = 0; i < 3; ++i) {
        const double gamma =
            std::acos(std::clamp(axis[i], -1.0, 1.0));
        const double most = std::cos(std::max(0.0, gamma - angle));
        const double least = std::cos(std::min(kPi, gamma + angle));
        high[i] = most >= 0.0 ? outer * most : inner * most;
        low[i] = least >= 0.0 ? inner * least : outer * least;
    }
    return {low, high};
}

void CubeSphereLod::Shape(double radius, std::uint32_t faceCells,
                          std::uint32_t leafCells) {
    if (!std::isfinite(radius) || radius <= 0.0) {
        throw std::invalid_argument(
            "Cube sphere radius must be positive and finite");
    }
    if (!std::has_single_bit(faceCells) || faceCells < leafCells) {
        throw std::invalid_argument(
            "Cube sphere faces need a power of two of at least the "
            "leaf cells along each side");
    }
    for (LodQuadtree& tree : trees_) {
        tree =
            LodQuadtree(faceCells + 1, faceCells + 1, 1.0, leafCells);
    }
    radius_ = radius;
    faceCells_ = faceCells;
}

CubeSphereLod::CubeSphereLod(double radius, std::uint32_t faceCells,
                             std::uint32_t leafCells) {
    Shape(radius, faceCells, leafCells);
}

CubeSphereLod::CubeSphereLod(
    double radius, std::span<const RasterView, kFaces> faces,
    std::uint32_t leafCells) {
    const std::size_t side = faces[0].width;
    for (const RasterView& face : faces) {
        if (face.width != side || face.height != side) {
            throw std::invalid_argument(
                "Cube sphere face rasters must be square and equal");
        }
    }
    if (side < 2 ||
        side - 1 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "Cube sphere face rasters are too small or too large");
    }
    Shape(radius, static_cast<std::uint32_t>(side - 1), leafCells);
    for (std::uint32_t face = 0; face < kFaces; ++face) {
        trees_[face] = LodQuadtree(faces[face], leafCells);
    }
}

glm::dvec3 CubeSphereLod::FaceDirection(std::uint32_t face, double s,
                                        double t) {
    const FaceFrame& frame = kFrames.at(face);
    return glm::normalize(frame.normal + PlaneOffset(s) * frame.u +
                          PlaneOffset(t) * frame.v);
}

std::pair<std::uint32_t, glm::dvec2>
CubeSphereLod::FacePoint(const glm::dvec3& direction) {
    std::uint32_t best = 0;
    double along = -std::numeric_limits<double>::infinity();
    for (std::uint32_t face = 0; face < kFaces; ++face) {
        const double d = glm::dot(direction, kFrames[face].normal);
        if (d > along) {
            along = d;
            best = face;
        }
    }
    const FaceFrame& frame = kFrames[best];
    const double a = glm::dot(direction, frame.u) / along;
    const double b = glm::dot(direction, frame.v) / along;
    return {best,
            glm::dvec2(std::atan(a), std::atan(b)) * (4.0 / kPi)};
}

glm::dvec3 CubeSphereLod::VertexDirection(std::uint32_t face,
                                          std::uint32_t col,
                                          std::uint32_t row) const {
    const double scale = 2.0 / static_cast<double>(faceCells_);
    return FaceDirection(face, static_cast<double>(col) * scale - 1.0,
                         static_cast<double>(row) * scale - 1.0);
}

glm::dvec3 CubeSphereLod::FineDirection(std::uint32_t face,
                                        std::uint32_t col,
                                        std::uint32_t row) const {
    const double scale = 2.0 / static_cast<double>(FineCells());
    return FaceDirection(face, static_cast<double>(col) * scale - 1.0,
                         static_cast<double>(row) * scale - 1.0);
}

void CubeSphereLod::SetRefinement(std::uint32_t levels) {
    if ((static_cast<std::uint64_t>(faceCells_) << levels) >
            (std::uint64_t{1} << 30) ||
        trees_[0].Levels() + levels > kMaxLevels) {
        throw std::invalid_argument("Cube sphere refinement too "
                                    "deep: at most 2^30 quads along "
                                    "a face and " +
                                    std::to_string(kMaxLevels) +
                                    " levels");
    }
    refinement_ = levels;
}

const glm::vec2& CubeSphereLod::NodeHeights(std::uint32_t face,
                                            std::uint32_t level,
                                            std::uint32_t x,
                                            std::uint32_t y) const {
    if (level >= refinement_) {
        return trees_.at(face).NodeHeights(level - refinement_, x, y);
    }
    const std::uint32_t up = refinement_ - level;
    return trees_.at(face).NodeHeights(0, x >> up, y >> up);
}

void CubeSphereLod::SetLeaves(std::uint32_t face,
                              std::span<const glm::vec2> leaves) {
    if (face >= kFaces) {
        throw std::invalid_argument("Cube sphere face out of range");
    }
    trees_[face].SetLeaves(leaves);
}

glm::vec2 CubeSphereLod::HeightRange() const {
    glm::vec2 range(1.0f, 0.0f);
    for (const LodQuadtree& tree : trees_) {
        const glm::vec2 h = tree.HeightRange();
        if (h.x > h.y) {
            continue;
        }
        range = range.x > range.y ? h
                                  : glm::vec2(std::min(range.x, h.x),
                                              std::max(range.y, h.y));
    }
    return range;
}

CubeSphereRegion CubeSphereLod::NodeRegion(std::uint32_t face,
                                           std::uint32_t level,
                                           std::uint32_t x,
                                           std::uint32_t y,
                                           float exaggeration) const {
    const glm::vec2& h = NodeHeights(face, level, x, y);
    CubeSphereRegion region;
    if (h.x > h.y) {
        region.inner = 1.0;
        region.outer = 0.0;
        return region;
    }
    const std::uint32_t cells = LeafCells() << level;
    const std::uint32_t col0 = x * cells;
    const std::uint32_t row0 = y * cells;
    region.axis =
        FineDirection(face, col0 + cells / 2, row0 + cells / 2);
    for (const std::uint32_t corner : {0u, 1u, 2u, 3u}) {
        const glm::dvec3 d =
            FineDirection(face, col0 + (corner & 1u) * cells,
                          row0 + (corner >> 1) * cells);
        region.angle = std::max(region.angle, Angle(region.axis, d));
    }
    region.angle *= 1.0 + 1e-12;
    const double low =
        radius_ + static_cast<double>(h.x) * exaggeration;
    const double high =
        radius_ + static_cast<double>(h.y) * exaggeration;
    region.inner =
        std::max(0.0, std::min(low, high)) * std::cos(region.angle);
    region.outer = std::max(0.0, std::max(low, high));
    return region;
}

// A level-L patch's points lie within the cap of its cells' centres.
// Along a grid line the angle between two vertices is at most pi / 4
// times their difference in s or t, so a cell of the coarsest level
// (2 / LeafCells() across in s and t) has its corners within pi / (2
// LeafCells()) of its centre, and every triangle point at least cos
// of that times the lowest vertex radius.
double CubeSphereLod::OccluderRadius(float exaggeration) const {
    const glm::vec2 h = HeightRange();
    if (Empty() || h.x > h.y) {
        return 0.0;
    }
    const double lowest =
        radius_ + static_cast<double>(std::min(h.x * exaggeration,
                                               h.y * exaggeration));
    return std::max(0.0, lowest) *
           std::cos(kPi / (2.0 * static_cast<double>(LeafCells())));
}

double CubeSphereLod::MinFirstRange(float exaggeration,
                                    float morphStart) const {
    double range = 0.0;
    // Refined levels: each node's sideways extent halves with its
    // level, as the range does, so its data leaf's (a refined node
    // spans 2^-(k - level) of it) sets every refined level's bound.
    if (refinement_ > 0) {
        const std::uint32_t leaves = trees_[0].LeavesPerSide();
        double widest = 0.0;
        for (std::uint32_t face = 0; face < kFaces; ++face) {
            for (std::uint32_t y = 0; y < leaves; ++y) {
                for (std::uint32_t x = 0; x < leaves; ++x) {
                    const CubeSphereRegion r = NodeRegion(
                        face, refinement_, x, y, exaggeration);
                    if (!r.Empty()) {
                        widest = std::max(
                            widest, 2.0 * r.outer *
                                        std::sin(std::min(
                                            r.angle, 0.5 * kPi)));
                    }
                }
            }
        }
        range =
            widest / (static_cast<double>(morphStart) *
                      std::ldexp(1.0, static_cast<int>(refinement_)));
    }
    for (std::uint32_t level = refinement_; level + 1 < Levels();
         ++level) {
        const std::uint32_t side =
            FineCells() / (LeafCells() << level);
        for (std::uint32_t face = 0; face < kFaces; ++face) {
            for (std::uint32_t y = 0; y < side; ++y) {
                for (std::uint32_t x = 0; x < side; ++x) {
                    const CubeSphereRegion r =
                        NodeRegion(face, level, x, y, exaggeration);
                    if (r.Empty()) {
                        continue;
                    }
                    const double spread =
                        std::cos(std::min(2.0 * r.angle, kPi));
                    const double diameter = std::sqrt(std::max(
                        {2.0 * r.outer * r.outer * (1.0 - spread),
                         r.inner * r.inner + r.outer * r.outer -
                             2.0 * r.inner * r.outer * spread,
                         0.0}));
                    range = std::max(
                        range,
                        diameter /
                            (static_cast<double>(morphStart) *
                             static_cast<double>(1u << level)));
                }
            }
        }
    }
    return range;
}

std::vector<LodRange> CubeSphereLod::Ranges(double firstRange,
                                            float morphStart) const {
    // As `LodQuadtree::Ranges`, over the refined levels too.
    std::vector<LodRange> ranges(Levels());
    double previous = 0.0;
    for (std::uint32_t level = 0; level < Levels(); ++level) {
        if (level + 1 == Levels()) {
            ranges[level] = {std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()};
            break;
        }
        const double end =
            firstRange * std::ldexp(1.0, static_cast<int>(level));
        ranges[level] = {
            static_cast<float>(previous +
                               morphStart * (end - previous)),
            static_cast<float>(end)};
        previous = end;
    }
    return ranges;
}

// A point at radius r is hidden behind a sphere of radius a from an
// eye at distance e when the angle between them at the centre exceeds
// acos(a / e) + acos(a / r), where the line between them touches the
// sphere. The region's nearest direction and largest radius are its
// best chance.
bool CubeSphereLod::Hidden(const CubeSphereRegion& region,
                           const glm::dvec3& eye,
                           const Frustum* frustum,
                           double occluder) const {
    if (region.Empty()) {
        return true;
    }
    const double e = glm::length(eye);
    if (occluder > 0.0 && e > occluder && region.outer > 0.0) {
        const double beyond =
            Angle(eye / e, region.axis) - region.angle;
        const double horizon =
            std::acos(occluder / e) +
            std::acos(std::min(1.0, occluder / region.outer));
        if (beyond > horizon) {
            return true;
        }
    }
    if (frustum != nullptr) {
        const auto [low, high] = region.Box();
        if (!frustum->Intersects(glm::vec3(low - eye),
                                 glm::vec3(high - eye))) {
            return true;
        }
    }
    return false;
}

void CubeSphereLod::Select(const glm::dvec3& eye,
                           std::span<const LodRange> ranges,
                           const Frustum* frustum, float exaggeration,
                           bool cullHorizon,
                           std::vector<LodPatch>& out) const {
    if (Empty()) {
        return;
    }
    const double occluder =
        cullHorizon ? OccluderRadius(exaggeration) : 0.0;
    for (std::uint32_t face = 0; face < kFaces; ++face) {
        SelectNode(face, Levels() - 1, 0, 0, eye, ranges, frustum,
                   exaggeration, occluder, out);
    }
}

void CubeSphereLod::SelectAll(std::vector<LodPatch>& out) const {
    if (Empty()) {
        return;
    }
    // The data's leaves at the data's resolution: four patches each.
    const std::uint32_t leaves = LeavesPerSide();
    for (std::uint32_t face = 0; face < kFaces; ++face) {
        for (std::uint32_t y = 0; y < leaves; ++y) {
            for (std::uint32_t x = 0; x < leaves; ++x) {
                const glm::vec2& h =
                    trees_[face].NodeHeights(0, x, y);
                if (h.x <= h.y) {
                    EmitQuarters(face, refinement_, x, y,
                                 glm::dvec3(0.0), nullptr, 1.0f, 0.0,
                                 out);
                }
            }
        }
    }
}

std::uint64_t CubeSphereLod::TriangleCount(
    std::span<const LodPatch> patches) const {
    const std::uint64_t cells = PatchCells();
    return 2 * cells * cells * patches.size();
}

// Strugar 2009, algorithm 1, as `LodQuadtree::SelectNode` with
// distances to the node's region.
bool CubeSphereLod::SelectNode(std::uint32_t face,
                               std::uint32_t level, std::uint32_t x,
                               std::uint32_t y, const glm::dvec3& eye,
                               std::span<const LodRange> ranges,
                               const Frustum* frustum,
                               float exaggeration, double occluder,
                               std::vector<LodPatch>& out) const {
    const CubeSphereRegion region =
        NodeRegion(face, level, x, y, exaggeration);
    if (region.Empty()) {
        return true;
    }
    const double distance = region.Distance(eye);
    if (level + 1 < Levels() && distance > ranges[level].morphEnd) {
        return false;
    }
    if (Hidden(region, eye, frustum, occluder)) {
        return true;
    }
    if (level == 0 || distance > ranges[level - 1].morphEnd) {
        EmitQuarters(face, level, x, y, eye, frustum, exaggeration,
                     occluder, out);
        return true;
    }
    for (std::uint32_t child = 0; child < 4; ++child) {
        const std::uint32_t childX = 2 * x + (child & 1u);
        const std::uint32_t childY = 2 * y + (child >> 1);
        if (!SelectNode(face, level - 1, childX, childY, eye, ranges,
                        frustum, exaggeration, occluder, out)) {
            EmitChildArea(face, level, childX, childY, eye, frustum,
                          exaggeration, occluder, out);
        }
    }
    return true;
}

void CubeSphereLod::EmitQuarters(std::uint32_t face,
                                 std::uint32_t level, std::uint32_t x,
                                 std::uint32_t y,
                                 const glm::dvec3& eye,
                                 const Frustum* frustum,
                                 float exaggeration, double occluder,
                                 std::vector<LodPatch>& out) const {
    if (level > 0) {
        for (std::uint32_t child = 0; child < 4; ++child) {
            EmitChildArea(face, level, 2 * x + (child & 1u),
                          2 * y + (child >> 1), eye, frustum,
                          exaggeration, occluder, out);
        }
        return;
    }
    const std::uint32_t cells = PatchCells();
    for (std::uint32_t quarter = 0; quarter < 4; ++quarter) {
        out.push_back({x * LeafCells() + (quarter & 1u) * cells,
                       y * LeafCells() + (quarter >> 1) * cells, 0,
                       face});
    }
}

void CubeSphereLod::EmitChildArea(
    std::uint32_t face, std::uint32_t level, std::uint32_t childX,
    std::uint32_t childY, const glm::dvec3& eye,
    const Frustum* frustum, float exaggeration, double occluder,
    std::vector<LodPatch>& out) const {
    const CubeSphereRegion region =
        NodeRegion(face, level - 1, childX, childY, exaggeration);
    if (Hidden(region, eye, frustum, occluder)) {
        return;
    }
    const std::uint32_t cells = LeafCells() << (level - 1);
    out.push_back({childX * cells, childY * cells, level, face});
}

}  // namespace ernie
