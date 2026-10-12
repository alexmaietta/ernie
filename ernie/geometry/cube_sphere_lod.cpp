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
    radii_ = glm::dvec3(radius);
    faceCells_ = faceCells;
}

// For unit d moving along a great circle with unit tangent t (d . t =
// 0), q = sum d_i^2 w_i (w_i = 1 / a_i^2) changes by 2 sum d_i t_i
// w_i per radian, at most w_max - w_min by Cauchy-Schwarz, and the
// radius q^(-1/2) by half a_max^3 times that. The normal's largest
// tilt from the radial direction is that of an ellipse of the extreme
// semi-axes, tan = (a^2 - c^2) / (2 a c) (the greatest difference
// between geodetic and geocentric latitude).
void CubeSphereLod::SetRadii(const glm::dvec3& radii) {
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(radii[i]) || radii[i] <= 0.0) {
            throw std::invalid_argument(
                "Cube sphere radii must be positive and finite");
        }
    }
    radii_ = radii;
    const double most = std::max({radii.x, radii.y, radii.z});
    const double least = std::min({radii.x, radii.y, radii.z});
    radius_ = most;
    radiusSlope_ = 0.5 * most * most * most *
                   (1.0 / (least * least) - 1.0 / (most * most));
    const double tilt = std::atan((most * most - least * least) /
                                  (2.0 * most * least));
    normalSin_ = std::sin(tilt);
    normalCos_ = std::cos(tilt);
    // n = d + e with |de| per radian at most the normal map's stretch
    // less 1, (a_max / a_min)^2 - 1, and the tilt e - (e . d) d
    // changing by at most twice that plus twice |e|.
    tiltRate_ = 2.0 * (most * most / (least * least) - 1.0) +
                2.0 * normalSin_;
}

double CubeSphereLod::SurfaceRadius(const glm::dvec3& d) const {
    const glm::dvec3 scaled = d / radii_;
    return 1.0 / std::sqrt(glm::dot(scaled, scaled));
}

glm::dvec3 CubeSphereLod::SurfaceNormal(const glm::dvec3& d) const {
    return glm::normalize(d / (radii_ * radii_));
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
        const RasterView& raster = faces[face];
        trees_[face] = LodQuadtree(raster, leafCells);
        std::vector<float>& heights = heights_[face];
        heights.resize(side * side);
        for (std::size_t row = 0; row < side; ++row) {
            for (std::size_t col = 0; col < side; ++col) {
                const float h = raster(col, row);
                heights[row * side + col] =
                    raster.IsValid(h) ? h : 0.0f;
            }
        }
        double step = 0.0;
        for (std::size_t row = 0; row < side; ++row) {
            for (std::size_t col = 0; col < side; ++col) {
                const double h = heights[row * side + col];
                if (col + 1 < side) {
                    step = std::max(
                        step,
                        std::abs(heights[row * side + col + 1] - h));
                }
                if (row + 1 < side) {
                    step = std::max(
                        step,
                        std::abs(heights[(row + 1) * side + col] -
                                 h));
                }
            }
        }
        maxStep_[face] = step;
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

float CubeSphereLod::FineHeight(std::uint32_t face, std::uint32_t col,
                                std::uint32_t row) const {
    const std::vector<float>& heights = heights_.at(face);
    const std::size_t side = faceCells_ + 1;
    const std::uint32_t mask = (1u << refinement_) - 1u;
    const std::uint32_t c = col >> refinement_;
    const std::uint32_t r = row >> refinement_;
    const std::uint32_t c1 = std::min(c + 1, faceCells_);
    const std::uint32_t r1 = std::min(r + 1, faceCells_);
    const float scale =
        std::ldexp(1.0f, -static_cast<int>(refinement_));
    const float tx = static_cast<float>(col & mask) * scale;
    const float ty = static_cast<float>(row & mask) * scale;
    const auto at = [&](std::uint32_t i, std::uint32_t j) {
        return heights[j * side + i];
    };
    const float top = at(c, r) + (at(c1, r) - at(c, r)) * tx;
    const float bottom = at(c, r1) + (at(c1, r1) - at(c, r1)) * tx;
    return top + (bottom - top) * ty;
}

glm::vec2 CubeSphereLod::NodeHeights(std::uint32_t face,
                                     std::uint32_t level,
                                     std::uint32_t x,
                                     std::uint32_t y) const {
    if (level >= refinement_) {
        return trees_.at(face).NodeHeights(level - refinement_, x, y);
    }
    const std::uint32_t up = refinement_ - level;
    const glm::vec2 leaf =
        trees_.at(face).NodeHeights(0, x >> up, y >> up);
    if (leaf.x > leaf.y || heights_.at(face).empty()) {
        return leaf;
    }
    // Bilinear heights reach their extremes at the data's vertices
    // or, within one quad, at the region's corners.
    const std::uint32_t cells = LeafCells() << level;
    const std::uint32_t col0 = x * cells;
    const std::uint32_t row0 = y * cells;
    glm::vec2 range(std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity());
    const auto take = [&range](float h) {
        range = glm::vec2(std::min(range.x, h), std::max(range.y, h));
    };
    if (cells < (1u << refinement_)) {
        for (const std::uint32_t corner : {0u, 1u, 2u, 3u}) {
            take(FineHeight(face, col0 + (corner & 1u) * cells,
                            row0 + (corner >> 1) * cells));
        }
        return range;
    }
    const std::vector<float>& heights = heights_[face];
    const std::size_t side = faceCells_ + 1;
    const std::uint32_t first = col0 >> refinement_;
    const std::uint32_t last = (col0 + cells) >> refinement_;
    const std::uint32_t top = row0 >> refinement_;
    const std::uint32_t bottom = (row0 + cells) >> refinement_;
    for (std::uint32_t row = top; row <= bottom; ++row) {
        for (std::uint32_t col = first; col <= last; ++col) {
            take(heights[row * side + col]);
        }
    }
    return range;
}

void CubeSphereLod::SetLeaves(std::uint32_t face,
                              std::span<const glm::vec2> leaves) {
    if (face >= kFaces) {
        throw std::invalid_argument("Cube sphere face out of range");
    }
    trees_[face].SetLeaves(leaves);
    heights_[face].clear();
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
    const glm::vec2 h = NodeHeights(face, level, x, y);
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
    // On an ellipsoid the base radius over the cap is within
    // `radiusSlope_` per radian of the axis's, and a height h along a
    // normal tilted from its direction moves a point radially by
    // between h cos(tilt) and h, and turns it by asin(|h| sin(tilt) /
    // |p|). Nearby points turn alike, so the cap centres on the
    // axis's point at the middle height and widens by how the turn
    // varies over the heights and across the node.
    const double spread = radiusSlope_ * region.angle;
    const double base = SurfaceRadius(region.axis);
    const double baseLow = std::max(
        std::min({radii_.x, radii_.y, radii_.z}), base - spread);
    const double baseHigh = std::min(radius_, base + spread);
    const double h0 =
        std::min(static_cast<double>(h.x) * exaggeration,
                 static_cast<double>(h.y) * exaggeration);
    const double h1 =
        std::max(static_cast<double>(h.x) * exaggeration,
                 static_cast<double>(h.y) * exaggeration);
    const double low = baseLow + (h0 < 0.0 ? h0 : h0 * normalCos_);
    const double high = baseHigh + (h1 > 0.0 ? h1 : h1 * normalCos_);
    if (normalSin_ > 0.0 && low > 0.0) {
        const double middle = 0.5 * (h0 + h1);
        region.axis =
            glm::normalize(BasePoint(region.axis) +
                           middle * SurfaceNormal(region.axis));
        // asin(x) <= x pi / 2 bounds the turn's change by its sine's.
        const double turn = (0.5 * (h1 - h0) * normalSin_ +
                             std::max(std::abs(h0), std::abs(h1)) *
                                 tiltRate_ * region.angle) /
                            low;
        region.angle += 0.5 * kPi * std::min(1.0, turn);
    }
    region.inner = std::max(0.0, low) * std::cos(region.angle);
    region.outer = std::max(0.0, high);
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
    const double least = static_cast<double>(
        std::min(h.x * exaggeration, h.y * exaggeration));
    const double lowest = std::min({radii_.x, radii_.y, radii_.z}) +
                          (least < 0.0 ? least : least * normalCos_);
    return std::max(0.0, lowest) *
           std::cos(kPi / (2.0 * static_cast<double>(LeafCells())));
}

double CubeSphereLod::MinFirstRange(float exaggeration,
                                    float morphStart) const {
    double range = 0.0;
    // Refined levels: a node's extent, the angle its cap widens by
    // and its heights' span (at most 2 w steps between neighbouring
    // vertices over w data quads) all scale with its size, halving
    // with each level as the ranges do, so a bound on each data leaf
    // scaled to its size sets every refined level's. Without a face's
    // heights its nodes' spans go unbounded here.
    if (refinement_ > 0) {
        const std::uint32_t leaves = trees_[0].LeavesPerSide();
        const std::uint32_t cells = LeafCells() << refinement_;
        const double scale =
            std::abs(static_cast<double>(exaggeration));
        double widest = 0.0;
        for (std::uint32_t face = 0; face < kFaces; ++face) {
            const double span = 2.0 * LeafCells() * maxStep_[face] *
                                scale *
                                (heights_[face].empty() ? 0.0 : 1.0);
            for (std::uint32_t y = 0; y < leaves; ++y) {
                for (std::uint32_t x = 0; x < leaves; ++x) {
                    const CubeSphereRegion r = NodeRegion(
                        face, refinement_, x, y, exaggeration);
                    if (r.Empty()) {
                        continue;
                    }
                    const glm::dvec3 axis =
                        FineDirection(face, x * cells + cells / 2,
                                      y * cells + cells / 2);
                    double angle = 0.0;
                    for (const std::uint32_t corner :
                         {0u, 1u, 2u, 3u}) {
                        angle = std::max(
                            angle,
                            Angle(
                                axis,
                                FineDirection(
                                    face,
                                    x * cells + (corner & 1u) * cells,
                                    y * cells +
                                        (corner >> 1) * cells)));
                    }
                    const glm::vec2 h =
                        trees_[face].NodeHeights(0, x, y);
                    const double highest =
                        std::max(std::abs(h.x), std::abs(h.y)) *
                        scale;
                    const double low =
                        std::min({radii_.x, radii_.y, radii_.z}) -
                        highest;
                    double turn = 0.0;
                    if (normalSin_ > 0.0 && low > 0.0) {
                        turn = 0.5 * kPi *
                               (0.5 * span * normalSin_ +
                                highest * tiltRate_ * angle) /
                               low;
                    }
                    widest = std::max(widest,
                                      2.0 * r.outer * (angle + turn) +
                                          2.0 * radiusSlope_ * angle +
                                          span);
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
