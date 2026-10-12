// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/overlay.h"

#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

/// Versions shared by every overlay, so a new or cleared overlay
/// never repeats one.
std::atomic<std::uint64_t> nextVersion{1};

/// Width in units of each line of `codes`, the widest.
float CodesWidth(std::span<const std::uint8_t> codes) {
    const OverlayFont& font = OverlayTextFont();
    float widest = 0.0f;
    float pen = 0.0f;
    for (std::size_t i = 0; i < codes.size(); ++i) {
        if (codes[i] == '\n') {
            pen = 0.0f;
            continue;
        }
        pen += font.glyphs[codes[i]].advance;
        if (i + 1 < codes.size()) {
            pen += font.Kerning(codes[i], codes[i + 1]);
        }
        widest = std::max(widest, pen);
    }
    return widest;
}

/// Two unit vectors perpendicular to unit `n` and to each other.
std::pair<glm::dvec3, glm::dvec3> Basis(const glm::dvec3& n) {
    const glm::dvec3 helper = std::abs(n.z) < 0.9
                                  ? glm::dvec3(0.0, 0.0, 1.0)
                                  : glm::dvec3(1.0, 0.0, 0.0);
    const glm::dvec3 u = glm::normalize(glm::cross(helper, n));
    return {u, glm::cross(n, u)};
}

glm::dvec3 Apply(const glm::dmat4& transform, const glm::dvec3& p) {
    return glm::dvec3(transform * glm::dvec4(p, 1.0));
}

void CheckIndices(std::span<const glm::dvec3> points,
                  std::span<const std::uint32_t> indices) {
    if (indices.size() % 3 != 0) {
        throw std::invalid_argument(
            "Overlay triangles need indices in threes");
    }
    for (const std::uint32_t index : indices) {
        if (index >= points.size()) {
            throw std::invalid_argument(
                "Overlay triangle index " + std::to_string(index) +
                " is past the " + std::to_string(points.size()) +
                " points");
        }
    }
}

}  // namespace

glm::vec2 OverlayTextSize(std::string_view text, float size) {
    const std::vector<std::uint8_t> codes = OverlayCodes(text);
    if (codes.empty()) {
        return glm::vec2(0.0f);
    }
    const float scale = size / kOverlayLineHeight;
    // The last line's capitals and descenders, not its spacing.
    const auto lines = static_cast<float>(
        std::count(codes.begin(), codes.end(), std::uint8_t{'\n'}));
    return glm::vec2(CodesWidth(codes),
                     lines * kOverlayLineHeight +
                         OverlayTextFont().textHeight) *
           scale;
}

Overlay::Overlay() {
    Touch();
}

OverlayItem Overlay::ItemOf(const OverlayStyle& style,
                            const glm::vec4& color) {
    OverlayItem item{color, style.depth, style.world};
    item.id = style.id;
    item.hover = style.hover;
    item.metric = style.metric;
    if (!style.anchor.IsEmpty()) {
        const auto found =
            std::find(anchors_.begin(), anchors_.end(), style.anchor);
        item.anchor =
            static_cast<std::uint32_t>(found - anchors_.begin()) + 1;
        if (found == anchors_.end()) {
            anchors_.push_back(style.anchor);
        }
    }
    return item;
}

Overlay& Overlay::Touch() {
    version_ = nextVersion.fetch_add(1);
    return *this;
}

Overlay& Overlay::Clear() {
    segments_.clear();
    triangles_.clear();
    sprites_.clear();
    batches_.clear();
    anchors_.clear();
    Touch();
    return *this;
}

std::size_t OverlayBatch::Count() const {
    // An image is one item, whatever its corners.
    if (kind == OverlayBatchKind::Image) {
        return image.Empty() ? 0 : 1;
    }
    if (points.Empty()) {
        if (kind == OverlayBatchKind::Numbers) {
            return static_cast<std::size_t>(values.Elements());
        }
        return kind == OverlayBatchKind::Texts && chars.Rank() == 2
                   ? static_cast<std::size_t>(chars.Shape()[0])
                   : 0;
    }
    return static_cast<std::size_t>(points.Elements()) / 3 /
           TrailLength();
}

std::size_t OverlayBatch::TrailLength() const {
    return kind == OverlayBatchKind::Trails && points.Rank() == 3
               ? std::max<std::size_t>(
                     static_cast<std::size_t>(points.Shape()[1]), 1)
               : 1;
}

Overlay& Overlay::Batch(OverlayBatch batch) {
    using Kind = OverlayBatchKind;
    const Kind kind = batch.kind;
    // Screen numbers and texts may go without points, at their
    // placement.
    const bool text = kind == Kind::Numbers || kind == Kind::Texts;
    const bool placed = text && batch.screen;
    // An image's arrays go with it: corners from the host may frame a
    // GPU image.
    const Tensor& lead =
        kind == Kind::Image ? batch.image
        : placed && batch.points.Empty()
            ? (kind == Kind::Numbers ? batch.values : batch.chars)
            : batch.points;
    const auto require = [&](const Tensor& array, const char* what,
                             std::int64_t columns, DType type,
                             bool needed) {
        if (array.Empty()) {
            if (needed) {
                throw std::invalid_argument(std::string("Overlay ") +
                                            what + " are needed");
            }
            return;
        }
        if (array.Type() != type || !array.IsContiguous()) {
            throw std::invalid_argument(
                std::string("Overlay ") + what +
                " must be contiguous " +
                (type == DType::Int32 ? "int32" : "float32"));
        }
        if (array.Elements() % columns != 0 ||
            (array.Rank() > 1 && array.Shape().back() != columns)) {
            throw std::invalid_argument(
                std::string("Overlay ") + what + " must have " +
                std::to_string(columns) + " numbers per item");
        }
        if (array.Location() != lead.Location()) {
            throw std::invalid_argument(
                std::string("Overlay ") + what +
                " must be on the same device as the points");
        }
        if (static_cast<std::size_t>(array.Elements() / columns) !=
            batch.Count()) {
            throw std::invalid_argument(
                std::string("Overlay ") + what +
                " must have one row per item");
        }
    };
    if (batch.screen && !text && kind != Kind::Image) {
        throw std::invalid_argument("Only overlay numbers, texts and "
                                    "images go on the screen");
    }
    if (kind == Kind::Image) {
        if (batch.image.Type() != DType::UInt8 ||
            !batch.image.IsContiguous() || batch.image.Rank() != 3 ||
            (batch.image.Shape()[2] != 3 &&
             batch.image.Shape()[2] != 4) ||
            batch.image.Shape()[0] < 1 ||
            batch.image.Shape()[1] < 1) {
            throw std::invalid_argument(
                "An overlay image must be contiguous uint8 (rows, "
                "cols, "
                "3 or 4)");
        }
        if (!batch.screen && (batch.points.Type() != DType::Float32 ||
                              !batch.points.IsContiguous() ||
                              batch.points.Elements() != 12)) {
            throw std::invalid_argument(
                "An overlay image in the world needs its four "
                "corners, "
                "(4, 3) float32");
        }
    }
    if (kind == Kind::Trails && batch.points.Rank() != 3) {
        throw std::invalid_argument(
            "Overlay trails' points must be (items, length, 3)");
    }
    // A trail's arrays have a row per trail (`Count`), its points a
    // row per point of each.
    if (kind == Kind::Trails) {
        if (!batch.points.IsContiguous() ||
            batch.points.Type() != DType::Float32 ||
            batch.points.Shape()[2] != 3) {
            throw std::invalid_argument(
                "Overlay trails' points must be contiguous float32 "
                "(items, length, 3)");
        }
    } else if (kind != Kind::Image) {
        require(batch.points, "points", 3, DType::Float32, !placed);
    }
    require(batch.heads, "heads", 1, DType::Int32, false);
    require(batch.halfHeights, "half heights", 1, DType::Float32,
            kind == Kind::Capsules);
    require(batch.poseIndices, "pose indices", 1, DType::Int32,
            false);
    if (!batch.count.Empty() &&
        (batch.count.Type() != DType::Int32 ||
         batch.count.Elements() != 1 ||
         batch.count.Location() != lead.Location())) {
        throw std::invalid_argument(
            "An overlay batch's count must be one int32 on the "
            "points' device");
    }
    if (!batch.poses.Empty()) {
        if (batch.poses.Type() != DType::Float32 ||
            !batch.poses.IsContiguous() ||
            batch.poses.Elements() % 7 != 0 ||
            batch.poses.Location() != lead.Location()) {
            throw std::invalid_argument(
                "Overlay poses must be contiguous float32 (m, 7) "
                "rows on the points' device: position, then "
                "quaternion x, y, z, w");
        }
        if (batch.poseIndices.Empty() &&
            static_cast<std::size_t>(batch.poses.Elements() / 7) <
                batch.Count()) {
            throw std::invalid_argument(
                "Overlay poses must be one per item without pose "
                "indices");
        }
        if (!batch.anchors.empty() || !batch.style.anchor.IsEmpty() ||
            batch.screen) {
            throw std::invalid_argument(
                "Overlay items follow poses or prims, not both, and "
                "not on the screen");
        }
    } else if (!batch.poseIndices.Empty()) {
        throw std::invalid_argument(
            "Overlay pose indices need poses");
    }
    require(batch.values, "values", 1, DType::Float32,
            kind == Kind::Numbers);
    if (kind == Kind::Texts) {
        if (batch.chars.Rank() != 2 ||
            batch.chars.Type() != DType::Int32 ||
            !batch.chars.IsContiguous() ||
            batch.chars.Location() != lead.Location() ||
            static_cast<std::size_t>(batch.chars.Shape()[0]) !=
                batch.Count() ||
            batch.chars.Shape()[1] < 1) {
            throw std::invalid_argument(
                "Overlay texts' chars must be contiguous int32 "
                "(items, "
                "length) on the points' device");
        }
    }
    if (!batch.poseWorlds.Empty() &&
        (batch.poses.Empty() ||
         batch.poseWorlds.Type() != DType::Int32 ||
         !batch.poseWorlds.IsContiguous() ||
         batch.poseWorlds.Location() != lead.Location() ||
         batch.poseWorlds.Elements() * 7 != batch.poses.Elements())) {
        throw std::invalid_argument(
            "Overlay pose worlds must be contiguous int32, one per "
            "pose, on the points' device");
    }
    require(batch.ends, "ends", 3, DType::Float32,
            kind == Kind::Lines || kind == Kind::Arrows);
    require(batch.halfExtents, "half extents", 3, DType::Float32,
            kind == Kind::Boxes);
    require(batch.orientations, "orientations", 4, DType::Float32,
            false);
    require(batch.radii, "radii", 1, DType::Float32,
            kind == Kind::Spheres || kind == Kind::Capsules);
    if (!batch.colors.Empty()) {
        const std::int64_t columns =
            batch.colors.Rank() > 1 ? batch.colors.Shape().back() : 3;
        // A trail's colours may be one per point, (n, length, 3 or
        // 4); lines' and arrows' one per end, (n, 2, 3 or 4).
        const auto rows =
            static_cast<std::size_t>(batch.colors.Elements()) /
            static_cast<std::size_t>(columns == 4 ? 4 : 3);
        const bool perPoint =
            (kind == Kind::Trails &&
             rows == batch.Count() * batch.TrailLength()) ||
            ((kind == Kind::Lines || kind == Kind::Arrows) &&
             rows == 2 * batch.Count());
        if (perPoint) {
            if (batch.colors.Type() != DType::Float32 ||
                !batch.colors.IsContiguous() ||
                batch.colors.Location() != lead.Location()) {
                throw std::invalid_argument(
                    "Overlay colours must be contiguous float32 on "
                    "the "
                    "points' device");
            }
        } else {
            require(batch.colors, "colours", columns == 4 ? 4 : 3,
                    DType::Float32, false);
        }
    }
    require(batch.sizes, "sizes", 1, DType::Float32, false);
    require(batch.worlds, "worlds", 1, DType::Int32, false);
    if (batch.anchors.empty() && !batch.style.anchor.IsEmpty()) {
        batch.anchors = {batch.style.anchor};
    }
    if (batch.screen) {
        batch.anchors.clear();
    }
    if (batch.anchors.size() > 1 &&
        batch.anchors.size() != batch.Count()) {
        throw std::invalid_argument(
            "Overlay anchors must be one, or one per item: " +
            std::to_string(batch.anchors.size()) + " for " +
            std::to_string(batch.Count()));
    }
    if (batch.format.decimals < 0 || batch.format.decimals > 6) {
        throw std::invalid_argument(
            "Overlay numbers take 0 to 6 decimals");
    }
    if (batch.Count() > 0) {
        batches_.push_back(std::move(batch));
    }
    Touch();
    return *this;
}

void Overlay::ReplaceBatchArray(std::size_t batch,
                                std::string_view array,
                                Tensor values) {
    if (batch >= batches_.size()) {
        throw std::invalid_argument("The overlay has no batch " +
                                    std::to_string(batch));
    }
    OverlayBatch& b = batches_[batch];
    const std::pair<std::string_view, Tensor*> arrays[] = {
        {"points", &b.points},
        {"ends", &b.ends},
        {"half_extents", &b.halfExtents},
        {"orientations", &b.orientations},
        {"radii", &b.radii},
        {"values", &b.values},
        {"colors", &b.colors},
        {"sizes", &b.sizes},
        {"worlds", &b.worlds},
        {"half_heights", &b.halfHeights},
        {"heads", &b.heads},
        {"count", &b.count},
        {"poses", &b.poses},
        {"pose_indices", &b.poseIndices},
        {"pose_worlds", &b.poseWorlds},
        {"chars", &b.chars},
        {"image", &b.image}};
    for (const auto& [name, field] : arrays) {
        if (name != array) {
            continue;
        }
        if (field->Empty()) {
            break;
        }
        if (values.Type() != field->Type() ||
            values.Elements() != field->Elements()) {
            throw std::invalid_argument(
                "Overlay batch array '" + std::string(array) +
                "' must keep its type and size");
        }
        *field = std::move(values);
        return;
    }
    throw std::invalid_argument("The overlay batch has no array '" +
                                std::string(array) + "'");
}

Overlay& Overlay::SetVisible(bool visible) {
    if (visible != visible_) {
        visible_ = visible;
        Touch();
    }
    return *this;
}

void Overlay::AddSegment(const glm::dvec3& a, const glm::dvec3& b,
                         const OverlayStyle& style,
                         const glm::vec4& color) {
    segments_.push_back(
        {a, b, style.width, ItemOf(style, color), style.dash});
}

Overlay& Overlay::Line(const glm::dvec3& a, const glm::dvec3& b,
                       const OverlayStyle& style) {
    AddSegment(a, b, style, style.color);
    Touch();
    return *this;
}

Overlay& Overlay::Polyline(std::span<const glm::dvec3> points,
                           const OverlayStyle& style, bool closed,
                           std::span<const glm::vec4> colors) {
    if (!colors.empty() && colors.size() != points.size()) {
        throw std::invalid_argument(
            "Overlay polylines need one colour per point, not " +
            std::to_string(colors.size()) + " for " +
            std::to_string(points.size()));
    }
    // From point i to point j, blended between their colours if
    // given.
    const auto segment = [&](std::size_t i, std::size_t j) {
        if (colors.empty()) {
            AddSegment(points[i], points[j], style, style.color);
            return;
        }
        AddSegment(points[i], points[j], style, colors[i]);
        segments_.back().endColor = colors[j];
    };
    for (std::size_t i = 1; i < points.size(); ++i) {
        segment(i - 1, i);
    }
    if (closed && points.size() > 2) {
        segment(points.size() - 1, 0);
    }
    Touch();
    return *this;
}

Overlay& Overlay::Lines(std::span<const glm::dvec3> endpoints,
                        const OverlayStyle& style,
                        std::span<const glm::vec4> colors) {
    if (endpoints.size() % 2 != 0) {
        throw std::invalid_argument(
            "Overlay lines need endpoints in pairs");
    }
    const std::size_t count = endpoints.size() / 2;
    if (!colors.empty() && colors.size() != count) {
        throw std::invalid_argument(
            "Overlay lines need one colour per segment, not " +
            std::to_string(colors.size()) + " for " +
            std::to_string(count));
    }
    segments_.reserve(segments_.size() + count);
    for (std::size_t i = 0; i < count; ++i) {
        AddSegment(endpoints[2 * i], endpoints[2 * i + 1], style,
                   colors.empty() ? style.color : colors[i]);
    }
    Touch();
    return *this;
}

Overlay& Overlay::Arrow(const glm::dvec3& from, const glm::dvec3& to,
                        const OverlayStyle& style, double head) {
    const glm::dvec3 shaft = to - from;
    const double length = glm::length(shaft);
    if (!(length > 0.0)) {
        Touch();
        return *this;
    }
    const glm::dvec3 axis = shaft / length;
    const double headLength =
        std::min(head > 0.0 ? head : 0.2 * length, length);
    const glm::dvec3 base = to - axis * headLength;
    AddSegment(from, base, style, style.color);
    // A cone of eight sides and its base.
    const auto [u, v] = Basis(axis);
    const double radius = 0.35 * headLength;
    constexpr int kSides = 8;
    for (int i = 0; i < kSides; ++i) {
        const double a0 = 2.0 * std::numbers::pi * i / kSides;
        const double a1 = 2.0 * std::numbers::pi * (i + 1) / kSides;
        const glm::dvec3 p0 =
            base + radius * (std::cos(a0) * u + std::sin(a0) * v);
        const glm::dvec3 p1 =
            base + radius * (std::cos(a1) * u + std::sin(a1) * v);
        triangles_.push_back(
            {{to, p0, p1}, ItemOf(style, style.color)});
        triangles_.push_back(
            {{base, p1, p0}, ItemOf(style, style.color)});
    }
    Touch();
    return *this;
}

Overlay& Overlay::Box(const glm::dvec3& low, const glm::dvec3& high,
                      const OverlayStyle& style,
                      const glm::dmat4& transform) {
    std::array<glm::dvec3, 8> corners;
    for (int i = 0; i < 8; ++i) {
        corners[i] =
            Apply(transform, {(i & 1) != 0 ? high.x : low.x,
                              (i & 2) != 0 ? high.y : low.y,
                              (i & 4) != 0 ? high.z : low.z});
    }
    // Corners differing in one bit share an edge.
    for (int i = 0; i < 8; ++i) {
        for (const int bit : {1, 2, 4}) {
            if ((i & bit) == 0) {
                AddSegment(corners[i], corners[i | bit], style,
                           style.color);
            }
        }
    }
    Touch();
    return *this;
}

Overlay& Overlay::Circle(const glm::dvec3& centre,
                         const glm::dvec3& normal, double radius,
                         const OverlayStyle& style, int segments) {
    const double length = glm::length(normal);
    if (!(length > 0.0) || segments < 3) {
        throw std::invalid_argument("An overlay circle needs a "
                                    "normal and at least 3 segments");
    }
    const auto [u, v] = Basis(normal / length);
    glm::dvec3 previous = centre + radius * u;
    for (int i = 1; i <= segments; ++i) {
        const double angle = 2.0 * std::numbers::pi * i / segments;
        const glm::dvec3 next =
            centre +
            radius * (std::cos(angle) * u + std::sin(angle) * v);
        AddSegment(previous, next, style, style.color);
        previous = next;
    }
    Touch();
    return *this;
}

Overlay& Overlay::Arc(const glm::dvec3& centre,
                      const glm::dvec3& normal,
                      const glm::dvec3& from, double radius,
                      double angle, const OverlayStyle& style,
                      bool filled, int segments) {
    const double length = glm::length(normal);
    if (!(length > 0.0) || segments < 1) {
        throw std::invalid_argument(
            "An overlay arc needs a normal and at least 1 segment");
    }
    const glm::dvec3 n = normal / length;
    // `from` in the plane, and a quarter turn on from it.
    const glm::dvec3 inPlane = from - glm::dot(from, n) * n;
    if (!(glm::length(inPlane) > 0.0)) {
        throw std::invalid_argument(
            "An overlay arc's start must lie off its normal");
    }
    const glm::dvec3 u = glm::normalize(inPlane);
    const glm::dvec3 v = glm::cross(n, u);
    const auto at = [&](int i) {
        const double a = angle * i / segments;
        return centre + radius * (std::cos(a) * u + std::sin(a) * v);
    };
    glm::dvec3 previous = at(0);
    for (int i = 1; i <= segments; ++i) {
        const glm::dvec3 next = at(i);
        if (filled) {
            triangles_.push_back({{centre, previous, next},
                                  ItemOf(style, style.color)});
        } else {
            AddSegment(previous, next, style, style.color);
        }
        previous = next;
    }
    Touch();
    return *this;
}

Overlay& Overlay::Curve(const glm::dvec3& p0, const glm::dvec3& p1,
                        const glm::dvec3& p2, const glm::dvec3& p3,
                        const OverlayStyle& style, int segments) {
    if (segments < 1) {
        throw std::invalid_argument(
            "An overlay curve needs at least 1 segment");
    }
    const auto at = [&](double t) {
        const double s = 1.0 - t;
        return s * s * s * p0 + 3.0 * s * s * t * p1 +
               3.0 * s * t * t * p2 + t * t * t * p3;
    };
    glm::dvec3 previous = p0;
    for (int i = 1; i <= segments; ++i) {
        const glm::dvec3 next = at(static_cast<double>(i) / segments);
        AddSegment(previous, next, style, style.color);
        previous = next;
    }
    Touch();
    return *this;
}

Overlay& Overlay::Sphere(const glm::dvec3& centre, double radius,
                         const OverlayStyle& style) {
    for (const glm::dvec3 axis :
         {glm::dvec3(1, 0, 0), glm::dvec3(0, 1, 0),
          glm::dvec3(0, 0, 1)}) {
        Circle(centre, axis, radius, style);
    }
    return *this;
}

Overlay& Overlay::Axes(const glm::dmat4& transform, double length,
                       const OverlayStyle& style) {
    const glm::dvec3 origin(transform[3]);
    const glm::vec4 colors[] = {{0.9f, 0.2f, 0.2f, style.color.a},
                                {0.3f, 0.85f, 0.3f, style.color.a},
                                {0.25f, 0.45f, 1.0f, style.color.a}};
    for (int k = 0; k < 3; ++k) {
        const glm::dvec3 axis = glm::dvec3(transform[k]);
        const double scale = glm::length(axis);
        const glm::dvec3 tip =
            scale > 0.0 ? origin + axis / scale * length : origin;
        AddSegment(origin, tip, style, colors[k]);
    }
    Touch();
    return *this;
}

Overlay& Overlay::WireMesh(std::span<const glm::dvec3> points,
                           std::span<const std::uint32_t> indices,
                           const OverlayStyle& style,
                           const glm::dmat4& transform) {
    CheckIndices(points, indices);
    std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
    for (std::size_t t = 0; t < indices.size(); t += 3) {
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t a = indices[t + k];
            const std::uint32_t b = indices[t + (k + 1) % 3];
            edges.emplace(std::min(a, b), std::max(a, b));
        }
    }
    for (const auto& [a, b] : edges) {
        AddSegment(Apply(transform, points[a]),
                   Apply(transform, points[b]), style, style.color);
    }
    Touch();
    return *this;
}

Overlay& Overlay::Triangles(std::span<const glm::dvec3> points,
                            std::span<const std::uint32_t> indices,
                            const OverlayStyle& style,
                            const glm::dmat4& transform) {
    CheckIndices(points, indices);
    for (std::size_t t = 0; t < indices.size(); t += 3) {
        triangles_.push_back(
            {{Apply(transform, points[indices[t]]),
              Apply(transform, points[indices[t + 1]]),
              Apply(transform, points[indices[t + 2]])},
             ItemOf(style, style.color)});
    }
    Touch();
    return *this;
}

Overlay& Overlay::Marker(const glm::dvec3& point,
                         const OverlayStyle& style,
                         MarkerShape shape) {
    const float half = 0.5f * style.size;
    sprites_.push_back({point,
                        {-half, -half, half, half},
                        shape,
                        ItemOf(style, style.color)});
    Touch();
    return *this;
}

Overlay& Overlay::Markers(std::span<const glm::dvec3> points,
                          const OverlayStyle& style,
                          MarkerShape shape,
                          std::span<const glm::vec4> colors) {
    if (!colors.empty() && colors.size() != points.size()) {
        throw std::invalid_argument(
            "Overlay markers need one colour per point, not " +
            std::to_string(colors.size()) + " for " +
            std::to_string(points.size()));
    }
    const float half = 0.5f * style.size;
    sprites_.reserve(sprites_.size() + points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        sprites_.push_back(
            {points[i],
             {-half, -half, half, half},
             shape,
             ItemOf(style,
                    colors.empty() ? style.color : colors[i])});
    }
    Touch();
    return *this;
}

void Overlay::AddText(const glm::dvec3& anchor,
                      const glm::vec2& corner, std::string_view text,
                      const OverlayStyle& style, bool screen) {
    const std::vector<std::uint8_t> codes = OverlayCodes(text);
    if (codes.empty()) {
        return;
    }
    const float scale = style.size / kOverlayLineHeight;
    const bool halo = style.halo > 0.0f && style.background.a > 0.0f;
    if (style.background.a > 0.0f && !halo) {
        const float pad = 0.25f * style.size;
        const glm::vec2 size = OverlayTextSize(text, style.size);
        sprites_.push_back(
            {anchor,
             {corner.x - pad, corner.y - pad, corner.x + size.x + pad,
              corner.y + size.y + pad},
             MarkerShape::Square,
             ItemOf(style, style.background),
             0,
             screen});
    }
    // A sprite per character over the box its field covers.
    const OverlayFont& font = OverlayTextFont();
    glm::vec2 pen(0.0f);
    for (std::size_t i = 0; i < codes.size(); ++i) {
        const std::uint8_t c = codes[i];
        if (c == '\n') {
            pen = {0.0f, pen.y + kOverlayLineHeight};
            continue;
        }
        const OverlayGlyph& glyph = font.glyphs[c];
        if (glyph.width > 0) {
            const glm::vec2 low =
                corner + (pen + glm::vec2(glyph.box)) * scale;
            const glm::vec2 high =
                corner +
                (pen + glm::vec2(glyph.box.z, glyph.box.w)) * scale;
            OverlaySprite sprite{anchor,
                                 {low.x, low.y, high.x, high.y},
                                 MarkerShape::Square,
                                 ItemOf(style, style.color),
                                 c,
                                 screen};
            if (halo) {
                sprite.halo = style.halo;
                sprite.haloColor = style.background;
            }
            sprites_.push_back(sprite);
        }
        pen.x += glyph.advance;
        if (i + 1 < codes.size()) {
            pen.x += font.Kerning(c, codes[i + 1]);
        }
    }
}

Overlay& Overlay::Label(const glm::dvec3& point,
                        std::string_view text,
                        const OverlayStyle& style) {
    // To the right of and above the point, so a marker there stays
    // visible.
    const float height = OverlayTextSize(text, style.size).y;
    AddText(point,
            {0.5f * style.size,
             -height - 3.0f * style.size / kOverlayLineHeight -
                 0.25f * style.size},
            text, style, false);
    Touch();
    return *this;
}

Overlay& Overlay::Text(const ScreenPlacement& placement,
                       std::string_view text,
                       const OverlayStyle& style) {
    const glm::vec2 size = OverlayTextSize(text, style.size);
    // The backing box's padding stays inside the image at its edges.
    const float pad =
        style.background.a > 0.0f ? 0.25f * style.size : 0.0f;
    const glm::vec2 corner = placement.offset -
                             placement.Align() * (size + 2.0f * pad) +
                             pad;
    OverlayStyle screen = style;
    screen.depth = OverlayDepth::Always;
    screen.anchor = Path();
    AddText(glm::dvec3(placement.at, 0.0), corner, text, screen,
            true);
    Touch();
    return *this;
}

Overlay& Overlay::Rect(const ScreenPlacement& placement,
                       const glm::vec2& size,
                       const OverlayStyle& style) {
    const glm::vec2 corner =
        placement.offset - placement.Align() * size;
    OverlayStyle screen = style;
    screen.depth = OverlayDepth::Always;
    screen.anchor = Path();
    sprites_.push_back(
        {glm::dvec3(placement.at, 0.0),
         {corner.x, corner.y, corner.x + size.x, corner.y + size.y},
         MarkerShape::Square,
         ItemOf(screen, screen.color),
         0,
         true});
    Touch();
    return *this;
}

Overlay& Overlay::Image(std::span<const glm::dvec3, 4> corners,
                        Tensor image, const OverlayStyle& style) {
    std::vector<float> points;
    for (const glm::dvec3& corner : corners) {
        points.insert(points.end(), {static_cast<float>(corner.x),
                                     static_cast<float>(corner.y),
                                     static_cast<float>(corner.z)});
    }
    OverlayBatch batch;
    batch.kind = OverlayBatchKind::Image;
    batch.points = Tensor::Copy<float>(points, {4, 3});
    batch.image = std::move(image);
    batch.style = style;
    return Batch(std::move(batch));
}

Overlay& Overlay::ScreenImage(const ScreenPlacement& placement,
                              Tensor image, const OverlayStyle& style,
                              const glm::vec2& size) {
    OverlayBatch batch;
    batch.kind = OverlayBatchKind::Image;
    batch.screen = true;
    batch.placement = placement;
    batch.imageSize = size;
    batch.image = std::move(image);
    batch.style = style;
    batch.style.depth = OverlayDepth::Always;
    return Batch(std::move(batch));
}

std::array<glm::dvec3, 4> CameraImageCorners(const Scene& scene,
                                             const Path& camera,
                                             double distance) {
    if (scene.TypeOf(camera) != tokens::kCamera ||
        !(distance > 0.0)) {
        throw std::invalid_argument("Camera image corners need a "
                                    "Camera and a distance above 0");
    }
    // The aperture's half extents at `distance`, by similar triangles
    // with the focal length; the camera looks down its -z, y up.
    const double focal =
        GetNumber(scene, camera, tokens::kFocalLength, 50.0);
    const double x = 0.5 *
                     GetNumber(scene, camera,
                               tokens::kHorizontalAperture, 20.955) /
                     focal * distance;
    const double y =
        0.5 *
        GetNumber(scene, camera, tokens::kVerticalAperture, 15.2908) /
        focal * distance;
    const glm::dmat4 pose = scene.WorldTransform(camera);
    const auto at = [&](double u, double v) {
        return glm::dvec3(pose * glm::dvec4(u, v, -distance, 1.0));
    };
    return {at(-x, y), at(x, y), at(x, -y), at(-x, -y)};
}

Overlay& Overlay::Button(const ScreenPlacement& placement,
                         std::string_view label,
                         const OverlayStyle& style,
                         const glm::vec2& size) {
    const glm::vec2 text = OverlayTextSize(label, style.size);
    const glm::vec2 pad(0.6f * style.size, 0.4f * style.size);
    const glm::vec2 box =
        size.x > 0.0f && size.y > 0.0f ? size : text + 2.0f * pad;
    const glm::vec2 corner =
        placement.offset - placement.Align() * box;
    OverlayStyle screen = style;
    screen.depth = OverlayDepth::Always;
    screen.anchor = Path();
    // The box, taking the hover colour; then the label, centred.
    sprites_.push_back(
        {glm::dvec3(placement.at, 0.0),
         {corner.x, corner.y, corner.x + box.x, corner.y + box.y},
         MarkerShape::Square,
         ItemOf(screen, screen.background),
         0,
         true});
    OverlayStyle caption = screen;
    caption.hover.reset();
    caption.background.a = 0.0f;
    AddText(glm::dvec3(placement.at, 0.0),
            corner + 0.5f * (box - text), label, caption, true);
    Touch();
    return *this;
}

std::uint32_t Overlay::ScreenItemAt(const glm::vec2& pixel,
                                    const glm::vec2& extent) const {
    // The last drawn is on top.
    for (auto it = sprites_.rbegin(); it != sprites_.rend(); ++it) {
        if (!it->screen || it->item.id == 0) {
            continue;
        }
        const glm::vec2 at =
            glm::vec2(it->anchor.x, it->anchor.y) * extent;
        if (pixel.x >= at.x + it->rect.x &&
            pixel.x < at.x + it->rect.z &&
            pixel.y >= at.y + it->rect.y &&
            pixel.y < at.y + it->rect.w) {
            return it->item.id;
        }
    }
    return 0;
}

Overlay& Overlay::SetHovered(std::uint32_t id) {
    if (id != hovered_) {
        hovered_ = id;
        Touch();
    }
    return *this;
}

std::vector<std::string>
SelectOverlays(std::string_view spec,
               std::span<const std::string> available) {
    std::set<std::string, std::less<>> named;
    std::string current;
    for (const char c : std::string(spec) + ",") {
        if (c == ',' || c == ' ' || c == '\t') {
            if (!current.empty()) {
                named.insert(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    const bool all = named.contains("*");
    std::vector<std::string> selected;
    for (const std::string& name : available) {
        if (all || named.contains(name)) {
            selected.push_back(name);
        }
    }
    return selected;
}

}  // namespace ernie
