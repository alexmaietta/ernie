// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/overlay.h"

#include <stb_easy_font.h>

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

OverlayItem ItemOf(const OverlayStyle& style,
                   const glm::vec4& color) {
    return {color, style.depth, style.world};
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

Overlay::Overlay() {
    Touch();
}

void Overlay::Touch() {
    version_ = nextVersion.fetch_add(1);
}

Overlay& Overlay::Clear() {
    segments_.clear();
    triangles_.clear();
    sprites_.clear();
    Touch();
    return *this;
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
    segments_.push_back({a, b, style.width, ItemOf(style, color)});
}

Overlay& Overlay::Line(const glm::dvec3& a, const glm::dvec3& b,
                       const OverlayStyle& style) {
    AddSegment(a, b, style, style.color);
    Touch();
    return *this;
}

Overlay& Overlay::Polyline(std::span<const glm::dvec3> points,
                           const OverlayStyle& style, bool closed) {
    for (std::size_t i = 1; i < points.size(); ++i) {
        AddSegment(points[i - 1], points[i], style, style.color);
    }
    if (closed && points.size() > 2) {
        AddSegment(points.back(), points.front(), style, style.color);
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

Overlay& Overlay::Label(const glm::dvec3& point,
                        std::string_view text,
                        const OverlayStyle& style) {
    // stb_easy_font builds each glyph from quads on a grid whose
    // capitals are 7 units tall and whose lines are 12 apart; scaled
    // so a line is `size` pixels.
    std::string ascii;
    for (const char c : text) {
        ascii.push_back(c == '\n' || (c >= 32 && c < 127) ? c : '?');
    }
    if (ascii.empty()) {
        Touch();
        return *this;
    }
    const float scale = style.size / 12.0f;
    // To the right of and above the point, so a marker there stays
    // visible.
    const float lines =
        static_cast<float>(stb_easy_font_height(ascii.data()));
    const float width =
        static_cast<float>(stb_easy_font_width(ascii.data()));
    const glm::vec2 offset(0.5f * style.size,
                           -lines * scale - 0.25f * style.size);
    if (style.background.a > 0.0f) {
        const float pad = 0.25f * style.size;
        OverlayItem item = ItemOf(style, style.background);
        sprites_.push_back({point,
                            {offset.x - pad, offset.y - pad,
                             offset.x + width * scale + pad,
                             offset.y + (lines - 3.0f) * scale + pad},
                            MarkerShape::Square,
                            item});
    }
    // About 270 bytes of quads per character, at 4 vertices of 16
    // bytes per quad.
    std::vector<char> buffer(ascii.size() * 300 + 1024);
    const int quads = stb_easy_font_print(
        0.0f, 0.0f, ascii.data(), nullptr, buffer.data(),
        static_cast<int>(buffer.size()));
    struct Vertex {
        float x, y, z;
        unsigned char color[4];
    };
    const auto* vertices =
        reinterpret_cast<const Vertex*>(buffer.data());
    for (int q = 0; q < quads; ++q) {
        const Vertex& first = vertices[4 * q];
        const Vertex& opposite = vertices[4 * q + 2];
        sprites_.push_back(
            {point,
             {offset.x + first.x * scale, offset.y + first.y * scale,
              offset.x + opposite.x * scale,
              offset.y + opposite.y * scale},
             MarkerShape::Square,
             ItemOf(style, style.color)});
    }
    Touch();
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
