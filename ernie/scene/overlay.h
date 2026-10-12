// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/tensor.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// How an overlay item meets the scene's surfaces.
enum class OverlayDepth : std::uint8_t {
    Test,    ///< Hidden behind surfaces, like scene geometry.
    XRay,    ///< Drawn faded where surfaces hide it.
    Always,  ///< Drawn over everything.
};

/// The shape of a marker, drawn at a fixed size in pixels.
enum class MarkerShape : std::uint8_t {
    Circle,   ///< A filled disc.
    Ring,     ///< A circle's outline.
    Square,   ///< A filled square, or any rectangle the sprite spans.
    Diamond,  ///< A filled square on its corner.
    Cross,    ///< An x.
    Plus,     ///< A +.
};

/// How overlay items look.
struct OverlayStyle {
    glm::vec4 color{
        1.0f};  ///< Display (sRGB-encoded) colour and opacity.
    float width = 2.0f;  ///< Line width in pixels.
    float size =
        12.0f;  ///< Marker diameter, or label text height, in pixels.
    OverlayDepth depth = OverlayDepth::Test;
    std::int32_t world = -1;  ///< The world whose views show the
                              ///< item; -1 shows it in every view.
    glm::vec4 background{
        0.0f, 0.0f, 0.0f,
        0.55f};  ///< Labels' backing box; alpha 0 for none.
};

/// What every overlay item carries besides its geometry.
struct OverlayItem {
    glm::vec4 color{1.0f};
    OverlayDepth depth = OverlayDepth::Test;
    std::int32_t world = -1;
};

/// A line segment of `width` pixels.
struct OverlaySegment {
    glm::dvec3 a{0.0};
    glm::dvec3 b{0.0};
    float width = 2.0f;
    OverlayItem item;
};

/// A filled triangle.
struct OverlayTriangle {
    std::array<glm::dvec3, 3> points{};
    OverlayItem item;
};

/// A rectangle facing the image at a fixed size in pixels, placed by
/// a world point: `rect` spans (x0, y0) to (x1, y1) pixels from where
/// `anchor` projects, y down. Markers and label text.
struct OverlaySprite {
    glm::dvec3 anchor{0.0};
    glm::vec4 rect{0.0f};
    MarkerShape shape = MarkerShape::Square;
    OverlayItem item;
};

/// The kinds of item an `OverlayBatch` holds many of.
enum class OverlayBatchKind : std::uint8_t {
    Lines,    ///< Segments from `points` to `ends`.
    Arrows,   ///< Lines with a cone at `ends`.
    Markers,  ///< Markers of `shape` at `points`.
    Boxes,    ///< Box edges around `points`, `halfExtents` out,
              ///< turned by `orientations`.
    Spheres,  ///< Three great circles of `radii` around `points`.
    Frames,   ///< The x, y and z axes at `points`, turned by
              ///< `orientations`, `length` long, in red, green and
              ///< blue.
};

/// Many overlay items of one kind from arrays on the host or a GPU
/// (`Overlay::Batch`), such as one per world of a batched
/// simulation: the renderer reads the arrays as they are, without
/// copies to the host, and draws each item from them on the GPU.
/// Arrays have one row per item: float32 but `worlds` (int32), and
/// contiguous; on a GPU, all on the renderer's compute device.
/// Optional ones are empty when absent, the style's value standing
/// for them.
struct OverlayBatch {
    OverlayBatchKind kind = OverlayBatchKind::Lines;
    Tensor points;        ///< (n, 3); the number of items.
    Tensor ends;          ///< (n, 3): lines' and arrows'.
    Tensor halfExtents;   ///< (n, 3): boxes'.
    Tensor orientations;  ///< (n, 4) quaternions x, y, z, w: boxes'
                          ///< and frames', optional.
    Tensor radii;         ///< (n,): spheres'.
    /// (n, 3) or (n, 4) display (sRGB-encoded) colours, and opacity.
    Tensor colors;
    /// (n,) pixels: line widths, or markers' sizes.
    Tensor sizes;
    Tensor worlds;  ///< (n,) int32: the world whose views show each.
    /// Arrows' heads' length; 0 makes each a fifth of its arrow.
    double head = 0.0;
    double length = 1.0;  ///< Frames' axis length.
    MarkerShape shape = MarkerShape::Circle;
    OverlayStyle style;
    /// The point the arrays' coordinates are relative to, so items
    /// far from the world's origin keep float32's precision.
    glm::dvec3 origin{0.0};
    /// The CUDA stream GPU arrays are written on; copies follow it.
    std::uintptr_t stream = 1;

    /// The number of items.
    std::size_t Count() const;
};

/// A named layer of debug drawing: lines, shapes, markers and labels
/// in world coordinates, drawn over the images of render products
/// that ask for it (`tokens::kOverlays`), after the sensor filters,
/// so they never reach observations that do not. Items stay until
/// `Clear`; a simulation redrawing each step clears and adds again.
///
/// Every shape reduces to segments, triangles and sprites
/// (`Segments`, `Triangles`, `Sprites`), which the renderer draws as
/// they are.
class Overlay {
  public:
    Overlay();

    /// Removes every item.
    Overlay& Clear();

    bool IsVisible() const {
        return visible_;
    }

    /// Hides or shows the layer everywhere, keeping its items.
    Overlay& SetVisible(bool visible);

    /// Advances with every change, and differs between overlays.
    std::uint64_t Version() const {
        return version_;
    }

    Overlay& Line(const glm::dvec3& a, const glm::dvec3& b,
                  const OverlayStyle& style = {});

    /// Segments through `points` in order, back to the first with
    /// `closed`.
    Overlay& Polyline(std::span<const glm::dvec3> points,
                      const OverlayStyle& style = {},
                      bool closed = false);

    /// Many segments at once: `endpoints` in pairs, each segment
    /// coloured by `colors` when given (one per segment), else by the
    /// style.
    /// @throws std::invalid_argument for an odd endpoint count or a
    /// colour count that differs.
    Overlay& Lines(std::span<const glm::dvec3> endpoints,
                   const OverlayStyle& style = {},
                   std::span<const glm::vec4> colors = {});

    /// A line from `from` to `to` with a cone for its head, `head`
    /// long (a fifth of the arrow when 0).
    Overlay& Arrow(const glm::dvec3& from, const glm::dvec3& to,
                   const OverlayStyle& style = {}, double head = 0.0);

    /// The edges of the box from `low` to `high`, placed by
    /// `transform`.
    Overlay& Box(const glm::dvec3& low, const glm::dvec3& high,
                 const OverlayStyle& style = {},
                 const glm::dmat4& transform = glm::dmat4(1.0));

    /// A circle of `radius` around `centre`, in the plane facing
    /// `normal`.
    Overlay& Circle(const glm::dvec3& centre,
                    const glm::dvec3& normal, double radius,
                    const OverlayStyle& style = {},
                    int segments = 48);

    /// Three great circles of a sphere, one around each axis.
    Overlay& Sphere(const glm::dvec3& centre, double radius,
                    const OverlayStyle& style = {});

    /// The x, y and z axes of `transform`, `length` long, in red,
    /// green and blue.
    Overlay& Axes(const glm::dmat4& transform, double length,
                  const OverlayStyle& style = {});

    /// The edges of the triangles `indices` (in threes) over
    /// `points`, placed by `transform`, each shared edge once.
    /// @throws std::invalid_argument for an index count not a
    /// multiple of 3 or an index out of range.
    Overlay& WireMesh(std::span<const glm::dvec3> points,
                      std::span<const std::uint32_t> indices,
                      const OverlayStyle& style = {},
                      const glm::dmat4& transform = glm::dmat4(1.0));

    /// The filled triangles `indices` (in threes) over `points`,
    /// placed by `transform`.
    /// @throws std::invalid_argument as `WireMesh`.
    Overlay& Triangles(std::span<const glm::dvec3> points,
                       std::span<const std::uint32_t> indices,
                       const OverlayStyle& style = {},
                       const glm::dmat4& transform = glm::dmat4(1.0));

    /// A marker of `style.size` pixels at `point`.
    Overlay& Marker(const glm::dvec3& point,
                    const OverlayStyle& style = {},
                    MarkerShape shape = MarkerShape::Circle);

    /// Markers at every point, each coloured by `colors` when given
    /// (one per point).
    /// @throws std::invalid_argument for a colour count that differs
    /// from the point count.
    Overlay& Markers(std::span<const glm::dvec3> points,
                     const OverlayStyle& style = {},
                     MarkerShape shape = MarkerShape::Circle,
                     std::span<const glm::vec4> colors = {});

    /// `text` (ASCII; lines split at '\n') `style.size` pixels tall,
    /// beside `point`, over a backing box of `style.background`.
    Overlay& Label(const glm::dvec3& point, std::string_view text,
                   const OverlayStyle& style = {});

    /// Many items of one kind from arrays (`OverlayBatch`), kept as
    /// they are and drawn on the GPU. The overlay holds the arrays:
    /// `Touch` it after writing into them in place.
    /// @throws std::invalid_argument for arrays of the wrong type or
    /// shape, rows that differ in number, or ones the kind needs
    /// missing.
    Overlay& Batch(OverlayBatch batch);

    const std::vector<OverlaySegment>& Segments() const {
        return segments_;
    }

    /// The filled triangles, of `Triangles` and arrow heads.
    const std::vector<OverlayTriangle>& Fills() const {
        return triangles_;
    }

    const std::vector<OverlaySprite>& Sprites() const {
        return sprites_;
    }

    const std::vector<OverlayBatch>& Batches() const {
        return batches_;
    }

    /// Whether the layer has no items.
    bool Empty() const {
        return segments_.empty() && triangles_.empty() &&
               sprites_.empty() && batches_.empty();
    }

    /// Marks the layer changed, so the renderer reads its batches'
    /// arrays again: after writing into them in place.
    Overlay& Touch();

  private:
    void AddSegment(const glm::dvec3& a, const glm::dvec3& b,
                    const OverlayStyle& style,
                    const glm::vec4& color);

    std::vector<OverlaySegment> segments_;
    std::vector<OverlayTriangle> triangles_;
    std::vector<OverlaySprite> sprites_;
    std::vector<OverlayBatch> batches_;
    bool visible_ = true;
    std::uint64_t version_ = 0;
};

/// The overlay names `spec` selects among `available`: "*" every one,
/// else the names it lists, separated by commas or spaces, in the
/// order of `available`.
std::vector<std::string>
SelectOverlays(std::string_view spec,
               std::span<const std::string> available);

}  // namespace ernie
