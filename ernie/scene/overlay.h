// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/util/tensor.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

class Scene;

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
    /// Lines' dashes and the gaps between them, each this many pixels
    /// long; 0 for solid lines.
    float dash = 0.0f;
    float size =
        12.0f;  ///< Marker diameter, or label text height, in pixels.
    OverlayDepth depth = OverlayDepth::Test;
    std::int32_t world = -1;  ///< The world whose views show the
                              ///< item; -1 shows it in every view, or
                              ///< in its anchor's world's.
    glm::vec4 background{
        0.0f, 0.0f, 0.0f,
        0.55f};  ///< Labels' backing box; alpha 0 for none.
    /// Text's halo: an outline this many pixels wide around each
    /// character in `background`'s colour, in place of the backing
    /// box, up to a fifth of `size`; 0 for the box.
    float halo = 0.0f;
    /// The prim the items are posed under: their coordinates are in
    /// its space, and they follow it as it moves, on the GPU too
    /// (a prim posed there). Empty for world coordinates.
    Path anchor;
    /// A number the application gives the items, which hit tests and
    /// picks report (`Overlay::ScreenItemAt`), such as a button's; 0
    /// for none.
    std::uint32_t id = 0;
    /// The colour items with `id` take while it is hovered
    /// (`Overlay::SetHovered`); none keeps their colour.
    std::optional<glm::vec4> hover;
    /// Markers' and text's `size` in metres rather than pixels: in
    /// the world they grow and shrink with distance as things there
    /// do, still facing the view.
    bool metric = false;
};

/// Where a screen item (`Overlay::Text`, `Overlay::Rect`) sits on
/// the image, whatever its size: the point `at` (fractions of the
/// image's width and height from its top left corner) moved `offset`
/// pixels (y down), with the item's own point `align` (fractions of
/// its width and height from its top left) there. Without `align`,
/// each axis keeps the item on the side of `at` away from the nearer
/// edge: its left (top) edge at `at` in the image's left (top) half,
/// its right (bottom) in the other, its centre at the centre; so
/// `at` {1, 0} puts it in the top right corner, and {0, 0.3} hangs it
/// down the left side from there.
struct ScreenPlacement {
    glm::vec2 at{0.0f};
    glm::vec2 offset{0.0f};
    std::optional<glm::vec2> align;

    /// `align`, else `at` snapped to 0, 0.5 or 1 on each axis.
    glm::vec2 Align() const {
        if (align) {
            return *align;
        }
        const auto snap = [](float t) {
            return t < 0.5f ? 0.0f : (t > 0.5f ? 1.0f : 0.5f);
        };
        return {snap(at.x), snap(at.y)};
    }
};

/// What every overlay item carries besides its geometry.
struct OverlayItem {
    glm::vec4 color{1.0f};
    OverlayDepth depth = OverlayDepth::Test;
    std::int32_t world = -1;
    /// 1 + the overlay's anchor (`Overlay::Anchors`) its coordinates
    /// are relative to, 0 for none.
    std::uint32_t anchor = 0;
    std::uint32_t id = 0;            ///< `OverlayStyle::id`.
    std::optional<glm::vec4> hover;  ///< `OverlayStyle::hover`.
    bool metric = false;             ///< `OverlayStyle::metric`.
};

/// A line segment of `width` pixels, dashed every `dash` pixels
/// (`OverlayStyle::dash`) unless 0.
struct OverlaySegment {
    glm::dvec3 a{0.0};
    glm::dvec3 b{0.0};
    float width = 2.0f;
    OverlayItem item;
    float dash = 0.0f;
    /// The colour at `b`, blended from `item.color` at `a`; none for
    /// one colour.
    std::optional<glm::vec4> endColor;
};

/// A filled triangle.
struct OverlayTriangle {
    std::array<glm::dvec3, 3> points{};
    OverlayItem item;
};

/// A rectangle facing the image at a fixed size in pixels, placed by
/// a world point: `rect` spans (x0, y0) to (x1, y1) pixels from where
/// `anchor` projects, y down. Markers, text and its backing boxes.
/// A screen sprite's `anchor` is instead a point of the image, x and
/// y fractions of it from its top left, drawn over everything.
struct OverlaySprite {
    glm::dvec3 anchor{0.0};
    glm::vec4 rect{0.0f};
    MarkerShape shape = MarkerShape::Square;
    OverlayItem item;
    /// A character of text (a Latin-1 code, `OverlayTextFont`)
    /// filling `rect` in place of the shape, or 0.
    std::uint8_t glyph = 0;
    bool screen = false;
    /// A character's halo (`OverlayStyle::halo`): pixels wide, and
    /// its colour.
    float halo = 0.0f;
    glm::vec4 haloColor{0.0f};
};

/// The font overlay text is drawn in, Inter, baked into signed
/// distance fields, so it stays sharp at any size and can take a
/// halo. Text is Latin-1 (given as UTF-8, anything else drawn as
/// '?'), laid out on a grid of which a line is `kOverlayLineHeight`
/// units, y down from the top of its capitals, and scaled so a line
/// is the style's `size` pixels.
inline constexpr float kOverlayLineHeight = 12.0f;
/// A field's texels per unit (40 a line); how far it reaches past
/// the outlines, in texels; and its levels per texel of distance, 128
/// on the outline and more inside.
inline constexpr float kOverlayFieldTexels =
    40.0f / kOverlayLineHeight;
inline constexpr int kOverlayFieldPadding = 8;
inline constexpr float kOverlayFieldLevels = 16.0f;

/// One character of the font (`OverlayTextFont`).
struct OverlayGlyph {
    /// The box its field covers, in units from the pen: x0, y0, x1,
    /// y1, y down.
    glm::vec4 box{0.0f};
    /// How far it moves the pen, in units.
    float advance = 0.0f;
    /// Its field's first texel in `OverlayFont::field`, and its
    /// texels across and down; 0 for a character with no outline,
    /// such as a space.
    std::uint32_t texel = 0;
    std::uint16_t width = 0;
    std::uint16_t height = 0;
};

/// First and number of codes the font's kerning covers, ' ' to 'ÿ'.
inline constexpr std::uint32_t kOverlayKernFirst = 32;
inline constexpr std::uint32_t kOverlayKernCodes = 224;

struct OverlayFont {
    /// By Latin-1 code; codes the font lacks are '?'.
    std::array<OverlayGlyph, 256> glyphs{};
    /// The pen's move between codes a and b besides a's advance,
    /// in sixteenths of a unit, at (a - kOverlayKernFirst) *
    /// kOverlayKernCodes + b - kOverlayKernFirst.
    std::vector<std::int8_t> kerning;
    /// Every glyph's distance field, row by row from its top.
    std::vector<std::uint8_t> field;
    /// A line's text from the top of its capitals to the bottom of
    /// its descenders, in units.
    float textHeight = 0.0f;

    /// The kerning between codes `a` and `b`, in units.
    float Kerning(std::uint8_t a, std::uint8_t b) const;
};

/// The font, baked on first use.
const OverlayFont& OverlayTextFont();

/// `text` (UTF-8) as the font's Latin-1 codes, '\n' kept, '?' for
/// what it lacks.
std::vector<std::uint8_t> OverlayCodes(std::string_view text);

/// Width and height in pixels of `text` (lines split at '\n') in the
/// font at `size` pixels a line.
glm::vec2 OverlayTextSize(std::string_view text, float size);

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
    Numbers,  ///< `values` printed in `text` at `points` (or, on
              ///< the screen without points, at `placement`, a
              ///< line apart), formatted on the GPU.
    Trails,   ///< A polyline per item through its row of `points`
              ///< (n, length, 3), a ring buffer whose newest point is
              ///< at `heads`, fading to its oldest with `fade`; a
              ///< segment with a NaN end is left out.
    Capsules,  ///< Capsules around `points`, along their local z axis
               ///< turned by `orientations`: a cylinder of `radii`
               ///< reaching `halfHeights` each way, capped by
               ///< hemispheres.
    Texts,     ///< Each row of `chars` as text, at `points` as
               ///< labels' or on the screen as `Numbers`'.
    Image,     ///< `image` on a quad: in the world between the four
               ///< `points` (its top left, top right, bottom right
               ///< and bottom left corners), or on the screen at
               ///< `placement`, `imageSize` pixels.
};

/// How a `Numbers` batch prints each of its values: `prefix`, the
/// value with `decimals` digits after the point, then `suffix`, in
/// the style's colour and `size`, over its `background`. In the world
/// the text sits beside its point as a label's; on the screen its
/// point `align` (fractions of its box) lands on the item's place.
struct OverlayNumberFormat {
    std::string prefix;
    std::string suffix;
    int decimals = 2;
    glm::vec2 align{0.0f};
};

/// Many overlay items of one kind from arrays on the host or a GPU
/// (`Overlay::Batch`), such as one per world of a batched
/// simulation, or a physics engine's bodies, contacts and particles:
/// the renderer reads the arrays as they are, without copies to the
/// host, and draws each item from them on the GPU. Arrays have one
/// row per item: float32 but `worlds`, `heads`, `poseIndices` and
/// `count` (int32), and contiguous; on a GPU, all on the renderer's
/// compute device, read again every frame. Optional ones are empty
/// when absent, the style's value standing for them.
struct OverlayBatch {
    OverlayBatchKind kind = OverlayBatchKind::Lines;
    /// (n, 3); the number of items. Screen items' x and y are
    /// fractions of the image (`ScreenPlacement::at`). Trails' (n,
    /// length, 3).
    Tensor points;
    /// (n, 3): lines' and arrows' ends, or with `directions` vectors
    /// from their points.
    Tensor ends;
    Tensor halfExtents;   ///< (n, 3): boxes'.
    Tensor orientations;  ///< (n, 4) quaternions x, y, z, w: boxes',
                          ///< frames' and capsules', optional.
    Tensor radii;         ///< (n,): spheres' and capsules'.
    /// (n,): capsules' half the length of their cylinders.
    Tensor halfHeights;
    /// (n,) int32: the index in each trail's row of its newest point;
    /// without it, the last.
    Tensor heads;
    /// Trails fade from opaque at their newest point to clear at
    /// their oldest.
    bool fade = false;
    /// `ends` are vectors from the points, scaled by `length`:
    /// velocities or forces drawn from a simulation's own arrays.
    bool directions = false;
    /// (1,) int32: how many of the rows are in use, the first, such
    /// as the contacts a simulation found this step (its contact
    /// count); the rest are left out, the GPU drawing only those.
    /// Every row without it.
    Tensor count;
    /// (m, 7) float32 poses, as a physics engine's bodies' (position,
    /// then quaternion x, y, z, w; Warp's `transform`, Newton's
    /// `body_q`): the items are posed under them, item i under pose
    /// `poseIndices[i]` (or pose i without them), turned and moved on
    /// the GPU every frame, then by `poseFrame`.
    Tensor poses;
    Tensor poseIndices;  ///< (n,) int32: each item's pose.
    /// (m,) int32: each pose's world (Newton's `body_world`), the
    /// world of the items under it without worlds of their own.
    Tensor poseWorlds;
    /// The poses' frame in the world, such as an engine's Y-up model
    /// turned Z-up.
    glm::dmat4 poseFrame{1.0};
    /// (n,): numbers'; their number when they have no points.
    Tensor values;
    /// (n, k) int32: texts' character codes (printable ASCII), each
    /// row up to its first 0; their number when they have no points.
    Tensor chars;
    /// (rows, cols, 3 or 4) uint8: an image's display (sRGB-encoded)
    /// colours and opacity, row 0 at its top, such as another
    /// camera's LdrColor; filtered between texels, times the style's
    /// colour.
    Tensor image;
    /// A screen image's size in pixels; its own when 0.
    glm::vec2 imageSize{0.0f};
    OverlayNumberFormat format;  ///< Numbers'.
    /// Numbers drawn on the image rather than in the world, over
    /// everything.
    bool screen = false;
    /// Where screen numbers without points go: item i a line below
    /// item i - 1.
    ScreenPlacement placement;
    /// The prims the items are posed under (`OverlayStyle::anchor`):
    /// one for every item, or one per item, such as a link of each
    /// world's robot. The style's anchor without them.
    std::vector<Path> anchors;
    /// (n, 3) or (n, 4) display (sRGB-encoded) colours, and opacity;
    /// a trail's may be one per point, (n, length, 3 or 4), each
    /// segment taking its newer point's.
    Tensor colors;
    /// (n,) pixels: line widths, or markers' sizes.
    Tensor sizes;
    /// (n,) int32: the world whose views show each. Without them and
    /// a style's world, items posed under prims or poses take their
    /// anchors' (`tokens::kWorld`, `poseWorlds`).
    Tensor worlds;
    /// Arrows' heads' length; 0 makes each a fifth of its arrow.
    double head = 0.0;
    /// Frames' axis length; the scale of `directions`.
    double length = 1.0;
    MarkerShape shape = MarkerShape::Circle;
    OverlayStyle style;
    /// The point the arrays' coordinates are relative to, so items
    /// far from the world's origin keep float32's precision.
    glm::dvec3 origin{0.0};
    /// The CUDA stream GPU arrays are written on; copies follow it.
    std::uintptr_t stream = 1;

    /// The number of items: rows, of which `count` may use fewer.
    std::size_t Count() const;

    /// Points in each trail's row; 1 for other kinds.
    std::size_t TrailLength() const;
};

/// A named layer of debug drawing: lines, shapes, markers and labels
/// in world coordinates or posed under prims
/// (`OverlayStyle::anchor`), and text and panels on the image itself
/// (`Text`, `Rect`), drawn over the images of render products that
/// ask for it
/// (`tokens::kOverlays`), after the sensor filters, so they never
/// reach observations that do not. Items stay until `Clear`; a
/// simulation redrawing each step clears and adds again, or keeps
/// batches of GPU arrays it writes (`Batch`).
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
    /// `closed`; with `colors` (one per point), each blending from
    /// its start's to its end's, such as a trail coloured by speed.
    /// @throws std::invalid_argument for a colour count that differs
    /// from the point count.
    Overlay& Polyline(std::span<const glm::dvec3> points,
                      const OverlayStyle& style = {},
                      bool closed = false,
                      std::span<const glm::vec4> colors = {});

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

    /// An arc of the circle of `radius` around `centre` in the plane
    /// facing `normal`, from direction `from` (projected into that
    /// plane) turning `angle` radians about `normal`, counter-
    /// clockwise looking down it; with `filled`, the sector it
    /// sweeps, in triangles of the style's colour, under the arc.
    /// @throws std::invalid_argument for a normal or `from` of no
    /// length in the plane, or fewer than 1 segment.
    Overlay& Arc(const glm::dvec3& centre, const glm::dvec3& normal,
                 const glm::dvec3& from, double radius, double angle,
                 const OverlayStyle& style = {}, bool filled = false,
                 int segments = 32);

    /// The cubic Bezier curve from `p0` to `p3`, pulled toward `p1`
    /// and `p2`, in `segments` segments.
    /// @throws std::invalid_argument for fewer than 1 segment.
    Overlay& Curve(const glm::dvec3& p0, const glm::dvec3& p1,
                   const glm::dvec3& p2, const glm::dvec3& p3,
                   const OverlayStyle& style = {}, int segments = 32);

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

    /// `text` (ASCII; lines split at '\n') `style.size` pixels a line
    /// on the image itself, a heads-up display, over a backing box of
    /// `style.background`, drawn over everything in every view (of
    /// `style.world`'s world, if set).
    Overlay& Text(const ScreenPlacement& placement,
                  std::string_view text,
                  const OverlayStyle& style = {});

    /// A filled rectangle of `size` pixels on the image, as `Text`,
    /// in the style's colour: a panel, or a bar.
    Overlay& Rect(const ScreenPlacement& placement,
                  const glm::vec2& size,
                  const OverlayStyle& style = {});

    /// `image` (`OverlayBatch::image`) on the quad between `corners`
    /// in the world: its top left, top right, bottom right and bottom
    /// left, in the space of the style's `anchor` if it has one. A
    /// GPU image is read again every frame, such as another camera's
    /// output written in place.
    /// @throws std::invalid_argument as `Batch`.
    Overlay& Image(std::span<const glm::dvec3, 4> corners,
                   Tensor image, const OverlayStyle& style = {});

    /// `image` on the screen, as `Rect`, `size` pixels (its own size
    /// when 0): a picture in the picture.
    /// @throws std::invalid_argument as `Batch`.
    Overlay& ScreenImage(const ScreenPlacement& placement,
                         Tensor image, const OverlayStyle& style = {},
                         const glm::vec2& size = glm::vec2(0.0f));

    /// A button on the image, as `Text`: a box of `size` pixels (or
    /// fitted around `label` when 0) in the style's `background`, or
    /// its `hover` colour while hovered, with `label` centred in its
    /// `color`. Give it the style's `id`, and hit test it
    /// (`ScreenItemAt`) or let the viewer report its clicks.
    Overlay& Button(const ScreenPlacement& placement,
                    std::string_view label,
                    const OverlayStyle& style = {},
                    const glm::vec2& size = glm::vec2(0.0f));

    /// The `id` of the topmost screen item with one covering `pixel`
    /// (y down) of an image of `extent` pixels, or 0: what a click or
    /// a pointer there would hit.
    std::uint32_t ScreenItemAt(const glm::vec2& pixel,
                               const glm::vec2& extent) const;

    /// Marks the items of `id` hovered (0 for none), so those with a
    /// `hover` colour take it; a new version only when it changes.
    Overlay& SetHovered(std::uint32_t id);

    std::uint32_t Hovered() const {
        return hovered_;
    }

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
    /// GPU ones are read again every frame; `Touch` it after writing
    /// into host ones in place.
    /// @throws std::invalid_argument for arrays of the wrong type or
    /// shape, rows that differ in number, ones the kind needs
    /// missing, or anchors neither one nor one per item.
    Overlay& Batch(OverlayBatch batch);

    /// Replaces array `array` ("points", "ends", "values"...: the
    /// field's name) of batch `batch` with `values` of the same type
    /// and shape, such as the renderer's own memory for it
    /// (`RenderDelegate::MapOverlay`), without a new version: the
    /// batch's layout stays.
    /// @throws std::invalid_argument for a batch or array that does
    /// not exist, or values of another type or shape.
    void ReplaceBatchArray(std::size_t batch, std::string_view array,
                           Tensor values);

    /// The prims items are posed under, which their
    /// `OverlayItem::anchor` count from 1.
    const std::vector<Path>& Anchors() const {
        return anchors_;
    }

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

    /// The item of `style` in `color`, with its anchor's index.
    OverlayItem ItemOf(const OverlayStyle& style,
                       const glm::vec4& color);

    /// Sprites of `text`'s characters and its backing box, its top
    /// left corner `corner` pixels from `anchor`'s place.
    void AddText(const glm::dvec3& anchor, const glm::vec2& corner,
                 std::string_view text, const OverlayStyle& style,
                 bool screen);

    std::vector<OverlaySegment> segments_;
    std::vector<OverlayTriangle> triangles_;
    std::vector<OverlaySprite> sprites_;
    std::vector<OverlayBatch> batches_;
    std::vector<Path> anchors_;
    std::uint32_t hovered_ = 0;
    bool visible_ = true;
    std::uint64_t version_ = 0;
};

/// What a pick of a rendered image found at a pixel
/// (`RenderDelegate::PickOverlay`): the overlay drawing it, and in it
/// a batch's item (`batch` the batch, `item` its index) or a host
/// item (`batch` -1, `item` its `OverlayStyle::id`).
struct OverlayHit {
    std::string overlay;
    std::int32_t batch = -1;
    std::uint32_t item = 0;
};

/// The corners of the image of the Camera at `camera`, `distance`
/// metres in front of it: its frustum's section there, from its focal
/// length and apertures, in the world, top left first as
/// `Overlay::Image` takes them. Its image drawn there sits where it
/// was seen from.
/// @throws std::invalid_argument for a prim that is no Camera or a
/// distance not above 0.
std::array<glm::dvec3, 4> CameraImageCorners(const Scene& scene,
                                             const Path& camera,
                                             double distance);

/// The overlay names `spec` selects among `available`: "*" every one,
/// else the names it lists, separated by commas or spaces, in the
/// order of `available`.
std::vector<std::string>
SelectOverlays(std::string_view spec,
               std::span<const std::string> available);

}  // namespace ernie
