// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/device/slang/dispatch.h"
#include "ernie/render/kernels.h"
#include "ernie/render/overlay/overlay_impl.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

namespace {

using Batch = OverlayData::Batch;

constexpr uint64_t kLeastPairs = 1 << 16;

struct Layout {
    int64_t worlds = 0;
    int64_t rows = 0;
    int64_t samples = 0;
};

struct HostRow {
    Float3 point;
    Float3 end;
    Float4 rotation = {0, 0, 0, 1};
    float scalar = 0;
    uint32_t extra = 0;
};

uint32_t PackColor(Float4 color) {
    return PackOverlayColor(color.x, color.y, color.z, color.w);
}

Float4 Rotation(Quatd q) {
    return {float(q.x), float(q.y), float(q.z), float(q.w)};
}

std::string_view KindName(OverlayKind kind) {
    constexpr std::string_view kNames[] = {
        "lines",  "markers", "boxes", "spheres", "circles",   "frames", "arrows", "capsules",
        "trails", "labels",  "texts", "numbers", "triangles", "images", "prims",  "outlines"};
    return kNames[uint32_t(kind)];
}

uint32_t PrimPieces(OverlayPrimShape shape) {
    switch (shape) {
    case OverlayPrimShape::Frames:
        return 3;
    case OverlayPrimShape::Bounds:
        return 12;
    case OverlayPrimShape::Frustums:
        return kOverlayFrustumLines;
    case OverlayPrimShape::Lights:
        return kOverlayLightLines;
    case OverlayPrimShape::CameraImages:
        return 2;
    }
    return 1;
}

uint32_t PiecesPerRow(OverlayKind kind, uint32_t samples) {
    switch (kind) {
    case OverlayKind::Images:
        return 2;
    case OverlayKind::Boxes:
        return 12;
    case OverlayKind::Spheres:
        return 3 * kOverlaySphereSegments;
    case OverlayKind::Circles:
        return kOverlayCircleSegments;
    case OverlayKind::Frames:
        return 3;
    case OverlayKind::Arrows:
        return 1 + kOverlayArrowHeadLines;
    case OverlayKind::Capsules:
        return 2 * kOverlayCapsuleSegments + 4 + 4 * kOverlayCapsuleArcSegments;
    case OverlayKind::Trails:
        return samples - 1;
    default:
        return 1;
    }
}

bool Textual(OverlayKind kind) {
    return kind == OverlayKind::Labels || kind == OverlayKind::Texts ||
           kind == OverlayKind::Numbers;
}

bool Hosted(const Batch& batch) {
    return batch.points.Empty() && batch.chars.Empty() && batch.scalars.Empty() &&
           batch.prims.Empty();
}

bool SameKey(const Batch& batch, OverlayKind kind, const OverlayStyle& style) {
    const OverlayStyle& key = batch.style;
    return Hosted(batch) && batch.kind == kind && kind != OverlayKind::Images &&
           kind != OverlayKind::Prims && kind != OverlayKind::Outlines &&
           key.depth == style.depth && key.anchor == style.anchor && key.dash == style.dash &&
           key.gap == style.gap && key.units == style.units &&
           key.screenSized == style.screenSized && key.facing == style.facing &&
           key.pickable == style.pickable &&
           (kind != OverlayKind::Markers || key.marker == style.marker) &&
           (!Textual(kind) || (key.background == style.background && key.halo == style.halo &&
                               key.align == style.align && key.width == style.width));
}

uint32_t PackPattern(float pixels) {
    return uint32_t(std::clamp(std::lround(pixels), 0l, long(kOverlayPatternMask)));
}

struct ImageLayout {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t rowStride = 0;
    uint32_t layerStride = 0;
    bool layered = false;
};

void PackAffix(std::string_view text, uint32_t (&words)[4], uint32_t& length,
               std::string_view what) {
    const std::vector<uint8_t> codes = OverlayCodes(text);
    if (codes.size() > kOverlayAffixChars) {
        throw Error("An overlay number's {} \"{}\" has {} characters; at most {} fit", what, text,
                    codes.size(), kOverlayAffixChars);
    }
    for (size_t at = 0; at < codes.size(); ++at) {
        words[at / 4] |= uint32_t(codes[at]) << (8 * (at % 4));
    }
    length = uint32_t(codes.size());
}

Batch& HostBatch(OverlayData& contents, OverlayKind kind, const OverlayStyle& style) {
    if (contents.batches.empty() || !SameKey(contents.batches.back(), kind, style)) {
        contents.batches.push_back({.kind = kind, .style = style});
    }
    return contents.batches.back();
}

void AppendRow(Batch& batch, const HostRow& row, uint32_t color, float size, uint32_t world,
               uint32_t id) {
    batch.hostIds.push_back(id);
    batch.hostPoints.push_back(row.point);
    batch.hostEnds.push_back(row.end);
    batch.hostColors.push_back(color);
    batch.hostSizes.push_back(size);
    batch.hostWorlds.push_back(world);
    batch.hostRotations.push_back(row.rotation);
    batch.hostScalars.push_back(row.scalar);
    batch.hostExtras.push_back(row.extra);
}

void AppendText(Batch& batch, std::string_view text, const HostRow& row,
                const OverlayStyle& style) {
    std::vector<uint8_t> codes = OverlayCodes(text);
    codes.resize(std::min<size_t>(codes.size(), kOverlayTextChars));
    const auto start = uint32_t(batch.hostCodes.size());
    if (start + codes.size() >= (1u << 24)) {
        throw UnsupportedError("An overlay's texts hold more than {} characters", 1u << 24);
    }
    batch.hostCodes.insert(batch.hostCodes.end(), codes.begin(), codes.end());
    batch.maxChars = std::max(batch.maxChars, uint32_t(codes.size()));
    HostRow packed = row;
    packed.extra = start | uint32_t(codes.size()) << 24;
    AppendRow(batch, packed, PackColor(style.color), style.size, style.world, style.id);
}

int64_t Stride(const device::Tensor& tensor, size_t dim) {
    if (!tensor.Strides().empty()) {
        return tensor.Strides()[dim];
    }
    int64_t stride = 1;
    for (size_t after = dim + 1; after < tensor.Shape().size(); ++after) {
        stride *= tensor.Shape()[after];
    }
    return stride;
}

Layout CheckPoints(const device::Tensor& points, std::string_view batch, bool trail) {
    if (points.Empty()) {
        throw Error("Overlay {} need points", batch);
    }
    const Span<const int64_t> shape = points.Shape();
    const bool shaped =
        trail ? shape.size() == 3 && shape[1] >= 2 : shape.size() == 2 || shape.size() == 3;
    if (!shaped || shape.back() != 3 || points.Type() != device::DType::Float32) {
        if (trail) {
            throw Error("Overlay {}' points are an [N, samples, 3] float32 tensor of at least 2 "
                        "samples",
                        batch);
        }
        throw Error("Overlay {}' points are an [N, 3] or [worlds, N, 3] float32 tensor", batch);
    }
    if (Stride(points, shape.size() - 1) != 1) {
        throw UnsupportedError("Overlay {}' points need their x, y and z side by side", batch);
    }
    const int64_t rows = trail ? shape[0] : shape[shape.size() - 2];
    if (rows > std::numeric_limits<uint32_t>::max()) {
        throw Error("Overlay {} of {} rows; at most {} fit", batch, rows,
                    std::numeric_limits<uint32_t>::max());
    }
    if (trail) {
        return {0, rows, shape[1]};
    }
    return {shape.size() == 3 ? shape[0] : 0, rows, 0};
}

void CheckField(const device::Tensor& field, const device::Tensor& points, Layout layout,
                std::string_view batch, std::string_view name, int64_t width,
                std::initializer_list<device::DType> types) {
    if (field.Empty()) {
        return;
    }
    std::vector<int64_t> expected;
    if (layout.worlds > 0) {
        expected.push_back(layout.worlds);
    }
    expected.push_back(layout.rows);
    if (width > 0) {
        expected.push_back(width);
    }
    const bool typed = std::ranges::find(types, field.Type()) != types.end();
    if (!typed || !std::ranges::equal(field.Shape(), expected)) {
        std::string names;
        for (const device::DType type : types) {
            names += std::string(names.empty() ? "" : " or ") +
                     (type == device::DType::Float32 ? "float32"
                      : type == device::DType::UInt8 ? "uint8"
                      : type == device::DType::Int32 ? "int32"
                                                     : "uint32");
        }
        throw Error("Overlay {}' {} are a [{}N{}] {} tensor, a row per point", batch, name,
                    layout.worlds > 0 ? "worlds, " : "",
                    width > 0 ? ", " + std::to_string(width) : "", names);
    }
    if (&field.GetDevice() != &points.GetDevice()) {
        throw Error("Overlay {}' {} are on another device than their points", batch, name);
    }
    if (width > 0 && Stride(field, expected.size() - 1) != 1) {
        throw UnsupportedError("Overlay {}' {} need their channels side by side", batch, name);
    }
    if (field.Type() == device::DType::UInt8) {
        for (size_t dim = 0; dim + 1 < expected.size(); ++dim) {
            if (Stride(field, dim) % 4 != 0) {
                throw UnsupportedError("Overlay {}' uint8 {} need rows a multiple of 4 bytes apart",
                                       batch, name);
            }
        }
    }
}

void Require(const device::Tensor& field, std::string_view batch, std::string_view name) {
    if (field.Empty()) {
        throw Error("Overlay {} need {}", batch, name);
    }
}

ImageLayout CheckImage(const device::Tensor& pixels, int64_t rows) {
    const Span<const int64_t> shape = pixels.Shape();
    const size_t rank = shape.size();
    if (pixels.Type() != device::DType::UInt8 || (rank != 3 && rank != 4) || shape[rank - 1] != 4 ||
        (rank == 4 && shape[0] != rows)) {
        throw Error("An overlay image is a [height, width, 4] uint8 tensor, or [N, height, width, "
                    "4] with one per row");
    }
    if (Stride(pixels, rank - 1) != 1 || Stride(pixels, rank - 2) != 4 ||
        Stride(pixels, rank - 3) % 4 != 0 || (rank == 4 && Stride(pixels, 0) % 4 != 0) ||
        pixels.Address() % 4 != 0) {
        throw UnsupportedError("An overlay image needs its pixels side by side, 4-byte aligned");
    }
    if (shape[rank - 2] > std::numeric_limits<uint32_t>::max() ||
        shape[rank - 3] > std::numeric_limits<uint32_t>::max()) {
        throw UnsupportedError("An overlay image of {} x {} pixels is too big", shape[rank - 2],
                               shape[rank - 3]);
    }
    return {.width = uint32_t(shape[rank - 2]),
            .height = uint32_t(shape[rank - 3]),
            .rowStride = uint32_t(Stride(pixels, rank - 3) / 4),
            .layerStride = rank == 4 ? uint32_t(Stride(pixels, 0) / 4) : 0,
            .layered = rank == 4};
}

void CheckPrims(const Batch& batch) {
    const std::initializer_list<device::DType> kWords = {device::DType::UInt32,
                                                         device::DType::Int32};
    const device::Tensor& prims = batch.prims;
    if (prims.Empty() || std::ranges::find(kWords, prims.Type()) == kWords.end() ||
        prims.Shape().size() != 1) {
        throw Error("Overlay prims are an [N] uint32 tensor of prim indices");
    }
    const Layout layout{0, prims.Shape()[0], 0};
    if (!batch.groups.Empty()) {
        if (batch.primShape != OverlayPrimShape::Bounds) {
            throw Error("Only overlay prim bounds gather prims into groups");
        }
        if (batch.groupCount == 0) {
            throw Error("Overlay prim groups need a group count");
        }
        CheckField(batch.groups, prims, layout, "prims", "groups", 0, kWords);
    }
    const Layout rows{0, batch.groups.Empty() ? layout.rows : int64_t(batch.groupCount), 0};
    CheckField(batch.colors, batch.groups.Empty() ? prims : batch.groups, rows, "prims", "colors",
               4, {device::DType::Float32, device::DType::UInt8});
    CheckField(batch.sizes, batch.groups.Empty() ? prims : batch.groups, rows, "prims", "widths", 0,
               {device::DType::Float32});
    if (!batch.pixels.Empty()) {
        if (batch.primShape != OverlayPrimShape::CameraImages) {
            throw Error("Only overlay camera images take pixels");
        }
        CheckImage(batch.pixels, layout.rows);
    }
}

void CheckFill(const Batch& batch) {
    const std::string_view name = KindName(batch.kind);
    const std::initializer_list<device::DType> kFloats = {device::DType::Float32};
    const std::initializer_list<device::DType> kWords = {device::DType::UInt32,
                                                         device::DType::Int32};
    Layout layout;
    if (batch.kind == OverlayKind::Triangles) {
        const Span<const int64_t> shape = batch.points.Shape();
        if (batch.points.Type() != device::DType::Float32 || shape.size() != 3 || shape[1] != 3 ||
            shape[2] != 3) {
            throw Error("Overlay triangles' points are an [N, 3, 3] float32 tensor");
        }
        if (Stride(batch.points, 2) != 1 || Stride(batch.points, 1) != 3) {
            throw UnsupportedError("Overlay triangles need each row's corners side by side");
        }
        layout = {0, shape[0], 0};
    } else {
        layout = CheckPoints(batch.points, name, false);
        if (layout.worlds > 0) {
            throw Error("Overlay images' points are an [N, 3] float32 tensor");
        }
        Require(batch.ends, name, "rights");
        Require(batch.rotations, name, "downs");
        CheckField(batch.ends, batch.points, layout, name, "rights", 3, kFloats);
        CheckField(batch.rotations, batch.points, layout, name, "downs", 3, kFloats);
        Require(batch.pixels, name, "pixels");
        CheckImage(batch.pixels, layout.rows);
    }
    CheckField(batch.colors, batch.points, layout, name, "colors", 4,
               {device::DType::Float32, device::DType::UInt8});
    CheckField(batch.worlds, batch.points, layout, name, "worlds", 0, kWords);
    if (!batch.itemCount.Empty() &&
        (std::ranges::find(kWords, batch.itemCount.Type()) == kWords.end() ||
         batch.itemCount.Shape().size() != 1 || batch.itemCount.Shape()[0] != 1)) {
        throw Error("Overlay {}' item counts are a [1] uint32 tensor", name);
    }
}

void CheckText(const Batch& batch) {
    const std::string_view name = KindName(batch.kind);
    const bool numbers = batch.kind == OverlayKind::Numbers;
    const device::Tensor& source = numbers ? batch.scalars : batch.chars;
    Require(source, name, numbers ? "values" : "chars");
    const Span<const int64_t> shape = source.Shape();
    if (numbers ? source.Type() != device::DType::Float32 || shape.size() != 1
                : source.Type() != device::DType::UInt8 || shape.size() != 2 || shape[1] < 1) {
        throw Error(numbers ? "Overlay numbers' values are an [N] float32 tensor"
                            : "Overlay texts' chars are an [N, characters] uint8 tensor");
    }
    if (!numbers && !source.Strides().empty() && source.Strides()[1] != 1) {
        throw UnsupportedError("Overlay texts' chars need their characters side by side");
    }
    const Layout layout{0, shape[0], 0};
    if (batch.points.Empty() && !batch.placement) {
        throw Error("Overlay {} need points, or a placement on the screen", name);
    }
    if (!batch.points.Empty()) {
        const Layout placed = CheckPoints(batch.points, name, false);
        if (placed.worlds > 0 || placed.rows != layout.rows) {
            throw Error("Overlay {}' points are an [N, 3] tensor, a row per text", name);
        }
    }
    const std::initializer_list<device::DType> kFloats = {device::DType::Float32};
    const std::initializer_list<device::DType> kWords = {device::DType::UInt32,
                                                         device::DType::Int32};
    CheckField(batch.colors, source, layout, name, "colors", 4,
               {device::DType::Float32, device::DType::UInt8});
    CheckField(batch.sizes, source, layout, name, "sizes", 0, kFloats);
    CheckField(batch.worlds, source, layout, name, "worlds", 0, kWords);
    if (!batch.itemCount.Empty() &&
        (std::ranges::find(kWords, batch.itemCount.Type()) == kWords.end() ||
         batch.itemCount.Shape().size() != 1 || batch.itemCount.Shape()[0] != 1)) {
        throw Error("Overlay {}' item counts are a [1] uint32 tensor", name);
    }
    if (batch.format.decimals > kOverlayMostDecimals) {
        throw Error("Overlay numbers with {} decimals; at most {} fit", batch.format.decimals,
                    kOverlayMostDecimals);
    }
}

void CheckBatch(const Batch& batch) {
    if (Textual(batch.kind)) {
        CheckText(batch);
        return;
    }
    if (batch.kind == OverlayKind::Prims) {
        CheckPrims(batch);
        return;
    }
    if (batch.kind == OverlayKind::Triangles || batch.kind == OverlayKind::Images) {
        CheckFill(batch);
        return;
    }
    const std::string_view name = KindName(batch.kind);
    const OverlayKind kind = batch.kind;
    const Layout layout = CheckPoints(batch.points, name, kind == OverlayKind::Trails);
    const std::initializer_list<device::DType> kFloats = {device::DType::Float32};
    const std::initializer_list<device::DType> kWords = {device::DType::UInt32,
                                                         device::DType::Int32};
    const std::string_view ends = kind == OverlayKind::Boxes     ? "half extents"
                                  : kind == OverlayKind::Circles ? "normals"
                                                                 : "ends";
    if (kind == OverlayKind::Lines || kind == OverlayKind::Arrows || kind == OverlayKind::Boxes ||
        kind == OverlayKind::Circles) {
        Require(batch.ends, name, ends);
    }
    CheckField(batch.ends, batch.points, layout, name, ends, 3, kFloats);
    CheckField(batch.rotations, batch.points, layout, name, "orientations", 4, kFloats);
    const std::string_view scalars = kind == OverlayKind::Frames   ? "lengths"
                                     : kind == OverlayKind::Arrows ? "head lengths"
                                                                   : "radii";
    if (kind == OverlayKind::Spheres || kind == OverlayKind::Circles ||
        kind == OverlayKind::Capsules) {
        Require(batch.scalars, name, scalars);
    }
    CheckField(batch.scalars, batch.points, layout, name, scalars, 0, kFloats);
    if (kind == OverlayKind::Capsules) {
        Require(batch.extras, name, "half heights");
        CheckField(batch.extras, batch.points, layout, name, "half heights", 0, kFloats);
    } else {
        CheckField(batch.extras, batch.points, layout, name, "heads", 0, kWords);
    }
    CheckField(batch.colors, batch.points, layout, name, "colors", 4,
               {device::DType::Float32, device::DType::UInt8});
    CheckField(batch.sizes, batch.points, layout, name,
               kind == OverlayKind::Markers ? "sizes" : "widths", 0, kFloats);
    if (!batch.worlds.Empty() && layout.worlds > 0) {
        throw Error("Overlay {} with a world per row can't also have worlds", name);
    }
    CheckField(batch.worlds, batch.points, layout, name, "worlds", 0, kWords);
    if (!batch.itemCount.Empty()) {
        const int64_t counts = std::max<int64_t>(layout.worlds, 1);
        if (std::ranges::find(kWords, batch.itemCount.Type()) == kWords.end() ||
            batch.itemCount.Shape().size() != 1 || batch.itemCount.Shape()[0] != counts) {
            throw Error("Overlay {}' item counts are a [{}] uint32 tensor", name, counts);
        }
        if (&batch.itemCount.GetDevice() != &batch.points.GetDevice()) {
            throw Error("Overlay {}' item counts are on another device than their points", name);
        }
    }
}

uint32_t Words(int64_t value, std::string_view what) {
    if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
        throw UnsupportedError("An overlay's {} are {} words apart; at most {} fit", what, value,
                               std::numeric_limits<uint32_t>::max());
    }
    return uint32_t(value);
}

void RequireDevice(const device::Tensor& tensor, device::Device& device, std::string_view what) {
    if (!tensor.Empty() && &tensor.GetDevice() != &device) {
        throw Error("Overlay {} are on {}, not the renderer's {}; Device::Copy or Device::Import "
                    "moves them",
                    what, tensor.GetDevice().Adapter().alias, device.Adapter().alias);
    }
}

}

Overlay::Overlay() : m_data(std::make_shared<OverlayData>()) {}

Overlay::~Overlay() = default;
Overlay::Overlay(const Overlay& other) = default;
Overlay::Overlay(Overlay&& other) noexcept = default;
Overlay& Overlay::operator=(const Overlay& other) = default;
Overlay& Overlay::operator=(Overlay&& other) noexcept = default;

OverlayData& Overlay::Edit() {
    if (!m_data) {
        m_data = std::make_shared<OverlayData>();
    } else if (m_data.use_count() > 1) {
        m_data = std::make_shared<OverlayData>(*m_data);
    }
    return *m_data;
}

Overlay& Overlay::Line(Float3 a, Float3 b, const OverlayStyle& style) {
    AppendRow(HostBatch(Edit(), OverlayKind::Lines, style), {a, b}, PackColor(style.color),
              style.width, style.world, style.id);
    return *this;
}

Overlay& Overlay::Lines(Span<const Float3> endpoints, const OverlayStyle& style,
                        Span<const Float4> colors) {
    if (endpoints.size() % 2 != 0) {
        throw Error("Overlay lines of {} endpoints; they come in pairs", endpoints.size());
    }
    if (!colors.empty() && colors.size() != endpoints.size() / 2) {
        throw Error("{} colors for {} overlay lines", colors.size(), endpoints.size() / 2);
    }
    if (endpoints.empty()) {
        return *this;
    }
    Batch& batch = HostBatch(Edit(), OverlayKind::Lines, style);
    for (size_t line = 0; line < endpoints.size() / 2; ++line) {
        AppendRow(batch, {endpoints[2 * line], endpoints[2 * line + 1]},
                  PackColor(colors.empty() ? style.color : colors[line]), style.width, style.world,
                  style.id);
    }
    return *this;
}

Overlay& Overlay::Polyline(Span<const Float3> points, bool closed, const OverlayStyle& style) {
    if (points.size() < 2) {
        return *this;
    }
    Batch& batch = HostBatch(Edit(), OverlayKind::Lines, style);
    const uint32_t color = PackColor(style.color);
    for (size_t at = 0; at + 1 < points.size(); ++at) {
        AppendRow(batch, {points[at], points[at + 1]}, color, style.width, style.world, style.id);
    }
    if (closed && points.size() > 2) {
        AppendRow(batch, {points[points.size() - 1], points[0]}, color, style.width, style.world,
                  style.id);
    }
    return *this;
}

Overlay& Overlay::Marker(Float3 point, const OverlayStyle& style) {
    return Markers(Span<const Float3>(&point, 1), style);
}

Overlay& Overlay::Markers(Span<const Float3> points, const OverlayStyle& style,
                          Span<const Float4> colors) {
    if (!colors.empty() && colors.size() != points.size()) {
        throw Error("{} colors for {} overlay markers", colors.size(), points.size());
    }
    if (points.empty()) {
        return *this;
    }
    Batch& batch = HostBatch(Edit(), OverlayKind::Markers, style);
    for (size_t at = 0; at < points.size(); ++at) {
        AppendRow(batch, {points[at]}, PackColor(colors.empty() ? style.color : colors[at]),
                  style.size, style.world, style.id);
    }
    return *this;
}

Overlay& Overlay::Box(Float3 center, Float3 halfExtents, const OverlayStyle& style,
                      Quatd orientation) {
    AppendRow(HostBatch(Edit(), OverlayKind::Boxes, style),
              {.point = center, .end = halfExtents, .rotation = Rotation(orientation)},
              PackColor(style.color), style.width, style.world, style.id);
    return *this;
}

Overlay& Overlay::Sphere(Float3 center, float radius, const OverlayStyle& style) {
    AppendRow(HostBatch(Edit(), OverlayKind::Spheres, style), {.point = center, .scalar = radius},
              PackColor(style.color), style.width, style.world, style.id);
    return *this;
}

Overlay& Overlay::Circle(Float3 center, Float3 normal, float radius, const OverlayStyle& style) {
    AppendRow(HostBatch(Edit(), OverlayKind::Circles, style),
              {.point = center, .end = normal, .scalar = radius}, PackColor(style.color),
              style.width, style.world, style.id);
    return *this;
}

Overlay& Overlay::Frame(Float3 origin, Quatd orientation, float length, const OverlayStyle& style) {
    AppendRow(HostBatch(Edit(), OverlayKind::Frames, style),
              {.point = origin, .rotation = Rotation(orientation), .scalar = length},
              PackColor(style.color), style.width, style.world, style.id);
    return *this;
}

Overlay& Overlay::Arrow(Float3 from, Float3 to, const OverlayStyle& style, float headLength) {
    AppendRow(HostBatch(Edit(), OverlayKind::Arrows, style),
              {.point = from, .end = to, .scalar = headLength}, PackColor(style.color), style.width,
              style.world, style.id);
    return *this;
}

Overlay& Overlay::Capsule(Float3 center, Quatd orientation, float radius, float halfHeight,
                          const OverlayStyle& style) {
    AppendRow(HostBatch(Edit(), OverlayKind::Capsules, style),
              {.point = center,
               .rotation = Rotation(orientation),
               .scalar = radius,
               .extra = std::bit_cast<uint32_t>(halfHeight)},
              PackColor(style.color), style.width, style.world, style.id);
    return *this;
}

Float2 ScreenPlacement::Alignment() const {
    const auto side = [](float value) { return value < 0.5f ? 0.0f : value > 0.5f ? 1.0f : 0.5f; };
    return align ? *align : Float2{side(at.x), side(at.y)};
}

Overlay& Overlay::Label(Float3 point, std::string_view text, const OverlayStyle& style) {
    AppendText(HostBatch(Edit(), OverlayKind::Labels, style), text, {.point = point}, style);
    return *this;
}

Overlay& Overlay::Text(const ScreenPlacement& placement, std::string_view text,
                       const OverlayStyle& style) {
    const Float2 align = placement.Alignment();
    AppendText(HostBatch(Edit(), OverlayKind::Texts, style), text,
               {.point = {placement.at.x, placement.at.y, 0},
                .end = {placement.offset.x, placement.offset.y, 0},
                .rotation = {align.x, align.y, 0, 1}},
               style);
    return *this;
}

Overlay& Overlay::Triangle(Float3 a, Float3 b, Float3 c, const OverlayStyle& style) {
    AppendRow(HostBatch(Edit(), OverlayKind::Triangles, style),
              {.point = a, .end = b, .rotation = {c.x, c.y, c.z, 1}, .extra = 7},
              PackColor(style.color), style.width, style.world, style.id);
    return *this;
}

Overlay& Overlay::Polygon(Span<const Float3> points, const OverlayStyle& style) {
    if (points.size() < 3) {
        return *this;
    }
    Batch& batch = HostBatch(Edit(), OverlayKind::Triangles, style);
    const uint32_t color = PackColor(style.color);
    const size_t last = points.size() - 2;
    for (size_t at = 1; at <= last; ++at) {
        const Float3 c = points[at + 1];
        const uint32_t edges = 2u | (at == 1 ? 1u : 0u) | (at == last ? 4u : 0u);
        AppendRow(
            batch,
            {.point = points[0], .end = points[at], .rotation = {c.x, c.y, c.z, 1}, .extra = edges},
            color, style.width, style.world, style.id);
    }
    return *this;
}

Overlay& Overlay::Curve(Span<const Float3> controls, const OverlayStyle& style,
                        uint32_t tessellation) {
    if (controls.size() < 4 || (controls.size() - 1) % 3 != 0) {
        throw Error("An overlay curve has 3N + 1 control points, not {}", controls.size());
    }
    if (tessellation == 0) {
        throw Error("An overlay curve needs at least one segment per span");
    }
    std::vector<Float3> points = {controls[0]};
    for (size_t span = 0; span + 3 < controls.size(); span += 3) {
        const Float3 a = controls[span];
        const Float3 b = controls[span + 1];
        const Float3 c = controls[span + 2];
        const Float3 d = controls[span + 3];
        for (uint32_t step = 1; step <= tessellation; ++step) {
            const float t = float(step) / float(tessellation);
            const float u = 1 - t;
            const float wa = u * u * u;
            const float wb = 3 * u * u * t;
            const float wc = 3 * u * t * t;
            const float wd = t * t * t;
            points.push_back({wa * a.x + wb * b.x + wc * c.x + wd * d.x,
                              wa * a.y + wb * b.y + wc * c.y + wd * d.y,
                              wa * a.z + wb * b.z + wc * c.z + wd * d.z});
        }
    }
    return Polyline(points, false, style);
}

Overlay& Overlay::Arc(Float3 center, Float3 normal, float radius, float begin, float end,
                      const OverlayStyle& style, bool filled) {
    const float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (!(length > 0) || !(radius > 0)) {
        throw Error("An overlay arc needs a normal and a positive radius");
    }
    const Float3 n = {normal.x / length, normal.y / length, normal.z / length};
    const Float3 other = std::abs(n.z) < 0.9f ? Float3{0, 0, 1} : Float3{1, 0, 0};
    Float3 u = {other.y * n.z - other.z * n.y, other.z * n.x - other.x * n.z,
                other.x * n.y - other.y * n.x};
    const float ul = std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    u = {u.x / ul, u.y / ul, u.z / ul};
    const Float3 v = {n.y * u.z - n.z * u.y, n.z * u.x - n.x * u.z, n.x * u.y - n.y * u.x};
    constexpr float kRadians = 3.14159265358979f / 180;
    const float sweep = (end - begin) * kRadians;
    const auto steps =
        uint32_t(std::clamp(std::ceil(std::abs(sweep) / (2 * 3.14159265f) * 64), 1.0f, 1024.0f));
    std::vector<Float3> points;
    if (filled) {
        points.push_back(center);
    }
    for (uint32_t step = 0; step <= steps; ++step) {
        const float angle = begin * kRadians + sweep * float(step) / float(steps);
        const float c = std::cos(angle) * radius;
        const float s = std::sin(angle) * radius;
        points.push_back({center.x + u.x * c + v.x * s, center.y + u.y * c + v.y * s,
                          center.z + u.z * c + v.z * s});
    }
    return filled ? Polygon(points, style) : Polyline(points, false, style);
}

Overlay& Overlay::Rectangle(Float3 center, Float3 halfRight, Float3 halfUp,
                            const OverlayStyle& style, bool filled) {
    const auto corner = [&](float across, float up) {
        return Float3{center.x + halfRight.x * across + halfUp.x * up,
                      center.y + halfRight.y * across + halfUp.y * up,
                      center.z + halfRight.z * across + halfUp.z * up};
    };
    const std::vector<Float3> corners = {corner(-1, -1), corner(1, -1), corner(1, 1),
                                         corner(-1, 1)};
    return filled ? Polygon(corners, style) : Polyline(corners, true, style);
}

Overlay& Overlay::Mesh(Span<const Float3> points, Span<const uint32_t> faceCounts,
                       Span<const uint32_t> indices, const OverlayStyle& style, bool wireframe) {
    size_t used = 0;
    for (const uint32_t count : faceCounts) {
        if (count < 3) {
            throw Error("An overlay mesh face has at least 3 corners, not {}", count);
        }
        used += count;
    }
    if (used != indices.size()) {
        throw Error("An overlay mesh's faces use {} indices, but it has {}", used, indices.size());
    }
    for (const uint32_t index : indices) {
        if (index >= points.size()) {
            throw Error("An overlay mesh index {} is past its {} points", index, points.size());
        }
    }
    size_t at = 0;
    std::vector<Float3> face;
    for (const uint32_t count : faceCounts) {
        face.clear();
        for (uint32_t corner = 0; corner < count; ++corner) {
            face.push_back(points[indices[at + corner]]);
        }
        at += count;
        if (wireframe) {
            Polyline(face, true, style);
        } else {
            Polygon(face, style);
        }
    }
    return *this;
}

Overlay& Overlay::Image(Float3 topLeft, Float3 right, Float3 down, device::Tensor pixels,
                        const OverlayStyle& style) {
    if (pixels.Empty()) {
        throw Error("An overlay image needs pixels");
    }
    CheckImage(pixels, 1);
    if (pixels.Shape().size() != 3) {
        throw Error("An overlay image is a [height, width, 4] uint8 tensor");
    }
    Batch batch{.kind = OverlayKind::Images, .style = style, .pixels = std::move(pixels)};
    AppendRow(batch, {.point = topLeft, .end = right, .rotation = {down.x, down.y, down.z, 1}},
              PackColor(style.color), style.width, style.world, style.id);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Prims(Span<const uint32_t> prims, OverlayPrimShape shape,
                        const OverlayStyle& style, float length, Span<const uint32_t> groups) {
    if (!groups.empty() && groups.size() != prims.size()) {
        throw Error("{} overlay prim groups for {} prims", groups.size(), prims.size());
    }
    if (!groups.empty() && shape != OverlayPrimShape::Bounds) {
        throw Error("Only overlay prim bounds gather prims into groups");
    }
    if (prims.empty()) {
        return *this;
    }
    Batch batch{.kind = OverlayKind::Prims, .style = style, .primShape = shape, .length = length};
    batch.hostPrims.assign(prims.begin(), prims.end());
    batch.hostGroups.assign(groups.begin(), groups.end());
    batch.groupCount =
        groups.empty() ? 0 : *std::ranges::max_element(groups.begin(), groups.end()) + 1;
    const size_t rows = groups.empty() ? prims.size() : batch.groupCount;
    batch.hostColors.assign(rows, PackColor(style.color));
    batch.hostSizes.assign(rows, style.width);
    batch.hostIds.assign(rows, style.id);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Outline(Span<const uint32_t> prims, const OverlayStyle& style) {
    if (prims.empty()) {
        return *this;
    }
    Batch batch{.kind = OverlayKind::Outlines, .style = style};
    batch.hostPrims.assign(prims.begin(), prims.end());
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayTriangles triangles) {
    Batch batch{.kind = OverlayKind::Triangles,
                .style = triangles.style,
                .points = std::move(triangles.points),
                .colors = std::move(triangles.colors),
                .worlds = std::move(triangles.worlds),
                .itemCount = std::move(triangles.itemCount),
                .owner = std::move(triangles.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayImages images) {
    Batch batch{.kind = OverlayKind::Images,
                .style = images.style,
                .points = std::move(images.points),
                .ends = std::move(images.rights),
                .colors = std::move(images.colors),
                .worlds = std::move(images.worlds),
                .itemCount = std::move(images.itemCount),
                .rotations = std::move(images.downs),
                .pixels = std::move(images.pixels),
                .owner = std::move(images.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayPrims prims) {
    Batch batch{.kind = OverlayKind::Prims,
                .style = prims.style,
                .colors = std::move(prims.colors),
                .sizes = std::move(prims.widths),
                .prims = std::move(prims.prims),
                .groups = std::move(prims.groups),
                .pixels = std::move(prims.pixels),
                .groupCount = prims.groupCount,
                .primShape = prims.shape,
                .length = prims.length,
                .owner = std::move(prims.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayTexts texts) {
    Batch batch{.kind = OverlayKind::Labels,
                .style = texts.style,
                .points = std::move(texts.points),
                .colors = std::move(texts.colors),
                .sizes = std::move(texts.sizes),
                .worlds = std::move(texts.worlds),
                .itemCount = std::move(texts.itemCount),
                .chars = std::move(texts.chars),
                .placement = texts.placement,
                .owner = std::move(texts.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayNumbers numbers) {
    Batch batch{.kind = OverlayKind::Numbers,
                .style = numbers.style,
                .points = std::move(numbers.points),
                .colors = std::move(numbers.colors),
                .sizes = std::move(numbers.sizes),
                .worlds = std::move(numbers.worlds),
                .itemCount = std::move(numbers.itemCount),
                .scalars = std::move(numbers.values),
                .placement = numbers.placement,
                .format = std::move(numbers.format),
                .owner = std::move(numbers.owner)};
    CheckBatch(batch);
    uint32_t prefix[4] = {};
    uint32_t suffix[4] = {};
    uint32_t length = 0;
    PackAffix(batch.format.prefix, prefix, length, "prefix");
    PackAffix(batch.format.suffix, suffix, length, "suffix");
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayLines lines) {
    Batch batch{.kind = OverlayKind::Lines,
                .style = lines.style,
                .points = std::move(lines.points),
                .ends = std::move(lines.ends),
                .colors = std::move(lines.colors),
                .sizes = std::move(lines.widths),
                .worlds = std::move(lines.worlds),
                .itemCount = std::move(lines.itemCount),
                .owner = std::move(lines.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayMarkers markers) {
    Batch batch{.kind = OverlayKind::Markers,
                .style = markers.style,
                .points = std::move(markers.points),
                .colors = std::move(markers.colors),
                .sizes = std::move(markers.sizes),
                .worlds = std::move(markers.worlds),
                .itemCount = std::move(markers.itemCount),
                .owner = std::move(markers.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayBoxes boxes) {
    Batch batch{.kind = OverlayKind::Boxes,
                .style = boxes.style,
                .points = std::move(boxes.points),
                .ends = std::move(boxes.halfExtents),
                .colors = std::move(boxes.colors),
                .sizes = std::move(boxes.widths),
                .worlds = std::move(boxes.worlds),
                .itemCount = std::move(boxes.itemCount),
                .rotations = std::move(boxes.orientations),
                .owner = std::move(boxes.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlaySpheres spheres) {
    Batch batch{.kind = OverlayKind::Spheres,
                .style = spheres.style,
                .points = std::move(spheres.points),
                .colors = std::move(spheres.colors),
                .sizes = std::move(spheres.widths),
                .worlds = std::move(spheres.worlds),
                .itemCount = std::move(spheres.itemCount),
                .scalars = std::move(spheres.radii),
                .owner = std::move(spheres.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayCircles circles) {
    Batch batch{.kind = OverlayKind::Circles,
                .style = circles.style,
                .points = std::move(circles.points),
                .ends = std::move(circles.normals),
                .colors = std::move(circles.colors),
                .sizes = std::move(circles.widths),
                .worlds = std::move(circles.worlds),
                .itemCount = std::move(circles.itemCount),
                .scalars = std::move(circles.radii),
                .owner = std::move(circles.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayFrames frames) {
    Batch batch{.kind = OverlayKind::Frames,
                .style = frames.style,
                .points = std::move(frames.points),
                .sizes = std::move(frames.widths),
                .worlds = std::move(frames.worlds),
                .itemCount = std::move(frames.itemCount),
                .rotations = std::move(frames.orientations),
                .scalars = std::move(frames.lengths),
                .length = frames.length,
                .owner = std::move(frames.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayArrows arrows) {
    Batch batch{.kind = OverlayKind::Arrows,
                .style = arrows.style,
                .points = std::move(arrows.points),
                .ends = std::move(arrows.ends),
                .colors = std::move(arrows.colors),
                .sizes = std::move(arrows.widths),
                .worlds = std::move(arrows.worlds),
                .itemCount = std::move(arrows.itemCount),
                .scalars = std::move(arrows.headLengths),
                .length = arrows.headLength,
                .owner = std::move(arrows.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayCapsules capsules) {
    Batch batch{.kind = OverlayKind::Capsules,
                .style = capsules.style,
                .points = std::move(capsules.points),
                .colors = std::move(capsules.colors),
                .sizes = std::move(capsules.widths),
                .worlds = std::move(capsules.worlds),
                .itemCount = std::move(capsules.itemCount),
                .rotations = std::move(capsules.orientations),
                .scalars = std::move(capsules.radii),
                .extras = std::move(capsules.halfHeights),
                .owner = std::move(capsules.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

Overlay& Overlay::Add(OverlayTrails trails) {
    Batch batch{.kind = OverlayKind::Trails,
                .style = trails.style,
                .points = std::move(trails.points),
                .colors = std::move(trails.colors),
                .sizes = std::move(trails.widths),
                .worlds = std::move(trails.worlds),
                .itemCount = std::move(trails.itemCount),
                .extras = std::move(trails.heads),
                .fade = trails.fade,
                .owner = std::move(trails.owner)};
    CheckBatch(batch);
    Edit().batches.push_back(std::move(batch));
    return *this;
}

bool Overlay::Empty() const {
    return !m_data || m_data->batches.empty();
}

OverlayPick::OverlayPick(std::shared_ptr<OverlayPickState> state) : m_state(std::move(state)) {}

bool OverlayPick::Ready() const {
    if (!m_state) {
        return true;
    }
    const std::lock_guard lock(m_state->mutex);
    return m_state->recorded && m_state->done.Ready();
}

void OverlayPick::Wait() const {
    if (!m_state) {
        return;
    }
    device::Operation done;
    {
        std::unique_lock lock(m_state->mutex);
        m_state->served.wait(lock, [&] { return m_state->recorded; });
        done = m_state->done;
    }
    done.Wait();
}

namespace {

std::optional<OverlayPickResult> PickResult(const std::shared_ptr<OverlayPickState>& state) {
    if (!state || !state->readback) {
        return std::nullopt;
    }
    OverlayPickResult result;
    std::memcpy(&result, state->readback->Mapped(), sizeof(result));
    return result;
}

}

std::optional<OverlayHit> OverlayPick::Hit() const {
    Wait();
    const std::optional<OverlayPickResult> result = PickResult(m_state);
    if (!result || result->hit == 0 || result->overlay >= m_state->names.size() ||
        result->batch >= m_state->sources.size()) {
        return std::nullopt;
    }
    return OverlayHit{.overlay = m_state->names[result->overlay],
                      .id = result->id,
                      .batch = m_state->sources[result->batch].batch,
                      .item = result->row,
                      .piece = result->piece,
                      .world = result->world,
                      .prim = result->prim,
                      .distance = result->distance,
                      .depth = result->depth,
                      .position = {result->position[0], result->position[1], result->position[2]}};
}

std::optional<OverlayRay> OverlayPick::Ray() const {
    Wait();
    const std::optional<OverlayPickResult> result = PickResult(m_state);
    if (!result) {
        return std::nullopt;
    }
    return OverlayRay{{result->origin[0], result->origin[1], result->origin[2]},
                      {result->direction[0], result->direction[1], result->direction[2]}};
}

bool OverlayData::Outlined() const {
    return std::ranges::any_of(batches, [](const Batch& batch) {
        return batch.kind == OverlayKind::Outlines && !batch.hostPrims.empty();
    });
}

OverlayPass::OverlayPass(device::Device& device)
    : m_device(device),
      m_program(device.CreateProgram(
          {.modules = {"ernie.render.kernels.overlay.overlay"},
           .library = &kRenderKernels,
           .entries = {"ProjectOverlays", "ScatterOverlays", "FindOverlayTiles", "FoldOverlayTiles",
                       "CompositeOverlays", "KeyOverlayRows", "StartOverlayGroups",
                       "MeasureOverlayGroups", "ClearOverlayBoxes", "UniteOverlayBoxes",
                       "OutlineOverlays", "ClearOverlayPicks", "MeasureOverlayPicks",
                       "ChooseOverlayPicks", "ResolveOverlayPicks"},
           .label = "overlays"})),
      m_unused(device.CreateBuffer({.label = "overlay unused"}, std::vector<uint32_t>(16, ~0u))),
      m_pairCount(device.CreateBuffer({.bytes = 4, .label = "overlay pairs"})),
      m_pairsNeeded(device.CreateBuffer(
          {.bytes = 4, .memory = device::Memory::Readback, .label = "overlay pairs needed"})) {
    *static_cast<uint32_t*>(m_pairsNeeded->Mapped()) = 0;
}

OverlayContents OverlayPass::Share(const Overlay& overlay) {
    return overlay.Empty() ? nullptr : overlay.m_data;
}

void OverlayPass::CheckDevice(const OverlayData& overlay, device::Device& device) {
    for (const Batch& batch : overlay.batches) {
        const std::string kind(KindName(batch.kind));
        for (const device::Tensor* tensor :
             {&batch.points, &batch.ends, &batch.colors, &batch.sizes, &batch.worlds,
              &batch.itemCount, &batch.rotations, &batch.scalars, &batch.extras, &batch.chars,
              &batch.prims, &batch.groups, &batch.pixels}) {
            RequireDevice(*tensor, device, kind + "' arrays");
        }
    }
}

void OverlayPass::Reserve(Ref<device::Buffer>& buffer, uint64_t bytes, const char* label) {
    if (!buffer || buffer->Bytes() < bytes) {
        buffer = m_device.CreateBuffer({.bytes = std::max<uint64_t>(bytes, 256), .label = label});
    }
}

void OverlayPass::Group(OverlayBatch& gpu, GroupKey key, uint32_t rows, uint32_t worldCount,
                        uint64_t worlds, uint32_t worldStride, uint64_t itemCounts) {
    const size_t at = m_groupParams.size();
    if (m_groupings.size() <= at) {
        m_groupings.emplace_back();
    }
    Grouping& grouping = m_groupings[at];
    Reserve(grouping.keys, uint64_t(rows) * 4, "overlay group keys");
    Reserve(grouping.order, uint64_t(rows) * 4, "overlay group order");
    Reserve(grouping.starts, (uint64_t(worldCount) + 3) * 4, "overlay group starts");
    m_groupParams.push_back({.worlds = {worlds},
                             .itemCounts = {itemCounts},
                             .keys = {grouping.keys->Address()},
                             .order = {grouping.order->Address()},
                             .starts = {grouping.starts->Address()},
                             .rows = rows,
                             .worldStride = worldStride,
                             .groups = worldCount,
                             .need = uint32_t(at)});
    m_groupKeys.push_back(key);
    const auto learned = m_groupStrides.find(key);
    const uint32_t guess = worldCount > 0 ? rows / worldCount * 2 + 64 : rows;
    const uint32_t need = learned != m_groupStrides.end() ? learned->second : guess;
    gpu.flags |= kOverlayGrouped;
    gpu.order = {grouping.order->Address()};
    gpu.groupStarts = {grouping.starts->Address()};
    gpu.groups = worldCount;
    gpu.capacity = std::min(rows, need + need / 4 + 16) * gpu.pieces;
}

void OverlayPass::Unite(OverlayBatch& gpu, uint32_t rows, uint32_t groups, uint64_t prims,
                        uint32_t primStride, uint64_t groupIds, uint32_t groupStride) {
    const size_t at = m_uniteParams.size();
    if (m_boxes.size() <= at) {
        m_boxes.emplace_back();
    }
    Reserve(m_boxes[at], uint64_t(groups) * kOverlayBoxWords * 4, "overlay boxes");
    m_uniteParams.push_back({.prims = {prims},
                             .groups = {groupIds},
                             .boxes = {m_boxes[at]->Address()},
                             .primCount = rows,
                             .primStride = primStride,
                             .groupStride = groupStride,
                             .boxCount = groups});
    gpu.boxes = {m_boxes[at]->Address()};
    gpu.flags |= kOverlayUnited;
}

void OverlayPass::AddBatches(const OverlayContents& overlay, uint32_t index, uint32_t worldCount,
                             std::vector<OverlayBatch>& batches, uint32_t& firstItem) {
    constexpr size_t kHostBuffers = 12;
    Uploaded& uploaded = m_uploads[overlay.get()];
    if (uploaded.contents.lock() != overlay) {
        uploaded = {overlay, {}};
        for (const Batch& batch : overlay->batches) {
            if (!Hosted(batch)) {
                continue;
            }
            const auto upload = [&](const auto& values, const char* label) {
                uploaded.buffers.push_back(
                    values.empty() ? m_unused : m_device.CreateBuffer({.label = label}, values));
            };
            upload(batch.hostPoints, "overlay points");
            upload(batch.hostEnds, "overlay ends");
            upload(batch.hostColors, "overlay colors");
            upload(batch.hostSizes, "overlay sizes");
            upload(batch.hostWorlds, "overlay worlds");
            upload(batch.hostRotations, "overlay rotations");
            upload(batch.hostScalars, "overlay scalars");
            upload(batch.hostExtras, "overlay extras");
            upload(batch.hostCodes, "overlay text");
            if (batch.kind == OverlayKind::Outlines) {
                std::vector<uint32_t> bits(*std::ranges::max_element(batch.hostPrims) / 32 + 1);
                for (const uint32_t prim : batch.hostPrims) {
                    bits[prim / 32] |= 1u << (prim % 32);
                }
                upload(bits, "overlay outline");
            } else {
                upload(batch.hostPrims, "overlay prims");
            }
            upload(batch.hostGroups, "overlay groups");
            upload(batch.hostIds, "overlay ids");
        }
    }

    const uint64_t unused = m_unused->Address();
    size_t hostBatch = 0;
    for (uint32_t batchIndex = 0; batchIndex < overlay->batches.size(); ++batchIndex) {
        const Batch& batch = overlay->batches[batchIndex];
        const OverlayStyle& style = batch.style;
        const Ref<device::Buffer>* buffers =
            Hosted(batch) ? uploaded.buffers.data() + kHostBuffers * hostBatch++ : nullptr;
        if (batch.kind == OverlayKind::Outlines) {
            m_outlines.push_back({.bits = {buffers[9]->Address()},
                                  .bitWords = uint32_t(buffers[9]->Bytes() / 4),
                                  .color = PackColor(style.color),
                                  .width = std::max(style.width, 0.5f),
                                  .overlay = index});
            continue;
        }
        OverlayBatch gpu{};
        gpu.ends = {unused};
        gpu.colors = {unused};
        gpu.sizes = {unused};
        gpu.worlds = {unused};
        gpu.itemCounts = {unused};
        gpu.rotations = {unused};
        gpu.scalars = {unused};
        gpu.extras = {unused};
        gpu.prims = {unused};
        gpu.primGroups = {unused};
        gpu.boxes = {unused};
        gpu.pieces = 1;
        gpu.length = batch.length;
        gpu.fade = batch.fade;
        gpu.chars = unused;
        gpu.background = PackColor(style.background);
        gpu.halo = PackColor(style.halo);
        gpu.kind = uint32_t(batch.kind);
        gpu.depth = uint32_t(style.depth);
        gpu.marker = uint32_t(style.marker);
        gpu.dash = PackPattern(style.dash);
        gpu.gap = PackPattern(style.gap > 0 ? style.gap : style.dash);
        gpu.flags = (style.units == OverlayUnits::Meters ? kOverlayMeters : 0) |
                    (style.screenSized ? kOverlayScreenSized : 0) |
                    (style.facing ? kOverlayFacing : 0) | (style.pickable ? kOverlayPickable : 0);
        gpu.id = style.id;
        gpu.ids = {unused};
        gpu.color = PackColor(style.color);
        gpu.size =
            batch.kind == OverlayKind::Markers || Textual(batch.kind) ? style.size : style.width;
        gpu.world = style.world;
        gpu.anchor = batch.kind == OverlayKind::Prims ? kNoPrim : style.anchor;
        gpu.overlay = index;
        gpu.alignX = style.align.x;
        gpu.alignY = style.align.y;
        gpu.justify = style.align.x;
        gpu.primShape = uint32_t(batch.primShape);
        if (Textual(batch.kind)) {
            gpu.length = style.width;
        }
        const bool haloed = Textual(batch.kind) && style.halo.w > 0;
        const auto textPieces = [&](uint32_t chars) { return 1 + (haloed ? 2 : 1) * chars; };
        if (!batch.pixels.Empty()) {
            RequireDevice(batch.pixels, m_device, std::string(KindName(batch.kind)) + "' pixels");
            const ImageLayout image = CheckImage(batch.pixels, batch.pixels.Shape()[0]);
            gpu.image = batch.pixels.Address();
            gpu.imageWidth = image.width;
            gpu.imageHeight = image.height;
            gpu.imageRowStride = image.rowStride;
            gpu.imageLayerStride = image.layerStride;
            gpu.flags |= image.layered ? kOverlayImageLayers : 0;
        }
        if (Hosted(batch)) {
            gpu.points = {buffers[0]->Address()};
            gpu.pointStride = 3;
            gpu.ends = {buffers[1]->Address()};
            gpu.endStride = 3;
            gpu.colors = {buffers[2]->Address()};
            gpu.colorStride = 1;
            gpu.sizes = {buffers[3]->Address()};
            gpu.sizeStride = 1;
            gpu.rotations = {buffers[5]->Address()};
            gpu.rotationStride = 4;
            gpu.scalars = {buffers[6]->Address()};
            gpu.scalarStride = 1;
            gpu.extras = {buffers[7]->Address()};
            gpu.extraStride = 1;
            if (!batch.hostIds.empty()) {
                gpu.ids = {buffers[11]->Address()};
                gpu.idStride = 1;
                gpu.flags |= kOverlayHasIds;
            }
            gpu.flags |= kOverlayByteColors | kOverlayHasColors | kOverlayHasSizes |
                         kOverlayHasRotations | kOverlayHasScalars | kOverlayHasExtras;
            gpu.pieces = PiecesPerRow(batch.kind, 0);
            if (Textual(batch.kind)) {
                gpu.chars = buffers[8]->Address();
                gpu.flags |= kOverlayPackedText;
                gpu.maxChars = batch.maxChars;
                gpu.pieces = textPieces(batch.maxChars);
            }
            auto rows = uint32_t(batch.hostPoints.size());
            if (batch.kind == OverlayKind::Prims) {
                gpu.prims = {buffers[9]->Address()};
                gpu.pointStride = 1;
                gpu.pieces = PrimPieces(batch.primShape);
                gpu.primCount = uint32_t(batch.hostPrims.size());
                rows = gpu.primCount;
                if (!batch.hostGroups.empty()) {
                    rows = batch.groupCount;
                    Unite(gpu, gpu.primCount, rows, buffers[9]->Address(), 1,
                          buffers[10]->Address(), 1);
                }
            }
            gpu.capacity = rows * gpu.pieces;
            if (rows > 0 && std::ranges::any_of(batch.hostWorlds, [](uint32_t world) {
                    return world != kGlobalWorld;
                })) {
                Group(gpu, {index, batchIndex}, rows, worldCount, buffers[4]->Address(), 1, unused);
            }
        } else {
            const std::string kind(KindName(batch.kind));
            const auto field = [&](const device::Tensor& tensor, std::string_view name) {
                RequireDevice(tensor, m_device, kind + "' " + std::string(name));
            };
            const bool trail = batch.kind == OverlayKind::Trails;
            const bool textual = Textual(batch.kind);
            const bool primal = batch.kind == OverlayKind::Prims;
            const bool triangles = batch.kind == OverlayKind::Triangles;
            const size_t rank = batch.points.Empty() ? 2 : batch.points.Shape().size();
            const bool perWorld = !trail && !textual && !primal && !triangles && rank == 3;
            const size_t rowDim = perWorld ? 1 : 0;
            const auto rowStride = [&](const device::Tensor& tensor, int64_t words,
                                       std::string_view name) {
                return Words(Stride(tensor, rowDim) / words, name);
            };
            const auto worldStride = [&](const device::Tensor& tensor, int64_t words,
                                         std::string_view name) {
                return perWorld ? Words(Stride(tensor, 0) / words, name) : 0u;
            };
            const device::Tensor& source = primal                               ? batch.prims
                                           : !textual                           ? batch.points
                                           : batch.kind == OverlayKind::Numbers ? batch.scalars
                                                                                : batch.chars;
            uint32_t rows =
                uint32_t(source.Shape()[trail || textual || primal || triangles ? 0 : rank - 2]);
            gpu.worldCount = perWorld ? uint32_t(batch.points.Shape()[0]) : 0;
            gpu.points = {unused};
            if (!batch.points.Empty()) {
                field(batch.points, "points");
                gpu.points = {batch.points.Address()};
                gpu.pointStride = rowStride(batch.points, 1, "points");
                gpu.pointWorldStride = worldStride(batch.points, 1, "points");
            }
            if (triangles) {
                gpu.ends = {batch.points.Address() + 12};
                gpu.endStride = gpu.pointStride;
                gpu.rotations = {batch.points.Address() + 24};
                gpu.rotationStride = gpu.pointStride;
            }
            if (primal) {
                field(batch.prims, "prims");
                gpu.prims = {batch.prims.Address()};
                gpu.pointStride = Words(Stride(batch.prims, 0), "prims");
                gpu.pieces = PrimPieces(batch.primShape);
                gpu.primCount = rows;
                if (!batch.groups.Empty()) {
                    field(batch.groups, "groups");
                    Unite(gpu, rows, batch.groupCount, batch.prims.Address(), gpu.pointStride,
                          batch.groups.Address(), Words(Stride(batch.groups, 0), "groups"));
                    rows = batch.groupCount;
                }
            }
            if (batch.kind == OverlayKind::Labels) {
                field(batch.chars, "chars");
                gpu.chars = batch.chars.Address();
                gpu.charStride = Words(Stride(batch.chars, 0), "chars");
                gpu.maxChars =
                    std::min<uint32_t>(uint32_t(batch.chars.Shape()[1]), kOverlayTextChars);
            }
            if (batch.kind == OverlayKind::Numbers) {
                PackAffix(batch.format.prefix, gpu.prefix, gpu.prefixLength, "prefix");
                PackAffix(batch.format.suffix, gpu.suffix, gpu.suffixLength, "suffix");
                gpu.decimals = batch.format.decimals;
                gpu.maxChars = std::min(gpu.prefixLength + gpu.suffixLength + 22 + gpu.decimals,
                                        kOverlayTextChars);
            }
            if (batch.placement) {
                const Float2 align = batch.placement->Alignment();
                gpu.flags |= kOverlayScreen;
                gpu.atX = batch.placement->at.x;
                gpu.atY = batch.placement->at.y;
                gpu.offsetX = batch.placement->offset.x;
                gpu.offsetY = batch.placement->offset.y;
                gpu.alignX = align.x;
                gpu.alignY = align.y;
            }
            if (trail) {
                gpu.trailLength = uint32_t(batch.points.Shape()[1]);
                gpu.endStride = Words(Stride(batch.points, 1), "points");
            }
            if (!batch.ends.Empty()) {
                field(batch.ends, "ends");
                gpu.ends = {batch.ends.Address()};
                gpu.endStride = rowStride(batch.ends, 1, "ends");
                gpu.endWorldStride = worldStride(batch.ends, 1, "ends");
            }
            if (!batch.colors.Empty()) {
                field(batch.colors, "colors");
                const bool bytes = batch.colors.Type() == device::DType::UInt8;
                const int64_t words = bytes ? 4 : 1;
                if (bytes && batch.colors.Address() % 4 != 0) {
                    throw UnsupportedError("Overlay {}' uint8 colors start on a 4-byte boundary",
                                           kind);
                }
                gpu.colors = {batch.colors.Address()};
                gpu.colorStride = rowStride(batch.colors, words, "colors");
                gpu.colorWorldStride = worldStride(batch.colors, words, "colors");
                gpu.flags |= kOverlayHasColors | (bytes ? kOverlayByteColors : 0);
            }
            if (!batch.sizes.Empty()) {
                field(batch.sizes, "widths");
                gpu.sizes = {batch.sizes.Address()};
                gpu.sizeStride = rowStride(batch.sizes, 1, "widths");
                gpu.sizeWorldStride = worldStride(batch.sizes, 1, "widths");
                gpu.flags |= kOverlayHasSizes;
            }
            if (!batch.rotations.Empty()) {
                field(batch.rotations, "orientations");
                gpu.rotations = {batch.rotations.Address()};
                gpu.rotationStride = rowStride(batch.rotations, 1, "orientations");
                gpu.rotationWorldStride = worldStride(batch.rotations, 1, "orientations");
                gpu.flags |= kOverlayHasRotations;
            }
            if (!batch.scalars.Empty()) {
                field(batch.scalars, "radii");
                gpu.scalars = {batch.scalars.Address()};
                gpu.scalarStride = rowStride(batch.scalars, 1, "radii");
                gpu.scalarWorldStride = worldStride(batch.scalars, 1, "radii");
                gpu.flags |= kOverlayHasScalars;
            }
            if (!batch.extras.Empty()) {
                field(batch.extras, trail ? "heads" : "half heights");
                gpu.extras = {batch.extras.Address()};
                gpu.extraStride = rowStride(batch.extras, 1, "extras");
                gpu.extraWorldStride = worldStride(batch.extras, 1, "extras");
                gpu.flags |= kOverlayHasExtras;
            }
            if (!batch.worlds.Empty()) {
                field(batch.worlds, "worlds");
                gpu.worlds = {batch.worlds.Address()};
                gpu.worldStride = Words(Stride(batch.worlds, 0), "worlds");
                gpu.flags |= kOverlayHasWorlds;
            }
            if (!batch.itemCount.Empty()) {
                field(batch.itemCount, "item counts");
                gpu.itemCounts = {batch.itemCount.Address()};
                gpu.itemCountStride = Words(Stride(batch.itemCount, 0), "item counts");
            }
            if (textual) {
                gpu.pieces = textPieces(gpu.maxChars);
            } else if (!primal) {
                gpu.pieces = PiecesPerRow(batch.kind, gpu.trailLength);
            }
            gpu.capacity = rows * gpu.pieces;
            if (!batch.worlds.Empty() && rows > 0) {
                Group(gpu, {index, batchIndex}, rows, worldCount, gpu.worlds.address,
                      gpu.worldStride, gpu.itemCounts.address);
            }
        }
        if (gpu.capacity == 0) {
            continue;
        }
        gpu.firstItem = firstItem;
        firstItem += gpu.capacity;
        batches.push_back(gpu);
        m_batchSources.push_back({index, batchIndex});
    }
}

void OverlayPass::Record(device::CommandEncoder& encoder, const Snapshot& snapshot,
                         uint32_t worldCount, std::span<const OverlayTarget> targets,
                         std::span<const OverlayContents> overlays,
                         std::span<const OverlayCameraSource> cameraImages,
                         std::span<const OverlayPickRequest> picks) {
    std::erase_if(m_uploads, [](const auto& entry) { return entry.second.contents.expired(); });
    if (overlays.size() > kMostOverlays) {
        throw UnsupportedError("{} overlays in one step; at most {} fit", overlays.size(),
                               kMostOverlays);
    }
    uint64_t shown = 0;
    for (const OverlayTarget& target : targets) {
        shown |= target.overlays;
    }
    std::vector<OverlayBatch> batches;
    uint32_t viewItemCount = 0;
    if (m_groupNeedsRead) {
        const auto* measured = static_cast<const uint32_t*>(m_groupNeedsRead->Mapped());
        for (size_t at = 0; at < m_measuredKeys.size(); ++at) {
            if (measured[at] > 0) {
                m_groupStrides[m_measuredKeys[at]] = measured[at];
            }
        }
    }
    m_groupParams.clear();
    m_groupKeys.clear();
    m_uniteParams.clear();
    m_outlines.clear();
    m_batchSources.clear();
    for (uint32_t index = 0; index < overlays.size(); ++index) {
        if (overlays[index] && ((shown >> index) & 1) != 0) {
            AddBatches(overlays[index], index, worldCount, batches, viewItemCount);
        }
    }
    uint64_t outlineMask = 0;
    for (const OverlayOutline& outline : m_outlines) {
        outlineMask |= uint64_t(1) << outline.overlay;
    }
    const bool outlined = std::ranges::any_of(targets, [&](const OverlayTarget& target) {
        return target.primIds != 0 && (target.overlays & outlineMask) != 0;
    });
    if (batches.empty() && !outlined && picks.empty()) {
        return;
    }

    std::vector<OverlayView> views;
    uint32_t tileCount = 0;
    for (const OverlayTarget& target : targets) {
        OverlayView view{};
        view.overlay = {target.overlay};
        view.depth = {target.depth};
        view.camera = target.camera;
        view.layer = target.layer;
        view.width = target.width;
        view.height = target.height;
        view.firstTile = tileCount;
        view.columnCount = (target.width + kOverlayTile - 1) / kOverlayTile;
        view.missDistance = target.missDistance;
        view.overlays = target.overlays;
        view.primIds = {target.primIds != 0 ? target.primIds : m_unused->Address()};
        view.outlined = target.primIds != 0 && (target.overlays & outlineMask) != 0 ? 1 : 0;
        tileCount += view.columnCount * ((target.height + kOverlayTile - 1) / kOverlayTile);
        views.push_back(view);
    }

    uint64_t fillCapacity = 0;
    for (const OverlayBatch& batch : batches) {
        const bool filled = batch.kind == uint32_t(OverlayKind::Triangles) ||
                            batch.kind == uint32_t(OverlayKind::Images) ||
                            (batch.kind == uint32_t(OverlayKind::Prims) &&
                             batch.primShape == uint32_t(OverlayPrimShape::CameraImages));
        fillCapacity += filled ? uint64_t(batch.capacity) * targets.size() : 0;
    }
    fillCapacity = std::min<uint64_t>(fillCapacity, std::numeric_limits<uint32_t>::max() / 2);
    Reserve(m_fills, std::max<uint64_t>(fillCapacity, 1) * sizeof(OverlayFill), "overlay fills");
    if (!m_fillCount) {
        m_fillCount = m_device.CreateBuffer({.bytes = 4, .label = "overlay fill count"});
    }
    std::vector<OverlayCameraImage> sources;
    for (const OverlayCameraSource& source : cameraImages) {
        const Span<const int64_t> shape = source.pixels.Shape();
        sources.push_back({.pixels = source.pixels.Address(),
                           .camera = source.camera,
                           .width = uint32_t(shape[shape.size() - 2]),
                           .height = uint32_t(shape[shape.size() - 3])});
    }
    if (!sources.empty()) {
        Reserve(m_cameraImages, sources.size() * sizeof(OverlayCameraImage),
                "overlay camera images");
    }
    if (!m_outlines.empty()) {
        Reserve(m_outlineBuffer, m_outlines.size() * sizeof(OverlayOutline), "overlay outlines");
    }
    std::vector<OverlayPickQuery> queries;
    for (const OverlayPickRequest& pick : picks) {
        queries.push_back({pick.target, pick.pixel.x, pick.pixel.y, pick.radius});
    }
    if (!queries.empty()) {
        Reserve(m_pickQueries, queries.size() * sizeof(OverlayPickQuery), "overlay pick queries");
        Reserve(m_pickResults, queries.size() * sizeof(OverlayPickResult), "overlay pick results");
    }

    const uint64_t workCount = uint64_t(viewItemCount) * targets.size();
    if (workCount > algorithms::Scan::kMaxItems) {
        throw UnsupportedError("{} overlay items across {} images; at most {} fit one step",
                               viewItemCount, targets.size(), algorithms::Scan::kMaxItems);
    }
    if (workCount > m_workCapacity || !m_scan) {
        m_workCapacity = uint32_t(std::min<uint64_t>(
            std::max<uint64_t>(workCount + workCount / 2, 64), algorithms::Scan::kMaxItems));
        m_scan.emplace(m_device, m_workCapacity);
        Reserve(m_records, uint64_t(m_workCapacity) * sizeof(OverlayRecord), "overlay records");
        Reserve(m_tileCounts, uint64_t(m_workCapacity) * 4, "overlay tile counts");
        Reserve(m_offsets, uint64_t(m_workCapacity) * 4, "overlay offsets");
    }
    const uint64_t needed = *static_cast<const uint32_t*>(m_pairsNeeded->Mapped());
    const uint64_t wanted =
        std::max(kLeastPairs, needed > 0 ? std::max(needed, 2 * workCount) : 8 * workCount);
    if (wanted > m_pairCapacity || 3 * wanted < m_pairCapacity) {
        if (needed > m_pairCapacity && m_pairCapacity > 0) {
            Log(LogLevel::Debug, "overlays needed {} tile pairs, so their buffers grow", needed);
        }
        m_pairCapacity =
            uint32_t(std::min<uint64_t>(wanted + wanted / 2, algorithms::RadixSort::kMaxItems));
        m_sort.emplace(m_device, m_pairCapacity, 4);
        Reserve(m_keys, uint64_t(m_pairCapacity) * 4, "overlay keys");
        Reserve(m_values, uint64_t(m_pairCapacity) * 4, "overlay values");
    }
    Reserve(m_views, views.size() * sizeof(OverlayView), "overlay views");
    Reserve(m_batches, std::max<size_t>(batches.size(), 1) * sizeof(OverlayBatch),
            "overlay batches");
    const uint64_t occupiedBytes = (kOverlayTileList + uint64_t(kOverlayTileWords) * tileCount) * 4;
    if (!m_occupied || m_occupied->Bytes() < occupiedBytes) {
        m_occupied = m_device.CreateBuffer({.bytes = std::max<uint64_t>(occupiedBytes, 256),
                                            .usage = device::BufferUsage::Indirect,
                                            .label = "overlay occupied tiles"});
    }

    const bool textual = std::ranges::any_of(batches, [](const OverlayBatch& batch) {
        return batch.kind >= uint32_t(OverlayKind::Labels);
    });
    if (textual && !m_fontField) {
        const OverlayFont& font = OverlayTextFont();
        std::vector<uint8_t> field = font.field;
        field.resize((field.size() + 3) / 4 * 4 + 4, 0);
        std::vector<int8_t> kerning = font.kerning;
        kerning.resize((kerning.size() + 3) / 4 * 4 + 4, 0);
        m_fontField = m_device.CreateBuffer({.label = "overlay font"}, field);
        m_fontGlyphs = m_device.CreateBuffer(
            {.label = "overlay glyphs"},
            std::vector<OverlayGlyph>(font.glyphs.begin(), font.glyphs.end()));
        m_fontKerning = m_device.CreateBuffer({.label = "overlay kerning"}, kerning);
    }

    OverlayParams params{};
    params.font = m_fontField ? m_fontField->Address() : m_unused->Address();
    params.glyphs = {m_fontGlyphs ? m_fontGlyphs->Address() : m_unused->Address()};
    params.kerning = m_fontKerning ? m_fontKerning->Address() : m_unused->Address();
    params.textHeight = m_fontField ? OverlayTextFont().textHeight : 0;
    params.scene = {snapshot.sceneData};
    params.layout = {snapshot.instances};
    params.views = {m_views->Address()};
    params.batches = {m_batches->Address()};
    params.records = {m_records->Address()};
    params.tileCounts = {m_tileCounts->Address()};
    params.offsets = {m_offsets->Address()};
    params.keys = {m_keys->Address()};
    params.values = {m_values->Address()};
    params.pairCount = {m_pairCount->Address()};
    params.occupied = {m_occupied->Address()};
    params.viewCount = uint32_t(views.size());
    params.batchCount = uint32_t(batches.size());
    params.viewItemCount = viewItemCount;
    params.workCount = uint32_t(workCount);
    params.tileCount = tileCount;
    params.pairCapacity = m_pairCapacity;
    params.fills = {m_fills->Address()};
    params.fillCount = {m_fillCount->Address()};
    params.fillCapacity = uint32_t(fillCapacity);
    params.cameraImages = {sources.empty() ? m_unused->Address() : m_cameraImages->Address()};
    params.cameraImageCount = uint32_t(sources.size());
    params.outlines = {m_outlines.empty() ? m_unused->Address() : m_outlineBuffer->Address()};
    params.outlineCount = uint32_t(m_outlines.size());
    params.picks = {queries.empty() ? m_unused->Address() : m_pickQueries->Address()};
    params.pickResults = {queries.empty() ? m_unused->Address() : m_pickResults->Address()};
    params.pickCount = uint32_t(queries.size());

    static const profiler::ZoneId zone = profiler::ZoneId::Named("Render/Overlays");
    const device::ZoneScope scope(encoder, zone);
    encoder.Barrier();
    const auto update = [&](device::Buffer& buffer, const auto& values) {
        const uint64_t bytes = values.size() * sizeof(values[0]);
        const auto* data = reinterpret_cast<const std::byte*>(values.data());
        for (uint64_t at = 0; at < bytes; at += device::CommandEncoder::kMaxUpdateBytes) {
            const uint64_t chunk = std::min(bytes - at, device::CommandEncoder::kMaxUpdateBytes);
            encoder.Update(buffer.Slice(at, chunk), data + at);
        }
    };
    update(*m_views, views);
    update(*m_batches, batches);
    if (!sources.empty()) {
        update(*m_cameraImages, sources);
    }
    if (!m_outlines.empty()) {
        update(*m_outlineBuffer, m_outlines);
    }
    if (!queries.empty()) {
        update(*m_pickQueries, queries);
    }
    encoder.Fill(m_occupied->Slice(0, uint64_t(kOverlayTileList) * 4));
    encoder.Fill(*m_fillCount);
    static const profiler::ZoneId projecting = profiler::ZoneId::Named("Render/Overlays/Project");
    static const profiler::ZoneId scanning = profiler::ZoneId::Named("Render/Overlays/Scan");
    static const profiler::ZoneId scattering = profiler::ZoneId::Named("Render/Overlays/Scatter");
    static const profiler::ZoneId sorting = profiler::ZoneId::Named("Render/Overlays/Sort");
    static const profiler::ZoneId ranging = profiler::ZoneId::Named("Render/Overlays/Tiles");
    static const profiler::ZoneId compositing =
        profiler::ZoneId::Named("Render/Overlays/Composite");
    static const profiler::ZoneId outlining = profiler::ZoneId::Named("Render/Overlays/Outline");
    const auto outline = [&] {
        if (!outlined) {
            return;
        }
        encoder.Barrier();
        encoder.BeginZone(outlining);
        const uint32_t tiles = std::min(tileCount, kOverlayMostGroups);
        encoder.Dispatch(m_program->Entry("OutlineOverlays"), params,
                         {.x = tiles, .y = (tileCount + tiles - 1) / tiles});
        encoder.EndZone();
    };
    static const profiler::ZoneId picking = profiler::ZoneId::Named("Render/Overlays/Pick");
    const auto pick = [&] {
        if (queries.empty()) {
            return;
        }
        encoder.Barrier();
        encoder.BeginZone(picking);
        const uint32_t count = params.pickCount;
        const uint64_t threads = uint64_t(count) * params.viewItemCount;
        encoder.Dispatch(m_program->Entry("ClearOverlayPicks"), params,
                         {.x = device::GroupCount(count, kOverlayGroup)});
        if (threads > 0) {
            encoder.Barrier();
            encoder.Dispatch(m_program->Entry("MeasureOverlayPicks"), params,
                             {.x = device::GroupCount(threads, kOverlayGroup)});
            encoder.Barrier();
            encoder.Dispatch(m_program->Entry("ChooseOverlayPicks"), params,
                             {.x = device::GroupCount(threads, kOverlayGroup)});
        }
        encoder.Barrier();
        encoder.Dispatch(m_program->Entry("ResolveOverlayPicks"), params,
                         {.x = device::GroupCount(count, kOverlayGroup)});
        encoder.Barrier();
        for (uint32_t at = 0; at < count; ++at) {
            encoder.Copy(m_pickResults->Slice(uint64_t(at) * sizeof(OverlayPickResult),
                                              sizeof(OverlayPickResult)),
                         picks[at].readback->Slice(0, sizeof(OverlayPickResult)));
        }
        encoder.EndZone();
    };
    if (batches.empty()) {
        outline();
        pick();
        return;
    }
    RecordGroups(encoder);
    RecordUnions(encoder, snapshot);
    encoder.Barrier();
    encoder.BeginZone(projecting);
    encoder.Dispatch(m_program->Entry("ProjectOverlays"), params,
                     {.x = device::GroupCount(params.workCount, kOverlayGroup)});
    encoder.EndZone();
    encoder.BeginZone(scanning);
    m_scan->ExclusiveSum(encoder, m_tileCounts->Slice(0, workCount * 4),
                         m_offsets->Slice(0, workCount * 4), params.workCount);
    encoder.EndZone();
    encoder.Barrier();
    encoder.BeginZone(scattering);
    encoder.Dispatch(m_program->Entry("ScatterOverlays"), params,
                     {.x = device::GroupCount(params.workCount, kOverlayGroup)});
    encoder.EndZone();
    encoder.BeginZone(sorting);
    m_sort->Sort(encoder, *m_keys, *m_values, *m_pairCount, m_pairCapacity,
                 uint32_t(std::max(1, std::bit_width(tileCount - 1))));
    encoder.EndZone();
    encoder.Barrier();
    encoder.BeginZone(ranging);
    encoder.Dispatch(m_program->Entry("FindOverlayTiles"), params,
                     {.x = device::GroupCount(m_pairCapacity, kOverlayGroup)});
    encoder.EndZone();
    encoder.Barrier();
    encoder.BeginZone(compositing);
    encoder.Dispatch(m_program->Entry("FoldOverlayTiles"), params, {.x = 1});
    encoder.Barrier();
    const uint32_t across = std::min(tileCount, kOverlayMostGroups);
    encoder.Dispatch(m_program->Entry("CompositeOverlays"), params,
                     {.x = across, .y = (tileCount + across - 1) / across},
                     m_occupied->Slice(4, 12));
    encoder.EndZone();
    outline();
    pick();
    encoder.Barrier();
    encoder.Copy(*m_pairCount, *m_pairsNeeded);
    if (!m_groupParams.empty()) {
        const uint64_t bytes = m_groupParams.size() * 4;
        encoder.Copy(m_groupNeeds->Slice(0, bytes), m_groupNeedsRead->Slice(0, bytes));
    }
    m_measuredKeys = m_groupKeys;
}

void OverlayPass::RecordUnions(device::CommandEncoder& encoder, const Snapshot& snapshot) {
    if (m_uniteParams.empty()) {
        return;
    }
    encoder.Barrier();
    for (OverlayUniteParams& params : m_uniteParams) {
        params.scene = {snapshot.sceneData};
        encoder.Dispatch(
            m_program->Entry("ClearOverlayBoxes"), params,
            {.x = device::GroupCount(params.boxCount * kOverlayBoxWords, kOverlayGroup)});
    }
    encoder.Barrier();
    for (const OverlayUniteParams& params : m_uniteParams) {
        encoder.Dispatch(m_program->Entry("UniteOverlayBoxes"), params,
                         {.x = device::GroupCount(params.primCount, kOverlayGroup)});
    }
}

void OverlayPass::RecordGroups(device::CommandEncoder& encoder) {
    if (m_groupParams.empty()) {
        return;
    }
    static const profiler::ZoneId grouping = profiler::ZoneId::Named("Render/Overlays/Group");
    const device::ZoneScope scope(encoder, grouping);
    uint32_t rows = 0;
    for (const OverlayGroupParams& params : m_groupParams) {
        rows = std::max(rows, params.rows);
    }
    if (rows > m_groupSortCapacity) {
        m_groupSortCapacity = rows + rows / 2;
        m_groupSort.emplace(m_device, m_groupSortCapacity, 4);
    }
    const uint64_t needBytes = std::max<uint64_t>(m_groupParams.size() * 4, 256);
    if (!m_groupNeeds || m_groupNeeds->Bytes() < needBytes) {
        m_groupNeeds = m_device.CreateBuffer({.bytes = needBytes, .label = "overlay group needs"});
        m_groupNeedsRead = m_device.CreateBuffer({.bytes = needBytes,
                                                  .memory = device::Memory::Readback,
                                                  .label = "overlay group needs read"});
        std::fill_n(static_cast<uint32_t*>(m_groupNeedsRead->Mapped()), needBytes / 4, 0u);
    }
    encoder.Fill(m_groupNeeds->Slice(0, m_groupParams.size() * 4));
    for (OverlayGroupParams& params : m_groupParams) {
        params.needs = {m_groupNeeds->Address()};
    }
    for (size_t at = 0; at < m_groupParams.size(); ++at) {
        const OverlayGroupParams& params = m_groupParams[at];
        const Grouping& buffers = m_groupings[at];
        encoder.Barrier();
        encoder.Dispatch(m_program->Entry("KeyOverlayRows"), params,
                         {.x = device::GroupCount(params.rows, kOverlayGroup)});
        m_groupSort->Sort(encoder, buffers.keys->Slice(0, uint64_t(params.rows) * 4),
                          buffers.order->Slice(0, uint64_t(params.rows) * 4), params.rows,
                          uint32_t(std::max(1, std::bit_width(params.groups + 1))));
        encoder.Barrier();
        encoder.Dispatch(m_program->Entry("StartOverlayGroups"), params,
                         {.x = device::GroupCount(params.groups + 3, kOverlayGroup)});
        encoder.Barrier();
        encoder.Dispatch(m_program->Entry("MeasureOverlayGroups"), params,
                         {.x = device::GroupCount(std::max(params.groups, 1u), kOverlayGroup)});
    }
}

device::Tensor CompositeOverlay(const device::Tensor& color, const device::Tensor& overlay,
                                const device::Operation& after) {
    const auto check = [](const device::Tensor& image, const char* what) {
        const Span<const int64_t> shape = image.Shape();
        if (image.Empty() || image.Type() != device::DType::UInt8 || shape.size() < 3 ||
            shape.back() != 4) {
            throw Error("A composited {} is a [..., height, width, 4] uint8 tensor", what);
        }
        if (!image.Strides().empty()) {
            int64_t expected = 1;
            for (size_t axis = shape.size(); axis-- > 0;) {
                if (shape[axis] > 1 && image.Strides()[axis] != expected) {
                    throw UnsupportedError("A composited {} is contiguous", what);
                }
                expected *= shape[axis];
            }
        }
    };
    check(color, "color");
    check(overlay, "overlay");
    if (!std::ranges::equal(color.Shape(), overlay.Shape())) {
        throw Error("An overlay composites over color of its own shape");
    }
    device::Device& device = color.GetDevice();
    if (&overlay.GetDevice() != &device) {
        throw Error("An overlay composites over color on its own device, not {}",
                    overlay.GetDevice().Adapter().alias);
    }
    const uint64_t pixels = uint64_t(color.Elements()) / 4;
    if (pixels > std::numeric_limits<uint32_t>::max()) {
        throw UnsupportedError("Compositing {} pixels at once; at most {} fit", pixels,
                               std::numeric_limits<uint32_t>::max());
    }
    device::Tensor blended = device.CreateTensor(
        device::DType::UInt8, {color.Shape().begin(), color.Shape().end()}, "composited overlay");
    const Ref<device::Program> program =
        device.CreateProgram({.modules = {"ernie.render.kernels.overlay.overlay"},
                              .library = &kRenderKernels,
                              .entries = {"BlendOverlay"},
                              .label = "overlays"});
    device::Queue& queue = device.GetQueue();
    queue.Follow(after);
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.Dispatch(
        program->Entry("BlendOverlay"),
        BlendParams{
            {color.Address()}, {overlay.Address()}, {blended.Address()}, uint32_t(pixels), 0},
        {.x = device::GroupCount(pixels, kOverlayGroup)});
    queue.Retire(std::make_shared<std::pair<device::Tensor, device::Tensor>>(color, overlay),
                 queue.Submit(encoder.Finish()));
    return blended;
}

}
