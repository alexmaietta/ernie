// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/operation.h"
#include "ernie/device/ref.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/overlay/kernels/style.h"
#include "ernie/stage/schema/ids.h"
#include "ernie/stage/schema/types.h"
#include "ernie/util/span.h"

namespace ernie {

struct ERNIE_API OverlayStyle {
    Float4 color = {1, 1, 1, 1};
    float width = 2;
    float size = 8;
    MarkerShape marker = MarkerShape::Circle;
    OverlayDepth depth = OverlayDepth::Test;
    uint32_t world = kGlobalWorld;
    uint32_t anchor = kNoPrim;
    Float4 background = {0, 0, 0, 0.55f};
    Float4 halo = {0, 0, 0, 0};
    Float2 align = {0, 0.5f};
    float dash = 0;
    float gap = 0;
    OverlayUnits units = OverlayUnits::Pixels;
    bool screenSized = false;
    bool facing = false;
    bool pickable = true;
    uint32_t id = 0;
    std::optional<Float2> screen;
};

struct ERNIE_API ScreenPlacement {
    Float2 at;
    Float2 offset;
    std::optional<Float2> align;

    Float2 Alignment() const;
};

struct ERNIE_API OverlayNumberFormat {
    std::string prefix;
    std::string suffix;
    uint32_t decimals = 2;
};

struct ERNIE_API OverlayLines {
    device::Tensor points;
    device::Tensor ends;
    device::Tensor colors;
    device::Tensor widths;
    device::Tensor worlds;
    device::Tensor itemCount;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayMarkers {
    device::Tensor points;
    device::Tensor colors;
    device::Tensor sizes;
    device::Tensor worlds;
    device::Tensor itemCount;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayBoxes {
    device::Tensor points;
    device::Tensor halfExtents;
    device::Tensor orientations;
    device::Tensor colors;
    device::Tensor widths;
    device::Tensor worlds;
    device::Tensor itemCount;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlaySpheres {
    device::Tensor points;
    device::Tensor radii;
    device::Tensor colors;
    device::Tensor widths;
    device::Tensor worlds;
    device::Tensor itemCount;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayCircles {
    device::Tensor points;
    device::Tensor normals;
    device::Tensor radii;
    device::Tensor colors;
    device::Tensor widths;
    device::Tensor worlds;
    device::Tensor itemCount;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayFrames {
    device::Tensor points;
    device::Tensor orientations;
    device::Tensor lengths;
    device::Tensor widths;
    device::Tensor worlds;
    device::Tensor itemCount;
    float length = 1;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayArrows {
    device::Tensor points;
    device::Tensor ends;
    device::Tensor vectors;
    float vectorScale = 1;
    device::Tensor headLengths;
    device::Tensor colors;
    device::Tensor widths;
    device::Tensor worlds;
    device::Tensor itemCount;
    float headLength = 0;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayCapsules {
    device::Tensor points;
    device::Tensor orientations;
    device::Tensor radii;
    device::Tensor halfHeights;
    device::Tensor colors;
    device::Tensor widths;
    device::Tensor worlds;
    device::Tensor itemCount;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayTrails {
    device::Tensor points;
    device::Tensor heads;
    device::Tensor colors;
    device::Tensor widths;
    device::Tensor worlds;
    device::Tensor itemCount;
    float fade = 0;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayTexts {
    device::Tensor points;
    device::Tensor chars;
    device::Tensor colors;
    device::Tensor sizes;
    device::Tensor worlds;
    device::Tensor itemCount;
    std::optional<ScreenPlacement> placement;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayNumbers {
    device::Tensor points;
    device::Tensor values;
    OverlayNumberFormat format;
    device::Tensor colors;
    device::Tensor sizes;
    device::Tensor worlds;
    device::Tensor itemCount;
    std::optional<ScreenPlacement> placement;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayTriangles {
    device::Tensor points;
    device::Tensor colors;
    device::Tensor worlds;
    device::Tensor itemCount;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayImages {
    device::Tensor points;
    device::Tensor rights;
    device::Tensor downs;
    device::Tensor pixels;
    device::Tensor colors;
    device::Tensor worlds;
    device::Tensor itemCount;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct ERNIE_API OverlayPrims {
    device::Tensor prims;
    device::Tensor groups;
    uint32_t groupCount = 0;
    OverlayPrimShape shape = OverlayPrimShape::Frames;
    float length = 1;
    device::Tensor pixels;
    device::Tensor colors;
    device::Tensor widths;
    OverlayStyle style;
    std::shared_ptr<const void> owner;
};

struct OverlayData;

struct ERNIE_API OverlayHit {
    std::string overlay;
    uint32_t id = 0;
    uint32_t batch = 0;
    uint32_t item = 0;
    uint32_t piece = 0;
    uint32_t world = kGlobalWorld;
    uint32_t prim = kNoPrim;
    float distance = 0;
    float depth = 0;
    Float3 position;
};

struct ERNIE_API OverlayRay {
    Float3 origin;
    Float3 direction;
};

struct OverlayPickState;

class ERNIE_API OverlayPick {
  public:
    OverlayPick() = default;
    explicit OverlayPick(std::shared_ptr<OverlayPickState> state);

    bool Ready() const;
    void Wait() const;
    std::optional<OverlayHit> Hit() const;
    std::optional<OverlayRay> Ray() const;

  private:
    std::shared_ptr<OverlayPickState> m_state;
};

class ERNIE_API Overlay {
  public:
    Overlay();
    ~Overlay();
    Overlay(const Overlay& other);
    Overlay(Overlay&& other) noexcept;
    Overlay& operator=(const Overlay& other);
    Overlay& operator=(Overlay&& other) noexcept;

    Overlay& Line(Float3 a, Float3 b, const OverlayStyle& style = {});
    Overlay& Lines(Span<const Float3> endpoints, const OverlayStyle& style = {},
                   Span<const Float4> colors = {});
    Overlay& Polyline(Span<const Float3> points, bool closed = false,
                      const OverlayStyle& style = {});
    Overlay& Marker(Float3 point, const OverlayStyle& style = {});
    Overlay& Markers(Span<const Float3> points, const OverlayStyle& style = {},
                     Span<const Float4> colors = {});
    Overlay& Box(Float3 center, Float3 halfExtents, const OverlayStyle& style = {},
                 Quatd orientation = {});
    Overlay& Sphere(Float3 center, float radius, const OverlayStyle& style = {});
    Overlay& Circle(Float3 center, Float3 normal, float radius, const OverlayStyle& style = {});
    Overlay& Frame(Float3 origin, Quatd orientation, float length = 1,
                   const OverlayStyle& style = {});
    Overlay& Arrow(Float3 from, Float3 to, const OverlayStyle& style = {}, float headLength = 0);
    Overlay& Capsule(Float3 center, Quatd orientation, float radius, float halfHeight,
                     const OverlayStyle& style = {});
    Overlay& Label(Float3 point, std::string_view text, const OverlayStyle& style = {});
    Overlay& Text(const ScreenPlacement& placement, std::string_view text,
                  const OverlayStyle& style = {});
    Overlay& Triangle(Float3 a, Float3 b, Float3 c, const OverlayStyle& style = {});
    Overlay& Polygon(Span<const Float3> points, const OverlayStyle& style = {});
    Overlay& Curve(Span<const Float3> controls, const OverlayStyle& style = {},
                   uint32_t tessellation = 16);
    Overlay& Arc(Float3 center, Float3 normal, float radius, float begin, float end,
                 const OverlayStyle& style = {}, bool filled = false);
    Overlay& Rectangle(Float3 center, Float3 halfRight, Float3 halfUp,
                       const OverlayStyle& style = {}, bool filled = false);
    Overlay& Mesh(Span<const Float3> points, Span<const uint32_t> faceCounts,
                  Span<const uint32_t> indices, const OverlayStyle& style = {},
                  bool wireframe = false);
    Overlay& Image(Float3 topLeft, Float3 right, Float3 down, device::Tensor pixels,
                   const OverlayStyle& style = {});
    Overlay& Prims(Span<const uint32_t> prims, OverlayPrimShape shape,
                   const OverlayStyle& style = {}, float length = 1,
                   Span<const uint32_t> groups = {}, Span<const Float4> colors = {});
    Overlay& Outline(Span<const uint32_t> prims, const OverlayStyle& style = {});
    Overlay& Add(OverlayLines lines);
    Overlay& Add(OverlayMarkers markers);
    Overlay& Add(OverlayBoxes boxes);
    Overlay& Add(OverlaySpheres spheres);
    Overlay& Add(OverlayCircles circles);
    Overlay& Add(OverlayFrames frames);
    Overlay& Add(OverlayArrows arrows);
    Overlay& Add(OverlayCapsules capsules);
    Overlay& Add(OverlayTrails trails);
    Overlay& Add(OverlayTexts texts);
    Overlay& Add(OverlayNumbers numbers);
    Overlay& Add(OverlayTriangles triangles);
    Overlay& Add(OverlayImages images);
    Overlay& Add(OverlayPrims prims);

    bool Empty() const;

  private:
    friend class OverlayPass;

    OverlayData& Edit();

    std::shared_ptr<OverlayData> m_data;
};

struct OverlaysImpl;

class ERNIE_API Overlays : public RefCounted {
  public:
    static Ref<Overlays> Create(device::Device& device);
    ~Overlays() override;

    void Set(std::string_view name, Overlay overlay);
    void Remove(std::string_view name);
    std::vector<std::string> Names() const;
    OverlayPick Pick(std::string_view product, Float2 pixel, float radius = 6, uint32_t layer = 0);

  private:
    friend struct OverlaysImpl;

    explicit Overlays(device::Device& device);

    std::unique_ptr<OverlaysImpl> m_impl;
};

ERNIE_API device::Tensor CompositeOverlay(const device::Tensor& color,
                                          const device::Tensor& overlay,
                                          const device::Operation& after = {});

}
