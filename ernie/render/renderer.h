// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/device/ref.h"
#include "ernie/device/stream.h"
#include "ernie/export.h"
#include "ernie/overlay/overlay.h"
#include "ernie/render/render_output.h"
#include "ernie/util/span.h"

namespace ernie {

class RendererImpl;
class Scene;

struct ERNIE_API RendererDesc {
    uint64_t maxPixelsPerLaunch = uint64_t(1) << 30;
    Ref<Overlays> overlays;
};

class ERNIE_API Renderer {
  public:
    explicit Renderer(Ref<Scene> scene, const RendererDesc& desc = {});
    ~Renderer();
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;

    RenderOutput Step(Span<const std::string> products, double dt, Stream stream = Stream::None());
    RenderOutput Step(std::string_view product, double dt, Stream stream = Stream::None());

    const Ref<Overlays>& GetOverlays() const;

  private:
    std::unique_ptr<RendererImpl> m_impl;
};

}
