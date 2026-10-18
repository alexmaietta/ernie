// SPDX-License-Identifier: BSD-3-Clause
/// @file renderer.h
/// @brief Rendering a scene's render settings to an image in one call.
///
///     Renderer renderer;
///     Tensor image = renderer.Render(scene);  // height x width x 4 bytes of LdrColor
#pragma once

#include "ernie/device/bindless.h"
#include "ernie/device/context.h"
#include "ernie/render/camera_pass.h"
#include "ernie/render/device_scene.h"
#include "ernie/scene/scene.h"
#include "ernie/util/tensor.h"

#include <memory>
#include <string>

namespace ernie {

struct RendererOptions {
    /// The device: its UUID in hex, or part of its name; empty for the best scoring one.
    std::string device;
    /// The Khronos validation layer with synchronization validation, when installed.
    bool validation = false;
};

/// @brief A device and what it keeps of the scene it last rendered, so rendering the same scene
/// again costs only what changed in it.
class Renderer {
  public:
    /// @throws DeviceError without a device that can ray trace.
    explicit Renderer(const RendererOptions& options = {});
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    /// @brief Renders the camera of the scene's first `RenderSettings` at its resolution, as
    /// Minimal sees it: albedo lit from the eye.
    /// @return LdrColor, height x width x 4 bytes (RGBA8, display encoded) on the host.
    /// @throws std::invalid_argument when the scene has no render settings or they name no camera.
    Tensor Render(const Scene& scene);

    DeviceContext& Context() {
        return *context_;
    }

  private:
    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<Arenas> arenas_;
    std::unique_ptr<BindlessHeap> heap_;
    std::unique_ptr<CameraPass> camera_;
    const Scene* scene_ = nullptr;
    std::unique_ptr<DeviceScene> device_;
    std::unique_ptr<DeviceColumn> ldr_;
};

}  // namespace ernie
