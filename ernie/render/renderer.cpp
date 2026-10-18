// SPDX-License-Identifier: BSD-3-Clause
/// @file renderer.cpp
/// @brief Rendering a scene's render settings: the device scene brought up to date, the camera
/// traced into a column, and the column read back.
#include "ernie/render/renderer.h"

#include "ernie/device/staging.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/render.h"
#include "ernie/util/profiler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

/// @return The camera a render setting names, as a pinhole view of `size` pixels. The vertical
/// field of view widens until the aperture's width fits too, as `expandAperture` conforms it.
PinholeView View(const Scene& scene, const Path& path, glm::uvec2 size) {
    const Camera camera(scene, path);
    const double focal = camera.GetFocalLength();
    const double aspect = static_cast<double>(size.x) / static_cast<double>(size.y);
    const double tangent = std::max(0.5 * camera.GetVerticalAperture() / focal,
                                    0.5 * camera.GetHorizontalAperture() / focal / aspect);
    const std::optional<PrimId> prim = scene.Find(path);
    return {.cameraToWorld = scene.WorldTransform(*prim),
            .verticalFov = 2.0 * std::atan(tangent),
            .size = size};
}

}  // namespace

Renderer::Renderer(const RendererOptions& options)
    : context_(std::make_unique<DeviceContext>(
          DeviceContextOptions{.validation = options.validation, .device = options.device})) {
    if (!context_->Capabilities().rayQuery) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          context_->Info().name + " cannot ray trace");
    }
    arenas_ =
        std::make_unique<Arenas>(*context_, MemoryUse::Device, "renderer", std::uint64_t{16} << 20);
    heap_ = std::make_unique<BindlessHeap>(*context_);
    camera_ = std::make_unique<CameraPass>(*context_, ShaderCompiler(), *heap_);
    ldr_ = std::make_unique<DeviceColumn>(*arenas_, "ldr colour", DType::UInt8,
                                          std::vector<std::int64_t>{4}, 0);
}

Renderer::~Renderer() {
    context_->WaitIdle();
}

Tensor Renderer::Render(const Scene& scene) {
    ERNIE_ZONE("Render");
    const std::vector<Path> settings = scene.Prims(RenderSettings::kName);
    if (settings.empty()) {
        throw std::invalid_argument("The scene has no RenderSettings to render");
    }
    const RenderSettings render(scene, settings.front());
    const Path camera = render.GetCamera();
    if (camera.IsEmpty() || !scene.Find(camera)) {
        throw std::invalid_argument("'" + settings.front().String() + "' names no camera");
    }
    const glm::dvec2 resolution = render.GetResolution();
    const glm::uvec2 size(std::max(1.0, resolution.x), std::max(1.0, resolution.y));
    const PinholeView view = View(scene, camera, size);

    if (&scene != scene_) {
        context_->WaitIdle();
        device_ = std::make_unique<DeviceScene>(*context_, *arenas_, scene, ShaderCompiler());
        scene_ = &scene;
    }
    const std::size_t pixels = std::size_t{size.x} * size.y;
    ldr_->Resize(pixels, context_->Main());
    device_->Follow(glm::dvec3(view.cameraToWorld[3]));
    const Ticket done = device_->Execute([&](graph::Graph& graph) {
        CameraOutputs outputs{.ldr = graph.Import(*ldr_, ldr_->Name())};
        camera_->Add(graph, view, outputs);
    });
    Readback readback(*context_, context_->Main(), ldr_->Storage(), ldr_->Range().offset,
                      pixels * 4, std::span(&done, 1));
    return Tensor::Copy(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(readback.Bytes().data()), pixels * 4),
        {size.y, size.x, 4});
}

}  // namespace ernie
