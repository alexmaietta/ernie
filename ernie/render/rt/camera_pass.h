// SPDX-License-Identifier: BSD-3-Clause
/// @file camera_pass.h
/// @brief Cameras traced through the scene by their lenses, every view in one dispatch: a shaded
/// colour image to look at, and each pixel's distance, normal, instance, prim and albedo in
/// buffers kernels, CUDA or PyTorch read in place, the views stacked one after another.
///
///     CameraOutputs outputs{.colour = graph.Import(image, "colour"), .colourIndex = index};
///     camera.Add(graph, {.cameraToWorld = orbit.Transform(), .size = {640, 480},
///                        .lens = MakePinhole(0.8, {640, 480})}, outputs);
#pragma once

#include "ernie/device/bindless.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/graph/graph.h"
#include "ernie/shaders/scene/transform.slang"
#include "ernie/shaders/sensors/camera.slang"
#include "ernie/shaders/spectral/spectrum.slang"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <optional>
#include <span>

namespace ernie {

/// @brief A lens model and its pixel intrinsics, as lens.slang defines them.
using Lens = shaders::Lens;

/// @return A pinhole lens for an image of `size` pixels with a vertical field of view of
/// `verticalFov` radians, square pixels and the principal point in the middle.
Lens MakePinhole(double verticalFov, glm::uvec2 size);

/// @brief A camera in the scene's frame, in double: the pass makes it relative to the render
/// origin before it rounds to float.
struct CameraView {
    /// Camera to world; the camera looks down -z with +y up, as `glm::lookAtRH`'s inverse.
    glm::dmat4 cameraToWorld{1.0};
    /// Pixels.
    glm::uvec2 size{1};
    Lens lens = MakePinhole(0.785398, {1, 1});
    /// The camera prim's transform-graph node, whose world the pass reads on the device in place
    /// of `cameraToWorld`, so poses CUDA or kernels write move the view; or `kNone`.
    std::uint32_t node = shaders::kNone;
    /// The batch world it sees: its prims and every world's shared ones.
    std::uint32_t world = 0;
    /// Metres along the view axis to the near and far clipping planes.
    glm::vec2 clipping{0.0f, 1e30f};
};

/// @brief What a camera pass writes; each left out is not written. The views' pixels are stacked
/// in each buffer, row by row, each view's after the one before. Each written one is replaced by
/// its version after the pass.
struct CameraOutputs {
    /// A storage image of the first view's size, and its slot in the pass's heap.
    std::optional<graph::Image> colour;
    BindlessIndex colourIndex;
    /// What `colour` shows where nothing is hit, as it is stored.
    glm::vec3 background{0.0f};
    /// A float a pixel: the hit distance in metres, infinity where nothing is hit.
    std::optional<graph::Buffer> distances;
    /// Three floats a pixel: the world normal, zero where nothing is hit.
    std::optional<graph::Buffer> normals;
    /// A uint32 a pixel: the instance row hit, or `kNoInstance`.
    std::optional<graph::Buffer> instances;
    /// A uint32 a pixel: the slot of the prim hit, its `PrimId::slot`, or `kNoInstance`.
    std::optional<graph::Buffer> prims;
    /// `kBands` floats a pixel: the albedo hit, per spectral band; zero where nothing is hit.
    std::optional<graph::Buffer> albedos;
    /// Four bytes a pixel: the colour as RGBA8, as `colour` holds it.
    std::optional<graph::Buffer> ldr;
};

/// @brief Traces cameras through the scene the graph's `TraceScene` describes, one ray a pixel,
/// shading what each hits by its albedo and a headlight.
class CameraPass {
  public:
    static constexpr std::uint32_t kNoInstance = shaders::kNoInstance;
    /// The spectral bands albedos carry: RGB mode's three, linear sRGB.
    static constexpr std::uint32_t kBands = shaders::kBands;

    CameraPass(DeviceContext& context, const ShaderCompiler& compiler, const BindlessHeap& heap);

    /// @return The modules, in import order, a kernel compiles with before its own to import
    /// `camera`.
    static std::span<const ShaderModule> Modules();

    /// @brief Adds the pass tracing every view in one dispatch, after the ray tracer's builds.
    /// @throws std::invalid_argument when the graph has no `TraceScene`, or a view's world is not
    /// one of its worlds.
    void Add(graph::Graph& graph, std::span<const CameraView> views, CameraOutputs& outputs) const;
    void Add(graph::Graph& graph, const CameraView& view, CameraOutputs& outputs) const;

  private:
    ComputeKernel kernel_;
};

}  // namespace ernie
