// SPDX-License-Identifier: BSD-3-Clause
/// @file camera_pass.h
/// @brief A pinhole camera traced through the scene: a shaded colour image to look at, and each
/// pixel's distance, normal and instance in buffers kernels, CUDA or PyTorch read in place.
///
///     CameraOutputs outputs{.colour = graph.Import(image, "colour"), .colourIndex = index};
///     camera.Add(graph, {.cameraToWorld = orbit.Transform(), .size = {640, 480}}, outputs);
#pragma once

#include "ernie/device/bindless.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/graph/graph.h"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include <cstdint>
#include <optional>

namespace ernie {

/// @brief A pinhole camera in the scene's frame, in double: the pass makes it relative to the
/// render origin before it rounds to float.
struct PinholeView {
    /// Camera to world; the camera looks down -z with +y up, as `glm::lookAtRH`'s inverse.
    glm::dmat4 cameraToWorld{1.0};
    /// Radians.
    double verticalFov = 0.785398;
    /// Pixels.
    glm::uvec2 size{1};
};

/// @brief What a camera pass writes; each left out is not written. Each written one is replaced by
/// its version after the pass.
struct CameraOutputs {
    /// A storage image of the view's size, and its slot in the pass's heap.
    std::optional<graph::Image> colour;
    BindlessIndex colourIndex;
    /// A float a pixel, row by row: the hit distance in metres, infinity where nothing is hit.
    std::optional<graph::Buffer> distances;
    /// Three floats a pixel: the world normal, zero where nothing is hit.
    std::optional<graph::Buffer> normals;
    /// A uint32 a pixel: the instance row hit, or `kNoInstance`.
    std::optional<graph::Buffer> instances;
};

/// @brief Traces a pinhole camera through the scene the graph's `TraceScene` describes, one ray a
/// pixel, shading what it hits with a headlight.
class CameraPass {
  public:
    static constexpr std::uint32_t kNoInstance = 0xffffffff;

    CameraPass(DeviceContext& context, const ShaderCompiler& compiler, const BindlessHeap& heap);

    /// @brief Adds the pass, after the ray tracer's builds.
    /// @throws std::invalid_argument when the graph has no `TraceScene`.
    void Add(graph::Graph& graph, const PinholeView& view, CameraOutputs& outputs) const;

  private:
    ComputeKernel kernel_;
};

}  // namespace ernie
