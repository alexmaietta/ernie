// SPDX-License-Identifier: BSD-3-Clause
/// @file camera_pass.cpp
/// @brief Tracing a pinhole camera as a frame-graph pass.
#include "ernie/render/camera_pass.h"

#include "ernie/render/rt/ray_tracer.h"
#include "ernie/shaders/device/bindless.h"
#include "ernie/shaders/sensors/camera.h"

#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

constexpr std::uint32_t kNoImage = 0xffffffff;

struct CameraParameters {
    TracerParameters tracer;
    glm::vec4 rows[3];
    glm::vec2 halfExtent;
    glm::uvec2 size;
    std::uint32_t colour, pad;
    VkDeviceAddress distances, normals, instances;
};
static_assert(sizeof(CameraParameters) == 112);

std::vector<std::uint32_t> Compile(const ShaderCompiler& compiler) {
    std::vector<ShaderModule> modules(RayTracer::Modules().begin(), RayTracer::Modules().end());
    modules.push_back(shaders::kBindless);
    modules.push_back(shaders::kCamera);
    return compiler.Spirv(modules, "main");
}

}  // namespace

CameraPass::CameraPass(DeviceContext& context, const ShaderCompiler& compiler,
                       const BindlessHeap& heap)
    : kernel_(context, Compile(compiler), "camera", sizeof(CameraParameters), &heap) {}

void CameraPass::Add(graph::Graph& graph, const PinholeView& view, CameraOutputs& outputs) const {
    const TraceScene* scene = graph.Board().Find<TraceScene>();
    if (scene == nullptr) {
        throw std::invalid_argument("A camera pass needs the ray tracer's TraceScene");
    }
    // Relative to the origin in double, then rounded: values near the camera stay small.
    const glm::dmat4 relative =
        glm::translate(glm::dmat4(1.0), -scene->origin) * view.cameraToWorld;
    CameraParameters parameters{.tracer = scene->parameters,
                                .size = view.size,
                                .colour = outputs.colour ? outputs.colourIndex.value : kNoImage};
    for (int r = 0; r < 3; ++r) {
        parameters.rows[r] = glm::vec4(glm::row(relative, r));
    }
    const double tangent = std::tan(0.5 * view.verticalFov);
    const double aspect = static_cast<double>(view.size.x) / static_cast<double>(view.size.y);
    parameters.halfExtent = glm::vec2(tangent * aspect, tangent);

    graph.Add(
        "Camera", graph::Kind::Compute,
        [&](graph::PassBuilder& pass) {
            scene->Read(pass);
            if (outputs.colour) {
                outputs.colour = pass.Write(*outputs.colour, graph::Use::StorageWrite);
            }
            for (std::optional<graph::Buffer>* buffer :
                 {&outputs.distances, &outputs.normals, &outputs.instances}) {
                if (*buffer) {
                    *buffer = pass.WriteAddress(**buffer, graph::Use::StorageWrite);
                }
            }
        },
        [this, parameters, outputs](graph::PassContext& pass) {
            CameraParameters recorded = parameters;
            const auto address = [&](const std::optional<graph::Buffer>& buffer) {
                return buffer ? pass.Get(*buffer).Address() : VkDeviceAddress{0};
            };
            recorded.distances = address(outputs.distances);
            recorded.normals = address(outputs.normals);
            recorded.instances = address(outputs.instances);
            kernel_.Dispatch(pass.commands,
                             {(parameters.size.x + 7) / 8, (parameters.size.y + 7) / 8, 1},
                             recorded);
        });
}

}  // namespace ernie
