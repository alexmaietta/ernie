// SPDX-License-Identifier: BSD-3-Clause
/// @file camera_pass.cpp
/// @brief Tracing pinhole cameras as a frame-graph pass.
#include "ernie/render/rt/camera_pass.h"

#include "ernie/render/rt/ray_tracer.h"
#include "ernie/shaders/device/bindless.h"
#include "ernie/shaders/sensors/camera.h"
#include "ernie/shaders/spectral/spectrum.h"
#include "ernie/util/math.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

using shaders::CameraParameters;
using shaders::View;

std::vector<std::uint32_t> Compile(const ShaderCompiler& compiler) {
    std::vector<ShaderModule> modules(RayTracer::Modules().begin(), RayTracer::Modules().end());
    modules.insert(modules.end(), {shaders::kBindless, shaders::kSpectrum, shaders::kCamera});
    return compiler.Spirv(modules, "main");
}

/// @return A view's record, its output addresses left for the pass to fill in once the graph has
/// placed them.
/// @return A kernel's handle of an output, or a null one when it is left out.
template <typename T>
shaders::BufferRef<T> BufferRefOrNull(const graph::PassContext& pass,
                                      const std::optional<graph::Buffer>& buffer) {
    return buffer ? pass.Get(*buffer).BufferRef<T>() : shaders::BufferRef<T>{};
}

View ToView(const PinholeView& view, const CameraOutputs& outputs, const glm::dvec3& origin) {
    // Relative to the origin in double, then rounded: values near the camera stay small.
    const glm::dmat4 relative = glm::translate(glm::dmat4(1.0), -origin) * view.cameraToWorld;
    const double tangent = std::tan(0.5 * view.verticalFov);
    const double aspect = static_cast<double>(view.size.x) / static_cast<double>(view.size.y);
    const std::array<glm::vec4, 3> rows = ToAffineRows(relative);
    return {.rows = {rows[0], rows[1], rows[2]},
            .halfExtent = glm::vec2(tangent * aspect, tangent),
            .size = view.size,
            .background = outputs.background,
            .colour = outputs.colour ? outputs.colourIndex.value : shaders::kNoImage};
}

}  // namespace

CameraPass::CameraPass(DeviceContext& context, const ShaderCompiler& compiler,
                       const BindlessHeap& heap)
    : kernel_(context, Compile(compiler), "camera", sizeof(CameraParameters), &heap) {}

void CameraPass::Add(graph::Graph& graph, const PinholeView& view, CameraOutputs& outputs) const {
    CameraView one{view, outputs};
    Add(graph, std::span(&one, 1));
    outputs = one.outputs;
}

void CameraPass::Add(graph::Graph& graph, std::span<CameraView> views) const {
    const TraceScene* scene = graph.Board().Find<TraceScene>();
    if (scene == nullptr) {
        throw std::invalid_argument("A camera pass needs the ray tracer's TraceScene");
    }
    if (views.empty()) {
        return;
    }
    std::vector<View> records;
    glm::uvec2 largest{0};
    for (const CameraView& view : views) {
        records.push_back(ToView(view.view, view.outputs, scene->origin));
        largest = glm::max(largest, view.view.size);
    }
    graph::Buffer staged = graph.Create(records.size() * sizeof(View), "camera views");
    // For the recording to look up: any version of a resource names it.
    std::vector<CameraOutputs> outputs;
    for (const CameraView& view : views) {
        outputs.push_back(view.outputs);
    }
    graph.Add(
        "Camera", graph::Kind::Compute,
        [&](graph::PassBuilder& pass) {
            scene->Read(pass);
            // Written by the pass itself before its dispatch reads it.
            staged = pass.WriteAddress(staged, graph::Use::StorageWrite);
            for (CameraView& view : views) {
                CameraOutputs& output = view.outputs;
                if (output.colour) {
                    output.colour = pass.Write(*output.colour, graph::Use::StorageWrite);
                }
                for (std::optional<graph::Buffer>* buffer :
                     {&output.distances, &output.normals, &output.instances, &output.albedos,
                      &output.ldr}) {
                    if (*buffer) {
                        *buffer = pass.WriteAddress(**buffer, graph::Use::StorageWrite);
                    }
                }
            }
        },
        [this, staged, largest, records = std::move(records), outputs = std::move(outputs),
         parameters =
             CameraParameters{.tracer = scene->parameters, .colours = scene->colourBuffer}](
            graph::PassContext& pass) mutable {
            for (std::size_t i = 0; i < records.size(); ++i) {
                records[i].distances = BufferRefOrNull<float>(pass, outputs[i].distances);
                records[i].normals = BufferRefOrNull<float>(pass, outputs[i].normals);
                records[i].instances = BufferRefOrNull<shaders::uint>(pass, outputs[i].instances);
                records[i].albedos = BufferRefOrNull<float>(pass, outputs[i].albedos);
                records[i].ldr = BufferRefOrNull<shaders::uint>(pass, outputs[i].ldr);
            }
            // The records hold the outputs' addresses, known only now, so the pass writes them
            // itself before it reads them.
            const graph::BufferSlice slice = pass.Get(staged);
            pass.commands.Update(*slice.buffer, slice.offset, std::as_bytes(std::span(records)));
            parameters.views = slice.BufferRef<View>();
            parameters.count = static_cast<std::uint32_t>(records.size());
            kernel_.Dispatch(
                pass.commands,
                {DivideRoundUp(largest.x, 8u), DivideRoundUp(largest.y, 8u), parameters.count},
                parameters);
        });
}

}  // namespace ernie
