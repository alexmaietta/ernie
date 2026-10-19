// SPDX-License-Identifier: BSD-3-Clause
/// @file camera_pass.cpp
/// @brief Tracing pinhole cameras as a frame-graph pass.
#include "ernie/render/rt/camera_pass.h"

#include "ernie/render/rt/ray_tracer.h"
#include "ernie/render/transform_graph.h"
#include "ernie/shaders/device/bindless.h"
#include "ernie/shaders/sensors/camera.h"
#include "ernie/shaders/sensors/lens.h"
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

/// @return A kernel's handle of an output, or a null one when it is left out.
template <typename T>
shaders::BufferRef<T> BufferRefOrNull(const graph::PassContext& pass,
                                      const std::optional<graph::Buffer>& buffer) {
    return buffer ? pass.Get(*buffer).BufferRef<T>() : shaders::BufferRef<T>{};
}

/// @return A view's record, its output addresses left for the pass to fill in once the graph has
/// placed them.
View ToView(const CameraView& view, const glm::dvec3& origin, std::uint32_t first) {
    // Relative to the origin in double, then rounded: values near the camera stay small.
    const glm::dmat4 relative = glm::translate(glm::dmat4(1.0), -origin) * view.cameraToWorld;
    const std::array<glm::vec4, 3> rows = ToAffineRows(relative);
    return {.rows = {rows[0], rows[1], rows[2]},
            .lens = view.lens,
            .size = view.size,
            .colour = shaders::kNoImage,
            .node = view.node,
            .first = first,
            .world = view.world};
}

}  // namespace

Lens MakePinhole(double verticalFov, glm::uvec2 size) {
    const auto focal = static_cast<float>(0.5 * size.y / std::tan(0.5 * verticalFov));
    return {
        .kind = shaders::kPinhole, .focal = glm::vec2(focal), .principal = 0.5f * glm::vec2(size)};
}

CameraPass::CameraPass(DeviceContext& context, const ShaderCompiler& compiler,
                       const BindlessHeap& heap)
    : kernel_(context, compiler.Spirv(Modules(), "main"), "camera", sizeof(CameraParameters),
              &heap) {}

std::span<const ShaderModule> CameraPass::Modules() {
    static const std::vector<ShaderModule> modules = [] {
        std::vector<ShaderModule> list(RayTracer::Modules().begin(), RayTracer::Modules().end());
        // The ray tracer's modules bring the buffer handles already.
        for (const ShaderModule& module : TransformGraph::WorldModules().subspan(1)) {
            list.push_back(module);
        }
        list.insert(list.end(),
                    {shaders::kBindless, shaders::kSpectrum, shaders::kLens, shaders::kCamera});
        return list;
    }();
    return modules;
}

void CameraPass::Add(graph::Graph& graph, const CameraView& view, CameraOutputs& outputs) const {
    Add(graph, std::span(&view, 1), outputs);
}

void CameraPass::Add(graph::Graph& graph, std::span<const CameraView> views,
                     CameraOutputs& outputs) const {
    const TraceScene* scene = graph.Board().Find<TraceScene>();
    if (scene == nullptr) {
        throw std::invalid_argument("A camera pass needs the ray tracer's TraceScene");
    }
    if (views.empty()) {
        return;
    }
    std::vector<View> records;
    glm::uvec2 largest{0};
    std::uint64_t pixels = 0;
    for (const CameraView& view : views) {
        if (view.world >= scene->worlds) {
            throw std::invalid_argument("A view of world " + std::to_string(view.world) +
                                        ", of a scene of " + std::to_string(scene->worlds));
        }
        records.push_back(ToView(view, scene->origin, static_cast<std::uint32_t>(pixels)));
        largest = glm::max(largest, view.size);
        pixels += std::uint64_t{view.size.x} * view.size.y;
    }
    if (outputs.colour) {
        records.front().colour = outputs.colourIndex.value;
        records.front().background = outputs.background;
    }
    graph::Buffer staged = graph.Create(records.size() * sizeof(View), "camera views");
    graph.Add(
        "Camera", graph::Kind::Compute,
        [&](graph::PassBuilder& pass) {
            scene->Read(pass);
            // Written by the pass itself before its dispatch reads it.
            staged = pass.WriteAddress(staged, graph::Use::StorageWrite);
            if (outputs.colour) {
                outputs.colour = pass.Write(*outputs.colour, graph::Use::StorageWrite);
            }
            for (std::optional<graph::Buffer>* buffer :
                 {&outputs.distances, &outputs.normals, &outputs.instances, &outputs.prims,
                  &outputs.albedos, &outputs.ldr}) {
                if (*buffer) {
                    *buffer = pass.WriteAddress(**buffer, graph::Use::StorageWrite);
                }
            }
        },
        [this, staged, largest, records = std::move(records), outputs,
         scene = *scene](graph::PassContext& pass) mutable {
            View written{.distances = BufferRefOrNull<float>(pass, outputs.distances),
                         .normals = BufferRefOrNull<float>(pass, outputs.normals),
                         .instances = BufferRefOrNull<shaders::uint>(pass, outputs.instances),
                         .prims = BufferRefOrNull<shaders::uint>(pass, outputs.prims),
                         .albedos = BufferRefOrNull<float>(pass, outputs.albedos),
                         .ldr = BufferRefOrNull<shaders::uint>(pass, outputs.ldr)};
            for (View& record : records) {
                record.distances = written.distances;
                record.normals = written.normals;
                record.instances = written.instances;
                record.prims = written.prims;
                record.albedos = written.albedos;
                record.ldr = written.ldr;
            }
            // The records hold the outputs' addresses, known only now, so the pass writes them
            // itself before it reads them.
            const graph::BufferSlice slice = pass.Get(staged);
            pass.commands.Update(*slice.buffer, slice.offset, std::as_bytes(std::span(records)));
            const CameraParameters parameters{
                .tracer = scene.parameters,
                .views = slice.BufferRef<View>(),
                .worlds = pass.Get(scene.transforms).BufferRef<glm::vec4>(),
                .colours = pass.Get(scene.colours).BufferRef<float>(),
                .instancePrims = pass.Get(scene.prims).BufferRef<shaders::uint>(),
                .count = static_cast<std::uint32_t>(records.size())};
            kernel_.Dispatch(
                pass.commands,
                {DivideRoundUp(largest.x, 8u), DivideRoundUp(largest.y, 8u), parameters.count},
                parameters);
        });
}

}  // namespace ernie
