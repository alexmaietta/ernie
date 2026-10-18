// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.cpp
/// @brief Keeping the acceleration structures up to date each frame.
#include "ernie/render/rt/ray_tracer.h"

#include "ernie/shaders/device/ref.h"
#include "ernie/shaders/rt/ray_tracer.h"
#include "ernie/shaders/scene/geometry.h"
#include "ernie/shaders/scene/shapes.h"
#include "ernie/util/profiler.h"

#include <array>

namespace ernie {

RayTracer::RayTracer(DeviceContext& context, Arenas& arenas, const ShaderCompiler& compiler,
                     const GeometryPool& geometry)
    : geometry_(geometry), bottoms_(context, arenas, geometry), tops_(context, compiler) {}

std::span<const ShaderModule> RayTracer::Modules() {
    static const std::array modules{shaders::kRef, shaders::kGeometry, shaders::kShapes,
                                    shaders::kRayTracer};
    return modules;
}

void RayTracer::AddBuild(graph::Graph& graph, const GeometryPool::Changes& changes,
                         const TransformGraph& transforms, const InstanceTable& instances) {
    ERNIE_ZONE("Update the ray tracer");
    const graph::Graph::ScopeGuard scope = graph.Scope("Ray tracer");
    bottoms_.Apply(changes);
    bottoms_.AddBuilds(graph);
    const graph::Acceleration top = tops_.AddBuild(graph, transforms, instances, bottoms_);
    const DeviceColumn& records = geometry_.Records();
    const DeviceColumn& colours = instances.Colours();
    graph.Board().Put(TraceScene{.parameters = {.scene = tops_.Structure()->Address(),
                                                .records = ToRef<shaders::GeometryRecord>(records)},
                                 .top = top,
                                 .records = graph.Import(records),
                                 .instances = instances.Count(),
                                 .colours = graph.Import(colours),
                                 .colourRef = ToRef<float>(colours),
                                 .origin = transforms.Origin()});
}

}  // namespace ernie
