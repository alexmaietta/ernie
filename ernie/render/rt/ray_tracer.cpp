// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.cpp
/// @brief Keeping the acceleration structures up to date each frame.
#include "ernie/render/rt/ray_tracer.h"

#include "ernie/shaders/device/buffer_ref.h"
#include "ernie/shaders/rt/ray_tracer.h"
#include "ernie/shaders/scene/geometry.h"
#include "ernie/shaders/scene/shapes.h"
#include "ernie/util/profiler.h"

#include <array>

namespace ernie {

RayTracer::RayTracer(DeviceContext& context, Arenas& arenas, const ShaderCompiler& compiler,
                     const GeometryPool& geometry)
    : geometry_(geometry), bottoms_(context, arenas, geometry), tops_(context, arenas, compiler) {}

std::span<const ShaderModule> RayTracer::Modules() {
    static const std::array modules{shaders::kBufferRef, shaders::kGeometry, shaders::kShapes,
                                    shaders::kRayTracer};
    return modules;
}

void RayTracer::AddBuild(graph::Graph& graph, const GeometryPool::Changes& changes,
                         const TransformGraph& transforms, const InstanceTable& instances,
                         bool rebuildTops) {
    ERNIE_ZONE("Update the ray tracer");
    const graph::Graph::ScopeGuard scope = graph.Scope("Ray tracer");
    bottoms_.Apply(changes);
    bottoms_.AddBuilds(graph);
    const graph::Acceleration top =
        tops_.AddBuild(graph, transforms, instances, bottoms_, rebuildTops);
    const DeviceColumn& records = geometry_.Records();
    graph.Board().Put(
        TraceScene{.parameters = {.tops = ToBufferRef<shaders::TopLevel>(tops_.Tops()),
                                  .sources = ToBufferRef<shaders::uint>(instances.Sources()),
                                  .records = ToBufferRef<shaders::GeometryRecord>(records),
                                  .worlds = tops_.Worlds(),
                                  .shared = tops_.SharedTop().value_or(shaders::kNoSharedTop)},
                   .top = top,
                   .tables = {graph.Import(tops_.Tops()), graph.Import(instances.Sources()),
                              graph.Import(records)},
                   .instances = instances.Count(),
                   .worlds = tops_.Worlds(),
                   .colours = graph.Import(instances.Colours()),
                   .prims = graph.Import(instances.Nodes()),
                   .transforms = graph.Import(transforms.Worlds()),
                   .origin = transforms.Origin()});
}

}  // namespace ernie
