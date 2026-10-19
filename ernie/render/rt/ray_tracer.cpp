// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.cpp
/// @brief Keeping the acceleration structures up to date each frame.
#include "ernie/render/rt/ray_tracer.h"

#include "ernie/shaders/device/buffer_ref.h"
#include "ernie/shaders/rt/frame_count.h"
#include "ernie/shaders/rt/ray_tracer.h"
#include "ernie/shaders/scene/geometry.h"
#include "ernie/shaders/scene/shapes.h"
#include "ernie/util/profiler.h"

#include <array>

namespace ernie {

namespace {

std::vector<std::uint32_t> CompileCounter(const ShaderCompiler& compiler) {
    const std::array modules{shaders::kBufferRef, shaders::kFrameCount};
    return compiler.Spirv(modules, "main");
}

}  // namespace

RayTracer::RayTracer(DeviceContext& context, Arenas& arenas, const ShaderCompiler& compiler,
                     const GeometryPool& geometry)
    : geometry_(geometry), bottoms_(context, arenas, geometry), tops_(context, arenas, compiler),
      counter_(context, CompileCounter(compiler), "count the frame",
               std::uint32_t{sizeof(shaders::BufferRef<std::uint32_t>)}),
      frame_(std::make_unique<DeviceColumn>(arenas, "frame number", DType::UInt32,
                                            std::vector<std::int64_t>{}, 1)) {}

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
    graph::Buffer frame = graph.Import(*frame_);
    if (!counting_) {
        // From zero, so the same steps key the same random numbers.
        graph.Add(
            "Start counting frames", graph::Kind::Copy,
            [&](graph::PassBuilder& pass) {
                pass.OneShot();
                frame = pass.Write(frame, graph::Use::CopyDst);
            },
            [frame](graph::PassContext& pass) {
                const graph::BufferSlice slice = pass.Get(frame);
                const std::uint32_t zero = 0;
                pass.commands.Update(*slice.buffer, slice.offset,
                                     std::as_bytes(std::span(&zero, 1)));
            });
        counting_ = true;
    }
    // Recorded once and run every frame, replays too.
    graph.Add(
        "Count the frame", graph::Kind::Compute,
        [&](graph::PassBuilder& pass) { frame = pass.WriteAddress(frame); },
        [this, frame](graph::PassContext& pass) {
            counter_.Dispatch(pass.commands, {1, 1, 1}, pass.Get(frame).BufferRef<std::uint32_t>());
        });
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
                   .frame = frame,
                   .origin = transforms.Origin()});
}

}  // namespace ernie
