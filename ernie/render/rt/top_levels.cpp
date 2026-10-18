// SPDX-License-Identifier: BSD-3-Clause
/// @file top_levels.cpp
/// @brief Writing instance records on the device and building the top level over them.
#include "ernie/render/rt/top_levels.h"

#include "ernie/shaders/rt/write_instances.h"
#include "ernie/util/math.h"
#include "ernie/util/profiler.h"

#include <algorithm>
#include <bit>
#include <vector>

namespace ernie {
namespace {

struct WriteParameters {
    VkDeviceAddress worlds, nodes, geometries, shapes, bottoms, instances;
    std::uint32_t count, pad;
};

std::vector<std::uint32_t> Compile(const ShaderCompiler& compiler) {
    std::vector<ShaderModule> modules(TransformGraph::WorldModules().begin(),
                                      TransformGraph::WorldModules().end());
    modules.push_back(shaders::kWriteInstances);
    return compiler.Spirv(modules, "main");
}

/// @return The capacity for `count` instances: doubling, and at least 64.
std::uint32_t Capacity(std::uint32_t count) {
    return std::max<std::uint32_t>(64, std::bit_ceil(count));
}

}  // namespace

TopLevels::TopLevels(DeviceContext& context, const ShaderCompiler& compiler)
    : context_(context), writer_(context, Compile(compiler), "write instances",
                                 std::uint32_t{sizeof(WriteParameters)}),
      pool_(context, "top levels") {}

graph::Acceleration TopLevels::AddBuild(graph::Graph& graph, const TransformGraph& transforms,
                                        const InstanceTable& instances,
                                        const BottomLevels& bottoms) {
    ERNIE_ZONE("Prepare the top level");
    const std::uint32_t count = instances.Count();
    const std::uint32_t capacity = top_ ? top_->Sizes().primitives[0] : 0;
    if (!top_ || count > capacity || (capacity > 64 && count < capacity / 4)) {
        // Frames in flight keep the old one: destroying it retires it.
        top_ = std::make_unique<AccelerationStructure>(
            pool_, Sizes(context_, Capacity(count), kRebuilt), "scene");
    }
    graph::Acceleration top = graph.Import(*top_);
    const graph::Acceleration levels = graph.Import(bottoms.Pool());
    graph::Buffer records = graph.Create(
        std::max<std::uint64_t>(count, 1) * sizeof(AccelerationInstance), "instance records");
    const TopBuild build{.target = top_.get(), .count = count};
    graph::Buffer scratch =
        graph.Create(ScratchBytes({}, std::span(&build, 1)), "top-level scratch");

    if (count > 0) {
        const graph::Buffer worlds = graph.Import(transforms.Worlds());
        const graph::Buffer nodes = graph.Import(instances.Nodes());
        const graph::Buffer geometries = graph.Import(instances.Geometries());
        const graph::Buffer shapes = graph.Import(instances.Shapes());
        const graph::Buffer addresses = graph.Import(bottoms.Addresses());
        graph.Add(
            "Write instances", graph::Kind::Compute,
            [&](graph::PassBuilder& pass) {
                for (const graph::Buffer& read : {worlds, nodes, geometries, shapes, addresses}) {
                    pass.ReadAddress(read);
                }
                records = pass.WriteAddress(records, graph::Use::StorageWrite);
            },
            [this, worlds, nodes, geometries, shapes, addresses, records,
             count](graph::PassContext& pass) {
                writer_.Dispatch(
                    pass.commands, {DivideRoundUp(count, 64u), 1, 1},
                    WriteParameters{pass.Get(worlds).Address(), pass.Get(nodes).Address(),
                                    pass.Get(geometries).Address(), pass.Get(shapes).Address(),
                                    pass.Get(addresses).Address(), pass.Get(records).Address(),
                                    count, 0});
            });
    }
    graph.Add(
        "Build the top level", graph::Kind::Build,
        [&](graph::PassBuilder& pass) {
            pass.Read(levels, graph::Use::BuildInput);
            if (count > 0) {
                pass.Read(records, graph::Use::BuildInput);
            }
            scratch = pass.Write(scratch, graph::Use::BuildScratch);
            top = pass.Build(top);
        },
        [build, records, scratch, count](graph::PassContext& pass) {
            TopBuild recorded = build;
            if (count > 0) {
                const graph::BufferSlice slice = pass.Get(records);
                recorded.instances = {slice.buffer, slice.offset};
            }
            const graph::BufferSlice space = pass.Get(scratch);
            RecordBuilds(pass.commands, {}, std::span(&recorded, 1), {space.buffer, space.offset});
        });
    return top;
}

}  // namespace ernie
