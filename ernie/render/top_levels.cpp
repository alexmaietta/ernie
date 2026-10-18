// SPDX-License-Identifier: BSD-3-Clause
/// @file top_levels.cpp
/// @brief Writing instance records on the device and building the top level over them.
#include "ernie/render/top_levels.h"

#include "ernie/util/profiler.h"

#include <algorithm>
#include <bit>
#include <vector>

namespace ernie {
namespace {

const ShaderModule kWriteInstances{"write_instances", R"(
import affine;
import world;
/// VkAccelerationStructureInstanceKHR.
struct Instance { float4 rows[3]; uint indexAndMask; uint offsetAndFlags; uint64_t bottom; };
struct Parameters {
    float4* worlds; uint* nodes; uint* geometries; float4* shapes; uint64_t* bottoms;
    Instance* instances; uint count;
};
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    if (row >= parameters.count) {
        return;
    }
    uint geometry = parameters.geometries[row];
    Affine m = Mul(LoadWorld(parameters.worlds, parameters.nodes[row]),
                   Load(parameters.shapes, row));
    uint64_t bottom = parameters.bottoms[geometry];
    Instance instance;
    instance.rows = m.rows;
    // An instance whose bottom level is not built yet is switched off.
    instance.indexAndMask = (bottom == 0 ? 0u : 0xff000000u) | (geometry & 0xffffffu);
    instance.offsetAndFlags = 0;
    instance.bottom = bottom;
    parameters.instances[row] = instance;
}
)"};

struct WriteParameters {
    VkDeviceAddress worlds, nodes, geometries, shapes, bottoms, instances;
    std::uint32_t count, pad;
};

std::vector<std::uint32_t> Compile(const ShaderCompiler& compiler) {
    std::vector<ShaderModule> modules(TransformGraph::WorldModules().begin(),
                                      TransformGraph::WorldModules().end());
    modules.push_back(kWriteInstances);
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

void TopLevels::AddBuild(graph::Graph& graph, const TransformGraph& transforms,
                         const InstanceTable& instances, const BottomLevels& bottoms) {
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
        const auto import = [&](const DeviceColumn& column) {
            return graph.Import(column, column.Name());
        };
        const graph::Buffer worlds = import(transforms.Worlds());
        const graph::Buffer nodes = import(instances.Nodes());
        const graph::Buffer geometries = import(instances.Geometries());
        const graph::Buffer shapes = import(instances.Shapes());
        const graph::Buffer addresses = import(bottoms.Addresses());
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
                    pass.commands, {(count + 63) / 64, 1, 1},
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
    graph.Board().Put(SceneStructure{.top = top, .address = top_->Address(), .instances = count});
}

}  // namespace ernie
