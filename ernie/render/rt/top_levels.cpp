// SPDX-License-Identifier: BSD-3-Clause
/// @file top_levels.cpp
/// @brief Writing every world's instance records on the device and building the top levels over
/// them.
#include "ernie/render/rt/top_levels.h"

#include "ernie/shaders/rt/write_instances.h"
#include "ernie/util/math.h"
#include "ernie/util/profiler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <vector>

namespace ernie {
namespace {

using shaders::WriteParameters;
static_assert(sizeof(shaders::Instance) == sizeof(AccelerationInstance));
static_assert(shaders::kNoRow == InstanceTable::kNoRow);

std::vector<std::uint32_t> Compile(const ShaderCompiler& compiler) {
    std::vector<ShaderModule> modules(TransformGraph::WorldModules().begin(),
                                      TransformGraph::WorldModules().end());
    modules.push_back(shaders::kWriteInstances);
    return compiler.Spirv(modules, "main");
}

/// @return The capacity for `count` instances: doubling, and at least 4.
std::uint32_t RoundUpCapacity(std::uint32_t count) {
    return std::max<std::uint32_t>(4, std::bit_ceil(count));
}

}  // namespace

TopLevels::TopLevels(DeviceContext& context, Arenas& arenas, const ShaderCompiler& compiler)
    : context_(context), writer_(context, Compile(compiler), "write instances",
                                 std::uint32_t{sizeof(WriteParameters)}),
      pool_(context, "top levels"), uploader_(context, context.Main()),
      tops_(std::make_unique<DeviceColumn>(
          arenas, "top levels", DType::UInt32,
          std::vector<std::int64_t>{sizeof(shaders::TopLevel) / sizeof(std::uint32_t)}, 0)) {}

graph::Acceleration TopLevels::AddBuild(graph::Graph& graph, const TransformGraph& transforms,
                                        const InstanceTable& instances, const BottomLevels& bottoms,
                                        bool rebuild) {
    ERNIE_ZONE("Prepare the top levels");
    // Each world's top level, then the shared prims' own, if any.
    std::vector<InstanceTable::WorldRecords> tops(instances.Worlds().begin(),
                                                  instances.Worlds().end());
    worlds_ = static_cast<std::uint32_t>(tops.size());
    if (const std::optional<InstanceTable::WorldRecords> shared = instances.Shared()) {
        tops.push_back(*shared);
    }
    EnsureCapacity(tops, instances.Capacities().instancesPerWorld, worlds_);
    if (uploader_.Staged()) {
        std::array buffers{graph.Import(*tops_)};
        graph::AddUploadPass(graph, "Upload top levels", uploader_, buffers);
    }
    if (!rebuild) {
        return graph.Import(pool_);
    }
    graph::Buffer records =
        graph.Create(std::max<std::uint64_t>(instances.Records(), 1) * sizeof(AccelerationInstance),
                     "instance records");
    if (instances.Records() > 0) {
        AddWritePass(graph, transforms, instances, bottoms, records);
    }
    return AddBuildPass(graph, bottoms, tops, records);
}

void TopLevels::EnsureCapacity(std::span<const InstanceTable::WorldRecords> tops,
                               std::uint32_t declared, std::size_t worlds) {
    // Frames in flight keep the old ones: destroying them retires them.
    structures_.resize(tops.size());
    hostTops_.resize(tops.size());
    bool changed = false;
    for (std::size_t index = 0; index < tops.size(); ++index) {
        std::unique_ptr<AccelerationStructure>& top = structures_[index];
        const std::uint32_t count = tops[index].count;
        const std::uint32_t capacity = top ? top->Sizes().primitives[0] : 0;
        // The shared top level, after the worlds', grows with its prims.
        const std::uint32_t fixed = index < worlds ? declared : 0;
        const bool resize = fixed > 0 ? capacity != fixed
                                      : count > capacity || (capacity > 4 && count < capacity / 4);
        if (!top || resize) {
            top = std::make_unique<AccelerationStructure>(
                pool_, Sizes(context_, fixed > 0 ? fixed : RoundUpCapacity(count), kRebuilt),
                index < worlds ? "world " + std::to_string(index) : std::string("shared"));
        }
        const shaders::TopLevel next{
            .address = top->Address(), .first = tops[index].first, .count = count};
        shaders::TopLevel& last = hostTops_[index];
        if (last.address != next.address || last.first != next.first || last.count != count) {
            last = next;
            changed = true;
        }
    }
    if (changed) {
        tops_->Resize(hostTops_.size(), context_.Main());
        uploader_.Write(tops_->Storage(), tops_->Range().offset,
                        std::span<const shaders::TopLevel>(hostTops_));
    }
}

void TopLevels::AddWritePass(graph::Graph& graph, const TransformGraph& transforms,
                             const InstanceTable& instances, const BottomLevels& bottoms,
                             graph::Buffer& records) const {
    // In `WriteParameters`' order.
    const std::array reads{graph.Import(transforms.Worlds()),    graph.Import(instances.Nodes()),
                           graph.Import(instances.Geometries()), graph.Import(instances.Shapes()),
                           graph.Import(bottoms.Addresses()),    graph.Import(instances.Sources())};
    const std::uint32_t count = instances.Records();
    graph.Add(
        "Write instances", graph::Kind::Compute,
        [&](graph::PassBuilder& pass) {
            for (const graph::Buffer& read : reads) {
                pass.ReadAddress(read);
            }
            records = pass.WriteAddress(records, graph::Use::StorageWrite);
        },
        [this, reads, records, count,
         placeholder = bottoms.Placeholder()](graph::PassContext& pass) {
            const auto address = [&](std::size_t read) { return pass.Get(reads[read]).Address(); };
            writer_.Dispatch(pass.commands, {DivideRoundUp(count, 64u), 1, 1},
                             WriteParameters{{address(0)},
                                             {address(1)},
                                             {address(2)},
                                             {address(3)},
                                             {address(4)},
                                             {address(5)},
                                             pass.Get(records).BufferRef<shaders::Instance>(),
                                             placeholder,
                                             count});
        });
}

graph::Acceleration TopLevels::AddBuildPass(graph::Graph& graph, const BottomLevels& bottoms,
                                            std::span<const InstanceTable::WorldRecords> worlds,
                                            graph::Buffer records) const {
    graph::Acceleration tops = graph.Import(pool_);
    const graph::Acceleration levels = graph.Import(bottoms.Pool());
    std::vector<TopBuild> builds;
    bool written = false;
    for (std::size_t world = 0; world < worlds.size(); ++world) {
        builds.push_back({.target = structures_[world].get(), .count = worlds[world].count});
        written = written || worlds[world].count > 0;
    }
    graph::Buffer scratch = graph.Create(ScratchBytes({}, builds), "top-level scratch");
    graph.Add(
        "Build the top levels", graph::Kind::Build,
        [&](graph::PassBuilder& pass) {
            pass.Read(levels, graph::Use::BuildInput);
            if (written) {
                pass.Read(records, graph::Use::BuildInput);
            }
            scratch = pass.Write(scratch, graph::Use::BuildScratch);
            tops = pass.Build(tops);
        },
        [builds = std::move(builds), worlds = std::vector(worlds.begin(), worlds.end()), records,
         scratch, written](graph::PassContext& pass) mutable {
            if (written) {
                const graph::BufferSlice slice = pass.Get(records);
                for (std::size_t world = 0; world < builds.size(); ++world) {
                    builds[world].instances = {slice.buffer,
                                               slice.offset + worlds[world].first *
                                                                  sizeof(AccelerationInstance)};
                }
            }
            const graph::BufferSlice space = pass.Get(scratch);
            RecordBuilds(pass.commands, {}, builds, {space.buffer, space.offset});
        });
    return tops;
}

}  // namespace ernie
