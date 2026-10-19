// SPDX-License-Identifier: BSD-3-Clause
/// @file instances.cpp
/// @brief Keeping instance rows dense as prims come and go, uploading the rows that changed, and
/// sorting them by world on the device.
#include "ernie/render/instances.h"

#include "ernie/shaders/device/buffer_ref.h"
#include "ernie/shaders/scene/instance_colours.h"
#include "ernie/shaders/scene/world_layout.h"
#include "ernie/util/math.h"
#include "ernie/util/profiler.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ernie {
namespace {

static_assert(shaders::kNoInstanceRow == InstanceTable::kNoRow);
static_assert(shaders::kNoSource == InstanceTable::kNoRow);

/// The most workgroups a layout kernel dispatches; each thread loops over the rest.
constexpr std::uint32_t kLayoutGroups = 1024;

std::vector<std::uint32_t> CompileColourer(const ShaderCompiler& compiler) {
    const std::array modules{shaders::kBufferRef, shaders::kInstanceColours};
    return compiler.Spirv(modules, "main");
}

std::vector<std::uint32_t> CompileLayout(const ShaderCompiler& compiler, std::string_view entry) {
    const std::array modules{shaders::kBufferRef, shaders::kWorldLayout};
    return compiler.Spirv(modules, entry);
}

}  // namespace

InstanceTable::InstanceTable(DeviceContext& context, Arenas& arenas, const Scene& scene,
                             const ShaderCompiler& compiler, WorldCapacities capacities)
    : context_(context), arenas_(arenas), scene_(scene), capacities_(capacities),
      colourer_(context, CompileColourer(compiler), "instance colours",
                std::uint32_t{sizeof(shaders::ColourParameters)}),
      clearer_(context, CompileLayout(compiler, "Clear"), "clear records",
               std::uint32_t{sizeof(shaders::LayoutParameters)}),
      scatterer_(context, CompileLayout(compiler, "Scatter"), "sort records by world",
                 std::uint32_t{sizeof(shaders::LayoutParameters)}),
      sharer_(context, CompileLayout(compiler, "FillShared"), "share records",
              std::uint32_t{sizeof(shaders::LayoutParameters)}),
      uploader_(context, context.Main()) {
    const auto column = [&](std::string_view name, DType type, std::vector<std::int64_t> element) {
        return std::make_unique<DeviceColumn>(arenas, name, type, std::move(element), 0);
    };
    nodes_ = column("instance nodes", DType::UInt32, {});
    geometries_ = column("instance geometries", DType::UInt32, {});
    shapes_ = column("instance shapes", DType::Float32, {3, 4});
    colours_ = column("instance colours", DType::Float32, {3});
    sources_ = column("top-level sources", DType::UInt32, {});
    rowWorlds_ = column("instance worlds", DType::Int32, {});
    cursors_ = column("world cursors", DType::UInt32, {});
    sharedRows_ = column("shared rows", DType::UInt32, {});
}

std::optional<std::uint32_t> InstanceTable::Row(PrimId prim) const {
    if (prim.slot >= rows_.size() || rows_[prim.slot] == kNoRow ||
        !(prims_[rows_[prim.slot]] == prim)) {
        return std::nullopt;
    }
    return rows_[prim.slot];
}

void InstanceTable::Set(std::uint32_t row, const PrimGeometry& placed) {
    prims_[row] = placed.prim;
    hostNodes_[row] = placed.prim.slot;
    hostGeometries_[row] = placed.geometry;
    hostShapes_[row] = ToAffineRows(placed.shape);
    hostColours_[row] = placed.colour;
    relayout_ = relayout_ || hostWorlds_[row] != placed.world;
    hostWorlds_[row] = placed.world;
    rows_[placed.prim.slot] = row;
    dirty_.push_back(row);
}

void InstanceTable::Remove(PrimId prim) {
    const std::optional<std::uint32_t> row = Row(prim);
    if (!row) {
        return;
    }
    // The last row moves into the gap.
    const auto last = static_cast<std::uint32_t>(prims_.size() - 1);
    if (*row != last) {
        ForEachHostColumn([&](auto& column) { column[*row] = column[last]; });
        rows_[prims_[*row].slot] = *row;
        dirty_.push_back(*row);
    }
    rows_[prim.slot] = kNoRow;
    ForEachHostColumn([](auto& column) { column.pop_back(); });
    relayout_ = true;
}

void InstanceTable::LayOutWorlds() {
    std::int32_t highest = 0;
    std::uint32_t shared = 0;
    for (const std::int32_t world : hostWorlds_) {
        highest = std::max(highest, world);
        shared += world < 0 ? 1u : 0u;
    }
    const std::size_t needed =
        std::max<std::size_t>(static_cast<std::size_t>(highest) + 1, reserved_);
    if (capacities_.worlds > 0 && needed > capacities_.worlds) {
        throw std::length_error(std::to_string(needed) + " worlds are more than the " +
                                std::to_string(capacities_.worlds) + " declared");
    }
    const std::size_t count = capacities_.worlds > 0 ? capacities_.worlds : needed;
    // With one world, a top level of their own would only add an empty one to trace.
    const bool ownTop = capacities_.sharedPrims == SharedPrims::SharedTopLevel && count > 1;
    std::vector<std::uint32_t> records(count, ownTop ? 0u : shared);
    for (const std::int32_t world : hostWorlds_) {
        if (world >= 0) {
            ++records[static_cast<std::size_t>(world)];
        }
    }
    const auto largest = std::ranges::max_element(records);
    const std::uint32_t declared = capacities_.instancesPerWorld;
    if (declared > 0 && *largest > declared) {
        throw std::length_error("World " + std::to_string(largest - records.begin()) + " has " +
                                std::to_string(*largest) + " instances, beyond the " +
                                std::to_string(declared) + " declared a world");
    }
    stride_ = declared > 0 ? declared : *largest;
    sharedCount_ = shared;
    worlds_.resize(count);
    for (std::size_t world = 0; world < count; ++world) {
        worlds_[world] = {.first = static_cast<std::uint32_t>(world) * stride_,
                          .count = records[world]};
    }
    shared_.reset();
    if (ownTop && shared > 0) {
        shared_ =
            WorldRecords{.first = static_cast<std::uint32_t>(count) * stride_, .count = shared};
    }
}

void InstanceTable::AddLayoutPasses(graph::Graph& graph, graph::Buffer rowWorlds,
                                    graph::Buffer& sources) const {
    graph::Buffer cursors = graph.Import(*cursors_);
    graph::Buffer shared = graph.Import(*sharedRows_);
    const shaders::LayoutParameters layout{.worlds = ToBufferRef<int>(*rowWorlds_),
                                           .cursors = ToBufferRef<shaders::uint>(*cursors_),
                                           .sharedRows = ToBufferRef<shaders::uint>(*sharedRows_),
                                           .sources = ToBufferRef<shaders::uint>(*sources_),
                                           .rows = Count(),
                                           .worldCount = static_cast<std::uint32_t>(worlds_.size()),
                                           .stride = stride_,
                                           .sharedCount = sharedCount_,
                                           .sharedFirst =
                                               shared_ ? shared_->first : shaders::kNoSource};
    const auto add = [&](const char* name, const ComputeKernel& kernel, std::uint32_t items,
                         const auto& declare) {
        if (items == 0) {
            return;
        }
        const std::uint32_t groups = std::min(DivideRoundUp(items, 64u), kLayoutGroups);
        shaders::LayoutParameters parameters = layout;
        parameters.threads = groups * 64;
        graph.Add(
            name, graph::Kind::Compute,
            [&](graph::PassBuilder& pass) {
                pass.OneShot();
                declare(pass);
            },
            [&kernel, parameters, groups](graph::PassContext& pass) {
                kernel.Dispatch(pass.commands, {groups, 1, 1}, parameters);
            });
    };
    add("Clear records", clearer_, std::max(Records(), layout.worldCount + 1),
        [&](graph::PassBuilder& pass) {
            sources = pass.WriteAddress(sources);
            cursors = pass.WriteAddress(cursors);
        });
    add("Sort records by world", scatterer_, Count(), [&](graph::PassBuilder& pass) {
        pass.ReadAddress(rowWorlds);
        sources = pass.WriteAddress(sources);
        cursors = pass.WriteAddress(cursors);
        shared = pass.WriteAddress(shared);
    });
    const std::uint32_t copies = shared_ ? sharedCount_ : layout.worldCount * sharedCount_;
    add("Share records", sharer_, copies, [&](graph::PassBuilder& pass) {
        pass.ReadAddress(shared);
        sources = pass.WriteAddress(sources);
    });
}

void InstanceTable::Apply(const GeometryPool::Changes& changes) {
    ERNIE_ZONE("Update instances");
    for (const PrimId prim : changes.removed) {
        Remove(prim);
    }
    for (const PrimGeometry& placed : changes.placed) {
        if (placed.prim.slot >= rows_.size()) {
            rows_.resize(placed.prim.slot + 1, kNoRow);
        }
        const std::optional<std::uint32_t> row = Row(placed.prim);
        if (row) {
            Set(*row, placed);
            continue;
        }
        ForEachHostColumn([](auto& column) { column.emplace_back(); });
        relayout_ = true;
        Set(static_cast<std::uint32_t>(prims_.size() - 1), placed);
    }
}

void InstanceTable::AddUploads(graph::Graph& graph) {
    // Rows that came, went or moved move the bound colours' targets too.
    retarget_ = retarget_ || relayout_;
    const std::size_t rows = prims_.size();
    for (DeviceColumn* column :
         {nodes_.get(), geometries_.get(), shapes_.get(), colours_.get(), rowWorlds_.get()}) {
        column->Resize(rows, context_.Main());
    }
    StageRows(uploader_, *nodes_, hostNodes_, dirty_);
    StageRows(uploader_, *geometries_, hostGeometries_, dirty_);
    StageRows(uploader_, *shapes_, hostShapes_, dirty_);
    StageRows(uploader_, *colours_, hostColours_, dirty_);
    StageRows(uploader_, *rowWorlds_, hostWorlds_, dirty_);
    dirty_.clear();
    const bool relayout = relayout_;
    if (relayout) {
        LayOutWorlds();
        sources_->Resize(Records(), context_.Main());
        cursors_->Resize(worlds_.size() + 1, context_.Main());
        sharedRows_->Resize(std::max(sharedCount_, 1u), context_.Main());
        relayout_ = false;
    }
    if (retarget_) {
        TargetColours();
    }
    std::vector<graph::Buffer> columns{graph.Import(*nodes_), graph.Import(*geometries_),
                                       graph.Import(*shapes_), graph.Import(*colours_),
                                       graph.Import(*rowWorlds_)};
    for (const ColourBinding& binding : colourBindings_) {
        columns.push_back(graph.Import(*binding.targets));
    }
    graph::AddUploadPass(graph, "Upload instances", uploader_, columns);
    if (relayout) {
        graph::Buffer sources = graph.Import(*sources_);
        AddLayoutPasses(graph, columns[4], sources);
    }
    AddColourPasses(graph);
}

void InstanceTable::BindColours(BucketId bucket, const DeviceColumn& column) {
    if (column.Type() != DType::Float32 || column.Element() != std::vector<std::int64_t>{3}) {
        throw std::invalid_argument(column.Name() + " does not hold three floats a row");
    }
    colourBindings_.push_back(
        {.bucket = bucket,
         .column = &column,
         .targets = std::make_unique<DeviceColumn>(arenas_, "colour targets", DType::UInt32,
                                                   std::vector<std::int64_t>{}, 0)});
    retarget_ = true;
}

void InstanceTable::TargetColours() {
    for (ColourBinding& binding : colourBindings_) {
        const Bucket& bucket = scene_.Store().GetBucket(binding.bucket);
        binding.hostTargets.resize(bucket.Rows());
        for (std::uint32_t row = 0; row < bucket.Rows(); ++row) {
            binding.hostTargets[row] = Row(bucket.prims[row]).value_or(kNoRow);
        }
        binding.targets->Resize(bucket.Rows(), context_.Main());
        uploader_.Write(binding.targets->Storage(), binding.targets->Range().offset,
                        std::span<const std::uint32_t>(binding.hostTargets));
    }
    retarget_ = false;
}

void InstanceTable::AddColourPasses(graph::Graph& graph) const {
    for (const ColourBinding& binding : colourBindings_) {
        const std::array reads{graph.Import(*binding.column), graph.Import(*binding.targets)};
        graph::Buffer colours = graph.Import(*colours_);
        const auto count = static_cast<std::uint32_t>(binding.hostTargets.size());
        graph.Add(
            "Bound colours", graph::Kind::Compute,
            [&](graph::PassBuilder& pass) {
                for (const graph::Buffer& read : reads) {
                    pass.ReadAddress(read);
                }
                colours = pass.WriteAddress(colours);
            },
            [this, reads, colours, count](graph::PassContext& pass) {
                colourer_.Dispatch(pass.commands, {DivideRoundUp(count, 64u), 1, 1},
                                   shaders::ColourParameters{
                                       .from = pass.Get(reads[0]).BufferRef<float>(),
                                       .targets = pass.Get(reads[1]).BufferRef<shaders::uint>(),
                                       .to = pass.Get(colours).BufferRef<float>(),
                                       .count = count});
            });
    }
}

}  // namespace ernie
