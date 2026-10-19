// SPDX-License-Identifier: BSD-3-Clause
/// @file bottom_levels.cpp
/// @brief Building, refitting and compacting each geometry's bottom level as frame-graph passes.
#include "ernie/render/rt/bottom_levels.h"

#include "ernie/util/profiler.h"

#include <algorithm>

namespace ernie {

BottomLevels::BottomLevels(DeviceContext& context, Arenas& arenas, const GeometryPool& geometry,
                           std::uint64_t scratchBudget)
    : context_(context), geometry_(geometry), scratchBudget_(scratchBudget),
      pool_(context, "bottom levels"), uploader_(context, context.Main()),
      // Addresses fit an int64 column bit for bit; kernels read them as uint64.
      addresses_(std::make_unique<DeviceColumn>(arenas, "bottom-level addresses", DType::Int64,
                                                std::vector<std::int64_t>{}, 0)) {}

BottomLevels::Level& BottomLevels::At(GeometryId geometry) {
    if (geometry >= levels_.size()) {
        levels_.resize(geometry + 1);
        hostAddresses_.resize(geometry + 1, 0);
    }
    return levels_[geometry];
}

void BottomLevels::SetAddress(GeometryId geometry, std::uint64_t address) {
    At(geometry);
    if (hostAddresses_[geometry] != address) {
        hostAddresses_[geometry] = address;
        dirtyAddresses_.push_back(geometry);
    }
    if (placeholder_ == 0 || geometry == placeholderGeometry_) {
        placeholder_ = address;
        placeholderGeometry_ = geometry;
    }
}

void BottomLevels::FindPlaceholder() {
    const auto built = std::ranges::find_if(hostAddresses_, [](std::uint64_t a) { return a != 0; });
    if (built != hostAddresses_.end()) {
        placeholder_ = *built;
        placeholderGeometry_ = static_cast<GeometryId>(built - hostAddresses_.begin());
    }
}

void BottomLevels::Queue(GeometryId geometry, bool refit) {
    const auto queued = std::ranges::find(pending_, geometry, &Work::geometry);
    if (queued == pending_.end()) {
        pending_.push_back({geometry, refit});
    } else {
        // A rebuild covers a refit.
        queued->refit = queued->refit && refit;
    }
}

void BottomLevels::Apply(const GeometryPool::Changes& changes) {
    for (const GeometryId geometry : changes.freed) {
        std::erase_if(pending_, [&](const Work& work) { return work.geometry == geometry; });
        // Frames in flight may trace it: destroying it retires it with their tickets.
        At(geometry) = {};
        SetAddress(geometry, 0);
    }
    for (const GeometryId geometry : changes.rebuilt) {
        At(geometry).refits = 0;
        Queue(geometry, false);
    }
    for (const GeometryId geometry : changes.refitted) {
        Level& level = At(geometry);
        // A level built static is rebuilt refittable the first time its points move.
        const bool refit =
            level.structure && level.deforming && level.refits < kRefitsBeforeRebuild;
        if (!refit) {
            level.deforming = true;
            level.refits = 0;
        }
        Queue(geometry, refit);
    }
}

BottomBuild BottomLevels::Build(GeometryId id, bool refit, std::vector<TriangleGeometry>& triangles,
                                std::vector<BoxGeometry>& boxes) const {
    const Geometry& geometry = geometry_.Get(id);
    BottomBuild build{.target = levels_[id].structure.get(), .update = refit};
    if (geometry.kind == GeometryKind::Triangles) {
        triangles.push_back(
            {.vertices = {&geometry_.Storage(geometry.vertices), geometry.vertices.offset},
             .vertexCount = geometry.vertexCount,
             .indices = {&geometry_.Storage(geometry.indices), geometry.indices.offset},
             .triangleCount = geometry.triangleCount});
        build.triangles = std::span(&triangles.back(), 1);
    } else {
        boxes.push_back(
            {.boxes = {&geometry_.Storage(geometry.box), geometry.box.offset}, .count = 1});
        build.boxes = std::span(&boxes.back(), 1);
    }
    return build;
}

void BottomLevels::AddBuilds(graph::Graph& graph) {
    ERNIE_ZONE("Prepare the bottom levels");
    Moves moves = TakeCompactedSizes();
    FrameBuilds builds = PlanBuilds();
    Compaction compaction = StartCompactionQuery(builds.built);
    if (placeholder_ == 0) {
        FindPlaceholder();
    }
    AddAddressUpload(graph);
    if (!builds.builds.empty() || !moves.empty()) {
        AddBuildPass(graph, std::move(builds), std::move(moves), compaction);
    }
    if (!compaction.geometries.empty()) {
        compactions_.push_back(std::move(compaction));
    }
}

BottomLevels::Moves BottomLevels::TakeCompactedSizes() {
    // Each level whose compacted size came back moves into a structure of that size.
    Moves moves;
    std::erase_if(compactions_, [&](Compaction& compaction) {
        const std::optional<std::vector<std::uint64_t>> sizes = compaction.query->Sizes();
        if (!sizes) {
            return false;
        }
        for (std::size_t i = 0; i < compaction.geometries.size(); ++i) {
            Level& level = levels_[compaction.geometries[i]];
            level.compacting = false;
            // Rebuilt, refitted or freed since: its size no longer applies.
            if (level.structure != compaction.structures[i] || level.deforming) {
                continue;
            }
            AccelerationSizes sizesCompacted = level.structure->Sizes();
            sizesCompacted.storage = (*sizes)[i];
            auto compacted = std::make_shared<AccelerationStructure>(
                pool_, sizesCompacted, level.structure->Name() + " compacted");
            ++totals_.compacted;
            totals_.saved += level.structure->Sizes().storage - sizesCompacted.storage;
            moves.emplace_back(level.structure, compacted);
            level.structure = compacted;
            SetAddress(compaction.geometries[i], compacted->Address());
        }
        return true;
    });
    return moves;
}

BottomLevels::FrameBuilds BottomLevels::PlanBuilds() {
    FrameBuilds frame;
    // Builds point into these, so they hold every geometry without growing.
    frame.triangles.reserve(pending_.size());
    frame.boxes.reserve(pending_.size());
    std::vector<Work> later;
    for (const Work& work : pending_) {
        const Geometry& geometry = geometry_.Get(work.geometry);
        Level& level = levels_[work.geometry];
        if (geometry.kind == GeometryKind::Triangles && geometry.triangleCount == 0) {
            level.structure.reset();
            SetAddress(work.geometry, 0);
            continue;
        }
        if (!work.refit) {
            std::vector<TriangleGeometry> sizing;
            std::vector<BoxGeometry> sizingBoxes;
            const BuildFlags flags = level.deforming ? kDeforming : kStatic;
            const BottomBuild input = Build(work.geometry, false, sizing, sizingBoxes);
            const AccelerationSizes sizes = !input.triangles.empty()
                                                ? Sizes(context_, input.triangles, flags)
                                                : Sizes(context_, input.boxes, flags);
            // Over the budget, the level keeps tracing as it was until a later frame.
            if (!frame.builds.empty() && frame.scratch + sizes.buildScratch > scratchBudget_) {
                later.push_back(work);
                continue;
            }
            // A new structure, so frames in flight keep tracing the old one until they finish.
            level.structure = std::make_shared<AccelerationStructure>(
                pool_, sizes, "geometry " + std::to_string(work.geometry));
            level.refits = 0;
        } else {
            ++level.refits;
        }
        ++(work.refit ? totals_.refitted : totals_.built);
        const BottomBuild build = Build(work.geometry, work.refit, frame.triangles, frame.boxes);
        frame.scratch +=
            work.refit ? build.target->Sizes().updateScratch : build.target->Sizes().buildScratch;
        frame.builds.push_back(build);
        frame.built.push_back(work.geometry);
        SetAddress(work.geometry, level.structure->Address());
    }
    pending_ = std::move(later);
    return frame;
}

BottomLevels::Compaction BottomLevels::StartCompactionQuery(const std::vector<GeometryId>& built) {
    // Static levels have their compacted sizes read back for a later frame.
    Compaction compaction;
    for (const GeometryId geometry : built) {
        Level& level = levels_[geometry];
        if (!level.deforming && !level.compacting) {
            level.compacting = true;
            compaction.geometries.push_back(geometry);
            compaction.structures.push_back(level.structure);
        }
    }
    if (!compaction.geometries.empty()) {
        compaction.query = std::make_unique<CompactionQuery>(
            context_, static_cast<std::uint32_t>(compaction.geometries.size()));
    }
    return compaction;
}

void BottomLevels::AddAddressUpload(graph::Graph& graph) {
    if (dirtyAddresses_.empty()) {
        return;
    }
    addresses_->Resize(hostAddresses_.size(), context_.Main());
    StageRows(uploader_, *addresses_, hostAddresses_, dirtyAddresses_);
    dirtyAddresses_.clear();
    graph::Buffer addresses = graph.Import(*addresses_);
    graph::AddUploadPass(graph, "Upload bottom-level addresses", uploader_,
                         std::span(&addresses, 1));
}

void BottomLevels::AddBuildPass(graph::Graph& graph, FrameBuilds frame, Moves moves,
                                const Compaction& compaction) {
    graph::Acceleration levels = graph.Import(pool_);
    std::vector<graph::Buffer> inputs = geometry_.Import(graph);
    graph::Buffer scratchBuffer =
        frame.scratch > 0 ? graph.Create(ScratchBytes(frame.builds, {}), "bottom-level scratch")
                          : graph::Buffer{};
    CompactionQuery* query = compaction.query.get();
    std::vector<const AccelerationStructure*> queried(compaction.structures.size());
    std::ranges::transform(compaction.structures, queried.begin(),
                           [](const auto& structure) { return structure.get(); });
    graph.Add(
        "Build bottom levels", graph::Kind::Build,
        [&](graph::PassBuilder& pass) {
            for (const graph::Buffer& input : inputs) {
                pass.Read(input, graph::Use::BuildInput);
            }
            if (scratchBuffer.Valid()) {
                scratchBuffer = pass.Write(scratchBuffer, graph::Use::BuildScratch);
            }
            pass.Read(levels, graph::Use::BuildInput);
            levels = pass.Build(levels);
        },
        [frame = std::move(frame), moves = std::move(moves), queried, query,
         scratchBuffer](graph::PassContext& pass) {
            for (const auto& [source, compacted] : moves) {
                RecordCompaction(pass.commands, *source, *compacted);
                // The copy reads the source: it goes once this frame's work is done.
                pass.Keep(source);
            }
            if (!frame.builds.empty()) {
                const graph::BufferSlice space = pass.Get(scratchBuffer);
                RecordBuilds(pass.commands, frame.builds, {}, {space.buffer, space.offset});
            }
            if (query != nullptr) {
                query->Record(pass.commands, queried);
            }
        });
}

}  // namespace ernie
