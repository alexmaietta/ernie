// SPDX-License-Identifier: BSD-3-Clause
/// @file bottom_levels.h
/// @brief A bottom-level acceleration structure for each geometry, built on the device as the
/// geometry changes: static geometry traced fast and compacted, deforming geometry refitted.
#pragma once

#include "ernie/device/acceleration.h"
#include "ernie/device/column.h"
#include "ernie/device/staging.h"
#include "ernie/graph/graph.h"
#include "ernie/render/geometry.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ernie {

/// @brief The bottom levels of a geometry pool. Builds wait for the frame's graph, batched in one
/// call under a scratch budget; geometry whose bottom level is not built yet has address 0, which
/// a top level takes as an instance switched off.
///
/// Static geometry is built to trace fast and compacted a few frames later, once its compacted
/// size has come back without a wait. Geometry refitted switches to a refittable build, and is
/// built anew after `kRefitsBeforeRebuild` refits, as refits loosen the tree.
class BottomLevels {
  public:
    static constexpr std::uint32_t kRefitsBeforeRebuild = 16;

    /// @param arenas Where the address column lives.
    /// @param scratchBudget The most scratch one frame's builds take; one build always goes.
    BottomLevels(DeviceContext& context, Arenas& arenas, const GeometryPool& geometry,
                 std::uint64_t scratchBudget = std::uint64_t{64} << 20);
    BottomLevels(const BottomLevels&) = delete;
    BottomLevels& operator=(const BottomLevels&) = delete;

    /// @brief Takes in what a geometry update changed: builds and refits to run, levels to free.
    void Apply(const GeometryPool::Changes& changes);
    /// @brief Adds the passes that compact finished levels, run this frame's builds and upload
    /// the addresses, after the geometry's uploads and before the top levels' builds.
    void AddBuilds(graph::Graph& graph);

    /// @return Each geometry's bottom level address, a uint64 a geometry id; 0 until built.
    const DeviceColumn& Addresses() const {
        return *addresses_;
    }
    /// @return The pool the levels live in, which top-level builds import to read.
    const AccelerationPool& Pool() const {
        return pool_;
    }
    /// @return Builds and refits not run yet, as over the budget.
    std::size_t Pending() const {
        return pending_.size();
    }
    /// @brief What the levels have cost so far, for memory and profiler reports.
    struct Statistics {
        std::uint32_t built = 0;
        std::uint32_t refitted = 0;
        std::uint32_t compacted = 0;
        /// Bytes compaction gave back.
        std::uint64_t saved = 0;
    };
    const Statistics& Totals() const {
        return totals_;
    }

  private:
    struct Level {
        std::shared_ptr<AccelerationStructure> structure;
        bool deforming = false;
        std::uint32_t refits = 0;
        /// Whether its compacted size is being read back.
        bool compacting = false;
    };
    struct Work {
        GeometryId geometry = 0;
        bool refit = false;
    };
    /// Compacted sizes on their way back, for the levels built with them.
    struct Compaction {
        std::unique_ptr<CompactionQuery> query;
        std::vector<GeometryId> geometries;
        std::vector<std::shared_ptr<AccelerationStructure>> structures;
    };

    /// Levels moving into compacted copies: each source, then its copy.
    using Moves = std::vector<
        std::pair<std::shared_ptr<AccelerationStructure>, std::shared_ptr<AccelerationStructure>>>;
    /// A frame's builds and refits, and the geometry they read.
    struct FrameBuilds {
        std::vector<TriangleGeometry> triangles;
        std::vector<BoxGeometry> boxes;
        std::vector<BottomBuild> builds;
        std::vector<GeometryId> built;
        std::uint64_t scratch = 0;
    };

    Level& At(GeometryId geometry);
    void Queue(GeometryId geometry, bool refit);
    void SetAddress(GeometryId geometry, std::uint64_t address);
    /// @return The triangles or box a geometry's level is built over.
    BottomBuild Build(GeometryId geometry, bool refit, std::vector<TriangleGeometry>& triangles,
                      std::vector<BoxGeometry>& boxes) const;
    /// @return The moves of levels whose compacted sizes have come back.
    Moves TakeCompactedSizes();
    /// @return The pending builds that fit the scratch budget; the rest wait for a later frame.
    FrameBuilds PlanBuilds();
    /// @return The query reading back the compacted sizes of the static levels just built.
    Compaction StartCompactionQuery(const std::vector<GeometryId>& built);
    void AddAddressUpload(graph::Graph& graph);
    void AddBuildPass(graph::Graph& graph, FrameBuilds builds, Moves moves,
                      const Compaction& compaction);

    DeviceContext& context_;
    const GeometryPool& geometry_;
    std::uint64_t scratchBudget_;
    AccelerationPool pool_;
    Uploader uploader_;
    std::vector<Level> levels_;
    std::vector<Work> pending_;
    std::vector<Compaction> compactions_;
    std::vector<std::uint64_t> hostAddresses_;
    std::vector<std::uint32_t> dirtyAddresses_;
    std::unique_ptr<DeviceColumn> addresses_;
    Statistics totals_;
};

}  // namespace ernie
