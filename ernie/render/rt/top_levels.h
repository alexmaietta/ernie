// SPDX-License-Identifier: BSD-3-Clause
/// @file top_levels.h
/// @brief A top-level acceleration structure for each batch world, rebuilt each frame on the
/// device from the instances' world transforms, every world in one build call with no loop over
/// the instances on the host.
#pragma once

#include "ernie/device/acceleration.h"
#include "ernie/device/column.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/staging.h"
#include "ernie/graph/graph.h"
#include "ernie/render/instances.h"
#include "ernie/render/rt/bottom_levels.h"
#include "ernie/render/transform_graph.h"
#include "ernie/shaders/rt/ray_tracer.slang"

#include <memory>
#include <vector>

namespace ernie {

/// @brief The scene's top levels, one a world, rebuilt every frame: a kernel writes each world's
/// instance records, laid out by the instance table, from their world transforms, shape transforms
/// and geometries' bottom levels, and one call builds them all. Instances whose bottom level is not
/// built are switched off. An instance's custom index is its geometry; its row in the instance
/// table is `InstanceIndex()`. Shared prims are in every world's top level. A world's capacity
/// grows by doubling and shrinks once it is four times what is needed, so instance counts that
/// wobble do not reallocate.
class TopLevels {
  public:
    TopLevels(DeviceContext& context, Arenas& arenas, const ShaderCompiler& compiler);
    TopLevels(const TopLevels&) = delete;
    TopLevels& operator=(const TopLevels&) = delete;

    /// @brief Adds the passes that write the records and build every world's top level, after the
    /// transforms, the instance uploads and the bottom levels' builds.
    /// @return The top levels' pool, for passes to read with `Use::Trace`.
    graph::Acceleration AddBuild(graph::Graph& graph, const TransformGraph& transforms,
                                 const InstanceTable& instances, const BottomLevels& bottoms);

    /// @return Each world's `TopLevel`, its address and records, which kernels select by view.
    const DeviceColumn& Tops() const {
        return *tops_;
    }
    std::uint32_t Worlds() const {
        return static_cast<std::uint32_t>(structures_.size());
    }

  private:
    /// Remakes the top levels whose worlds outgrow them or use under a quarter of them; with a
    /// `declared` capacity a world, makes each once at that size.
    void EnsureCapacity(std::span<const InstanceTable::WorldRecords> worlds,
                        std::uint32_t declared);
    /// Adds the pass writing each record from its instance's world and shape transforms.
    void AddWritePass(graph::Graph& graph, const TransformGraph& transforms,
                      const InstanceTable& instances, const BottomLevels& bottoms,
                      graph::Buffer& records) const;
    graph::Acceleration AddBuildPass(graph::Graph& graph, const BottomLevels& bottoms,
                                     std::span<const InstanceTable::WorldRecords> worlds,
                                     graph::Buffer records) const;

    DeviceContext& context_;
    ComputeKernel writer_;
    AccelerationPool pool_;
    Uploader uploader_;
    std::vector<std::unique_ptr<AccelerationStructure>> structures_;
    std::vector<shaders::TopLevel> hostTops_;
    std::unique_ptr<DeviceColumn> tops_;
};

}  // namespace ernie
