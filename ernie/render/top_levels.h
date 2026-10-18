// SPDX-License-Identifier: BSD-3-Clause
/// @file top_levels.h
/// @brief The top-level acceleration structure rays are traced through, rebuilt each frame on the
/// device from the instances' world transforms with no loop on the host.
#pragma once

#include "ernie/device/acceleration.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/graph/graph.h"
#include "ernie/render/bottom_levels.h"
#include "ernie/render/instances.h"
#include "ernie/render/transform_graph.h"

#include <cstdint>
#include <memory>

namespace ernie {

/// @brief What a frame's passes trace, left on the graph's blackboard by `TopLevels::AddBuild`.
struct SceneStructure {
    /// The top level, for passes to read with `Use::Trace`.
    graph::Acceleration top;
    /// Its address, as kernels trace it: `RaytracingAccelerationStructure(address)`.
    std::uint64_t address = 0;
    std::uint32_t instances = 0;
};

/// @brief The scene's top level, rebuilt every frame: a kernel writes each instance's record from
/// its world transform, its shape transform and its geometry's bottom level, and the build follows.
/// Instances whose bottom level is not built are switched off. An instance's custom index is its
/// geometry; its row in the instance table is `InstanceIndex()`. Capacity grows by doubling and
/// shrinks once it is four times what is needed, so instance counts that wobble do not reallocate.
class TopLevels {
  public:
    TopLevels(DeviceContext& context, const ShaderCompiler& compiler);
    TopLevels(const TopLevels&) = delete;
    TopLevels& operator=(const TopLevels&) = delete;

    /// @brief Adds the passes that write the instances and build the top level, after the
    /// transforms, the instance uploads and the bottom levels' builds, and puts a
    /// `SceneStructure` on the graph's blackboard.
    void AddBuild(graph::Graph& graph, const TransformGraph& transforms,
                  const InstanceTable& instances, const BottomLevels& bottoms);

    /// @return The top level as last built, or null before the first build.
    const AccelerationStructure* Structure() const {
        return top_.get();
    }

  private:
    DeviceContext& context_;
    ComputeKernel writer_;
    AccelerationPool pool_;
    std::unique_ptr<AccelerationStructure> top_;
};

}  // namespace ernie
