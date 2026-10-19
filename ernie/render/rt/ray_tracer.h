// SPDX-License-Identifier: BSD-3-Clause
/// @file ray_tracer.h
/// @brief The ray tracer: the acceleration structures the scene is traced through, kept up to date
/// each frame, and the Slang modules its passes trace them with.
///
///     const TraceScene& scene = graph.Board().Get<TraceScene>();
///     graph.Add("Shade", graph::Kind::Compute,
///               [&](graph::PassBuilder& pass) { scene.Read(pass); ... },
///               [&](graph::PassContext& pass) { kernel.Dispatch(..., {scene.parameters, ...}); });
#pragma once

#include "ernie/device/shader_compiler.h"
#include "ernie/graph/graph.h"
#include "ernie/render/geometry.h"
#include "ernie/render/instances.h"
#include "ernie/render/rt/bottom_levels.h"
#include "ernie/render/rt/top_levels.h"
#include "ernie/render/transform_graph.h"
#include "ernie/shaders/rt/ray_tracer.slang"

#include <glm/vec3.hpp>

#include <array>
#include <cstdint>
#include <span>

namespace ernie {

/// @brief What a tracing kernel hands its `RayTracer`, as ray_tracer.slang defines it.
using TracerParameters = shaders::TracerParameters;

/// @brief What a frame's passes trace, left on the graph's blackboard by `RayTracer::AddBuild`.
struct TraceScene {
    TracerParameters parameters;
    /// Every batch world's top level, in one pool.
    graph::Acceleration top;
    /// What the parameters point at: the top levels, the records' instance rows and the geometry.
    std::array<graph::Buffer, 3> tables;
    std::uint32_t instances = 0;
    /// The batch worlds, each with a top level; views pick theirs by index.
    std::uint32_t worlds = 1;
    /// Each instance row's albedo, three linear RGB floats, and its prim's slot.
    graph::Buffer colours;
    graph::Buffer prims;
    /// The transform graph's worlds, for poses read on the device.
    graph::Buffer transforms;
    /// The frame's number, a uint counted on the device, which random numbers are keyed by.
    graph::Buffer frame;
    /// The render origin the structures are relative to: rays start from points less it.
    glm::dvec3 origin{0.0};

    /// @brief Declares a pass's reads of everything tracing touches.
    void Read(graph::PassBuilder& pass) const {
        pass.Read(top);
        for (const graph::Buffer& buffer : tables) {
            pass.ReadAddress(buffer);
        }
        for (const graph::Buffer& buffer : {colours, prims, transforms, frame}) {
            pass.ReadAddress(buffer);
        }
    }
};

/// @brief The scene's acceleration structures on a device with ray queries: a bottom level for
/// each geometry, built and compacted as it changes, and a top level over the instances rebuilt
/// every frame.
class RayTracer {
  public:
    RayTracer(DeviceContext& context, Arenas& arenas, const ShaderCompiler& compiler,
              const GeometryPool& geometry);
    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;

    /// @return The modules, in import order, a tracing kernel compiles with before its own; it
    /// then imports `ray_tracer`.
    static std::span<const ShaderModule> Modules();

    /// @brief Adds the passes that bring the structures up to date with the frame's geometry
    /// changes, after the transforms and the geometry's and instances' uploads, and puts a
    /// `TraceScene` on the graph's blackboard, after a pass counting the frame on the device.
    /// @param rebuildTops Whether to build the top levels, as `TopLevels::AddBuild` takes it.
    void AddBuild(graph::Graph& graph, const GeometryPool::Changes& changes,
                  const TransformGraph& transforms, const InstanceTable& instances,
                  bool rebuildTops = true);

    BottomLevels& Bottoms() {
        return bottoms_;
    }
    const TopLevels& Tops() const {
        return tops_;
    }

  private:
    const GeometryPool& geometry_;
    BottomLevels bottoms_;
    TopLevels tops_;
    ComputeKernel counter_;
    /// The frame's number, counted on the device from zero once started.
    std::unique_ptr<DeviceColumn> frame_;
    bool counting_ = false;
};

}  // namespace ernie
