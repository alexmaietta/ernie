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

#include <cstdint>
#include <span>

namespace ernie {

/// @brief What a tracing kernel hands its `RayTracer`, as ray_tracer.slang defines it.
using TracerParameters = shaders::TracerParameters;

/// @brief What a frame's passes trace, left on the graph's blackboard by `RayTracer::AddBuild`.
struct TraceScene {
    TracerParameters parameters;
    graph::Acceleration top;
    graph::Buffer records;
    std::uint32_t instances = 0;
    /// Each instance row's albedo, three linear RGB floats, and kernels' handle of them.
    graph::Buffer colours;
    shaders::BufferRef<float> colourBuffer;
    /// The render origin the structures are relative to: rays start from points less it.
    glm::dvec3 origin{0.0};

    /// @brief Declares a pass's reads of everything tracing touches.
    void Read(graph::PassBuilder& pass) const {
        pass.Read(top);
        pass.ReadAddress(records);
        pass.ReadAddress(colours);
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
    /// `TraceScene` on the graph's blackboard.
    void AddBuild(graph::Graph& graph, const GeometryPool::Changes& changes,
                  const TransformGraph& transforms, const InstanceTable& instances);

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
};

}  // namespace ernie
