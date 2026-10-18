// SPDX-License-Identifier: BSD-3-Clause
/// @file device_scene.h
/// @brief The device scene: the managers that keep a scene's device state, and the graph each
/// frame runs to bring it up to date and then run the passes that read it, in one submission.
///
///     DeviceScene device(context, arenas, scene, compiler);
///     device.Transforms().Bind(bucket, "xformOp:translate",
///                              device.Columns().Bind(bucket, "xformOp:translate"));
///     const Ticket done = device.Execute([&](graph::Graph& graph) {
///         const graph::Buffer worlds = graph.Import(device.Transforms().Worlds(), "worlds");
///         graph.Add("Shade", ...);
///     });
#pragma once

#include "ernie/device/shader_compiler.h"
#include "ernie/graph/graph.h"
#include "ernie/render/geometry.h"
#include "ernie/render/instances.h"
#include "ernie/render/rt/ray_tracer.h"
#include "ernie/render/scene_columns.h"
#include "ernie/render/transform_graph.h"

#include <functional>
#include <memory>
#include <span>

namespace ernie {

/// @brief A scene on a device and the graph each frame runs: the columns bound on the device
/// follow the scene, world transforms compose from them, the geometry and its acceleration
/// structures follow, and then the frame's own passes run, finding what to trace as a
/// `TraceScene` on the graph's blackboard. Devices without ray queries get no ray tracer yet.
class DeviceScene {
  public:
    /// @param arenas Where the device columns live; `MemoryUse::Shared` to share them with CUDA.
    DeviceScene(DeviceContext& context, Arenas& arenas, const Scene& scene,
                const ShaderCompiler& compiler);
    DeviceScene(const DeviceScene&) = delete;
    DeviceScene& operator=(const DeviceScene&) = delete;

    SceneColumns& Columns() {
        return columns_;
    }
    TransformGraph& Transforms() {
        return transforms_;
    }
    GeometryPool& Geometry() {
        return geometry_;
    }
    InstanceTable& Instances() {
        return instances_;
    }
    /// @return The ray tracer, or null on devices without ray queries.
    RayTracer* Tracer() {
        return tracer_.get();
    }
    /// @return Where the frames' transients come from.
    graph::TransientPool& Pool() {
        return pool_;
    }

    /// @brief Runs a frame on the main queue after `waits`: the columns' sync, the transforms,
    /// then the passes `passes` adds.
    /// @param binary Semaphores the frame waits on and signals, as a swapchain image's.
    /// @return The frame's ticket.
    /// @throws graph::GraphError when the passes cannot run.
    Ticket Execute(const std::function<void(graph::Graph&)>& passes,
                   std::span<const Ticket> waits = {}, BinarySemaphores binary = {});

  private:
    DeviceContext& context_;
    graph::TransientPool pool_;
    SceneColumns columns_;
    TransformGraph transforms_;
    GeometryPool geometry_;
    InstanceTable instances_;
    std::unique_ptr<RayTracer> tracer_;
};

}  // namespace ernie
