// SPDX-License-Identifier: BSD-3-Clause
/// @file device_scene.cpp
/// @brief Building and running each frame's graph over the device scene.
#include "ernie/render/device_scene.h"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

namespace ernie {

DeviceScene::DeviceScene(DeviceContext& context, Arenas& arenas, const Scene& scene,
                         const ShaderCompiler& compiler)
    : context_(context), scene_(scene), pool_(context), columns_(context, arenas, scene),
      transforms_(context, arenas, scene, compiler), geometry_(context, arenas, scene),
      instances_(context, arenas) {
    if (context.Capabilities().rayQuery) {
        tracer_ = std::make_unique<RayTracer>(context, arenas, compiler, geometry_);
    }
}

bool DeviceScene::Follow(const glm::dvec3& viewpoint) {
    if (glm::length(viewpoint - transforms_.Origin()) <= kRebaseDistance) {
        return false;
    }
    ERNIE_ZONE("Rebase the render origin");
    transforms_.SetOrigin(glm::round(viewpoint / kRebaseDistance) * kRebaseDistance);
    return true;
}

Ticket DeviceScene::Execute(const std::function<void(graph::Graph&)>& passes,
                            std::span<const Ticket> waits, BinarySemaphores binary) {
    ERNIE_ZONE("Run a frame");
    captured_.reset();
    graph::Graph graph(context_, pool_);
    AddFrame(graph, passes);
    return graph.Execute(context_.Main(), waits, binary);
}

Ticket DeviceScene::Step(const std::function<void(graph::Graph&)>& passes, std::uint64_t key,
                         std::span<const Ticket> waits) {
    const CaptureKey now{scene_.Store().Version(), transforms_.Origin(), key};
    replayed_ = captured_ && now == capturedKey_ && Settled();
    if (replayed_) {
        ERNIE_ZONE("Replay a frame");
        return captured_->Submit(waits);
    }
    ERNIE_ZONE("Run a frame");
    captured_.reset();
    graph::Graph graph(context_, pool_);
    AddFrame(graph, passes);
    if (!Settled() || !graph.Capturable()) {
        return graph.Execute(context_.Main(), waits);
    }
    captured_ = graph.Capture(context_.Main());
    capturedKey_ = now;
    return captured_->Submit(waits);
}

bool DeviceScene::Settled() const {
    return !tracer_ || tracer_->Bottoms().Idle();
}

void DeviceScene::AddFrame(graph::Graph& graph, const std::function<void(graph::Graph&)>& passes) {
    columns_.AddSync(graph);
    transforms_.AddUpdate(graph);
    const GeometryPool::Changes changes = geometry_.Update();
    instances_.Apply(changes);
    geometry_.AddUploads(graph);
    instances_.AddUploads(graph);
    if (tracer_) {
        tracer_->AddBuild(graph, changes, transforms_, instances_);
    }
    if (passes) {
        passes(graph);
    }
}

}  // namespace ernie
