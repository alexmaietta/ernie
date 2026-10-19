// SPDX-License-Identifier: BSD-3-Clause
/// @file device_scene.cpp
/// @brief Building and running each frame's graph over the device scene.
#include "ernie/render/device_scene.h"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>

namespace ernie {

DeviceScene::DeviceScene(DeviceContext& context, Arenas& arenas, const Scene& scene,
                         const ShaderCompiler& compiler, WorldCapacities capacities)
    : context_(context), scene_(scene), pool_(context), columns_(context, arenas, scene),
      transforms_(context, arenas, scene, compiler), geometry_(context, arenas, scene),
      instances_(context, arenas, scene, compiler, capacities) {
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
    captured_.clear();
    graph::Graph graph(context_, pool_);
    AddFrame(graph, passes);
    return graph.Execute(context_.Main(), waits, binary);
}

Ticket DeviceScene::Step(const std::function<void(graph::Graph&)>& passes, std::uint64_t key,
                         std::span<const Ticket> waits) {
    const CaptureKey now{scene_.Store().Version(), transforms_.Origin(), key};
    // Frames of another version or origin record what no longer stands.
    std::erase_if(captured_, [&](const Captured& frame) {
        return frame.key.version != now.version || frame.key.origin != now.origin;
    });
    const auto found = std::ranges::find(captured_, now, &Captured::key);
    replayed_ = found != captured_.end() && Settled();
    if (replayed_) {
        ERNIE_ZONE("Replay a frame");
        return found->graph->Submit(waits);
    }
    ERNIE_ZONE("Run a frame");
    if (found != captured_.end()) {
        captured_.erase(found);
    }
    graph::Graph graph(context_, pool_);
    AddFrame(graph, passes);
    if (!Settled() || !graph.Capturable()) {
        return graph.Execute(context_.Main(), waits);
    }
    if (captured_.size() == kKeptFrames) {
        captured_.erase(captured_.begin());
    }
    captured_.push_back({now, graph.Capture(context_.Main())});
    return captured_.back().graph->Submit(waits);
}

bool DeviceScene::Settled() const {
    return !tracer_ || tracer_->Bottoms().Idle();
}

void DeviceScene::AddFrame(graph::Graph& graph, const std::function<void(graph::Graph&)>& passes) {
    // Instances move only with the host's edits, a rebase, bottom levels still building, or
    // bound columns written on the device unseen; otherwise the last frame's worlds and top
    // levels stand.
    const MovedKey now{scene_.Store().Version(), transforms_.Origin()};
    const bool moved = !moved_ || *moved_ != now || !Settled() || transforms_.Bound();
    moved_ = now;
    columns_.AddSync(graph);
    transforms_.AddUpdate(graph, moved);
    const GeometryPool::Changes changes = geometry_.Update();
    instances_.Apply(changes);
    geometry_.AddUploads(graph);
    instances_.AddUploads(graph);
    if (tracer_) {
        tracer_->AddBuild(graph, changes, transforms_, instances_, moved);
    }
    if (passes) {
        passes(graph);
    }
}

}  // namespace ernie
