// SPDX-License-Identifier: BSD-3-Clause
/// @file device_scene.cpp
/// @brief Building and running each frame's graph over the device scene.
#include "ernie/render/device_scene.h"

namespace ernie {

DeviceScene::DeviceScene(DeviceContext& context, Arenas& arenas, const Scene& scene,
                         const ShaderCompiler& compiler)
    : context_(context), pool_(context), columns_(context, arenas, scene),
      transforms_(context, arenas, scene, compiler) {}

Ticket DeviceScene::Execute(const std::function<void(graph::Graph&)>& passes,
                            std::span<const Ticket> waits, BinarySemaphores binary) {
    ERNIE_ZONE("Run a frame");
    graph::Graph graph(context_, pool_);
    columns_.AddSync(graph);
    transforms_.AddUpdate(graph);
    if (passes) {
        passes(graph);
    }
    return graph.Execute(context_.Main(), waits, binary);
}

}  // namespace ernie
