// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <memory>

#include "ernie/pipeline/node.h"

namespace ernie::pipeline {

inline constexpr const char* kCameraNodes = "ernie.render.kernels.pipeline.nodes.camera";
inline constexpr const char* kIspNodes = "ernie.render.kernels.pipeline.nodes.isp";
inline constexpr const char* kLidarNodes = "ernie.render.kernels.pipeline.nodes.lidar";

void RegisterBuiltinNodes(NodeRegistry& registry);

const NodeRegistry& BuiltinNodes();

std::shared_ptr<const NodeRegistry> PipelineNodes();

void RegisterPipelineNode(NodeDef def);

}
