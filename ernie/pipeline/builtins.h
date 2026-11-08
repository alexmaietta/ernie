// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/pipeline/node.h"

namespace ernie::pipeline {

inline constexpr const char* kCameraNodes = "ernie.render.kernels.pipeline.nodes.camera";
inline constexpr const char* kIspNodes = "ernie.render.kernels.pipeline.nodes.isp";

void RegisterBuiltinNodes(NodeRegistry& registry);

const NodeRegistry& BuiltinNodes();

}
