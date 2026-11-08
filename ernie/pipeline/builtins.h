// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/pipeline/node.h"

namespace ernie::pipeline {

inline constexpr const char* kCameraNodes = "ernie.render.kernels.pipeline.nodes.camera";

void RegisterBuiltinNodes(NodeRegistry& registry);

}
