// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>
#include <vector>

#include "ernie/device/ref.h"
#include "ernie/scene/scene.h"
#include "ernie/xr/frame.h"

namespace ernie::xr {

struct RigState {
    Ref<Scene> scene;
    XformPrim root;
    XformPrim head;
    CameraPrim leftEye;
    CameraPrim rightEye;
    XformPrim leftHand;
    XformPrim rightHand;
    std::string product;
    std::string colorVar;
    std::string depthVar;
    std::vector<std::string> vars;
    Uint2 recommended;
    Pose rootPose{{0, 0, 0}, {0, 0, 0, 1}, true};
};

}
