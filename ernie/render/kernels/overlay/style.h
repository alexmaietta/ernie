// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

ERNIE_ENUM(OverlayDepth){
    Test,
    XRay,
    Always,
};

ERNIE_ENUM(MarkerShape){
    Circle,
    Ring,
    Square,
    Diamond,
    Cross,
    Plus,
};

ERNIE_ENUM(OverlayUnits){
    Pixels,
    Meters,
};

ERNIE_ENUM(OverlayPrimShape){
    Frames,
    Bounds,
    Frustums,
    Lights,
    CameraImages,
};

}
