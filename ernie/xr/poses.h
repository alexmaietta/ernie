// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/rotation.h"
#include "ernie/stage/schema/types.h"
#include "ernie/xr/frame.h"

namespace ernie::xr {

inline Double3 Add(Double3 a, Double3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Double3 Scale(Double3 v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}

inline Pose Offset(const Pose& pose, const Pose& offset) {
    Pose placed = pose;
    placed.position = Add(pose.position, Rotate(pose.orientation, offset.position));
    placed.orientation = Multiply(pose.orientation, offset.orientation);
    return placed;
}

}
