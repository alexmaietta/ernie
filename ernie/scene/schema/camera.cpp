// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.cpp
/// @brief The camera's registry entry and Ernie's camera API schemas'.
#include "ernie/scene/schema/camera.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(Camera, Concrete, kNoExtent, ERNIE_CAMERA_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ErnieLensAPI, SingleApply, kInheritExtent, ERNIE_LENS_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ErniePolarizerAPI, SingleApply, kInheritExtent, ERNIE_POLARIZER_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ErnieSensorAPI, SingleApply, kInheritExtent, ERNIE_SENSOR_API_ATTRIBUTES)

}  // namespace ernie
