// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.cpp
/// @brief The camera's registry entry and Ernie's camera API schemas'.
#include "ernie/scene/schemas/camera.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(Camera, "Camera", &Xformable::Definition(), Concrete, kNoExtent,
                    "A view looking down -z, y up, with a lens and film.", ERNIE_CAMERA_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ErnieLensAPI, "ErnieLensAPI", nullptr, SingleApply, kInheritExtent,
                    "A camera's lens beyond USD's: fisheyes, distortion, vignetting and colour.",
                    ERNIE_LENS_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ErniePolarizerAPI, "ErniePolarizerAPI", nullptr, SingleApply, kInheritExtent,
                    "A polarizer in front of a camera's lens, or a polarization sensor's mosaic.",
                    ERNIE_POLARIZER_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ErnieSensorAPI, "ErnieSensorAPI", nullptr, SingleApply, kInheritExtent,
                    "A camera's sensor: its spectral range, noise, ADC and quantum efficiency.",
                    ERNIE_SENSOR_API_ATTRIBUTES)

}  // namespace ernie
