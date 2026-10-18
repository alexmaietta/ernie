// SPDX-License-Identifier: BSD-3-Clause
/// @file camera.cpp
/// @brief The camera's registry entry.
#include "ernie/scene/schemas/camera.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(Camera, "Camera", &Xformable::Definition(), Concrete, kNoExtent,
                    "A view looking down -z, y up, with a lens and film.", ERNIE_CAMERA_ATTRIBUTES)

}  // namespace ernie
