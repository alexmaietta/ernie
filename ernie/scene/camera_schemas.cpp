// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/camera_schemas.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(Camera, tokens::kCamera, &Imageable::Definition(),
                    false,
                    "A camera looking down its -z axis: "
                    "UsdGeomCamera's projection, apertures, "
                    "exposure and shutter, and Ernie's lens, scan, "
                    "spectral sampling, sensor and "
                    "view poses.",
                    ERNIE_CAMERA_ATTRIBUTES)

std::optional<std::vector<float>>
Camera::GetSensorQe(std::string_view channel) const {
    const auto authored = [](bool has, std::vector<float> pairs) {
        return has ? std::optional(std::move(pairs)) : std::nullopt;
    };
    if (channel == "r") {
        return authored(HasSensorQeR(), GetSensorQeR());
    }
    if (channel == "g") {
        return authored(HasSensorQeG(), GetSensorQeG());
    }
    if (channel == "b") {
        return authored(HasSensorQeB(), GetSensorQeB());
    }
    if (channel == "mono") {
        return authored(HasSensorQeMono(), GetSensorQeMono());
    }
    return std::nullopt;
}

std::vector<const PrimTypeDef*> CameraSchemaTypes() {
    return {&Camera::Definition()};
}

}  // namespace ernie
