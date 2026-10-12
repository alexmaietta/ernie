// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/session/frame_params.h"

#include <vector>

namespace ernie {

class Scene;

/// The scene's visible Volumes (UsdVol, `volume_schemas.h`), each
/// with its bound Material's volume inputs (defaults without one) and
/// the fields they name: the density field (`inputs:densityField`),
/// the temperature field and the emission field, each an OpenVDBAsset
/// the Volume's `field:<name>` relationship targets; a name the
/// Volume has no field for, or a target of another type, leaves that
/// field out.
std::vector<VolumeParams> VolumesOf(const Scene& scene);

}  // namespace ernie
