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

class Path;

/// A field's or a Volume's local bounds from its attributes alone, no
/// grid read: a CloudLayersField's from its extent and its layers'
/// bottoms and tops, a DenseField's from its array's shape (on the
/// host or a device) and voxel size, and a Volume's as its fields'
/// together in its frame. None for others (an OpenVDBAsset's or a
/// NanoVDBField's grids hold theirs).
std::optional<Bounds3> VolumeBoundsOf(const Scene& scene,
                                      const Path& prim);

}  // namespace ernie
