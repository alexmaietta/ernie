// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/app/property_fields.h"
#include "ernie/scene/brdf.h"
#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/sensor_filters.h"

#include <string_view>
#include <vector>

namespace ernie::app {

/// The attributes Ernie reads on the prim at `path`, by its type, in
/// sections with the fallbacks Ernie uses: shape parameters of the
/// UsdGeom primitives, UsdLux light inputs, materials and their
/// BRDF's parameters (from `brdfs`), cameras, render prims and sensor
/// filters (their types from `filters`), and what every prim may
/// carry (visibility, world). Empty for prims Ernie reads nothing
/// from.
std::vector<PropertySection>
SchemaOf(const Scene& scene, const Path& path,
         const BrdfRegistry& brdfs,
         const SensorFilterRegistry& filters);

/// A Camera prim's sections from its declaration (`Camera`): lens,
/// exposure, shutter, sensor and views, without the "Prim" section
/// every prim has.
std::vector<PropertySection> CameraSchema();

/// A RenderSettings prim's sections: image, culling and level of
/// detail, and shadows.
std::vector<PropertySection> RenderSettingsSchema();

/// Whether prims of `type` are placed by their transform, for the
/// Transform section.
bool HasTransform(std::string_view type);

}  // namespace ernie::app
