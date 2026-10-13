// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"

#include <string_view>

namespace ernie::app {

class Viewer;

/// A Camera prim's sections in the Properties panel beyond its
/// attributes: its exposure value and the viewport's meter, each
/// channel's quantum efficiency over wavelength and as binned into
/// its bands, and the sensor chain the viewport runs through it, to
/// edit (bypass, reorder, remove, add and the filters' inputs; a
/// camera without a product of its own customizes the default chain)
/// and to view at any stage, the viewport then looking through the
/// camera and showing the image right after that filter. With a
/// `filter`, only the sections whose titles match it.
void DrawCameraSections(Viewer& viewer, const Path& camera,
                        float labelWidth, std::string_view filter);

}  // namespace ernie::app
