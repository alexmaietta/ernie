// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/app/prim_schemas.h"
#include "ernie/scene/path.h"

#include <span>
#include <string_view>

namespace ernie::app {

class Viewer;

/// A Camera prim's sections in the Properties panel beyond its other
/// attributes: its exposure (the Exposure section of `sections`, its
/// schema's, which changes its products' outputs, with its exposure
/// value and the viewport's meter), each channel's quantum efficiency
/// over wavelength and as binned into its bands, its render products
/// (their settings and vars, and new ones), and the sensor chain of
/// one of them (the viewport's unless chosen) to edit (bypass,
/// reorder, remove, add and the filters' inputs; a camera without a
/// product customizes the default chain) and to view at any stage,
/// the viewport then looking through the camera and showing the image
/// right after that filter. With a `filter`, only the sections whose
/// titles match it.
void DrawCameraSections(Viewer& viewer, const Path& camera,
                        std::span<const PropertySection> sections,
                        float labelWidth, std::string_view filter);

}  // namespace ernie::app
