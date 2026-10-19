// SPDX-License-Identifier: BSD-3-Clause
/// @file products.h
/// @brief The render products a step renders, read from UsdRender's prims: each one's camera, image
/// size, lens and vars.
#pragma once

#include "ernie/render/rt/camera_pass.h"
#include "ernie/scene/scene.h"

#include <glm/vec2.hpp>

#include <string>
#include <vector>

namespace ernie {

/// @brief A product as a step renders it.
struct ProductSpec {
    /// The RenderProduct prim, or the settings' own path for their default product.
    Path product;
    PrimId camera;
    /// The batch world its camera is in; a camera in every world sees the first.
    std::uint32_t world = 0;
    glm::uvec2 size{1};
    Lens lens;
    /// Its vars' source names, in order.
    std::vector<std::string> vars;
};

/// @return The products of the scene's first RenderSettings: its `products`, or one default
/// product of the settings' own, which holds LdrColor. A product authoring a camera, resolution,
/// pixel aspect ratio or conform policy overrides its settings'; its vars are its `orderedVars`'
/// source names, or LdrColor alone.
/// @throws std::invalid_argument when there are no settings, a product names no camera, a path is
/// no product or var, or a var or lens is one the Minimal tier does not render.
std::vector<ProductSpec> ReadProducts(const Scene& scene);

/// @return A camera prim's lens for an image of `size` pixels, its aperture conformed to the
/// image's aspect by `policy`, as UsdGeomCamera's `conformApertureToAspectRatio` describes.
/// @throws std::invalid_argument for a lens model not rendered yet.
Lens ReadLens(const Scene& scene, PrimId camera, glm::uvec2 size, double pixelAspectRatio,
              std::string_view policy);

}  // namespace ernie
