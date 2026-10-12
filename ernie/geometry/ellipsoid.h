// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

namespace ernie {

/// The point of the ellipsoid of semi-axes `radii` (along x, y and z,
/// centred at the origin) whose normal passes through `point`: the
/// foot of the geodetic normal, where a point at that place's height
/// stands. Newton's method on the normal's multiplier, from the
/// radial point; points within a tenth of the radii of the centre get
/// the radial point.
///
/// References:
/// - P. Cozzi and K. Ring, "3D Engine Design for Virtual Globes", A
///   K Peters/CRC Press, 2011, section 2.3.
///   https://doi.org/10.1201/9781439865583
/// - CesiumJS, `scaleToGeodeticSurface`.
///   https://github.com/CesiumGS/cesium/blob/main/packages/engine/Source/Core/scaleToGeodeticSurface.js
glm::dvec3 GeodeticSurfacePoint(const glm::dvec3& point,
                                const glm::dvec3& radii);

/// The ellipsoid's outward unit normal at its point `surface`:
/// parallel to surface_i / radii_i^2.
glm::dvec3 GeodeticNormal(const glm::dvec3& surface,
                          const glm::dvec3& radii);

}  // namespace ernie
