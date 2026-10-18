// SPDX-License-Identifier: BSD-3-Clause
/// @file xform.h
/// @brief UsdGeomXformable's transform ops: composing them, and authoring them from parts.
#pragma once

#include "ernie/scene/scene.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace ernie {

/// The order Euler rotation ops (`rotateXYZ` ...) turn about their axes, first axis first.
enum class RotationOrder : std::uint8_t { XYZ, XZY, YXZ, YZX, ZXY, ZYX };

/// @return The op name of an order, as `xformOp:rotateXYZ`.
std::string_view RotateOp(RotationOrder order);
/// @return The rotation of Euler angles in degrees, the order's first axis turning first.
glm::dmat3 EulerRotation(const glm::dvec3& degrees, RotationOrder order);
/// @return Angles in degrees that `EulerRotation` turns back into `rotation`.
glm::dvec3 EulerAngles(const glm::dmat3& rotation, RotationOrder order);

/// @brief A transform as translate, rotate or orient, and scale ops; unset parts are left out.
struct XformOps {
    std::optional<glm::dvec3> translate;
    /// Euler angles in degrees, in `rotationOrder`.
    std::optional<glm::dvec3> rotate;
    RotationOrder rotationOrder = RotationOrder::XYZ;
    std::optional<glm::dquat> orient;
    std::optional<glm::dvec3> scale;
};

/// @brief Authors a prim's ops and their order, keeping a reset stack.
/// @throws std::invalid_argument if there is no prim at `path`, or `xform` has both rotate and
/// orient.
void SetXform(Scene& scene, const Path& path, const XformOps& xform);

/// @return The prim's local transform from its ops in `xformOpOrder`; without an order,
/// `xformOp:transform` alone.
/// @throws std::invalid_argument for an op Ernie does not know or a value of the wrong kind.
glm::dmat4 ComposeXformOps(const Scene& scene, PrimId prim);
/// @return Whether the prim's transform ignores its ancestors' (`!resetXformStack!`).
bool ResetsXformStack(const Scene& scene, PrimId prim);

}  // namespace ernie
