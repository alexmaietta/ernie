// SPDX-License-Identifier: BSD-3-Clause
/// @file xform.h
/// @brief UsdGeomXformable's transform ops: composing them, and authoring them from parts.
#pragma once

#include "ernie/export.h"
#include "ernie/scene/scene.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// The order Euler rotation ops (`rotateXYZ` ...) turn about their axes, first axis first.
enum class RotationOrder : std::uint8_t { XYZ, XZY, YXZ, YZX, ZXY, ZYX };

/// @return The op name of an order, as `xformOp:rotateXYZ`.
ERNIE_API std::string_view RotateOp(RotationOrder order);
/// @return The rotation of Euler angles in degrees, the order's first axis turning first.
ERNIE_API glm::dmat3 EulerRotation(const glm::dvec3& degrees, RotationOrder order);
/// @return Angles in degrees that `EulerRotation` turns back into `rotation`.
ERNIE_API glm::dvec3 EulerAngles(const glm::dmat3& rotation, RotationOrder order);

/// @brief A transform as translate, rotate or orient, and scale ops; unset parts are left out.
struct XformOps {
    std::optional<glm::dvec3> translate;
    /// Euler angles in degrees, in `rotationOrder`.
    std::optional<glm::dvec3> rotate;
    RotationOrder rotationOrder = RotationOrder::XYZ;
    std::optional<glm::dquat> orient;
    std::optional<glm::dvec3> scale;
};

/// @return A camera's transform at `eye` looking at `target`, `up` upward in the view: cameras
/// look down their -z with +y up, as USD's do.
/// @throws std::invalid_argument when `target` is `eye`, or `up` is along the view.
ERNIE_API glm::dmat4 LookAtTransform(const glm::dvec3& eye, const glm::dvec3& target,
                                     const glm::dvec3& up = glm::dvec3(0.0, 0.0, 1.0));
/// @return A distant light's transform shining from `towardLight`, the direction to its source:
/// distant lights shine down their -z, so their +z is turned to it.
/// @throws std::invalid_argument for a zero direction.
ERNIE_API glm::dmat4 DistantLightTransform(const glm::dvec3& towardLight);

/// @brief Authors a prim's ops and their order, keeping a reset stack.
/// @throws std::invalid_argument if there is no prim at `path`, or `xform` has both rotate and
/// orient.
ERNIE_API void SetXform(Scene& scene, const Path& path, const XformOps& xform);

/// @return The prim's local transform from its ops in `xformOpOrder`; without an order,
/// `xformOp:transform` alone.
/// @throws std::invalid_argument for an op Ernie does not know or a value of the wrong kind.
ERNIE_API glm::dmat4 ComposeXformOps(const Scene& scene, PrimId prim);
/// @return Whether the prim's transform ignores its ancestors' (`!resetXformStack!`).
ERNIE_API bool ResetsXformStack(const Scene& scene, PrimId prim);

/// The transform ops Ernie knows; the Euler rotations in `RotationOrder`'s order.
enum class XformOpKind : std::uint8_t {
    Translate,
    Scale,
    RotateX,
    RotateY,
    RotateZ,
    RotateXYZ,
    RotateXZY,
    RotateYXZ,
    RotateYZX,
    RotateZXY,
    RotateZYX,
    Orient,
    Transform,
};

/// @return The kind of an op attribute, as `xformOp:rotateXYZ:arm` is a `RotateXYZ`; none for an
/// attribute that is no op Ernie knows.
ERNIE_API std::optional<XformOpKind> ParseXformOpKind(std::string_view attribute);

/// @brief One op of a prim's transform: its attribute, kind and whether the order inverts it.
struct XformOp {
    std::string attribute;
    XformOpKind kind = XformOpKind::Transform;
    bool invert = false;
    /// The attribute's token, when interned as the order was read, to read its value by.
    std::optional<Token> token;
    bool operator==(const XformOp&) const = default;
};
/// @return The prim's ops, outermost first, as `ComposeXformOps` multiplies them: those of
/// `xformOpOrder` without its reset token, or without an order `xformOp:transform` alone.
/// @throws std::invalid_argument for an entry that is no xformOp, or an op Ernie does not know.
ERNIE_API std::vector<XformOp> XformOpsInOrder(const Scene& scene, PrimId prim);
/// @return One op's matrix, inverted if the order says so; identity when it has no value.
/// @throws std::invalid_argument for a value of the wrong kind.
ERNIE_API glm::dmat4 XformOpMatrix(const Scene& scene, PrimId prim, const XformOp& op);

}  // namespace ernie
