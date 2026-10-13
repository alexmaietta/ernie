// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/value.h"
#include "ernie/util/numbers.h"
#include "ernie/util/tensor.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

class Scene;

/// The axis order of an Euler-angle rotate op: the rotation about the
/// first letter's axis applies first, about fixed (extrinsic) axes.
/// `ZYX` is the yaw, pitch, roll sequence of aerospace.
enum class RotationOrder : std::uint8_t {
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX
};

/// The rotate op of `order`, such as `xformOp:rotateXYZ`.
std::string_view RotateOp(RotationOrder order);

/// The rotation of Euler angles `degrees` (one per axis, as x, y, z,
/// whatever the order) applied in `order`: for XYZ, Rz * Ry * Rx with
/// column vectors.
glm::dmat3 EulerRotation(glm::dvec3 degrees, RotationOrder order);

/// Euler angles in degrees (x, y, z) whose `EulerRotation` in `order`
/// is `rotation`, a rotation matrix. The middle axis' angle is in
/// [-90, 90]; at gimbal lock (it at +-90) the last axis' angle is 0.
///
/// Reference: K. Shoemake, "Euler Angle Conversion", Graphics Gems
/// IV, Academic Press, 1994, pp. 222-229.
glm::dvec3 EulerAngles(const glm::dmat3& rotation,
                       RotationOrder order);

/// A part of `XformOps`, set or not, read as `std::optional<T>` is
/// (`if (part)`, `*part`).
template <typename T> class XformPart {
  public:
    XformPart() = default;
    XformPart(std::nullopt_t) {}
    XformPart(const T& value) : value_(value) {}

    explicit operator bool() const {
        return value_.has_value();
    }
    bool has_value() const {
        return value_.has_value();
    }
    const T& operator*() const {
        return *value_;
    }
    const T* operator->() const {
        return &*value_;
    }
    void reset() {
        value_.reset();
    }

  private:
    std::optional<T> value_;
};

/// A vector part of `XformOps`: a `glm::dvec3`, three numbers
/// (`{1, 2, 3}`), or any contiguous range of three (`std::array`,
/// `std::vector`, `std::span`), so that no glm type is needed.
class XformVector : public XformPart<glm::dvec3> {
  public:
    using XformPart::XformPart;
    XformVector() = default;
    XformVector(double x, double y, double z)
        : XformPart(glm::dvec3(x, y, z)) {}
    /// @throws std::invalid_argument unless `numbers` holds three.
    template <NumberRange R>
    XformVector(const R& numbers) : XformPart(VectorOf<3>(numbers)) {}
};

/// The rotation part of `XformOps` as a quaternion: a `glm::dquat`,
/// or its real part then its imaginary one, as OpenUSD's
/// `GfQuatd(real, imaginary)` (`{1, {0, 0, 0}}` turns nothing), never
/// four numbers alone (w may be first or last).
class XformRotation : public XformPart<glm::dquat> {
  public:
    using XformPart::XformPart;
    XformRotation() = default;
    XformRotation(double real, const glm::dvec3& imaginary)
        : XformPart(glm::dquat(real, imaginary.x, imaginary.y,
                               imaginary.z)) {}
    /// @throws std::invalid_argument unless `imaginary` holds three
    /// numbers.
    template <NumberRange R>
    XformRotation(double real, const R& imaginary)
        : XformRotation(real, VectorOf<3>(imaginary)) {}
};

/// A prim's local transform as translate, rotate (Euler angles or a
/// quaternion) and scale, the ops of USD's UsdGeomXformCommonAPI.
/// Unset parts are left out. Each part takes glm types or plain
/// numbers: `{.translate = {1, 2, 3}, .orient = {1, {0, 0, 0}}}`.
struct XformOps {
    XformVector translate;
    XformVector
        rotate;  ///< Euler angles in degrees, in `rotationOrder`.
    RotationOrder rotationOrder = RotationOrder::XYZ;
    XformRotation
        orient;  ///< A unit quaternion, in place of `rotate`.
    XformVector scale;
};

/// Sets the prim at `path` to `xform`: its set parts as
/// `xformOp:translate`, a rotate op or `xformOp:orient`, and
/// `xformOp:scale`, with `xformOpOrder` listing them in that order
/// (so points scale, then rotate, then move), after a leading
/// `!resetXformStack!` if it had one.
/// @throws std::invalid_argument if the prim does not exist, or
/// `xform` has both `rotate` and `orient`.
void SetXform(Scene& scene, const Path& path, const XformOps& xform);

/// The local transform of the prim at `path` from its transform ops
/// (`Scene::LocalTransform`): the product of the ops `xformOpOrder`
/// lists, or `xformOp:transform` without an order. An op with no
/// value is the identity. Op values are values or CPU tensors
/// (`Scene::Write`): 1, 3 or 4 numbers, quaternions as x, y, z, w,
/// and 4 x 4 matrices in the math convention (row-major, column
/// vectors, translation in the last column).
/// @throws std::invalid_argument for an op Ernie does not know, a
/// value of the wrong kind, or a tensor on a GPU.
///
/// Reference: Pixar, UsdGeomXformable,
/// https://openusd.org/release/api/class_usd_geom_xformable.html
glm::dmat4 ComposeXformOps(const Scene& scene, const Path& path);

/// Copies an op's tensor on a GPU (`attribute`'s, a prim's row of an
/// array) to the host.
using GpuOpReader = std::function<Tensor(std::string_view attribute,
                                         const Tensor& device)>;

/// `ComposeXformOps` reading ops on a GPU through `read` (none
/// throws, as above): for the few prims the CPU needs posed under
/// prims posed on the GPU, such as lights.
glm::dmat4 ComposeXformOps(const Scene& scene, const Path& path,
                           const GpuOpReader* read);

/// Whether the prim at `path` ignores its ancestors' transforms
/// (`!resetXformStack!`).
bool ResetsXformStack(const Scene& scene, const Path& path);

/// The attribute values that give the prim at `path` the local
/// transform `local`. A prim without `xformOpOrder` gets
/// `xformOp:transform`. One whose ops are translate, one rotate op or
/// orient, and scale (each optional, in that order) keeps them, set
/// to `local` decomposed, when they can represent it (no shear, and
/// no rotation or scale an absent op would need); otherwise its order
/// becomes `xformOp:transform` alone (after a leading
/// `!resetXformStack!`). For tools that move prims, such as the
/// viewer's gizmo, through `Scene::Set` or an edit history.
std::vector<std::pair<std::string, Value>>
LocalTransformEdits(const Scene& scene, const Path& path,
                    const glm::dmat4& local);

/// Whether `attribute` is `xformOpOrder` or a transform op
/// (`xformOp:*`).
bool IsXformAttribute(std::string_view attribute);

}  // namespace ernie
