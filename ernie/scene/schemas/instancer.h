// SPDX-License-Identifier: BSD-3-Clause
/// @file instancer.h
/// @brief UsdGeomPointInstancer: prototypes placed at many instances.
///
/// References:
/// - Pixar, UsdGeomPointInstancer,
///   https://openusd.org/release/api/class_usd_geom_point_instancer.html
#pragma once

#include "ernie/scene/schemas/geom.h"

#include <array>
#include <string_view>

// clang-format off
#define ERNIE_INSTANCE_VECTORS(X, Name, usdName, text)                                             \
    X(Tensor, Name, usdName, Tensor(), text, .group = "Instances", .components = 3,                \
      .dirty = Dirty::Points)

#define ERNIE_POINT_INSTANCER_ATTRIBUTES(X)                                                        \
    X(std::vector<Path>, Prototypes, "prototypes", std::vector<Path>{},                            \
      "The prims instanced, indexed by the proto indices.", .group = "Instances",                  \
      .dirty = Dirty::Topology)                                                                    \
    X(Tensor, ProtoIndices, "protoIndices", Tensor(), "The prototype of each instance.",           \
      .group = "Instances", .integer = true, .dirty = Dirty::Topology)                             \
    ERNIE_INSTANCE_VECTORS(X, Positions, "positions",                                              \
                           "(n, 3) positions in the instancer's space; the number of instances.")  \
    X(Tensor, Orientations, "orientations", Tensor(), "(n, 4) rotations as quaternions.",          \
      .group = "Instances", .components = 4, .dirty = Dirty::Points)                               \
    ERNIE_INSTANCE_VECTORS(X, Scales, "scales", "(n, 3) scales.")                                  \
    ERNIE_INSTANCE_VECTORS(X, Velocities, "velocities", "(n, 3) metres per second.")               \
    ERNIE_INSTANCE_VECTORS(X, Accelerations, "accelerations", "(n, 3) metres per second squared.") \
    ERNIE_INSTANCE_VECTORS(X, AngularVelocities, "angularVelocities", "(n, 3) degrees per second.") \
    X(Tensor, Ids, "ids", Tensor(), "int64 ids that follow instances across time samples.",        \
      .group = "Instances", .integer = true, .dirty = Dirty::Topology)                             \
    X(Tensor, InvisibleIds, "invisibleIds", Tensor(), "The ids of instances to hide.",             \
      .group = "Instances", .integer = true, .dirty = Dirty::Visibility)                            \
    X(Choice<QuaternionOrder>, QuaternionOrder, "ernie:quaternionOrder", QuaternionOrder::Xyzw,    \
      "The component order of the orientations: xyzw (USD, Warp, PhysX) or wxyz (Isaac Lab, "      \
      "MuJoCo).",                                                                                  \
      .group = "Instances", .dirty = Dirty::Points)                                                \
    X(Tensor, Worlds, "ernie:worlds", Tensor(), "int (n,): each instance's world.",               \
      .group = "Instances", .integer = true)
// clang-format on

namespace ernie {

/// The component order of quaternions in arrays.
enum class QuaternionOrder { Xyzw, Wxyz };

template <> struct ChoiceTokens<QuaternionOrder> {
    static constexpr std::array<std::string_view, 2> kTokens{"xyzw", "wxyz"};
};

ERNIE_CONCRETE_SCHEMA(PointInstancer, Boundable, ERNIE_POINT_INSTANCER_ATTRIBUTES,
                      "Prototypes placed at instances, bounded by their bounds at each instance.");

}  // namespace ernie
