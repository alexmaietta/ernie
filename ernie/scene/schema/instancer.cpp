// SPDX-License-Identifier: BSD-3-Clause
/// @file instancer.cpp
/// @brief The point instancer's registry entry and extent: its prototypes' bounds at each
/// instance, as UsdGeomPointInstancer computes it.
#include "ernie/scene/schema/instancer.h"

#include "ernie/util/algorithm.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <optional>
#include <vector>

namespace ernie {
namespace {

/// An instance's transform: translate, then rotate, then scale innermost.
glm::dmat4 InstanceTransform(std::span<const float> position, std::span<const float> rotation,
                             std::span<const float> scale, bool wFirst) {
    glm::dmat4 m =
        glm::translate(glm::dmat4(1.0), glm::dvec3(position[0], position[1], position[2]));
    if (!rotation.empty()) {
        const glm::dquat q = wFirst
                                 ? glm::dquat(rotation[0], rotation[1], rotation[2], rotation[3])
                                 : glm::dquat(rotation[3], rotation[0], rotation[1], rotation[2]);
        m *= glm::dmat4(glm::mat3_cast(glm::normalize(q)));
    }
    if (!scale.empty()) {
        m = glm::scale(m, glm::dvec3(scale[0], scale[1], scale[2]));
    }
    return m;
}

/// One row's instances: its prototypes' boxes, each placed at the instances that use it.
struct Instances {
    std::vector<Box> prototypes;
    std::vector<std::int64_t> indices;
    std::vector<float> positions;
    std::vector<float> rotations;
    std::vector<float> scales;
    bool wFirst = false;

    Box Bounds() const {
        Box box;
        for (std::size_t i = 0; i < positions.size() / 3; ++i) {
            const std::int64_t p = i < indices.size() ? indices[i] : 0;
            if (p >= 0 && static_cast<std::size_t>(p) < prototypes.size()) {
                box.Extend(prototypes[static_cast<std::size_t>(p)].Transformed(InstanceTransform(
                    Part(positions, i, 3), Part(rotations, i, 4), Part(scales, i, 3), wFirst)));
            }
        }
        return box;
    }

  private:
    /// @return Instance i's `n` values, or none when the array is unset or too short.
    static std::span<const float> Part(const std::vector<float>& values, std::size_t i,
                                       std::size_t n) {
        return values.size() >= (i + 1) * n ? std::span(values).subspan(i * n, n)
                                            : std::span<const float>();
    }
};

void InstancerExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto prototypes = rows.Column<std::vector<Target>>(PointInstancer::kPrototypes);
    const auto indices = rows.Column<Tensor>(PointInstancer::kProtoIndices);
    const auto positions = rows.Column<Tensor>(PointInstancer::kPositions);
    const auto orientations = rows.Column<Tensor>(PointInstancer::kOrientations);
    const auto scales = rows.Column<Tensor>(PointInstancer::kScales);
    const auto order = rows.Column<Token>(PointInstancer::kQuaternionOrder);
    const auto authored = rows.Column<std::vector<float>>(Boundable::kExtent);
    ComputeRows(range, out, [&](std::uint32_t row) {
        const bool onHost = indices[row].OnHost() && positions[row].OnHost() &&
                            orientations[row].OnHost() && scales[row].OnHost();
        if (!onHost) {
            return AuthoredExtent(authored[row]);
        }
        const Instances instances{
            .prototypes = Map(prototypes[row],
                              [&](const Target& target) {
                                  const std::optional<PrimId> prim = rows.Resolve(target);
                                  return prim ? rows.scene->BoundsInParent(*prim) : Box();
                              }),
            .indices = CopyInts(indices[row], "protoIndices"),
            .positions = CopyFloats(positions[row], 3, "positions"),
            .rotations = CopyFloats(orientations[row], 4, "orientations"),
            .scales = CopyFloats(scales[row], 3, "scales"),
            .wFirst = rows.Text(order[row]) == "wxyz",
        };
        return instances.Bounds();
    });
}

}  // namespace

ERNIE_DEFINE_SCHEMA(PointInstancer, Concrete, ComputedFromOtherPrims(InstancerExtent),
                    ERNIE_POINT_INSTANCER_ATTRIBUTES, .openNamespaces = {tokens::kPrimvarsPrefix})

}  // namespace ernie
