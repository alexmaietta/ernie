// SPDX-License-Identifier: BSD-3-Clause
/// @file instancer.cpp
/// @brief The point instancer's registry entry and extent: its prototypes' bounds at each
/// instance, as UsdGeomPointInstancer computes it.
#include "ernie/scene/schemas/instancer.h"

#include "ernie/util/algorithm.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <optional>
#include <vector>

namespace ernie {
namespace {

bool OnHost(std::initializer_list<const Tensor*> tensors) {
    return std::ranges::none_of(tensors, [](const Tensor* t) { return IsGpu(t->Location().kind); });
}

/// @return A CPU array's floats, none when unset.
std::vector<float> FloatsOrNone(const Tensor& array, std::size_t components,
                                std::string_view what) {
    return array.Empty() ? std::vector<float>{} : CopyFloats(array, components, what);
}

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

void InstancerExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto prototypes = rows.Column<std::vector<Target>>(PointInstancer::kPrototypes);
    const auto indices = rows.Column<Tensor>(PointInstancer::kProtoIndices);
    const auto positions = rows.Column<Tensor>(PointInstancer::kPositions);
    const auto orientations = rows.Column<Tensor>(PointInstancer::kOrientations);
    const auto scales = rows.Column<Tensor>(PointInstancer::kScales);
    const auto order = rows.Column<Token>(PointInstancer::kQuaternionOrder);
    const auto authored = rows.Column<std::vector<float>>(Boundable::kExtent);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        Box& box = out[row - range.begin] = Box();
        if (!OnHost({&indices[row], &positions[row], &orientations[row], &scales[row]})) {
            box = AuthoredExtent(authored[row]);
            continue;
        }
        const std::vector<Box> protoBoxes = Map(prototypes[row], [&](const Target& target) {
            const std::optional<PrimId> prim = rows.store.Find(target.path);
            return prim ? rows.scene->BoundsInParent(*prim) : Box();
        });
        const std::vector<float> xyz = FloatsOrNone(positions[row], 3, "positions");
        const std::vector<float> rotations = FloatsOrNone(orientations[row], 4, "orientations");
        const std::vector<float> scaling = FloatsOrNone(scales[row], 3, "scales");
        const std::vector<std::int64_t> proto = indices[row].Empty()
                                                    ? std::vector<std::int64_t>{}
                                                    : CopyInts(indices[row], "protoIndices");
        const bool wFirst = rows.Text(order[row]) == "wxyz";
        const std::size_t count = xyz.size() / 3;
        for (std::size_t i = 0; i < count; ++i) {
            const std::int64_t p = i < proto.size() ? proto[i] : 0;
            if (p < 0 || static_cast<std::size_t>(p) >= protoBoxes.size()) {
                continue;
            }
            const auto part = [&](const std::vector<float>& values, std::size_t n) {
                return values.size() >= (i + 1) * n ? std::span(values).subspan(i * n, n)
                                                    : std::span<const float>();
            };
            box.Extend(protoBoxes[static_cast<std::size_t>(p)].Transformed(
                InstanceTransform(part(xyz, 3), part(rotations, 4), part(scaling, 3), wFirst)));
        }
    }
}

}  // namespace

ERNIE_DEFINE_SCHEMA(PointInstancer, tokens::kPointInstancer, &Boundable::Definition(), Concrete,
                    ComputedFromOtherPrims(InstancerExtent),
                    "Prototypes placed at instances, bounded by their bounds at each instance.",
                    ERNIE_POINT_INSTANCER_ATTRIBUTES, .openNamespaces = {tokens::kPrimvarsPrefix})

}  // namespace ernie
