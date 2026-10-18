// SPDX-License-Identifier: BSD-3-Clause
/// @file mesh.cpp
/// @brief Point-based geometry's registry entries and extents, as UsdGeom computes them.
#include "ernie/scene/schemas/mesh.h"

#include <optional>
#include <vector>

namespace ernie {
namespace {

/// @return The box of a CPU tensor's points, each grown by half its width (one per point, one for
/// all, or none); none for an empty tensor or one on a device.
std::optional<Box> BoundPoints(const Tensor& points, const Tensor& widths) {
    if (points.Empty() || IsGpu(points.Location().kind) ||
        (!widths.Empty() && IsGpu(widths.Location().kind))) {
        return std::nullopt;
    }
    const Tensor xyz = points.Contiguous();
    const std::vector<float> diameters =
        widths.Empty() ? std::vector<float>{} : CopyFloats(widths, "widths");
    const auto count = static_cast<std::size_t>(xyz.Elements() / 3);
    const auto half = [&](std::size_t i) {
        return diameters.size() == count ? 0.5 * diameters[i]
               : diameters.size() == 1   ? 0.5 * diameters[0]
                                         : 0.0;
    };
    Box box;
    const auto extend = [&]<typename T>(const T* data) {
        for (std::size_t i = 0; i < count; ++i) {
            const glm::dvec3 point(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
            box.Extend(Box::Around(point, glm::dvec3(half(i))));
        }
    };
    if (xyz.Type() == DType::Float64) {
        extend(xyz.As<double>());
    } else {
        extend(xyz.As<float>());
    }
    return box;
}

/// @return The authored `extent`, read while the points aren't on the host.
Box AuthoredExtent(const std::vector<float>& extent) {
    if (extent.size() != 6) {
        return {};
    }
    return {{extent[0], extent[1], extent[2]}, {extent[3], extent[4], extent[5]}};
}

/// Meshes from their points; point clouds grown by their widths.
template <bool kWidths>
void PointBasedExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto points = rows.Column<Tensor>(PointBased::kPoints);
    const auto widths = kWidths ? rows.Column<Tensor>(Points::kWidths) : std::span<const Tensor>();
    const auto authored = rows.Column<std::vector<float>>(Boundable::kExtent);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        const std::optional<Box> box = BoundPoints(points[row], kWidths ? widths[row] : Tensor());
        out[row - range.begin] = box ? *box : AuthoredExtent(authored[row]);
    }
}

}  // namespace

ERNIE_DEFINE_SCHEMA(PointBased, "PointBased", &Gprim::Definition(), Abstract, kInheritExtent,
                    "Geometry from an array of points.", ERNIE_POINT_BASED_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Mesh, tokens::kMesh, &PointBased::Definition(), Concrete,
                    ComputedExtent(PointBasedExtent<false>),
                    "A polygon mesh, or the control mesh of a subdivision surface.",
                    ERNIE_MESH_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Points, tokens::kPoints, &PointBased::Definition(), Concrete,
                    ComputedExtent(PointBasedExtent<true>),
                    "A point cloud of spheres, each its own width.", ERNIE_POINTS_ATTRIBUTES)

}  // namespace ernie
