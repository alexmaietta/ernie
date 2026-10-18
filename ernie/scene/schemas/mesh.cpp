// SPDX-License-Identifier: BSD-3-Clause
/// @file mesh.cpp
/// @brief Point-based geometry's registry entries and extents, as UsdGeom computes them.
#include "ernie/scene/schemas/mesh.h"

#include <algorithm>
#include <optional>
#include <vector>

namespace ernie {
namespace {

/// @return The box of a CPU tensor's points, each grown by half its width when there is one per
/// point, else by half the widest (as UsdGeomCurves does); none for an empty tensor or one on a
/// device.
std::optional<Box> BoundPoints(const Tensor& points, const Tensor& widths) {
    if (points.Empty() || IsGpu(points.Location().kind) ||
        (!widths.Empty() && IsGpu(widths.Location().kind))) {
        return std::nullopt;
    }
    const Tensor xyz = points.Contiguous();
    const std::vector<float> diameters =
        widths.Empty() ? std::vector<float>{} : CopyFloats(widths, "widths");
    const auto count = static_cast<std::size_t>(xyz.Elements() / 3);
    const double widest = diameters.empty() ? 0.0 : *std::ranges::max_element(diameters);
    const auto half = [&](std::size_t i) {
        return 0.5 * (diameters.size() == count ? diameters[i] : widest);
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

/// Meshes from their points; point clouds and curves grown by their widths. Curves are bounded by
/// their control points, which hold every basis but catmullRom and Hermite closely, as in USD.
template <bool kWidths>
void PointBasedExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto points = rows.Column<Tensor>(PointBased::kPoints);
    const auto widths = kWidths ? rows.Column<Tensor>(Points::kWidths) : std::span<const Tensor>();
    const auto authored = rows.Column<std::vector<float>>(Boundable::kExtent);
    ComputeRows(range, out, [&](std::uint32_t row) {
        const std::optional<Box> box = BoundPoints(points[row], kWidths ? widths[row] : Tensor());
        return box ? *box : AuthoredExtent(authored[row]);
    });
}

}  // namespace

ERNIE_DEFINE_SCHEMA(PointBased, Abstract, kInheritExtent, ERNIE_POINT_BASED_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Mesh, Concrete, ComputedExtent(PointBasedExtent<false>), ERNIE_MESH_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Points, Concrete, ComputedExtent(PointBasedExtent<true>),
                    ERNIE_POINTS_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Curves, Abstract, ComputedExtent(PointBasedExtent<true>),
                    ERNIE_CURVES_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(BasisCurves, Concrete, kInheritExtent, ERNIE_BASIS_CURVES_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(NurbsCurves, Concrete, kInheritExtent, ERNIE_NURBS_CURVES_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(HermiteCurves, Concrete, kInheritExtent, ERNIE_HERMITE_CURVES_ATTRIBUTES)

}  // namespace ernie
