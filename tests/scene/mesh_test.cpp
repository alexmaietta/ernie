// SPDX-License-Identifier: BSD-3-Clause
/// @file mesh_test.cpp
/// @brief Tests of point-based schemas, array attributes and their extents.
#include "ernie/scene/schemas/mesh.h"

#include "tests/common.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace ernie {
namespace {

using test::Floats;
using test::Ints;

Tensor Triangle() {
    return Floats({0, 0, 0, 2, 0, 0, 0, 3, -1}, {3, 3});
}

TEST(Mesh, ArraysAreSharedAndCheckedForTypeAndShape) {
    Scene scene;
    Mesh mesh = Mesh::Define(scene, "/Tri");
    EXPECT_TRUE(mesh.GetPoints().Empty()) << "unauthored";
    const Tensor points = Triangle();
    mesh.SetPoints(points);
    EXPECT_EQ(mesh.GetPoints().Data(), points.Data()) << "shared, not copied";
    EXPECT_NE(scene.Read(mesh.Id(), PointBased::kPoints), nullptr);
    mesh.SetFaceVertexCounts(Ints({3}));

    EXPECT_THROW(mesh.SetPoints(Floats({0, 0, 0, 1}, {2, 2})), std::invalid_argument)
        << "points are threes";
    EXPECT_THROW(mesh.SetFaceVertexCounts(Floats({3}, {1})), std::invalid_argument)
        << "counts are integers";
    EXPECT_THROW(scene.Write(mesh.Id(), Sphere::kRadius, points), std::invalid_argument)
        << "not an attribute of a mesh";
    EXPECT_THROW(scene.Write(mesh.Id(), Gprim::kDisplayOpacity, points), std::invalid_argument)
        << "not an array";
    scene.Write(mesh.Id(), "primvars:st", Floats({0, 0, 1, 0, 0, 1}, {3, 2}));
    EXPECT_NE(scene.Read(mesh.Id(), "primvars:st"), nullptr) << "primvars take any name";
    EXPECT_EQ(mesh.GetSubdivisionScheme(), "catmullClark") << "USD's fallback";
    EXPECT_EQ(mesh.GetSubdivisionRefineLevel(), 1);
    EXPECT_THROW(mesh.SetSubdivisionRefineLevel(9), std::invalid_argument);
}

TEST(Mesh, ExtentsBoundPointsAndPointCloudsTheirWidths) {
    Scene scene;
    Mesh mesh = Mesh::Define(scene, "/Tri");
    EXPECT_TRUE(scene.Extent(mesh.Id())->Empty()) << "no points";
    mesh.SetPoints(Triangle());
    const Box tri = *scene.Extent(mesh.Id());
    EXPECT_EQ(tri.min, glm::dvec3(0.0, 0.0, -1.0));
    EXPECT_EQ(tri.max, glm::dvec3(2.0, 3.0, 0.0));

    Points cloud = Points::Define(scene, "/Cloud");
    cloud.SetPoints(Floats({0, 0, 0, 10, 0, 0}, {2, 3}));
    cloud.SetWidths(Floats({2, 4}, {2}));
    const Box spheres = *scene.Extent(cloud.Id());
    EXPECT_EQ(spheres.min, glm::dvec3(-1.0, -2.0, -2.0));
    EXPECT_EQ(spheres.max, glm::dvec3(12.0, 2.0, 2.0));
    cloud.SetWidths(Floats({1}, {1}));
    EXPECT_EQ(scene.Extent(cloud.Id())->max, glm::dvec3(10.5, 0.5, 0.5)) << "one width for all";
}

TEST(Curves, ExtentsBoundControlPointsGrownByTheWidestWidth) {
    Scene scene;
    BasisCurves hair = BasisCurves::Define(scene, "/Hair");
    EXPECT_EQ(hair.GetType(), "cubic") << "USD's fallback";
    hair.SetCurveVertexCounts(Ints({4}));
    hair.SetPoints(Floats({0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 1}, {4, 3}));
    hair.SetWidths(Floats({0.2f, 0.4f}, {2}));
    const Box box = *scene.Extent(hair.Id());
    EXPECT_NEAR(box.min.x, -0.2, 1e-6) << "varying widths grow by the widest";
    EXPECT_NEAR(box.max.z, 1.2, 1e-6);

    NurbsCurves nurbs = NurbsCurves::Define(scene, "/Nurbs");
    nurbs.SetPoints(Floats({0, 0, 0, 1, 1, 1}, {2, 3}));
    EXPECT_EQ(scene.Extent(nurbs.Id())->max, glm::dvec3(1.0)) << "no widths";
    EXPECT_THROW(HermiteCurves::Define(scene, "/Hermite").SetTangents(Floats({1, 0}, {1, 2})),
                 std::invalid_argument)
        << "tangents are threes";
}

TEST(Mesh, PointsTheHostCannotReadFallBackToTheAuthoredExtent) {
    Scene scene;
    Mesh mesh = Mesh::Define(scene, "/Tri");
    mesh.SetExtent({-1, -1, -1, 1, 1, 1});
    EXPECT_EQ(scene.Extent(mesh.Id())->max, glm::dvec3(1.0)) << "authored, no points yet";
    mesh.SetPoints(Triangle());
    EXPECT_EQ(scene.Extent(mesh.Id())->max, glm::dvec3(2.0, 3.0, 0.0)) << "computed wins";
}

}  // namespace
}  // namespace ernie
