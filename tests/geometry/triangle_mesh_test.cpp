// SPDX-License-Identifier: BSD-3-Clause
/// @file triangle_mesh_test.cpp
/// @brief Tests of the implicit primitives' meshes, closed, sized and wound outward, and of fan
/// triangulation.
#include "ernie/geometry/triangle_mesh.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace ernie {
namespace {

constexpr double kPi = std::numbers::pi;

glm::dvec3 Point(const TriangleMesh& mesh, std::uint32_t i) {
    return mesh.points[i];
}

double Area(const TriangleMesh& mesh) {
    double area = 0.0;
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
        const glm::dvec3 a = Point(mesh, mesh.indices[t]);
        area += 0.5 * glm::length(glm::cross(Point(mesh, mesh.indices[t + 1]) - a,
                                             Point(mesh, mesh.indices[t + 2]) - a));
    }
    return area;
}

/// @return The volume by the divergence theorem: positive only when every triangle winds outward.
double SignedVolume(const TriangleMesh& mesh) {
    double volume = 0.0;
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
        volume +=
            glm::dot(Point(mesh, mesh.indices[t]), glm::cross(Point(mesh, mesh.indices[t + 1]),
                                                              Point(mesh, mesh.indices[t + 2]))) /
            6.0;
    }
    return volume;
}

std::pair<glm::dvec3, glm::dvec3> Bounds(const TriangleMesh& mesh) {
    glm::dvec3 low(1e30);
    glm::dvec3 high(-1e30);
    for (const glm::vec3& point : mesh.points) {
        low = glm::min(low, glm::dvec3(point));
        high = glm::max(high, glm::dvec3(point));
    }
    return {low, high};
}

/// Expects unit normals, and every triangle that is not degenerate facing its vertices' normals.
void ExpectConsistent(const TriangleMesh& mesh) {
    ASSERT_EQ(mesh.points.size(), mesh.normals.size());
    ASSERT_EQ(mesh.indices.size() % 3, 0u);
    for (const glm::vec3& normal : mesh.normals) {
        ASSERT_NEAR(glm::length(normal), 1.0f, 1e-5f);
    }
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
        const std::uint32_t a = mesh.indices[t];
        const std::uint32_t b = mesh.indices[t + 1];
        const std::uint32_t c = mesh.indices[t + 2];
        ASSERT_LT(std::max({a, b, c}), mesh.points.size());
        const glm::dvec3 face =
            glm::cross(Point(mesh, b) - Point(mesh, a), Point(mesh, c) - Point(mesh, a));
        if (glm::length(face) < 1e-9) {
            continue;
        }
        const glm::dvec3 normals = mesh.normals[a] + mesh.normals[b] + mesh.normals[c];
        ASSERT_GT(glm::dot(face, normals), 0.0) << "triangle " << t / 3;
    }
}

TEST(Primitives, BoxesAreClosedAndSized) {
    const TriangleMesh box = BoxMesh({2.0f, 4.0f, 6.0f});
    ExpectConsistent(box);
    EXPECT_EQ(box.points.size(), 24u) << "four corners a face, for flat normals";
    EXPECT_EQ(box.indices.size() / 3, 12u);
    EXPECT_NEAR(Area(box), 2.0 * (8.0 + 24.0 + 12.0), 1e-4);
    EXPECT_NEAR(SignedVolume(box), 48.0, 1e-4);
    const auto [low, high] = Bounds(box);
    EXPECT_EQ(low, glm::dvec3(-1.0, -2.0, -3.0));
    EXPECT_EQ(high, glm::dvec3(1.0, 2.0, 3.0));
}

TEST(Primitives, SpheresApproachTheirAreaAndVolume) {
    const TriangleMesh sphere = SphereMesh(2.0f, 96, 48);
    ExpectConsistent(sphere);
    for (const glm::vec3& point : sphere.points) {
        ASSERT_NEAR(glm::length(point), 2.0f, 1e-5f);
    }
    EXPECT_NEAR(Area(sphere), 16.0 * kPi, 0.01 * 16.0 * kPi);
    EXPECT_NEAR(SignedVolume(sphere), 32.0 / 3.0 * kPi, 0.01 * 32.0 / 3.0 * kPi);
}

TEST(Primitives, CylindersConesAndCapsulesAreClosed) {
    const double r = 1.5;
    const double h = 4.0;
    const TriangleMesh cylinder = CylinderMesh(1.5f, 4.0f, Axis::Z, 128);
    ExpectConsistent(cylinder);
    EXPECT_NEAR(Area(cylinder), 2.0 * kPi * r * h + 2.0 * kPi * r * r, 0.5);
    EXPECT_NEAR(SignedVolume(cylinder), kPi * r * r * h, 0.01 * kPi * r * r * h);

    const TriangleMesh cone = ConeMesh(1.5f, 4.0f, Axis::Z, 128);
    ExpectConsistent(cone);
    EXPECT_NEAR(Area(cone), kPi * r * std::hypot(r, h) + kPi * r * r, 0.3);
    EXPECT_NEAR(SignedVolume(cone), kPi * r * r * h / 3.0, 0.01 * kPi * r * r * h / 3.0);
    EXPECT_NEAR(Bounds(cone).second.z, 2.0, 1e-6) << "apex at +height / 2";

    const TriangleMesh capsule = CapsuleMesh(1.5f, 4.0f, Axis::Z, 128, 64);
    ExpectConsistent(capsule);
    const double volume = kPi * r * r * h + 4.0 / 3.0 * kPi * r * r * r;
    EXPECT_NEAR(Area(capsule), 2.0 * kPi * r * h + 4.0 * kPi * r * r, 0.7);
    EXPECT_NEAR(SignedVolume(capsule), volume, 0.01 * volume);
    EXPECT_NEAR(Bounds(capsule).second.z, 0.5 * h + r, 1e-6) << "caps beyond the height";
    EXPECT_NO_THROW(CapsuleMesh(1.0f, 0.0f)) << "a capsule of height 0 is a sphere";
}

TEST(Primitives, AxesTurnShapesWithoutMirroringThem) {
    for (const Axis axis : {Axis::X, Axis::Y}) {
        const TriangleMesh cone = ConeMesh(1.0f, 4.0f, axis);
        ExpectConsistent(cone);
        EXPECT_GT(SignedVolume(cone), 0.0) << "still wound outward";
        const glm::dvec3 high = Bounds(cone).second;
        EXPECT_NEAR(axis == Axis::X ? high.x : high.y, 2.0, 1e-6) << "apex along the axis";
    }
    const TriangleMesh x = PlaneMesh(2.0f, 6.0f, Axis::X);
    ExpectConsistent(x);
    EXPECT_EQ(x.normals[0], glm::vec3(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(Bounds(x).second.y, 3.0, 1e-6) << "length along y";
    EXPECT_NEAR(Bounds(x).second.z, 1.0, 1e-6) << "width along z";
    const TriangleMesh y = PlaneMesh(2.0f, 6.0f, Axis::Y);
    EXPECT_EQ(y.normals[0], glm::vec3(0.0f, 1.0f, 0.0f));
    EXPECT_NEAR(Bounds(y).second.z, 3.0, 1e-6);
    const TriangleMesh z = PlaneMesh(2.0f, 6.0f);
    ExpectConsistent(z);
    EXPECT_NEAR(Area(z), 12.0, 1e-5);

    EXPECT_EQ(ParseAxis("Y"), Axis::Y);
    EXPECT_FALSE(ParseAxis("W"));
}

TEST(Primitives, RejectInvalidSizes) {
    EXPECT_THROW(BoxMesh({1.0f, 0.0f, 1.0f}), std::invalid_argument);
    EXPECT_THROW(SphereMesh(-1.0f), std::invalid_argument);
    EXPECT_THROW(SphereMesh(1.0f, 2), std::invalid_argument);
    EXPECT_THROW(CylinderMesh(1.0f, std::nanf("")), std::invalid_argument);
    EXPECT_THROW(CapsuleMesh(1.0f, -1.0f), std::invalid_argument);
    EXPECT_THROW(PlaneMesh(1.0f, 0.0f), std::invalid_argument);
}

TEST(TriangulateFaces, FansPolygonsAndKeepsTriangles) {
    const std::vector<std::int64_t> counts{3, 4, 5};
    const std::vector<std::int64_t> indices{0, 1, 2, 0, 2, 3, 4, 1, 2, 3, 4, 5};
    EXPECT_EQ(TriangulateFaces(counts, indices, 6),
              (std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3, 0, 3, 4, 1, 2, 3, 1, 3, 4, 1, 4, 5}));
    EXPECT_TRUE(TriangulateFaces({}, {}, 0).empty());
}

TEST(TriangulateFaces, RejectsMalformedTopology) {
    const std::vector<std::int64_t> three{3};
    const std::vector<std::int64_t> triangle{0, 1, 2};
    EXPECT_THROW(TriangulateFaces(std::vector<std::int64_t>{2}, std::vector<std::int64_t>{0, 1}, 3),
                 std::invalid_argument);
    EXPECT_THROW(TriangulateFaces(std::vector<std::int64_t>{4}, triangle, 3),
                 std::invalid_argument);
    EXPECT_THROW(TriangulateFaces(three, std::vector<std::int64_t>{0, 1, 2, 3}, 4),
                 std::invalid_argument);
    EXPECT_THROW(TriangulateFaces(three, triangle, 2), std::invalid_argument);
    EXPECT_THROW(TriangulateFaces(three, std::vector<std::int64_t>{0, -1, 2}, 3),
                 std::invalid_argument);
}

}  // namespace
}  // namespace ernie
