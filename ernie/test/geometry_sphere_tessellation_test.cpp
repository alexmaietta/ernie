// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/sphere_tessellation.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using ernie::MeshData;
using ernie::SphereTessellation;

constexpr double kPi = std::numbers::pi;

glm::dvec3 Point(const MeshData& mesh, std::uint32_t i) {
    return {mesh.points[3 * i], mesh.points[3 * i + 1],
            mesh.points[3 * i + 2]};
}

/// The solid angle of the spherical triangle through unit vectors
/// a, b, c (Van Oosterom and Strackee 1983).
double SolidAngle(glm::dvec3 a, glm::dvec3 b, glm::dvec3 c) {
    a = glm::normalize(a);
    b = glm::normalize(b);
    c = glm::normalize(c);
    return 2.0 * std::atan2(std::abs(glm::dot(a, glm::cross(b, c))),
                            1.0 + glm::dot(a, b) + glm::dot(b, c) +
                                glm::dot(c, a));
}

/// Checks `mesh` is a closed sphere of `radius`: every vertex on it
/// with its normal outward, every edge shared by exactly two
/// triangles (V - E + F = 2), every triangle wound outward, its
/// triangles covering 4 pi steradians, and texture coordinates per
/// corner within the map.
void ExpectClosedSphere(const MeshData& mesh, float radius) {
    const std::size_t vertices = mesh.points.size() / 3;
    const std::size_t triangles = mesh.indices.size() / 3;
    ASSERT_EQ(mesh.normals.size(), mesh.points.size());
    ASSERT_EQ(mesh.uvs.size(), 2 * mesh.indices.size());
    for (std::uint32_t i = 0; i < vertices; ++i) {
        const glm::dvec3 p = Point(mesh, i);
        EXPECT_NEAR(glm::length(p), radius,
                    1e-5 * static_cast<double>(radius));
        const glm::dvec3 n(mesh.normals[3 * i],
                           mesh.normals[3 * i + 1],
                           mesh.normals[3 * i + 2]);
        EXPECT_GT(glm::dot(n, p),
                  0.999 * static_cast<double>(radius));
    }
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> edges;
    double solidAngle = 0.0;
    for (std::size_t t = 0; t < triangles; ++t) {
        const std::uint32_t* v = mesh.indices.data() + 3 * t;
        const glm::dvec3 a = Point(mesh, v[0]);
        const glm::dvec3 b = Point(mesh, v[1]);
        const glm::dvec3 c = Point(mesh, v[2]);
        EXPECT_GT(glm::dot(glm::cross(b - a, c - a), a + b + c), 0.0)
            << "triangle " << t;
        solidAngle += SolidAngle(a, b, c);
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t from = v[k];
            const std::uint32_t to = v[(k + 1) % 3];
            ++edges[{std::min(from, to), std::max(from, to)}];
        }
    }
    for (const auto& [edge, count] : edges) {
        EXPECT_EQ(count, 2) << edge.first << "-" << edge.second;
    }
    EXPECT_EQ(static_cast<long>(vertices) -
                  static_cast<long>(edges.size()) +
                  static_cast<long>(triangles),
              2);
    EXPECT_NEAR(solidAngle, 4.0 * kPi, 1e-6);
    for (std::size_t i = 0; i < mesh.uvs.size(); i += 2) {
        EXPECT_GE(mesh.uvs[i], 0.0f);
        EXPECT_LE(mesh.uvs[i], 1.5f);
        EXPECT_GE(mesh.uvs[i + 1], 0.0f);
        EXPECT_LE(mesh.uvs[i + 1], 1.0f);
    }
}

}  // namespace

TEST(SphereTessellationTest, EveryKindIsAClosedSphere) {
    for (const SphereTessellation kind :
         {SphereTessellation::Uv, SphereTessellation::Icosphere,
          SphereTessellation::CubeSphere,
          SphereTessellation::Healpix}) {
        SCOPED_TRACE(static_cast<int>(kind));
        ExpectClosedSphere(ernie::TessellatedSphere(2.5f, kind, 3),
                           2.5f);
    }
}

// Triangles: a UV sphere's rings less one a cell at each pole, an
// icosahedron's 20 faces in 4^level, a cube's 6 faces in level^2
// quads and HEALPix's 12 N_side^2 pixels, two triangles each.
TEST(SphereTessellationTest, CountsTheirTriangles) {
    const auto triangles = [](SphereTessellation kind,
                              std::uint32_t level) {
        return ernie::TessellatedSphere(1.0f, kind, level)
                   .indices.size() /
               3;
    };
    EXPECT_EQ(triangles(SphereTessellation::Uv, 8),
              4u * 8 * 8 - 4 * 8);
    EXPECT_EQ(triangles(SphereTessellation::Icosphere, 3), 20u * 64);
    EXPECT_EQ(triangles(SphereTessellation::CubeSphere, 5), 12u * 25);
    EXPECT_EQ(triangles(SphereTessellation::Healpix, 4), 24u * 16);
    EXPECT_EQ(triangles(SphereTessellation::Healpix, 0), 24u * 256);
}

// HEALPix pixels cover equal areas: each pair of triangles, one
// pixel, within a few percent of 4 pi / (12 N_side^2) (their edges
// are not great circles, which the triangles' are).
TEST(SphereTessellationTest, HealpixPixelsHaveEqualAreas) {
    constexpr std::uint32_t kNside = 8;
    const MeshData mesh = ernie::TessellatedSphere(
        1.0f, SphereTessellation::Healpix, kNside);
    const double pixel = 4.0 * kPi / (12.0 * kNside * kNside);
    double smallest = 1e30;
    double largest = 0.0;
    for (std::size_t t = 0; t + 1 < mesh.indices.size() / 3; t += 2) {
        double area = 0.0;
        for (std::size_t k = t; k < t + 2; ++k) {
            const std::uint32_t* v = mesh.indices.data() + 3 * k;
            area += SolidAngle(Point(mesh, v[0]), Point(mesh, v[1]),
                               Point(mesh, v[2]));
        }
        smallest = std::min(smallest, area / pixel);
        largest = std::max(largest, area / pixel);
    }
    std::printf(
        "HEALPix N_side %u pixel areas %.4f to %.4f of 4 pi / "
        "12 N^2\n",
        kNside, smallest, largest);
    EXPECT_GT(smallest, 0.9);
    EXPECT_LT(largest, 1.1);
}

TEST(SphereTessellationTest, NamesAndLimits) {
    EXPECT_EQ(ernie::SphereTessellationNamed("healpix"),
              SphereTessellation::Healpix);
    EXPECT_EQ(ernie::SphereTessellationNamed("cubeSphere"),
              SphereTessellation::CubeSphere);
    EXPECT_FALSE(
        ernie::SphereTessellationNamed("geodesic").has_value());
    EXPECT_THROW(ernie::TessellatedSphere(
                     1.0f, SphereTessellation::Icosphere, 10),
                 std::invalid_argument);
    EXPECT_THROW(
        ernie::TessellatedSphere(0.0f, SphereTessellation::Uv),
        std::invalid_argument);
}
