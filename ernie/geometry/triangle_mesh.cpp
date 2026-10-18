// SPDX-License-Identifier: BSD-3-Clause
/// @file triangle_mesh.cpp
/// @brief Tessellating UsdGeom's implicit primitives, and fan triangulation.
#include "ernie/geometry/triangle_mesh.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace ernie {
namespace {

constexpr float kPi = std::numbers::pi_v<float>;

/// A point of a surface of revolution's profile: distance from the axis, height, and the outward
/// normal's radial and axial parts.
struct ProfilePoint {
    float radius;
    float height;
    float normalRadial;
    float normalAxial;
};

glm::vec2 Around(std::uint32_t slice, std::uint32_t slices) {
    const float phi = 2.0f * kPi * static_cast<float>(slice) / static_cast<float>(slices);
    return {std::cos(phi), std::sin(phi)};
}

class Builder {
  public:
    std::uint32_t Add(const glm::vec3& point, const glm::vec3& normal) {
        mesh_.points.push_back(point);
        mesh_.normals.push_back(glm::normalize(normal));
        return static_cast<std::uint32_t>(mesh_.points.size() - 1);
    }

    void Triangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        mesh_.indices.insert(mesh_.indices.end(), {a, b, c});
    }

    /// A surface of revolution around +z through `profile`, listed from the top down.
    void Revolve(const std::vector<ProfilePoint>& profile, std::uint32_t slices) {
        const auto first = static_cast<std::uint32_t>(mesh_.points.size());
        for (const ProfilePoint& p : profile) {
            for (std::uint32_t j = 0; j <= slices; ++j) {
                const glm::vec2 around = Around(j, slices);
                Add({p.radius * around, p.height}, {p.normalRadial * around, p.normalAxial});
            }
        }
        const std::uint32_t row = slices + 1;
        for (std::uint32_t i = 0; i + 1 < profile.size(); ++i) {
            for (std::uint32_t j = 0; j < slices; ++j) {
                const std::uint32_t a = first + i * row + j;
                const std::uint32_t c = a + row;
                Triangle(a, c, a + 1);
                Triangle(a + 1, c, c + 1);
            }
        }
    }

    /// A disc of `radius` at `height` facing +z (`up`) or -z.
    void Disc(float radius, float height, bool up, std::uint32_t slices) {
        const glm::vec3 normal(0.0f, 0.0f, up ? 1.0f : -1.0f);
        const std::uint32_t centre = Add({0.0f, 0.0f, height}, normal);
        const auto first = static_cast<std::uint32_t>(mesh_.points.size());
        for (std::uint32_t j = 0; j <= slices; ++j) {
            Add({radius * Around(j, slices), height}, normal);
        }
        for (std::uint32_t j = 0; j < slices; ++j) {
            if (up) {
                Triangle(centre, first + j, first + j + 1);
            } else {
                Triangle(centre, first + j + 1, first + j);
            }
        }
    }

    /// A quad with corners `centre` -+ `u` -+ `v`, facing u x v.
    void Quad(const glm::vec3& centre, const glm::vec3& u, const glm::vec3& v) {
        const glm::vec3 normal = glm::cross(u, v);
        const std::uint32_t a = Add(centre - u - v, normal);
        const std::uint32_t b = Add(centre + u - v, normal);
        const std::uint32_t c = Add(centre + u + v, normal);
        const std::uint32_t d = Add(centre - u + v, normal);
        Triangle(a, b, c);
        Triangle(a, c, d);
    }

    /// @return The mesh, turned from +z to `axis`.
    TriangleMesh Take(Axis axis) {
        if (axis != Axis::Z) {
            const auto turn = [axis](const glm::vec3& p) {
                return axis == Axis::X ? glm::vec3(p.z, p.x, p.y) : glm::vec3(p.y, p.z, p.x);
            };
            std::ranges::transform(mesh_.points, mesh_.points.begin(), turn);
            std::ranges::transform(mesh_.normals, mesh_.normals.begin(), turn);
        }
        return std::move(mesh_);
    }

  private:
    TriangleMesh mesh_;
};

void CheckPositive(float value, const char* what) {
    if (!(value > 0.0f) || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("A primitive's ") + what +
                                    " must be positive and finite");
    }
}

void CheckDivisions(std::uint32_t slices, std::uint32_t stacks = 2) {
    if (slices < 3) {
        throw std::invalid_argument("A primitive needs at least 3 slices");
    }
    if (stacks < 2) {
        throw std::invalid_argument("A primitive needs at least 2 stacks");
    }
}

/// Adds `count + 1` profile points around a circular arc of `radius` centred at `height`, from
/// polar angle `from` to `to` (0 at +z).
void Arc(std::vector<ProfilePoint>& profile, float radius, float height, float from, float to,
         std::uint32_t count) {
    for (std::uint32_t i = 0; i <= count; ++i) {
        const float theta = from + (to - from) * static_cast<float>(i) / static_cast<float>(count);
        // Poles land exactly on the axis, as sin(pi) is not 0 in float.
        const bool pole = theta == 0.0f || theta == kPi;
        const float s = pole ? 0.0f : std::sin(theta);
        const float c = pole ? (theta == 0.0f ? 1.0f : -1.0f) : std::cos(theta);
        profile.push_back({radius * s, height + radius * c, s, c});
    }
}

}  // namespace

std::optional<Axis> ParseAxis(std::string_view name) {
    if (name == "X") {
        return Axis::X;
    }
    if (name == "Y") {
        return Axis::Y;
    }
    if (name == "Z") {
        return Axis::Z;
    }
    return std::nullopt;
}

TriangleMesh BoxMesh(const glm::vec3& size) {
    CheckPositive(size.x, "width");
    CheckPositive(size.y, "depth");
    CheckPositive(size.z, "height");
    const glm::vec3 half = 0.5f * size;
    Builder builder;
    for (int axis = 0; axis < 3; ++axis) {
        for (const float sign : {1.0f, -1.0f}) {
            glm::vec3 normal(0.0f);
            normal[axis] = sign;
            glm::vec3 u(0.0f);
            glm::vec3 v(0.0f);
            u[(axis + 1) % 3] = sign * half[(axis + 1) % 3];
            v[(axis + 2) % 3] = half[(axis + 2) % 3];
            builder.Quad(normal * half[axis], u, v);
        }
    }
    return builder.Take(Axis::Z);
}

TriangleMesh SphereMesh(float radius, std::uint32_t slices, std::uint32_t stacks) {
    CheckPositive(radius, "radius");
    CheckDivisions(slices, stacks);
    std::vector<ProfilePoint> profile;
    Arc(profile, radius, 0.0f, 0.0f, kPi, stacks);
    Builder builder;
    builder.Revolve(profile, slices);
    return builder.Take(Axis::Z);
}

TriangleMesh CylinderMesh(float radius, float height, Axis axis, std::uint32_t slices) {
    CheckPositive(radius, "radius");
    CheckPositive(height, "height");
    CheckDivisions(slices);
    const float half = 0.5f * height;
    Builder builder;
    builder.Revolve({{radius, half, 1.0f, 0.0f}, {radius, -half, 1.0f, 0.0f}}, slices);
    builder.Disc(radius, half, true, slices);
    builder.Disc(radius, -half, false, slices);
    return builder.Take(axis);
}

TriangleMesh ConeMesh(float radius, float height, Axis axis, std::uint32_t slices) {
    CheckPositive(radius, "radius");
    CheckPositive(height, "height");
    CheckDivisions(slices);
    const float half = 0.5f * height;
    const float slant = std::hypot(radius, height);
    Builder builder;
    builder.Revolve({{0.0f, half, height / slant, radius / slant},
                     {radius, -half, height / slant, radius / slant}},
                    slices);
    builder.Disc(radius, -half, false, slices);
    return builder.Take(axis);
}

TriangleMesh CapsuleMesh(float radius, float height, Axis axis, std::uint32_t slices,
                         std::uint32_t stacks) {
    CheckPositive(radius, "radius");
    if (!(height >= 0.0f) || !std::isfinite(height)) {
        throw std::invalid_argument("A capsule's height must be finite and not negative");
    }
    CheckDivisions(slices, stacks);
    const std::uint32_t cap = std::max(stacks / 2, 1u);
    const float half = 0.5f * height;
    std::vector<ProfilePoint> profile;
    Arc(profile, radius, half, 0.0f, 0.5f * kPi, cap);
    Arc(profile, radius, -half, 0.5f * kPi, kPi, cap);
    Builder builder;
    builder.Revolve(profile, slices);
    return builder.Take(axis);
}

TriangleMesh PlaneMesh(float width, float length, Axis axis) {
    CheckPositive(width, "width");
    CheckPositive(length, "length");
    const float w = 0.5f * width;
    const float l = 0.5f * length;
    Builder builder;
    switch (axis) {
    case Axis::Z:
        builder.Quad({}, {w, 0.0f, 0.0f}, {0.0f, l, 0.0f});
        break;
    case Axis::Y:
        builder.Quad({}, {0.0f, 0.0f, l}, {w, 0.0f, 0.0f});
        break;
    case Axis::X:
        builder.Quad({}, {0.0f, l, 0.0f}, {0.0f, 0.0f, w});
        break;
    }
    return builder.Take(Axis::Z);
}

std::vector<std::uint32_t> TriangulateFaces(std::span<const std::int64_t> faceVertexCounts,
                                            std::span<const std::int64_t> faceVertexIndices,
                                            std::size_t points) {
    std::vector<std::uint32_t> triangles;
    std::size_t next = 0;
    for (const std::int64_t count : faceVertexCounts) {
        if (count < 3) {
            throw std::invalid_argument("Every face needs at least 3 vertices");
        }
        const auto size = static_cast<std::size_t>(count);
        if (next + size > faceVertexIndices.size()) {
            throw std::invalid_argument("faceVertexCounts add up to more than faceVertexIndices");
        }
        const std::span<const std::int64_t> face = faceVertexIndices.subspan(next, size);
        if (std::ranges::any_of(face, [&](std::int64_t index) {
                return index < 0 || static_cast<std::size_t>(index) >= points;
            })) {
            throw std::invalid_argument("faceVertexIndices has an index out of range");
        }
        for (std::size_t k = 1; k + 1 < size; ++k) {
            for (const std::int64_t corner : {face[0], face[k], face[k + 1]}) {
                triangles.push_back(static_cast<std::uint32_t>(corner));
            }
        }
        next += size;
    }
    if (next != faceVertexIndices.size()) {
        throw std::invalid_argument("faceVertexCounts add up to fewer than faceVertexIndices");
    }
    return triangles;
}

}  // namespace ernie
