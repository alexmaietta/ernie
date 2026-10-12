// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/sphere_tessellation.h"

#include "ernie/geometry/cube_sphere_lod.h"

#include <glm/glm.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ernie {

namespace {

constexpr double kPi = std::numbers::pi;

/// Unit directions and triangles of a tessellation, before vertices
/// on shared edges are merged.
struct Directions {
    std::vector<glm::dvec3> points;
    std::vector<std::array<std::uint32_t, 3>> triangles;

    std::uint32_t Add(const glm::dvec3& direction) {
        points.push_back(glm::normalize(direction));
        return static_cast<std::uint32_t>(points.size() - 1);
    }

    /// A grid of (columns + 1) x (rows + 1) points from `at(col,
    /// row)`, two triangles a cell.
    template <typename At>
    void Grid(std::uint32_t columns, std::uint32_t rows, At at) {
        const auto first = static_cast<std::uint32_t>(points.size());
        for (std::uint32_t row = 0; row <= rows; ++row) {
            for (std::uint32_t col = 0; col <= columns; ++col) {
                Add(at(col, row));
            }
        }
        const std::uint32_t stride = columns + 1;
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t col = 0; col < columns; ++col) {
                const std::uint32_t a = first + row * stride + col;
                triangles.push_back({a, a + 1, a + stride + 1});
                triangles.push_back({a, a + stride + 1, a + stride});
            }
        }
    }
};

void Uv(Directions& out, std::uint32_t stacks) {
    out.Grid(2 * stacks, stacks,
             [&](std::uint32_t col, std::uint32_t row) {
                 const double longitude =
                     2.0 * kPi * col / (2.0 * stacks) - kPi;
                 const double latitude =
                     kPi * row / stacks - 0.5 * kPi;
                 return glm::dvec3(
                     std::cos(latitude) * std::cos(longitude),
                     std::cos(latitude) * std::sin(longitude),
                     std::sin(latitude));
             });
}

void Icosphere(Directions& out, std::uint32_t subdivisions) {
    // The icosahedron's vertices, (0, +-1, +-g) cyclically with g the
    // golden ratio, and its faces.
    const double g = 0.5 * (1.0 + std::sqrt(5.0));
    const std::array<glm::dvec3, 12> corners = {{{-1, g, 0},
                                                 {1, g, 0},
                                                 {-1, -g, 0},
                                                 {1, -g, 0},
                                                 {0, -1, g},
                                                 {0, 1, g},
                                                 {0, -1, -g},
                                                 {0, 1, -g},
                                                 {g, 0, -1},
                                                 {g, 0, 1},
                                                 {-g, 0, -1},
                                                 {-g, 0, 1}}};
    constexpr std::array<std::array<int, 3>, 20> faces = {
        {{0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
         {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
         {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
         {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}}};
    // Each face as a triangular grid of n = 2^subdivisions segments a
    // side: n^2 = 4^subdivisions triangles.
    const std::uint32_t n = 1u << subdivisions;
    for (const auto& face : faces) {
        const glm::dvec3 a =
            corners[static_cast<std::size_t>(face[0])];
        const glm::dvec3 b =
            corners[static_cast<std::size_t>(face[1])];
        const glm::dvec3 c =
            corners[static_cast<std::size_t>(face[2])];
        const auto first =
            static_cast<std::uint32_t>(out.points.size());
        // Row i (0 at a) holds i + 1 points from the a-b edge to a-c.
        const auto index = [&](std::uint32_t i, std::uint32_t j) {
            return first + i * (i + 1) / 2 + j;
        };
        for (std::uint32_t i = 0; i <= n; ++i) {
            for (std::uint32_t j = 0; j <= i; ++j) {
                const double u = static_cast<double>(i - j) / n;
                const double v = static_cast<double>(j) / n;
                out.Add(a + u * (b - a) + v * (c - a));
            }
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            for (std::uint32_t j = 0; j <= i; ++j) {
                out.triangles.push_back({index(i, j), index(i + 1, j),
                                         index(i + 1, j + 1)});
                if (j < i) {
                    out.triangles.push_back({index(i, j),
                                             index(i + 1, j + 1),
                                             index(i, j + 1)});
                }
            }
        }
    }
}

void CubeSphere(Directions& out, std::uint32_t cells) {
    for (std::uint32_t face = 0; face < CubeSphereLod::kFaces;
         ++face) {
        out.Grid(cells, cells,
                 [&](std::uint32_t col, std::uint32_t row) {
                     return CubeSphereLod::FaceDirection(
                         face, 2.0 * col / cells - 1.0,
                         2.0 * row / cells - 1.0);
                 });
    }
}

/// The point (`x`, `y`) in [0, 1]^2 of HEALPix base pixel `face`, as
/// healpix_base's `xyf2loc`: its ring coordinate jr = jrll - x - y
/// gives the polar caps (z = +-(1 - nr^2 / 3)) and the equatorial
/// belt (z linear in jr), and its longitude follows jpll.
glm::dvec3 HealpixPoint(std::uint32_t face, double x, double y) {
    constexpr std::array<int, 12> jrll = {2, 2, 2, 2, 3, 3,
                                          3, 3, 4, 4, 4, 4};
    constexpr std::array<int, 12> jpll = {1, 3, 5, 7, 0, 2,
                                          4, 6, 1, 3, 5, 7};
    const double jr = jrll[face] - x - y;
    double nr = 1.0;
    double z = 0.0;
    if (jr < 1.0) {
        nr = jr;
        z = 1.0 - nr * nr / 3.0;
    } else if (jr > 3.0) {
        nr = 4.0 - jr;
        z = nr * nr / 3.0 - 1.0;
    } else {
        z = (2.0 - jr) * 2.0 / 3.0;
    }
    double around = jpll[face] * nr + x - y;
    if (around < 0.0) {
        around += 8.0;
    }
    if (around >= 8.0) {
        around -= 8.0;
    }
    const double phi = nr < 1e-15 ? 0.0 : 0.25 * kPi * around / nr;
    const double ring = std::sqrt(std::max(1.0 - z * z, 0.0));
    return {ring * std::cos(phi), ring * std::sin(phi), z};
}

void Healpix(Directions& out, std::uint32_t nside) {
    for (std::uint32_t face = 0; face < 12; ++face) {
        out.Grid(
            nside, nside, [&](std::uint32_t col, std::uint32_t row) {
                return HealpixPoint(face,
                                    static_cast<double>(col) / nside,
                                    static_cast<double>(row) / nside);
            });
    }
}

/// Grid cells of 1e-9 for merging: points on shared edges come from
/// different faces' formulas and agree to rounding.
struct Cell {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;
    bool operator==(const Cell&) const = default;
};

struct CellHash {
    std::size_t operator()(const Cell& cell) const {
        return std::hash<std::int64_t>()(cell.x * 73856093 ^
                                         cell.y * 19349663 ^
                                         cell.z * 83492791);
    }
};

/// `directions` with points closer than 1e-8 merged, degenerate
/// triangles dropped and the rest wound outward, as a mesh of
/// `radius`.
MeshData Merged(const Directions& directions, float radius) {
    constexpr double kScale = 1e9;
    std::unordered_map<Cell, std::uint32_t, CellHash> cells;
    std::vector<glm::dvec3> kept;
    std::vector<std::uint32_t> remap(directions.points.size());
    for (std::size_t i = 0; i < directions.points.size(); ++i) {
        const glm::dvec3& p = directions.points[i];
        const Cell cell{std::llround(p.x * kScale),
                        std::llround(p.y * kScale),
                        std::llround(p.z * kScale)};
        std::optional<std::uint32_t> found;
        for (std::int64_t dx = -1; dx <= 1 && !found; ++dx) {
            for (std::int64_t dy = -1; dy <= 1 && !found; ++dy) {
                for (std::int64_t dz = -1; dz <= 1 && !found; ++dz) {
                    const auto near = cells.find(
                        {cell.x + dx, cell.y + dy, cell.z + dz});
                    if (near != cells.end() &&
                        glm::length(kept[near->second] - p) < 1e-8) {
                        found = near->second;
                    }
                }
            }
        }
        if (!found) {
            found = static_cast<std::uint32_t>(kept.size());
            kept.push_back(p);
            cells.emplace(cell, *found);
        }
        remap[i] = *found;
    }

    MeshData mesh;
    for (const glm::dvec3& p : kept) {
        const glm::vec3 point =
            static_cast<float>(radius) * glm::vec3(p);
        mesh.points.insert(mesh.points.end(),
                           {point.x, point.y, point.z});
        const glm::vec3 normal(p);
        mesh.normals.insert(mesh.normals.end(),
                            {normal.x, normal.y, normal.z});
    }
    for (const auto& triangle : directions.triangles) {
        std::array<std::uint32_t, 3> t = {remap[triangle[0]],
                                          remap[triangle[1]],
                                          remap[triangle[2]]};
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) {
            continue;
        }
        const glm::dvec3& a = kept[t[0]];
        const glm::dvec3& b = kept[t[1]];
        const glm::dvec3& c = kept[t[2]];
        if (glm::dot(glm::cross(b - a, c - a), a + b + c) < 0.0) {
            std::swap(t[1], t[2]);
        }
        // Equirectangular coordinates, joined across the
        // antimeridian, a pole taking the mean longitude of the other
        // corners.
        std::array<glm::dvec2, 3> st;
        std::array<bool, 3> pole{};
        for (std::size_t k = 0; k < 3; ++k) {
            const glm::dvec3& p = kept[t[k]];
            const double horizontal = std::hypot(p.x, p.y);
            pole[k] = horizontal < 1e-12;
            st[k] = {std::atan2(p.y, p.x) / (2.0 * kPi) + 0.5,
                     std::atan2(p.z, horizontal) / kPi + 0.5};
        }
        double lowest = 1.0;
        double highest = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            if (!pole[k]) {
                lowest = std::min(lowest, st[k].x);
                highest = std::max(highest, st[k].x);
            }
        }
        if (highest - lowest > 0.5) {
            for (std::size_t k = 0; k < 3; ++k) {
                if (!pole[k] && st[k].x < 0.5) {
                    st[k].x += 1.0;
                }
            }
        }
        for (std::size_t k = 0; k < 3; ++k) {
            if (pole[k]) {
                double sum = 0.0;
                int count = 0;
                for (std::size_t m = 0; m < 3; ++m) {
                    if (!pole[m]) {
                        sum += st[m].x;
                        ++count;
                    }
                }
                st[k].x = count > 0 ? sum / count : 0.5;
            }
        }
        mesh.indices.insert(mesh.indices.end(), t.begin(), t.end());
        for (const glm::dvec2& corner : st) {
            mesh.uvs.insert(mesh.uvs.end(),
                            {static_cast<float>(corner.x),
                             static_cast<float>(corner.y)});
        }
    }
    return mesh;
}

}  // namespace

std::optional<SphereTessellation>
SphereTessellationNamed(std::string_view name) {
    if (name == "uv") {
        return SphereTessellation::Uv;
    }
    if (name == "icosphere") {
        return SphereTessellation::Icosphere;
    }
    if (name == "cubeSphere") {
        return SphereTessellation::CubeSphere;
    }
    if (name == "healpix") {
        return SphereTessellation::Healpix;
    }
    return std::nullopt;
}

std::uint32_t DefaultSphereLevel(SphereTessellation tessellation) {
    switch (tessellation) {
    case SphereTessellation::Uv:
        return 24;
    case SphereTessellation::Icosphere:
        return 4;
    case SphereTessellation::CubeSphere:
    case SphereTessellation::Healpix:
        return 16;
    }
    return 16;
}

MeshData TessellatedSphere(float radius,
                           SphereTessellation tessellation,
                           std::uint32_t level) {
    if (!(radius > 0.0f) || !std::isfinite(radius)) {
        throw std::invalid_argument(
            "A sphere's radius must be positive and finite");
    }
    if (level == 0) {
        level = DefaultSphereLevel(tessellation);
    }
    const auto limit = [&](std::uint32_t lowest,
                           std::uint32_t highest, const char* what) {
        if (level < lowest || level > highest) {
            throw std::invalid_argument(
                std::string("A ") + what +
                " sphere's level must be " + std::to_string(lowest) +
                " to " + std::to_string(highest) + ", not " +
                std::to_string(level));
        }
    };
    Directions directions;
    switch (tessellation) {
    case SphereTessellation::Uv:
        limit(2, 4096, "UV");
        Uv(directions, level);
        break;
    case SphereTessellation::Icosphere:
        limit(1, 9, "icosphere");
        Icosphere(directions, level);
        break;
    case SphereTessellation::CubeSphere:
        limit(1, 1024, "cube");
        CubeSphere(directions, level);
        break;
    case SphereTessellation::Healpix:
        limit(1, 1024, "HEALPix");
        Healpix(directions, level);
        break;
    }
    return Merged(directions, radius);
}

}  // namespace ernie
