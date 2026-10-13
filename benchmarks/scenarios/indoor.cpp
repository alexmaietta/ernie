// SPDX-License-Identifier: BSD-3-Clause
#include "indoor.h"

#include "ernie/geometry/primitives.h"
#include "ernie/scene/geometry_schemas.h"

#include <array>
#include <cmath>

namespace ernie::bench::indoor {

namespace {

constexpr float kDoorway = 1.0f;
constexpr float kWallThickness = 0.1f;

}  // namespace

Layout MakeLayout(std::uint64_t seed) {
    Random random(seed * 7u + 101u);
    const int columns = random.Int(2, 3);
    constexpr int kRows = 2;
    std::vector<float> xs{0.0f};
    std::vector<float> ys{0.0f};
    for (int i = 0; i < columns; ++i) {
        xs.push_back(xs.back() + random.Uniform(3.0f, 5.0f));
    }
    for (int j = 0; j < kRows; ++j) {
        ys.push_back(ys.back() + random.Uniform(3.0f, 5.0f));
    }
    // Centred on the origin, where the worlds overlap.
    const glm::vec2 centre(xs.back() / 2.0f, ys.back() / 2.0f);
    Layout layout;
    for (int j = 0; j < kRows; ++j) {
        for (int i = 0; i < columns; ++i) {
            const auto ui = static_cast<std::size_t>(i);
            const auto uj = static_cast<std::size_t>(j);
            layout.rooms.push_back(
                {glm::vec2(xs[ui], ys[uj]) - centre,
                 glm::vec2(xs[ui + 1], ys[uj + 1]) - centre});
        }
    }
    // Each grid line's segments, a doorway in the middle of the
    // inner ones.
    const auto segment = [&](glm::vec2 a, glm::vec2 b, bool inner) {
        a -= centre;
        b -= centre;
        if (!inner) {
            layout.walls.emplace_back(a, b);
            return;
        }
        const glm::vec2 mid = (a + b) / 2.0f;
        const glm::vec2 half =
            glm::normalize(b - a) * (kDoorway / 2.0f);
        layout.walls.emplace_back(a, mid - half);
        layout.walls.emplace_back(mid + half, b);
    };
    for (std::size_t i = 0; i < xs.size(); ++i) {
        for (std::size_t j = 0; j + 1 < ys.size(); ++j) {
            segment({xs[i], ys[j]}, {xs[i], ys[j + 1]},
                    i > 0 && i + 1 < xs.size());
        }
    }
    for (std::size_t j = 0; j < ys.size(); ++j) {
        for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
            segment({xs[i], ys[j]}, {xs[i + 1], ys[j]},
                    j > 0 && j + 1 < ys.size());
        }
    }
    return layout;
}

Shell DefineShell(Scene& scene, const Path& root,
                  std::vector<Path>& prototypes) {
    constexpr std::array<std::array<glm::vec3, 2>, 4> kFloors = {{
        {{{0.55f, 0.38f, 0.22f}, {0.5f, 0.34f, 0.2f}}},
        {{{0.75f, 0.72f, 0.68f}, {0.3f, 0.3f, 0.32f}}},
        {{{0.4f, 0.42f, 0.5f}, {0.4f, 0.42f, 0.5f}}},
        {{{0.65f, 0.6f, 0.5f}, {0.6f, 0.55f, 0.45f}}},
    }};
    Shell shell{.firstFloor = static_cast<int>(prototypes.size()),
                .floors = static_cast<int>(kFloors.size())};
    for (std::size_t i = 0; i < kFloors.size(); ++i) {
        const Path look("/Looks/Floor" + std::to_string(i));
        DefineTexturedMaterial(
            scene, look,
            CheckerTexture(256, 8, kFloors[i][0], kFloors[i][1], 0.2f,
                           static_cast<std::uint32_t>(10 + i)),
            256);
        MeshData floor = PlaneMesh(1.0f, 1.0f);
        ProjectUvs(floor, 4.0f);
        prototypes.push_back(root.Child("Floor" + std::to_string(i)));
        DefineMesh(scene, prototypes.back(), floor)
            .SetMaterialBinding(look);
    }
    DefineTexturedMaterial(
        scene, "/Looks/Plaster",
        CheckerTexture(256, 1, {0.85f, 0.83f, 0.78f},
                       {0.85f, 0.83f, 0.78f}, 0.08f, 20),
        256);
    MeshData wall = BoxMesh(glm::vec3(1.0f));
    ProjectUvs(wall, 1.0f);
    shell.wall = static_cast<int>(prototypes.size());
    prototypes.push_back(root.Child("Wall"));
    DefineMesh(scene, prototypes.back(), wall)
        .SetMaterialBinding("/Looks/Plaster");
    return shell;
}

void PlaceShell(const Layout& layout, const Shell& shell, int world,
                std::vector<Placement>& placements) {
    for (std::size_t r = 0; r < layout.rooms.size(); ++r) {
        const Room& room = layout.rooms[r];
        const glm::vec2 centre = room.Centre();
        const glm::vec2 size = room.max - room.min;
        placements.push_back(
            {.proto = shell.firstFloor +
                      static_cast<int>(r) % shell.floors,
             .world = world,
             .position = {centre, 0.0f},
             .scale = {size, 1.0f}});
    }
    for (const auto& [a, b] : layout.walls) {
        const glm::vec2 along = b - a;
        const float length = glm::length(along);
        if (length < 0.05f) {
            continue;
        }
        placements.push_back(
            {.proto = shell.wall,
             .world = world,
             .position = {(a + b) / 2.0f, kWallHeight / 2.0f},
             .orientation = Yaw(std::atan2(along.y, along.x)),
             .scale = {length + kWallThickness, kWallThickness,
                       kWallHeight}});
    }
}

glm::vec3 PointIn(const Room& room, float margin, float z,
                  Random& random) {
    return {random.Uniform(room.min.x + margin, room.max.x - margin),
            random.Uniform(room.min.y + margin, room.max.y - margin),
            z};
}

glm::dmat4 ViewIn(const Layout& layout, int world, int view,
                  int step) {
    Random random(static_cast<std::uint64_t>(world) * 7907u +
                  static_cast<std::uint64_t>(view) * 13u + 1u);
    const Room& room =
        layout.rooms[static_cast<std::size_t>(world + view) %
                     layout.rooms.size()];
    const glm::dvec3 eye(PointIn(room, 0.8f, 1.4f, random));
    const double heading =
        static_cast<double>(random.Uniform(0.0f, 6.283f)) +
        0.03 * step;
    return LookAt(eye, eye + glm::dvec3(std::cos(heading),
                                        std::sin(heading), -0.25));
}

}  // namespace ernie::bench::indoor
