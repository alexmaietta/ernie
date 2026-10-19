// SPDX-License-Identifier: BSD-3-Clause
/// @file hide_seek.cpp
/// @brief Madrona's Hide & Seek benchmark scene rebuilt in Ernie, to compare batch rendering on the
/// same GPU: each world ±18 m of rooms walled off with doors, 3 to 9 boxes and 2 ramps, and 3
/// hiders and 3 seekers each looking out through a 100° camera 0.5 m above it, over one shared
/// ground. Meshes come from madrona_benchmark's data directory (`ERNIE_MADRONA_DATA`), each asset's
/// arrays shared by every instance, as Madrona's assets are. Madrona counts worlds a second, each
/// world six views; this reports both.
#include "benchmarks/bench.h"
#include "ernie/device/context.h"
#include "ernie/render/renderer.h"
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/mesh.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ernie::bench {
namespace {

constexpr int kAgents = 6;
constexpr double kBounds = 18.0;
constexpr double kPi = 3.14159265358979323846;

/// A mesh's arrays, shared by every prim drawing it.
struct MeshArrays {
    Tensor points;
    Tensor counts;
    Tensor indices;
};

/// @return An OBJ file's positions and faces; texture and normal indices are skipped.
MeshArrays ReadObj(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    std::vector<float> points;
    std::vector<std::int32_t> counts;
    std::vector<std::int32_t> indices;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream words(line);
        std::string kind;
        words >> kind;
        if (kind == "v") {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            words >> x >> y >> z;
            points.insert(points.end(), {x, y, z});
        } else if (kind == "f") {
            std::int32_t corners = 0;
            for (std::string corner; words >> corner; ++corners) {
                indices.push_back(std::stoi(corner.substr(0, corner.find('/'))) - 1);
            }
            counts.push_back(corners);
        }
    }
    const auto length = [](const auto& values) { return static_cast<std::int64_t>(values.size()); };
    return {.points = Tensor::Copy<float>(points, {length(points) / 3, 3}),
            .counts = Tensor::Copy<std::int32_t>(counts, {length(counts)}),
            .indices = Tensor::Copy<std::int32_t>(indices, {length(indices)})};
}

struct Assets {
    MeshArrays agent;
    MeshArrays cube;
    MeshArrays wall;
    MeshArrays ramp;
    MeshArrays elongated;

    explicit Assets(const std::filesystem::path& data)
        : agent(ReadObj(data / "agent_render.obj")), cube(ReadObj(data / "cube_render.obj")),
          wall(ReadObj(data / "wall_render.obj")), ramp(ReadObj(data / "ramp_render.obj")),
          elongated(ReadObj(data / "elongated_render.obj")) {}
};

/// An axis-aligned wall segment of the unit square, from `low` to `high`.
struct Wall {
    glm::dvec2 low;
    glm::dvec2 high;

    bool Horizontal() const {
        return low.y == high.y;
    }
};

/// @return Madrona's walls: the unit square's sides, then cross walls each with a door and doors
/// cut into walls, as many of each as its generator draws.
std::vector<Wall> MakeWalls(std::mt19937& random) {
    std::vector<Wall> walls{{{0, 0}, {1, 0}}, {{0, 0}, {0, 1}}, {{0, 1}, {1, 1}}, {{1, 0}, {1, 1}}};
    std::uniform_real_distribution<double> unit(0.2, 0.8);
    const auto cutDoor = [&](std::size_t index) {
        const Wall wall = walls[index];
        const double at = unit(random);
        constexpr double kDoor = 0.08;
        const glm::dvec2 along = wall.high - wall.low;
        walls[index] = {wall.low, wall.low + (at - kDoor) * along};
        walls.push_back({wall.low + (at + kDoor) * along, wall.high});
    };
    const int connects = 1 + std::uniform_int_distribution<int>(0, 5)(random);
    const int doors = 4 + std::uniform_int_distribution<int>(0, 2)(random);
    for (int i = 0; i < connects; ++i) {
        const double at = unit(random);
        walls.push_back(i % 2 == 0 ? Wall{{at, 0}, {at, 1}} : Wall{{0, at}, {1, at}});
        cutDoor(walls.size() - 1);
    }
    for (int i = 0; i < doors; ++i) {
        cutDoor(std::uniform_int_distribution<std::size_t>(0, walls.size() - 1)(random));
    }
    return walls;
}

/// @brief Defines a mesh prim drawing `arrays`, shared, at `transform` under its world.
void DefineMesh(Scene& scene, const std::string& path, const MeshArrays& arrays,
                const glm::dmat4& transform, const glm::dvec3& colour) {
    Mesh::Define(scene, path);
    const PrimId prim = scene.Require(path);
    scene.Write(prim, Mesh::kPoints, arrays.points);
    scene.Write(prim, Mesh::kFaceVertexCounts, arrays.counts);
    scene.Write(prim, Mesh::kFaceVertexIndices, arrays.indices);
    scene.Set(prim, Gprim::kDisplayColor, colour);
    scene.Set(prim, tokens::kTransform, transform);
}

glm::dmat4 Placed(const glm::dvec3& at, double yaw, const glm::dvec3& scale = glm::dvec3(1.0)) {
    return glm::scale(glm::rotate(glm::translate(glm::dmat4(1.0), at), yaw, {0, 0, 1}), scale);
}

/// @brief Defines one world: its walls, boxes, ramps and agents, and a camera and product an agent.
void BuildWorld(Scene& scene, const Assets& assets, int world, std::mt19937& random,
                glm::uvec2 size, const std::vector<Path>& vars, std::vector<Path>& products) {
    const std::string root = "/World" + std::to_string(world);
    Xform::Define(scene, root).SetWorld(world);
    std::uniform_real_distribution<double> spot(-kBounds, kBounds);
    std::uniform_real_distribution<double> turn(0.0, kPi);
    int index = 0;
    for (const Wall& wall : MakeWalls(random)) {
        const glm::dvec2 low = 2.0 * kBounds * wall.low - kBounds;
        const glm::dvec2 high = 2.0 * kBounds * wall.high - kBounds;
        const glm::dvec2 centre = 0.5 * (low + high);
        const glm::dvec3 scale = wall.Horizontal() ? glm::dvec3(high.x - centre.x, 0.2, 1.0)
                                                   : glm::dvec3(0.2, high.y - centre.y, 1.0);
        DefineMesh(scene, root + "/Wall" + std::to_string(index++), assets.wall,
                   Placed({centre, 0.0}, 0.0, scale), {0.6, 0.6, 0.6});
    }
    const int boxes = std::uniform_int_distribution<int>(3, 9)(random);
    const int elongated = std::uniform_int_distribution<int>(3, boxes)(random);
    for (int box = 0; box < boxes; ++box) {
        DefineMesh(scene, root + "/Box" + std::to_string(box),
                   box < elongated ? assets.elongated : assets.cube,
                   Placed({spot(random), spot(random), 1.0}, turn(random)), {0.8, 0.5, 0.2});
    }
    for (int ramp = 0; ramp < 2; ++ramp) {
        DefineMesh(scene, root + "/Ramp" + std::to_string(ramp), assets.ramp,
                   Placed({spot(random), spot(random), 1.0}, turn(random)), {0.3, 0.6, 0.3});
    }
    for (int agent = 0; agent < kAgents; ++agent) {
        const std::string name = root + "/Agent" + std::to_string(agent);
        const glm::dvec3 at(spot(random), spot(random), 1.0);
        const double yaw = turn(random);
        DefineMesh(scene, name, assets.agent, Placed(at, yaw),
                   agent < 3 ? glm::dvec3(0.2, 0.4, 0.9) : glm::dvec3(0.9, 0.2, 0.2));
        // Looking along the agent's +y at its eye, 0.5 m up, as Madrona's view does.
        const glm::dvec3 eye = at + glm::dvec3(0.0, 0.0, 0.5);
        const glm::dvec3 ahead(-std::sin(yaw), std::cos(yaw), 0.0);
        const std::string camera = name + "Camera";
        // A square film 100° high at the default focal length, its near plane past the agent's
        // own body, which Madrona's views do not see either.
        Camera::Define(scene, camera)
            .SetVerticalAperture(2.0 * 0.05 * std::tan(0.5 * 100.0 * kPi / 180.0))
            .SetHorizontalAperture(2.0 * 0.05 * std::tan(0.5 * 100.0 * kPi / 180.0))
            .SetClippingRange({1.2, 1e6});
        scene.Set(camera, tokens::kTransform, LookAtTransform(eye, eye + ahead, {0, 0, 1}));
        products.emplace_back("/Render/View" + std::to_string(world) + "_" + std::to_string(agent));
        RenderProduct::Define(scene, products.back())
            .SetCamera(camera)
            .SetResolution(glm::dvec2(size))
            .SetOrderedVars(vars);
    }
}

void BuildHideSeek(Scene& scene, const Assets& assets, int worlds, glm::uvec2 size) {
    GroundPlane::Define(scene, "/Ground");
    const std::vector<Path> vars{Path("/Render/Vars/Colour"), Path("/Render/Vars/Depth")};
    RenderVar::Define(scene, vars[0]).SetSourceName(std::string(render_vars::kLdrColor));
    RenderVar::Define(scene, vars[1]).SetSourceName(std::string(render_vars::kDistanceToCamera));
    std::mt19937 random(5);
    std::vector<Path> products;
    for (int world = 0; world < worlds; ++world) {
        BuildWorld(scene, assets, world, random, size, vars, products);
    }
    RenderSettings::Define(scene, "/Render/Settings").SetProducts(products);
}

/// @brief Writes the first world's six views side by side as a binary PPM, to check the scene by
/// eye against Madrona's `--dump-last-frame`.
void DumpFirstWorld(Renderer& renderer, Scene& scene, std::uint32_t side, const char* path) {
    const RenderBatch& batch = renderer.Step(scene).front();
    const Tensor colour = renderer.Read(*batch.Find(render_vars::kLdrColor));
    const auto* texels = static_cast<const std::uint8_t*>(colour.Data());
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << kAgents * side << " " << side << "\n255\n";
    for (std::uint32_t row = 0; row < side; ++row) {
        for (int view = 0; view < kAgents; ++view) {
            for (std::uint32_t column = 0; column < side; ++column) {
                const std::size_t at =
                    4 * ((static_cast<std::size_t>(view) * side + row) * side + column);
                file.write(reinterpret_cast<const char*>(texels + at), 3);
            }
        }
    }
    std::printf("  wrote %s\n", path);
}

Timing StepHideSeek(int worlds, std::uint32_t side, double targetMs) {
    const char* data = std::getenv("ERNIE_MADRONA_DATA");
    if (data == nullptr) {
        std::printf("skipped: set ERNIE_MADRONA_DATA to madrona_benchmark's data directory\n");
        return {0.0, targetMs};
    }
    std::unique_ptr<Renderer> renderer;
    try {
        renderer = std::make_unique<Renderer>();
    } catch (const std::exception& error) {
        std::printf("skipped: %s\n", error.what());
        return {0.0, targetMs};
    }
    const Assets assets{std::filesystem::path(data)};
    Scene scene;
    BuildHideSeek(scene, assets, worlds, glm::uvec2(side));
    renderer->Step(scene);
    renderer->Context().WaitIdle();
    const double ms = Fastest([] {},
                              [&] {
                                  renderer->Step(scene);
                                  renderer->Context().WaitIdle();
                              });
    std::printf("  %d worlds, %d views of %u x %u: %.0f worlds, %.0f views a second, on %s\n",
                worlds, worlds * kAgents, side, side, 1000.0 * worlds / ms,
                1000.0 * worlds * kAgents / ms, renderer->Context().Info().name.c_str());
    if (const char* dump = std::getenv("ERNIE_BENCH_DUMP")) {
        DumpFirstWorld(*renderer, scene, side, dump);
    }
    return {ms, targetMs};
}

bool RegisterHideSeek(int worlds, std::uint32_t side, double targetMs) {
    return Register("render.hide_seek." + std::to_string(worlds) + "_worlds_" +
                        std::to_string(side) + "x" + std::to_string(side),
                    [=] { return StepHideSeek(worlds, side, targetMs); });
}

// Targets: madrona_benchmark's hideseek_headless step times, ray tracer, on an RTX 5070 Laptop
// GPU (its step simulates too).
const bool kHideSeek = RegisterHideSeek(1024, 64, 49.4) && RegisterHideSeek(1024, 128, 149.4) &&
                       RegisterHideSeek(1024, 256, 501.1) && RegisterHideSeek(4096, 64, 193.5);

}  // namespace
}  // namespace ernie::bench
