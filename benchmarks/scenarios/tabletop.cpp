// SPDX-License-Identifier: BSD-3-Clause
// A robot arm at a textured table beside a cabinet, with cubes to
// pick: each world's arm in its own pose and its cubes in their own
// places, seen from cameras around the table. The arm's links are
// finely tessellated, some 100 thousand triangles a world.

#include "../batch.h"

#include "ernie/geometry/primitives.h"
#include "ernie/scene/geometry_schemas.h"

#include <glm/gtc/constants.hpp>

#include <array>
#include <cmath>

namespace ernie::bench {

namespace {

constexpr float kTableTop = 0.775f;  // the table's height
constexpr std::array<float, 4> kLinks = {0.33f, 0.32f, 0.25f, 0.10f};

class Tabletop : public BatchScenario {
  protected:
    std::vector<Param> WorldParams() const override {
        return {{"detail", "96",
                 "Slices around the arm's links and joints; their "
                 "triangles grow as its square"},
                {"cubes", "4", "Cubes on each world's table"}};
    }

    void BuildWorlds(Scene& scene, const Case& c) override {
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(0.5);
        BuildFurniture(scene, c);
        BuildArms(scene, c);
    }

    void Poses(const Case& c, int step,
               std::span<glm::dmat4> poses) const override {
        // Around the table's front, each world's cameras spread
        // evenly, all drifting round.
        const int views = ViewsPerWorld(c);
        for (std::size_t v = 0; v < poses.size(); ++v) {
            const int world = static_cast<int>(v) / views;
            const int k = static_cast<int>(v) % views;
            Random random(static_cast<std::uint64_t>(world) + 5u);
            const double heading =
                -glm::half_pi<double>() +
                static_cast<double>(random.Uniform(-0.3f, 0.3f)) +
                glm::two_pi<double>() * k / views + 0.01 * step;
            poses[v] = LookAt({1.6 * std::cos(heading),
                               1.6 * std::sin(heading), 1.4},
                              {0.0, 0.0, 0.85});
        }
    }

    double FocalLength() const override {
        return 20.0;
    }

  private:
    void BuildFurniture(Scene& scene, const Case& c) {
        const Path root("/World/Furniture");
        DefineTexturedMaterial(
            scene, "/Looks/Wood",
            CheckerTexture(256, 1, {0.55f, 0.36f, 0.2f},
                           {0.55f, 0.36f, 0.2f}, 0.35f, 1),
            256);
        DefineTexturedMaterial(
            scene, "/Looks/Painted",
            CheckerTexture(256, 1, {0.85f, 0.85f, 0.8f},
                           {0.85f, 0.85f, 0.8f}, 0.1f, 2),
            256);
        DefineTexturedMaterial(
            scene, "/Looks/Cube",
            CheckerTexture(64, 4, {0.9f, 0.15f, 0.1f},
                           {0.95f, 0.95f, 0.9f}, 0.0f, 3),
            64);
        const auto mesh = [&](const char* name, MeshData data,
                              const char* material, float perMetre) {
            ProjectUvs(data, perMetre);
            const Path path = root.Child(name);
            DefineMesh(scene, path, data)
                .SetMaterialBinding(material);
            return path;
        };
        const std::vector<Path> prototypes = {
            mesh("Top", BoxMesh({1.2f, 0.8f, 0.05f}), "/Looks/Wood",
                 1.0f),
            mesh("Leg", CylinderMesh(0.03f, 0.75f, Axis::Z, 24),
                 "/Looks/Wood", 1.0f),
            mesh("Cabinet", BoxMesh({0.5f, 0.4f, 0.9f}),
                 "/Looks/Painted", 1.0f),
            mesh("Drawer", BoxMesh({0.44f, 0.02f, 0.25f}),
                 "/Looks/Wood", 2.0f),
            mesh("Cube", BoxMesh(glm::vec3(0.05f)), "/Looks/Cube",
                 20.0f),
        };
        std::vector<Placement> placements;
        for (int world = 0; world < Worlds(c); ++world) {
            placements.push_back(
                {.proto = 0,
                 .world = world,
                 .position = {0.0f, 0.0f, kTableTop - 0.025f}});
            for (const float x : {-0.55f, 0.55f}) {
                for (const float y : {-0.35f, 0.35f}) {
                    placements.push_back(
                        {.proto = 1,
                         .world = world,
                         .position = {x, y, 0.375f}});
                }
            }
            // The cabinet behind the table, its drawers facing it.
            placements.push_back({.proto = 2,
                                  .world = world,
                                  .position = {0.0f, 0.9f, 0.45f}});
            for (int d = 0; d < 3; ++d) {
                placements.push_back(
                    {.proto = 3,
                     .world = world,
                     .position = {
                         0.0f, 0.69f,
                         0.16f + 0.29f * static_cast<float>(d)}});
            }
            Random random(static_cast<std::uint64_t>(world) * 13u +
                          7u);
            for (int i = 0; i < GetInt(c, "cubes"); ++i) {
                placements.push_back(
                    {.proto = 4,
                     .world = world,
                     .position = {random.Uniform(-0.1f, 0.5f),
                                  random.Uniform(-0.3f, 0.3f),
                                  kTableTop + 0.025f},
                     .orientation =
                         Yaw(random.Uniform(0.0f, 1.57f))});
            }
        }
        DefineInstances(scene, root, prototypes, placements);
    }

    void BuildArms(Scene& scene, const Case& c) {
        const Path root("/World/Arms");
        const auto detail =
            static_cast<std::uint32_t>(GetInt(c, "detail"));
        std::vector<Path> prototypes;
        const auto mesh = [&](const std::string& name,
                              const MeshData& data) {
            prototypes.push_back(root.Child(name));
            DefineMesh(scene, prototypes.back(), data);
        };
        // Prototypes 0 to 3, the links, a capsule of each length from
        // its joint along z.
        for (std::size_t i = 0; i < kLinks.size(); ++i) {
            mesh("Link" + std::to_string(i),
                 CapsuleMesh(0.045f, kLinks[i], Axis::Z, detail,
                             detail / 2));
        }
        mesh("Joint", SphereMesh(0.065f, detail, detail / 2));  // 4
        mesh("Base",
             CylinderMesh(0.09f, 0.15f, Axis::Z, detail));  // 5
        mesh("Finger", BoxMesh({0.015f, 0.02f, 0.06f}));    // 6

        const glm::vec3 white(0.9f);
        const glm::vec3 grey(0.25f);
        std::vector<Placement> placements;
        for (int world = 0; world < Worlds(c); ++world) {
            // Joint angles: the shoulder's yaw, then pitches.
            Random random(static_cast<std::uint64_t>(world) * 101u +
                          11u);
            const std::array<float, 4> pitches = {
                random.Uniform(0.2f, 0.9f),
                random.Uniform(0.6f, 1.6f),
                random.Uniform(0.2f, 1.0f),
                random.Uniform(-0.5f, 0.5f)};
            glm::quat rotation = Yaw(random.Uniform(-1.0f, 1.0f));
            glm::vec3 joint(-0.4f, 0.0f, kTableTop + 0.075f);
            placements.push_back({.proto = 5,
                                  .world = world,
                                  .position = joint,
                                  .colour = grey});
            joint.z += 0.075f;
            for (std::size_t i = 0; i < kLinks.size(); ++i) {
                rotation =
                    rotation *
                    glm::angleAxis(pitches[i],
                                   glm::vec3(0.0f, 1.0f, 0.0f));
                const glm::vec3 along =
                    rotation * glm::vec3(0.0f, 0.0f, 1.0f);
                placements.push_back({.proto = 4,
                                      .world = world,
                                      .position = joint,
                                      .colour = grey});
                placements.push_back(
                    {.proto = static_cast<int>(i),
                     .world = world,
                     .position = joint + along * (kLinks[i] / 2.0f),
                     .orientation = rotation,
                     .colour = white});
                joint += along * kLinks[i];
            }
            for (const float side : {-1.0f, 1.0f}) {
                placements.push_back(
                    {.proto = 6,
                     .world = world,
                     .position =
                         joint + rotation * glm::vec3(side * 0.03f,
                                                      0.0f, 0.03f),
                     .orientation = rotation,
                     .colour = grey});
            }
        }
        DefineInstances(scene, root, prototypes, placements);
    }
};

}  // namespace

ERNIE_BENCH_SCENARIO(
    Tabletop, "tabletop",
    "A robot arm at a table per world, cubes to pick, a "
    "cabinet behind");

}  // namespace ernie::bench
