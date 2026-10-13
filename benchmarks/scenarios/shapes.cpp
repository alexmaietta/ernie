// SPDX-License-Identifier: BSD-3-Clause
// A walled arena of boxes, spheres, capsules and cylinders, laid out
// afresh in every world, seen by agents standing in it: each view is
// an agent's, turning as it looks around.

#include "../batch.h"

#include "ernie/scene/geometry_schemas.h"

#include <glm/gtc/constants.hpp>

#include <cmath>

namespace ernie::bench {

namespace {

class Shapes : public BatchScenario {
  protected:
    std::vector<Param> WorldParams() const override {
        return {{"objects", "16", "Shapes in each world's arena"},
                {"arena", "10", "Side of the arena, metres"}};
    }

    void BuildWorlds(Scene& scene, const Case& c) override {
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(1.0);
        const Path root("/World/Shapes");
        const std::vector<Path> prototypes = {
            root.Child("Box"), root.Child("Sphere"),
            root.Child("Capsule"), root.Child("Cylinder")};
        Cube::Define(scene, prototypes[0]).SetSize(1.0);
        Sphere::Define(scene, prototypes[1]).SetRadius(0.5);
        Capsule::Define(scene, prototypes[2])
            .SetRadius(0.25)
            .SetHeight(1.0);
        Cylinder::Define(scene, prototypes[3])
            .SetRadius(0.5)
            .SetHeight(1.0);

        const float half =
            static_cast<float>(GetDouble(c, "arena")) / 2.0f;
        const int objects = GetInt(c, "objects");
        std::vector<Placement> placements;
        for (int world = 0; world < Worlds(c); ++world) {
            const glm::vec3 wall(0.6f, 0.6f, 0.65f);
            for (int side = 0; side < 4; ++side) {
                const float sign = side % 2 == 0 ? 1.0f : -1.0f;
                const bool alongX = side < 2;
                placements.push_back(
                    {.proto = 0,
                     .world = world,
                     .position =
                         alongX ? glm::vec3(0.0f, sign * half, 0.5f)
                                : glm::vec3(sign * half, 0.0f, 0.5f),
                     .scale =
                         alongX ? glm::vec3(2.0f * half, 0.2f, 1.0f)
                                : glm::vec3(0.2f, 2.0f * half, 1.0f),
                     .colour = wall});
            }
            Random random(static_cast<std::uint64_t>(world) * 7919u +
                          1u);
            for (int i = 0; i < objects; ++i) {
                const float size = random.Uniform(0.4f, 1.2f);
                placements.push_back(
                    {.proto = random.Int(0, 3),
                     .world = world,
                     .position = {random.Uniform(-half + 1.0f,
                                                 half - 1.0f),
                                  random.Uniform(-half + 1.0f,
                                                 half - 1.0f),
                                  size / 2.0f},
                     .orientation = Yaw(random.Uniform(0.0f, 6.283f)),
                     .scale = glm::vec3(size),
                     .colour = {random.Uniform(0.2f, 0.9f),
                                random.Uniform(0.2f, 0.9f),
                                random.Uniform(0.2f, 0.9f)}});
            }
            // The agents, which see one another.
            for (int agent = 0; agent < ViewsPerWorld(c); ++agent) {
                placements.push_back(
                    {.proto = 2,
                     .world = world,
                     .position = Agent(c, world, agent),
                     .colour = {0.9f, 0.2f, 0.2f}});
            }
        }
        DefineInstances(scene, root, prototypes, placements);
    }

    void Poses(const Case& c, int step,
               std::span<glm::dmat4> poses) const override {
        const int views = ViewsPerWorld(c);
        for (std::size_t v = 0; v < poses.size(); ++v) {
            const int world = static_cast<int>(v) / views;
            const int agent = static_cast<int>(v) % views;
            const glm::dvec3 at(Agent(c, world, agent));
            const double heading =
                Heading(world, agent) + 0.05 * step;
            const glm::dvec3 ahead(std::cos(heading),
                                   std::sin(heading), 0.0);
            // Eyes in front of the agent's own capsule.
            const glm::dvec3 eye =
                at + 0.35 * ahead + glm::dvec3(0, 0, 0.3);
            poses[v] =
                LookAt(eye, eye + ahead - glm::dvec3(0.0, 0.0, 0.1));
        }
    }

    double FocalLength() const override {
        return 12.0;
    }

  private:
    glm::vec3 Agent(const Case& c, int world, int agent) const {
        const float half =
            static_cast<float>(GetDouble(c, "arena")) / 2.0f;
        Random random(static_cast<std::uint64_t>(world) * 104729u +
                      static_cast<std::uint64_t>(agent) + 17u);
        return {random.Uniform(-half + 1.0f, half - 1.0f),
                random.Uniform(-half + 1.0f, half - 1.0f), 0.75f};
    }

    static double Heading(int world, int agent) {
        Random random(static_cast<std::uint64_t>(world) * 31u +
                      static_cast<std::uint64_t>(agent) * 7u + 3u);
        return random.Uniform(0.0f, 6.283f);
    }
};

}  // namespace

ERNIE_BENCH_SCENARIO(
    Shapes, "shapes",
    "An arena of simple shapes per world, seen by agents in it");

}  // namespace ernie::bench
