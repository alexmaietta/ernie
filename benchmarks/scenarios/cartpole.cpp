// SPDX-License-Identifier: BSD-3-Clause
// A cart on a rail balancing a pole, each world's in its own state,
// seen from the front: the simplest scene, mostly the cost of a view.

#include "../batch.h"

#include "ernie/scene/geometry_schemas.h"

#include <glm/gtc/constants.hpp>

#include <cmath>

namespace ernie::bench {

namespace {

class Cartpole : public BatchScenario {
  protected:
    std::vector<Param> WorldParams() const override {
        return {{"spread", "1.0",
                 "Range of the carts' positions, metres"}};
    }

    void BuildWorlds(Scene& scene, const Case& c) override {
        GroundPlane::Define(scene, "/World/Ground")
            .SetChecker(true)
            .SetCheckerSize(0.5);
        const Path root("/World/Cartpoles");
        const std::vector<Path> prototypes = {root.Child("Rail"),
                                              root.Child("Cart"),
                                              root.Child("Pole")};
        Cube::Define(scene, prototypes[0]).SetSize(1.0);
        Cube::Define(scene, prototypes[1]).SetSize(1.0);
        Capsule::Define(scene, prototypes[2])
            .SetRadius(0.025)
            .SetHeight(1.0);

        const float spread =
            static_cast<float>(GetDouble(c, "spread"));
        std::vector<Placement> placements;
        for (int world = 0; world < Worlds(c); ++world) {
            Random random(static_cast<std::uint64_t>(world));
            const float x = random.Uniform(-spread, spread);
            const float angle = random.Uniform(-0.4f, 0.4f);
            const glm::vec3 cart(x, 0.0f, 1.0f);
            const glm::quat tilt =
                glm::angleAxis(angle, glm::vec3(0.0f, 1.0f, 0.0f));
            placements.push_back({.proto = 0,
                                  .world = world,
                                  .position = {0.0f, 0.0f, 1.0f},
                                  .scale = {5.0f, 0.03f, 0.03f},
                                  .colour = {0.1f, 0.1f, 0.12f}});
            placements.push_back({.proto = 1,
                                  .world = world,
                                  .position = cart,
                                  .scale = {0.4f, 0.25f, 0.16f},
                                  .colour = {0.8f, 0.45f, 0.15f}});
            placements.push_back(
                {.proto = 2,
                 .world = world,
                 .position =
                     cart + tilt * glm::vec3(0.0f, -0.15f, 0.5f),
                 .orientation = tilt,
                 .colour = {0.9f, 0.8f, 0.6f}});
        }
        DefineInstances(scene, root, prototypes, placements);
    }

    void Poses(const Case& c, int step,
               std::span<glm::dmat4> poses) const override {
        // From the front, swaying a few centimetres.
        for (std::size_t v = 0; v < poses.size(); ++v) {
            const double sway =
                0.05 *
                std::sin(0.2 * step + 0.01 * static_cast<double>(v));
            poses[v] = LookAt({sway, -3.0, 1.3}, {0.0, 0.0, 1.3});
        }
        (void)c;
    }

    double FocalLength() const override {
        return 18.0;
    }
};

}  // namespace

ERNIE_BENCH_SCENARIO(
    Cartpole, "cartpole",
    "A cart balancing a pole per world, seen from the front");

}  // namespace ernie::bench
