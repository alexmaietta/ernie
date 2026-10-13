// SPDX-License-Identifier: BSD-3-Clause
// Furnished houses: a few floor plans of textured rooms, tables,
// chairs, beds, sofas, shelves, lamps and plants built of shared
// parts, each world one of the houses, seen from inside its rooms.

#include "indoor.h"

#include "../batch.h"

#include "ernie/geometry/primitives.h"
#include "ernie/scene/geometry_schemas.h"

#include <array>
#include <cmath>

namespace ernie::bench {

namespace {

/// A part of a piece of furniture: a unit prototype (a box, a
/// cylinder or sphere of diameter 1, a cone) moved and scaled.
struct Part {
    int proto;
    glm::vec3 offset;
    glm::vec3 size;
};

enum Parts {
    kWood,
    kFabric,
    kMetal,
    kCeramic,
    kShade,
    kLeaves,
    kDark,
    kParts
};

const std::vector<std::vector<Part>>& Furniture() {
    static const std::vector<std::vector<Part>> pieces = {
        // A table.
        {{kWood, {0, 0, 0.73f}, {1.2f, 0.8f, 0.04f}},
         {kWood, {0.55f, 0.35f, 0.355f}, {0.05f, 0.05f, 0.71f}},
         {kWood, {-0.55f, 0.35f, 0.355f}, {0.05f, 0.05f, 0.71f}},
         {kWood, {0.55f, -0.35f, 0.355f}, {0.05f, 0.05f, 0.71f}},
         {kWood, {-0.55f, -0.35f, 0.355f}, {0.05f, 0.05f, 0.71f}},
         {kCeramic, {0.2f, 0.1f, 0.8f}, {0.12f, 0.12f, 0.12f}}},
        // A chair.
        {{kWood, {0, 0, 0.45f}, {0.45f, 0.45f, 0.05f}},
         {kWood, {0, 0.2f, 0.72f}, {0.45f, 0.05f, 0.5f}},
         {kMetal, {0.2f, 0.2f, 0.225f}, {0.04f, 0.04f, 0.45f}},
         {kMetal, {-0.2f, 0.2f, 0.225f}, {0.04f, 0.04f, 0.45f}},
         {kMetal, {0.2f, -0.2f, 0.225f}, {0.04f, 0.04f, 0.45f}},
         {kMetal, {-0.2f, -0.2f, 0.225f}, {0.04f, 0.04f, 0.45f}}},
        // A bed.
        {{kWood, {0, 0, 0.15f}, {1.6f, 2.0f, 0.3f}},
         {kFabric, {0, 0.05f, 0.4f}, {1.5f, 1.85f, 0.2f}},
         {kFabric, {0.35f, -0.75f, 0.56f}, {0.5f, 0.3f, 0.12f}},
         {kFabric, {-0.35f, -0.75f, 0.56f}, {0.5f, 0.3f, 0.12f}}},
        // A sofa.
        {{kFabric, {0, 0, 0.2f}, {2.0f, 0.9f, 0.4f}},
         {kFabric, {0, 0.35f, 0.65f}, {2.0f, 0.2f, 0.5f}},
         {kFabric, {0.9f, 0, 0.55f}, {0.2f, 0.9f, 0.3f}},
         {kFabric, {-0.9f, 0, 0.55f}, {0.2f, 0.9f, 0.3f}}},
        // A shelf of vases.
        {{kWood, {0.48f, 0, 0.9f}, {0.04f, 0.35f, 1.8f}},
         {kWood, {-0.48f, 0, 0.9f}, {0.04f, 0.35f, 1.8f}},
         {kWood, {0, 0, 0.1f}, {1.0f, 0.35f, 0.03f}},
         {kWood, {0, 0, 0.6f}, {1.0f, 0.35f, 0.03f}},
         {kWood, {0, 0, 1.1f}, {1.0f, 0.35f, 0.03f}},
         {kWood, {0, 0, 1.6f}, {1.0f, 0.35f, 0.03f}},
         {kCeramic, {-0.25f, 0, 0.72f}, {0.2f, 0.2f, 0.2f}},
         {kCeramic, {0.2f, 0, 1.2f}, {0.15f, 0.15f, 0.15f}},
         {kMetal, {0.25f, 0, 0.25f}, {0.15f, 0.15f, 0.25f}}},
        // A lamp.
        {{kMetal, {0, 0, 0.02f}, {0.25f, 0.25f, 0.04f}},
         {kMetal, {0, 0, 0.72f}, {0.03f, 0.03f, 1.4f}},
         {kShade, {0, 0, 1.45f}, {0.4f, 0.4f, 0.3f}}},
        // A plant.
        {{kCeramic, {0, 0, 0.175f}, {0.3f, 0.3f, 0.35f}},
         {kLeaves, {0, 0, 0.6f}, {0.45f, 0.45f, 0.45f}},
         {kLeaves, {0.12f, 0.05f, 0.85f}, {0.35f, 0.35f, 0.35f}},
         {kLeaves, {-0.1f, -0.06f, 0.8f}, {0.3f, 0.3f, 0.3f}}},
        // A television on a stand.
        {{kWood, {0, 0, 0.25f}, {1.4f, 0.4f, 0.5f}},
         {kDark, {0, 0, 0.85f}, {1.1f, 0.06f, 0.65f}}},
    };
    return pieces;
}

class House : public BatchScenario {
  protected:
    std::vector<Param> WorldParams() const override {
        return {
            {"layouts", "16",
             "Distinct houses; world w is house w % layouts"},
            {"objects", "40", "Pieces of furniture in each house"},
            {"detail", "48", "Slices around the round parts"}};
    }

    void BuildWorlds(Scene& scene, const Case& c) override {
        const Path root("/World/House");
        std::vector<Path> prototypes;
        DefineParts(scene, root, GetInt(c, "detail"), prototypes);
        const indoor::Shell shell =
            indoor::DefineShell(scene, root, prototypes);

        const int layouts = GetInt(c, "layouts");
        layouts_.clear();
        std::vector<std::vector<Placement>> furnished(
            static_cast<std::size_t>(layouts));
        for (int l = 0; l < layouts; ++l) {
            layouts_.push_back(
                indoor::MakeLayout(static_cast<std::uint64_t>(l)));
            Random random(static_cast<std::uint64_t>(l) * 977u + 5u);
            const indoor::Layout& layout = layouts_.back();
            for (int i = 0; i < GetInt(c, "objects"); ++i) {
                const auto& piece =
                    Furniture()[static_cast<std::size_t>(random.Int(
                        0,
                        static_cast<int>(Furniture().size()) - 1))];
                const indoor::Room& room =
                    layout.rooms[static_cast<std::size_t>(random.Int(
                        0,
                        static_cast<int>(layout.rooms.size()) - 1))];
                const glm::vec3 at =
                    indoor::PointIn(room, 1.0f, 0.0f, random);
                const glm::quat turn = Yaw(
                    1.5708f * static_cast<float>(random.Int(0, 3)));
                for (const Part& part : piece) {
                    furnished[static_cast<std::size_t>(l)].push_back(
                        {.proto = part.proto,
                         .position = at + turn * part.offset,
                         .orientation = turn,
                         .scale = part.size});
                }
            }
        }
        std::vector<Placement> placements;
        for (int world = 0; world < Worlds(c); ++world) {
            const auto l = static_cast<std::size_t>(world % layouts);
            indoor::PlaceShell(layouts_[l], shell, world, placements);
            for (Placement placement : furnished[l]) {
                placement.world = world;
                placements.push_back(placement);
            }
        }
        DefineInstances(scene, root, prototypes, placements);
        AddStat("layouts", layouts);
    }

    void Poses(const Case& c, int step,
               std::span<glm::dmat4> poses) const override {
        const int views = ViewsPerWorld(c);
        for (std::size_t v = 0; v < poses.size(); ++v) {
            const int world = static_cast<int>(v) / views;
            poses[v] = indoor::ViewIn(
                layouts_[static_cast<std::size_t>(world) %
                         layouts_.size()],
                world, static_cast<int>(v) % views, step);
        }
    }

    glm::dvec3 SunDirection() const override {
        return {0.3, -0.2, 0.93};  // high, to reach into the rooms
    }

  private:
    /// The parts' prototypes, `Parts` order, each with a material.
    static void DefineParts(Scene& scene, const Path& root,
                            int detail,
                            std::vector<Path>& prototypes) {
        const auto slices = static_cast<std::uint32_t>(detail);
        struct Look {
            const char* name;
            glm::vec3 a;
            glm::vec3 b;
            int cells;
            float noise;
        };
        const std::array<Look, kParts> looks = {{
            {"Wood",
             {0.5f, 0.32f, 0.18f},
             {0.5f, 0.32f, 0.18f},
             1,
             0.4f},
            {"Fabric",
             {0.3f, 0.4f, 0.55f},
             {0.35f, 0.45f, 0.6f},
             32,
             0.1f},
            {"Metal",
             {0.6f, 0.6f, 0.62f},
             {0.6f, 0.6f, 0.62f},
             1,
             0.05f},
            {"Ceramic",
             {0.85f, 0.8f, 0.7f},
             {0.2f, 0.4f, 0.6f},
             4,
             0.05f},
            {"Shade",
             {0.95f, 0.9f, 0.75f},
             {0.95f, 0.9f, 0.75f},
             1,
             0.05f},
            {"Leaves",
             {0.15f, 0.45f, 0.15f},
             {0.2f, 0.55f, 0.2f},
             16,
             0.3f},
            {"Dark",
             {0.05f, 0.05f, 0.06f},
             {0.05f, 0.05f, 0.06f},
             1,
             0.0f},
        }};
        const std::array<MeshData, kParts> meshes = {
            BoxMesh(glm::vec3(1.0f)),
            BoxMesh(glm::vec3(1.0f)),
            CylinderMesh(0.5f, 1.0f, Axis::Z, slices),
            SphereMesh(0.5f, slices, slices / 2),
            ConeMesh(0.5f, 1.0f, Axis::Z, slices),
            SphereMesh(0.5f, slices, slices / 2),
            BoxMesh(glm::vec3(1.0f)),
        };
        for (std::size_t i = 0; i < kParts; ++i) {
            const Path look = Path("/Looks").Child(looks[i].name);
            DefineTexturedMaterial(
                scene, look,
                CheckerTexture(128, looks[i].cells, looks[i].a,
                               looks[i].b, looks[i].noise,
                               static_cast<std::uint32_t>(30 + i)),
                128);
            MeshData mesh = meshes[i];
            ProjectUvs(mesh, 2.0f);
            prototypes.push_back(root.Child(looks[i].name));
            DefineMesh(scene, prototypes.back(), mesh)
                .SetMaterialBinding(look);
        }
    }

    std::vector<indoor::Layout> layouts_;
};

}  // namespace

ERNIE_BENCH_SCENARIO(
    House, "house",
    "Furnished houses of a few rooms, seen from inside");

}  // namespace ernie::bench
