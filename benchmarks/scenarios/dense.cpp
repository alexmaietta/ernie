// SPDX-License-Identifier: BSD-3-Clause
// Rooms crowded with detailed objects: a library of finely
// tessellated, textured prototypes, hundreds placed through each of a
// few floor plans, millions of triangles a world, seen from inside.

#include "indoor.h"

#include "../batch.h"

#include "ernie/geometry/primitives.h"
#include "ernie/scene/geometry_schemas.h"

#include <algorithm>
#include <cmath>

namespace ernie::bench {

namespace {

constexpr int kLooks = 8;

class Dense : public BatchScenario {
  protected:
    std::vector<Param> WorldParams() const override {
        return {
            {"layouts", "16",
             "Distinct floor plans; world w has plan w % layouts"},
            {"objects", "330", "Objects in each floor plan"},
            {"library", "256", "Distinct object meshes"},
            {"triangles", "20000",
             "Mean triangles of an object mesh"}};
    }

    void BuildWorlds(Scene& scene, const Case& c) override {
        const Path root("/World/Dense");
        std::vector<Path> prototypes;
        const int library = GetInt(c, "library");
        DefineLibrary(scene, root, library, GetInt(c, "triangles"),
                      prototypes);
        const indoor::Shell shell =
            indoor::DefineShell(scene, root, prototypes);

        const int layouts = GetInt(c, "layouts");
        layouts_.clear();
        std::vector<std::vector<Placement>> filled(
            static_cast<std::size_t>(layouts));
        for (int l = 0; l < layouts; ++l) {
            layouts_.push_back(indoor::MakeLayout(
                static_cast<std::uint64_t>(l) + 1000u));
            Random random(static_cast<std::uint64_t>(l) * 541u + 9u);
            const indoor::Layout& layout = layouts_.back();
            for (int i = 0; i < GetInt(c, "objects"); ++i) {
                const indoor::Room& room =
                    layout.rooms[static_cast<std::size_t>(random.Int(
                        0,
                        static_cast<int>(layout.rooms.size()) - 1))];
                const float size = random.Uniform(0.15f, 0.6f);
                // On the floor, or up on what would be shelves.
                const float z = random.Uniform(0.0f, 1.0f) < 0.5f
                                    ? size / 2.0f
                                    : random.Uniform(0.5f, 1.9f);
                filled[static_cast<std::size_t>(l)].push_back(
                    {.proto = random.Int(0, library - 1),
                     .position =
                         indoor::PointIn(room, 0.4f, z, random),
                     .orientation = Yaw(random.Uniform(0.0f, 6.283f)),
                     .scale = glm::vec3(size)});
            }
        }
        std::vector<Placement> placements;
        for (int world = 0; world < Worlds(c); ++world) {
            const auto l = static_cast<std::size_t>(world % layouts);
            indoor::PlaceShell(layouts_[l], shell, world, placements);
            for (Placement placement : filled[l]) {
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
        return {0.3, -0.2, 0.93};
    }

  private:
    /// `count` prototypes of about `triangles` triangles each, of
    /// unit size: spheres, capsules, cylinders and cones, finely
    /// sliced.
    static void DefineLibrary(Scene& scene, const Path& root,
                              int count, int triangles,
                              std::vector<Path>& prototypes) {
        for (int i = 0; i < kLooks; ++i) {
            const Path look("/Looks/Object" + std::to_string(i));
            Random random(static_cast<std::uint64_t>(i) + 200u);
            const glm::vec3 a(random.Uniform(0.2f, 0.9f),
                              random.Uniform(0.2f, 0.9f),
                              random.Uniform(0.2f, 0.9f));
            DefineTexturedMaterial(
                scene, look,
                CheckerTexture(256, 2 << (i % 4), a, a * 0.6f, 0.3f,
                               static_cast<std::uint32_t>(40 + i)),
                256);
        }
        for (int i = 0; i < count; ++i) {
            Random random(static_cast<std::uint64_t>(i) * 31u + 77u);
            const float target = static_cast<float>(triangles) *
                                 random.Uniform(0.5f, 1.5f);
            // s slices and s / 2 stacks make about s^2 triangles.
            const auto rings = static_cast<std::uint32_t>(
                std::max(8.0f, std::sqrt(target)));
            MeshData mesh;
            switch (i % 3) {
            case 0:
                mesh = SphereMesh(0.5f, rings, rings / 2);
                break;
            case 1:
                mesh = CapsuleMesh(0.25f, 0.5f, Axis::Z, rings,
                                   rings / 2);
                break;
            default:
                mesh = CapsuleMesh(0.4f, 0.2f, Axis::X, rings,
                                   rings / 2);
                break;
            }
            ProjectUvs(mesh, 2.0f);
            prototypes.push_back(
                root.Child("Object" + std::to_string(i)));
            DefineMesh(scene, prototypes.back(), mesh)
                .SetMaterialBinding("/Looks/Object" +
                                    std::to_string(i % kLooks));
        }
    }

    std::vector<indoor::Layout> layouts_;
};

}  // namespace

ERNIE_BENCH_SCENARIO(
    Dense, "dense",
    "Rooms crowded with detailed objects, seen from inside");

}  // namespace ernie::bench
