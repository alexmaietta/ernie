// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/bvh/builder.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/builders/instance_builder.h"
#include "ernie/scene/builders/mesh_builder.h"
#include "ernie/scene/builders/surface_builder.h"
#include "ernie/scene/builders/transform_builder.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/publish/publisher.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
#include "support/devices.h"

namespace ernie {
namespace {

const std::vector<Float3> kQuad = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};

struct Fixture {
    device::Device& device;
    Store& store;
    Catalog& catalog;
    uint32_t balls[2];
    uint32_t quads[2];
    uint32_t sun;

    View One(uint32_t prim, SchemaId schema) {
        return View(catalog, schema, std::vector<uint32_t>{prim});
    }

    void Move(uint32_t prim, Double3 to) {
        One(prim, SchemaId::Xformable).Get<Double3>(kXformableTranslate).Set(to);
    }

    uint32_t PointsRow(uint32_t prim) {
        return catalog.GetPrims().Row(prim, SchemaId::PointBased);
    }

    ArrayRef NewArray(ColumnType type, std::span<const std::byte> bytes, uint32_t count) {
        ArrayHeap& arrays = store.Arrays(type);
        const ArrayRef array = arrays.Allocate(count);
        arrays.Write(array, bytes);
        return array;
    }
};

struct Work {
    uint32_t hierarchyRows = 0;
    uint32_t blasBuilt = 0;
    uint32_t blasRefit = 0;
    uint32_t tlasBuilt = 0;
    uint32_t tlasRefit = 0;
    uint32_t instancesWritten = 0;
    uint32_t surfacesWritten = 0;
    bool hierarchyRebuilt = false;
    bool instancesRelaid = false;

    bool operator==(const Work&) const = default;
};

void PrintTo(const Work& work, std::ostream* out) {
    *out << fmt::format("{{hierarchy rows {}, BLAS built {} refit {}, TLAS built {} refit {}, "
                        "instances written {}, surfaces written {}, hierarchy rebuilt {}, "
                        "instances relaid {}}}",
                        work.hierarchyRows, work.blasBuilt, work.blasRefit, work.tlasBuilt,
                        work.tlasRefit, work.instancesWritten, work.surfacesWritten,
                        work.hierarchyRebuilt, work.instancesRelaid);
}

struct Case {
    const char* name;
    std::function<void(Fixture&)> change;
    Work expected;
};

const Case kCases[] = {
    {"nothing written does nothing", [](Fixture&) {}, {}},
    {"an attribute no system reads does nothing",
     [](Fixture& f) { f.One(f.quads[0], SchemaId::Gprim).Get<bool>(kGprimDoubleSided).Set(false); },
     {}},
    {"a display color rewrites its world's surfaces without running instances",
     [](Fixture& f) {
         f.One(f.balls[0], SchemaId::Gprim).Get<Float3>(kGprimDisplayColor).Set(Float3{1, 0, 0});
     },
     {.surfacesWritten = 3}},
    {"authored normals rewrite their world's surfaces without running instances",
     [](Fixture& f) {
         f.One(f.quads[0], SchemaId::PointBased)
             .GetArray<Float3>(kPointBasedNormals)
             .Set(0, std::vector<Float3>(4, Float3{0, 0, 1}));
     },
     {.surfacesWritten = 3}},
    {"moving a prim refits its world only",
     [](Fixture& f) { f.Move(f.balls[0], {2, 0, 0}); },
     {.hierarchyRows = 3, .tlasRefit = 1, .instancesWritten = 3}},
    {"moving prims in two worlds refits both",
     [](Fixture& f) {
         f.Move(f.balls[0], {2, 0, 0});
         f.Move(f.balls[1], {2, 0, 0});
     },
     {.hierarchyRows = 6, .tlasRefit = 2, .instancesWritten = 6}},
    {"moving a global prim refits every TLAS",
     [](Fixture& f) { f.Move(f.sun, {0, 0, 9}); },
     {.hierarchyRows = 7, .tlasRefit = 3, .instancesWritten = 7}},
    {"resizing a shape rebuilds its world",
     [](Fixture& f) { f.One(f.balls[1], SchemaId::Sphere).Get<float>(kSphereRadius).Set(2.0f); },
     {.tlasBuilt = 1, .instancesWritten = 3, .surfacesWritten = 3}},
    {"hiding a prim rebuilds its world",
     [](Fixture& f) {
         f.One(f.balls[0], SchemaId::Imageable)
             .Get<Visibility>(kImageableVisibility)
             .Set(Visibility::Invisible);
     },
     {.hierarchyRows = 3, .tlasBuilt = 1, .instancesWritten = 3, .surfacesWritten = 3}},
    {"changing a prim's purpose rebuilds its world",
     [](Fixture& f) {
         f.One(f.balls[0], SchemaId::Imageable).Get<Purpose>(kImageablePurpose).Set(Purpose::Guide);
     },
     {.hierarchyRows = 3, .tlasBuilt = 1, .instancesWritten = 3, .surfacesWritten = 3}},
    {"moving shared points in place refits the BLAS once and every world using it",
     [](Fixture& f) {
         const ArrayRef points = f.store.Array(kPointBasedPoints, f.PointsRow(f.quads[0]));
         const std::vector<Float3> raised = {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
         f.store.Arrays(ColumnType::Float3Array).Write(points, std::as_bytes(std::span(raised)));
         f.store.SetArray(kPointBasedPoints, f.PointsRow(f.quads[0]), points);
     },
     {.blasRefit = 1, .tlasRefit = 2, .instancesWritten = 6}},
    {"giving one clone its own points builds one BLAS and rebuilds its world",
     [](Fixture& f) {
         f.store.SetArray(kPointBasedPoints, f.PointsRow(f.quads[1]),
                          f.NewArray(ColumnType::Float3Array, std::as_bytes(std::span(kQuad)), 4));
     },
     {.blasBuilt = 1, .tlasBuilt = 1, .instancesWritten = 3, .surfacesWritten = 3}},
    {"new topology for one clone builds one BLAS and rebuilds its world",
     [](Fixture& f) {
         const std::vector<int32_t> counts = {3, 3};
         const std::vector<int32_t> indices = {0, 1, 2, 0, 2, 3};
         const uint32_t row = f.catalog.GetPrims().Row(f.quads[0], SchemaId::Mesh);
         f.store.SetArray(kMeshFaceVertexCounts, row,
                          f.NewArray(ColumnType::IntArray, std::as_bytes(std::span(counts)), 2));
         f.store.SetArray(kMeshFaceVertexIndices, row,
                          f.NewArray(ColumnType::IntArray, std::as_bytes(std::span(indices)), 6));
     },
     {.blasBuilt = 1, .tlasBuilt = 1, .instancesWritten = 3, .surfacesWritten = 3}},
    {"defining a prim updates and rebuilds only its world",
     [](Fixture& f) {
         f.catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/Env0/Extra"});
     },
     {.hierarchyRows = 4, .tlasBuilt = 1, .instancesWritten = 4, .surfacesWritten = 4}},
    {"removing a prim updates and rebuilds only its world",
     [](Fixture& f) { f.catalog.Remove(f.balls[1]); },
     {.hierarchyRows = 2, .tlasBuilt = 1, .instancesWritten = 2, .surfacesWritten = 2}},
    {"defining a global prim rebuilds every TLAS, since every world holds it",
     [](Fixture& f) { f.catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/Moon"}); },
     {.hierarchyRows = 8, .tlasBuilt = 3, .instancesWritten = 10, .surfacesWritten = 10}},
    {"defining a world lays everything out again",
     [](Fixture& f) { f.catalog.Define(SchemaId::World, std::vector<std::string>{"/Env2"}); },
     {.hierarchyRows = 8,
      .tlasBuilt = 4,
      .instancesWritten = 8,
      .surfacesWritten = 8,
      .hierarchyRebuilt = true,
      .instancesRelaid = true}},
};

TEST(UpdatesTest, EachChangeDoesExactlyTheWorkItNeeds) {
    test::ForEachDevice([](device::Device& device) {
        if (device.Adapter().backend == device::Backend::Hip) {
            return;
        }
        for (const Case& c : kCases) {
            SCOPED_TRACE(c.name);
            Store store(device);
            Catalog catalog(store);
            Timeline timeline(catalog);
            const std::unique_ptr<bvh::Builder> builder = bvh::Builder::Create(device);
            Hierarchy hierarchy(catalog);
            Geometry geometry(catalog, *builder);
            Instances instances(catalog, *builder);
            Surfaces surfaces(catalog, instances);
            timeline.Add(hierarchy);
            timeline.Add(geometry);
            timeline.Add(instances);
            timeline.Add(surfaces);

            Fixture f{device, store, catalog, {}, {}, kNoPrim};
            catalog.Define(SchemaId::World, std::vector<std::string>{"/Env0"});
            f.balls[0] =
                catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/Env0/Ball"}).firstPrim;
            f.quads[0] = catalog
                             .Define<Mesh>(std::vector<std::string>{"/Env0/Quad"},
                                           {.points = kQuad,
                                            .faceVertexIndices = {0, 1, 2, 3},
                                            .faceVertexCounts = {4}})
                             .firstPrim;
            catalog.Clone(catalog.Find("/Env0"), std::vector<std::string>{"/Env1"});
            f.balls[1] = catalog.Find("/Env1/Ball");
            f.quads[1] = catalog.Find("/Env1/Quad");
            f.sun = catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/Sun"}).firstPrim;
            timeline.Publish(1).Wait();

            const uint32_t hierarchyRebuilds = hierarchy.Rebuilds();
            const uint32_t instanceRelayouts = instances.Relayouts();
            const uint32_t hierarchyRuns = hierarchy.Runs();
            const uint32_t geometryRuns = geometry.Runs();
            const uint32_t instancesRuns = instances.Runs();
            const uint32_t surfacesRuns = surfaces.Runs();
            const uint64_t submitted = device.GetQueue().Submitted();
            c.change(f);
            timeline.Publish(2).Wait();

            Work work;
            if (hierarchy.Runs() != hierarchyRuns) {
                work.hierarchyRows = hierarchy.Dispatched();
            }
            if (geometry.Runs() != geometryRuns) {
                work.blasBuilt = geometry.Stats().built;
                work.blasRefit = geometry.Stats().refit;
            }
            if (instances.Runs() != instancesRuns) {
                work.tlasBuilt = instances.Stats().built;
                work.tlasRefit = instances.Stats().refit;
                work.instancesWritten = instances.Stats().written;
            }
            if (surfaces.Runs() != surfacesRuns) {
                work.surfacesWritten = surfaces.Written();
            }
            work.hierarchyRebuilt = hierarchy.Rebuilds() != hierarchyRebuilds;
            work.instancesRelaid = instances.Relayouts() != instanceRelayouts;
            EXPECT_EQ(work, c.expected);
            if (std::string_view(c.name) == "nothing written does nothing") {
                EXPECT_EQ(device.GetQueue().Submitted(), submitted);
            }
        }
    });
}

}
}
