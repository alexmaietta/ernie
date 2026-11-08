// SPDX-License-Identifier: BSD-3-Clause
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/structure/structure.h"
#include "support/devices.h"

namespace ernie {
namespace {

template <class T>
std::vector<T> Read(Store& store, ColumnRef column, uint32_t first, uint32_t count) {
    const std::vector<std::byte> bytes =
        store.Schema(column.schema).Read(column.index, first, count);
    std::vector<T> cells(bytes.size() / sizeof(T));
    std::memcpy(cells.data(), bytes.data(), bytes.size());
    return cells;
}

std::vector<uint64_t> Stamps(Store& store, ColumnRef column, uint32_t first, uint32_t count) {
    const std::vector<std::byte> bytes =
        store.Schema(column.schema).Read(store.StampColumn(column), first, count);
    std::vector<uint64_t> stamps(count);
    std::memcpy(stamps.data(), bytes.data(), bytes.size());
    return stamps;
}

std::vector<std::string> Paths(std::string_view prefix, uint32_t count) {
    std::vector<std::string> paths;
    for (uint32_t i = 0; i < count; ++i) {
        paths.push_back(fmt::format("{}{}", prefix, i));
    }
    return paths;
}

template <class Test> void ForEachCatalog(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        Store store(device);
        Catalog catalog(store);
        test(store, catalog);
    });
}

TEST(CatalogTest, ADefinedPrimHasItsRecordPathAndFallbacks) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        const std::vector<std::string> paths = {"/Ball"};
        const PrimBlock block = catalog.Define(SchemaId::Sphere, paths);
        const uint32_t prim = block.firstPrim;
        EXPECT_EQ(catalog.Find("/Ball"), prim);
        const PrimRecord& record = catalog.GetPrims().At(prim);
        EXPECT_EQ(record.schema, SchemaId::Sphere);
        EXPECT_EQ(record.parent, kNoPrim);
        EXPECT_EQ(record.world, kGlobalWorld);
        const uint32_t sphere = catalog.GetPrims().Row(prim, SchemaId::Sphere);
        const uint32_t xform = catalog.GetPrims().Row(prim, SchemaId::Xformable);
        EXPECT_EQ(Read<float>(store, kSphereRadius, sphere, 1), std::vector<float>{1.0f});
        EXPECT_EQ(Read<float>(store, kXformableScale, xform, 1),
                  (std::vector<float>{1.0f, 1.0f, 1.0f}));
        EXPECT_EQ(Read<uint32_t>(store,
                                 {SchemaId::Sphere, nullptr, store.PrimsColumn(SchemaId::Sphere)},
                                 sphere, 1),
                  std::vector<uint32_t>{prim});
    });
}

TEST(CatalogTest, ABatchGetsContiguousRowsInEveryTable) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        const std::vector<std::string> paths = Paths("/Ball", 300);
        const PrimBlock block = catalog.Define(SchemaId::Sphere, paths);
        ASSERT_EQ(block.copies, 300u);
        for (const SchemaId table : {SchemaId::Sphere, SchemaId::Xformable, SchemaId::Imageable}) {
            const uint32_t first = catalog.GetPrims().Row(block.Prim(0, 0), table);
            for (uint32_t i = 0; i < 300; ++i) {
                EXPECT_EQ(catalog.GetPrims().Row(block.Prim(i, 0), table), first + i);
            }
            EXPECT_TRUE(store.Schema(table).Contiguous(0, first, 300));
        }
    });
}

TEST(CatalogTest, BroadcastValuesLandOnEveryPrimWhetherStagedOrFilled) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        for (const uint32_t count : {4u, 12000u}) {
            const std::vector<std::string> paths = Paths(fmt::format("/Batch{}_", count), count);
            const PrimBlock block =
                catalog.Define<Sphere>(paths, {.translate = {1.0, 2.0, 3.0}, .radius = 2.5f});
            const uint32_t sphere = catalog.GetPrims().Row(block.Prim(0, 0), SchemaId::Sphere);
            const uint32_t xform = catalog.GetPrims().Row(block.Prim(0, 0), SchemaId::Xformable);
            const std::vector<float> radii = Read<float>(store, kSphereRadius, sphere, count);
            const std::vector<double> translates =
                Read<double>(store, kXformableTranslate, xform, count);
            for (uint32_t i = 0; i < count; ++i) {
                ASSERT_EQ(radii[i], 2.5f) << i;
                ASSERT_EQ(translates[i * 3 + 2], 3.0) << i;
            }
        }
    });
}

TEST(CatalogTest, DefiningStampsTrackedRowsAndTheirWorld) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        store.SetStamp(7);
        const std::vector<std::string> paths = Paths("/Ball", 3);
        const PrimBlock block = catalog.Define(SchemaId::Sphere, paths);
        const uint32_t xform = catalog.GetPrims().Row(block.Prim(0, 0), SchemaId::Xformable);
        EXPECT_EQ(Stamps(store, kXformableTranslate, xform, 3), (std::vector<uint64_t>{7, 7, 7}));
        const std::vector<std::byte> world = store.Worlds().Read(0, 0, 1);
        uint64_t global = 0;
        std::memcpy(&global, world.data(), sizeof(global));
        EXPECT_EQ(global, 7u);
        EXPECT_EQ(catalog.Log().size(), 3u);
        EXPECT_TRUE(catalog.Log()[0].added);
        EXPECT_EQ(catalog.Log()[0].ordinal, 7u);
    });
}

TEST(CatalogTest, ArraysAreSharedAcrossABatchAndFreedWithTheLastPrim) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        const std::vector<std::string> paths = Paths("/Mesh", 3);
        const PrimBlock block =
            catalog.Define<Mesh>(paths, {.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
                                         .faceVertexIndices = {0, 1, 2},
                                         .faceVertexCounts = {3}});
        std::vector<ArrayRef> arrays;
        for (uint32_t i = 0; i < 3; ++i) {
            arrays.push_back(store.Array(
                kPointBasedPoints, catalog.GetPrims().Row(block.Prim(i, 0), SchemaId::PointBased)));
        }
        EXPECT_NE(arrays[0].address, 0u);
        EXPECT_EQ(arrays[1].address, arrays[0].address);
        EXPECT_EQ(arrays[0].count, 3u);
        ArrayHeap& heap = store.Arrays(ColumnType::Float3Array);
        EXPECT_EQ(heap.References(arrays[0]), 3u);
        catalog.Remove(block.Prim(0, 0));
        EXPECT_EQ(heap.References(arrays[0]), 2u);
        catalog.Remove(block.Prim(1, 0));
        catalog.Remove(block.Prim(2, 0));
        EXPECT_EQ(heap.ArrayCount(), 0u);
    });
}

TEST(CatalogTest, RemovedPrimsComeBackWithANewGeneration) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        const std::vector<std::string> first = {"/A"};
        const uint32_t prim = catalog.Define(SchemaId::Sphere, first).firstPrim;
        const uint32_t row = catalog.GetPrims().Row(prim, SchemaId::Sphere);
        const uint32_t generation = catalog.GetPrims().Generation(prim);
        catalog.Remove(prim);
        EXPECT_FALSE(catalog.GetPrims().Alive(prim));
        EXPECT_EQ(catalog.Find("/A"), kNoPrim);
        EXPECT_EQ(Read<uint32_t>(store,
                                 {SchemaId::Sphere, nullptr, store.PrimsColumn(SchemaId::Sphere)},
                                 row, 1),
                  std::vector<uint32_t>{kNoPrim});

        const std::vector<std::string> second = {"/B"};
        const uint32_t again = catalog.Define(SchemaId::Sphere, second).firstPrim;
        EXPECT_EQ(again, prim);
        EXPECT_EQ(catalog.GetPrims().Generation(again), generation + 1);
        EXPECT_EQ(catalog.GetPrims().Row(again, SchemaId::Sphere), row);
    });
}

TEST(CatalogTest, RemovingAPrimRemovesItsSubtree) {
    ForEachCatalog([](Store&, Catalog& catalog) {
        for (const char* path : {"/Robot", "/Robot/Arm", "/Robot/Arm/Hand"}) {
            const std::vector<std::string> paths = {path};
            catalog.Define(SchemaId::Xform, paths);
        }
        catalog.Remove(catalog.Find("/Robot"));
        EXPECT_EQ(catalog.GetPrims().LiveCount(), 0u);
        EXPECT_EQ(catalog.Find("/Robot/Arm/Hand"), kNoPrim);
    });
}

TEST(CatalogTest, WorldsHoldTheirSubtreesAndDontNest) {
    ForEachCatalog([](Store&, Catalog& catalog) {
        for (const char* path : {"/Env", "/Env/Ball"}) {
            const std::vector<std::string> paths = {path};
            catalog.Define(std::string_view(path) == "/Env" ? SchemaId::World : SchemaId::Sphere,
                           paths);
        }
        const std::vector<std::string> sun = {"/Sun"};
        catalog.Define(SchemaId::Sphere, sun);
        EXPECT_EQ(catalog.GetPrims().At(catalog.Find("/Env")).world, 0u);
        EXPECT_EQ(catalog.GetPrims().At(catalog.Find("/Env/Ball")).world, 0u);
        EXPECT_EQ(catalog.GetPrims().At(catalog.Find("/Sun")).world, kGlobalWorld);
        const std::vector<std::string> nested = {"/Env/Inner"};
        EXPECT_THROW(catalog.Define(SchemaId::World, nested), Error);
        EXPECT_EQ(catalog.GetPrims().WorldCount(), 1u);
    });
}

TEST(CatalogTest, ClonesCopyValuesIntoNewWorlds) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        const std::vector<std::string> env = {"/Env0"};
        catalog.Define(SchemaId::World, env);
        const std::vector<std::string> ball = {"/Env0/Ball"};
        catalog.Define<Sphere>(ball, {.radius = 4.0f});
        store.SetStamp(3);
        const std::vector<std::string> targets = Paths("/Env", 4);
        const std::vector<std::string> fresh(targets.begin() + 1, targets.end());
        const PrimBlock block = catalog.Clone(catalog.Find("/Env0"), fresh);
        EXPECT_EQ(block.count, 2u);
        EXPECT_EQ(block.copies, 3u);
        EXPECT_EQ(block.firstWorld, 1u);
        EXPECT_EQ(catalog.GetPrims().WorldCount(), 4u);
        for (uint32_t copy = 0; copy < 3; ++copy) {
            const uint32_t prim = catalog.Find(fmt::format("/Env{}/Ball", copy + 1));
            ASSERT_EQ(prim, block.Prim(copy, 1));
            EXPECT_EQ(catalog.GetPrims().At(prim).world, copy + 1);
            const uint32_t row = catalog.GetPrims().Row(prim, SchemaId::Sphere);
            EXPECT_EQ(Read<float>(store, kSphereRadius, row, 1), std::vector<float>{4.0f});
            EXPECT_EQ(Stamps(store, kXformableTranslate,
                             catalog.GetPrims().Row(prim, SchemaId::Xformable), 1),
                      std::vector<uint64_t>{3});
        }
    });
}

TEST(CatalogTest, MovingTheSourceKeepsItsWorldNumbers) {
    ForEachCatalog([](Store&, Catalog& catalog) {
        const std::vector<std::string> env = {"/Env0"};
        catalog.Define(SchemaId::World, env);
        const std::vector<std::string> targets = {"/Env1", "/Env2"};
        const PrimBlock block = catalog.Clone(catalog.Find("/Env0"), targets, {.moveSource = true});
        EXPECT_EQ(block.copies, 3u);
        EXPECT_EQ(block.firstWorld, 0u);
        EXPECT_EQ(catalog.GetPrims().WorldCount(), 3u);
        for (uint32_t world = 0; world < 3; ++world) {
            EXPECT_EQ(catalog.GetPrims().At(catalog.Find(fmt::format("/Env{}", world))).world,
                      world);
        }
        EXPECT_EQ(catalog.GetPrims().LiveCount(), 3u);
    });
}

TEST(CatalogTest, WritesStagedBeforeACloneReachTheClones) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        const std::vector<std::string> ball = {"/Ball"};
        const uint32_t prim = catalog.Define(SchemaId::Sphere, ball).firstPrim;
        const float radius = 9.0f;
        store.Stage(kSphereRadius, catalog.GetPrims().Row(prim, SchemaId::Sphere),
                    std::as_bytes(std::span(&radius, 1)));
        const std::vector<std::string> targets = {"/Copy"};
        const PrimBlock block = catalog.Clone(prim, targets);
        const uint32_t row = catalog.GetPrims().Row(block.Prim(0, 0), SchemaId::Sphere);
        EXPECT_EQ(Read<float>(store, kSphereRadius, row, 1), std::vector<float>{9.0f});
    });
}

TEST(CatalogTest, ClonesShareArraysUntilTheLastCopyIsRemoved) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        const std::vector<std::string> mesh = {"/Mesh"};
        const uint32_t prim =
            catalog.Define<Mesh>(mesh, {.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}}).firstPrim;
        const ArrayRef points =
            store.Array(kPointBasedPoints, catalog.GetPrims().Row(prim, SchemaId::PointBased));
        const std::vector<std::string> targets = Paths("/Copy", 2);
        const PrimBlock block = catalog.Clone(prim, targets);
        ArrayHeap& heap = store.Arrays(ColumnType::Float3Array);
        EXPECT_EQ(heap.References(points), 3u);
        catalog.Remove(prim);
        EXPECT_EQ(heap.References(points), 2u);
        const std::vector<std::byte> bytes = heap.Read(points);
        EXPECT_EQ(bytes.size(), 36u);
        catalog.Remove(block.Prim(0, 0));
        catalog.Remove(block.Prim(1, 0));
        EXPECT_EQ(heap.ArrayCount(), 0u);
    });
}

TEST(CatalogTest, TargetsToPrimsDefinedLaterResolveWhenTheyAppear) {
    ForEachCatalog([](Store& store, Catalog& catalog) {
        const std::vector<std::string> ball = {"/Ball"};
        const uint32_t prim =
            catalog.Define<Sphere>(ball, {.materialBinding = {"/Look"}}).firstPrim;
        const uint32_t row = catalog.GetPrims().Row(prim, SchemaId::MaterialBindingAPI);
        ArrayHeap& heap = store.Arrays(ColumnType::PrimArray);
        const ArrayRef before = store.Array(kMaterialBindingAPIMaterialBinding, row);
        EXPECT_EQ(Items<uint32_t>(heap.Read(before)), std::vector<uint32_t>{kNoPrim});
        const std::vector<std::string> look = {"/Look"};
        const uint32_t material = catalog.Define(SchemaId::UsdPreviewSurface, look).firstPrim;
        const ArrayRef after = store.Array(kMaterialBindingAPIMaterialBinding, row);
        EXPECT_EQ(Items<uint32_t>(heap.Read(after)), std::vector<uint32_t>{material});
    });
}

TEST(CatalogTest, AFailedDefineLeavesNothingBehind) {
    ForEachCatalog([](Store&, Catalog& catalog) {
        const std::vector<std::string> paths = {"/A", "/A"};
        EXPECT_THROW(catalog.Define(SchemaId::Sphere, paths), Error);
        const std::vector<std::string> orphan = {"/Missing/Child"};
        EXPECT_THROW(catalog.Define(SchemaId::Sphere, orphan), Error);
        EXPECT_THROW(catalog.Define(SchemaId::Xformable, std::vector<std::string>{"/X"}), Error);
        EXPECT_EQ(catalog.GetPrims().LiveCount(), 0u);
        EXPECT_EQ(catalog.Find("/A"), kNoPrim);
    });
}

TEST(CatalogTest, SelectMatchesPatternsAndSchemas) {
    ForEachCatalog([](Store&, Catalog& catalog) {
        catalog.Define(SchemaId::Xform, std::vector<std::string>{"/World"});
        catalog.Define(SchemaId::Sphere, Paths("/World/Ball", 3));
        catalog.Define(SchemaId::Cube, std::vector<std::string>{"/World/Box"});
        EXPECT_EQ(catalog.Select("/World/*", SchemaId::None).size(), 4u);
        EXPECT_EQ(catalog.Select("/World/*", SchemaId::Sphere).size(), 3u);
        EXPECT_EQ(catalog.Select("/World/Ball*", SchemaId::Gprim).size(), 3u);
    });
}

TEST(CatalogTest, FilesLiveAsLongAsAnyOwner) {
    ForEachCatalog([](Store&, Catalog& catalog) {
        const uint32_t a =
            catalog.Define(SchemaId::Sphere, std::vector<std::string>{"/A"}).firstPrim;
        const FileBytes bytes = std::make_shared<const std::vector<std::byte>>(4);
        catalog.AddFile("tex.png", bytes, a);
        const PrimBlock copy = catalog.Clone(a, std::vector<std::string>{"/B"});
        catalog.Remove(a);
        EXPECT_NE(catalog.FindFile("tex.png"), nullptr);
        catalog.Remove(copy.Prim(0, 0));
        EXPECT_EQ(catalog.FindFile("tex.png"), nullptr);
    });
}

}
}
