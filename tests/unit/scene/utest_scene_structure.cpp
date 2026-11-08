// SPDX-License-Identifier: BSD-3-Clause
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/structure/structure.h"
#include "support/devices.h"

namespace ernie {
namespace {

template <class T>
std::vector<T> Read(ColumnStore& store, ColumnRef column, uint32_t first, uint32_t count) {
    const std::vector<std::byte> bytes =
        store.Schema(column.schema).Read(column.index, first, count);
    std::vector<T> cells(bytes.size() / sizeof(T));
    std::memcpy(cells.data(), bytes.data(), bytes.size());
    return cells;
}

std::vector<uint64_t> Stamps(ColumnStore& store, ColumnRef column, uint32_t first, uint32_t count) {
    const std::vector<std::byte> bytes =
        store.Schema(column.schema).Read(store.StampColumn(column), first, count);
    std::vector<uint64_t> stamps(count);
    std::memcpy(stamps.data(), bytes.data(), bytes.size());
    return stamps;
}

std::vector<std::string> Paths(std::string_view prefix, uint32_t count) {
    std::vector<std::string> paths;
    for (uint32_t i = 0; i < count; ++i) {
        paths.push_back(std::format("{}{}", prefix, i));
    }
    return paths;
}

template <class Test> void ForEachCatalog(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        ColumnStore store(device);
        Structure structure(store);
        test(store, structure);
    });
}

TEST(StructureTest, ADefinedPrimHasItsRecordPathAndFallbacks) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        const std::vector<std::string> paths = {"/Ball"};
        const PrimBlock block = structure.Define(SchemaId::Sphere, paths);
        const uint32_t prim = block.firstPrim;
        EXPECT_EQ(structure.Find("/Ball"), prim);
        const PrimRecord& record = structure.GetPrims().At(prim);
        EXPECT_EQ(record.schema, SchemaId::Sphere);
        EXPECT_EQ(record.parent, kNoPrim);
        EXPECT_EQ(record.world, kGlobalWorld);
        const uint32_t sphere = structure.GetPrims().Row(prim, SchemaId::Sphere);
        const uint32_t xform = structure.GetPrims().Row(prim, SchemaId::Xformable);
        EXPECT_EQ(Read<float>(store, kSphereRadius, sphere, 1), std::vector<float>{1.0f});
        EXPECT_EQ(Read<float>(store, kXformableScale, xform, 1),
                  (std::vector<float>{1.0f, 1.0f, 1.0f}));
        EXPECT_EQ(Read<uint32_t>(store,
                                 {SchemaId::Sphere, nullptr, store.PrimsColumn(SchemaId::Sphere)},
                                 sphere, 1),
                  std::vector<uint32_t>{prim});
    });
}

TEST(StructureTest, ABatchGetsContiguousRowsInEveryTable) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        const std::vector<std::string> paths = Paths("/Ball", 300);
        const PrimBlock block = structure.Define(SchemaId::Sphere, paths);
        ASSERT_EQ(block.copies, 300u);
        for (const SchemaId table : {SchemaId::Sphere, SchemaId::Xformable, SchemaId::Imageable}) {
            const uint32_t first = structure.GetPrims().Row(block.Prim(0, 0), table);
            for (uint32_t i = 0; i < 300; ++i) {
                EXPECT_EQ(structure.GetPrims().Row(block.Prim(i, 0), table), first + i);
            }
            EXPECT_TRUE(store.Schema(table).Contiguous(0, first, 300));
        }
    });
}

TEST(StructureTest, BroadcastValuesLandOnEveryPrimWhetherStagedOrFilled) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        for (const uint32_t count : {4u, 12000u}) {
            const std::vector<std::string> paths = Paths(std::format("/Batch{}_", count), count);
            const PrimBlock block =
                structure.Define<Sphere>(paths, {.translate = {1.0, 2.0, 3.0}, .radius = 2.5f});
            const uint32_t sphere = structure.GetPrims().Row(block.Prim(0, 0), SchemaId::Sphere);
            const uint32_t xform = structure.GetPrims().Row(block.Prim(0, 0), SchemaId::Xformable);
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

TEST(StructureTest, DefiningStampsTrackedRowsAndTheirWorld) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        store.SetStamp(7);
        const std::vector<std::string> paths = Paths("/Ball", 3);
        const PrimBlock block = structure.Define(SchemaId::Sphere, paths);
        const uint32_t xform = structure.GetPrims().Row(block.Prim(0, 0), SchemaId::Xformable);
        EXPECT_EQ(Stamps(store, kXformableTranslate, xform, 3), (std::vector<uint64_t>{7, 7, 7}));
        const std::vector<std::byte> world = store.Worlds().Read(0, 0, 1);
        uint64_t global = 0;
        std::memcpy(&global, world.data(), sizeof(global));
        EXPECT_EQ(global, 7u);
        EXPECT_EQ(structure.Log().size(), 3u);
        EXPECT_TRUE(structure.Log()[0].added);
        EXPECT_EQ(structure.Log()[0].ordinal, 7u);
    });
}

TEST(StructureTest, ArraysAreSharedAcrossABatchAndFreedWithTheLastPrim) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        const std::vector<std::string> paths = Paths("/Mesh", 3);
        const PrimBlock block =
            structure.Define<Mesh>(paths, {.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
                                           .faceVertexIndices = {0, 1, 2},
                                           .faceVertexCounts = {3}});
        std::vector<ArrayRef> arrays;
        for (uint32_t i = 0; i < 3; ++i) {
            arrays.push_back(
                store.Array(kPointBasedPoints,
                            structure.GetPrims().Row(block.Prim(i, 0), SchemaId::PointBased)));
        }
        EXPECT_NE(arrays[0].address, 0u);
        EXPECT_EQ(arrays[1].address, arrays[0].address);
        EXPECT_EQ(arrays[0].count, 3u);
        ArrayHeap& heap = store.Arrays(ColumnType::Float3Array);
        EXPECT_EQ(heap.References(arrays[0]), 3u);
        structure.Remove(block.Prim(0, 0));
        EXPECT_EQ(heap.References(arrays[0]), 2u);
        structure.Remove(block.Prim(1, 0));
        structure.Remove(block.Prim(2, 0));
        EXPECT_EQ(heap.ArrayCount(), 0u);
    });
}

TEST(StructureTest, RemovedPrimsComeBackWithANewGeneration) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        const std::vector<std::string> first = {"/A"};
        const uint32_t prim = structure.Define(SchemaId::Sphere, first).firstPrim;
        const uint32_t row = structure.GetPrims().Row(prim, SchemaId::Sphere);
        const uint32_t generation = structure.GetPrims().Generation(prim);
        structure.Remove(prim);
        EXPECT_FALSE(structure.GetPrims().Alive(prim));
        EXPECT_EQ(structure.Find("/A"), kNoPrim);
        EXPECT_EQ(Read<uint32_t>(store,
                                 {SchemaId::Sphere, nullptr, store.PrimsColumn(SchemaId::Sphere)},
                                 row, 1),
                  std::vector<uint32_t>{kNoPrim});

        const std::vector<std::string> second = {"/B"};
        const uint32_t again = structure.Define(SchemaId::Sphere, second).firstPrim;
        EXPECT_EQ(again, prim);
        EXPECT_EQ(structure.GetPrims().Generation(again), generation + 1);
        EXPECT_EQ(structure.GetPrims().Row(again, SchemaId::Sphere), row);
    });
}

TEST(StructureTest, RemovingAPrimRemovesItsSubtree) {
    ForEachCatalog([](ColumnStore&, Structure& structure) {
        for (const char* path : {"/Robot", "/Robot/Arm", "/Robot/Arm/Hand"}) {
            const std::vector<std::string> paths = {path};
            structure.Define(SchemaId::Xform, paths);
        }
        structure.Remove(structure.Find("/Robot"));
        EXPECT_EQ(structure.GetPrims().LiveCount(), 0u);
        EXPECT_EQ(structure.Find("/Robot/Arm/Hand"), kNoPrim);
    });
}

TEST(StructureTest, WorldsHoldTheirSubtreesAndDontNest) {
    ForEachCatalog([](ColumnStore&, Structure& structure) {
        for (const char* path : {"/Env", "/Env/Ball"}) {
            const std::vector<std::string> paths = {path};
            structure.Define(std::string_view(path) == "/Env" ? SchemaId::World : SchemaId::Sphere,
                             paths);
        }
        const std::vector<std::string> sun = {"/Sun"};
        structure.Define(SchemaId::Sphere, sun);
        EXPECT_EQ(structure.GetPrims().At(structure.Find("/Env")).world, 0u);
        EXPECT_EQ(structure.GetPrims().At(structure.Find("/Env/Ball")).world, 0u);
        EXPECT_EQ(structure.GetPrims().At(structure.Find("/Sun")).world, kGlobalWorld);
        const std::vector<std::string> nested = {"/Env/Inner"};
        EXPECT_THROW(structure.Define(SchemaId::World, nested), Error);
        EXPECT_EQ(structure.GetPrims().WorldCount(), 1u);
    });
}

TEST(StructureTest, ClonesCopyValuesIntoNewWorlds) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        const std::vector<std::string> env = {"/Env0"};
        structure.Define(SchemaId::World, env);
        const std::vector<std::string> ball = {"/Env0/Ball"};
        structure.Define<Sphere>(ball, {.radius = 4.0f});
        store.SetStamp(3);
        const std::vector<std::string> targets = Paths("/Env", 4);
        const std::vector<std::string> fresh(targets.begin() + 1, targets.end());
        const PrimBlock block = structure.Clone(structure.Find("/Env0"), fresh);
        EXPECT_EQ(block.count, 2u);
        EXPECT_EQ(block.copies, 3u);
        EXPECT_EQ(block.firstWorld, 1u);
        EXPECT_EQ(structure.GetPrims().WorldCount(), 4u);
        for (uint32_t copy = 0; copy < 3; ++copy) {
            const uint32_t prim = structure.Find(std::format("/Env{}/Ball", copy + 1));
            ASSERT_EQ(prim, block.Prim(copy, 1));
            EXPECT_EQ(structure.GetPrims().At(prim).world, copy + 1);
            const uint32_t row = structure.GetPrims().Row(prim, SchemaId::Sphere);
            EXPECT_EQ(Read<float>(store, kSphereRadius, row, 1), std::vector<float>{4.0f});
            EXPECT_EQ(Stamps(store, kXformableTranslate,
                             structure.GetPrims().Row(prim, SchemaId::Xformable), 1),
                      std::vector<uint64_t>{3});
        }
    });
}

TEST(StructureTest, MovingTheSourceKeepsItsWorldNumbers) {
    ForEachCatalog([](ColumnStore&, Structure& structure) {
        const std::vector<std::string> env = {"/Env0"};
        structure.Define(SchemaId::World, env);
        const std::vector<std::string> targets = {"/Env1", "/Env2"};
        const PrimBlock block =
            structure.Clone(structure.Find("/Env0"), targets, {.moveSource = true});
        EXPECT_EQ(block.copies, 3u);
        EXPECT_EQ(block.firstWorld, 0u);
        EXPECT_EQ(structure.GetPrims().WorldCount(), 3u);
        for (uint32_t world = 0; world < 3; ++world) {
            EXPECT_EQ(structure.GetPrims().At(structure.Find(std::format("/Env{}", world))).world,
                      world);
        }
        EXPECT_EQ(structure.GetPrims().LiveCount(), 3u);
    });
}

TEST(StructureTest, WritesStagedBeforeACloneReachTheClones) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        const std::vector<std::string> ball = {"/Ball"};
        const uint32_t prim = structure.Define(SchemaId::Sphere, ball).firstPrim;
        const float radius = 9.0f;
        store.Stage(kSphereRadius, structure.GetPrims().Row(prim, SchemaId::Sphere),
                    std::as_bytes(std::span(&radius, 1)));
        const std::vector<std::string> targets = {"/Copy"};
        const PrimBlock block = structure.Clone(prim, targets);
        const uint32_t row = structure.GetPrims().Row(block.Prim(0, 0), SchemaId::Sphere);
        EXPECT_EQ(Read<float>(store, kSphereRadius, row, 1), std::vector<float>{9.0f});
    });
}

TEST(StructureTest, ClonesShareArraysUntilTheLastCopyIsRemoved) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        const std::vector<std::string> mesh = {"/Mesh"};
        const uint32_t prim =
            structure.Define<Mesh>(mesh, {.points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}}).firstPrim;
        const ArrayRef points =
            store.Array(kPointBasedPoints, structure.GetPrims().Row(prim, SchemaId::PointBased));
        const std::vector<std::string> targets = Paths("/Copy", 2);
        const PrimBlock block = structure.Clone(prim, targets);
        ArrayHeap& heap = store.Arrays(ColumnType::Float3Array);
        EXPECT_EQ(heap.References(points), 3u);
        structure.Remove(prim);
        EXPECT_EQ(heap.References(points), 2u);
        const std::vector<std::byte> bytes = heap.Read(points);
        EXPECT_EQ(bytes.size(), 36u);
        structure.Remove(block.Prim(0, 0));
        structure.Remove(block.Prim(1, 0));
        EXPECT_EQ(heap.ArrayCount(), 0u);
    });
}

TEST(StructureTest, TargetsToPrimsDefinedLaterResolveWhenTheyAppear) {
    ForEachCatalog([](ColumnStore& store, Structure& structure) {
        const std::vector<std::string> ball = {"/Ball"};
        const uint32_t prim =
            structure.Define<Sphere>(ball, {.materialBinding = {"/Look"}}).firstPrim;
        const uint32_t row = structure.GetPrims().Row(prim, SchemaId::MaterialBindingAPI);
        ArrayHeap& heap = store.Arrays(ColumnType::PrimArray);
        const ArrayRef before = store.Array(kMaterialBindingAPIMaterialBinding, row);
        EXPECT_EQ(Items<uint32_t>(heap.Read(before)), std::vector<uint32_t>{kNoPrim});
        const std::vector<std::string> look = {"/Look"};
        const uint32_t material = structure.Define(SchemaId::UsdPreviewSurface, look).firstPrim;
        const ArrayRef after = store.Array(kMaterialBindingAPIMaterialBinding, row);
        EXPECT_EQ(Items<uint32_t>(heap.Read(after)), std::vector<uint32_t>{material});
    });
}

TEST(StructureTest, AFailedDefineLeavesNothingBehind) {
    ForEachCatalog([](ColumnStore&, Structure& structure) {
        const std::vector<std::string> paths = {"/A", "/A"};
        EXPECT_THROW(structure.Define(SchemaId::Sphere, paths), Error);
        const std::vector<std::string> orphan = {"/Missing/Child"};
        EXPECT_THROW(structure.Define(SchemaId::Sphere, orphan), Error);
        EXPECT_THROW(structure.Define(SchemaId::Xformable, std::vector<std::string>{"/X"}), Error);
        EXPECT_EQ(structure.GetPrims().LiveCount(), 0u);
        EXPECT_EQ(structure.Find("/A"), kNoPrim);
    });
}

TEST(StructureTest, SelectMatchesPatternsAndSchemas) {
    ForEachCatalog([](ColumnStore&, Structure& structure) {
        structure.Define(SchemaId::Xform, std::vector<std::string>{"/World"});
        structure.Define(SchemaId::Sphere, Paths("/World/Ball", 3));
        structure.Define(SchemaId::Cube, std::vector<std::string>{"/World/Box"});
        EXPECT_EQ(structure.Select("/World/*", SchemaId::None).size(), 4u);
        EXPECT_EQ(structure.Select("/World/*", SchemaId::Sphere).size(), 3u);
        EXPECT_EQ(structure.Select("/World/Ball*", SchemaId::Gprim).size(), 3u);
    });
}

TEST(StructureTest, FilesLiveAsLongAsAnyOwner) {
    ForEachCatalog([](ColumnStore&, Structure& structure) {
        const uint32_t a =
            structure.Define(SchemaId::Sphere, std::vector<std::string>{"/A"}).firstPrim;
        const FileBytes bytes = std::make_shared<const std::vector<std::byte>>(4);
        structure.AddFile("tex.png", bytes, a);
        const PrimBlock copy = structure.Clone(a, std::vector<std::string>{"/B"});
        structure.Remove(a);
        EXPECT_NE(structure.FindFile("tex.png"), nullptr);
        structure.Remove(copy.Prim(0, 0));
        EXPECT_EQ(structure.FindFile("tex.png"), nullptr);
    });
}

}
}
