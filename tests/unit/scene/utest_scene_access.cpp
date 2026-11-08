// SPDX-License-Identifier: BSD-3-Clause
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/poses.h"
#include "ernie/scene/views/view.h"
#include "ernie/scene/views/view_state.h"
#include "support/devices.h"

namespace ernie {
namespace {

std::vector<std::string> Paths(std::string_view prefix, uint32_t count) {
    std::vector<std::string> paths;
    for (uint32_t i = 0; i < count; ++i) {
        paths.push_back(fmt::format("{}{}", prefix, i));
    }
    return paths;
}

std::vector<float> Radii(Structure& structure, std::span<const uint32_t> prims) {
    std::vector<float> radii;
    for (const uint32_t prim : prims) {
        const View one(structure, SchemaId::Sphere, std::vector<uint32_t>{prim});
        radii.push_back(one.Get<float>(kSphereRadius).Get().front());
    }
    return radii;
}

uint64_t RowStamp(ColumnStore& store, Structure& structure, ColumnRef column, uint32_t prim) {
    const uint32_t row = structure.GetPrims().Row(prim, column.schema);
    return Items<uint64_t>(store.Schema(column.schema).Read(store.StampColumn(column), row, 1))
        .front();
}

uint64_t WorldStamp(ColumnStore& store, uint32_t world) {
    const uint32_t slot = WorldSlot(world);
    return Items<uint64_t>(store.Worlds().Read(0, slot, 1)).front();
}

std::vector<uint32_t> Interleaved(Structure& structure, uint32_t count) {
    const PrimBlock a = structure.Define(SchemaId::Sphere, Paths("/A", count));
    const PrimBlock b = structure.Define(SchemaId::Sphere, Paths("/B", count));
    std::vector<uint32_t> prims;
    for (uint32_t i = 0; i < count; ++i) {
        prims.push_back(i % 2 == 0 ? a.Prim(i, 0) : b.Prim(i, 0));
    }
    return prims;
}

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        ColumnStore store(device);
        Structure structure(store);
        test(device, store, structure);
    });
}

TEST(ViewTest, ResolvesOnceUntilTheStructureChanges) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 8));
        const View view(structure, SchemaId::Sphere, block);
        ViewState& state = *view.State();
        const Ref<device::Buffer> first = state.DeviceRows(SchemaId::Sphere);
        EXPECT_EQ(state.DeviceRows(SchemaId::Sphere), first);
        structure.Define(SchemaId::Sphere, std::vector<std::string>{"/Other"});
        EXPECT_NE(state.DeviceRows(SchemaId::Sphere), first);
    });
}

TEST(ViewTest, ABlockIsDenseAndAMixIsNot) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 8));
        EXPECT_TRUE(View(structure, SchemaId::Sphere, block).Dense(SchemaId::Sphere));
        EXPECT_TRUE(View(structure, SchemaId::Sphere, block).Dense(SchemaId::Xformable));
        const View mixed(structure, SchemaId::Sphere, Interleaved(structure, 6));
        EXPECT_FALSE(mixed.Dense(SchemaId::Sphere));
    });
}

TEST(ViewTest, ADenseViewNeverBuildsARowIndex) {
    ForEachScene([](device::Device& device, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 8));
        const View view(structure, SchemaId::Sphere, block);
        UntypedAttribute radius = view.Get(kSphereRadius);
        const std::vector<float> radii = {1, 2, 3, 4, 5, 6, 7, 8};
        view.Get<float>(kSphereRadius).Set(radii);
        const device::Tensor gathered = radius.Read();
        EXPECT_EQ(gathered.ToHost<float>(), radii);
        if (device.SharedDevice()) {
            radius.Write(gathered.DLPack()).Wait();
        }
        radius.MarkChanged().Wait();
        std::vector<float> poses(8 * 7, 0.0f);
        int64_t shape[] = {8, 7};
        DLTensor tensor{};
        tensor.data = poses.data();
        tensor.device = {kDLCPU, 0};
        tensor.ndim = 2;
        tensor.dtype = {kDLFloat, 32, 1};
        tensor.shape = shape;
        view.WritePoses(tensor).Wait();
        view.ReadPoses();
        EXPECT_FALSE(view.State()->Rows(SchemaId::Sphere).device);
        EXPECT_FALSE(view.State()->Rows(SchemaId::Xformable).device);

        const View mixed(structure, SchemaId::Sphere, Interleaved(structure, 6));
        mixed.Get(kSphereRadius).MarkChanged().Wait();
        EXPECT_TRUE(mixed.State()->Rows(SchemaId::Sphere).device);
    });
}

TEST(ViewTest, RefusesPrimsOfAnotherSchema) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const uint32_t cube =
            structure.Define(SchemaId::Cube, std::vector<std::string>{"/Box"}).firstPrim;
        EXPECT_THROW(View(structure, SchemaId::Sphere, std::vector<uint32_t>{cube}), Error);
        const View view(structure, SchemaId::Cube, std::vector<uint32_t>{cube});
        EXPECT_THROW(view.Get<float>(kSphereRadius), Error);
        EXPECT_NO_THROW(view.Get<Double3>(kXformableTranslate));
    });
}

TEST(AttributeTest, HostWritesLandOnTheirPrimsDenseOrScattered) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 5));
        const View dense(structure, SchemaId::Sphere, block);
        dense.Get<float>(kSphereRadius).Set(std::vector<float>{1, 2, 3, 4, 5});
        EXPECT_EQ(dense.Get<float>(kSphereRadius).Get(), (std::vector<float>{1, 2, 3, 4, 5}));

        const std::vector<uint32_t> prims = Interleaved(structure, 6);
        const View mixed(structure, SchemaId::Sphere, prims);
        mixed.Get<float>(kSphereRadius).Set(std::vector<float>{6, 7, 8, 9, 10, 11});
        EXPECT_EQ(mixed.Get<float>(kSphereRadius).Get(), (std::vector<float>{6, 7, 8, 9, 10, 11}));
        EXPECT_EQ(Radii(structure, prims), (std::vector<float>{6, 7, 8, 9, 10, 11}));
        mixed.Get<float>(kSphereRadius).Set(0.5f);
        EXPECT_EQ(Radii(structure, prims), std::vector<float>(6, 0.5f));
    });
}

TEST(AttributeTest, ManyScatteredRunsLandAndStampTheirRows) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        const std::vector<uint32_t> prims = Interleaved(structure, 100);
        const View mixed(structure, SchemaId::Sphere, prims);
        const uint32_t skipped = prims[1] - 1;
        const View one(structure, SchemaId::Sphere, std::vector<uint32_t>{skipped});
        one.Get<float>(kSphereRadius).Set(7.0f);
        store.Flush();
        store.SetStamp(9);
        std::vector<float> radii(prims.size());
        std::vector<Double3> places(prims.size());
        for (uint32_t i = 0; i < radii.size(); ++i) {
            radii[i] = float(i) + 0.25f;
            places[i] = {double(i), 1, 2};
        }
        mixed.Get<float>(kSphereRadius).Set(radii);
        mixed.Get<Double3>(kXformableTranslate).Set(places);
        store.Flush();
        EXPECT_EQ(mixed.Get<float>(kSphereRadius).Get(), radii);
        EXPECT_EQ(Radii(structure, prims), radii);
        EXPECT_EQ(mixed.Get<Double3>(kXformableTranslate).Get(), places);
        EXPECT_EQ(RowStamp(store, structure, kXformableTranslate, prims[0]), 9u);
        EXPECT_EQ(RowStamp(store, structure, kXformableTranslate, prims[99]), 9u);
        EXPECT_EQ(one.Get<float>(kSphereRadius).Get().front(), 7.0f);
        EXPECT_NE(RowStamp(store, structure, kXformableTranslate, skipped), 9u);
    });
}

TEST(AttributeTest, WritesStampExactlyTheirRowsAndWorlds) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        structure.Define(SchemaId::World, std::vector<std::string>{"/Env0"});
        structure.Define(SchemaId::World, std::vector<std::string>{"/Env1"});
        const uint32_t a =
            structure.Define(SchemaId::Sphere, std::vector<std::string>{"/Env0/A"}).firstPrim;
        const uint32_t b =
            structure.Define(SchemaId::Sphere, std::vector<std::string>{"/Env1/B"}).firstPrim;
        store.SetStamp(9);
        const View view(structure, SchemaId::Sphere, std::vector<uint32_t>{a});
        view.Get<Double3>(kXformableTranslate).Set(Double3{1, 2, 3});
        store.Flush();
        EXPECT_EQ(RowStamp(store, structure, kXformableTranslate, a), 9u);
        EXPECT_EQ(RowStamp(store, structure, kXformableTranslate, b), 1u);
        EXPECT_EQ(WorldStamp(store, 0), 9u);
        EXPECT_EQ(WorldStamp(store, 1), 1u);
    });
}

TEST(AttributeTest, UntrackedWritesStillStampTheirWorlds) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        structure.Define(SchemaId::World, std::vector<std::string>{"/Env0"});
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Env0/Ball", 4));
        store.SetStamp(4);
        View(structure, SchemaId::Sphere, block).Get<float>(kSphereRadius).Set(3.0f);
        store.Flush();
        EXPECT_EQ(WorldStamp(store, 0), 4u);
        EXPECT_EQ(WorldStamp(store, kGlobalWorld), 0u);
    });
}

TEST(AttributeTest, DeviceTensorsWriteInPlaceOnEveryBackend) {
    ForEachScene([](device::Device& device, ColumnStore&, Structure& structure) {
        const View view(structure, SchemaId::Sphere,
                        structure.Define(SchemaId::Sphere, Paths("/Ball", 4)));
        const std::vector<float> radii = {1, 2, 3, 4};
        const device::Tensor values(device.CreateBuffer({.label = "radii"}, radii), 0,
                                    device::DType::Float32, {4});
        view.Get(kSphereRadius).Write(values).Wait();
        EXPECT_EQ(view.Get<float>(kSphereRadius).Get(), radii);

        const std::vector<uint64_t> bits = {0b0110};
        const device::Tensor mask(device.CreateBuffer({.label = "mask"}, bits), 0,
                                  device::DType::UInt64, {1});
        const std::vector<float> doubled = {10, 20, 30, 40};
        const device::Tensor more(device.CreateBuffer({.label = "radii"}, doubled), 0,
                                  device::DType::Float32, {4});
        view.Get(kSphereRadius).Write(more, Stream::None(), mask).Wait();
        EXPECT_EQ(view.Get<float>(kSphereRadius).Get(), (std::vector<float>{1, 20, 30, 4}));

        const View other(structure, SchemaId::Sphere, Interleaved(structure, 4));
        other.Get(kSphereRadius).Write(view.Get(kSphereRadius).Read()).Wait();
        EXPECT_EQ(other.Get<float>(kSphereRadius).Get(), (std::vector<float>{1, 20, 30, 4}));

        const device::Tensor wide(device.CreateBuffer({.label = "radii"}, doubled), 0,
                                  device::DType::Float32, {2, 2});
        EXPECT_THROW(view.Get(kSphereRadius).Write(wide), Error);
        EXPECT_THROW(view.Get(kSphereRadius).Write(device::Tensor()), Error);
    });
}

TEST(AttributeTest, AMaskedWriteLeavesOtherRowsAlone) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 4));
        const View view(structure, SchemaId::Sphere, block);
        store.SetStamp(6);
        std::vector<Double3> values = {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}};
        int64_t shape[] = {4, 3};
        DLTensor source{};
        source.data = values.data();
        source.device = {kDLCPU, 0};
        source.ndim = 2;
        source.dtype = {kDLFloat, 64, 1};
        source.shape = shape;
        uint64_t bits = 0b0101;
        int64_t maskShape[] = {1};
        DLTensor mask{};
        mask.data = &bits;
        mask.device = {kDLCPU, 0};
        mask.ndim = 1;
        mask.dtype = {kDLUInt, 64, 1};
        mask.shape = maskShape;
        view.Get(kXformableTranslate).Write(source, Stream::None(), &mask);
        const std::vector<Double3> after = view.Get<Double3>(kXformableTranslate).Get();
        EXPECT_EQ(after[0].x, 1.0);
        EXPECT_EQ(after[1].x, 0.0);
        EXPECT_EQ(after[2].x, 3.0);
        EXPECT_EQ(after[3].x, 0.0);
        EXPECT_EQ(RowStamp(store, structure, kXformableTranslate, block.Prim(1, 0)), 1u);
        EXPECT_EQ(RowStamp(store, structure, kXformableTranslate, block.Prim(2, 0)), 6u);
    });
}

TEST(AttributeTest, RemovedPrimsAreSkippedAndTheirIndexIsntReclaimed) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const uint32_t a =
            structure.Define(SchemaId::Sphere, std::vector<std::string>{"/A"}).firstPrim;
        const uint32_t b =
            structure.Define(SchemaId::Sphere, std::vector<std::string>{"/B"}).firstPrim;
        const View view(structure, SchemaId::Sphere, std::vector<uint32_t>{a, b});
        structure.Remove(a);
        const uint32_t c =
            structure.Define(SchemaId::Sphere, std::vector<std::string>{"/C"}).firstPrim;
        ASSERT_EQ(c, a);
        view.Get<float>(kSphereRadius).Set(std::vector<float>{7, 8});
        EXPECT_EQ(Radii(structure, std::vector<uint32_t>{c, b}), (std::vector<float>{1, 8}));
    });
}

TEST(AttributeTest, ArraysArentFilled) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Mesh, Paths("/Mesh", 2));
        const View view(structure, SchemaId::Mesh, block);
        const ArrayRef none{};
        EXPECT_THROW(view.Get(kPointBasedPoints).Fill(std::as_bytes(std::span(&none, 1))), Error);
        EXPECT_NO_THROW(view.Get(kPointBasedPoints).MarkChanged());
    });
}

TEST(MappingTest, ADenseMappingAliasesTheColumnAndStampsOnClose) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 300));
        const View view(structure, SchemaId::Sphere, block);
        UntypedAttribute radius = view.Get(kSphereRadius);
        store.SetStamp(5);
        Mapping mapping = radius.Map(MapAccess::Write);
        ASSERT_TRUE(mapping.Aliased());
        const uint32_t first = structure.GetPrims().Row(block.Prim(0, 0), SchemaId::Sphere);
        const device::BufferRange column =
            *store.Schema(SchemaId::Sphere).Contiguous(kSphereRadius.index, first, 300);
        EXPECT_EQ(mapping.Values().Address(), column.Address());
        std::vector<float> values(300, 2.0f);
        device.GetQueue().Write(column, values.data());
        mapping.Close();
        EXPECT_EQ(view.Get<float>(kSphereRadius).Get(), values);
        EXPECT_EQ(WorldStamp(store, kGlobalWorld), 5u);
    });
}

TEST(MappingTest, AScatteredMappingStagesAndScattersOnClose) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        const std::vector<uint32_t> prims = Interleaved(structure, 6);
        const View view(structure, SchemaId::Sphere, prims);
        view.Get<float>(kSphereRadius).Set(std::vector<float>{1, 2, 3, 4, 5, 6});
        store.SetStamp(8);
        Mapping mapping = view.Get(kXformableTranslate).Map(MapAccess::ReadWrite);
        ASSERT_FALSE(mapping.Aliased());
        EXPECT_EQ(mapping.Rows().Shape()[0], 6);
        Mapping read = view.Get(kSphereRadius).Map(MapAccess::Read);
        EXPECT_EQ(read.Values().ToHost<float>(), (std::vector<float>{1, 2, 3, 4, 5, 6}));
        read.Close();
        std::vector<double> values(18);
        for (uint32_t i = 0; i < 6; ++i) {
            values[i * 3] = i + 10.0;
        }
        device.GetQueue().Write(mapping.Values().GetBuffer(), values.data());
        mapping.Close();
        const std::vector<Double3> after = view.Get<Double3>(kXformableTranslate).Get();
        for (uint32_t i = 0; i < 6; ++i) {
            EXPECT_EQ(after[i].x, i + 10.0);
            EXPECT_EQ(RowStamp(store, structure, kXformableTranslate, prims[i]), 8u);
        }
    });
}

TEST(MappingTest, AnOpenMappingBlocksRemovalUntilItCloses) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 3));
        const View view(structure, SchemaId::Sphere, block);
        {
            Mapping mapping = view.Get(kSphereRadius).Map(MapAccess::Write);
            EXPECT_THROW(structure.Remove(block.Prim(0, 0)), Error);
            structure.Define(SchemaId::Sphere, Paths("/More", 600));
            mapping.Close();
        }
        EXPECT_NO_THROW(structure.Remove(block.Prim(0, 0)));
        {
            Mapping dropped = view.Get(kSphereRadius).Map(MapAccess::Write);
        }
        EXPECT_NO_THROW(structure.Remove(block.Prim(1, 0)));
        Mapping moved = view.Get(kSphereRadius).Map(MapAccess::Read);
        Mapping target = std::move(moved);
        EXPECT_FALSE(moved.Open());
        EXPECT_THROW(structure.Remove(block.Prim(2, 0)), Error);
        target.Close();
        EXPECT_THROW(target.Close(), Error);
    });
}

TEST(AttributeTest, DeviceTensorsWriteAndReadWithoutTheHost) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        if (!device.SharedDevice()) {
            return;
        }
        const PrimBlock source = structure.Define(SchemaId::Sphere, Paths("/Source", 6));
        const View from(structure, SchemaId::Sphere, source);
        from.Get<float>(kSphereRadius).Set(std::vector<float>{1, 2, 3, 4, 5, 6});
        const device::Tensor gathered = from.Get(kSphereRadius).Read();
        const std::vector<uint32_t> prims = Interleaved(structure, 6);
        const View to(structure, SchemaId::Sphere, prims);
        store.SetStamp(12);
        to.Get(kSphereRadius).Write(gathered.DLPack()).Wait();
        EXPECT_EQ(Radii(structure, prims), (std::vector<float>{1, 2, 3, 4, 5, 6}));
        EXPECT_EQ(WorldStamp(store, kGlobalWorld), 12u);
    });
}

DLTensor HostTensor(void* data, int64_t* shape, uint8_t bits) {
    DLTensor tensor{};
    tensor.data = data;
    tensor.device = {kDLCPU, 0};
    tensor.ndim = 2;
    tensor.dtype = {kDLFloat, bits, 1};
    tensor.shape = shape;
    return tensor;
}

const std::vector<Float3> kTriangle = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};

TEST(ArrayAttributeTest, ArraysAreSetAndReadPerPrim) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Mesh, Paths("/Mesh", 2));
        const View view(structure, SchemaId::Mesh, block);
        ArrayAttribute<Float3> points = view.GetArray<Float3>(kPointBasedPoints);
        points.Set(1, kTriangle);
        EXPECT_TRUE(points.Get(0).empty());
        EXPECT_EQ(points.Get(1).size(), 3u);
        EXPECT_EQ(points.Get(1)[1].x, 1.0f);
        EXPECT_EQ(store.ArrayRows(kPointBasedPoints, 0, ~uint64_t(0)).size(), 1u);
        EXPECT_THROW(view.GetArray<int32_t>(kPointBasedPoints), Error);
    });
}

TEST(ArrayAttributeTest, AWriteMappingUnsharesClonesIntoOneRun) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        const uint32_t mesh =
            structure.Define<Mesh>(std::vector<std::string>{"/Mesh"}, {.points = kTriangle})
                .firstPrim;
        const PrimBlock copies = structure.Clone(mesh, Paths("/Copy", 3));
        std::vector<uint32_t> prims = {mesh};
        for (uint32_t copy = 0; copy < 3; ++copy) {
            prims.push_back(copies.Prim(copy, 0));
        }
        const View view(structure, SchemaId::Mesh, prims);
        ArrayHeap& heap = store.Arrays(ColumnType::Float3Array);
        const ArrayRef shared =
            store.Array(kPointBasedPoints, structure.GetPrims().Row(mesh, SchemaId::PointBased));
        ASSERT_EQ(heap.References(shared), 4u);

        Mapping mapping = view.Get(kPointBasedPoints).Map(MapAccess::Write);
        ASSERT_TRUE(mapping.Aliased());
        EXPECT_EQ(mapping.Values().Shape()[0], 4);
        EXPECT_EQ(mapping.Values().Shape()[1], 3);
        EXPECT_EQ(mapping.Values().Shape()[2], 3);
        EXPECT_EQ(heap.ArrayCount(), 4u);
        std::vector<Float3> moved = kTriangle;
        moved[0].z = 5;
        const ArrayRef second = store.Array(
            kPointBasedPoints, structure.GetPrims().Row(prims[1], SchemaId::PointBased));
        EXPECT_EQ(heap.References(second), 1u);
        device.GetQueue().Write(heap.Range(second).Slice(0, 36), moved.data());
        const Ordinal stamp = store.Stamp();
        mapping.Close();
        EXPECT_EQ(store.ArrayRows(kPointBasedPoints, stamp - 1, stamp).size(), 4u);
        ArrayAttribute<Float3> points = view.GetArray<Float3>(kPointBasedPoints);
        EXPECT_EQ(points.Get(1)[0].z, 5.0f);
        EXPECT_EQ(points.Get(0)[0].z, 0.0f);
        EXPECT_EQ(points.Get(3)[1].x, 1.0f);
    });
}

TEST(ArrayAttributeTest, ArraysOfDifferentLengthsMapAsPackedValuesWithOffsets) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Mesh, Paths("/Mesh", 3));
        const View view(structure, SchemaId::Mesh, block);
        ArrayAttribute<Float3> points = view.GetArray<Float3>(kPointBasedPoints);
        points.Set(0, kTriangle);
        points.Set(2, std::vector<Float3>(4, Float3{2, 2, 2}));

        Mapping read = view.Get(kPointBasedPoints).Map(MapAccess::Read);
        EXPECT_FALSE(read.Aliased());
        EXPECT_EQ(read.Offsets().ToHost<uint32_t>(), (std::vector<uint32_t>{0, 3, 3, 7}));
        ASSERT_EQ(read.Values().Shape().size(), 2u);
        EXPECT_EQ(read.Values().Shape()[0], 7);
        const std::vector<float> values = read.Values().ToHost<float>();
        EXPECT_EQ(values[3], 1.0f);
        EXPECT_EQ(values[20], 2.0f);
        read.Close();

        Mapping write = view.Get(kPointBasedPoints).Map(MapAccess::Write);
        EXPECT_TRUE(write.Aliased());
        std::vector<Float3> moved(7, Float3{9, 9, 9});
        device.GetQueue().Write(write.Values().GetBuffer().Slice(write.Values().Offset(), 84),
                                moved.data());
        const Ordinal stamp = store.Stamp();
        write.Close();
        EXPECT_EQ(store.ArrayRows(kPointBasedPoints, stamp - 1, stamp).size(), 3u);
        EXPECT_EQ(points.Get(0), std::vector<Float3>(3, Float3{9, 9, 9}));
        EXPECT_TRUE(points.Get(1).empty());
        EXPECT_EQ(points.Get(2), std::vector<Float3>(4, Float3{9, 9, 9}));

        Mapping again = view.Get(kPointBasedPoints).Map(MapAccess::ReadWrite);
        EXPECT_TRUE(again.Aliased());
        EXPECT_EQ(again.Values().Address(), write.Values().Address());
        again.Close();
    });
}

TEST(ArrayAttributeTest, AReadMappingLeavesClonesShared) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        const uint32_t mesh =
            structure.Define<Mesh>(std::vector<std::string>{"/Mesh"}, {.points = kTriangle})
                .firstPrim;
        const PrimBlock copies = structure.Clone(mesh, Paths("/Copy", 2));
        const View view(structure, SchemaId::Mesh,
                        std::vector<uint32_t>{mesh, copies.Prim(0, 0), copies.Prim(1, 0)});
        Mapping mapping = view.Get(kPointBasedPoints).Map(MapAccess::Read);
        EXPECT_FALSE(mapping.Aliased());
        EXPECT_EQ(mapping.Values().Shape()[0], 3);
        EXPECT_EQ(mapping.Values().Shape()[1], 3);
        EXPECT_TRUE(mapping.Offsets().Empty());
        mapping.Close();
        const ArrayRef shared =
            store.Array(kPointBasedPoints, structure.GetPrims().Row(mesh, SchemaId::PointBased));
        EXPECT_EQ(store.Arrays(ColumnType::Float3Array).References(shared), 3u);
    });
}

TEST(PosesTest, Float32AndFloat64PosesLandExactlyAndStampBothColumns) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 2));
        const View view(structure, SchemaId::Sphere, block);
        store.SetStamp(4);
        std::vector<float> narrow = {0.1f, 2, 3, 0, 0, 0, 1, -4, 5.5f, 1e7f, 0, 1, 0, 0};
        int64_t shape[] = {2, 7};
        view.WritePoses(HostTensor(narrow.data(), shape, 32));
        const std::vector<Double3> translates = view.Get<Double3>(kXformableTranslate).Get();
        const std::vector<Quatd> orients = view.Get<Quatd>(kXformableOrient).Get();
        EXPECT_EQ(translates[0].x, double(0.1f));
        EXPECT_EQ(translates[1].z, 1e7);
        EXPECT_EQ(orients[0].w, 1.0);
        EXPECT_EQ(orients[1].y, 1.0);
        EXPECT_EQ(RowStamp(store, structure, kXformableTranslate, block.Prim(1, 0)), 4u);
        EXPECT_EQ(RowStamp(store, structure, kXformableOrient, block.Prim(1, 0)), 4u);

        std::vector<double> wide = {1e9 + 0.25, 0, 0, 0, 0, 0, 1, 0, 0, -6371000.125, 1, 0, 0, 0};
        view.WritePoses(HostTensor(wide.data(), shape, 64));
        EXPECT_EQ(view.Get<Double3>(kXformableTranslate).Get()[0].x, 1e9 + 0.25);
        const device::Tensor gathered = view.ReadPoses(device::DType::Float64);
        EXPECT_EQ(gathered.ToHost<double>(), wide);
        const std::vector<float> narrowed = view.ReadPoses().ToHost<float>();
        EXPECT_EQ(narrowed[9], -6371000.125f);
    });
}

TEST(PosesTest, WxyzPosesLandAsXyzwAndReadBackInEitherOrder) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 2));
        const View view(structure, SchemaId::Sphere, block);
        int64_t shape[] = {2, 7};
        std::vector<float> narrow = {1, 2, 3, 0.5f, 0.1f, 0.2f, 0.3f, 4, 5, 6, 1, 0, 0, 0};
        view.WritePoses(HostTensor(narrow.data(), shape, 32), Stream::None(),
                        QuaternionOrder::Wxyz);
        const std::vector<Quatd> orients = view.Get<Quatd>(kXformableOrient).Get();
        EXPECT_EQ(orients[0].x, double(0.1f));
        EXPECT_EQ(orients[0].y, double(0.2f));
        EXPECT_EQ(orients[0].z, double(0.3f));
        EXPECT_EQ(orients[0].w, 0.5);
        EXPECT_EQ(orients[1].w, 1.0);
        EXPECT_EQ(view.Get<Double3>(kXformableTranslate).Get()[1].y, 5.0);
        EXPECT_EQ(view.ReadPoses(device::DType::Float32, Stream::Host(), QuaternionOrder::Wxyz)
                      .ToHost<float>(),
                  narrow);
        const std::vector<float> xyzw = view.ReadPoses().ToHost<float>();
        EXPECT_EQ(std::vector<float>(xyzw.begin() + 3, xyzw.begin() + 7),
                  (std::vector<float>{0.1f, 0.2f, 0.3f, 0.5f}));

        std::vector<double> wide = {0, 0, 0, 0.25, 0.5, 0.75, 1, 0, 0, 0, 0, 0, 0, 1};
        view.WritePoses(HostTensor(wide.data(), shape, 64), Stream::None(), QuaternionOrder::Wxyz);
        EXPECT_EQ(view.Get<Quatd>(kXformableOrient).Get()[0].w, 0.25);
        EXPECT_EQ(view.ReadPoses(device::DType::Float64, Stream::Host(), QuaternionOrder::Wxyz)
                      .ToHost<double>(),
                  wide);
    });
}

TEST(PosesTest, SplitSourcesAndRowsGatherPosesFromBodyBuffers) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 2));
        const View view(structure, SchemaId::Sphere, block);
        std::vector<float> positions = {0, 0, 0, 1, 2, 3, 4, 5, 6};
        std::vector<float> rotations = {1, 0, 0, 0, 0.5f, 0.1f, 0.2f, 0.3f, 0, 0, 0, 1};
        std::vector<int32_t> rows = {2, 1};
        int64_t translationShape[] = {3, 3};
        int64_t rotationShape[] = {3, 4};
        int64_t rowShape[] = {2};
        DLTensor indices{};
        indices.data = rows.data();
        indices.device = {kDLCPU, 0};
        indices.ndim = 1;
        indices.dtype = {kDLInt, 32, 1};
        indices.shape = rowShape;
        view.WritePoses(HostTensor(positions.data(), translationShape, 32),
                        HostTensor(rotations.data(), rotationShape, 32), Stream::None(),
                        QuaternionOrder::Wxyz, &indices);
        const std::vector<Double3> translates = view.Get<Double3>(kXformableTranslate).Get();
        const std::vector<Quatd> orients = view.Get<Quatd>(kXformableOrient).Get();
        EXPECT_EQ(translates[0], (Double3{4, 5, 6}));
        EXPECT_EQ(translates[1], (Double3{1, 2, 3}));
        EXPECT_EQ(orients[0], (Quatd{0, 0, 1, 0}));
        EXPECT_EQ(orients[1].x, double(0.1f));
        EXPECT_EQ(orients[1].w, 0.5);

        std::vector<float> packed(3 * 7, 0.0f);
        int64_t packedShape[] = {3, 7};
        EXPECT_THROW(view.WritePoses(HostTensor(packed.data(), packedShape, 32)), Error);
        packed[0 * 7 + 6] = packed[1 * 7 + 6] = packed[2 * 7 + 6] = 1;
        packed[2 * 7 + 0] = 9;
        view.WritePoses(HostTensor(packed.data(), packedShape, 32), Stream::None(),
                        QuaternionOrder::Xyzw, &indices);
        EXPECT_EQ(view.Get<Double3>(kXformableTranslate).Get()[0].x, 9.0);
    });
}

TEST(PosesTest, AMaskedPoseWriteLeavesOtherPrimsAlone) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 2));
        const View view(structure, SchemaId::Sphere, block);
        std::vector<float> poses = {1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 1};
        int64_t shape[] = {2, 7};
        uint64_t bits = 0b10;
        int64_t maskShape[] = {1};
        DLTensor mask{};
        mask.data = &bits;
        mask.device = {kDLCPU, 0};
        mask.ndim = 1;
        mask.dtype = {kDLUInt, 64, 1};
        mask.shape = maskShape;
        view.WritePoses(HostTensor(poses.data(), shape, 32), Stream::None(), QuaternionOrder::Xyzw,
                        nullptr, &mask);
        const std::vector<Double3> translates = view.Get<Double3>(kXformableTranslate).Get();
        EXPECT_EQ(translates[0].x, 0.0);
        EXPECT_EQ(translates[1].x, 2.0);
    });
}

TEST(PosesTest, DevicePosesWriteWithoutTheHost) {
    ForEachScene([](device::Device& device, ColumnStore&, Structure& structure) {
        const PrimBlock from = structure.Define(SchemaId::Sphere, Paths("/From", 3));
        const View source(structure, SchemaId::Sphere, from);
        std::vector<float> poses(21, 0);
        for (uint32_t i = 0; i < 3; ++i) {
            poses[i * 7] = float(i + 1);
            poses[i * 7 + 6] = 1;
        }
        int64_t shape[] = {3, 7};
        source.WritePoses(HostTensor(poses.data(), shape, 32));
        const device::Tensor gathered = source.ReadPoses();
        const View target(structure, SchemaId::Sphere, Interleaved(structure, 3));
        target.WritePoses(gathered).Wait();
        EXPECT_EQ(target.Get<Double3>(kXformableTranslate).Get()[2].x, 3.0);
        EXPECT_THROW(target.WritePoses(gathered, gathered), Error);
        if (device.SharedDevice()) {
            const View again(structure, SchemaId::Sphere,
                             structure.Define(SchemaId::Sphere, Paths("/Again", 3)));
            again.WritePoses(gathered.DLPack()).Wait();
            EXPECT_EQ(again.Get<Double3>(kXformableTranslate).Get()[2].x, 3.0);
        }
    });
}
}
}
