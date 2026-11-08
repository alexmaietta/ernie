// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "ernie/device/buffer.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/schema/columns.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/views/attribute.h"
#include "ernie/scene/views/view.h"
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

struct Owned {
    DLManagedTensor managed{};
    std::vector<float> values;
    Ref<device::Buffer> buffer;
    int64_t shape[1] = {0};
    int* deletions = nullptr;
};

DLManagedTensor* Manage(Owned* owned, void* data, DLDevice device, int64_t rows, int& deletions) {
    owned->shape[0] = rows;
    owned->deletions = &deletions;
    owned->managed.manager_ctx = owned;
    owned->managed.dl_tensor.data = data;
    owned->managed.dl_tensor.device = device;
    owned->managed.dl_tensor.ndim = 1;
    owned->managed.dl_tensor.dtype = {kDLFloat, 32, 1};
    owned->managed.dl_tensor.shape = owned->shape;
    owned->managed.deleter = [](DLManagedTensor* managed) {
        Owned* self = static_cast<Owned*>(managed->manager_ctx);
        ++*self->deletions;
        delete self;
    };
    return &owned->managed;
}

DLManagedTensor* HostTensor(std::vector<float> values, int& deletions) {
    Owned* owned = new Owned;
    owned->values = std::move(values);
    return Manage(owned, owned->values.data(), {kDLCPU, 0}, int64_t(owned->values.size()),
                  deletions);
}

std::vector<float>& HostValues(DLManagedTensor* tensor) {
    return static_cast<Owned*>(tensor->manager_ctx)->values;
}

bool InPlace(device::Device& device) {
    const device::Backend backend = device.Adapter().backend;
    return backend == device::Backend::Cuda || backend == device::Backend::Hip;
}

DLManagedTensor* DeviceTensor(device::Device& device, uint32_t rows, int& deletions) {
    Owned* owned = new Owned;
    owned->buffer = device.CreateBuffer(
        {.bytes = rows * sizeof(float), .usage = device::BufferUsage::Interop, .label = "user"});
    const DLDeviceType api = device.Adapter().backend == device::Backend::Cuda ? kDLCUDA : kDLROCM;
    const device::ExternalAddress address = owned->buffer->External(api);
    return Manage(owned, reinterpret_cast<void*>(address.address), address.device, rows, deletions);
}

uint64_t GlobalStamp(ColumnStore& store) {
    return Items<uint64_t>(store.Worlds().Read(0, 0, 1)).front();
}

PrimBlock Attachable(Structure& structure, std::string_view prefix, uint32_t count) {
    return structure.Define(SchemaId::Sphere, Paths(prefix, count), [](RowWriter&) {}, true);
}

struct Moved {
    uint64_t copied = 0;
    uint64_t uploaded = 0;
    uint64_t readBack = 0;
    int64_t allocated = 0;
};

template <class Work> Moved Measure(ColumnStore& store, Work work) {
    device::Device& device = store.GetDevice();
    store.Flush();
    store.DeviceData();
    device.GetQueue().Mark().Wait();
    const device::TrafficStats before = device.Traffic();
    const uint64_t memory = device.MemoryUsage().deviceBytes;
    work();
    store.Flush();
    device.GetQueue().Mark().Wait();
    const device::TrafficStats after = device.Traffic();
    return {after.copyBytes - before.copyBytes, after.uploadBytes - before.uploadBytes,
            after.readbackBytes - before.readbackBytes,
            int64_t(device.MemoryUsage().deviceBytes) - int64_t(memory)};
}

void ExpectOnlyAWorldStampMoved(const Moved& moved) {
    EXPECT_EQ(moved.copied, 0u);
    EXPECT_EQ(moved.uploaded, sizeof(Ordinal));
    EXPECT_EQ(moved.readBack, 0u);
    EXPECT_EQ(moved.allocated, 0);
}

template <class Test> void ForEachScene(Test test) {
    test::ForEachDevice([&](device::Device& device) {
        ColumnStore store(device);
        Structure structure(store);
        test(device, store, structure);
    });
}

TEST(AttachmentTest, CopiedTensorsAreRefusedAndReadOnlyOnesTakeNoSceneValues) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 2));
        UntypedAttribute radius = View(structure, SchemaId::Sphere, block).Get(kSphereRadius);
        std::vector<float> values = {1, 2};
        int64_t shape = 2;
        DLManagedTensorVersioned tensor{};
        tensor.dl_tensor.data = values.data();
        tensor.dl_tensor.device = {kDLCPU, 0};
        tensor.dl_tensor.ndim = 1;
        tensor.dl_tensor.dtype = {kDLFloat, 32, 1};
        tensor.dl_tensor.shape = &shape;
        tensor.flags = DLPACK_FLAG_BITMASK_IS_COPIED;
        EXPECT_THROW(radius.Attach(&tensor), Error);
        tensor.flags = DLPACK_FLAG_BITMASK_READ_ONLY;
        EXPECT_THROW(radius.Attach(&tensor, Stream::None(), AttachFrom::Scene), Error);
        EXPECT_FALSE(radius.Attached());
        radius.Attach(&tensor).Wait();
        EXPECT_TRUE(radius.Attached());
        radius.Detach().Wait();
    });
}

TEST(AttachmentTest, AHostTensorIsCopiedInWhenMarkChangeded) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 4));
        const View view(structure, SchemaId::Sphere, block);
        Attribute<float> radius = view.Get<float>(kSphereRadius);
        int deletions = 0;
        DLManagedTensor* tensor = HostTensor({1, 2, 3, 4}, deletions);
        radius.Attach(tensor).Wait();
        EXPECT_TRUE(radius.Attached());
        EXPECT_EQ(radius.Get(), (std::vector<float>{1, 2, 3, 4}));

        HostValues(tensor) = {5, 6, 7, 8};
        EXPECT_EQ(radius.Get(), (std::vector<float>{1, 2, 3, 4}));
        store.SetStamp(9);
        radius.MarkChanged().Wait();
        EXPECT_EQ(radius.Get(), (std::vector<float>{5, 6, 7, 8}));
        EXPECT_EQ(GlobalStamp(store), 9u);

        EXPECT_THROW(radius.Set(1.0f), Error);
        EXPECT_THROW(radius.Map(MapAccess::Write), Error);
        EXPECT_THROW(structure.Remove(block.Prim(0, 0)), Error);
        EXPECT_THROW(radius.Attach(HostTensor({0, 0, 0, 0}, deletions)), Error);
        EXPECT_EQ(deletions, 0);

        radius.Detach().Wait();
        EXPECT_EQ(deletions, 1);
        EXPECT_FALSE(radius.Attached());
        EXPECT_EQ(radius.Get(), (std::vector<float>{5, 6, 7, 8}));
        radius.Set(2.0f);
        structure.Remove(block.Prim(0, 0));
    });
}

TEST(AttachmentTest, AttachingFromTheSceneFillsTheTensor) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 3));
        const View view(structure, SchemaId::Sphere, block);
        Attribute<float> radius = view.Get<float>(kSphereRadius);
        radius.Set(std::vector<float>{1, 2, 3});
        int deletions = 0;
        DLManagedTensor* tensor = HostTensor({0, 0, 0}, deletions);
        radius.Attach(tensor, Stream::None(), AttachFrom::Scene).Wait();
        EXPECT_EQ(HostValues(tensor), (std::vector<float>{1, 2, 3}));
        radius.Detach();
        EXPECT_EQ(deletions, 1);
    });
}

TEST(AttachmentTest, ARefusedAttachmentLeavesTheTensorWithTheCaller) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 3));
        const View view(structure, SchemaId::Sphere, block);
        int deletions = 0;
        DLManagedTensor* tensor = HostTensor({1, 2}, deletions);
        EXPECT_THROW(view.Get<float>(kSphereRadius).Attach(tensor), Error);
        EXPECT_EQ(deletions, 0);
        tensor->deleter(tensor);
        EXPECT_THROW(view.Get<float>(kSphereRadius).Detach(), Error);
    });
}

TEST(AttachmentTest, TheStoreFreesTensorsStillAttached) {
    test::ForEachDevice([](device::Device& device) {
        int deletions = 0;
        {
            ColumnStore store(device);
            Structure structure(store);
            const View view(structure, SchemaId::Sphere,
                            structure.Define(SchemaId::Sphere, Paths("/Ball", 2)));
            view.Get<float>(kSphereRadius).Attach(HostTensor({1, 2}, deletions));
        }
        EXPECT_EQ(deletions, 1);
    });
}

TEST(AttachmentTest, DeviceTensorsBecomeTheStorageOfAnAttachableBatch) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        if (!InPlace(device)) {
            return;
        }
        structure.Define(SchemaId::Sphere, Paths("/Before", 3));
        const PrimBlock block = Attachable(structure, "/Ball", 300);
        structure.Define(SchemaId::Sphere, Paths("/After", 3));
        const View view(structure, SchemaId::Sphere, block);
        Attribute<float> radius = view.Get<float>(kSphereRadius);
        std::vector<float> values(300);
        for (uint32_t i = 0; i < values.size(); ++i) {
            values[i] = float(i);
        }
        radius.Set(values);

        int deletions = 0;
        DLManagedTensor* tensor = DeviceTensor(device, 300, deletions);
        const Ref<device::Buffer> user(static_cast<Owned*>(tensor->manager_ctx)->buffer);
        radius.Attach(tensor, Stream::None(), AttachFrom::Scene).Wait();
        std::vector<float> held(300);
        device.GetQueue().Read(*user, held.data());
        EXPECT_EQ(held, values);
        {
            Mapping mapping = radius.Map(MapAccess::Read);
            EXPECT_EQ(mapping.Values().Address(), user->Address());
            mapping.Close();
        }

        values[7] = 70;
        values[299] = 2990;
        device.GetQueue().Write(*user, values.data());
        store.SetStamp(11);
        radius.MarkChanged().Wait();
        EXPECT_EQ(GlobalStamp(store), 11u);
        EXPECT_EQ(radius.Get(), values);
        EXPECT_EQ(radius.Read().ToHost<float>(), values);

        radius.Detach().Wait();
        EXPECT_EQ(deletions, 1);
        device.GetQueue().Write(*user, std::vector<float>(300, -1).data());
        EXPECT_EQ(radius.Get(), values);
        EXPECT_EQ(radius.Read().ToHost<float>(), values);
    });
}

TEST(AttachmentTest, WriteMappingsOfAttachedValuesAliasTheTensorAndStampOnClose) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        if (!InPlace(device)) {
            return;
        }
        const PrimBlock block = Attachable(structure, "/Ball", 64);
        const View view(structure, SchemaId::Sphere, block);
        Attribute<float> radius = view.Get<float>(kSphereRadius);
        int deletions = 0;
        DLManagedTensor* tensor = DeviceTensor(device, 64, deletions);
        const Ref<device::Buffer> user(static_cast<Owned*>(tensor->manager_ctx)->buffer);
        radius.Attach(tensor, Stream::None(), AttachFrom::Scene).Wait();

        const std::vector<float> values(64, 5);
        store.SetStamp(21);
        {
            Mapping mapping = radius.Map(MapAccess::Write);
            EXPECT_TRUE(mapping.Aliased());
            EXPECT_EQ(mapping.Values().Address(), user->Address());
            device.GetQueue().Write(*user, values.data());
            mapping.Close().Wait();
        }
        EXPECT_EQ(GlobalStamp(store), 21u);
        EXPECT_EQ(radius.Get(), values);
        radius.Detach().Wait();
    });
}

TEST(AttachmentTest, AttachmentsByCopyDontMapForWriting) {
    ForEachScene([](device::Device&, ColumnStore&, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 4));
        const View view(structure, SchemaId::Sphere, block);
        Attribute<float> radius = view.Get<float>(kSphereRadius);
        int deletions = 0;
        radius.Attach(HostTensor({1, 2, 3, 4}, deletions)).Wait();
        EXPECT_THROW(radius.Map(MapAccess::Write), Error);
        EXPECT_NO_THROW(radius.Map(MapAccess::Read).Close());
        radius.Detach().Wait();
    });
}

TEST(AttachmentTest, InPlaceAttachmentNeedsAnAttachableBatch) {
    ForEachScene([](device::Device& device, ColumnStore&, Structure& structure) {
        if (!InPlace(device)) {
            return;
        }
        structure.Define(SchemaId::Sphere, Paths("/Before", 3));
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 4));
        const View view(structure, SchemaId::Sphere, block);
        int deletions = 0;
        DLManagedTensor* tensor = DeviceTensor(device, 4, deletions);
        EXPECT_THROW(view.Get<float>(kSphereRadius).Attach(tensor), Error);
        EXPECT_EQ(deletions, 0);
        tensor->deleter(tensor);
    });
}

TEST(ZeroCopyTest, ADenseMappingMovesNoBytesAndAScatteredOneStages) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 1000));
        Attribute<float> dense = View(structure, SchemaId::Sphere, block).Get<float>(kSphereRadius);
        dense.Set(1.0f);
        ExpectOnlyAWorldStampMoved(Measure(store, [&] {
            Mapping mapping = dense.Map(MapAccess::ReadWrite);
            mapping.Close();
        }));

        const PrimBlock other = structure.Define(SchemaId::Sphere, Paths("/Other", 1000));
        std::vector<uint32_t> prims;
        for (uint32_t i = 0; i < 1000; ++i) {
            prims.push_back((i % 2 == 0 ? block : other).Prim(i, 0));
        }
        Attribute<float> scattered =
            View(structure, SchemaId::Sphere, prims).Get<float>(kSphereRadius);
        scattered.Map(MapAccess::ReadWrite).Close();
        Mapping mapping;
        const Moved moved = Measure(store, [&] { mapping = scattered.Map(MapAccess::ReadWrite); });
        EXPECT_GE(moved.allocated, 4000);
        mapping.Close();
    });
}

TEST(ZeroCopyTest, DeviceTensorsAreReadInPlaceOrCopiedOnce) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        if (!device.SharedDevice()) {
            return;
        }
        const PrimBlock block = structure.Define(SchemaId::Sphere, Paths("/Ball", 1000));
        Attribute<float> radius =
            View(structure, SchemaId::Sphere, block).Get<float>(kSphereRadius);
        radius.Read();
        int deletions = 0;
        Owned* owned = new Owned;
        owned->buffer = device.CreateBuffer(
            {.bytes = 4000, .usage = device::BufferUsage::Interop, .label = "user"});
        const device::ExternalAddress address =
            owned->buffer->External(device.SharedDevice()->device_type);
        DLManagedTensor* tensor = Manage(owned, reinterpret_cast<void*>(address.address),
                                         address.device, 1000, deletions);
        const Moved moved = Measure(store, [&] { radius.Write(tensor->dl_tensor); });
        EXPECT_EQ(moved.uploaded, sizeof(Ordinal));
        EXPECT_EQ(moved.readBack, 0u);
        EXPECT_EQ(moved.copied, InPlace(device) ? 0u : 4000u);
        if (InPlace(device)) {
            EXPECT_EQ(moved.allocated, 0);
        }
        tensor->deleter(tensor);
    });
}

TEST(ZeroCopyTest, AnInPlaceAttachmentMovesNoBytesPerFrame) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        if (!InPlace(device)) {
            return;
        }
        const PrimBlock block = Attachable(structure, "/Ball", 1000);
        Attribute<float> radius =
            View(structure, SchemaId::Sphere, block).Get<float>(kSphereRadius);
        int deletions = 0;
        DLManagedTensor* tensor = DeviceTensor(device, 1000, deletions);
        const Ref<device::Buffer> user(static_cast<Owned*>(tensor->manager_ctx)->buffer);
        radius.Attach(tensor).Wait();
        store.Flush();
        for (uint32_t frame = 0; frame < 3; ++frame) {
            const std::vector<float> values(1000, float(frame + 1));
            device.GetQueue().Write(*user, values.data());
            EXPECT_EQ(radius.Read().ToHost<float>(), values);
            ExpectOnlyAWorldStampMoved(Measure(store, [&] {
                radius.MarkChanged();
                store.Flush();
            }));
        }
        radius.Detach();
    });
}

struct OwnedPoints {
    DLManagedTensor managed{};
    std::vector<float> values;
    Ref<device::Buffer> buffer;
    int64_t shape[3] = {0, 0, 3};
    int* deletions = nullptr;
};

DLManagedTensor* Points(OwnedPoints* owned, void* data, DLDevice device, int64_t prims,
                        int64_t count, int& deletions) {
    owned->shape[0] = prims;
    owned->shape[1] = count;
    owned->deletions = &deletions;
    owned->managed.manager_ctx = owned;
    owned->managed.dl_tensor.data = data;
    owned->managed.dl_tensor.device = device;
    owned->managed.dl_tensor.ndim = 3;
    owned->managed.dl_tensor.dtype = {kDLFloat, 32, 1};
    owned->managed.dl_tensor.shape = owned->shape;
    owned->managed.deleter = [](DLManagedTensor* managed) {
        OwnedPoints* self = static_cast<OwnedPoints*>(managed->manager_ctx);
        ++*self->deletions;
        delete self;
    };
    return &owned->managed;
}

DLManagedTensor* HostPoints(uint32_t prims, uint32_t count, float value, int& deletions) {
    OwnedPoints* owned = new OwnedPoints;
    owned->values.assign(size_t(prims) * count * 3, value);
    return Points(owned, owned->values.data(), {kDLCPU, 0}, prims, count, deletions);
}

const std::vector<Float3> kTriangle = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};

TEST(ArrayAttachmentTest, HostPointsAreCopiedInWhenMarkChangededAndClonesKeepTheirOwn) {
    ForEachScene([](device::Device&, ColumnStore& store, Structure& structure) {
        const PrimBlock block = structure.Define(SchemaId::Mesh, Paths("/Mesh", 3));
        const View view(structure, SchemaId::Mesh, block);
        ArrayAttribute<Float3> points = view.GetArray<Float3>(kPointBasedPoints);
        points.Set(0, kTriangle);
        int deletions = 0;
        DLManagedTensor* tensor = HostPoints(3, 4, 1.0f, deletions);
        view.Get(kPointBasedPoints).Attach(tensor).Wait();
        EXPECT_EQ(points.Get(2), std::vector<Float3>(4, Float3{1, 1, 1}));
        EXPECT_THROW(points.Set(0, kTriangle), Error);
        EXPECT_THROW(view.Get(kPointBasedPoints).Map(MapAccess::Write), Error);

        const PrimBlock copy = structure.Clone(block.Prim(1, 0), Paths("/Copy", 1));
        const View copied(structure, SchemaId::Mesh, copy);

        static_cast<OwnedPoints*>(tensor->manager_ctx)->values.assign(36, 5.0f);
        EXPECT_EQ(points.Get(1)[0].x, 1.0f);
        const Ordinal stamp = store.Stamp() + 1;
        store.SetStamp(stamp);
        view.Get(kPointBasedPoints).MarkChanged().Wait();
        EXPECT_EQ(store.ArrayRows(kPointBasedPoints, stamp - 1, stamp).size(), 3u);
        EXPECT_EQ(points.Get(1), std::vector<Float3>(4, Float3{5, 5, 5}));
        EXPECT_EQ(copied.GetArray<Float3>(kPointBasedPoints).Get(0),
                  std::vector<Float3>(4, Float3{1, 1, 1}));

        view.Get(kPointBasedPoints).Detach();
        EXPECT_EQ(deletions, 1);
        EXPECT_EQ(points.Get(1), std::vector<Float3>(4, Float3{5, 5, 5}));
        points.Set(0, kTriangle);
    });
}

TEST(ArrayAttachmentTest, DevicePointsAreTheArraysInPlace) {
    ForEachScene([](device::Device& device, ColumnStore& store, Structure& structure) {
        if (!InPlace(device)) {
            return;
        }
        const PrimBlock block = structure.Define(SchemaId::Mesh, Paths("/Mesh", 2));
        const View view(structure, SchemaId::Mesh, block);
        ArrayAttribute<Float3> points = view.GetArray<Float3>(kPointBasedPoints);
        points.Set(0, kTriangle);
        points.Set(1, kTriangle);
        ArrayHeap& heap = store.Arrays(ColumnType::Float3Array);

        int deletions = 0;
        OwnedPoints* owned = new OwnedPoints;
        owned->buffer = device.CreateBuffer(
            {.bytes = 72, .usage = device::BufferUsage::Interop, .label = "user points"});
        const Ref<device::Buffer> user = owned->buffer;
        const DLDeviceType api =
            device.Adapter().backend == device::Backend::Cuda ? kDLCUDA : kDLROCM;
        const device::ExternalAddress address = user->External(api);
        DLManagedTensor* tensor = Points(owned, reinterpret_cast<void*>(address.address),
                                         address.device, 2, 3, deletions);
        UntypedAttribute attribute = view.Get(kPointBasedPoints);
        attribute.Attach(tensor, Stream::None(), AttachFrom::Scene).Wait();
        std::vector<Float3> held(6);
        device.GetQueue().Read(*user, held.data());
        EXPECT_EQ(held[4], kTriangle[1]);
        {
            Mapping mapping = attribute.Map(MapAccess::Read);
            EXPECT_TRUE(mapping.Aliased());
            EXPECT_EQ(mapping.Values().Address(), user->Address());
            mapping.Close();
        }

        const std::vector<Float3> moved(6, Float3{3, 3, 3});
        device.GetQueue().Write(*user, moved.data());
        EXPECT_EQ(points.Get(1), std::vector<Float3>(3, Float3{3, 3, 3}));
        const Ordinal stamp = store.Stamp() + 1;
        store.SetStamp(stamp);
        ExpectOnlyAWorldStampMoved(Measure(store, [&] {
            attribute.MarkChanged();
            store.Flush();
        }));
        EXPECT_EQ(store.ArrayRows(kPointBasedPoints, stamp - 1, stamp).size(), 2u);

        attribute.Detach().Wait();
        EXPECT_EQ(deletions, 1);
        device.GetQueue().Write(*user, std::vector<Float3>(6).data());
        EXPECT_EQ(points.Get(1), std::vector<Float3>(3, Float3{3, 3, 3}));
        EXPECT_FALSE(heap.Borrowed(store.Array(
            kPointBasedPoints, structure.GetPrims().Row(block.Prim(1, 0), SchemaId::PointBased))));
    });
}

}
}
