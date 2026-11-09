// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/tensor.h"

#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"
#include "support/devices.h"

namespace ernie::device {
namespace {

using test::ForEachDevice;

static_assert(ElementBytes(DType::UInt8) == 1);
static_assert(ElementBytes(DType::Float32) == 4);
static_assert(ElementBytes(DType::Int64) == 8);
static_assert(ElementType<double>() == DType::Float64);

Tensor Filled(Device& device, std::vector<float> values, std::vector<int64_t> shape) {
    Tensor tensor = device.CreateTensor(DType::Float32, std::move(shape), "filled");
    device.GetQueue().Write(tensor.GetBuffer().Slice(tensor.Offset(), values.size() * 4),
                            values.data());
    return tensor;
}

TEST(TensorTest, ShapesSelectsAndReadsBack) {
    ForEachDevice([](Device& device) {
        std::vector<float> values(6);
        std::iota(values.begin(), values.end(), 1.0f);
        const Tensor tensor = Filled(device, values, {2, 3});
        EXPECT_FALSE(tensor.Empty());
        EXPECT_EQ(tensor.Elements(), 6);
        EXPECT_EQ(tensor.Strides()[0], 3);
        EXPECT_EQ(tensor.Strides()[1], 1);
        EXPECT_EQ(tensor.ToHost<float>(), values);

        const Tensor row = tensor.Select(1);
        ASSERT_EQ(row.Shape().size(), 1u);
        EXPECT_EQ(row.Shape()[0], 3);
        EXPECT_EQ(row.Address(), tensor.Address() + 12);
        EXPECT_EQ(row.ToHost<float>(), (std::vector<float>{4, 5, 6}));
        const Download download = row.ToHostAsync();
        download.Wait();
        EXPECT_EQ(download.Bytes().size(), 12u);

        EXPECT_THROW(tensor.Select(2), Error);
        EXPECT_THROW(tensor.ToHost<double>(), std::invalid_argument);
        EXPECT_THROW(Tensor().ToHostAsync(), Error);
        EXPECT_THROW(device.CreateTensor(DType::Float32, {2, -1}), Error);
    });
}

TEST(TensorTest, ReadsBackStridedViews) {
    ForEachDevice([](Device& device) {
        std::vector<float> values(24);
        std::iota(values.begin(), values.end(), 0.0f);
        const Tensor tensor = Filled(device, values, {2, 3, 4});
        const Ref<Buffer> buffer(&tensor.GetBuffer());

        const Tensor rows(buffer, tensor.Offset() + 4 * 4, DType::Float32, {2, 2, 4}, {12, 4, 1});
        EXPECT_EQ(rows.ToHost<float>(),
                  (std::vector<float>{4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23}));
        EXPECT_THROW(rows.ToHostAsync(), UnsupportedError);

        const Tensor columns(buffer, tensor.Offset() + 4, DType::Float32, {2, 3, 2}, {12, 4, 2});
        EXPECT_EQ(columns.ToHost<float>(),
                  (std::vector<float>{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}));

        const Tensor transposed(buffer, tensor.Offset(), DType::Float32, {4, 3}, {1, 4});
        EXPECT_EQ(transposed.ToHost<float>(),
                  (std::vector<float>{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}));

        const Tensor single(buffer, tensor.Offset() + 5 * 4, DType::Float32, {1, 3, 1}, {24, 4, 1});
        EXPECT_EQ(single.ToHost<float>(), (std::vector<float>{5, 9, 13}));
    });
}

TEST(TensorTest, CopiesBetweenTensorsOfOneSize) {
    ForEachDevice([](Device& device) {
        const Tensor from = Filled(device, {1, 2, 3, 4}, {4});
        const Tensor to = device.CreateTensor(DType::Float32, {2, 2});
        device.Copy(from, to).Wait();
        EXPECT_EQ(to.ToHost<float>(), (std::vector<float>{1, 2, 3, 4}));
        EXPECT_THROW(device.Copy(from, device.CreateTensor(DType::Float32, {3})), Error);
    });
}

TEST(TensorTest, CopiesBetweenDevicesThroughTheHost) {
    std::optional<std::string> cpu;
    for (const AdapterInfo& adapter : EnumerateAdapters()) {
        if (adapter.type == AdapterType::Cpu && adapter.unsupported.empty()) {
            cpu = adapter.alias;
        }
    }
    if (!cpu) {
        GTEST_SKIP() << "no CPU adapter";
    }
    const Ref<Device> first = Device::Create();
    if (first->Adapter().alias == *cpu) {
        GTEST_SKIP() << "the default adapter is the CPU one";
    }
    const Ref<Device> second = Device::Create({.adapter = *cpu});
    std::vector<float> values(4096);
    std::iota(values.begin(), values.end(), 0.5f);
    const Tensor from = Filled(*first, values, {64, 64});
    const Tensor middle = second->CreateTensor(DType::Float32, {64, 64});
    second->Copy(from, middle).Wait();
    EXPECT_EQ(middle.ToHost<float>(), values);
    const Tensor back = first->CreateTensor(DType::Float32, {4096});
    first->Copy(middle, back).Wait();
    EXPECT_EQ(back.ToHost<float>(), values);
    second->GetQueue().WaitIdle();
    first->GetQueue().WaitIdle();
}

TEST(TensorTest, ImportsHostMemory) {
    ForEachDevice([](Device& device) {
        std::vector<int32_t> host = {9, 8, 7};
        int64_t shape[] = {3};
        DLTensor fromHost{.data = host.data(),
                          .device = {kDLCPU, 0},
                          .ndim = 1,
                          .dtype = {kDLInt, 32, 1},
                          .shape = shape};
        EXPECT_EQ(device.Import(fromHost).ToHost<int32_t>(), host);
        fromHost.dtype = {kDLFloat, 16, 1};
        EXPECT_THROW(device.Import(fromHost), UnsupportedError);
    });
}

TEST(TensorTest, TravelsThroughDlpackWhenMemoryIsShared) {
    ForEachDevice([](Device& device) {
        const Tensor tensor = Filled(device, {1, 2, 3, 4, 5, 6}, {3, 2});
        if (!device.SharedDevice()) {
            EXPECT_THROW(tensor.DLPack(), UnsupportedError);
            return;
        }
        device.GetQueue().WaitIdle();
        const DLTensor view = tensor.DLPack();
        EXPECT_EQ(view.ndim, 2);
        EXPECT_EQ(view.shape[0], 3);
        EXPECT_EQ(view.dtype.code, kDLFloat);
        EXPECT_EQ(view.dtype.bits, 32);
        EXPECT_EQ(device.Import(view).ToHost<float>(), tensor.ToHost<float>());

        DLManagedTensorVersioned* exported = tensor.Export();
        ASSERT_NE(exported, nullptr);
        EXPECT_EQ(exported->dl_tensor.ndim, 2);
        exported->deleter(exported);
    });
}

}
}
