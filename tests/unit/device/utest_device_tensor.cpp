// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/tensor.h"

#include <cstdint>
#include <numeric>
#include <stdexcept>
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

TEST(TensorTest, CopiesBetweenTensorsOfOneSize) {
    ForEachDevice([](Device& device) {
        const Tensor from = Filled(device, {1, 2, 3, 4}, {4});
        const Tensor to = device.CreateTensor(DType::Float32, {2, 2});
        device.Copy(from, to).Wait();
        EXPECT_EQ(to.ToHost<float>(), (std::vector<float>{1, 2, 3, 4}));
        EXPECT_THROW(device.Copy(from, device.CreateTensor(DType::Float32, {3})), Error);
    });
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
