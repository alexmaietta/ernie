// SPDX-License-Identifier: BSD-3-Clause
/// @file cuda_test.cpp
/// @brief Tests of CUDA interop: CUDA writes a column in place and Ernie reads it, Ernie writes
/// and CUDA reads, ordered by shared timelines, with no copies; a tensor CUDA allocated itself is
/// copied once on the device, and counted.
#include "ernie/device/cuda.h"
#include "ernie/device/staging.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstring>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

/// Declares the CUDA interop `interop` of `context`, skipping the test without one.
#define ERNIE_CUDA(interop, context)                                                               \
    const std::unique_ptr<CudaInterop> interop = CudaInterop::Create(*(context));                  \
    if (!interop) {                                                                                \
        GTEST_SKIP() << "No CUDA device pairs with " << (context)->Info().name;                    \
    }

std::vector<float> Read(DeviceContext& context, const DeviceColumn& column, const Ticket& after) {
    Readback readback(context, context.Main(), column.Storage(), column.Range().offset,
                      column.Bytes(), std::span(&after, 1));
    std::vector<float> values(column.Bytes() / sizeof(float));
    std::memcpy(values.data(), readback.Bytes().data(), column.Bytes());
    return values;
}

TEST(CudaInterop, CudaWritesAColumnInPlaceThatErnieReads) {
    ERNIE_CONTEXT(context);
    ERNIE_CUDA(cuda, context);
    Arenas arenas(*context, MemoryUse::Shared, "shared", 1 << 16);
    arenas.Allocate(256);
    const DeviceColumn points(arenas, "points", DType::Float32, {3}, 1000);

    const Tensor view = cuda->View(points);
    EXPECT_EQ(view.Location(), (Device{DeviceKind::Cuda, cuda->Ordinal()}));
    EXPECT_EQ(view.Shape(), (std::vector<std::int64_t>{1000, 3}));
    DLManagedTensorVersioned* exported = view.ToDlpack();
    EXPECT_EQ(exported->dl_tensor.device.device_type, kDLCUDA);
    EXPECT_EQ(exported->dl_tensor.device.device_id, cuda->Ordinal());
    exported->deleter(exported);

    const Transfers before = context->Transfers();
    const CudaDriver& driver = cuda->Driver();
    const CudaInterop::Current current(*cuda);
    // CUDA's stream waits for Ernie's earlier work, writes, then signals Ernie's queue.
    cuda->Wait(nullptr, context->Main().Last());
    driver.Check(driver.memsetD32Async(reinterpret_cast<CUdeviceptr>(view.Data()),
                                       std::bit_cast<unsigned int>(1.5f), 3000, nullptr),
                 "Writing in CUDA");
    const Ticket written = cuda->Signal(nullptr);
    EXPECT_EQ(Read(*context, points, written), std::vector<float>(3000, 1.5f));
    EXPECT_EQ(context->Transfers().copied, before.copied) << "written in place";
    context->WaitIdle();
    ExpectClean(*context);
}

TEST(CudaInterop, CudaReadsWhatErnieWroteOnceItsTicketIsReached) {
    ERNIE_CONTEXT(context);
    ERNIE_CUDA(cuda, context);
    Arenas arenas(*context, MemoryUse::Shared, "shared", 1 << 16);
    const DeviceColumn values(arenas, "values", DType::UInt32, {}, 4096);
    const Tensor view = cuda->View(values);
    const Ticket written = context->Main().Submit([&](VkCommandBuffer commands) {
        vkCmdFillBuffer(commands, values.Storage().Handle(), values.Range().offset, values.Bytes(),
                        7);
    });
    cuda->Wait(nullptr, written);
    const CudaDriver& driver = cuda->Driver();
    const CudaInterop::Current current(*cuda);
    driver.Check(driver.streamSynchronize(nullptr), "Synchronizing");
    std::vector<std::uint32_t> seen(4096);
    driver.Check(
        driver.memcpyDtoH(seen.data(), reinterpret_cast<CUdeviceptr>(view.Data()), values.Bytes()),
        "Reading in CUDA");
    EXPECT_EQ(seen, std::vector<std::uint32_t>(4096, 7));
    EXPECT_TRUE(written.Done());
    ExpectClean(*context);
}

TEST(CudaInterop, AssigningItsOwnViewCopiesNothingAndAnyOtherTensorCopiesOnce) {
    ERNIE_CONTEXT(context);
    ERNIE_CUDA(cuda, context);
    Arenas arenas(*context, MemoryUse::Shared, "shared", 1 << 16);
    const DeviceColumn poses(arenas, "poses", DType::Float32, {8}, 100);
    const std::uint64_t bytes = poses.Bytes();
    const CudaDriver& driver = cuda->Driver();
    const CudaInterop::Current current(*cuda);

    const Transfers before = context->Transfers();
    const Ticket inPlace = cuda->Assign(poses, cuda->View(poses), nullptr);
    inPlace.Wait();
    EXPECT_EQ(context->Transfers(), before) << "its own view is assigned in place";

    CUdeviceptr foreign = 0;
    driver.Check(driver.memAlloc(&foreign, bytes), "Allocating in CUDA");
    driver.Check(driver.memsetD32Async(foreign, std::bit_cast<unsigned int>(2.0f), 800, nullptr),
                 "Writing in CUDA");
    const Tensor tensor(reinterpret_cast<void*>(foreign), DType::Float32, {100, 8}, {},
                        Device{DeviceKind::Cuda, cuda->Ordinal()});
    const Ticket copied = cuda->Assign(poses, tensor, nullptr);
    EXPECT_EQ(Read(*context, poses, copied), std::vector<float>(800, 2.0f));
    EXPECT_EQ(context->Transfers().copied - before.copied, bytes) << "one copy, counted";
    EXPECT_EQ(context->Transfers().uploaded, before.uploaded) << "never through the host";
    driver.Check(driver.streamSynchronize(nullptr), "Synchronizing");
    driver.memFree(foreign);

    const Tensor wrongSize(reinterpret_cast<void*>(foreign), DType::Float32, {10}, {},
                           Device{DeviceKind::Cuda, cuda->Ordinal()});
    EXPECT_THROW(cuda->Assign(poses, wrongSize, nullptr), std::invalid_argument);
    context->WaitIdle();
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
