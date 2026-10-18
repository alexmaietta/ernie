// SPDX-License-Identifier: BSD-3-Clause
/// @file columns.h
/// @brief Helpers shared by tests of device columns: reading their rows back, and CUDA interop
/// that skips without a CUDA device.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/cuda.h"
#include "ernie/device/staging.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace ernie::test {

/// @return A float column's first `rows` rows, all by default, once `after` is reached.
inline std::vector<float>
ReadFloats(DeviceContext& context, const DeviceColumn& column, const Ticket& after = {},
           std::uint64_t rows = std::numeric_limits<std::uint64_t>::max()) {
    const std::uint64_t bytes = std::min(rows, column.Rows()) * column.RowBytes();
    Readback readback(context, context.Main(), column.Storage(), column.Range().offset, bytes,
                      std::span(&after, after.timeline ? 1 : 0));
    std::vector<float> values(bytes / sizeof(float));
    std::memcpy(values.data(), readback.Bytes().data(), bytes);
    return values;
}

}  // namespace ernie::test

/// Declares the CUDA interop `interop` of `context`, skipping the test without one.
#define ERNIE_CUDA(interop, context)                                                               \
    const std::unique_ptr<::ernie::CudaInterop> interop =                                          \
        ::ernie::CudaInterop::Create(*(context));                                                  \
    if (!interop) {                                                                                \
        GTEST_SKIP() << "No CUDA device pairs with " << (context)->Info().name;                    \
    }
