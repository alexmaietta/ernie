// SPDX-License-Identifier: BSD-3-Clause
/// @file device.h
/// @brief Binding the device to Python: columns any DLPack library shares in place.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/cuda.h"
#include "ernie/device/queue.h"
#include "ernie/util/tensor.h"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ernie::python {

/// @return A DLPack stream argument as a CUDA stream: None and 1 are the legacy default stream, 2
/// the per-thread one, anything else a `cudaStream_t`.
CUstream ToStream(const nanobind::object& stream);

/// @brief A device and the CUDA device paired with it, if any.
struct Interop {
    DeviceContext* context = nullptr;
    CudaInterop* cuda = nullptr;

    /// @throws nanobind::buffer_error without a paired CUDA device.
    CudaInterop& Cuda() const;
};

/// @brief A column Python shares through DLPack, and the ticket of the writes, CUDA's or a
/// render's, Ernie or a reader has yet to wait for.
struct Column {
    Interop owner;
    /// The column, owned by this or by a renderer.
    std::unique_ptr<DeviceColumn> owned;
    const DeviceColumn* column = nullptr;
    /// The tensor a wrapped column shares, which `__dlpack__` hands back as it is.
    std::optional<Tensor> wrapped;
    /// The shape it shows; empty for the column's own.
    std::vector<std::int64_t> shape;
    Ticket written;
    /// The step that rendered a renderer's output: what readers wait for in place of the queues'
    /// last work, so a later step rendering into another output set need not finish first.
    std::optional<Ticket> rendered;
    /// What keeps a column another object owns alive.
    nanobind::object keeper;

    Column(Interop interop, std::unique_ptr<DeviceColumn> device,
           std::optional<Tensor> tensor = std::nullopt);
    /// @brief A view of a column another object owns, shown as `view`.
    Column(Interop interop, const DeviceColumn& device, std::vector<std::int64_t> view,
           Ticket ready);

    std::vector<std::int64_t> Shape() const;
    nanobind::object Dlpack(const nanobind::object& stream, const nanobind::object& maxVersion);
    void Assign(const nanobind::object& source, const nanobind::object& stream);
    /// @return The rows copied to the host after the pending writes.
    Tensor Read() const;
};

/// @brief Adds `DeviceContext` and `DeviceColumn` to the module.
void BindDevice(nanobind::module_& module);

}  // namespace ernie::python
