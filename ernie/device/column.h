// SPDX-License-Identifier: BSD-3-Clause
/// @file column.h
/// @brief Columns whose source of truth is device memory: per-step data such as poses, written
/// and read in place on the GPU, which the host sees only through an explicit readback.
#pragma once

#include "ernie/device/arena.h"
#include "ernie/util/tensor.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief Rows of one element shape in one range of device arenas. Kernels, and through export
/// CUDA, Warp or PyTorch, write rows in place; nothing is mirrored on the host.
class DeviceColumn {
  public:
    /// @param element The shape of one row, such as {4, 3} for a transform; empty for a scalar.
    DeviceColumn(Arenas& arenas, std::string_view name, DType type,
                 std::vector<std::int64_t> element, std::uint64_t rows);
    ~DeviceColumn();
    DeviceColumn(const DeviceColumn&) = delete;
    DeviceColumn& operator=(const DeviceColumn&) = delete;

    /// @brief Grows or shrinks to `rows`; new rows are undefined. Within the capacity the rows
    /// stay where they are. Past it the capacity at least doubles, so growing a row at a time
    /// costs amortized constant time: the rows are copied to a new range on `queue` after
    /// `waits`, and the old one is freed once the copy is done.
    /// @return The copy's ticket; a default one when nothing was copied.
    Ticket Resize(std::uint64_t rows, Queue& queue, std::span<const Ticket> waits = {});

    const std::string& Name() const {
        return name_;
    }
    DType Type() const {
        return type_;
    }
    const std::vector<std::int64_t>& Element() const {
        return element_;
    }
    std::uint64_t Rows() const {
        return rows_;
    }
    /// @return The rows' bytes; the range may hold more, for growth.
    std::uint64_t Bytes() const {
        return rows_ * rowBytes_;
    }
    std::uint64_t Capacity() const {
        return capacity_;
    }
    std::uint64_t RowBytes() const {
        return rowBytes_;
    }
    /// @return The rows' range: (arena, offset) on every backend.
    const BufferRange& Range() const {
        return range_;
    }
    const Buffer& Storage() const {
        return arenas_.Storage(range_);
    }
    /// @return The first row's device address, as Vulkan shaders read it.
    VkDeviceAddress Address() const {
        return arenas_.Address(range_);
    }

  private:
    BufferRange Allocate(std::uint64_t rows);

    Arenas& arenas_;
    std::string name_;
    DType type_;
    std::vector<std::int64_t> element_;
    std::uint64_t rowBytes_;
    std::uint64_t rows_;
    std::uint64_t capacity_;
    BufferRange range_;
};

}  // namespace ernie
