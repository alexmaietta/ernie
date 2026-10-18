// SPDX-License-Identifier: BSD-3-Clause
/// @file column.h
/// @brief Columns whose source of truth is device memory: per-step data such as poses, written
/// and read in place on the GPU, which the host sees only through an explicit readback.
#pragma once

#include "ernie/device/arena.h"
#include "ernie/device/staging.h"
#include "ernie/util/algorithm.h"
#include "ernie/util/tensor.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief Rows of one element shape in device memory: a range of Ernie's arenas, or memory
/// another API owns and shares in place. Kernels, and through export CUDA, Warp or PyTorch,
/// write rows in place; nothing is mirrored on the host.
class DeviceColumn {
  public:
    /// @param element The shape of one row, such as {4, 3} for a transform; empty for a scalar.
    DeviceColumn(Arenas& arenas, std::string_view name, DType type,
                 std::vector<std::int64_t> element, std::uint64_t rows);
    /// @brief A column over memory it does not own, such as a CUDA tensor imported in place,
    /// from `offset` in `storage`. It cannot grow; it keeps `owner` alive, and both until the
    /// device is done with them.
    DeviceColumn(DeviceContext& context, std::shared_ptr<const Buffer> storage,
                 std::uint64_t offset, std::string_view name, DType type,
                 std::vector<std::int64_t> element, std::uint64_t rows,
                 std::shared_ptr<const void> owner = {});
    ~DeviceColumn();
    DeviceColumn(const DeviceColumn&) = delete;
    DeviceColumn& operator=(const DeviceColumn&) = delete;

    /// @brief Grows or shrinks to `rows`; new rows are undefined. Within the capacity the rows
    /// stay where they are. Past it the capacity at least doubles, so growing a row at a time
    /// costs amortized constant time: the rows are copied to a new range on `queue` after
    /// `waits`, and the old one is freed once the copy is done.
    /// @return The copy's ticket; a default one when nothing was copied.
    /// @throws std::logic_error when a column over another API's memory would grow.
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
    /// @return The rows, then the element's shape: the column as a tensor sees it.
    std::vector<std::int64_t> Shape() const {
        std::vector<std::int64_t> shape{static_cast<std::int64_t>(rows_)};
        shape.insert(shape.end(), element_.begin(), element_.end());
        return shape;
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
    /// @return Whether the memory is another API's, shared in place.
    bool Adopted() const {
        return arenas_ == nullptr;
    }
    /// @return The rows' range: (arena, offset) on every backend; for adopted memory, the offset
    /// in `Storage`.
    const BufferRange& Range() const {
        return range_;
    }
    const Buffer& Storage() const {
        return arenas_ != nullptr ? arenas_->Storage(range_) : *storage_;
    }
    /// @return The first row's device address, as Vulkan shaders read it.
    VkDeviceAddress Address() const {
        return Storage().Address() + range_.offset;
    }

  private:
    BufferRange Allocate(std::uint64_t rows);

    Arenas* arenas_ = nullptr;
    DeviceContext* context_ = nullptr;
    std::shared_ptr<const Buffer> storage_;
    std::shared_ptr<const void> owner_;
    std::string name_;
    DType type_;
    std::vector<std::int64_t> element_;
    std::uint64_t rowBytes_;
    std::uint64_t rows_;
    std::uint64_t capacity_;
    BufferRange range_;
};

/// @brief Stages `rows` of a host mirror into the same rows of `column`. The rows are sorted and
/// made unique in place, and those past the mirror's end dropped; the caller clears them.
template <typename T>
void StageRows(Uploader& uploader, const DeviceColumn& column, const std::vector<T>& host,
               std::vector<std::uint32_t>& rows) {
    SortUnique(rows);
    std::erase_if(rows, [&](std::uint32_t row) { return row >= host.size(); });
    uploader.WriteRows(column.Storage(), column.Range().offset, column.RowBytes(),
                       std::as_bytes(std::span(host)), rows);
}

}  // namespace ernie
