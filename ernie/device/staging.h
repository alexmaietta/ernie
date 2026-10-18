// SPDX-License-Identifier: BSD-3-Clause
/// @file staging.h
/// @brief Moving bytes between the host and buffers: batched uploads, and readbacks that complete
/// with a ticket.
#pragma once

#include "ernie/device/buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ernie {

/// @brief Uploads to buffers on one queue. Writes to a mapped destination land in place; the
/// rest are staged and copied as batched regions when flushed, the staging retired once the
/// copies are done.
class Uploader {
  public:
    /// @param chunk The size staging buffers are allocated in; larger writes get their own.
    Uploader(DeviceContext& context, Queue& queue, std::uint64_t chunk = std::uint64_t{16} << 20);
    ~Uploader();
    Uploader(const Uploader&) = delete;
    Uploader& operator=(const Uploader&) = delete;

    /// @brief Writes bytes into `destination` at `offset`: at once if it is mapped, else at the
    /// next `Flush`.
    /// @throws std::out_of_range if the bytes do not fit.
    void Write(const Buffer& destination, std::uint64_t offset, std::span<const std::byte> bytes);
    template <typename T>
    void Write(const Buffer& destination, std::uint64_t offset, std::span<const T> values) {
        Write(destination, offset, std::as_bytes(values));
    }
    /// @brief Writes some rows of `host`, rows of `rowBytes` laid out as the destination's from
    /// `offset`, a run of consecutive rows at a time.
    /// @param rows The rows to write, ascending and each once.
    void WriteRows(const Buffer& destination, std::uint64_t offset, std::uint64_t rowBytes,
                   std::span<const std::byte> host, std::span<const std::uint32_t> rows);
    /// @return The staged bytes waiting for `Flush`; flush when it grows past a limit.
    std::uint64_t Pending() const {
        return pending_;
    }
    /// @brief Submits the staged copies. Work submitted after them on the same queue, or waiting
    /// on the ticket, sees the bytes.
    /// @return The copies' ticket; a default one when nothing was staged.
    Ticket Flush(std::span<const Ticket> waits = {});

  private:
    struct Copy {
        std::size_t staging;
        VkBuffer destination;
        VkBufferCopy region;
    };

    DeviceContext& context_;
    Queue& queue_;
    std::uint64_t chunk_;
    std::vector<std::shared_ptr<Buffer>> staging_;
    std::uint64_t used_ = 0;
    std::uint64_t pending_ = 0;
    std::vector<Copy> copies_;
};

/// @brief Bytes of a buffer copied to the host after `waits`, readable once its ticket is done.
class Readback {
  public:
    Readback(DeviceContext& context, Queue& queue, const Buffer& source, std::uint64_t offset,
             std::uint64_t size, std::span<const Ticket> waits = {});
    ~Readback();
    Readback(const Readback&) = delete;
    Readback& operator=(const Readback&) = delete;

    Ticket Ready() const {
        return ticket_;
    }
    /// @return The bytes, waiting for the copy if it is not done.
    std::span<const std::byte> Bytes();

  private:
    DeviceContext& context_;
    std::shared_ptr<Buffer> host_;
    Ticket ticket_;
};

}  // namespace ernie
