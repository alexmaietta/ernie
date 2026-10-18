// SPDX-License-Identifier: BSD-3-Clause
/// @file pool.h
/// @brief Memory for the frame graph's transient resources: images pooled across frames by
/// description, and buffers as ranges of arenas.
#pragma once

#include "ernie/device/arena.h"
#include "ernie/device/image.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ernie::graph {

/// @brief Lends graphs images and buffer ranges for one execution. An image a graph returns may
/// be lent to a later graph on the same queue, whose barriers order it after the earlier
/// graph's work; images are freed only once the device is done with them.
class TransientPool {
  public:
    /// @param arenaSize Bytes per arena of transient buffer ranges.
    explicit TransientPool(DeviceContext& context,
                           std::uint64_t arenaSize = std::uint64_t{64} << 20);
    ~TransientPool();
    TransientPool(const TransientPool&) = delete;
    TransientPool& operator=(const TransientPool&) = delete;

    struct ImageLease {
        DeviceImage* image = nullptr;
        /// Whether an earlier graph used it.
        bool recycled = false;
    };
    /// @brief Lends an idle image of `description`, or makes one.
    ImageLease AcquireImage(const ImageDescription& description);
    void Release(const DeviceImage& image);

    /// @brief Lends a range of `bytes`, given back once every queue's work so far is done.
    BufferRange AcquireBuffer(std::uint64_t bytes);
    void Release(const BufferRange& range);
    const Arenas& Buffers() const {
        return buffers_;
    }

    /// @brief Frees idle images no graph leased in the last `frames` trims, once the device is
    /// done with them.
    void Trim(std::uint32_t frames = 0);
    /// @return The images held, lent or idle.
    std::size_t Images() const {
        return images_.size();
    }

  private:
    struct Entry {
        std::shared_ptr<DeviceImage> image;
        bool leased = false;
        /// Trims since it was last leased.
        std::uint32_t idle = 0;
    };

    DeviceContext& context_;
    Arenas buffers_;
    std::vector<Entry> images_;
    std::uint64_t made_ = 0;
};

}  // namespace ernie::graph
