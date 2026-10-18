// SPDX-License-Identifier: BSD-3-Clause
/// @file pool.cpp
/// @brief Lending pooled images and arena ranges to graphs.
#include "ernie/graph/pool.h"

#include <algorithm>
#include <string>

namespace ernie::graph {

TransientPool::TransientPool(DeviceContext& context, std::uint64_t arenaSize)
    : context_(context), buffers_(context, MemoryUse::Device, "graph transients", arenaSize) {}

TransientPool::~TransientPool() {
    for (Entry& entry : images_) {
        context_.Retire(std::move(entry.image));
    }
}

TransientPool::ImageLease TransientPool::AcquireImage(const ImageDescription& description) {
    for (Entry& entry : images_) {
        if (!entry.leased && entry.image->Description() == description) {
            entry.leased = true;
            entry.idle = 0;
            return {entry.image.get(), true};
        }
    }
    Entry& entry = images_.emplace_back(
        Entry{std::make_shared<DeviceImage>(context_, description,
                                            "graph transient " + std::to_string(made_++)),
              true, 0});
    return {entry.image.get(), false};
}

void TransientPool::Release(const DeviceImage& image) {
    const auto found = std::ranges::find_if(
        images_, [&](const Entry& entry) { return entry.image.get() == &image; });
    if (found != images_.end()) {
        found->leased = false;
    }
}

BufferRange TransientPool::AcquireBuffer(std::uint64_t bytes) {
    return buffers_.Allocate(bytes);
}

void TransientPool::Release(const BufferRange& range) {
    buffers_.Free(range);
}

void TransientPool::Trim(std::uint32_t frames) {
    std::erase_if(images_, [&](Entry& entry) {
        if (entry.leased || entry.idle++ < frames) {
            return false;
        }
        context_.Retire(std::move(entry.image));
        return true;
    });
}

}  // namespace ernie::graph
