// SPDX-License-Identifier: BSD-3-Clause
/// @file hash_index.h
/// @brief An open-addressed index of 32-bit ids whose keys live with their owner.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace ernie {

/// @brief Finds ids by hash, testing candidates with the owner's own equality, so the index holds
/// no keys and a lookup touches one flat array (linear probing, at most half full).
class HashIndex {
  public:
    /// @return The id `equal` accepts among those with `hash`, if any.
    template <typename Equal>
    std::optional<std::uint32_t> Find(std::uint64_t hash, const Equal& equal) const {
        if (slots_.empty()) {
            return std::nullopt;
        }
        const std::uint32_t tag = Tag(hash);
        for (std::size_t i = tag & Mask();; i = (i + 1) & Mask()) {
            const Slot& slot = slots_[i];
            if (slot.id == kEmpty) {
                return std::nullopt;
            }
            if (slot.tag == tag && equal(slot.id)) {
                return slot.id;
            }
        }
    }

    /// @return The id `equal` accepts among those with `hash`, or `id` once added.
    template <typename Equal>
    std::uint32_t Insert(std::uint64_t hash, std::uint32_t id, const Equal& equal) {
        if (2 * (size_ + 1) > slots_.size()) {
            Rehash(std::max<std::size_t>(16, 2 * slots_.size()));
        }
        const std::uint32_t tag = Tag(hash);
        for (std::size_t i = tag & Mask();; i = (i + 1) & Mask()) {
            Slot& slot = slots_[i];
            if (slot.id == kEmpty) {
                slot = {tag, id};
                ++size_;
                return id;
            }
            if (slot.tag == tag && equal(slot.id)) {
                return slot.id;
            }
        }
    }

    /// Makes room for `count` ids without rehashing.
    void Reserve(std::size_t count) {
        std::size_t slots = 16;
        while (slots < 2 * count) {
            slots *= 2;
        }
        if (slots > slots_.size()) {
            Rehash(slots);
        }
    }

    std::size_t Size() const {
        return size_;
    }

  private:
    static constexpr std::uint32_t kEmpty = std::numeric_limits<std::uint32_t>::max();
    struct Slot {
        std::uint32_t tag = 0;
        std::uint32_t id = kEmpty;
    };

    static std::uint32_t Tag(std::uint64_t hash) {
        return static_cast<std::uint32_t>(hash ^ (hash >> 32));
    }
    std::size_t Mask() const {
        return slots_.size() - 1;
    }
    void Rehash(std::size_t slots) {
        std::vector<Slot> old(slots, Slot{});
        old.swap(slots_);
        for (const Slot& slot : old) {
            if (slot.id != kEmpty) {
                std::size_t i = slot.tag & Mask();
                while (slots_[i].id != kEmpty) {
                    i = (i + 1) & Mask();
                }
                slots_[i] = slot;
            }
        }
    }

    std::vector<Slot> slots_;
    std::size_t size_ = 0;
};

}  // namespace ernie
