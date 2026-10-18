// SPDX-License-Identifier: BSD-3-Clause
/// @file column.cpp
/// @brief Allocating device columns and growing them by device copies.
#include "ernie/device/column.h"

#include "ernie/util/on_failure.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ernie {
namespace {

std::uint64_t ElementBytes(DType type, const std::vector<std::int64_t>& element) {
    return ByteSize(type) * std::accumulate(element.begin(), element.end(), std::uint64_t{1},
                                            [](std::uint64_t product, std::int64_t extent) {
                                                return product * static_cast<std::uint64_t>(extent);
                                            });
}

}  // namespace

DeviceColumn::DeviceColumn(Arenas& arenas, std::string_view name, DType type,
                           std::vector<std::int64_t> element, std::uint64_t rows)
    : arenas_(&arenas), name_(name), type_(type), element_(std::move(element)),
      rowBytes_(ElementBytes(type, element_)), rows_(rows), capacity_(rows),
      range_(Allocate(rows)) {}

DeviceColumn::DeviceColumn(DeviceContext& context, std::shared_ptr<const Buffer> storage,
                           std::uint64_t offset, std::string_view name, DType type,
                           std::vector<std::int64_t> element, std::uint64_t rows,
                           std::shared_ptr<const void> owner)
    : context_(&context), storage_(std::move(storage)), owner_(std::move(owner)), name_(name),
      type_(type), element_(std::move(element)), rowBytes_(ElementBytes(type, element_)),
      rows_(rows), capacity_(rows), range_{0, offset, rows * rowBytes_} {
    if (offset > storage_->Size() || range_.size > storage_->Size() - offset) {
        throw std::out_of_range(name_ + "'s rows run past the end of their memory");
    }
}

DeviceColumn::~DeviceColumn() {
    if (arenas_ != nullptr) {
        arenas_->Free(range_);
    } else {
        context_->Retire(std::move(storage_));
        context_->Retire(std::move(owner_));
    }
}

BufferRange DeviceColumn::Allocate(std::uint64_t rows) {
    if (rowBytes_ != 0 && rows > std::numeric_limits<std::uint64_t>::max() / rowBytes_) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          std::to_string(rows) + " rows of " + name_ + " overflow");
    }
    // Arenas hand out no empty ranges.
    return arenas_->Allocate(std::max<std::uint64_t>(rowBytes_ * rows, 1));
}

Ticket DeviceColumn::Resize(std::uint64_t rows, Queue& queue, std::span<const Ticket> waits) {
    if (rows <= capacity_) {
        rows_ = rows;
        return {};
    }
    if (arenas_ == nullptr) {
        throw std::logic_error(name_ + " is another API's memory, which Ernie cannot grow");
    }
    // Nothing changes until the new range holds the rows, so a failure leaves the column whole.
    const std::uint64_t capacity = std::max(rows, 2 * capacity_);
    const BufferRange grown = Allocate(capacity);
    const OnFailure freeGrown([&] { arenas_->Free(grown); });
    const std::uint64_t kept = rowBytes_ * rows_;
    Ticket ticket;
    if (kept > 0) {
        ticket = queue.Submit(
            [&](VkCommandBuffer commands) {
                const VkBufferCopy region{range_.offset, grown.offset, kept};
                CopyBuffer(commands, arenas_->Storage(range_).Handle(),
                           arenas_->Storage(grown).Handle(), std::span(&region, 1));
            },
            waits);
    }
    arenas_->Free(std::exchange(range_, grown));
    capacity_ = capacity;
    rows_ = rows;
    return ticket;
}

}  // namespace ernie
