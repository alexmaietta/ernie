// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "ernie/scene/columns/kernels/authored.h"
#include "ernie/scene/publish/table.h"

namespace ernie {

class SnapshotHeader {
  public:
    virtual ~SnapshotHeader() = default;

    virtual void Describe(const Table& table) = 0;
    virtual std::span<const std::byte> Seal(const AuthoredData& data, uint32_t worldCount) = 0;
};

}
