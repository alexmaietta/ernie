// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ernie/device/operation.h"
#include "ernie/export.h"
#include "ernie/scene/kernels/ids.h"
#include "ernie/scene/records/semantics.h"
#include "ernie/scene/schema/types.h"

namespace ernie {

struct ComposedWorldDesc {
    std::vector<uint32_t> worlds;
    std::vector<Double3> offsets;
    bool global = true;
    bool stageLights = true;
};

using Ordinal = uint64_t;
using FileBytes = std::shared_ptr<const std::vector<std::byte>>;

struct ERNIE_API CloneDesc {
    bool moveSource = false;
    bool attachable = false;
};

struct ERNIE_API BatchDesc {
    bool attachable = false;
};

struct ERNIE_API PrimRange {
    uint32_t firstPrim = kNoPrim;
    uint32_t count = 0;
    uint32_t copies = 0;
    uint32_t firstWorld = kGlobalWorld;
    uint32_t worldsPerCopy = 0;

    uint32_t Index(uint32_t copy, uint32_t index) const {
        return firstPrim + copy * count + index;
    }

    uint32_t Size() const {
        return count * copies;
    }
};

struct ERNIE_API Box {
    Float3 lower;
    Float3 upper;
};

struct ERNIE_API Snapshot {
    Ordinal ordinal = 0;
    device::Operation ready;
    uint64_t tables = 0;
    std::shared_ptr<const void> pinned;
};

}
