// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "ernie/bvh/kernels/bvh.h"

namespace ernie::bvh {

struct Inspection {
    std::vector<std::string> errors;
    uint32_t nodes = 0;
    uint32_t leaves = 0;
    uint32_t records = 0;
    uint32_t depth = 0;
    double sahCost = 0;
    double childrenPerNode = 0;
    double recordsPerLeaf = 0;
};

Inspection Inspect(std::span<const std::byte> bytes, std::span<const BvhBox> boxes = {});

}
