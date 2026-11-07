// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

namespace ernie {

static const uint kNoPrim = 0xFFFFFFFF;
static const uint kNoRow = 0xFFFFFFFF;
static const uint kGlobalWorld = 0xFFFFFFFF;

ERNIE_SHARED uint WorldSlot(uint world) {
    return world == kGlobalWorld ? 0 : world + 1;
}

ERNIE_SHARED uint SlotWorld(uint slot) {
    return slot == 0 ? kGlobalWorld : slot - 1;
}

ERNIE_SHARED uint TlasIndex(uint world, uint worldCount) {
    return world < worldCount ? world : worldCount;
}

}
