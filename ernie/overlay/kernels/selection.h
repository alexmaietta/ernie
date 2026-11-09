// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"

#ifndef __SLANG__
#include "ernie/scene/records/render_tables.h"
#endif

namespace ernie {

struct OverlayUniteParams {
    Ptr<RenderTables> tables;
    Ptr<uint> prims;
    Ptr<uint> groups;
    Ptr<uint> boxes;
    uint primCount;
    uint primStride;
    uint groupStride;
    uint boxCount;
};

ERNIE_CHECK_SIZE(OverlayUniteParams, 48);

}
