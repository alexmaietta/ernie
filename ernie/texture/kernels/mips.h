// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/kernels/shared.h"
#include "ernie/device/kernels/texture.h"

namespace ernie {

static const uint kMipGroup = 64;

struct MipJob {
    Ptr<uint> texels;
    uint source;
    uint target;
    uint sourceWidth;
    uint sourceHeight;
    uint width;
    uint height;
    uint first;
    TexelFormat format;
    TexelColorSpace colorSpace;
    uint padding;
};

ERNIE_CHECK_SIZE(MipJob, 48);

struct MipParams {
    Ptr<MipJob> jobs;
    uint jobCount;
    uint total;
};

ERNIE_CHECK_SIZE(MipParams, 16);

}
