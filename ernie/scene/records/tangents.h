// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/stage/schema/types.h"

namespace ernie {

struct TangentRecord {
    ArrayRef tangents;
    uint primvar;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(TangentRecord, 32);

}
