// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

struct BoundableBox {
    Float3 lower;
    Float3 upper;
};

ERNIE_CHECK_SIZE(BoundableBox, 24);

}
