// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/schema/types.h"

namespace ernie {

struct MeshRecord {
    uint64_t blas;
    ArrayRef corners;
    ArrayRef faces;
    ArrayRef starts;
    ArrayRef adjacency;
    ArrayRef normals;
    ArrayRef bounds;
};

ERNIE_CHECK_SIZE(MeshRecord, 104);

}
