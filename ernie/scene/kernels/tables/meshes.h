// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/schema/types.h"

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

struct TangentRecord {
    ArrayRef tangents;
    uint primvar;
    uint padding0;
    uint padding1;
    uint padding2;
};

ERNIE_CHECK_SIZE(TangentRecord, 32);

}
