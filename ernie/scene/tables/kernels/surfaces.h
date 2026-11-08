// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/instances.h"
#include "ernie/scene/records/meshes.h"
#include "ernie/scene/records/prims.h"
#include "ernie/scene/records/transforms.h"
#include "ernie/scene/schema/primvars.h"
#include "ernie/scene/tables/kernels/instance_layout.h"

namespace ernie {

struct SurfaceParams {
    Ptr<AuthoredData> scene;
    Ptr<InstanceLayout> layout;
    Ptr<MeshRecord> meshes;
    Ptr<SurfaceRecord> surfaces;
    Ptr<uint> tlases;
    uint normalsName;
    uint padding;
};

ERNIE_CHECK_SIZE(SurfaceParams, 48);

static const uint kSurfaceGroup = 64;

}
