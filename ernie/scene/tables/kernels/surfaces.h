// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/instances.h"
#include "ernie/scene/records/meshes.h"
#include "ernie/scene/tables/kernels/authored.h"
#include "ernie/scene/tables/kernels/instance_layout.h"

namespace ernie {

struct SurfaceParams {
    Ptr<AuthoredData> scene;
    Ptr<InstanceLayout> layout;
    Ptr<MeshRecord> meshes;
    Ptr<SurfaceRecord> surfaces;
    Ptr<uint> tlases;
};

ERNIE_CHECK_SIZE(SurfaceParams, 40);

static const uint kSurfaceGroup = 64;

}
