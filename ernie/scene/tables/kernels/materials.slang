// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/materials.h"

using namespace ernie;

[shader("compute")]
[numthreads(kMaterialGroup, 1, 1)]
void WriteMaterials(uniform MaterialParams p) {
    const uint prim = LinearThreadIndex();
    AuthoredData* scene = p.scene;
    if (prim >= scene->primCount || (scene->prims.flags[prim] & kPrimAlive) == 0) {
        return;
    }
    const SchemaId schema = SchemaId(scene->prims.schema[prim]);
    p.temperatures[prim] =
        HasApi(scene, prim, kApiThermalAPI)
            ? scene->schemas.thermalAPI.thermalTemperature[ApiRow(scene, prim, kApiThermalAPI)]
            : 0;
    if (HasApi(scene, prim, kApiCheckerboardAPI)) {
        const uint row = ApiRow(scene, prim, kApiCheckerboardAPI);
        CheckerboardAPIData* c = &scene->schemas.checkerboardAPI;
        CheckerRecord record;
        record.colorA = c.checkerColorA[row];
        record.colorB = c.checkerColorB[row];
        record.enabled = c.checkerboard[row];
        record.size = c.checkerSize[row];
        record.axis = schema == SchemaId::Plane ? scene->schemas.plane.axis[PrimRow(scene, prim, kSlotPlane)] : 2;
        record.padding0 = 0;
        record.padding1 = 0;
        record.padding2 = 0;
        p.checkers[row] = record;
    }
}
