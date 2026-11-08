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

[shader("compute")]
[numthreads(kMaterialGroup, 1, 1)]
void PackMaterials(uniform PackParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.count) {
        return;
    }
    const PackEntry entry = p.entries[index];
    if (entry.pages == 0) {
        p.words[entry.destination] = entry.value;
        return;
    }
    const Ptr<uint64_t> pages = Ptr<uint64_t>(entry.pages);
    const Ptr<uint> source = Ptr<uint>(pages[entry.row >> kPageRowsLog2]) +
                             (entry.row & (kPageRows - 1)) * entry.words;
    for (uint word = 0; word < entry.words; ++word) {
        p.words[entry.destination + word] = source[word];
    }
}
