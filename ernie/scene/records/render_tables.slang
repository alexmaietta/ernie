// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;

#include "ernie/scene/records/render_tables.h"

namespace ernie {

bool PrimAlive(RenderTables* tables, uint prim) {
    return prim < tables->prims.count && (tables->prims.flags[prim] & kPrimAlive) != 0;
}

SchemaId PrimSchema(RenderTables* tables, uint prim) {
    return SchemaId(tables->prims.schema[prim]);
}

uint PrimRow(RenderTables* tables, uint prim, uint slot) {
    return tables->prims.rows[tables->prims.slots[prim] + slot];
}

bool HasApi(RenderTables* tables, uint prim, uint api) {
    const uint64_t apis = tables->prims.apis[prim];
    return ((api < 32 ? uint(apis) >> api : uint(apis >> 32) >> (api - 32)) & 1) != 0;
}

uint ApiRow(RenderTables* tables, uint prim, uint api) {
    const uint64_t apis = tables->prims.apis[prim];
    const uint low = uint(apis);
    const uint below =
        api < 32 ? countbits(low & ((1u << api) - 1))
                 : countbits(low) + countbits(uint(apis >> 32) & ((1u << (api - 32)) - 1));
    const uint typed = kSchemaSlotCounts[tables->prims.schema[prim]];
    return tables->prims.rows[tables->prims.slots[prim] + typed + below];
}

WorldTransform PrimWorld(RenderTables* tables, uint prim) {
    return tables->transforms[PrimRow(tables, prim, kSlotXformable)];
}

bool InCollection(RenderTables* tables, uint row, uint prim) {
    if (row == kNoRow) {
        return true;
    }
    const CollectionRecord collection = tables->collections[row];
    uint at = prim;
    while (at != kNoPrim) {
        if (Lists(collection.excludes, at)) {
            return false;
        }
        if (Lists(collection.includes, at)) {
            return true;
        }
        if (collection.explicitOnly != 0) {
            break;
        }
        at = tables->prims.parent[at];
    }
    return collection.includeRoot != 0;
}

ResolvedImageable PrimImageable(RenderTables* tables, uint prim) {
    return tables->imageables[PrimRow(tables, prim, kSlotImageable)];
}

RenderOrigin TlasOrigin(RenderTables* tables, uint world) {
    return tables->origins[TlasIndex(world, tables->worldCount)];
}

float3 RenderTranslation(RenderTables* tables, WorldTransform w, uint world) {
    return RelativeTranslation(TlasOrigin(tables, world), w);
}

float3 RenderPoint(RenderTables* tables, float3 point, uint world) {
    const RenderOrigin origin = TlasOrigin(tables, world);
    return (point - origin.high.Vector()) - origin.low.Vector();
}

float3 RenderShift(RenderTables* tables, uint from, uint world) {
    const RenderOrigin origin = TlasOrigin(tables, world);
    InstanceLayout* layout = tables->instances;
    if (from >= tables->worldCount || layout->worldOrigins[from] == kNoRow) {
        return -origin.high.Vector() - origin.low.Vector();
    }
    const WorldTransform anchor = tables->transforms[layout->worldOrigins[from]];
    return (anchor.high.Vector() - origin.high.Vector()) +
           (anchor.low.Vector() - origin.low.Vector());
}

}
