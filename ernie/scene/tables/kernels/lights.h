// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/lights.h"
#include "ernie/scene/records/prims.h"
#include "ernie/scene/records/transforms.h"
#include "ernie/scene/tables/kernels/instance_layout.h"

namespace ernie {

static const uint kLightGroup = 64;

struct LightParams {
    Ptr<AuthoredData> scene;
    Ptr<LightLayout> layout;
    Ptr<InstanceLayout> instances;
    Ptr<RenderOrigin> origins;
    Ptr<WorldTransform> transforms;
    Ptr<ResolvedImageable> imageables;
    Ptr<TextureRecord> textures;
    Ptr<uint64_t> textureBindings;
    Ptr<uint> tlases;
    Ptr<Float3> spectra;
    Ptr<CollectionRecord> collections;
    Ptr<LightShapeRecord> shapes;
    Ptr<MinimalLightRecord> minimal;
    uint tlasCount;
    uint lightLink;
    uint shadowLink;
    uint collectionRows;
    uint shapeRows;
    uint padding;
};

ERNIE_CHECK_SIZE(LightParams, 128);

#ifdef __SLANG__

bool InCollection(AuthoredData* scene, uint row, uint prim) {
    if (row == kNoRow) {
        return true;
    }
    CollectionAPIData* collection = &scene->schemas.collectionAPI;
    const ArrayRef includes = collection.collectionIncludes[row];
    const ArrayRef excludes = collection.collectionExcludes[row];
    const bool explicitOnly = CollectionExpansionRule(collection.collectionExpansionRule[row]) ==
                              CollectionExpansionRule::ExplicitOnly;
    uint at = prim;
    while (at != kNoPrim) {
        if (Lists(excludes, at)) {
            return false;
        }
        if (Lists(includes, at)) {
            return true;
        }
        if (explicitOnly) {
            break;
        }
        at = scene->prims.parent[at];
    }
    return collection.collectionIncludeRoot[row] != 0;
}

uint LinkRow(AuthoredData* scene, uint prim, uint instance) {
    CollectionAPIData* collection = &scene->schemas.collectionAPI;
    const uint row = ApiInstanceRow(scene, prim, kApiCollectionAPI, instance, collection.instance,
                                    collection.next);
    if (row == kNoRow) {
        return kNoRow;
    }
    const bool everything = collection.collectionIncludeRoot[row] != 0 &&
                            collection.collectionIncludes[row].count == 0 &&
                            collection.collectionExcludes[row].count == 0;
    return everything ? kNoRow : row;
}

#endif

}
