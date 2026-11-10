// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/records/bounds.h"
#include "ernie/scene/records/instances.h"
#include "ernie/scene/records/lights.h"
#include "ernie/scene/records/materials.h"
#include "ernie/scene/records/meshes.h"
#include "ernie/scene/records/points.h"
#include "ernie/scene/records/prims.h"
#include "ernie/scene/records/products.h"
#include "ernie/scene/records/semantics.h"
#include "ernie/scene/records/sensors.h"
#include "ernie/scene/records/spectra.h"
#include "ernie/scene/records/tangents.h"
#include "ernie/scene/records/textures.h"
#include "ernie/scene/records/transforms.h"
#include "ernie/stage/schema/ids.h"
#include "ernie/stage/schema/schema.h"

namespace ernie {

struct RenderTables {
    PrimTable prims;
    Ptr<WorldTransform> transforms;
    Ptr<ResolvedImageable> imageables;
    Ptr<MotionRecord> motions;
    Ptr<MeshRecord> meshes;
    Ptr<SubsetRecord> subsets;
    Ptr<PointsRecord> points;
    Ptr<TangentRecord> tangents;
    Ptr<SurfaceRecord> surfaces;
    Ptr<BoundableBox> primBounds;
    Ptr<TextureRecord> textures;
    Ptr<uint64_t> textureBindings;
    Ptr<InstanceLayout> instances;
    Ptr<LightLayout> lights;
    Ptr<MinimalLightRecord> minimalLights;
    Ptr<CollectionRecord> collections;
    Ptr<LightShapeRecord> lightShapes;
    Ptr<RenderOrigin> origins;
    Ptr<WorldTransform> slotWorlds;
    Ptr<CameraRecord> cameras;
    Ptr<DistortionRecord> distortions;
    Ptr<DetectorRecord> detectors;
    Ptr<LidarRecord> lidars;
    Ptr<RadarRecord> radars;
    Ptr<ProductRecord> products;
    Ptr<MaterialRecord> materials;
    Ptr<uint> materialWords;
    Ptr<CheckerRecord> checkers;
    SpectraRecord spectra;
    Ptr<float> temperatures;
    Ptr<RadioRecord> radios;
    uint temperaturePrimvar;
    uint padding0;
    SemanticsData semantics;
    uint worldCount;
    uint padding;
};

ERNIE_CHECK_SIZE(RenderTables, 408);

}
