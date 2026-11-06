// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.icamera;
__exported import ernie.render.kernels.render.irender;
import ernie.render.kernels.sensors.camera.camera;
import ernie.device.kernels.device;
import ernie.io.kernels.encoding;

using namespace ernie;

bool Wants(CapturePixel target, RenderVarSource source) {
    return target.capture->vars[uint(source)] != 0;
}

void Write<T>(CapturePixel target, RenderVarSource source, T value) {
    if (Wants(target, source)) {
        Ptr<T>(target.capture->vars[uint(source)])[target.index] = value;
    }
}

uint PackSrgb(float3 color, bool hit) {
    const uint r = EncodeUnorm8(EncodeSrgb(color.x));
    const uint g = EncodeUnorm8(EncodeSrgb(color.y));
    const uint b = EncodeUnorm8(EncodeSrgb(color.z));
    return r | g << 8 | b << 16 | (hit ? 255u : 0u) << 24;
}

uint4 StableWords(StableId id) {
    return uint4(id.words[0], id.words[1], id.words[2], id.words[3]);
}

float3 DisplayRadiance(float3 display) {
    return saturate(display);
}

void WriteColor(CapturePixel target, float3 radiance, bool hit) {
    Write(target, RenderVarSource::HdrColor, float4(radiance, hit ? 1.0f : 0.0f));
    Write(target, RenderVarSource::LdrColor, PackSrgb(saturate(radiance), hit));
}

void WriteSegmentation(RenderParams params, CapturePixel target, uint prim) {
    const bool wants = Wants(target, RenderVarSource::NonStableInstanceSegmentation) ||
                       Wants(target, RenderVarSource::SemanticSegmentation) ||
                       Wants(target, RenderVarSource::SemanticInstanceSegmentation) ||
                       Wants(target, RenderVarSource::StableIdSegmentation);
    if (!wants) {
        return;
    }
    if (prim == kNoPrim) {
        Write(target, RenderVarSource::NonStableInstanceSegmentation, 0u);
        Write(target, RenderVarSource::SemanticSegmentation, 0u);
        Write(target, RenderVarSource::SemanticInstanceSegmentation, uint4(0));
        Write(target, RenderVarSource::StableIdSegmentation, uint4(0));
        return;
    }
    const SemanticsData semantics = params.semantics;
    const uint row = target.capture->taxonomy * semantics.stride + prim;
    const uint labeled = semantics.semanticPrims[row];
    Write(target, RenderVarSource::NonStableInstanceSegmentation,
          labeled == kNoPrim ? 1u : labeled + 2);
    Write(target, RenderVarSource::SemanticSegmentation, semantics.semanticIds[row]);
    Write(target, RenderVarSource::SemanticInstanceSegmentation,
          labeled == kNoPrim ? uint4(0, 0, 0, 1) : StableWords(semantics.stableIds[labeled]));
    Write(target, RenderVarSource::StableIdSegmentation, StableWords(semantics.stableIds[prim]));
}

void WriteInstance(RenderParams params, CapturePixel target, SurfacePoint point, bool hit) {
    if (!Wants(target, RenderVarSource::InstanceId) &&
        !Wants(target, RenderVarSource::InstancerStableId)) {
        return;
    }
    SlotHit drawn = Drawn(kNoPrim, kNoPrim, 0, 0);
    if (hit) {
        drawn = params.layout->hits[point.slot];
    }
    if (drawn.instancer == kNoPrim) {
        Write(target, RenderVarSource::InstanceId, int64_t(-1));
        Write(target, RenderVarSource::InstancerStableId, uint4(0));
        return;
    }
    SceneData* scene = params.scene;
    int64_t id = 0;
    if (SchemaId(scene->prims.schema[drawn.instancer]) == SchemaId::PointInstancer) {
        const ArrayRef ids =
            scene->schemas.pointInstancer.ids[PrimRow(scene, drawn.instancer, kSlotPointInstancer)];
        id = drawn.instance < ids.count ? Ptr<int64_t>(ids.address)[drawn.instance]
                                        : int64_t(drawn.instance);
    }
    Write(target, RenderVarSource::InstanceId, id);
    Write(target, RenderVarSource::InstancerStableId,
          StableWords(params.semantics.stableIds[drawn.instancer]));
}

void WritePrimId(CapturePixel target, uint prim) {
    if (target.capture->primIds != 0) {
        Ptr<uint>(target.capture->primIds)[target.index] = prim;
    }
}

void WriteSurface(RenderParams params, CapturePixel target, PathSample sample) {
    WritePrimId(target, sample.hit ? sample.point.prim : kNoPrim);
    if (!sample.hit) {
        Write(target, RenderVarSource::DistanceToCamera, target.capture->missDistance);
        Write(target, RenderVarSource::DistanceToImagePlane, target.capture->missDistance);
        Write(target, RenderVarSource::Normal, float4(0));
        Write(target, RenderVarSource::DiffuseAlbedo, PackSrgb(float3(0), false));
        WriteSegmentation(params, target, kNoPrim);
        WriteInstance(params, target, sample.point, false);
        return;
    }
    const SurfacePoint point = sample.point;
    Write(target, RenderVarSource::DistanceToCamera, point.distance);
    if (Wants(target, RenderVarSource::DistanceToImagePlane)) {
        const ICamera camera = LoadCamera(params.scene, params.layout, target.capture->sensor);
        Write(target, RenderVarSource::DistanceToImagePlane,
              camera.ImagePlaneDistance(point.position));
    }
    Write(target, RenderVarSource::Normal, float4(point.shadingNormal, 1));
    Write(target, RenderVarSource::DiffuseAlbedo, PackSrgb(sample.albedo, true));
    WriteSegmentation(params, target, point.prim);
    WriteInstance(params, target, point, true);
}
