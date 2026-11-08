// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.render.context;
import ernie.device.kernels.device;
import ernie.io.kernels.encoding;

using namespace ernie;

bool Wants(CapturePixel pixel, RenderVarSource source) {
    return pixel.capture->vars[uint(source)] != 0;
}

void Write<T>(CapturePixel pixel, RenderVarSource source, T value) {
    if (Wants(pixel, source)) {
        Ptr<T>(pixel.capture->vars[uint(source)])[pixel.index] = value;
    }
}

void WriteNormal(CapturePixel pixel, float3 normal) {
    if (Wants(pixel, RenderVarSource::Normal)) {
        Ptr<float> normals = Ptr<float>(pixel.capture->vars[uint(RenderVarSource::Normal)]);
        const uint base = pixel.index * 3;
        normals[base] = normal.x;
        normals[base + 1] = normal.y;
        normals[base + 2] = normal.z;
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

void WriteColor(CapturePixel pixel, float3 radiance, bool hit) {
    Write(pixel, RenderVarSource::HdrColor, float4(radiance, hit ? 1.0f : 0.0f));
    Write(pixel, RenderVarSource::LdrColor, PackSrgb(saturate(radiance), hit));
}

void WriteSegmentation(RenderParams params, CapturePixel pixel, uint prim) {
    const bool wants = Wants(pixel, RenderVarSource::NonStableInstanceSegmentation) ||
                       Wants(pixel, RenderVarSource::SemanticSegmentation) ||
                       Wants(pixel, RenderVarSource::SemanticInstanceSegmentation) ||
                       Wants(pixel, RenderVarSource::StableIdSegmentation);
    if (!wants) {
        return;
    }
    if (prim == kNoPrim) {
        Write(pixel, RenderVarSource::NonStableInstanceSegmentation, 0u);
        Write(pixel, RenderVarSource::SemanticSegmentation, 0u);
        Write(pixel, RenderVarSource::SemanticInstanceSegmentation, uint4(0));
        Write(pixel, RenderVarSource::StableIdSegmentation, uint4(0));
        return;
    }
    const SemanticsData semantics = params.tables->semantics;
    const uint row = pixel.capture->taxonomy * semantics.stride + prim;
    const uint labeled = semantics.semanticPrims[row];
    Write(pixel, RenderVarSource::NonStableInstanceSegmentation,
          labeled == kNoPrim ? 1u : labeled + 2);
    Write(pixel, RenderVarSource::SemanticSegmentation, semantics.semanticIds[row]);
    Write(pixel, RenderVarSource::SemanticInstanceSegmentation,
          labeled == kNoPrim ? uint4(0, 0, 0, 1) : StableWords(semantics.stableIds[labeled]));
    Write(pixel, RenderVarSource::StableIdSegmentation, StableWords(semantics.stableIds[prim]));
}

void WriteInstance(RenderParams params, CapturePixel pixel, uint slot, bool hit) {
    if (!Wants(pixel, RenderVarSource::InstanceId) &&
        !Wants(pixel, RenderVarSource::InstancerStableId)) {
        return;
    }
    SlotHit drawn = Drawn(kNoPrim, kNoPrim, 0, 0);
    if (hit) {
        drawn = params.tables->instances->hits[slot];
    }
    if (drawn.instancer == kNoPrim) {
        Write(pixel, RenderVarSource::InstanceId, int64_t(-1));
        Write(pixel, RenderVarSource::InstancerStableId, uint4(0));
        return;
    }
    SceneData* scene = params.scene;
    int64_t id = 0;
    if (PrimSchema(params.tables, drawn.instancer) == SchemaId::PointInstancer) {
        const ArrayRef ids =
            scene->schemas.pointInstancer.ids[PrimRow(scene, drawn.instancer, kSlotPointInstancer)];
        id = drawn.instance < ids.count ? Ptr<int64_t>(ids.address)[drawn.instance]
                                        : int64_t(drawn.instance);
    }
    Write(pixel, RenderVarSource::InstanceId, id);
    Write(pixel, RenderVarSource::InstancerStableId,
          StableWords(params.tables->semantics.stableIds[drawn.instancer]));
}

void WritePrimId(CapturePixel pixel, uint prim) {
    if (pixel.capture->primIds != 0) {
        Ptr<uint>(pixel.capture->primIds)[pixel.index] = prim;
    }
}
