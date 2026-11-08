// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.texture.kernels.texture;

#include "ernie/scene/kernels/instance_layout.h"
#include "ernie/scene/kernels/lights.h"

using namespace ernie;

static const uint kShapingSteps = 64;
static const uint kIesSteps = 32;

float3 UnitLuminance(float3 rgb) {
    const float luminance = LinearSrgbLuminance(rgb);
    return luminance > 0 ? rgb / luminance : float3(0);
}

float3 CustomRgb(Ptr<Float3> spectra, ArrayRef pairs) {
    if (pairs.count < 2) {
        return float3(0);
    }
    Ptr<Float2> points = Ptr<Float2>(pairs.address);
    float3 xyz = float3(0);
    uint segment = 0;
    for (uint row = 0; row < kSpectralRows; ++row) {
        const float nm = kSpectralFirstNm + row * kSpectralStepNm;
        while (segment + 2 < pairs.count && points[segment + 1].x < nm) {
            ++segment;
        }
        const float2 a = points[segment].Vector();
        const float2 b = points[segment + 1].Vector();
        if (nm < a.x || nm > b.x || b.x <= a.x) {
            continue;
        }
        xyz += lerp(a.y, b.y, (nm - a.x) / (b.x - a.x)) * spectra[row].Vector();
    }
    if (xyz.y <= 0) {
        return float3(0);
    }
    return UnitLuminance(max(XyzToLinearSrgb(xyz / xyz.y) / spectra[kWhiteRow].Vector(), 0));
}

float3 Chromaticity(SceneData* scene, Ptr<Float3> spectra, uint prim, uint row) {
    LightAPIData* light = &scene->schemas.lightAPI;
    const float3 color = light.color[row].Vector();
    const float kelvin = light.colorTemperature[row];
    PhysicalIlluminant illuminant = PhysicalIlluminant::White;
    uint illuminantRow = kNoRow;
    if (HasApi(scene, prim, kApiPhysicalLightIlluminantAPI)) {
        illuminantRow = ApiRow(scene, prim, kApiPhysicalLightIlluminantAPI);
        illuminant = PhysicalIlluminant(
            scene->schemas.physicalLightIlluminantAPI.physicalIlluminant[illuminantRow]);
    }
    switch (illuminant) {
    case PhysicalIlluminant::Blackbody:
        return color * TemperatureRgb(spectra, kelvin, false);
    case PhysicalIlluminant::IlluminantD:
        return color * TemperatureRgb(spectra, kelvin, true);
    case PhysicalIlluminant::Custom:
        return color *
               CustomRgb(spectra,
                         scene->schemas.physicalLightIlluminantAPI.physicalCustomIlluminant[illuminantRow]);
    default:
        if (light.enableColorTemperature[row] != 0) {
            return color * UnitLuminance(TemperatureRgb(spectra, kelvin, false) /
                                         TemperatureRgb(spectra, 6500, false));
        }
        return color;
    }
}

float ShapingLuminance(LightRecord record, float cosine) {
    return LinearSrgbLuminance(ShapingWeight(record, cosine));
}

float IesWeight(LightRecord record, uint64_t handle, float3 emitted) {
    if (handle == 0) {
        return 1;
    }
    const float2 st = IesSt(record, emitted);
    if (st.x < 0) {
        return 0;
    }
    TextureLookup lookup;
    lookup.st = st;
    lookup.lod = -64;
    lookup.filter = TextureFilter::Linear;
    lookup.u = float3(0);
    return max(TextureFromHandle(handle).Sample(lookup).r, 0.0f) * record.iesScale;
}

float EmissionIntegral(LightRecord record, uint64_t handle, bool hemisphere, bool shaped) {
    const bool cone = shaped && (record.flags & kLightShaped) != 0;
    if (!cone && handle == 0) {
        return 1;
    }
    float upper = hemisphere ? 0.5f * kPi : kPi;
    if (cone) {
        upper = min(upper, record.coneAngle);
    }
    const uint arounds = handle != 0 ? kIesSteps : 1;
    float total = 0;
    for (uint step = 0; step < kShapingSteps; ++step) {
        const float polar = (step + 0.5f) / kShapingSteps * upper;
        const float measure = hemisphere ? cos(polar) * sin(polar) : sin(polar);
        const float shaping = cone ? ShapingLuminance(record, cos(polar)) : 1;
        for (uint around = 0; around < arounds; ++around) {
            const float angle = (around + 0.5f) / arounds * 2 * kPi;
            total += shaping * measure *
                     IesWeight(record, handle, EmissionDirection(record, polar, angle));
        }
    }
    return total * (upper / kShapingSteps) * (2 * kPi / arounds) / (hemisphere ? kPi : 4 * kPi);
}

void Profile(inout LightRecord record, uint64_t handle, bool hemisphere, bool normalize) {
    record.iesScale = 1;
    if (handle != 0 && normalize) {
        const float mean = EmissionIntegral(record, handle, hemisphere, false);
        record.iesScale = mean > 0 ? 1 / mean : 1;
    }
}

DomeTextureFormat DomeFormat(SceneData* scene, TextureRecord texture, uint row) {
    DomeLightData* dome = &scene->schemas.domeLight;
    const DomeTextureFormat format = DomeTextureFormat(dome.textureFormat[row]);
    if (format != DomeTextureFormat::Automatic) {
        return format;
    }
    const uint2 size = uint2(texture.width, texture.height);
    if (size.x * 4 == size.y * 3) {
        return DomeTextureFormat::CubeMapVerticalCross;
    }
    return size.x == size.y && size.x > 0 ? DomeTextureFormat::Angular
                                          : DomeTextureFormat::Latlong;
}

[shader("compute")]
[numthreads(kLightGroup, 1, 1)]
void WriteDomeMaps(uniform LightParams p, uint3 id: SV_DispatchThreadID) {
    LightLayout* layout = p.layout;
    if (id.x >= kDomeMapCells || id.y >= layout->domeCount) {
        return;
    }
    SceneData* scene = p.scene;
    const uint domePrim = layout->domePrims[id.y];
    const uint row = PrimRow(scene, domePrim, kSlotDomeLight);
    const TextureRecord texture = p.textures[domePrim];
    const uint binding = texture.textures;
    const uint64_t handle = binding != 0 ? scene->textureBindings[binding] : 0;
    const float2 uv = float2((id.x % kDomeMapWidth + 0.5f) / kDomeMapWidth,
                             (id.x / kDomeMapWidth + 0.5f) / kDomeMapHeight);
    float value = 1;
    if (handle != 0) {
        TextureLookup lookup;
        lookup.st = DomeTextureSt(DomeFormat(scene, texture, row), DomeMapDirection(uv));
        lookup.lod = -0.5f * log2(float(kDomeMapCells));
        lookup.filter = TextureFilter::Linear;
        lookup.u = float3(0);
        value = max(LinearSrgbLuminance(TextureFromHandle(handle).Sample(lookup).rgb), 0.0f);
    }
    Ptr<float> map = layout->maps + id.y * kDomeMapFloats;
    map[id.x] = value * sin(kPi * uv.y);
}

[shader("compute")]
[numthreads(kLightGroup, 1, 1)]
void WriteDomeRows(uniform LightParams p, uint3 id: SV_DispatchThreadID) {
    LightLayout* layout = p.layout;
    if (id.x >= kDomeMapHeight || id.y >= layout->domeCount) {
        return;
    }
    Ptr<float> map = layout->maps + id.y * kDomeMapFloats;
    Ptr<float> cells = map + id.x * kDomeMapWidth;
    float total = 0;
    for (uint i = 0; i < kDomeMapWidth; ++i) {
        total += cells[i];
    }
    float running = 0;
    for (uint i = 0; i < kDomeMapWidth; ++i) {
        running += total > 0 ? cells[i] / total : 1.0f / kDomeMapWidth;
        cells[i] = i + 1 == kDomeMapWidth ? 1.0f : running;
    }
    map[kDomeMapCells + id.x] = total;
}

[shader("compute")]
[numthreads(kLightGroup, 1, 1)]
void WriteDomeMarginals(uniform LightParams p, uint3 id: SV_DispatchThreadID) {
    LightLayout* layout = p.layout;
    if (id.x >= layout->domeCount) {
        return;
    }
    Ptr<float> map = layout->maps + id.x * kDomeMapFloats;
    Ptr<float> rows = map + kDomeMapCells;
    float total = 0;
    for (uint i = 0; i < kDomeMapHeight; ++i) {
        total += rows[i];
    }
    float running = 0;
    for (uint i = 0; i < kDomeMapHeight; ++i) {
        running += total > 0 ? rows[i] / total : 1.0f / kDomeMapHeight;
        rows[i] = i + 1 == kDomeMapHeight ? 1.0f : running;
    }
    map[kDomeMapCells + kDomeMapHeight] = total * kPi / (2.0f * kDomeMapCells);
}

LightRecord DarkLight(uint instancer) {
    LightRecord record;
    record.position = Float3(float3(0));
    record.prim = kNoPrim;
    record.axisX = Float3(float3(1, 0, 0));
    record.type = uint(LightType::Unsupported);
    record.axisY = Float3(float3(0, 1, 0));
    record.flags = 0;
    record.axisZ = Float3(float3(0, 0, 1));
    record.power = 0;
    record.radiance = Float3(float3(0));
    record.radius = 0;
    record.focusTint = Float3(float3(0));
    record.focus = 0;
    record.width = 0;
    record.height = 0;
    record.length = 0;
    record.cosAngle = 1;
    record.diffuse = 0;
    record.specular = 0;
    record.coneAngle = kPi;
    record.coneSoftness = 0;
    record.forwardAxis = uint(ShapingAxis::MinusZ);
    record.spread = 0;
    record.lightLink = kNoRow;
    record.shadowLink = kNoRow;
    record.shadowColor = Float3(float3(0));
    record.shadowDistance = -1;
    record.shadowFalloff = -1;
    record.shadowFalloffGamma = 1;
    record.textures = 0;
    record.textureFormat = 0;
    record.map = Ptr<float>(0);
    record.portals = 0;
    record.portalPrims = 0;
    record.ies = 0;
    record.iesAngleScale = 0;
    record.iesScale = 1;
    record.instancer = instancer;
    record.copyOffset = 0;
    return record;
}

[shader("compute")]
[numthreads(kLightGroup, 1, 1)]
void WriteLights(uniform LightParams p, uint3 id: SV_DispatchThreadID) {
    LightLayout* layout = p.layout;
    const uint tlas = p.tlases[id.y];
    const uint first = layout->firstLights[tlas];
    if (id.x >= layout->firstLights[tlas + 1] - first) {
        return;
    }
    const uint slot = first + id.x;
    const uint item = layout->prims[slot];
    SceneData* scene = p.scene;
    Ptr<Float3> spectra = p.spectra;
    uint prim = item;
    uint instancer = kNoPrim;
    uint copyOffset = 0;
    WorldTransform w;
    if ((item & kInstancerSlot) != 0) {
        const uint pair = item & ~kInstancerSlot;
        InstanceLayout* instances = p.instances;
        const uint block = layout->copies[2 * pair];
        const ResolvedCopy copy =
            ResolveCopy(scene, p.transforms, p.imageables, instances, block,
                        layout->copies[2 * pair + 1], true);
        instancer = instances->blocks[block].instancer;
        if (!copy.valid || !copy.shown) {
            layout->records[slot] = DarkLight(instancer);
            return;
        }
        prim = copy.prim;
        copyOffset = copy.offset;
        w = copy.world;
    } else {
        w = PrimWorld(scene, p.transforms, prim);
    }
    const float3 x = w.x.Vector();
    const float3 y = w.y.Vector();
    const float3 z = w.z.Vector();
    const float3 scale = float3(length(x), length(y), length(z));

    LightRecord record;
    record.position = Float3(RelativeTranslation(p.origins[TlasIndex(tlas, layout->worldCount)], w));
    const uint shownBase = layout->offsetBases[tlas];
    const uint source = scene->prims.world[instancer != kNoPrim ? instancer : prim];
    if (shownBase != kNoRow && source != kGlobalWorld) {
        record.position = Float3(record.position.Vector() + layout->offsets[shownBase + source].Vector());
    }
    record.prim = prim;
    record.axisX = Float3(x / max(scale.x, 1e-30f));
    record.axisY = Float3(y / max(scale.y, 1e-30f));
    record.axisZ = Float3(z / max(scale.z, 1e-30f));
    record.type = uint(LightType::Unsupported);
    record.flags = 0;
    record.radius = 0;
    record.width = 0;
    record.height = 0;
    record.length = 0;
    record.cosAngle = 1;
    record.focus = 0;
    record.focusTint = Float3(float3(0));
    record.coneAngle = kPi;
    record.coneSoftness = 0;
    record.forwardAxis = uint(ShapingAxis::MinusZ);
    record.spread = 0;
    record.lightLink = LinkRow(scene, prim, p.lightLink);
    record.shadowLink = LinkRow(scene, prim, p.shadowLink);
    record.shadowColor = Float3(float3(0));
    record.shadowDistance = -1;
    record.shadowFalloff = -1;
    record.shadowFalloffGamma = 1;
    record.textures = 0;
    record.textureFormat = 0;
    record.map = layout->maps;
    record.portals = 0;
    record.portalPrims = 0;
    record.ies = 0;
    record.iesAngleScale = 0;
    record.iesScale = 1;
    record.instancer = instancer;
    record.copyOffset = copyOffset;

    LightAPIData* light = &scene->schemas.lightAPI;
    const uint row = ApiRow(scene, prim, kApiLightAPI);
    record.diffuse = light.diffuse[row];
    record.specular = light.specular[row];
    if (light.visibleInPrimaryRay[row] != 0) {
        record.flags |= kLightVisible;
    }
    if (light.normalize[row] != 0) {
        record.flags |= kLightNormalize;
    }
    if (HasApi(scene, prim, kApiShadowAPI)) {
        ShadowAPIData* shadow = &scene->schemas.shadowAPI;
        const uint at = ApiRow(scene, prim, kApiShadowAPI);
        if (shadow.shadowEnable[at] == 0) {
            record.flags |= kLightShadowless;
        }
        record.shadowColor = shadow.shadowColor[at];
        record.shadowDistance = shadow.shadowDistance[at];
        record.shadowFalloff = shadow.shadowFalloff[at];
        record.shadowFalloffGamma = shadow.shadowFalloffGamma[at];
    }
    uint64_t ies = 0;
    bool iesNormalize = false;
    if (HasApi(scene, prim, kApiShapingAPI)) {
        ShapingAPIData* shaping = &scene->schemas.shapingAPI;
        const uint at = ApiRow(scene, prim, kApiShapingAPI);
        record.flags |= kLightShaped;
        record.focus = max(shaping.shapingFocus[at], 0.0f);
        record.focusTint = shaping.shapingFocusTint[at];
        record.coneAngle = radians(clamp(shaping.shapingConeAngle[at], 0.0f, 180.0f));
        record.coneSoftness = clamp(shaping.shapingConeSoftness[at], 0.0f, 1.0f);
        const uint binding = p.textures[prim].shaping;
        ies = binding != 0 ? scene->textureBindings[binding] : 0;
        if (ies != 0) {
            record.ies = binding;
            record.iesAngleScale = shaping.shapingIesAngleScale[at];
            iesNormalize = shaping.shapingIesNormalize[at] != 0;
        }
    }
    if (HasApi(scene, prim, kApiOmniShapingAPI)) {
        record.forwardAxis = scene->schemas.omniShapingAPI.rtxShapingForwardAxis[
            ApiRow(scene, prim, kApiOmniShapingAPI)];
    }

    float area = 0;
    float projected = 0;
    float flux = 0;
    bool distant = false;
    float domeMean = -1;
    switch (SchemaId(scene->prims.schema[prim])) {
    case SchemaId::DomeLight: {
        DomeLightData* dome = &scene->schemas.domeLight;
        const uint at = PrimRow(scene, prim, kSlotDomeLight);
        record.type = uint(LightType::Dome);
        if (DomePoleAxis(dome.poleAxis[at]) != DomePoleAxis::Y) {
            record.axisY = Float3(z / max(scale.z, 1e-30f));
            record.axisZ = Float3(-y / max(scale.y, 1e-30f));
        }
        record.textures = p.textures[prim].textures;
        record.textureFormat = uint(DomeFormat(scene, p.textures[prim], at));
        record.map = layout->maps + layout->domeSlots[slot] * kDomeMapFloats;
        record.portals = dome.portals[at].count;
        record.portalPrims = dome.portals[at].address;
        domeMean = record.map[kDomeMapCells + kDomeMapHeight];
        break;
    }
    case SchemaId::PortalLight: {
        const uint at = PrimRow(scene, prim, kSlotPortalLight);
        record.type = uint(LightType::Portal);
        record.width = scene->schemas.portalLight.width[at] * scale.x;
        record.height = scene->schemas.portalLight.height[at] * scale.y;
        break;
    }
    case SchemaId::DistantLight: {
        const float angle = scene->schemas.distantLight.angle[PrimRow(scene, prim, kSlotDistantLight)];
        const float half = sin(radians(angle) * 0.25f);
        record.type = uint(LightType::Distant);
        record.spread = 2 * half * half;
        record.cosAngle = 1 - record.spread;
        area = 4 * kPi * half * half;
        distant = true;
        break;
    }
    case SchemaId::SphereLight: {
        const uint at = PrimRow(scene, prim, kSlotSphereLight);
        record.type = uint(LightType::Sphere);
        record.radius = scene->schemas.sphereLight.radius[at] * max(scale.x, max(scale.y, scale.z));
        if (scene->schemas.sphereLight.treatAsPoint[at] != 0) {
            record.flags |= kLightTreatAsPoint;
        }
        area = 4 * kPi * record.radius * record.radius;
        projected = kPi * record.radius * record.radius;
        Profile(record, ies, false, iesNormalize);
        flux = projected * 4 * kPi * EmissionIntegral(record, ies, false, true);
        break;
    }
    case SchemaId::RectLight: {
        const uint at = PrimRow(scene, prim, kSlotRectLight);
        record.type = uint(LightType::Rect);
        record.width = scene->schemas.rectLight.width[at] * scale.x;
        record.height = scene->schemas.rectLight.height[at] * scale.y;
        area = record.width * record.height;
        projected = area;
        Profile(record, ies, true, iesNormalize);
        flux = area * kPi * EmissionIntegral(record, ies, true, true);
        record.textures = p.textures[prim].textures;
        const uint64_t image = record.textures != 0 ? scene->textureBindings[record.textures] : 0;
        if (image != 0) {
            TextureLookup lookup;
            lookup.st = float2(0.5f);
            lookup.lod = 64;
            lookup.filter = TextureFilter::Linear;
            lookup.u = float3(0);
            const float mean =
                LinearSrgbLuminance(max(TextureFromHandle(image).Sample(lookup).rgb, float3(0)));
            flux *= mean;
            projected *= mean;
        }
        break;
    }
    case SchemaId::DiskLight: {
        const uint at = PrimRow(scene, prim, kSlotDiskLight);
        record.type = uint(LightType::Disk);
        record.radius = scene->schemas.diskLight.radius[at] * max(scale.x, scale.y);
        area = kPi * record.radius * record.radius;
        projected = area;
        Profile(record, ies, true, iesNormalize);
        flux = area * kPi * EmissionIntegral(record, ies, true, true);
        break;
    }
    case SchemaId::CylinderLight: {
        const uint at = PrimRow(scene, prim, kSlotCylinderLight);
        record.type = uint(LightType::Cylinder);
        record.length = scene->schemas.cylinderLight.length[at] * scale.x;
        record.radius = scene->schemas.cylinderLight.radius[at] * max(scale.y, scale.z);
        if (scene->schemas.cylinderLight.treatAsLine[at] != 0) {
            record.flags |= kLightTreatAsLine;
        }
        area = 2 * kPi * record.radius * record.length;
        projected = 2 * record.radius * record.length;
        Profile(record, ies, false, iesNormalize);
        flux = area * kPi * EmissionIntegral(record, ies, false, true);
        break;
    }
    default:
        break;
    }

    const float3 chroma = Chromaticity(scene, spectra, prim, row);
    float3 radiance = chroma * light.intensity[row] * exp2(light.exposure[row]);
    if ((record.flags & kLightNormalize) != 0 && area > 0) {
        radiance /= area;
    }
    if (distant && HasApi(scene, prim, kApiPhotometricDistantLightAPI)) {
        const float illuminance = scene->schemas.photometricDistantLightAPI.photometricIlluminance[
            ApiRow(scene, prim, kApiPhotometricDistantLightAPI)];
        radiance = UnitLuminance(chroma) * illuminance / (area > 0 ? area : 1);
    }
    if (!distant && HasApi(scene, prim, kApiPhotometricAreaLightAPI)) {
        PhotometricAreaLightAPIData* photometric = &scene->schemas.photometricAreaLightAPI;
        const uint at = ApiRow(scene, prim, kApiPhotometricAreaLightAPI);
        const float illuminance = photometric.photometricIlluminance[at];
        const float distance = photometric.photometricIlluminanceDistance[at];
        float luminance = 0;
        if (illuminance > 0 && distance > 0) {
            const float facing = projected * ShapingLuminance(record, 1) *
                                 IesWeight(record, ies, ForwardAxis(record));
            luminance = facing > 0 ? illuminance * distance * distance / facing : 0;
        } else {
            luminance = flux > 0 ? photometric.photometricPower[at] / flux : 0;
        }
        radiance = UnitLuminance(chroma) * luminance;
    }
    if (LightType(record.type) == LightType::Portal) {
        radiance = float3(0);
    }
    record.radiance = Float3(radiance);
    record.power = LinearSrgbLuminance(radiance) * (distant ? area : flux);
    if (domeMean >= 0) {
        record.power = LinearSrgbLuminance(radiance) * domeMean * kPi;
    }
    if (PrimImageable(scene, p.imageables, prim).visibility == uint(Visibility::Invisible)) {
        record.type = uint(LightType::Unsupported);
        record.power = 0;
    }
    layout->records[slot] = record;
}

[shader("compute")]
[numthreads(kLightGroup, 1, 1)]
void WriteLightCdf(uniform LightParams p, uint3 id: SV_DispatchThreadID) {
    if (id.x >= p.tlasCount) {
        return;
    }
    LightLayout* layout = p.layout;
    const uint tlas = p.tlases[id.x];
    const uint first = layout->firstLights[tlas];
    const uint last = layout->firstLights[tlas + 1];
    float total = 0;
    for (uint i = first; i < last; ++i) {
        total += max(layout->records[i].power, 0.0f);
    }
    float running = 0;
    for (uint i = first; i < last; ++i) {
        running += total > 0 ? max(layout->records[i].power, 0.0f) / total : 1.0f / (last - first);
        layout->cdf[i] = i + 1 == last ? 1.0f : running;
    }
}
