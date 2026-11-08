// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.light.ilight;
import ernie.render.kernels.light.area;
import ernie.texture.kernels.texture;

using namespace ernie;

float3 DomeNative(LightRecord light, float3 direction) {
    return float3(dot(direction, light.axisX.Vector()), dot(direction, light.axisY.Vector()),
                  dot(direction, light.axisZ.Vector()));
}

float3 DomeWorld(LightRecord light, float3 native) {
    return native.x * light.axisX.Vector() + native.y * light.axisY.Vector() +
           native.z * light.axisZ.Vector();
}

float2 DomeMapUv(float3 n) {
    const float u = atan2(n.z, n.x) / (2 * kPi);
    return float2(u - floor(u), acos(clamp(n.y, -1.0f, 1.0f)) / kPi);
}

uint SearchCdf(Ptr<float> cdf, uint count, float u) {
    uint low = 0;
    uint high = count - 1;
    while (low < high) {
        const uint middle = (low + high) / 2;
        if (cdf[middle] < u) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

struct DomeMapSample {
    float3 direction;
    float pdf;

}

float DomeMapDensity(float rowProbability, float columnProbability, float v) {

    const float sine = sin(kPi * v);
    if (sine <= 0) {
        return 0;
    }
    return rowProbability * columnProbability * kDomeMapCells / (2 * kPi * kPi * sine);
}

DomeMapSample SampleDomeMap(Ptr<float> map, float2 u) {
    Ptr<float> marginal = map + kDomeMapCells;
    const uint row = SearchCdf(marginal, kDomeMapHeight, u.x);
    const float rowLow = row > 0 ? marginal[row - 1] : 0.0f;
    const float rowProbability = marginal[row] - rowLow;
    Ptr<float> conditional = map + row * kDomeMapWidth;
    const uint column = SearchCdf(conditional, kDomeMapWidth, u.y);
    const float columnLow = column > 0 ? conditional[column - 1] : 0.0f;
    const float columnProbability = conditional[column] - columnLow;
    const float y = rowProbability > 0 ? saturate((u.x - rowLow) / rowProbability) : 0.5f;
    const float x = columnProbability > 0 ? saturate((u.y - columnLow) / columnProbability) : 0.5f;
    const float2 uv = float2((column + x) / kDomeMapWidth, (row + y) / kDomeMapHeight);
    DomeMapSample sample;
    sample.direction = DomeMapDirection(uv);
    sample.pdf = DomeMapDensity(rowProbability, columnProbability, uv.y);
    return sample;
}

float DomeMapPdf(Ptr<float> map, float3 n) {
    const float2 uv = DomeMapUv(n);
    const uint column = min(uint(uv.x * kDomeMapWidth), kDomeMapWidth - 1);
    const uint row = min(uint(uv.y * kDomeMapHeight), kDomeMapHeight - 1);
    Ptr<float> marginal = map + kDomeMapCells;
    Ptr<float> conditional = map + row * kDomeMapWidth;
    const float rowProbability = marginal[row] - (row > 0 ? marginal[row - 1] : 0.0f);
    const float columnProbability =
        conditional[column] - (column > 0 ? conditional[column - 1] : 0.0f);
    return DomeMapDensity(rowProbability, columnProbability, uv.y);
}

struct DomeLight<Spectrum : ISpectrum> : ILight<Spectrum> {
    no_diff LightRecord record;
    no_diff LightContext context;
    no_diff uint64_t texture;
    no_diff ArrayRef portals;
    no_diff uint first;
    no_diff uint last;

    static DomeLight<Spectrum> Load(LightContext context, LightRecord record) {
        RenderTables* tables = context.tables;
        DomeLight<Spectrum> light;
        light.record = record;
        light.context = context;
        light.texture = record.textures != 0 ? tables->textureBindings[record.textures] : 0;
        light.portals.address = 0;
        light.portals.count = 0;
        light.first = 0;
        light.last = 0;
        if (record.portals > 0) {
            light.portals.address = record.portalPrims;
            light.portals.count = record.portals;
            light.first = context.lights->firstLights[context.tlas];
            light.last = context.lights->firstLights[context.tlas + 1];
        }
        return light;
    }

    float3 Texel(float3 direction) {
        if (texture == 0) {
            return float3(1);
        }
        const float2 st =
            DomeTextureSt(DomeTextureFormat(record.textureFormat), DomeNative(record, direction));
        return max(TextureFromHandle(texture).SampleLinear(st, kBaseLevel).rgb, float3(0));
    }

    Spectrum Emitted(float3 direction) {
        return Spectrum.FromRgb(record.radiance.Vector() * Texel(direction));
    }

    bool IsPortal(LightRecord portal) {
        return LightType(portal.type) == LightType::Portal && Lists(portals, portal.prim);
    }

    float Facing(LightRecord portal, float3 position) {
        if (dot(position - portal.position.Vector(), portal.axisZ.Vector()) >= 0) {
            return 0;
        }
        return AreaLight<Spectrum>.Load(context, portal).RectSolidAngle(position);
    }

    bool Through(LightRecord portal, float3 position, float3 direction) {
        const float3 normal = portal.axisZ.Vector();
        const float facing = dot(direction, normal);
        if (facing <= 0) {
            return false;
        }
        const float3 center = portal.position.Vector();
        const float t = dot(center - position, normal) / facing;
        if (t <= 0) {
            return false;
        }
        const float3 offset = position + t * direction - center;
        return abs(dot(offset, portal.axisX.Vector())) <= 0.5f * portal.width &&
               abs(dot(offset, portal.axisY.Vector())) <= 0.5f * portal.height;
    }

    float PortalPdf(float3 position, float3 direction) {
        float total = 0;
        uint hits = 0;
        for (uint i = first; i < last; ++i) {
            const LightRecord portal = context.lights->records[i];
            if (!IsPortal(portal)) {
                continue;
            }
            const float solid = Facing(portal, position);
            if (solid <= 0) {
                continue;
            }
            total += solid;
            if (Through(portal, position, direction)) {
                ++hits;
            }
        }
        return total > 0 ? hits / total : 0;
    }

    LightSample<Spectrum> Empty() {
        LightSample<Spectrum> sample;
        sample.direction = float3(0, 0, 1);
        sample.distance = kInfinity;
        sample.radiance = Spectrum.Constant(0);
        sample.pdf = 0;
        sample.delta = false;
        sample.diffuse = record.diffuse;
        sample.specular = record.specular;
        return sample;
    }

    LightSample<Spectrum> SamplePortals(float3 position, float2 u) {
        float total = 0;
        for (uint i = first; i < last; ++i) {
            if (IsPortal(context.lights->records[i])) {
                total += Facing(context.lights->records[i], position);
            }
        }
        if (total <= 0) {
            return Empty();
        }
        const float target = u.x * total;
        float before = 0;
        for (uint i = first; i < last; ++i) {
            const LightRecord portal = context.lights->records[i];
            if (!IsPortal(portal)) {
                continue;
            }
            const float solid = Facing(portal, position);
            if (solid <= 0) {
                continue;
            }
            if (target < before + solid || before + solid >= total) {
                const float remapped = saturate((target - before) / solid);
                float3 emitted;
                float3 at;
                const LightSample<Spectrum> through =
                    AreaLight<Spectrum>.Load(context, portal).SampleRect(
                        position, float2(remapped, u.y), emitted, at);
                if (through.pdf <= 0) {
                    return Empty();
                }
                LightSample<Spectrum> sample = Empty();
                sample.direction = through.direction;
                sample.pdf = PortalPdf(position, through.direction);
                sample.radiance = Emitted(through.direction);
                return sample;
            }
            before += solid;
        }
        return Empty();
    }

    LightSample<Spectrum> SampleDetached(float3 position, float2 u) {
        if (record.portals > 0) {
            return SamplePortals(position, u);
        }
        const DomeMapSample chosen = SampleDomeMap(record.map, u);
        LightSample<Spectrum> sample = Empty();
        sample.direction = normalize(DomeWorld(record, chosen.direction));
        sample.pdf = chosen.pdf;
        sample.radiance = Emitted(sample.direction);
        return sample;
    }

    [Differentiable]
    LightSample<Spectrum> Sample(float3 position, no_diff float2 u) {
        return no_diff SampleDetached(detach(position), u);
    }

    [Differentiable]
    Spectrum Radiance(float3 position, float3 incoming) {
        return no_diff Emitted(detach(incoming));
    }

    float Pdf(float3 position, float3 incoming) {
        if (record.portals > 0) {
            return PortalPdf(position, incoming);
        }
        return DomeMapPdf(record.map, DomeNative(record, incoming));
    }
}
