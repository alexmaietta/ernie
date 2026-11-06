// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.light.ilight;
import ernie.render.kernels.light.area;
import ernie.texture.kernels.texture;

using namespace ernie;

struct DomeLight<Spectrum : ISpectrum> : ILight<Spectrum> {
    no_diff LightRecord record;
    no_diff uint64_t texture;
    no_diff ArrayRef portals;
    LightLayout* lights;
    no_diff uint first;
    no_diff uint last;

    static DomeLight<Spectrum> FromRecord(LightRecord record) {
        DomeLight<Spectrum> light;
        light.record = record;
        light.texture = 0;
        light.portals.address = 0;
        light.portals.count = 0;
        light.lights = nullptr;
        light.first = 0;
        light.last = 0;
        return light;
    }

    static DomeLight<Spectrum> Load(LightRecord record, SceneData* scene, LightLayout* lights,
                                    uint tlas) {
        DomeLight<Spectrum> light = FromRecord(record);
        if (record.textures != 0) {
            light.texture = scene->textureBindings[record.textures];
        }
        if (record.portals > 0) {
            light.portals =
                scene->schemas.domeLight.portals[PrimRow(scene, record.prim, kSlotDomeLight)];
            light.lights = lights;
            light.first = lights->firstLights[tlas];
            light.last = lights->firstLights[tlas + 1];
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
        return AreaLight<Spectrum>.FromRecord(portal).RectSolidAngle(position);
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
            const LightRecord portal = lights->records[i];
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
            if (IsPortal(lights->records[i])) {
                total += Facing(lights->records[i], position);
            }
        }
        if (total <= 0) {
            return Empty();
        }
        const float target = u.x * total;
        float before = 0;
        for (uint i = first; i < last; ++i) {
            const LightRecord portal = lights->records[i];
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
                    AreaLight<Spectrum>.FromRecord(portal).SampleRect(
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
