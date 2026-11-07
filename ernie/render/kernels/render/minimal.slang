// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.render.irender;
import ernie.render.kernels.light.distant;
import ernie.render.kernels.light.scatter;
import ernie.render.kernels.material.diffuse;
import ernie.render.kernels.material.preview;
import ernie.render.kernels.surface.primvars;
import ernie.render.kernels.random;
import ernie.render.kernels.basis;

using namespace ernie;

static const uint kMinimalAnisotropy = 1;

struct MinimalLighting {
    DistantLight<RgbSpectrum> light;
    bool headlight;
    bool castShadows;
    float shadowVisibility;
    float3 ambient;
    float3 ground;
    bool hemispheric;

    static MinimalLighting Load(HitQuery query, uint row, LightLayout* lights) {
        SceneData* scene = query.data;
        MinimalAPIData* minimal = &scene->schemas.minimalAPI;
        MinimalLighting lighting;
        lighting.castShadows = minimal.castShadows[row] != 0;
        lighting.shadowVisibility = minimal.shadowVisibility[row];
        const float intensity = minimal.ambientLightIntensity[row];
        lighting.ambient = minimal.ambientLightColor[row].Vector() * intensity;
        lighting.ground = minimal.ambientGroundColor[row].Vector() * intensity;
        lighting.hemispheric = minimal.ambientHemispheric[row] != 0;
        lighting.headlight = true;
        lighting.light = DistantLight<RgbSpectrum>(
            float3(0, 0, 1), RgbSpectrum.Constant(kHeadlightIrradiance), 1, 1);
        const uint first = lights->firstLights[query.tlas];
        const uint last = lights->firstLights[query.tlas + 1];
        for (uint i = first; i < last; ++i) {
            const LightRecord record = lights->records[i];
            if (LightType(record.type) == LightType::Distant) {
                const DistantLight<RgbSpectrum> sun = DistantLight<RgbSpectrum>.Load(
                    LightContext(scene, lights, query.tlas), record);
                const float spread = sun.spread > 0 ? sun.SolidAngle() : 1;
                lighting.light = DistantLight<RgbSpectrum>(
                    sun.direction, sun.radiance.Scale(spread), sun.diffuse, sun.specular);
                lighting.headlight = false;
                break;
            }
        }
        return lighting;
    }

    float3 Shade<M : IMaterial<RgbSpectrum>>(Ray ray, SurfacePoint point, M material,
                                                 inout PathSample sample) {
        const Basis frame = Basis(point.shadingNormal);
        const float3 surrounding =
            hemispheric ? lerp(ground, ambient, 0.5f * (point.shadingNormal.z + 1)) : ambient;
        const float3 unlit = surrounding * material.Albedo(point).ToRgb() +
                             material.Emission(point, frame.ToLocal(-ray.direction)).ToRgb();
        DistantLight<RgbSpectrum> source = light;
        if (headlight) {
            source.direction = -ray.direction;
        }
        const LightSample<RgbSpectrum> toward = source.Sample(point.position, float2(0.5f));
        sample.lit = Scattered<RgbSpectrum, M.Bsdf>(material.Surface(point), frame,
                                                        -ray.direction, toward)
                         .ToRgb();
        sample.testShadow = castShadows && !headlight && any(sample.lit > 0);
        sample.shadowVisibility = shadowVisibility;
        if (sample.testShadow) {
            sample.shadow = SpawnRay(point, toward.direction, min(toward.distance, kBvhFar),
                                     ray.time);
        }
        return unlit;
    }
}

struct MinimalRender<let Mode : MinimalMode> : IRender {
    uint row;
    uint product;
    LightLayout* lights;

    static MinimalRender<Mode> Load(RenderParams params, Capture* capture, HitQuery query) {
        SceneData* scene = query.data;
        const uint row = ApiRow(scene, capture->product, kApiMinimalAPI);
        MinimalRender<Mode> render;
        render.row = row;
        render.lights = params.lights;
        render.product = capture->product;
        return render;
    }

    PathSample TracePath(HitQuery query, Ray ray, float spread, inout Rng rng) {
        const PrimHit primHit = query.Intersect(ray);
        if (!primHit.IsHit()) {
            return PathSample.Miss();
        }
        PathSample sample = PathSample.Miss();
        sample.hit = true;
        sample.point = ComputeSurfacePoint(query, ray, primHit, spread);
        sample.radiance = float3(0);
        sample.albedo = float3(0);
        if (Mode == MinimalMode::Data) {
            return sample;
        }
        if (Mode == MinimalMode::DisplayColor) {
            SceneData* scene = query.data;
            sample.albedo = SurfaceDisplayColor(scene, query.layout, sample.point.prim, sample.point.slot);
            sample.radiance = MinimalLighting.Load(query, row, lights).Shade(
                ray, sample.point, DiffuseMaterial<RgbSpectrum>(RgbSpectrum(sample.albedo)),
                sample);
            return sample;
        }
        MaterialQuery materials = MaterialQuery.Load(query, product);
        materials.minimal = true;
        materials.maxAnisotropy = min(materials.maxAnisotropy, kMinimalAnisotropy);
        const uint material = materials.BoundMaterial(sample.point.prim);
        const bool preview = materials.Schema(material) == SchemaId::UsdPreviewSurface;
        if (Mode == MinimalMode::TextureDiffuse) {
            RgbSpectrum albedo;
            if (preview) {
                albedo = PreviewSurface<RgbSpectrum>.LoadAlbedo(materials, material,
                                                                    sample.point);
            } else {
                albedo = DiffuseMaterial<RgbSpectrum>.Load(materials, material, sample.point)
                             .Albedo(sample.point);
            }
            sample.albedo = albedo.ToRgb();
            sample.radiance = MinimalLighting.Load(query, row, lights).Shade(
                ray, sample.point, DiffuseMaterial<RgbSpectrum>(albedo), sample);
            return sample;
        }
        if (preview) {
            return Shade<PreviewSurface<RgbSpectrum>>(query, materials, ray, material, sample);
        }
        return Shade<DiffuseMaterial<RgbSpectrum>>(query, materials, ray, material, sample);
    }

    PathSample Shade<M : IMaterial<RgbSpectrum>>(HitQuery query, MaterialQuery materials,
                                                     Ray ray, uint material, PathSample sample) {
        const M loaded = M.Load(materials, material, sample.point);
        sample.albedo = loaded.Albedo(sample.point).ToRgb();
        sample.radiance = MinimalLighting.Load(query, row, lights).Shade(ray, sample.point, loaded, sample);
        return sample;
    }
}
