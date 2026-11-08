// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.integrator.iintegrator;
import ernie.render.kernels.light.distant;
import ernie.render.kernels.light.scatter;
import ernie.render.kernels.material.diffuse;
import ernie.render.kernels.material.preview;
import ernie.render.kernels.surface.primvars;
import ernie.render.kernels.random;
import ernie.render.kernels.basis;

using namespace ernie;

static const uint kMinimalAnisotropy = 1;

struct MinimalShade {
    float3 albedo;
    float3 unlit;
    float3 lit;
    bool testShadow;
    Ray shadow;
}

struct MinimalLighting {
    DistantLight<RgbSpectrum> light;
    bool headlight;
    bool castShadows;
    float shadowVisibility;
    float3 ambient;
    float3 ground;
    bool hemispheric;

    static MinimalLighting Load(HitQuery query, uint row, LightLayout* lights) {
        ProductRecord* product = &query.tables->products[row];
        MinimalLighting lighting;
        lighting.castShadows = product.castShadows != 0;
        lighting.shadowVisibility = product.shadowVisibility;
        lighting.ambient = product.ambient.Vector();
        lighting.ground = product.ground.Vector();
        lighting.hemispheric = product.hemispheric != 0;
        lighting.headlight = true;
        lighting.light = DistantLight<RgbSpectrum>(
            float3(0, 0, 1), RgbSpectrum.Constant(kHeadlightIrradiance), 1, 1);
        const uint first = lights->firstLights[query.tlas];
        const uint last = lights->firstLights[query.tlas + 1];
        for (uint i = first; i < last; ++i) {
            const LightRecord record = lights->records[i];
            if (LightType(record.type) == LightType::Distant) {
                const DistantLight<RgbSpectrum> sun = DistantLight<RgbSpectrum>.Load(
                    LightContext(query.scene, lights, query.tlas), record);
                const float spread = sun.spread > 0 ? sun.SolidAngle() : 1;
                lighting.light = DistantLight<RgbSpectrum>(
                    sun.direction, sun.radiance.Scale(spread), sun.diffuse, sun.specular);
                lighting.headlight = false;
                break;
            }
        }
        return lighting;
    }

    MinimalShade Shade<M : IMaterial<RgbSpectrum>>(Ray ray, SurfacePoint point, M material) {
        const Basis frame = Basis(point.shadingNormal);
        const float3 surrounding =
            hemispheric ? lerp(ground, ambient, 0.5f * (point.shadingNormal.z + 1)) : ambient;
        MinimalShade shade;
        shade.albedo = material.Albedo(point).ToRgb();
        shade.unlit = surrounding * shade.albedo +
                      material.Emission(point, frame.ToLocal(-ray.direction)).ToRgb();
        DistantLight<RgbSpectrum> source = light;
        if (headlight) {
            source.direction = -ray.direction;
        }
        const LightSample<RgbSpectrum> toward = source.Sample(point.position, float2(0.5f));
        shade.lit = Scattered<RgbSpectrum, M.Bsdf>(material.Surface(point), frame, -ray.direction,
                                                   toward)
                        .ToRgb();
        shade.testShadow = castShadows && !headlight && any(shade.lit > 0);
        if (shade.testShadow) {
            shade.shadow =
                SpawnRay(point, toward.direction, min(toward.distance, kBvhFar), ray.time);
        }
        return shade;
    }
}

struct MinimalIntegrator<let Mode : MinimalMode> : IIntegrator {
    typealias Spectrum = RgbSpectrum;

    HitQuery query;
    uint row;
    uint product;
    LightLayout* lights;

    static MinimalIntegrator<Mode> Load(SensorContext context) {
        const uint product = context.capture->product;
        MinimalIntegrator<Mode> integrator;
        integrator.query = context.query;
        integrator.row = PrimRow(context.query.tables, product, kSlotRenderProduct);
        integrator.product = product;
        integrator.lights = context.params.tables->lights;
        return integrator;
    }

    PathResult<RgbSpectrum> Trace(Ray ray, float spread, inout Rng rng) {
        PathResult<RgbSpectrum> result = PathResult<RgbSpectrum>.Miss();
        const PrimHit primHit = query.Intersect(ray);
        if (!primHit.IsHit()) {
            return result;
        }
        SurfacePoint point = ComputeSurfacePoint(query, ray, primHit, spread);
        result.point = point;
        if (Mode == MinimalMode::Data) {
            return result;
        }

        const MinimalLighting lighting = MinimalLighting.Load(query, row, lights);
        MinimalShade shade;
        if (Mode == MinimalMode::DisplayColor) {
            const float3 color = SurfaceDisplayColor(query.tables, point.slot);
            shade = lighting.Shade(ray, point, DiffuseMaterial<RgbSpectrum>(RgbSpectrum(color)));
        } else {
            MaterialQuery materials = MaterialQuery.Load(query, product);
            materials.minimal = true;
            materials.maxAnisotropy = min(materials.maxAnisotropy, kMinimalAnisotropy);
            const uint material = materials.BoundMaterial(point.prim);
            const bool preview = materials.Schema(material) == SchemaId::UsdPreviewSurface;
            if (Mode == MinimalMode::TextureDiffuse) {
                RgbSpectrum albedo;
                if (preview) {
                    albedo = PreviewSurface<RgbSpectrum>.LoadAlbedo(materials, material, point);
                } else {
                    albedo = DiffuseMaterial<RgbSpectrum>.Load(materials, material, point)
                                 .Albedo(point);
                }
                shade = lighting.Shade(ray, point, DiffuseMaterial<RgbSpectrum>(albedo));
            } else if (preview) {
                const PreviewSurface<RgbSpectrum> loaded =
                    PreviewSurface<RgbSpectrum>.Load(materials, material, point);
                shade = lighting.Shade(ray, point, loaded);
            } else {
                const DiffuseMaterial<RgbSpectrum> loaded =
                    DiffuseMaterial<RgbSpectrum>.Load(materials, material, point);
                shade = lighting.Shade(ray, point, loaded);
            }
        }

        float3 lit = shade.lit;
        if (shade.testShadow && query.Occluded(shade.shadow)) {
            lit *= lighting.shadowVisibility;
        }
        result.albedo = shade.albedo;
        result.radiance = RgbSpectrum(shade.unlit + lit);
        return result;
    }
}
