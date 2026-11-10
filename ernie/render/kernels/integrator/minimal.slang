// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.integrator.iintegrator;
import ernie.render.kernels.light.distant;
import ernie.render.kernels.light.scatter;
import ernie.render.kernels.material.dispatch;
import ernie.render.kernels.material.thermal;
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

    static MinimalLighting Load(HitQuery query, uint row) {
        ProductRecord* product = &query.tables->products[row];
        MinimalLighting lighting;
        lighting.castShadows = product.castShadows != 0;
        lighting.shadowVisibility = product.shadowVisibility;
        lighting.ambient = product.ambient.Vector();
        lighting.ground = product.ground.Vector();
        lighting.hemispheric = product.hemispheric != 0;
        const MinimalLightRecord record = query.tables->minimalLights[query.tlas];
        lighting.headlight = record.headlight != 0;
        lighting.light =
            lighting.headlight
                ? DistantLight<RgbSpectrum>(float3(0, 0, 1),
                                            RgbSpectrum.Constant(kHeadlightIrradiance), 1, 1)
                : DistantLight<RgbSpectrum>(record.direction.Vector(),
                                            RgbSpectrum(record.radiance.Vector()), record.diffuse,
                                            record.specular);
        return lighting;
    }

    MinimalShade Shade(Ray ray, SurfacePoint point, OpticalClosure<RgbSpectrum> closure) {
        const Basis frame = Basis(point.shadingNormal);
        const float3 surrounding =
            hemispheric ? lerp(ground, ambient, 0.5f * (point.shadingNormal.z + 1)) : ambient;
        MinimalShade shade;
        shade.albedo = closure.albedo.ToRgb();
        shade.unlit = surrounding * shade.albedo + closure.emission.ToRgb();
        DistantLight<RgbSpectrum> source = light;
        if (headlight) {
            source.direction = -ray.direction;
        }
        const LightSample<RgbSpectrum> toward = source.Sample(point.position, float2(0.5f));
        shade.lit = Scattered<RgbSpectrum, OpticalClosure<RgbSpectrum>>(closure, frame,
                                                                        -ray.direction, toward)
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

    static MinimalIntegrator<Mode> Load(SensorContext context) {
        const uint product = context.capture->plan.product;
        MinimalIntegrator<Mode> integrator;
        integrator.query = context.query;
        integrator.row = PrimRow(context.query.tables, product, kSlotRenderProduct);
        integrator.product = product;
        return integrator;
    }

    PathResult<RgbSpectrum> Trace(Ray ray, float spread, inout Rng rng) {
        PathResult<RgbSpectrum> result = PathResult<RgbSpectrum>.Miss();
        const PrimHit primHit = query.Intersect(ray, kGeometryMask | kTraceCullBackFaces);
        if (!primHit.IsHit()) {
            return result;
        }
        SurfacePoint point = ComputeSurfacePoint(query, ray, primHit, spread);
        result.point = point;
        if (Mode == MinimalMode::Data) {
            return result;
        }

        const MinimalLighting lighting = MinimalLighting.Load(query, row);
        MinimalShade shade;
        if (Mode == MinimalMode::DisplayColor) {
            const float3 color = SurfaceDisplayColor(query.tables, point.slot, point.element);
            shade = lighting.Shade(ray, point,
                                   OpticalClosure<RgbSpectrum>.Diffuse(RgbSpectrum(color)));
        } else {
            MaterialQuery materials = MaterialQuery.Load(query, product);
            materials.minimal = true;
            materials.maxAnisotropy = min(materials.maxAnisotropy, kMinimalAnisotropy);
            const uint material = point.material;
            if (Mode == MinimalMode::TextureDiffuse) {
                ShadeContext context = ShadeContext(materials, material, point, -ray.direction);
                const RgbSpectrum albedo = AlbedoDefinition<RgbSpectrum>(context);
                shade = lighting.Shade(ray, context.point,
                                       OpticalClosure<RgbSpectrum>.Diffuse(albedo));
            } else {
                ShadeContext context = ShadeContext(materials, material, point, -ray.direction);
                OpticalClosure<RgbSpectrum> closure = OpticalClosure<RgbSpectrum>();
                ShadeDefinition<RgbSpectrum>(context, closure);
                AddThermalEmission<RgbSpectrum>(query.tables, context.point, closure);
                shade = lighting.Shade(ray, context.point, closure);
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
