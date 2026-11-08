// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.integrator.iintegrator;
import ernie.render.kernels.light.area;
import ernie.render.kernels.light.distant;
import ernie.render.kernels.light.dome;
import ernie.render.kernels.light.scatter;
import ernie.render.kernels.material.diffuse;
import ernie.render.kernels.material.preview;
import ernie.render.kernels.random;
import ernie.render.kernels.basis;

using namespace ernie;

struct ChosenLight {
    uint index;
    float probability;
};

ChosenLight ChooseLight(LightLayout* lights, uint tlas, float u) {
    const uint first = lights->firstLights[tlas];
    const uint last = lights->firstLights[tlas + 1];
    uint low = first;
    uint high = last - 1;
    while (low < high) {
        const uint middle = (low + high) / 2;
        if (lights->cdf[middle] < u) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    ChosenLight chosen;
    chosen.index = low;
    chosen.probability = lights->cdf[low] - (low > first ? lights->cdf[low - 1] : 0.0f);
    return chosen;
}

static const uint kMaxPassThroughs = 16;

struct PathVertex<Spectrum : ISpectrum> {
    SurfacePoint point;
    Spectrum radiance;
    float3 albedo;
    float opacity;
    BsdfSample<Spectrum> next;
}

struct MaterialBsdf<Spectrum : ISpectrum> : IBsdf<Spectrum> {
    PreviewSurfaceBsdf<Spectrum> preview;
    no_diff bool lambert;

    [Differentiable]
    __init(PreviewSurfaceBsdf<Spectrum> preview) {
        this.preview = preview;
        lambert = false;
    }

    [Differentiable]
    __init(Lambert<Spectrum> diffuse) {
        preview = PreviewSurfaceBsdf<Spectrum>(diffuse.albedo, Spectrum.Constant(0), 1, 0, 1);
        lambert = true;
    }

    [Differentiable]
    BsdfValue<Spectrum> Evaluate(float3 outgoing, float3 incoming) {
        if (lambert) {
            return Lambert<Spectrum>(preview.diffuse).Evaluate(outgoing, incoming);
        }
        return preview.Evaluate(outgoing, incoming);
    }

    [Differentiable]
    BsdfSample<Spectrum> Sample(float3 outgoing, no_diff float3 u) {
        if (lambert) {
            return Lambert<Spectrum>(preview.diffuse).Sample(outgoing, u);
        }
        return preview.Sample(outgoing, u);
    }

    float Pdf(float3 outgoing, float3 incoming) {
        if (lambert) {
            return Lambert<Spectrum>(preview.diffuse).Pdf(outgoing, incoming);
        }
        return preview.Pdf(outgoing, incoming);
    }
}

struct RealTimeIntegrator<S : ISpectrum> : IIntegrator {
    typealias Spectrum = S;

    HitQuery query;
    MaterialQuery materials;
    uint bounces;
    LightLayout* lights;

    static RealTimeIntegrator<S> Load(SensorContext context) {
        RenderTables* tables = context.query.tables;
        const uint product = context.capture->plan.product;
        const uint row = PrimRow(tables, product, kSlotRenderProduct);
        RealTimeIntegrator<S> integrator;
        integrator.query = context.query;
        integrator.materials = MaterialQuery.Load(context.query, product);
        integrator.bounces = tables->products[row].maxBounces;
        integrator.lights = context.params.tables->lights;
        return integrator;
    }

    PathResult<S> Trace(Ray primary, float spread, inout Rng rng) {
        PathResult<S> result = PathResult<S>.Miss();
        Spectrum radiance = Spectrum.Constant(0);
        Spectrum throughput = Spectrum.Constant(1);
        Ray ray = primary;
        uint bounce = 0;
        uint passes = 0;
        float3 origin = primary.origin;
        uint originPrim = kNoPrim;
        float scatteredPdf = 0;
        bool scatteredDelta = true;
        while (true) {
            const uint mask = kGeometryMask | (bounce == 0 ? kCameraEmitterMask : kEmitterMask);
            const PrimHit primHit = query.Intersect(ray, mask);
            if (!primHit.IsHit()) {
                radiance = radiance + throughput * Sky(origin, ray.direction, originPrim,
                                                       scatteredPdf, scatteredDelta);
                break;
            }
            const SurfacePoint point = ComputeSurfacePoint(query, ray, primHit, spread);
            if (IsEmitter(uint(PrimSchema(query.tables, point.prim)))) {
                if (bounce == 0) {
                    result.point = point;
                }
                radiance = radiance + throughput * Emitter(origin, originPrim, point,
                                                           scatteredPdf, scatteredDelta);
                break;
            }
            const uint material = materials.BoundMaterial(point.prim);
            const float3 u = float3(rng.Next2(), rng.Next());
            const PathVertex<Spectrum> vertex = Shade(ray, material, point, u, bounce, rng);
            if (rng.Next() >= vertex.opacity && passes < kMaxPassThroughs) {
                ++passes;
                ray = SpawnRay(point, ray.direction, kBvhFar, ray.time);
                continue;
            }
            if (bounce == 0) {
                result.point = point;
                result.albedo = vertex.albedo;
            }
            radiance = radiance + throughput * vertex.radiance;
            if (bounce >= bounces || vertex.next.pdf <= 0) {
                break;
            }
            throughput = throughput * vertex.next.weight.Sum();
            if (throughput.MaxValue() <= 0) {
                break;
            }
            const Basis frame = Basis(vertex.point.shadingNormal);
            origin = vertex.point.position;
            originPrim = vertex.point.prim;
            scatteredPdf = vertex.next.pdf;
            scatteredDelta = vertex.next.delta;
            ray = SpawnRay(vertex.point, frame.ToWorld(vertex.next.direction), kBvhFar, ray.time);
            ++bounce;
        }
        result.radiance = radiance;
        return result;
    }

    float Selection(uint tlas, uint index) {
        const uint first = lights->firstLights[tlas];
        const uint last = lights->firstLights[tlas + 1];
        if (last - first <= kLoopedLights) {
            return 1;
        }
        return lights->cdf[index] - (index > first ? lights->cdf[index - 1] : 0.0f);
    }

    float Weight(float scatteredPdf, bool scatteredDelta, float lightPdf) {
        if (scatteredDelta || lightPdf <= 0) {
            return 1;
        }
        return scatteredPdf * scatteredPdf / (scatteredPdf * scatteredPdf + lightPdf * lightPdf);
    }

    Spectrum Sky(float3 origin, float3 direction, uint originPrim,
                 float scatteredPdf, bool scatteredDelta) {
        const bool camera = originPrim == kNoPrim;
        Spectrum seen = Spectrum.Constant(0);
        const LightContext context = LightContext(query.tables, lights, query.tlas);
        const uint first = lights->firstLights[query.tlas];
        const uint last = lights->firstLights[query.tlas + 1];
        for (uint i = first; i < last; ++i) {
            const LightRecord record = lights->records[i];
            const LightType type = LightType(record.type);
            if ((type != LightType::Distant && type != LightType::Dome) ||
                (camera && (record.flags & kLightVisible) == 0) ||
                (!camera && !InCollection(query.tables, record.lightLink, originPrim))) {
                continue;
            }
            Spectrum emitted;
            float pdf = 0;
            if (type == LightType::Dome) {
                const DomeLight<Spectrum> light = DomeLight<Spectrum>.Load(context, record);
                emitted = light.Radiance(origin, direction);
                pdf = camera ? 0 : light.Pdf(origin, direction);
            } else {
                const DistantLight<Spectrum> light = DistantLight<Spectrum>.Load(context, record);
                emitted = light.Radiance(origin, direction);
                pdf = light.Pdf(origin, direction);
            }
            if (camera) {
                seen = seen + emitted;
            } else {
                pdf *= Selection(query.tlas, i);
                seen = seen + emitted.Scale(Weight(scatteredPdf, scatteredDelta, pdf));
            }
        }
        return seen;
    }

    Spectrum Emitter(float3 origin, uint originPrim, SurfacePoint point,
              float scatteredPdf, bool scatteredDelta) {
        const bool camera = originPrim == kNoPrim;
        const uint first = lights->firstLights[query.tlas];
        const uint last = lights->firstLights[query.tlas + 1];
        const SlotHit drawn = query.layout->hits[point.slot];
        for (uint i = first; i < last; ++i) {
            const LightRecord candidate = lights->records[i];
            if (candidate.prim != point.prim || candidate.instancer != drawn.instancer ||
                (drawn.instancer != kNoPrim && candidate.copyOffset != drawn.offset)) {
                continue;
            }
            if (!camera && !InCollection(query.tables, lights->records[i].lightLink, originPrim)) {
                return Spectrum.Constant(0);
            }
            const AreaLight<Spectrum> light =
                AreaLight<Spectrum>.Load(LightContext(query.tables, lights, query.tlas), candidate);
            const Spectrum emitted = light.EmittedToward(origin, point.position);
            if (camera) {
                return emitted;
            }
            const float pdf = light.PointPdf(origin, point.position) * Selection(query.tlas, i);
            return emitted.Scale(Weight(scatteredPdf, scatteredDelta, pdf));
        }
        return Spectrum.Constant(0);
    }

    LightSample<Spectrum> SampleLight(LightRecord record, float3 position,
                                      float2 u) {
        const LightContext context = LightContext(query.tables, lights, query.tlas);
        switch (LightType(record.type)) {
        case LightType::Distant:
            return DistantLight<Spectrum>.Load(context, record).Sample(position, u);
        case LightType::Dome:
            return DomeLight<Spectrum>.Load(context, record).Sample(position, u);
        default:
            return AreaLight<Spectrum>.Load(context, record).Sample(position, u);
        }
    }

    PathVertex<Spectrum> Surfaced<M : IMaterial<Spectrum>>(M loaded, SurfacePoint point,
                                                          float3 outgoing) {
        PathVertex<Spectrum> vertex;
        vertex.point = point;
        vertex.albedo = loaded.Albedo(point).ToRgb();
        vertex.opacity = loaded.Opacity(point);
        vertex.radiance = loaded.Emission(point, outgoing);
        return vertex;
    }

    PathVertex<Spectrum> Shade(Ray ray, uint material, SurfacePoint point,
                               float3 u, uint bounce, inout Rng rng) {
        const bool primary = bounce == 0;
        const float weighted = bounce < bounces ? 1.0f : 0.0f;
        PathVertex<Spectrum> vertex;
        MaterialBsdf<Spectrum> bsdf;
        if (materials.Schema(material) == SchemaId::UsdPreviewSurface) {
            const PreviewSurface<Spectrum> loaded =
                PreviewSurface<Spectrum>.Load(materials, material, point);
            vertex = Surfaced(loaded, point, Basis(point.shadingNormal).ToLocal(-ray.direction));
            bsdf = MaterialBsdf<Spectrum>(loaded.Surface(point));
        } else {
            const DiffuseMaterial<Spectrum> loaded =
                DiffuseMaterial<Spectrum>.Load(materials, material, point);
            vertex = Surfaced(loaded, point, Basis(point.shadingNormal).ToLocal(-ray.direction));
            bsdf = MaterialBsdf<Spectrum>(loaded.Surface(point));
        }
        const Basis frame = Basis(point.shadingNormal);
        const float3 outgoing = frame.ToLocal(-ray.direction);
        vertex.next = bsdf.Sample(outgoing, u);
        const uint first = lights->firstLights[query.tlas];
        const uint last = lights->firstLights[query.tlas + 1];
        const bool sampled = last - first > kLoopedLights;
        uint begin = first;
        uint end = last;
        float scale = 1;
        float selection = weighted;
        if (sampled) {
            const ChosenLight chosen = ChooseLight(lights, query.tlas, rng.Next());
            begin = chosen.index;
            end = chosen.probability > 0 ? begin + 1 : begin;
            scale = chosen.probability > 0 ? 1 / chosen.probability : 0;
            selection = chosen.probability * weighted;
        }
        bool lit = sampled;
        for (uint i = begin; i <= end; ++i) {
            LightSample<Spectrum> sample;
            Shadowing shadowing = Shadowing.None();
            float chance = 0;
            float factor = 1;
            if (i == end) {
                if (lit || !primary) {
                    break;
                }
                sample = DistantLight<Spectrum>(-ray.direction,
                                                Spectrum.Constant(kHeadlightIrradiance), 1, 1)
                             .Sample(point.position, float2(0.5f));
            } else {
                const LightRecord record = lights->records[i];
                const float2 v = rng.Next2();
                const LightType type = LightType(record.type);
                if (type == LightType::Portal || type == LightType::Unsupported) {
                    continue;
                }
                lit = true;
                if (!InCollection(query.tables, record.lightLink, point.prim)) {
                    continue;
                }
                sample = SampleLight(record, point.position, v);
                shadowing = Shadowing.FromRecord(record);
                chance = selection;
                factor = scale;
            }
            vertex.radiance =
                vertex.radiance +
                Direct<Spectrum, MaterialBsdf<Spectrum>>(query, ray, point, frame, bsdf, sample,
                                                         shadowing, chance)
                    .Scale(factor);
        }
        return vertex;
    }
}
