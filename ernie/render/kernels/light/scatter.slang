// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;
__exported import ernie.render.kernels.light.ilight;
__exported import ernie.render.kernels.material.ibsdf;
__exported import ernie.render.kernels.surface.point;
import ernie.render.kernels.basis;

using namespace ernie;

[Differentiable]
Spectrum Scattered<Spectrum : ISpectrum, B : IBsdf<Spectrum>>(B bsdf, Basis frame, float3 outgoing,
                                         LightSample<Spectrum> sample) {
    const float3 incoming = frame.ToLocal(sample.direction);
    if (sample.pdf <= 0 || incoming.z <= 0) {
        return Spectrum.Constant(0);
    }
    const BsdfValue<Spectrum> value = bsdf.Evaluate(frame.ToLocal(outgoing), incoming);
    return value.Scaled(sample.diffuse, sample.specular) * sample.radiance *
           (incoming.z / sample.pdf);
}

struct Shadowing {
    bool cast;
    float3 color;
    float distance;
    float falloff;
    float gamma;
    uint links;

    static Shadowing None() {
        Shadowing shadowing;
        shadowing.cast = false;
        shadowing.color = float3(0);
        shadowing.distance = -1;
        shadowing.falloff = -1;
        shadowing.gamma = 1;
        shadowing.links = kNoRow;
        return shadowing;
    }

    static Shadowing FromRecord(LightRecord record) {
        Shadowing shadowing;
        shadowing.cast = (record.flags & kLightShadowless) == 0;
        shadowing.color = record.shadowColor.Vector();
        shadowing.distance = record.shadowDistance;
        shadowing.falloff = record.shadowFalloff;
        shadowing.gamma = record.shadowFalloffGamma;
        shadowing.links = record.shadowLink;
        return shadowing;
    }

    float Strength(HitQuery query, Ray shadow) {
        const bool hard = distance <= 0 || falloff <= 0;
        const float t = query.OccluderDistance(shadow, links, hard);
        if (!isfinite(t)) {
            return 0;
        }
        if (hard) {
            return 1;
        }
        const float start = distance - falloff;
        return t <= start ? 1 : pow(saturate(1 - (t - start) / falloff), gamma);
    }
}

Spectrum Direct<Spectrum : ISpectrum, B : IBsdf<Spectrum>>(HitQuery query, Ray ray,
                                                           SurfacePoint point, Basis frame, B bsdf,
                                                           LightSample<Spectrum> sample,
                                                           Shadowing shadowing, float selection) {
    Spectrum lit = Scattered<Spectrum, B>(bsdf, frame, -ray.direction, sample);
    if (!sample.delta && sample.pdf > 0 && selection > 0) {
        const float chosen = sample.pdf * selection;
        const float scattered =
            bsdf.Pdf(frame.ToLocal(-ray.direction), frame.ToLocal(sample.direction));
        lit = lit.Scale(chosen * chosen / (chosen * chosen + scattered * scattered));
    }
    if (!shadowing.cast || lit.MaxValue() <= 0) {
        return lit;
    }
    float reach = min(sample.distance, kBvhFar);
    if (shadowing.distance > 0) {
        reach = min(reach, shadowing.distance);
    }
    const Ray shadow = SpawnRay(point, sample.direction, reach, ray.time);
    const float strength = shadowing.Strength(query, shadow);
    if (strength <= 0) {
        return lit;
    }
    return lit * Spectrum.FromRgb(lerp(float3(1), shadowing.color, strength));
}
