// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
__exported import ernie.render.kernels.light.ilight;
import ernie.render.kernels.basis;

using namespace ernie;

struct DistantLight<Spectrum : ISpectrum> : ILight<Spectrum> {
    float3 direction;
    Spectrum radiance;
    float diffuse;
    float specular;
    no_diff float cosAngle;
    no_diff float spread;
    no_diff bool visible;

    [Differentiable]
    __init(float3 direction, Spectrum radiance, float diffuse, float specular) {
        this.direction = direction;
        this.radiance = radiance;
        this.diffuse = diffuse;
        this.specular = specular;
        cosAngle = 1;
        spread = 0;
        visible = false;
    }

    static DistantLight<Spectrum> FromRecord(LightRecord record) {
        DistantLight<Spectrum> light = DistantLight<Spectrum>(record.axisZ.Vector(),
                                                Spectrum.FromRgb(record.radiance.Vector()),
                                                record.diffuse, record.specular);
        light.cosAngle = record.cosAngle;
        light.spread = record.spread;
        light.visible = (record.flags & kLightVisible) != 0;
        return light;
    }

    float SolidAngle() {
        return 2 * kPi * spread;
    }

    [Differentiable]
    LightSample<Spectrum> Sample(float3 position, no_diff float2 u) {
        LightSample<Spectrum> sample;
        sample.distance = kInfinity;
        sample.radiance = radiance;
        sample.diffuse = diffuse;
        sample.specular = specular;
        sample.delta = spread <= 0;
        if (sample.delta) {
            sample.direction = direction;
            sample.pdf = 1;
            return sample;
        }
        const float cosine = 1 - u.x * spread;
        const float sine = sqrt(max(1 - cosine * cosine, 0.0f));
        const float phi = 2 * kPi * u.y;
        const Basis frame = Basis(direction);
        sample.direction = frame.ToWorld(float3(sine * cos(phi), sine * sin(phi), cosine));
        sample.pdf = 1 / SolidAngle();
        return sample;
    }

    [Differentiable]
    Spectrum Radiance(float3 position, float3 incoming) {
        if (spread <= 0 || dot(incoming, direction) < cosAngle) {
            return Spectrum.Constant(0);
        }
        return radiance;
    }

    float Pdf(float3 position, float3 incoming) {
        if (spread <= 0 || dot(incoming, direction) < cosAngle) {
            return 0;
        }
        return 1 / SolidAngle();
    }
}
