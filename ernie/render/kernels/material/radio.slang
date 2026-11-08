// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

static const uint kElectricalConstant = 0;
static const uint kElectricalItu = 1;
static const uint kElectricalConductor = 2;

struct ElectricalValue {
    uint kind;
    float4 data;

    static ElectricalValue Constant(float permittivity, float conductivity) {
        ElectricalValue value;
        value.kind = kElectricalConstant;
        value.data = float4(permittivity, conductivity, 0, 0);
        return value;
    }

    static ElectricalValue Itu(float a, float b, float c, float d) {
        ElectricalValue value;
        value.kind = kElectricalItu;
        value.data = float4(a, b, c, d);
        return value;
    }

    static ElectricalValue Conductor() {
        ElectricalValue value;
        value.kind = kElectricalConductor;
        value.data = float4(0);
        return value;
    }

    float2 Permittivity(float gigahertz) {
        float permittivity = data.x;
        float conductivity = data.y;
        if (kind == kElectricalItu) {
            permittivity = data.x * pow(gigahertz, data.y);
            conductivity = data.z * pow(gigahertz, data.w);
        }
        return float2(permittivity, -17.98f * conductivity / max(gigahertz, 1e-6f));
    }

    float Reflectance(float gigahertz) {
        if (kind == kElectricalConductor) {
            return 1;
        }
        const float2 eta = Permittivity(gigahertz);
        const float magnitude = sqrt(length(eta));
        const float angle = 0.5f * atan2(eta.y, eta.x);
        const float2 root = magnitude * float2(cos(angle), sin(angle));
        const float2 numerator = float2(1 - root.x, -root.y);
        const float2 denominator = float2(1 + root.x, root.y);
        return dot(numerator, numerator) / max(dot(denominator, denominator), 1e-12f);
    }
}

struct RadioClosure {
    ElectricalValue electrical;
    float rmsHeight;
    float correlationLength;

    static RadioClosure Dielectric() {
        RadioClosure closure;
        closure.electrical = ElectricalValue.Constant(5, 0.01f);
        closure.rmsHeight = 0.001f;
        closure.correlationLength = 0.01f;
        return closure;
    }

    float Backscatter(float gigahertz, float cosine) {
        const float c = clamp(cosine, 1e-3f, 1.0f);
        const float slope = max(1.41421356f * rmsHeight / max(correlationLength, 1e-6f), 1e-3f);
        const float variance = slope * slope;
        const float tangent = (1 - c * c) / (c * c);
        return electrical.Reflectance(gigahertz) * exp(-tangent / (2 * variance)) /
               (2 * variance * c * c * c * c);
    }
}
