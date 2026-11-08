// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.projection.iprojection;

using namespace ernie;

static const int kBrownConradySteps = 10;

struct BrownConrady : IDifferentiable {
    no_diff float4 radial;
    no_diff float3 divisor;
    no_diff float2 tangential;
    no_diff float4 thinPrism;
    no_diff float focalLength;

    static BrownConrady Load(RenderTables* tables, uint row, float focalLength) {
        const DistortionRecord record = tables->distortions[row];
        BrownConrady model;
        model.radial = record.radial.Vector();
        model.divisor = record.radialDivisor.Vector();
        model.tangential = record.tangential.Vector();
        model.thinPrism = record.thinPrism.Vector();
        model.focalLength = focalLength;
        return model;
    }

    [Differentiable]
    float Scale(float r2) {
        return (1 + r2 * (radial.x + r2 * (radial.y + r2 * (radial.z + r2 * radial.w)))) /
               (1 + r2 * (divisor.x + r2 * (divisor.y + r2 * divisor.z)));
    }

    [Differentiable]
    float2 Shift(float2 p, float r2) {
        const float r4 = r2 * r2;
        return float2(2 * tangential.x * p.x * p.y + tangential.y * (r2 + 2 * p.x * p.x) +
                          thinPrism.x * r2 + thinPrism.y * r4,
                      tangential.x * (r2 + 2 * p.y * p.y) + 2 * tangential.y * p.x * p.y +
                          thinPrism.z * r2 + thinPrism.w * r4);
    }

    [Differentiable]
    float2 Distort(float2 ideal) {
        const float2 p = float2(ideal.x, -ideal.y) / focalLength;
        const float r2 = dot(p, p);
        const float2 q = p * Scale(r2) + Shift(p, r2);
        return float2(q.x, -q.y) * focalLength;
    }

    [Differentiable]
    float2 Undistort(float2 film) {
        const float2 q = float2(film.x, -film.y) / focalLength;
        float2 p = q;
        [MaxIters(kBrownConradySteps)]
        for (int step = 0; step < kBrownConradySteps; ++step) {
            const float r2 = dot(p, p);
            p = (q - Shift(p, r2)) / Scale(r2);
        }
        return float2(p.x, -p.y) * focalLength;
    }
}
